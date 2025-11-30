#!/usr/bin/env python3
"""
mkv_shrink.py - shrink MKV files by re-encoding only the video stream.
Audio, subtitles, chapters, and attachments are copied losslessly.

Usage:
    python mkv_shrink.py input.mkv output.mkv
"""

import argparse
import subprocess
import shutil
import sys
import json
from pathlib import Path
import re
import statistics
from collections import Counter
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mkv_shrink.log", encoding="utf-8"),
    ],
)

QUALITY_PRESETS = {
    "h264": {
        "cpu": ["-preset", "slow", "-crf", "21"],
        "nvenc": ["-preset", "p5", "-cq", "27"],
    },
    "hevc": {
        "cpu": ["-preset", "slow", "-crf", "28"],
        "nvenc": ["-preset", "p5", "-cq", "28"],
    },
}


def probe_streams(mkv_path: Path) -> list[dict]:
    """
    Use ffprobe to inspect *all* streams in an MKV file.
    Returns a list of dicts with type, codec, channels, language, etc.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_type,codec_name,channels,field_order,interlaced_frame,"
        "r_frame_rate,avg_frame_rate:stream_tags=language,title",
        "-of",
        "json",
        str(mkv_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not result.stdout.strip():
        print(f"[WARN] ffprobe returned no output for {mkv_path}")
        return []
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse ffprobe output: {e}")
        return []

    streams = []
    for s in data.get("streams", []):
        entry = {
            "index": s.get("index"),
            "type": s.get("codec_type"),
            "codec": s.get("codec_name"),
            "language": s.get("tags", {}).get("language", "und"),
            "title": s.get("tags", {}).get("title"),
            "field_order": s.get("field_order"),
            "interlaced": s.get("interlaced_frame"),
            "r_frame_rate": s.get("r_frame_rate"),
        }
        if "channels" in s:  # audio streams only
            entry["channels"] = s["channels"]
        streams.append(entry)
    return streams


def filter_streams_by_language(
    streams: list[dict], allowed_langs=("eng", "jpn")
) -> dict:
    """
    Filter streams by language.
    Returns dict with 'video', 'audio', 'subs', 'other' containing lists of stream indexes.
    """
    selected = {"video": [], "audio": [], "subs": [], "other": []}

    for s in streams:
        stype = s.get("type")
        lang = (s.get("language") or "und").lower()

        if stype == "video":
            selected["video"].append(s["index"])
        elif stype == "audio":
            if lang in allowed_langs:
                selected["audio"].append(s["index"])
        elif stype == "subtitle":
            if lang in allowed_langs:
                selected["subs"].append(s["index"])
        else:
            selected["other"].append(s["index"])

    return selected


def choose_encoder(codec: str, prefer_nvenc: bool = True) -> list[str]:
    """Return encoder args for h264 or hevc, using NVENC if available."""
    assert codec in ("h264", "hevc"), f"Unsupported codec: {codec}"

    if prefer_nvenc:
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                check=True,
            )
            encoders = result.stdout
            nvenc_name = f"{codec}_nvenc"
            if nvenc_name in encoders:
                print(f"[INFO] Using NVENC encoder: {nvenc_name}")
                return ["-c:v", nvenc_name] + QUALITY_PRESETS[codec]["nvenc"]
            print(f"[WARN] {nvenc_name} not available, falling back to CPU.")
        except Exception as e:
            print(f"[WARN] Could not check NVENC support: {e}, using CPU.")

    return ["-c:v", f"lib{codec}"] + QUALITY_PRESETS[codec]["cpu"]


def order_and_label_audio(
    streams: list[dict], allowed_langs=("eng", "jpn")
) -> list[dict]:
    """
    Return audio streams in preferred order with labels:
    1. Surround (>=6 channels)
    2. Stereo (2 channels)
    3. Mono (1 channel)
    4. Commentary (title contains 'commentary')
    Keeps only allowed languages.
    """
    ordered = []
    for lang in allowed_langs:
        surround = [
            s
            for s in streams
            if s["type"] == "audio"
            and s["language"] == lang
            and s.get("channels", 0) >= 6
        ]
        stereo = [
            s
            for s in streams
            if s["type"] == "audio"
            and s["language"] == lang
            and s.get("channels", 0) == 2
            and "commentary" not in (s.get("title") or "").lower()
        ]
        mono = [
            s
            for s in streams
            if s["type"] == "audio"
            and s["language"] == lang
            and s.get("channels", 0) == 1
            and "commentary" not in (s.get("title") or "").lower()
        ]
        commentary = [
            s
            for s in streams
            if s["type"] == "audio"
            and s["language"] == lang
            and "commentary" in (s.get("title") or "").lower()
        ]
        ordered.extend(surround + stereo + mono + commentary)
    return ordered


def probe_duration_seconds(input_file: str) -> float | None:
    """Return duration in seconds for the given file, or None on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                input_file,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip() or 0)
    except Exception as exc:
        print(f"[WARN] Could not read duration for {input_file}: {exc}")
        return None


def build_crop_offsets(input_file: str) -> list[int]:
    """
    Build cropdetect offsets biased to the middle of the movie to avoid intro/outro AR shifts.
    """
    duration = probe_duration_seconds(input_file)
    offsets: list[int] = []

    if duration and duration > 0:
        for frac in (0.35, 0.50, 0.65):
            ts = int(duration * frac)
            ts = max(300, ts)
            if duration > 900:
                ts = min(ts, int(duration) - 600)
            if ts > 0:
                offsets.append(ts)

    # Legacy early offsets as fallback
    offsets.extend([600, 1200, 1800])

    # Deduplicate preserving order
    seen = set()
    ordered = []
    for off in offsets:
        if off not in seen and off > 0:
            ordered.append(off)
            seen.add(off)
    return ordered


def probe_video_aspect(input_file: str) -> tuple[int | None, int | None, float | None, float | None]:
    """Return (width, height, sar, dar) for the first video stream, or None values on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,sample_aspect_ratio,display_aspect_ratio",
                "-of",
                "default=nw=1:nk=1",
                input_file,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        if len(lines) < 4:
            return None, None, None, None
        w = int(lines[0])
        h = int(lines[1])

        def parse_ratio(val: str) -> float | None:
            try:
                if ":" in val:
                    a, b = val.split(":", 1)
                    b = float(b)
                    if b == 0:
                        return None
                    return float(a) / b
                return float(val)
            except Exception:
                return None

        sar = parse_ratio(lines[2])
        dar = parse_ratio(lines[3])
        if dar is None and sar and w and h:
            dar = (w / h) * sar
        return w, h, sar, dar
    except Exception:
        return None, None, None, None


def detect_crop(
    input_file: str,
    start_seconds: int = 600,
    frames: int = 400,
    limit: int = 24,
    allow_fallback: bool = True,
) -> list[str]:
    """
    Detect stable crop region using ffmpeg cropdetect.
    - Skips intros (starts later)
    - Filters outliers around the median
    - Prevents side crops for HD sources (keeps full width)
    Returns ['crop=w:h:x:y'] or [] if no reliable crop found.
    """
    try:
        print("[INFO] Analyzing for letterboxing/pillarboxing...")

        # run cropdetect
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-ss",
            str(start_seconds),
            "-i",
            input_file,
            "-vf",
            f"cropdetect=limit={limit}:round=2",
            "-frames:v",
            str(frames),
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        crops = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", result.stderr)
        if not crops:
            print("[INFO] No crop data found.")
            return []

        # parse and filter invalid
        valid = [
            (int(w), int(h), int(x), int(y))
            for w, h, x, y in crops
            if int(w) > 0 and int(h) > 0
        ]
        if not valid:
            print("[WARN] No valid crop entries found.")
            return []

        # median filter
        med_w = statistics.median([w for w, _, _, _ in valid])
        med_h = statistics.median([h for _, h, _, _ in valid])
        filtered = [
            c for c in valid if abs(c[0] - med_w) <= 16 and abs(c[1] - med_h) <= 16
        ]
        if not filtered:
            print("[WARN] No consistent crop readings.")
            return []

        # most common near-median crop
        w, h, x, y = Counter(filtered).most_common(1)[0][0]

        # enforce even dimensions for encoders
        if w < 64 or h < 64 or w % 2 or h % 2:
            print(f"[WARN] Ignoring odd/too-small crop {w}x{h}.")
            return []

        # check for HD width and disable side crops
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                input_file,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        m = re.search(r"(\d+)[,\s]+(\d+)", probe.stdout)
        if m:
            src_w, src_h = map(int, m.groups())
            if src_w >= 1280 and w < src_w:
                if x > 0 or (src_w - (w + x)) > 0:
                    print(
                        f"[INFO] HD source ({src_w}x{src_h}); removing only letterbox (no side crop)."
                    )
                    x, w = 0, src_w

        crop_str = f"crop={w}:{h}:{x}:{y}"
        print(f"[INFO] Stable crop region detected: {crop_str}")

        # Fallback for scope/letterboxed content: if AR is suspiciously tall, retry with a more sensitive limit,
        # but keep whichever crop is wider.
        ar = w / h
        if allow_fallback and ar < 2.2 and limit > 2:
            print(
                "[INFO] Crop AR looks tall for scope; retrying cropdetect with lower limit for faint bars."
            )
            fallback = detect_crop(
                input_file,
                start_seconds=start_seconds,
                frames=frames,
                limit=2,
                allow_fallback=False,
            )
            if fallback:
                m_fb = re.search(r"crop=(\d+):(\d+):", fallback[0])
                if m_fb:
                    w2, h2 = map(int, m_fb.groups())
                    ar2 = w2 / h2 if h2 else 0
                    if ar2 > ar:
                        print(
                            f"[INFO] Using fallback crop {fallback[0]} (AR ~{ar2:.3f}) over {crop_str} (AR ~{ar:.3f})"
                        )
                        return fallback
            # keep original if fallback not better

        return [crop_str]

    except Exception as e:
        print(f"[WARN] Crop detection failed: {e}")
        return []


def detect_crop_multi(input_file: str) -> list[str]:
    """
    Try multiple start offsets to find a stable crop region.
    Uses detect_crop() internally for each offset.
    Returns the first consistent result, or [] if none found.
    """
    offsets = build_crop_offsets(input_file)
    all_results: list[str] = []

    for offset in offsets:
        result = detect_crop(input_file, start_seconds=offset)
        if result:
            all_results.extend(result)

    if not all_results:
        print("[INFO] No stable crop found across multiple passes.")
        return []

    # Prefer most common; on near-ties favor the widest AR to avoid pillarboxing.
    from collections import Counter

    counts = Counter(all_results)
    max_count = counts.most_common(1)[0][1]

    def aspect_ratio(crop: str) -> float:
        m = re.search(r"crop=(\d+):(\d+):", crop)
        if not m:
            return 0.0
        w, h = map(int, m.groups())
        return (w / h) if h else 0.0

    candidates = [c for c, cnt in counts.items() if cnt >= max_count - 1]
    best = sorted(
        candidates,
        key=lambda c: (counts[c], aspect_ratio(c), c),
        reverse=True,
    )[0]

    print(f"[INFO] Chosen stable crop after multi-pass: {best}")
    return [best]


def shrink_mkv(input_file: str, output_file: str, codec: str = "h264") -> None:
    """Run ffmpeg to shrink the MKV (video re-encode only)."""
    video_opts = choose_encoder(codec, prefer_nvenc=True)

    is_foreign = "foreign" in input_file.lower()
    no_crop = "no_crop" in input_file.lower()

    streams = probe_streams(input_file)
    video_stream = next((s for s in streams if s["type"] == "video"), None)

    src_w, src_h, src_sar, src_dar = probe_video_aspect(input_file)

    # Check for telecine / inerlacing
    vf = []
    if video_stream:
        field_order = (video_stream.get("field_order") or "").lower()
        interlaced_flag = str(video_stream.get("interlaced") or "0")

        if field_order in {"tt", "bb", "tb", "bt"} or interlaced_flag == "1":
            print(f"[INFO] Detected interlaced ({field_order}), applying bwdif filter")
            vf = ["-vf", "bwdif=1:0:0"]
        else:
            vf = []

    if not no_crop:
        crop_filter = detect_crop_multi(input_file)

    if is_foreign:
        audio_streams = [s for s in streams if s.get("type") == "audio"]
        keep = {
            "video": [s["index"] for s in streams if s.get("type") == "video"],
            "audio": [s["index"] for s in streams if s.get("type") == "audio"],
            "subs": [s["index"] for s in streams if s.get("type") == "subtitle"],
            "other": [
                s["index"]
                for s in streams
                if s.get("type") not in ("video", "audio", "subtitle")
            ],
        }
    else:
        audio_streams = order_and_label_audio(streams, allowed_langs=("eng", "jpn"))
        keep = filter_streams_by_language(streams, allowed_langs=("eng", "jpn", "und"))

    cmd = ["ffmpeg", "-y", "-i", input_file]

    # Only the first stream
    cmd += ["-map", "0:v:0"]

    filters = []
    if vf:  # from interlace/telecine detection
        filters.append(vf[-1].replace("-vf ", ""))

    if not no_crop and crop_filter:
        crop_str = crop_filter[0]
        filters.append(crop_str)
        filters.append("setsar=1")  # normalize SAR after crop

        # Set DAR based on cropped dimensions and source SAR (if valid); fallback to crop AR.
        m = re.search(r"crop=(\d+):(\d+):", crop_str)
        if m:
            w, h = map(int, m.groups())
            if h:
                if src_sar and src_sar > 0:
                    dar = (w * src_sar) / h
                    filters.append(f"setdar={dar:.6f}")
                    print(
                        f"[INFO] Setting DAR from crop+source SAR ({src_sar:.3f}) => {dar:.3f}"
                    )
                else:
                    crop_ar = w / h
                    filters.append(f"setdar={crop_ar:.6f}")
                    print(f"[INFO] Setting DAR to crop AR ({crop_ar:.3f}) from {crop_str}")

    if filters:
        cmd += ["-vf", ",".join(filters)]
        print(f"[INFO] Using filters: {filters}")

    # Then encoder settings
    cmd += video_opts

    # Audio: ordered, but don’t touch titles
    for a_index, s in enumerate(audio_streams):
        cmd += [
            "-map",
            f"0:{s['index']}",
            f"-c:a:{a_index}",
            "copy",
            f"-metadata:s:a:{a_index}",
            f"language={s['language']}",
        ]

    # Subs: just copy filtered ones
    s_index = 0
    for idx in keep["subs"]:
        cmd += ["-map", f"0:{idx}", f"-c:s:{s_index}", "copy"]
        s_index += 1

    # Keep attachments/chapters
    cmd += ["-map_chapters", "0", "-map_metadata", "0"]

    cmd += ["-movflags", "+faststart", output_file]

    print("[INFO] Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] ffmpeg failed: {result.stderr}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Shrink MKV by re-encoding video only."
    )
    parser.add_argument("input", help="Input MKV file")
    parser.add_argument("output", help="Output MKV file")
    parser.add_argument(
        "--codec",
        choices=["h264", "hevc"],
        default="h264",
        help="Video codec to use (default: h264)",
    )
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("Error: ffmpeg not found in PATH")

    shrink_mkv(args.input, args.output, args.codec)


if __name__ == "__main__":
    main()
