#!/usr/bin/env python3
"""
mkv_shrink_watcher.py
---------------------

Watches an incoming directory for MKV files, detects when they are fully written,
shrinks them using `shrink_mkv`, and moves the completed files to an output directory.
"""

import time
import shutil
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from mkv_shrink.shrink import shrink_mkv
from mkv_shrink.config import load_config
import logging
import argparse

log = logging.getLogger(__name__)


class MKVHandler(FileSystemEventHandler):
    """Handles filesystem events and tracks MKV files until they stabilize in size.

    Attributes:
        tracking (dict[Path, tuple[int, int]]): Maps file paths to (last_size, stable_count).
        stable_times (int): Number of consecutive checks where size must remain stable.
    """

    def __init__(self, stable_times: int):
        """Initializes the MKVHandler.

        Args:
            stable_times (int): Number of consecutive checks required
                for a file's size to be considered stable.
        """
        self.tracking = {}  # path -> (last_size, stable_count)
        self.stable_times = stable_times

    def on_created(self, event):
        """Called when a new file is created in the watched directory.

        Args:
            event (FileSystemEvent): The event object containing file path and type.
        """
        if not event.is_directory and event.src_path.lower().endswith(".mkv"):
            path = Path(event.src_path)
            log.info(f"New MKV detected: {path}")
            self.tracking[path] = (path.stat().st_size, 0)

    def check_stable_files(self) -> list[Path]:
        """Checks tracked MKV files to see if their size has stabilized.

        Returns:
            list[Path]: A list of file paths that have been stable long enough
                to be processed.
        """
        done = []
        for path, (last_size, stable_count) in list(self.tracking.items()):
            if not path.exists():
                del self.tracking[path]
                continue

            current_size = path.stat().st_size
            if current_size == last_size:
                stable_count += 1
            else:
                stable_count = 0

            self.tracking[path] = (current_size, stable_count)

            log.debug(
                f"{path}: size={current_size}, last={last_size}, stable={stable_count}"
            )

            if stable_count >= self.stable_times:
                done.append(path)
                del self.tracking[path]

        return done


def process_file(path: Path, output_dir: Path, done_dir: Path, codec: str) -> None:
    """Shrinks and moves a single MKV file when ready.

    Args:
        path (Path): Path to the input MKV file.
        output_dir (Path): Directory where the processed MKV will be saved.
        done_dir (Path): Directory where the original MKV will be archived.
        codec (str): Video codec to use for shrinking (e.g., "h264" or "hevc").
    """
    output_file = output_dir / path.name
    tmp_file = output_file.with_suffix(".tmp.mkv")

    log.info(f"Processing {path} -> {output_file}")
    try:
        shrink_mkv(str(path), str(tmp_file), codec=codec)
        shutil.move(tmp_file, output_file)

        # Move original into DONE_DIR
        done_file = done_dir / path.name
        shutil.move(str(path), done_file)

        log.info(f"Finished {output_file}")
        log.info(f"Original moved to {done_file}")
    except Exception as e:
        log.error(f"Failed processing {path}: {e}")
        if tmp_file.exists():
            tmp_file.unlink(missing_ok=True)


def is_file_ready(path: Path) -> bool:
    """Checks whether a file is ready for processing (not locked).

    Attempts to open and read a small portion of the file.

    Args:
        path (Path): Path to the file to check.

    Returns:
        bool: True if the file is readable, False otherwise.
    """
    try:
        with open(path, "rb") as f:
            f.read(1024)  # try to read a small chunk
        return True
    except OSError as e:
        print(f"[DEBUG] File not ready yet ({path}): {e}")
        return False


def run_watcher(
    incoming_dir: Path,
    output_dir: Path,
    done_dir: Path,
    check_interval: int,
    stable_times: int,
    codec: str,
) -> None:
    """Main watcher loop that monitors for new MKV files and processes them.

    Args:
        incoming_dir (Path): Directory to watch for new MKV files.
        output_dir (Path): Directory where processed MKVs will be saved.
        done_dir (Path): Directory where originals will be moved after processing.
        check_interval (int): Seconds to wait between file size checks.
        stable_times (int): Number of consecutive stable checks before processing.
        codec (str): Codec to use for shrinking.
    """

    observer = Observer()
    handler = MKVHandler(stable_times)
    observer.schedule(handler, str(incoming_dir), recursive=False)
    observer.start()
    log.info("Watching %s -- press Ctrl+C to stop", incoming_dir)

    for d in (incoming_dir, output_dir, done_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Track existing files
    for f in incoming_dir.glob("*.mkv"):
        log.info(f"Tracking existing file at startup: {f}")
        handler.tracking[f] = (f.stat().st_size, 0)

    try:
        while True:
            stable_files = handler.check_stable_files()
            for f in stable_files:
                if is_file_ready(f):
                    process_file(f, output_dir, done_dir, codec)
                else:
                    handler.tracking[f] = (f.stat().st_size, 0)
            time.sleep(check_interval)
    except KeyboardInterrupt:
        log.info("Shutting down watcher...")
        observer.stop()

    finally:
        observer.stop()
        observer.join()


def main() -> None:
    """Entry point for the watcher script.

    Loads configuration, then starts the MKV watcher using parameters
    from the config file.
    """

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Watch a directory for MKV files and shrink them automatically."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a custom config.toml file (default: built-in config).",
    )
    parser.add_argument(
        "--incoming",
        type=Path,
        help="Override incoming directory path from config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override output directory path from config.",
    )
    parser.add_argument(
        "--done",
        type=Path,
        help="Override done directory path from config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    run_watcher(
        incoming_dir=Path(cfg["incoming"]["path"]),
        output_dir=Path(cfg["output"]["path"]),
        done_dir=Path(cfg["done"]["path"]),
        check_interval=cfg["watcher"]["check_interval"],
        stable_times=cfg["watcher"]["stable_times"],
        codec=cfg["shrink"]["codec"],
    )


if __name__ == "__main__":
    main()
