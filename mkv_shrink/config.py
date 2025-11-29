from pathlib import Path
import tomllib
import os

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.toml"


def load_config(config_path: Path | None = None) -> dict:
    """Load TOML configuration, allowing overrides via environment variables."""
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    # Optional overrides via env vars
    cfg["incoming"]["path"] = os.getenv("MKV_INCOMING", cfg["incoming"]["path"])
    cfg["output"]["path"] = os.getenv("MKV_OUTPUT", cfg["output"]["path"])
    cfg["done"]["path"] = os.getenv("MKV_DONE", cfg["done"]["path"])

    return cfg
