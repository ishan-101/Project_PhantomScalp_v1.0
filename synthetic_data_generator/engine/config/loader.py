# synthetic_data_generator/engine/config/loader.py

import yaml
import hashlib
from pathlib import Path
from functools import lru_cache


# Path to the config file (relative to this loader)
_CONFIG_PATH = Path(__file__).parent / "synthetic_config.yaml"


class ConfigError(Exception):
    """Raised when the configuration is missing or invalid."""
    pass


def _hash_config_text(text: str) -> str:
    """Return a short SHA256 hash of the YAML text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_minimal_structure(cfg: dict):
    """
    Minimal but strict validation for Option B config.
    Ensures critical sections exist.
    """
    required_top = [
        "meta",
        "global",
        "rows",
        "paths",
        "partitioning",
        "writer",
        "sharder",
        "validation",
        "engines",
        "manifest",
        "logging",
    ]

    for key in required_top:
        if key not in cfg:
            raise ConfigError(f"Config missing required top-level section: '{key}'")

    # Validate rows (must be positive integers)
    for engine, val in cfg["rows"].items():
        if not isinstance(val, int) or val <= 0:
            raise ConfigError(f"rows.{engine} must be a positive integer. Got: {val}")

    # Validate required path fields
    if "base" not in cfg["paths"]:
        raise ConfigError("paths.base is required.")

    # Validate partitioning
    part = cfg["partitioning"]
    if "columns" not in part or not isinstance(part["columns"], list):
        raise ConfigError("partitioning.columns must be a list (e.g., ['date']).")

    return True


@lru_cache(maxsize=1)
def load_config() -> dict:
    """
    Load, validate, and return the synthetic configuration.
    Cached so engines/orchestrator do not repeatedly load the YAML.
    """
    if not _CONFIG_PATH.exists():
        raise ConfigError(f"Config file not found: {_CONFIG_PATH}")

    # Read YAML text
    text = _CONFIG_PATH.read_text(encoding="utf-8")
    try:
        cfg = yaml.safe_load(text)
    except Exception as e:
        raise ConfigError(f"Failed to parse YAML config: {e}")

    # Basic structure validation
    _validate_minimal_structure(cfg)

    # Attach SHA256 hash for manifest tracking
    cfg["_config_hash"] = _hash_config_text(text)

    return cfg
