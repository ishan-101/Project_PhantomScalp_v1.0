"""Manifest module: centralized dataset registry and provenance ledger.

Philosophy (documented for future maintainers):
- Manifest  dataset contents; it records facts about produced datasets.
- Manifest  validator; validation logic lives in dedicated validators.
- Manifest  logger; no side effects like prints or logging.
- Records facts only  no inference or mutation of datasets.
- Immutable per dataset entry; append-only across runs.
- Must remain human-readable and machine-readable.

This module provides an engine-agnostic, deterministic, append-safe manifest
implementation that can be consumed by engines, runners, validators, and CI
pipelines without introducing circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import json
import os
import time


# ---------------------------------------------------------------------------
# Manifest entry structure (commented design for clarity and future changes)
#
# A single manifest entry encodes the provenance of one produced dataset file.
# All fields are derivable and deterministic; no optional ambiguity, no
# engine-specific hacks.
#
# {
#   "engine": str,
#   "dataset_type": str,
#   "symbol": str,
#   "exchange": str,
#   "row_count": int,
#   "columns": list[str],
#   "schema_hash": str,
#   "partition_date": str,
#   "file_path": str,
#   "file_size_bytes": int,
#   "created_at_utc": str,
#   "config_fingerprint": str
# }
#
# Notes:
# - All fields must be derivable from generation context, not inferred from
#   reading data. Schema hash should be derived from ordered column names.
# - created_at_utc is ISO 8601 (UTC) with seconds precision for determinism.
# - file_path is stored relative to repository root or configured base path
#   to preserve portability across environments.
# - config_fingerprint allows detecting config drift (e.g., from
#   synthetic_config.yaml and runtime overrides).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Public API (signatures)
# ---------------------------------------------------------------------------
def build_manifest_entry(
    *,
    engine: str,
    dataset_type: str,
    symbol: str,
    exchange: str,
    row_count: int,
    columns: Sequence[str],
    partition_date: str,
    file_path: Path,
    config: Dict,
) -> Dict:
    """Construct a deterministic manifest entry dictionary."""


def validate_manifest_entry(entry: Dict) -> Dict:
    """Validate manifest entry completeness and types; returns the entry."""


def write_manifest(entries: Sequence[Dict], path: Path) -> None:
    """Write manifest entries atomically to a JSON file."""


def read_manifest(path: Path) -> List[Dict]:
    """Read manifest JSON file and return a list of entries."""


def append_manifest(entry: Dict, path: Path) -> None:
    """Append a validated manifest entry to an existing manifest file."""


def hash_schema(columns: Sequence[str]) -> str:
    """Create a deterministic hash of ordered column names."""


def hash_config(cfg: Dict) -> str:
    """Create a deterministic hash of configuration mapping."""


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ManifestEntry:
    engine: str
    dataset_type: str
    symbol: str
    exchange: str
    row_count: int
    columns: List[str]
    schema_hash: str
    partition_date: str
    file_path: str
    file_size_bytes: int
    created_at_utc: str
    config_fingerprint: str

    def to_dict(self) -> Dict:
        # Use asdict from dataclasses to maintain stable key ordering (insertion
        # order of field definitions) and ensure JSON-serializable primitives.
        return asdict(self)


_REQUIRED_FIELDS = {
    "engine": str,
    "dataset_type": str,
    "symbol": str,
    "exchange": str,
    "row_count": int,
    "columns": list,
    "schema_hash": str,
    "partition_date": str,
    "file_path": str,
    "file_size_bytes": int,
    "created_at_utc": str,
    "config_fingerprint": str,
}


def _isoformat_utc_now() -> str:
    # Use time.gmtime for UTC without reliance on system tz; second precision for
    # deterministic string formatting.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ensure_path(path: Path) -> Path:
    if isinstance(path, str):
        return Path(path)
    return path


def _stable_json_dumps(obj: Dict) -> str:
    # Stable, deterministic JSON string for hashing: sorted keys, no whitespace
    # variability.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def hash_schema(columns: Sequence[str]) -> str:
    ordered = list(columns)
    payload = "|".join(ordered)
    return sha256(payload.encode("utf-8")).hexdigest()


def hash_config(cfg: Dict) -> str:
    return sha256(_stable_json_dumps(cfg).encode("utf-8")).hexdigest()


def build_manifest_entry(
    *,
    engine: str,
    dataset_type: str,
    symbol: str,
    exchange: str,
    row_count: int,
    columns: Sequence[str],
    partition_date: str,
    file_path: Path,
    config: Dict,
) -> Dict:
    manifest_path = _ensure_path(file_path)
    file_size = manifest_path.stat().st_size if manifest_path.exists() else 0

    entry = ManifestEntry(
        engine=str(engine),
        dataset_type=str(dataset_type),
        symbol=str(symbol),
        exchange=str(exchange),
        row_count=int(row_count),
        columns=list(columns),
        schema_hash=hash_schema(columns),
        partition_date=str(partition_date),
        file_path=str(manifest_path),
        file_size_bytes=int(file_size),
        created_at_utc=_isoformat_utc_now(),
        config_fingerprint=hash_config(config),
    )
    return entry.to_dict()


def validate_manifest_entry(entry: Dict) -> Dict:
    # Basic structural validation: required keys and types.
    for key, expected_type in _REQUIRED_FIELDS.items():
        if key not in entry:
            raise ValueError(f"Manifest entry missing required key: {key}")
        value = entry[key]
        if expected_type is list:
            if not isinstance(value, list) or not all(
                isinstance(col, str) for col in value
            ):
                raise TypeError(f"Manifest entry field '{key}' must be list[str]")
        elif not isinstance(value, expected_type):
            raise TypeError(
                f"Manifest entry field '{key}' expected {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

    # Additional checks: non-empty columns, non-negative sizes/rows.
    if not entry["columns"]:
        raise ValueError("Manifest entry columns must be non-empty")
    if entry["row_count"] < 0:
        raise ValueError("Manifest entry row_count must be non-negative")
    if entry["file_size_bytes"] < 0:
        raise ValueError("Manifest entry file_size_bytes must be non-negative")

    return entry


def write_manifest(entries: Sequence[Dict], path: Path) -> None:
    manifest_path = _ensure_path(path)
    # Validate all entries before writing.
    validated_entries = [validate_manifest_entry(entry) for entry in entries]

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(validated_entries, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, manifest_path)


def read_manifest(path: Path) -> List[Dict]:
    manifest_path = _ensure_path(path)
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Manifest file must contain a list of entries")
    return [validate_manifest_entry(entry) for entry in data]


def append_manifest(entry: Dict, path: Path) -> None:
    manifest_path = _ensure_path(path)
    existing = read_manifest(manifest_path)
    existing.append(validate_manifest_entry(entry))
    write_manifest(existing, manifest_path)

