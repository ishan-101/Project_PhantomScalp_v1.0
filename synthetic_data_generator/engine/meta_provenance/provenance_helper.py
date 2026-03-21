"""
Infrastructure-grade provenance utilities for synthetic data engines.

This module centralizes provenance record construction, hashing, and
validation to guarantee deterministic, engine-agnostic metadata that can be
consumed by manifests and validators across all synthetic data engines.

Guarantees:
    * Deterministic outputs for identical inputs
    * Pure functions with no side effects or IO
    * Engine-agnostic helpers with standard library dependencies only

Usage example (import-only, non-executable):
    from synthetic_data_generator.engine.meta_provenance import provenance_helper as prov

    config_hash = prov.hash_config(config)
    record = prov.build_provenance_record(
        dataset_name="spot",
        engine_name="spot_engine",
        engine_version="1.2.0",
        config_version="v1.0.0",
        config_hash=config_hash,
        time_range_start="2025-12-01T00:00:00Z",
        time_range_end="2025-12-01T01:00:00Z",
        rows=10000,
        symbol=prov.normalize_symbol("btc-usdt"),
        exchange="binance",
        environment="dev",
        seed=12345,
    )
    assert prov.validate_provenance_schema(record)
"""

from __future__ import annotations

import datetime
import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional


REQUIRED_FIELDS: Iterable[str] = (
    "dataset_name",
    "engine_name",
    "engine_version",
    "config_version",
    "config_hash",
    "generated_at_utc",
    "time_range_start",
    "time_range_end",
    "rows",
    "symbol",
    "exchange",
    "environment",
)

OPTIONAL_FIELDS: Iterable[str] = (
    "git_commit",
    "git_branch",
    "seed",
    "dependencies",
    "python_version",
    "platform",
    "notes",
)

VALID_ENVIRONMENTS = {"dev", "staging", "prod"}


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string without microseconds."""

    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def normalize_symbol(symbol: str) -> str:
    """
    Normalize a symbol/ticker to a consistent uppercase, hyphen-free form.

    This helper is intentionally conservative: it strips whitespace, uppercases the
    value, and replaces common separators with a single canonical separator.
    """

    normalized = symbol.strip()
    normalized = normalized.replace("/", "-").replace("_", "-")
    return normalized.upper()


def hash_config(config: Mapping[str, Any]) -> str:
    """
    Compute a deterministic SHA-256 hash for a configuration mapping.

    The mapping is serialized with sorted keys and no whitespace to ensure stable
    hashing across runs and interpreters.
    """

    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_provenance_record(
    *,
    dataset_name: str,
    engine_name: str,
    engine_version: str,
    config_version: str,
    config_hash: str,
    time_range_start: str,
    time_range_end: str,
    rows: int,
    symbol: str,
    exchange: str,
    environment: str,
    generated_at_utc: Optional[str] = None,
    git_commit: Optional[str] = None,
    git_branch: Optional[str] = None,
    seed: Optional[int] = None,
    dependencies: Optional[List[str]] = None,
    python_version: Optional[str] = None,
    platform_info: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Construct a provenance record with standardized fields.

    Inputs are not mutated. Optional values are only included when provided.
    """

    record: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "engine_name": engine_name,
        "engine_version": engine_version,
        "config_version": config_version,
        "config_hash": config_hash,
        "generated_at_utc": generated_at_utc or utc_now_iso(),
        "time_range_start": time_range_start,
        "time_range_end": time_range_end,
        "rows": int(rows),
        "symbol": symbol,
        "exchange": exchange,
        "environment": environment,
    }

    optional_values = {
        "git_commit": git_commit,
        "git_branch": git_branch,
        "seed": seed,
        "dependencies": list(dependencies) if dependencies is not None else None,
        "python_version": python_version,
        "platform": platform_info,
        "notes": notes,
    }

    for key, value in optional_values.items():
        if value is not None:
            record[key] = value

    return record


def _is_iso8601(value: str) -> bool:
    try:
        datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def validate_provenance_schema(record: Mapping[str, Any]) -> bool:
    """
    Validate a provenance record against required fields and basic typing rules.

    Returns True when the record satisfies presence, type, and format checks;
    returns False otherwise. The input mapping is never mutated.
    """

    for field in REQUIRED_FIELDS:
        if field not in record:
            return False

    if record.get("environment") not in VALID_ENVIRONMENTS:
        return False

    rows_value = record.get("rows")
    if not isinstance(rows_value, int) or rows_value < 0:
        return False

    for date_field in ("generated_at_utc", "time_range_start", "time_range_end"):
        value = record.get(date_field)
        if not isinstance(value, str) or not _is_iso8601(value):
            return False

    str_fields = ("dataset_name", "engine_name", "engine_version", "config_version", "config_hash")
    for str_field in str_fields:
        if not isinstance(record.get(str_field), str):
            return False

    return True
