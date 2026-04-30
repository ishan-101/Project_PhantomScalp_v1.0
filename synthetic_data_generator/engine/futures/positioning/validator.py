"""Strict validator for Futures Positioning dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


class PositioningValidationError(Exception):
    """Raised when futures positioning dataset fails strict checks."""


REQUIRED_COLUMNS: List[str] = [
    "meta__timestamp",
    "meta__sequence_id",
    "fut__long_short_ratio",
    "fut__long_short_ratio_change",
    "fut__net_long_position_proxy",
    "fut__net_short_position_proxy",
    "fut__position_skew",
    "fut__net_position_change_velocity",
]

FEATURE_COLUMNS = [c for c in REQUIRED_COLUMNS if c.startswith("fut__")]


def _load_futures_base_schema() -> Dict:
    schema_path = Path(__file__).resolve().parents[1] / "schema" / "Futures Base Schema V1 0 · json"
    with schema_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_positioning_df(df: pd.DataFrame) -> bool:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise PositioningValidationError(f"Missing required columns: {missing}")

    ts = df["meta__timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise PositioningValidationError("meta__timestamp must be datetime dtype")
    if getattr(ts.dtype, "tz", None) is None or str(getattr(ts.dtype, "tz", "")) != "UTC":
        raise PositioningValidationError("meta__timestamp must be UTC tz-aware")

    seq = df["meta__sequence_id"]
    if str(seq.dtype) != "int64":
        raise PositioningValidationError("meta__sequence_id must be int64")

    if not ts.is_monotonic_increasing:
        raise PositioningValidationError("meta__timestamp must be monotonic increasing")
    if not seq.is_monotonic_increasing:
        raise PositioningValidationError("meta__sequence_id must be monotonic increasing")

    null_counts = df[REQUIRED_COLUMNS].isna().sum()
    bad_nulls = {k: int(v) for k, v in null_counts.items() if int(v) > 0}
    if bad_nulls:
        raise PositioningValidationError(f"Zero-null policy violated: {bad_nulls}")

    for col in FEATURE_COLUMNS:
        if str(df[col].dtype) != "float32":
            raise PositioningValidationError(f"{col} must be float32, got {df[col].dtype}")

    schema = _load_futures_base_schema()
    rules = schema.get("validation_rules", {})
    if not rules.get("strict_dtype_enforcement", False):
        raise PositioningValidationError("Futures base schema strict dtype enforcement disabled")
    if not rules.get("fail_on_null", False):
        raise PositioningValidationError("Futures base schema null-fail disabled")
    if not rules.get("fail_on_non_monotonic_sequence", False):
        raise PositioningValidationError("Futures base schema monotonic sequence fail disabled")

    return True
