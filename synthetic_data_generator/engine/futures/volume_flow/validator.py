"""Strict validator for Futures Volume Flow dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


class VolumeFlowValidationError(Exception):
    """Raised when volume-flow dataset fails strict validation."""


REQUIRED_COLUMNS: List[str] = [
    "meta__timestamp",
    "meta__sequence_id",
    "fut__trade_volume",
    "fut__volume_delta",
    "fut__volume_delta_ratio",
    "fut__large_trade_volume",
    "fut__volume_burst_intensity",
]

FEATURE_COLUMNS: List[str] = [c for c in REQUIRED_COLUMNS if c.startswith("fut__")]


def _load_futures_base_schema() -> Dict:
    schema_path = Path(__file__).resolve().parents[1] / "schema" / "Futures Base Schema V1 0 · json"
    with schema_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_volume_flow_df(df: pd.DataFrame) -> bool:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise VolumeFlowValidationError(f"Missing required columns: {missing}")

    ts = df["meta__timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise VolumeFlowValidationError("meta__timestamp must be datetime dtype")
    if getattr(ts.dtype, "tz", None) is None:
        raise VolumeFlowValidationError("meta__timestamp must be UTC tz-aware")
    if str(getattr(ts.dtype, "tz", "")) != "UTC":
        raise VolumeFlowValidationError("meta__timestamp timezone must be UTC")

    seq = df["meta__sequence_id"]
    if str(seq.dtype) != "int64":
        raise VolumeFlowValidationError("meta__sequence_id must be int64")

    if not ts.is_monotonic_increasing:
        raise VolumeFlowValidationError("meta__timestamp must be monotonic increasing")
    if not seq.is_monotonic_increasing:
        raise VolumeFlowValidationError("meta__sequence_id must be monotonic increasing")

    null_counts = df[REQUIRED_COLUMNS].isna().sum()
    non_zero_nulls = {k: int(v) for k, v in null_counts.items() if int(v) > 0}
    if non_zero_nulls:
        raise VolumeFlowValidationError(f"Zero-null policy violated: {non_zero_nulls}")

    for col in FEATURE_COLUMNS:
        if str(df[col].dtype) != "float32":
            raise VolumeFlowValidationError(f"{col} must be float32, got {df[col].dtype}")

    schema = _load_futures_base_schema()
    rules = schema.get("validation_rules", {})
    if not rules.get("strict_dtype_enforcement", False):
        raise VolumeFlowValidationError("Futures base schema does not enforce strict dtypes")
    if not rules.get("fail_on_null", False):
        raise VolumeFlowValidationError("Futures base schema does not enforce null-fail")
    if not rules.get("fail_on_non_monotonic_sequence", False):
        raise VolumeFlowValidationError("Futures base schema does not enforce monotonic sequence")

    return True
