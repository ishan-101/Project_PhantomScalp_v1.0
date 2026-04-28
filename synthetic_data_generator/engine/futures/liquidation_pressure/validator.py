"""Strict validator for Futures Liquidation Pressure dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


class LiquidationPressureValidationError(Exception):
    """Raised when liquidation-pressure dataset fails strict validation."""


REQUIRED_COLUMNS: List[str] = [
    "meta__timestamp",
    "meta__sequence_id",
    "fut__long_liquidation_volume",
    "fut__short_liquidation_volume",
    "fut__liquidation_imbalance",
    "fut__liquidation_cluster_distance",
    "fut__liquidation_pressure_index",
    "fut__liquidation_cascade_probability",
    "fut__liquidation_velocity",
    "fut__liquidation_heat_pressure",
]

FEATURE_COLUMNS: List[str] = [c for c in REQUIRED_COLUMNS if c.startswith("fut__")]


def _load_futures_base_schema() -> Dict:
    schema_path = Path(__file__).resolve().parents[1] / "schema" / "Futures Base Schema V1 0 · json"
    with schema_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_liquidation_pressure_df(df: pd.DataFrame) -> bool:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise LiquidationPressureValidationError(f"Missing required columns: {missing}")

    ts = df["meta__timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise LiquidationPressureValidationError("meta__timestamp must be datetime dtype")
    if getattr(ts.dtype, "tz", None) is None:
        raise LiquidationPressureValidationError("meta__timestamp must be UTC tz-aware")
    if str(getattr(ts.dtype, "tz", "")) != "UTC":
        raise LiquidationPressureValidationError("meta__timestamp timezone must be UTC")

    seq = df["meta__sequence_id"]
    if str(seq.dtype) != "int64":
        raise LiquidationPressureValidationError("meta__sequence_id must be int64")

    if not ts.is_monotonic_increasing:
        raise LiquidationPressureValidationError("meta__timestamp must be monotonic increasing")
    if not seq.is_monotonic_increasing:
        raise LiquidationPressureValidationError("meta__sequence_id must be monotonic increasing")

    null_counts = df[REQUIRED_COLUMNS].isna().sum()
    non_zero_nulls = {k: int(v) for k, v in null_counts.items() if int(v) > 0}
    if non_zero_nulls:
        raise LiquidationPressureValidationError(f"Zero-null policy violated: {non_zero_nulls}")

    for col in FEATURE_COLUMNS:
        if str(df[col].dtype) != "float32":
            raise LiquidationPressureValidationError(f"{col} must be float32, got {df[col].dtype}")

    cascade = pd.to_numeric(df["fut__liquidation_cascade_probability"], errors="coerce")
    if ((cascade < 0.0) | (cascade > 1.0)).any():
        raise LiquidationPressureValidationError("fut__liquidation_cascade_probability must be within [0, 1]")

    imbalance = pd.to_numeric(df["fut__liquidation_imbalance"], errors="coerce")
    if ((imbalance < -1.0) | (imbalance > 1.0)).any():
        raise LiquidationPressureValidationError("fut__liquidation_imbalance must be within [-1, 1]")

    schema = _load_futures_base_schema()
    rules = schema.get("validation_rules", {})
    if not rules.get("strict_dtype_enforcement", False):
        raise LiquidationPressureValidationError("Futures base schema does not enforce strict dtypes")
    if not rules.get("fail_on_null", False):
        raise LiquidationPressureValidationError("Futures base schema does not enforce null-fail")
    if not rules.get("fail_on_non_monotonic_sequence", False):
        raise LiquidationPressureValidationError("Futures base schema does not enforce monotonic sequence")

    return True
