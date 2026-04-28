"""Strict validator for Futures Funding Rate dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


class FundingRateValidationError(Exception):
    """Raised when funding-rate dataset fails strict validation."""


REQUIRED_FUNDING_COLUMNS: List[str] = [
    "meta__timestamp",
    "meta__sequence_id",
    "fut__funding_rate",
    "fut__funding_rate_change",
    "fut__funding_rate_velocity",
    "fut__funding_rate_zscore",
    "fut__funding_oi_divergence",
    "fut__funding_price_divergence",
    "fut__predicted_funding_shift",
    "fut__funding_stress_score",
    "fut__funding_regime_flag",
]

FEATURE_COLUMNS: List[str] = [c for c in REQUIRED_FUNDING_COLUMNS if c.startswith("fut__")]


def _load_futures_base_schema() -> Dict:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schema"
        / "Futures Base Schema V1 0 · json"
    )
    with schema_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_funding_rate_df(df: pd.DataFrame) -> bool:
    missing = [c for c in REQUIRED_FUNDING_COLUMNS if c not in df.columns]
    if missing:
        raise FundingRateValidationError(f"Missing required columns: {missing}")

    ts = df["meta__timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise FundingRateValidationError("meta__timestamp must be datetime dtype")
    if getattr(ts.dtype, "tz", None) is None:
        raise FundingRateValidationError("meta__timestamp must be UTC tz-aware")
    if str(getattr(ts.dtype, "tz", "")) != "UTC":
        raise FundingRateValidationError("meta__timestamp timezone must be UTC")

    seq = df["meta__sequence_id"]
    if str(seq.dtype) != "int64":
        raise FundingRateValidationError("meta__sequence_id must be int64")

    if not ts.is_monotonic_increasing:
        raise FundingRateValidationError("meta__timestamp must be monotonic increasing")
    if not seq.is_monotonic_increasing:
        raise FundingRateValidationError("meta__sequence_id must be monotonic increasing")

    null_counts = df[REQUIRED_FUNDING_COLUMNS].isna().sum()
    non_zero_nulls = {k: int(v) for k, v in null_counts.items() if int(v) > 0}
    if non_zero_nulls:
        raise FundingRateValidationError(f"Zero-null policy violated: {non_zero_nulls}")

    for col in FEATURE_COLUMNS:
        if str(df[col].dtype) != "float32":
            raise FundingRateValidationError(f"{col} must be float32, got {df[col].dtype}")

    schema = _load_futures_base_schema()
    rules = schema.get("validation_rules", {})
    if not rules.get("strict_dtype_enforcement", False):
        raise FundingRateValidationError("Futures base schema does not enforce strict dtypes")
    if not rules.get("fail_on_null", False):
        raise FundingRateValidationError("Futures base schema does not enforce null-fail")
    if not rules.get("fail_on_non_monotonic_sequence", False):
        raise FundingRateValidationError("Futures base schema does not enforce monotonic sequence")

    return True
