"""Strict validator for Futures Funding Rate dataset."""

from __future__ import annotations

from typing import List

import pandas as pd


class FundingRateValidationError(Exception):
    """Raised when funding-rate dataset fails strict validation."""


REQUIRED_FUNDING_COLUMNS: List[str] = [
    "meta__timestamp",
    "meta__sequence_id",
    "fut__funding_rate",
    "fut__funding_rate_change",
    "fut__funding_rate_velocity",
    "fut__funding_rate_acceleration",
    "fut__funding_rate_zscore",
    "fut__funding_pressure_index",
    "fut__funding_extreme_flag",
    "fut__funding_oi_stress",
    "fut__funding_rate_regime_flag",
]

FLOAT32_COLUMNS: List[str] = [
    "fut__funding_rate",
    "fut__funding_rate_change",
    "fut__funding_rate_velocity",
    "fut__funding_rate_acceleration",
    "fut__funding_rate_zscore",
    "fut__funding_pressure_index",
    "fut__funding_oi_stress",
]


DEPRECATED_COLUMNS = {
    "fut__funding_oi_divergence",
    "fut__funding_price_divergence",
    "fut__predicted_funding_shift",
    "fut__funding_stress_score",
}


def validate_funding_rate_df(df: pd.DataFrame) -> bool:
    missing = [c for c in REQUIRED_FUNDING_COLUMNS if c not in df.columns]
    if missing:
        raise FundingRateValidationError(f"Missing required columns: {missing}")

    extra = [c for c in df.columns if c not in REQUIRED_FUNDING_COLUMNS]
    if extra:
        raise FundingRateValidationError(f"Unexpected non-schema columns present: {extra}")

    deprecated_present = sorted(DEPRECATED_COLUMNS.intersection(set(df.columns)))
    if deprecated_present:
        raise FundingRateValidationError(f"Deprecated funding columns present: {deprecated_present}")

    ts = df["meta__timestamp"]
    if str(ts.dtype) != "datetime64[ns, UTC]":
        raise FundingRateValidationError(f"meta__timestamp must be datetime64[ns, UTC], got {ts.dtype}")

    seq = df["meta__sequence_id"]
    if str(seq.dtype) != "int64":
        raise FundingRateValidationError(f"meta__sequence_id must be int64, got {seq.dtype}")

    if not ts.is_monotonic_increasing:
        raise FundingRateValidationError("meta__timestamp must be monotonic increasing")
    if not seq.is_monotonic_increasing:
        raise FundingRateValidationError("meta__sequence_id must be monotonic increasing")

    tuple_idx = pd.MultiIndex.from_arrays([ts, seq])
    if not tuple_idx.is_monotonic_increasing:
        raise FundingRateValidationError("(meta__timestamp, meta__sequence_id) must be monotonic increasing")

    null_counts = df[REQUIRED_FUNDING_COLUMNS].isna().sum()
    non_zero_nulls = {k: int(v) for k, v in null_counts.items() if int(v) > 0}
    if non_zero_nulls:
        raise FundingRateValidationError(f"Zero-null policy violated: {non_zero_nulls}")

    for col in FLOAT32_COLUMNS:
        if str(df[col].dtype) != "float32":
            raise FundingRateValidationError(f"{col} must be float32, got {df[col].dtype}")

    if str(df["fut__funding_extreme_flag"].dtype) != "bool":
        raise FundingRateValidationError(
            f"fut__funding_extreme_flag must be bool, got {df['fut__funding_extreme_flag'].dtype}"
        )

    if str(df["fut__funding_rate_regime_flag"].dtype) != "int32":
        raise FundingRateValidationError(
            f"fut__funding_rate_regime_flag must be int32, got {df['fut__funding_rate_regime_flag'].dtype}"
        )

    return True
