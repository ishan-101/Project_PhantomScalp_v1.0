"""Strict validator for Futures Leverage Metrics dataset."""

from __future__ import annotations

from typing import List

import pandas as pd


class LeverageMetricsValidationError(Exception):
    """Raised when leverage metrics dataset fails strict validation."""


REQUIRED_COLUMNS: List[str] = [
    "meta__timestamp",
    "meta__sequence_id",
    "fut__oi_volume_interaction",
    "fut__funding_oi_interaction",
    "fut__basis_oi_interaction",
    "fut__leverage_pressure_index",
]

FLOAT32_COLUMNS: List[str] = [
    "fut__oi_volume_interaction",
    "fut__funding_oi_interaction",
    "fut__basis_oi_interaction",
    "fut__leverage_pressure_index",
]


def validate_leverage_metrics_df(df: pd.DataFrame) -> bool:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise LeverageMetricsValidationError(f"Missing required columns: {missing}")

    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra:
        raise LeverageMetricsValidationError(f"Unexpected non-schema columns present: {extra}")

    if str(df["meta__timestamp"].dtype) != "datetime64[ns, UTC]":
        raise LeverageMetricsValidationError(
            f"meta__timestamp must be datetime64[ns, UTC], got {df['meta__timestamp'].dtype}"
        )
    if str(df["meta__sequence_id"].dtype) != "int64":
        raise LeverageMetricsValidationError(
            f"meta__sequence_id must be int64, got {df['meta__sequence_id'].dtype}"
        )

    if not df["meta__timestamp"].is_monotonic_increasing:
        raise LeverageMetricsValidationError("meta__timestamp must be monotonic increasing")
    if not df["meta__sequence_id"].is_monotonic_increasing:
        raise LeverageMetricsValidationError("meta__sequence_id must be monotonic increasing")

    tuple_idx = pd.MultiIndex.from_arrays([df["meta__timestamp"], df["meta__sequence_id"]])
    if not tuple_idx.is_monotonic_increasing:
        raise LeverageMetricsValidationError("(meta__timestamp, meta__sequence_id) must be monotonic increasing")

    null_counts = df[REQUIRED_COLUMNS].isna().sum()
    nulls = {k: int(v) for k, v in null_counts.items() if int(v) > 0}
    if nulls:
        raise LeverageMetricsValidationError(f"Zero-null policy violated: {nulls}")

    for col in FLOAT32_COLUMNS:
        if str(df[col].dtype) != "float32":
            raise LeverageMetricsValidationError(f"{col} must be float32, got {df[col].dtype}")

    return True
