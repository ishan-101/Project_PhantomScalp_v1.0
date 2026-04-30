"""Strict validator for Futures Derivatives Regime dataset."""

from __future__ import annotations

from typing import List

import pandas as pd


class DerivativesRegimeValidationError(Exception):
    """Raised when derivatives regime dataset fails strict validation."""


REQUIRED_COLUMNS: List[str] = [
    "meta__timestamp",
    "meta__sequence_id",
    "fut__derivatives_stress_index",
    "fut__leverage_regime_flag",
    "fut__basis_extreme_flag",
]


def validate_derivatives_regime_df(df: pd.DataFrame) -> bool:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DerivativesRegimeValidationError(f"Missing required columns: {missing}")

    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra:
        raise DerivativesRegimeValidationError(f"Unexpected non-schema columns present: {extra}")

    if str(df["meta__timestamp"].dtype) != "datetime64[ns, UTC]":
        raise DerivativesRegimeValidationError(
            f"meta__timestamp must be datetime64[ns, UTC], got {df['meta__timestamp'].dtype}"
        )
    if str(df["meta__sequence_id"].dtype) != "int64":
        raise DerivativesRegimeValidationError(
            f"meta__sequence_id must be int64, got {df['meta__sequence_id'].dtype}"
        )
    if str(df["fut__derivatives_stress_index"].dtype) != "float32":
        raise DerivativesRegimeValidationError(
            f"fut__derivatives_stress_index must be float32, got {df['fut__derivatives_stress_index'].dtype}"
        )
    if str(df["fut__leverage_regime_flag"].dtype) != "int32":
        raise DerivativesRegimeValidationError(
            f"fut__leverage_regime_flag must be int32, got {df['fut__leverage_regime_flag'].dtype}"
        )
    if str(df["fut__basis_extreme_flag"].dtype) != "bool":
        raise DerivativesRegimeValidationError(
            f"fut__basis_extreme_flag must be bool, got {df['fut__basis_extreme_flag'].dtype}"
        )

    if not df["meta__timestamp"].is_monotonic_increasing:
        raise DerivativesRegimeValidationError("meta__timestamp must be monotonic increasing")
    if not df["meta__sequence_id"].is_monotonic_increasing:
        raise DerivativesRegimeValidationError("meta__sequence_id must be monotonic increasing")

    tuple_idx = pd.MultiIndex.from_arrays([df["meta__timestamp"], df["meta__sequence_id"]])
    if not tuple_idx.is_monotonic_increasing:
        raise DerivativesRegimeValidationError("(meta__timestamp, meta__sequence_id) must be monotonic increasing")

    null_counts = df[REQUIRED_COLUMNS].isna().sum()
    nulls = {k: int(v) for k, v in null_counts.items() if int(v) > 0}
    if nulls:
        raise DerivativesRegimeValidationError(f"Zero-null policy violated: {nulls}")

    if df.duplicated(subset=["meta__timestamp", "meta__sequence_id"]).any():
        raise DerivativesRegimeValidationError("Duplicate (meta__timestamp, meta__sequence_id) rows detected")

    return True
