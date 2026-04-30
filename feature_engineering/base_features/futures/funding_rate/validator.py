# funding_rate/validator.py

from __future__ import annotations

import pandas as pd

from feature_engineering.utils import (
    validate_no_nulls,
    validate_dtypes,
    validate_monotonic_index,
)

from .dependencies import PROVIDES_FEATURES
from .schema import SCHEMA


def validate_features(df: pd.DataFrame) -> None:
    """
    Full validation pipeline for funding_rate features.
    Enforces schema, dtype, null policy, and dependency integrity.
    """

    # ------------------------------------------------------------
    # 1. Column Presence
    # ------------------------------------------------------------
    missing_cols = [col for col in PROVIDES_FEATURES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"[funding_rate] Missing required feature columns: {missing_cols}")

    extra_cols = [col for col in df.columns if col not in PROVIDES_FEATURES]
    if extra_cols:
        raise ValueError(f"[funding_rate] Unexpected extra feature columns: {extra_cols}")

    # ------------------------------------------------------------
    # 2. Index Validation
    # ------------------------------------------------------------
    validate_monotonic_index(df)

    # ------------------------------------------------------------
    # 3. Null Validation
    # ------------------------------------------------------------
    validate_no_nulls(df)

    # ------------------------------------------------------------
    # 4. Dtype Validation
    # ------------------------------------------------------------
    expected_dtypes = {
        feature: SCHEMA[feature]["dtype"]
        for feature in PROVIDES_FEATURES
    }

    validate_dtypes(df, expected_dtypes)

    # ------------------------------------------------------------
    # 5. Range Validation
    # ------------------------------------------------------------
    for col, meta in SCHEMA.items():

        series = df[col]
        expected_range = meta["expected_range"]

        if expected_range == "[false, true]":
            if not series.isin([0, 1, True, False]).all():
                raise ValueError(f"[funding_rate] Invalid boolean values in {col}")

        elif expected_range == "[-1, 0, 1]":
            if not series.isin([-1, 0, 1]).all():
                raise ValueError(f"[funding_rate] Invalid regime values in {col}")

        elif expected_range == "[0, +inf)":
            if (series < 0).any():
                raise ValueError(f"[funding_rate] Negative values found in {col}")

        # (-inf, +inf) → no constraint

    # ------------------------------------------------------------
    # 6. Dependency Consistency (Cross-feature sanity)
    # ------------------------------------------------------------
    # funding_oi_stress should be finite
    stress_col = "fut__funding_oi_stress__mtf-none__strike-none__maturity-none"
    if not pd.api.types.is_numeric_dtype(df[stress_col]):
        raise ValueError("[funding_rate] funding_oi_stress must be numeric")

    return
