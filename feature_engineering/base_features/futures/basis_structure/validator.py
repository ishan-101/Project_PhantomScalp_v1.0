# basis_structure/validator.py

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
    Full validation pipeline for basis_structure features.
    Enforces schema, dtype, null policy, and structural integrity.
    """

    # ------------------------------------------------------------
    # 1. Column Presence
    # ------------------------------------------------------------
    missing_cols = [col for col in PROVIDES_FEATURES if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"[basis_structure] Missing required feature columns: {missing_cols}"
        )

    extra_cols = [col for col in df.columns if col not in PROVIDES_FEATURES]
    if extra_cols:
        raise ValueError(
            f"[basis_structure] Unexpected extra feature columns: {extra_cols}"
        )

    # ------------------------------------------------------------
    # 2. Index Validation
    # ------------------------------------------------------------
    validate_monotonic_index(df)

    # ------------------------------------------------------------
    # 3. Null Validation (ZERO-NULL POLICY)
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
    for col in PROVIDES_FEATURES:

        series = df[col]
        meta = SCHEMA[col]
        expected_range = meta["expected_range"]

        if expected_range == "[-1, 0, 1]":
            if not series.isin([-1, 0, 1]).all():
                raise ValueError(
                    f"[basis_structure] Invalid regime values in {col}"
                )

        elif expected_range == "[0, +inf)":
            if (series < 0).any():
                raise ValueError(
                    f"[basis_structure] Negative values found in {col}"
                )

        # (-inf, +inf) → no constraint

    # ------------------------------------------------------------
    # 6. Financial Sanity Checks
    # ------------------------------------------------------------
    basis_col = "fut__basis__mtf-none__strike-none__maturity-none"

    if not pd.api.types.is_numeric_dtype(df[basis_col]):
        raise ValueError("[basis_structure] basis must be numeric")

    # basis should not explode to extreme values (sanity bound)
    if df[basis_col].abs().max() > 10:
        raise ValueError(
            "[basis_structure] basis values are unrealistically large (check normalization)"
        )

    return
