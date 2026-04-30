# open_interest/validator.py

from __future__ import annotations

import pandas as pd

from feature_engineering.utils import (
    validate_no_nulls,
    validate_dtypes,
    validate_monotonic_index,
)

from .dependencies import (
    PROVIDES_FEATURES,
)

from .schema import SCHEMA


# ============================================================
# MAIN VALIDATOR
# ============================================================

def validate_features(df: pd.DataFrame) -> None:
    """
    Full validation pipeline for open_interest features.
    Enforces schema, dtype, null policy, and structural integrity.
    """

    # ------------------------------------------------------------
    # 1. Column Presence (Schema Enforcement)
    # ------------------------------------------------------------
    missing_cols = [col for col in PROVIDES_FEATURES if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"[open_interest] Missing required feature columns: {missing_cols}"
        )

    extra_cols = [col for col in df.columns if col not in PROVIDES_FEATURES]
    if extra_cols:
        raise ValueError(
            f"[open_interest] Unexpected extra feature columns: {extra_cols}"
        )

    # ------------------------------------------------------------
    # 2. Index Validation
    # ------------------------------------------------------------
    validate_monotonic_index(df)

    # ------------------------------------------------------------
    # 3. Null Validation (STRICT ZERO-NULL POLICY)
    # ------------------------------------------------------------
    validate_no_nulls(df)

    # ------------------------------------------------------------
    # 4. Dtype Validation (Schema-Aligned)
    # ------------------------------------------------------------
    expected_dtypes = {
        feature: SCHEMA[feature]["dtype"]
        for feature in PROVIDES_FEATURES
    }

    validate_dtypes(df, expected_dtypes)

    # ------------------------------------------------------------
    # 5. Range Validation (Schema Enforcement)
    # ------------------------------------------------------------
    for col, meta in SCHEMA.items():

        series = df[col]

        expected_range = meta["expected_range"]

        if expected_range == "[0, +inf)":
            if (series < 0).any():
                raise ValueError(
                    f"[open_interest] Negative values found in {col}, expected >= 0"
                )

        elif expected_range == "(-inf, +inf)":
            # No constraint
            pass

        # Extendable for future strict ranges

    # ------------------------------------------------------------
    # 6. Determinism Check (Optional Hook)
    # ------------------------------------------------------------
    # Placeholder for future checksum / reproducibility checks

    return