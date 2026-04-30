# feature_engineering/futures/global_validator.py

from __future__ import annotations

import pandas as pd

# Subfamily validators
from .open_interest.validator import validate_features as validate_oi
from .funding_rate.validator import validate_features as validate_fr
from .basis_structure.validator import validate_features as validate_basis

# Dependencies
from .funding_rate.dependencies import REQUIRES_FEATURES as FR_DEP


def validate_global_features(
    oi_df: pd.DataFrame,
    fr_df: pd.DataFrame,
    basis_df: pd.DataFrame,
) -> None:
    """
    Global validator across futures subfamilies.
    Enforces:
    - Individual module validity
    - Cross-module dependencies
    - Index alignment
    - Merge integrity
    """

    # ------------------------------------------------------------
    # 1. Individual validation
    # ------------------------------------------------------------
    validate_oi(oi_df)
    validate_fr(fr_df)
    validate_basis(basis_df)

    # ------------------------------------------------------------
    # 2. Index alignment
    # ------------------------------------------------------------
    if not (oi_df.index.equals(fr_df.index) and oi_df.index.equals(basis_df.index)):
        raise ValueError("[global] Index mismatch across subfamilies")

    # ------------------------------------------------------------
    # 3. Dependency validation (funding depends on OI)
    # ------------------------------------------------------------
    missing = [col for col in FR_DEP if col not in oi_df.columns]
    if missing:
        raise ValueError(f"[global] Missing upstream OI features required by funding_rate: {missing}")

    # ------------------------------------------------------------
    # 4. Merge integrity
    # ------------------------------------------------------------
    merged = pd.concat([oi_df, fr_df, basis_df], axis=1)

    if merged.columns.duplicated().any():
        raise ValueError("[global] Duplicate feature columns after merge")

    if merged.isnull().any().any():
        raise ValueError("[global] Nulls detected after merge")

    return