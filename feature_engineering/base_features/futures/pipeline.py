# feature_engineering/futures/pipeline.py

from __future__ import annotations

import pandas as pd

# Feature modules
from .open_interest.features import compute_features as compute_oi
from .funding_rate.features import compute_features as compute_fr
from .basis_structure.features import compute_features as compute_basis

# Global validator
from .global_validator import validate_global_features


def run_futures_pipeline(
    snapshot: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Minimal futures feature pipeline.

    Steps:
    1. Compute open_interest features
    2. Compute funding_rate (uses OI)
    3. Compute basis_structure
    4. Merge all features
    5. Run global validation
    """

    # ------------------------------------------------------------
    # 1. Open Interest
    # ------------------------------------------------------------
    oi_df = compute_oi(snapshot, config)

    # ------------------------------------------------------------
    # 2. Funding Rate (depends on OI)
    # ------------------------------------------------------------
    fr_df = compute_fr(snapshot, oi_df, config)

    # ------------------------------------------------------------
    # 3. Basis Structure
    # ------------------------------------------------------------
    basis_df = compute_basis(snapshot, config)

    # ------------------------------------------------------------
    # 4. Merge
    # ------------------------------------------------------------
    merged = pd.concat([oi_df, fr_df, basis_df], axis=1)

    # ------------------------------------------------------------
    # 5. Global Validation
    # ------------------------------------------------------------
    validate_global_features(oi_df, fr_df, basis_df)

    return merged