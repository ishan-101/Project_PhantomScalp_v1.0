"""Engine for fut__basis_oi_interaction."""

from __future__ import annotations

import pandas as pd


class BasisOIInteractionError(Exception):
    """Raised when required columns for basis-OI interaction are missing."""


def add_basis_oi_interaction(df: pd.DataFrame) -> pd.DataFrame:
    required = ["fut__basis_zscore", "fut__oi_zscore"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise BasisOIInteractionError(f"Missing required columns for fut__basis_oi_interaction: {missing}")

    out = df.copy(deep=True)
    out["fut__basis_oi_interaction"] = (
        pd.to_numeric(out["fut__basis_zscore"], errors="coerce")
        * pd.to_numeric(out["fut__oi_zscore"], errors="coerce")
    )
    out["fut__basis_oi_interaction"] = (
        out["fut__basis_oi_interaction"]
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0.0)
        .astype("float32")
    )
    return out
