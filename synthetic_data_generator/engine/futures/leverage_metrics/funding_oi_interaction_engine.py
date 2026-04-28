"""Engine for fut__funding_oi_interaction."""

from __future__ import annotations

import pandas as pd


class FundingOIInteractionError(Exception):
    """Raised when required columns for funding-OI interaction are missing."""


def add_funding_oi_interaction(df: pd.DataFrame) -> pd.DataFrame:
    required = ["fut__funding_rate_zscore", "fut__oi_zscore"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise FundingOIInteractionError(f"Missing required columns for fut__funding_oi_interaction: {missing}")

    out = df.copy(deep=True)
    out["fut__funding_oi_interaction"] = (
        pd.to_numeric(out["fut__funding_rate_zscore"], errors="coerce")
        * pd.to_numeric(out["fut__oi_zscore"], errors="coerce")
    )
    out["fut__funding_oi_interaction"] = (
        out["fut__funding_oi_interaction"]
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0.0)
        .astype("float32")
    )
    return out
