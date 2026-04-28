"""Engine for fut__oi_volume_interaction."""

from __future__ import annotations

import pandas as pd


class OIVolumeInteractionError(Exception):
    """Raised when required columns for OI-volume interaction are missing."""


def add_oi_volume_interaction(df: pd.DataFrame) -> pd.DataFrame:
    required = ["fut__oi_zscore", "fut__volume_delta_ratio"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise OIVolumeInteractionError(f"Missing required columns for fut__oi_volume_interaction: {missing}")

    out = df.copy(deep=True)
    out["fut__oi_volume_interaction"] = (
        pd.to_numeric(out["fut__oi_zscore"], errors="coerce")
        * pd.to_numeric(out["fut__volume_delta_ratio"], errors="coerce")
    )
    out["fut__oi_volume_interaction"] = (
        out["fut__oi_volume_interaction"]
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0.0)
        .astype("float32")
    )
    return out
