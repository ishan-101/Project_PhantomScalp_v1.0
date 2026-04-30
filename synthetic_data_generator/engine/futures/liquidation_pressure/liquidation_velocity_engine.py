"""Liquidation velocity engine."""

from __future__ import annotations

import pandas as pd


def add_liquidation_velocity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)

    total_liq = (
        pd.to_numeric(out["fut__long_liquidation_volume"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["fut__short_liquidation_volume"], errors="coerce").fillna(0.0)
    )

    baseline = total_liq.ewm(span=90, adjust=False, min_periods=5).mean().replace(0.0, pd.NA)
    velocity = (total_liq.diff().fillna(0.0) / baseline).fillna(0.0)
    out["fut__liquidation_velocity"] = velocity.clip(-5.0, 5.0).astype("float32")
    return out
