"""OI acceleration feature engine."""

from __future__ import annotations

import pandas as pd


def add_oi_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    vel = pd.to_numeric(out["fut__oi_velocity"], errors="coerce").fillna(0.0)
    out["fut__oi_acceleration"] = vel.diff().fillna(0.0).astype("float32")
    return out
