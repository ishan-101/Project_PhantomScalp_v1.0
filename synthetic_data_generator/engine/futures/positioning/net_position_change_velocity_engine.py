"""Engine for position skew velocity (first derivative of skew)."""

from __future__ import annotations

import pandas as pd


def add_net_position_change_velocity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    skew = pd.to_numeric(out["fut__position_skew"], errors="coerce").replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    out["fut__net_position_change_velocity"] = skew.diff().fillna(0.0).astype("float32")
    return out
