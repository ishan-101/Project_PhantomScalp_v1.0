"""Engine for long/short ratio first-difference."""

from __future__ import annotations

import pandas as pd


def add_long_short_ratio_change(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    lsr = pd.to_numeric(out["fut__long_short_ratio"], errors="coerce").replace([float("inf"), float("-inf")], 1.0).fillna(1.0)
    out["fut__long_short_ratio_change"] = lsr.diff().fillna(0.0).astype("float32")
    return out
