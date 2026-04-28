"""OI change feature engine."""

from __future__ import annotations

import pandas as pd


def add_oi_change(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    oi = pd.to_numeric(out["fut__open_interest"], errors="coerce").fillna(0.0)
    out["fut__oi_change"] = oi.diff().fillna(0.0).astype("float32")
    return out
