"""OI turnover feature engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_oi_turnover(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    num = pd.to_numeric(out["fut__oi_change"], errors="coerce").fillna(0.0).abs()
    den = pd.to_numeric(out["fut__open_interest"], errors="coerce").fillna(0.0)
    safe_den = den.where(den > 1e-12, 1.0)

    turnover = (num / safe_den).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["fut__oi_turnover"] = turnover.astype("float32")
    return out
