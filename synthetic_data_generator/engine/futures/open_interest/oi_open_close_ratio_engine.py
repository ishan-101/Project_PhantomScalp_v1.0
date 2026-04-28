"""OI open-close ratio feature engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_oi_open_close_ratio(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    opening = pd.to_numeric(out["open_flow"], errors="coerce").fillna(0.0)
    closing = pd.to_numeric(out["close_flow"], errors="coerce").fillna(0.0)

    safe_closing = closing.where(closing > 1e-12, 1.0)
    ratio = (opening / safe_closing).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["fut__oi_open_close_ratio"] = ratio.astype("float32")
    return out
