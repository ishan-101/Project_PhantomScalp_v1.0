"""OI z-score feature engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_oi_zscore(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    out = df.copy(deep=True)
    oi = pd.to_numeric(out["fut__open_interest"], errors="coerce").fillna(0.0)

    rolling_mean = oi.rolling(window=window, min_periods=1).mean()
    rolling_std = oi.rolling(window=window, min_periods=1).std(ddof=0)
    safe_std = rolling_std.where(rolling_std > 1e-12, 1.0)

    out["fut__oi_zscore"] = ((oi - rolling_mean) / safe_std).replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32")
    return out
