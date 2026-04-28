"""Funding-rate rolling past-only z-score engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_funding_rate_zscore(df: pd.DataFrame, window: int = 256) -> pd.DataFrame:
    out = df.copy(deep=True)
    fr = pd.to_numeric(out["fut__funding_rate"], errors="coerce").fillna(0.0)

    mu = fr.rolling(window=window, min_periods=16).mean().shift(1)
    sd = fr.rolling(window=window, min_periods=16).std(ddof=0).shift(1)

    z = (fr - mu) / sd.replace(0.0, np.nan)
    out["fut__funding_rate_zscore"] = (
        z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8.0, 8.0).astype("float32")
    )
    return out
