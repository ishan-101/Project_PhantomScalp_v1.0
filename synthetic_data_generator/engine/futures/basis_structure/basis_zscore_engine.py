"""Basis z-score engine (past-only rolling)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_basis_zscore(df: pd.DataFrame, window: int = 512, min_periods: int = 32) -> pd.DataFrame:
    out = df.copy(deep=True)
    basis = pd.to_numeric(out["fut__perp_spot_basis"], errors="coerce").fillna(0.0)

    mu = basis.rolling(window=window, min_periods=min_periods).mean().shift(1)
    sd = basis.rolling(window=window, min_periods=min_periods).std(ddof=0).shift(1).replace(0.0, np.nan)
    z = ((basis - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["fut__basis_zscore"] = z.clip(-8.0, 8.0).astype("float32")
    return out
