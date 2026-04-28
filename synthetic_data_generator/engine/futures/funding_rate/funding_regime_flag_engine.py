"""Deterministic funding regime classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_funding_regime_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)

    fr = pd.to_numeric(out["fut__funding_rate"], errors="coerce").fillna(0.0)
    z = pd.to_numeric(out["fut__funding_rate_zscore"], errors="coerce").fillna(0.0)
    oi_div = pd.to_numeric(out["fut__funding_oi_divergence"], errors="coerce").fillna(0.0)
    px_div = pd.to_numeric(out["fut__funding_price_divergence"], errors="coerce").fillna(0.0)
    stress = pd.to_numeric(out["fut__funding_stress_score"], errors="coerce").fillna(0.0)

    bull_crowded = (fr > 0) & (z > 1.0) & (stress >= 0.45)
    bear_crowded = (fr < 0) & (z < -1.0) & (stress >= 0.45)

    # Squeeze risk regimes require elevated stress + opposing divergence structure.
    long_squeeze = (fr > 0) & (px_div < -0.25) & (oi_div < 0.0) & (stress >= 0.55)
    short_squeeze = (fr < 0) & (px_div > 0.25) & (oi_div > 0.0) & (stress >= 0.55)

    regime = np.zeros(len(out), dtype=np.float32)
    regime = np.where(bull_crowded, 1.0, regime)
    regime = np.where(bear_crowded, 2.0, regime)
    regime = np.where(long_squeeze, 3.0, regime)
    regime = np.where(short_squeeze, 4.0, regime)

    out["fut__funding_regime_flag"] = pd.Series(regime, index=out.index).astype("float32")
    return out
