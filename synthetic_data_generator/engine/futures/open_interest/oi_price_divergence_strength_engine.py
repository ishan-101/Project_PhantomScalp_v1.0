"""OI-price divergence strength feature engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_oi_price_divergence_strength(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    out = df.copy(deep=True)
    div = pd.to_numeric(out["fut__oi_price_divergence"], errors="coerce").fillna(0.0)

    scale = div.abs().rolling(window=window, min_periods=1).mean()
    safe_scale = scale.where(scale > 1e-12, 1.0)
    strength = (div.abs() / safe_scale).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    out["fut__oi_price_divergence_strength"] = strength.astype("float32")
    return out
