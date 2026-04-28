"""Large trade volume feature engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


class LargeTradeVolumeEngineError(Exception):
    pass


def add_large_trade_volume(df: pd.DataFrame, window: int = 200, quantile: float = 0.95) -> pd.DataFrame:
    """Compute cumulative volume from trades above adaptive rolling quantile threshold."""
    if "trade_size" not in df.columns:
        raise LargeTradeVolumeEngineError("Missing trade_size column")

    out = df.copy(deep=True)
    size = pd.to_numeric(out["trade_size"], errors="coerce")
    if size.isna().any() or (size <= 0).any():
        raise LargeTradeVolumeEngineError("trade_size contains invalid values")

    dynamic_threshold = size.rolling(window=window, min_periods=max(10, window // 4)).quantile(quantile)
    dynamic_threshold = dynamic_threshold.bfill().fillna(size.expanding(min_periods=1).quantile(quantile))

    large_mask = size >= dynamic_threshold
    large_contrib = np.where(large_mask.to_numpy(), size.to_numpy(), 0.0)
    out["fut__large_trade_volume"] = pd.Series(large_contrib, index=out.index).cumsum().astype("float32")
    return out
