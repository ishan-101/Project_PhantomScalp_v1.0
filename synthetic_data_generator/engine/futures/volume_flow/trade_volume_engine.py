"""Trade volume feature engine."""

from __future__ import annotations

import pandas as pd


class TradeVolumeEngineError(Exception):
    pass


def add_trade_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative executed trade volume from validated trade_size."""
    if "trade_size" not in df.columns:
        raise TradeVolumeEngineError("Missing trade_size column")

    out = df.copy(deep=True)
    size = pd.to_numeric(out["trade_size"], errors="coerce")
    if (size <= 0).any():
        raise TradeVolumeEngineError("trade_size contains non-positive values; invalid executed-trade input")

    out["fut__trade_volume"] = size.cumsum().astype("float32")
    return out
