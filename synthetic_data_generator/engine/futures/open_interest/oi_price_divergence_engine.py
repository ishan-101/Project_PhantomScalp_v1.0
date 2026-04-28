"""OI-price divergence feature engine."""

from __future__ import annotations

import pandas as pd


def add_oi_price_divergence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)

    price = pd.to_numeric(out["event_price"], errors="coerce").fillna(0.0)
    oi_change = pd.to_numeric(out["fut__oi_change"], errors="coerce").fillna(0.0)

    price_ret = price.pct_change().replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    out["fut__oi_price_divergence"] = (oi_change - price_ret).astype("float32")
    return out
