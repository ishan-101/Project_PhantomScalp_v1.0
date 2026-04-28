"""Funding-vs-price divergence engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_funding_price_divergence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)

    fr = pd.to_numeric(out["fut__funding_rate"], errors="coerce").fillna(0.0)
    fr_chg = pd.to_numeric(out["fut__funding_rate_change"], errors="coerce").fillna(0.0)
    ret = pd.to_numeric(out.get("__funding_price_return", 0.0), errors="coerce").fillna(0.0)

    ret_vol = ret.rolling(96, min_periods=8).std(ddof=0).replace(0.0, np.nan)
    normalized_ret = (ret / ret_vol).replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-6.0, 6.0) / 6.0
    funding_pressure = np.tanh((fr_chg * 10000.0).clip(-8.0, 8.0) / 2.0)

    # Divergence: funding pressure moving against/without price confirmation.
    divergence = funding_pressure - normalized_ret

    # Hidden leverage buildup: flat price + funding acceleration.
    flat_price = (normalized_ret.abs() < 0.08).astype(float)
    hidden_leverage = flat_price * np.tanh((fr.abs() * 10000.0) / 4.0)

    # Trapped positioning patterns.
    trapped_longs = np.where((normalized_ret < -0.15) & (funding_pressure > 0.15), 1.0, 0.0)
    trapped_shorts = np.where((normalized_ret > 0.15) & (funding_pressure < -0.15), 1.0, 0.0)

    score = divergence + 0.25 * hidden_leverage + 0.2 * (trapped_longs - trapped_shorts)
    out["fut__funding_price_divergence"] = (
        pd.Series(score, index=out.index).replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-1.0, 1.0).astype("float32")
    )
    return out
