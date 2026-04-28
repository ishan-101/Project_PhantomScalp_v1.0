"""Directional liquidation imbalance engine."""

from __future__ import annotations

import pandas as pd


def add_liquidation_imbalance(df: pd.DataFrame, epsilon: float = 1e-9) -> pd.DataFrame:
    out = df.copy(deep=True)

    long_liq = pd.to_numeric(out["fut__long_liquidation_volume"], errors="coerce").fillna(0.0)
    short_liq = pd.to_numeric(out["fut__short_liquidation_volume"], errors="coerce").fillna(0.0)

    imbalance = (long_liq - short_liq) / (long_liq + short_liq + float(epsilon))
    out["fut__liquidation_imbalance"] = imbalance.clip(-1.0, 1.0).astype("float32")
    return out
