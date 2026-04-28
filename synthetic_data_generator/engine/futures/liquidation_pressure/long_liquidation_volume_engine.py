"""Long liquidation volume estimator (forced sell-side liquidation flow)."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED = [
    "__liq__sell_aggr_notional",
    "__liq__price_return",
    "fut__oi_velocity",
    "fut__funding_rate_zscore",
    "fut__funding_oi_stress",
]


def add_long_liquidation_volume(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)

    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise ValueError(f"Long liquidation engine missing required columns: {missing}")

    sell_notional = pd.to_numeric(out["__liq__sell_aggr_notional"], errors="coerce").fillna(0.0)
    px_ret = pd.to_numeric(out["__liq__price_return"], errors="coerce").fillna(0.0)
    oi_vel = pd.to_numeric(out["fut__oi_velocity"], errors="coerce").fillna(0.0)
    fr_z = pd.to_numeric(out["fut__funding_rate_zscore"], errors="coerce").fillna(0.0)
    crowd_stress = pd.to_numeric(out["fut__funding_oi_stress"], errors="coerce").fillna(0.0)

    sell_spike = np.tanh(sell_notional / (sell_notional.rolling(180, min_periods=5).median().replace(0.0, np.nan))).fillna(0.0)
    down_impulse = np.tanh((-px_ret).clip(lower=0.0) / (px_ret.abs().rolling(240, min_periods=10).median().replace(0.0, np.nan))).fillna(0.0)
    oi_collapse = np.tanh((-oi_vel).clip(lower=0.0) * 7.5)

    # positive funding => long crowding vulnerability, lagged to avoid lookahead.
    prior_positive_funding = (fr_z.shift(1).fillna(0.0) + crowd_stress.shift(1).fillna(0.0)).clip(lower=0.0)
    crowding_gate = np.tanh(prior_positive_funding)

    liquidation_intensity = sell_spike * down_impulse * oi_collapse * crowding_gate

    # Convert intensity into forced execution estimate anchored on aggressive sell notional.
    estimated = sell_notional * liquidation_intensity.clip(0.0, 1.0)

    out["fut__long_liquidation_volume"] = (
        pd.Series(estimated, index=out.index)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .clip(lower=0.0)
        .astype("float32")
    )
    return out
