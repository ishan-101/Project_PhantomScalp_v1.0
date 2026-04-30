"""Residual liquidation heat-pressure engine."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def add_liquidation_heat_pressure(df: pd.DataFrame, half_life_seconds: float = 60.0) -> pd.DataFrame:
    out = df.copy(deep=True)

    ts = pd.to_datetime(out["meta__timestamp"], utc=True)
    total_liq = (
        pd.to_numeric(out["fut__long_liquidation_volume"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["fut__short_liquidation_volume"], errors="coerce").fillna(0.0)
    ).to_numpy(dtype="float64")

    heat = np.zeros(len(out), dtype="float64")
    decay_lambda = math.log(2.0) / float(half_life_seconds)

    for i in range(len(out)):
        if i == 0:
            heat[i] = total_liq[i]
            continue
        dt_sec = max(0.0, (ts.iloc[i] - ts.iloc[i - 1]).total_seconds())
        decay = math.exp(-decay_lambda * dt_sec)
        heat[i] = (heat[i - 1] * decay) + total_liq[i]

    out["fut__liquidation_heat_pressure"] = pd.Series(heat, index=out.index).clip(lower=0.0).astype("float32")
    return out
