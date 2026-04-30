"""Funding pressure index engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_settlement_weight(ts: pd.Series) -> pd.Series:
    minutes_of_day = ts.dt.hour * 60 + ts.dt.minute + (ts.dt.second / 60.0)
    slots = np.array([0.0, 480.0, 960.0, 1440.0], dtype=np.float64)
    arr = minutes_of_day.to_numpy(dtype=np.float64)
    idx = np.searchsorted(slots, arr, side="right")
    next_slots = slots[idx]
    minutes_to_next = next_slots - arr
    centered = ((minutes_to_next - 20.0) / 18.0) ** 2
    return pd.Series(np.exp(-centered), index=ts.index, dtype="float64").clip(0.05, 1.0)


def add_funding_pressure_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    z = pd.to_numeric(out["fut__funding_rate_zscore"], errors="coerce").fillna(0.0)
    vel = pd.to_numeric(out["fut__funding_rate_velocity"], errors="coerce").fillna(0.0)

    if "__funding_settlement_decay_weight" in out.columns:
        settle_w = pd.to_numeric(out["__funding_settlement_decay_weight"], errors="coerce").fillna(0.0)
    else:
        ts = pd.to_datetime(out["meta__timestamp"], utc=True)
        settle_w = _compute_settlement_weight(ts)

    pressure = z * vel * settle_w
    out["fut__funding_pressure_index"] = pressure.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32")
    return out
