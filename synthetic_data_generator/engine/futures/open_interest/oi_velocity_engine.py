"""OI velocity feature engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_oi_velocity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    ts_us = out["meta__timestamp"].astype("int64") / 1000.0
    step = pd.Series(ts_us).diff().fillna(0.0).to_numpy(dtype="float64")
    safe_step = np.where(step <= 0.0, 1.0, step)

    delta = pd.to_numeric(out["fut__oi_change"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    out["fut__oi_velocity"] = (delta / safe_step).astype("float32")
    return out
