"""Funding-rate acceleration engine (second derivative)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_funding_rate_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    velocity = pd.to_numeric(out["fut__funding_rate_velocity"], errors="coerce").fillna(0.0)
    seq = pd.to_numeric(out["meta__sequence_id"], errors="coerce").ffill().fillna(0)
    event_step = seq.diff().abs().replace(0.0, np.nan).fillna(1.0).clip(lower=1.0)
    acceleration = (velocity.diff().fillna(0.0) / event_step).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["fut__funding_rate_acceleration"] = acceleration.astype("float32")
    return out
