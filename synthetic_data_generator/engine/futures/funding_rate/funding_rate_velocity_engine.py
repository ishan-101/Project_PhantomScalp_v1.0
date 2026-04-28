"""Funding-rate velocity engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_funding_rate_velocity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    delta = pd.to_numeric(out["fut__funding_rate_change"], errors="coerce").fillna(0.0)

    seq = pd.to_numeric(out["meta__sequence_id"], errors="coerce").ffill().fillna(0)
    event_step = seq.diff().abs().replace(0.0, np.nan).fillna(1.0)

    velocity = (delta / event_step.clip(lower=1.0)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["fut__funding_rate_velocity"] = velocity.astype("float32")
    return out
