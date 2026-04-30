"""Funding regime flag engine with hysteresis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_funding_rate_regime_flag(
    df: pd.DataFrame,
    enter_threshold: float = 0.00010,
    exit_threshold: float = 0.00005,
) -> pd.DataFrame:
    out = df.copy(deep=True)
    fr = pd.to_numeric(out["fut__funding_rate"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    regime = np.zeros(len(out), dtype=np.int32)
    state = 0
    for i, value in enumerate(fr):
        if state == 0:
            if value >= enter_threshold:
                state = 1
            elif value <= -enter_threshold:
                state = -1
        elif state == 1:
            if value <= exit_threshold:
                state = 0
        elif state == -1:
            if value >= -exit_threshold:
                state = 0
        regime[i] = state

    out["fut__funding_rate_regime_flag"] = pd.Series(regime, index=out.index, dtype="int32")
    return out
