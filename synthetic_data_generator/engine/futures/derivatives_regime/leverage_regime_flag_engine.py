"""Engine for fut__leverage_regime_flag."""

from __future__ import annotations

import numpy as np
import pandas as pd


class LeverageRegimeFlagEngineError(Exception):
    """Raised when leverage-regime classification cannot be computed."""


REQUIRED_INPUT_COLS = ["fut__oi_zscore", "fut__funding_oi_stress"]
EXPANSION_THRESHOLD = 1.0
DELEVERAGE_THRESHOLD = -1.0
HYSTERESIS_EXIT_BAND = 0.35


def add_leverage_regime_flag(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        raise LeverageRegimeFlagEngineError(f"Missing required input columns: {missing}")

    out = df.copy(deep=True)
    oi = pd.to_numeric(out["fut__oi_zscore"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    funding_stress = pd.to_numeric(out["fut__funding_oi_stress"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    leverage_score = 0.5 * oi + 0.5 * funding_stress

    flags = np.zeros(len(out), dtype=np.int32)
    prev = np.int32(0)
    for i, score in enumerate(leverage_score.to_numpy(dtype="float64")):
        if prev == 1:
            if score < HYSTERESIS_EXIT_BAND:
                prev = 0
        elif prev == -1:
            if score > -HYSTERESIS_EXIT_BAND:
                prev = 0
        else:
            if score >= EXPANSION_THRESHOLD:
                prev = 1
            elif score <= DELEVERAGE_THRESHOLD:
                prev = -1
        flags[i] = prev

    out["fut__leverage_regime_flag"] = pd.Series(flags, index=out.index, dtype="int32")
    return out
