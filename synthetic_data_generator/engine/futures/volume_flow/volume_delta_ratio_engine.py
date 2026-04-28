"""Volume delta ratio feature engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


class VolumeDeltaRatioEngineError(Exception):
    pass


def add_volume_delta_ratio(df: pd.DataFrame) -> pd.DataFrame:
    required = ["fut__volume_delta", "fut__trade_volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise VolumeDeltaRatioEngineError(f"Missing columns: {missing}")

    out = df.copy(deep=True)
    delta = pd.to_numeric(out["fut__volume_delta"], errors="coerce")
    total = pd.to_numeric(out["fut__trade_volume"], errors="coerce")
    if total.isna().any() or (total <= 0).any():
        raise VolumeDeltaRatioEngineError("fut__trade_volume contains invalid values for normalization")

    ratio = delta / total
    ratio = np.clip(ratio, -1.0, 1.0)
    out["fut__volume_delta_ratio"] = pd.Series(ratio, index=out.index).astype("float32")
    return out
