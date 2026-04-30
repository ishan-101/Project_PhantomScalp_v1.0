"""Funding/OI stress interaction engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_funding_oi_stress(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    fz = pd.to_numeric(out["fut__funding_rate_zscore"], errors="coerce").fillna(0.0)
    oi_z = pd.to_numeric(out["fut__oi_zscore"], errors="coerce").fillna(0.0)
    stress = (fz * oi_z).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["fut__funding_oi_stress"] = stress.astype("float32")
    return out
