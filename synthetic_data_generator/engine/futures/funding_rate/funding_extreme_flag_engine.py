"""Funding extreme flag engine."""

from __future__ import annotations

import pandas as pd


def add_funding_extreme_flag(df: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    out = df.copy(deep=True)
    z = pd.to_numeric(out["fut__funding_rate_zscore"], errors="coerce").fillna(0.0)
    out["fut__funding_extreme_flag"] = (z.abs() > float(threshold)).astype("bool")
    return out
