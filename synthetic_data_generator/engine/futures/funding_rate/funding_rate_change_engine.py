"""Funding-rate first-difference engine."""

from __future__ import annotations

import pandas as pd


def add_funding_rate_change(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    fr = pd.to_numeric(out["fut__funding_rate"], errors="coerce").fillna(0.0)
    out["fut__funding_rate_change"] = fr.diff().fillna(0.0).astype("float32")
    return out
