"""Engine for log-ratio positioning skew."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_position_skew(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    net_long = pd.to_numeric(out["fut__net_long_position_proxy"], errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0)
    net_short = pd.to_numeric(out["fut__net_short_position_proxy"], errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0)

    eps = 1e-6
    skew = np.log((net_long + eps) / (net_short + eps))
    out["fut__position_skew"] = pd.Series(skew, index=out.index).replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32")
    return out
