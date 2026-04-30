"""Engine for net-short notional proxy derived from OI-scaled crowding."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_net_short_position_proxy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    oi = pd.to_numeric(out["fut__open_interest"], errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0)
    lsr = pd.to_numeric(out["fut__long_short_ratio"], errors="coerce").replace([np.inf, -np.inf], 1.0).fillna(1.0).clip(1e-6)

    short_share = 1.0 / (1.0 + lsr)
    out["fut__net_short_position_proxy"] = (oi * short_share).astype("float32")
    return out
