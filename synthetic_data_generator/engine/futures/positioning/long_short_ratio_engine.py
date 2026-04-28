"""Engine to derive institutionally-informed long/short crowding ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_long_short_ratio(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)

    fr = pd.to_numeric(out["fut__funding_rate"], errors="coerce").fillna(0.0)
    fr_z = pd.to_numeric(out["fut__funding_rate_zscore"], errors="coerce").fillna(0.0)
    oi_chg = pd.to_numeric(out["fut__oi_change"], errors="coerce").fillna(0.0)
    oi_z = pd.to_numeric(out["fut__oi_zscore"], errors="coerce").fillna(0.0)
    stress = pd.to_numeric(out["__pos__funding_oi_stress"], errors="coerce").fillna(0.0)
    regime = pd.to_numeric(out["fut__funding_rate_regime_flag"], errors="coerce").fillna(0.0)

    oi_expansion = np.tanh((oi_chg * 35.0) + (oi_z * 0.55))
    directional_pressure = np.tanh((fr * 2400.0) + (fr_z * 0.42))
    regime_persistence = pd.Series(regime, index=out.index).ewm(alpha=0.10, adjust=False).mean().clip(-1.0, 1.0)
    stress_boost = np.tanh(stress * 1.4)

    crowding_signal = (
        0.40 * directional_pressure
        + 0.30 * (directional_pressure * oi_expansion)
        + 0.18 * regime_persistence
        + 0.12 * (stress_boost * np.sign(directional_pressure))
    )

    # map bounded crowding signal [-1,1] into stable ratio [0.5,2.0], neutral at 1.0
    ratio = np.exp(np.clip(crowding_signal, -0.693147, 0.693147))
    ratio_series = pd.Series(ratio, index=out.index).replace([np.inf, -np.inf], 1.0).fillna(1.0).clip(0.5, 2.0)

    out["fut__long_short_ratio"] = ratio_series.astype("float32")
    return out
