"""Funding-vs-OI structural divergence engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_funding_oi_divergence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)

    fr = pd.to_numeric(out["fut__funding_rate"], errors="coerce").fillna(0.0)
    dfr = pd.to_numeric(out["fut__funding_rate_change"], errors="coerce").fillna(0.0)
    oi_chg = pd.to_numeric(out["fut__oi_change"], errors="coerce").fillna(0.0)
    oi_z = pd.to_numeric(out.get("fut__oi_zscore", 0.0), errors="coerce").fillna(0.0)

    oi_dir = np.sign(oi_chg)
    funding_dir = np.sign(pd.Series(np.where(dfr.to_numpy() == 0.0, fr.to_numpy(), dfr.to_numpy()), index=out.index))

    # Alignment captures crowded directional build; anti-alignment captures squeeze unwind risk.
    directional_structure = funding_dir * oi_dir
    magnitude = dfr.abs().rolling(32, min_periods=1).mean() + oi_chg.abs().rolling(32, min_periods=1).mean()

    unwind_risk = np.where((dfr > 0) & (oi_chg < 0), 1.0, 0.0) + np.where((dfr < 0) & (oi_chg < 0), 1.0, 0.0)
    build_risk = np.where((dfr > 0) & (oi_chg > 0), 1.0, 0.0) + np.where((dfr < 0) & (oi_chg > 0), 1.0, 0.0)

    divergence = (
        0.45 * directional_structure
        + 0.20 * np.tanh(oi_z / 3.0)
        + 0.20 * (build_risk - unwind_risk)
        + 0.15 * np.tanh(magnitude / (magnitude.rolling(64, min_periods=1).mean() + 1e-9))
    )

    out["fut__funding_oi_divergence"] = (
        pd.Series(divergence, index=out.index).replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-1.0, 1.0).astype("float32")
    )
    return out
