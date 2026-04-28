"""Funding stress score engine (liquidation precursor metric)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_funding_stress_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)

    fr = pd.to_numeric(out["fut__funding_rate"], errors="coerce").fillna(0.0)
    z = pd.to_numeric(out["fut__funding_rate_zscore"], errors="coerce").fillna(0.0)
    oi_div = pd.to_numeric(out["fut__funding_oi_divergence"], errors="coerce").fillna(0.0)
    px_div = pd.to_numeric(out["fut__funding_price_divergence"], errors="coerce").fillna(0.0)
    shift = pd.to_numeric(out["fut__predicted_funding_shift"], errors="coerce").fillna(0.0)

    oi_expand = pd.to_numeric(out.get("__funding_oi_expansion_pressure", 0.0), errors="coerce").fillna(0.0)
    extension = pd.to_numeric(out.get("__funding_directional_extension", 0.0), errors="coerce").fillna(0.0)

    extreme_funding = np.tanh((fr.abs() * 10000.0) / 6.0)
    extreme_z = np.tanh(np.abs(z) / 3.0)
    abnormal_oi = np.tanh(np.abs(oi_expand) * 6.0)
    divergence = np.tanh((np.abs(oi_div) + np.abs(px_div)) * 1.2)
    trapped_risk = np.tanh((np.abs(extension) + np.abs(shift)) * 1.1)

    stress = 0.28 * extreme_funding + 0.22 * extreme_z + 0.20 * abnormal_oi + 0.18 * divergence + 0.12 * trapped_risk
    out["fut__funding_stress_score"] = (
        pd.Series(stress, index=out.index).replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(0.0, 1.0).astype("float32")
    )
    return out
