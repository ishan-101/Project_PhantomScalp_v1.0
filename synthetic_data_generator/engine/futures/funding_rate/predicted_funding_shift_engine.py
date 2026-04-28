"""Deterministic early-warning predicted funding shift engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_predicted_funding_shift(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)

    vel = pd.to_numeric(out["fut__funding_rate_velocity"], errors="coerce").fillna(0.0)
    z = pd.to_numeric(out["fut__funding_rate_zscore"], errors="coerce").fillna(0.0)
    oi_div = pd.to_numeric(out["fut__funding_oi_divergence"], errors="coerce").fillna(0.0)
    px_div = pd.to_numeric(out["fut__funding_price_divergence"], errors="coerce").fillna(0.0)

    crowd = pd.to_numeric(out.get("__funding_trade_imbalance", 0.0), errors="coerce").fillna(0.0)
    flow = pd.to_numeric(out.get("__funding_orderflow_imbalance", 0.0), errors="coerce").fillna(0.0)

    # Flip-likelihood rises when velocity opposes stretched z-score and divergence stress is elevated.
    stretch_reversal = -np.sign(z) * np.tanh((vel * 5000.0).clip(-5.0, 5.0))
    divergence_stress = 0.55 * oi_div + 0.45 * px_div
    crowd_instability = np.tanh((crowd + flow) * 1.2) * np.tanh(np.abs(z) / 2.5)

    shift_intensity = 0.45 * stretch_reversal + 0.35 * divergence_stress + 0.20 * crowd_instability

    out["fut__predicted_funding_shift"] = (
        pd.Series(shift_intensity, index=out.index)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .clip(-1.0, 1.0)
        .astype("float32")
    )
    return out
