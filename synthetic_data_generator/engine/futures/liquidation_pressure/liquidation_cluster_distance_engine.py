"""Nearest liquidation cluster distance estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED = [
    "__liq__price",
    "__liq__oi_build_density",
    "fut__open_interest",
    "fut__oi_zscore",
    "fut__funding_rate_zscore",
]


def add_liquidation_cluster_distance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)

    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise ValueError(f"Cluster distance engine missing required columns: {missing}")

    price = (
    pd.to_numeric(
        out["__liq__price"],
        errors="coerce"
    )
    .ffill()
    .bfill()
    .fillna(0.0))
    oi = pd.to_numeric(out["fut__open_interest"], errors="coerce").fillna(0.0)
    oi_z = pd.to_numeric(out["fut__oi_zscore"], errors="coerce").fillna(0.0)
    funding_z = pd.to_numeric(out["fut__funding_rate_zscore"], errors="coerce").fillna(0.0)
    build_density = pd.to_numeric(out["__liq__oi_build_density"], errors="coerce").fillna(0.0)

    # Position build zones: high OI growth while price relatively stable define crowded entries.
    stable_price = price.pct_change().abs().rolling(90, min_periods=5).median().fillna(0.0)
    build_weight = np.tanh((build_density.abs() * 4.5) + (oi_z.abs() * 0.35) + (funding_z.abs() * 0.20))
    zone_score = build_weight / (1.0 + 40.0 * stable_price)

    zone_price = (price * zone_score).rolling(180, min_periods=10).sum() / zone_score.rolling(180, min_periods=10).sum().replace(0.0, np.nan)
    zone_price = zone_price.ffill().bfill().fillna(price)

    # Estimated trigger buffer narrows under crowding (high zone_score and high OI).
    rel_distance = ((price - zone_price).abs() / price.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    crowding_compression = (1.0 + np.tanh(zone_score * 1.5 + oi.clip(lower=0.0).pct_change().fillna(0.0) * 8.0)).clip(0.35, 2.0)
    cluster_distance = (rel_distance / crowding_compression).ffill().bfill().fillna(0.0)

    out["fut__liquidation_cluster_distance"] = cluster_distance.clip(lower=0.0).astype("float32")
    return out
