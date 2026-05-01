"""Financial semantics checks for open-interest features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import compute_features


OI = "fut__open_interest__mtf-none__strike-none__maturity-none"
OI_CHG = "fut__oi_change__mtf-none__strike-none__maturity-none"
DIV = "fut__oi_price_divergence__mtf-none__strike-none__maturity-none"
ZS = "fut__oi_zscore__mtf-none__strike-none__maturity-none"
TURN = "fut__oi_turnover__mtf-none__strike-none__maturity-none"


def run_semantic_checks() -> None:
    idx = pd.date_range("2025-02-01", periods=180, freq="min", tz="UTC")

    t = np.arange(len(idx), dtype=float)
    oi = 1200.0 + 50.0 * np.sin(t / 6.0)
    price = 100.0 + 5.0 * np.sin(t / 8.0) + 0.05 * t
    volume = 100.0 + 20.0 * np.cos(t / 5.0)

    snapshot = pd.DataFrame(
        {"open_interest": oi, "price": price, "volume": volume},
        index=idx,
    )

    out = compute_features(snapshot, pd.DataFrame(index=idx), {"rolling_window": 30})

    inc_mask = out[OI].diff() > 0
    if not (out.loc[inc_mask, OI_CHG] > 0).all():
        raise AssertionError("Semantic check failed: OI increase should imply positive oi_change")

    snapshot2 = snapshot.copy()
    snapshot2.loc[snapshot2.index[60:120], "open_interest"] -= np.linspace(0.0, 120.0, 60)
    snapshot2.loc[snapshot2.index[60:120], "price"] += np.linspace(0.0, 20.0, 60)
    out2 = compute_features(snapshot2, pd.DataFrame(index=idx), {"rolling_window": 30})

    seg = out2.index[80:110]
    if not (out2.loc[seg, DIV] < 0).any():
        raise AssertionError("Semantic check failed: price up with OI down should create negative divergence")

    z_mean = float(out[ZS].mean())
    if abs(z_mean) > 0.5:
        raise AssertionError(f"Semantic check failed: z-score mean too far from zero ({z_mean})")

    if not (out[TURN] >= 0).all():
        raise AssertionError("Semantic check failed: turnover must be non-negative")


if __name__ == "__main__":
    run_semantic_checks()
    print("Open-interest semantic checks passed.")
