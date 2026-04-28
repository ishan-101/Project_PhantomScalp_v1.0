"""Smoke test for market_regime base feature family."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_engineering.base_features.market_regime.features import compute_market_regime_features  # noqa: E402
from feature_engineering.base_features.market_regime.validator import FeatureValidationError, validate_market_regime_features  # noqa: E402


def _build_synthetic_input() -> pd.DataFrame:
    rows = 5
    data = {
        "price__micro_volatility": np.linspace(0.5, 0.9, rows, dtype=float),
        "opt__best_bid_iv": np.linspace(0.4, 0.6, rows, dtype=float),
        "opt__best_ask_iv": np.linspace(0.5, 0.7, rows, dtype=float),
        "spread__l1": np.linspace(0.01, 0.05, rows, dtype=float),
        "previous_spread__l1": np.linspace(0.015, 0.055, rows, dtype=float),
        "ob__top_level_size_bid": np.full(rows, 100.0),
        "ob__top_level_size_ask": np.full(rows, 110.0),
        "ob__total_depth_bid": np.full(rows, 500.0),
        "ob__total_depth_ask": np.full(rows, 520.0),
        "ob__cancellation_rate": np.linspace(0.1, 0.3, rows, dtype=float),
        "ob__new_order_rate": np.linspace(0.2, 0.4, rows, dtype=float),
        "ob__top_of_book_turnover": np.linspace(0.8, 1.2, rows, dtype=float),
        "ob__queue_resilience": np.linspace(0.6, 0.9, rows, dtype=float),
        "ob__depth_elasticity": np.linspace(0.5, 0.9, rows, dtype=float),
        "ob__hidden_to_visible_ratio": np.linspace(0.1, 0.2, rows, dtype=float),
        "of__signed_volume": np.linspace(1000, 1200, rows, dtype=float),
        "previous_of__signed_volume": np.linspace(950, 1150, rows, dtype=float),
        "of__aggressor_volume_ratio": np.linspace(0.2, 0.5, rows, dtype=float),
        "of__execution_flow_polarity": np.linspace(-0.2, 0.6, rows, dtype=float),
        "of__trade_burst_intensity": np.linspace(0.8, 1.5, rows, dtype=float),
        "price__near_term_return_volatility": np.linspace(0.45, 0.85, rows, dtype=float),
        "price__tick_direction": np.array([1, 1, -1, 1, -1], dtype=float),
        "tick_return": np.linspace(-0.01, 0.02, rows, dtype=float),
        "previous_tick_return": np.linspace(-0.015, 0.015, rows, dtype=float),
        "previous_regime__realized_vol": np.linspace(0.45, 0.75, rows, dtype=float),
        "previous_ob__top_of_book_turnover": np.linspace(0.75, 1.05, rows, dtype=float),
        "previous_ob__queue_resilience": np.linspace(0.55, 0.85, rows, dtype=float),
    }
    return pd.DataFrame(data)


def run_smoke_test() -> None:
    base_input = _build_synthetic_input()
    features = compute_market_regime_features(base_input)

    assert features.shape[1] == 15, "Expected exactly 15 regime features"
    if features.isna().any().any():
        raise AssertionError("Nulls detected in computed features")

    validation_result = validate_market_regime_features(features)
    assert validation_result["validated"], "Validation did not return success flag"

    corrupted = features.copy()
    corrupted.loc[0, "regime__spread_zscore"] = np.inf
    try:
        validate_market_regime_features(corrupted)
        raise AssertionError("Validator did not fail on corrupted feature")
    except FeatureValidationError:
        pass

    print("Input columns =>", list(base_input.columns))
    print("Output features =>", list(features.columns))
    print("market_regime base features — SAFE TO FREEZE")


if __name__ == "__main__":
    run_smoke_test()
