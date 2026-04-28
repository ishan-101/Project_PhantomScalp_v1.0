"""Smoke test for orderflow_tick base features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_engineering.base_features.orderflow_tick.features import (
    FeatureComputationError,
    compute_orderflow_tick_features,
)
from feature_engineering.base_features.orderflow_tick.validator import (
    FeatureValidationError,
    validate_orderflow_tick_features,
)


SMALL_TRADE_THRESHOLD = 1.0
LARGE_TRADE_THRESHOLD = 3.0


def _print_heading(title: str) -> None:
    print(f"\n=== {title} ===")


def _make_synthetic_rows(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    rows = []
    prev_signed_volume = 0.0
    prev_price = 100.0
    prev_aggressor = 1

    for _ in range(n):
        trade_count = rng.integers(1, 5)
        prices = np.round(prev_price + rng.normal(0, 0.5, size=trade_count), 4)
        sizes = np.round(rng.uniform(0.1, 5.0, size=trade_count), 4)
        aggressors = rng.choice([-1, 1], size=trade_count)
        timestamps = np.cumsum(rng.uniform(0.01, 0.5, size=trade_count))

        mid_price = float(np.mean([prices.min(), prices.max()]))
        spread = float(np.clip(rng.normal(0.05, 0.01), 0.001, None))
        visible_depth = float(np.clip(rng.normal(50.0, 10.0), 1.0, None))
        vwap = float(np.average(prices, weights=sizes))

        rows.append(
            {
                "trade_price": prices,
                "trade_size": sizes,
                "aggressor_side": aggressors,
                "trade_count": int(trade_count),
                "timestamp": timestamps,
                "mid_price": mid_price,
                "spread": spread,
                "visible_depth": visible_depth,
                "vwap": vwap,
                "previous_trade_price": prev_price,
                "previous_aggressor_side": prev_aggressor,
                "previous_signed_volume": prev_signed_volume,
            }
        )

        prev_price = float(prices[-1])
        prev_signed_volume = float(np.sum(sizes * aggressors))
        prev_aggressor = int(np.sign(prev_signed_volume) or prev_aggressor)

    return pd.DataFrame(rows)


def _assert_no_nans(df: pd.DataFrame, feature_names: list[str]) -> None:
    lingering = {col: int(df[col].isna().sum()) for col in feature_names}
    lingering = {col: count for col, count in lingering.items() if count > 0}
    if lingering:
        raise AssertionError(f"Unexpected NaNs present: {lingering}")


def main() -> None:
    _print_heading("Step 1 — Generate Synthetic Data")
    base_df = _make_synthetic_rows()
    print(f"PASS: generated synthetic dataset with shape {base_df.shape}")

    _print_heading("Step 2 — Compute Orderflow Features")
    features_df = compute_orderflow_tick_features(
        base_df,
        small_trade_threshold=SMALL_TRADE_THRESHOLD,
        large_trade_threshold=LARGE_TRADE_THRESHOLD,
        decay_lambda=0.6,
    )

    expected_cols = [
        "of__signed_volume",
        "of__imbalance_ratio",
        "of__large_trade_count",
        "of__avg_trade_size",
        "of__time_between_trades",
        "of__aggressor_flag_ratio",
        "of__trade_price_vs_vwap",
        "of__run_length_up",
        "of__run_length_down",
        "of__aggressive_buy_size",
        "of__aggressive_sell_size",
        "of__sequence_entropy",
        "of__small_trade_vol",
        "of__medium_trade_vol",
        "of__large_trade_vol",
        "of__vwap_pressure",
        "of__aggressor_volume_ratio",
        "of__execution_flow_polarity",
        "of__market_pressure_tilt",
        "of__impact_adjusted_flow",
        "of__aggression_persistence",
        "of__trade_burst_intensity",
        "of__toxicity_proxy",
        "of__realized_sign_rate",
        "of__price_impact_per_unit_volume",
        "of__initiator_persistence",
        "of__time_decay_of_flow",
    ]

    missing = [col for col in expected_cols if col not in features_df.columns]
    if missing:
        raise FeatureComputationError(f"Missing computed features: {missing}")
    if len(expected_cols) != 27:
        raise FeatureComputationError("Feature set size mismatch; expected 27 features")

    _print_heading("Step 3 — Validate Features")
    validate_orderflow_tick_features(features_df)
    _assert_no_nans(features_df, expected_cols)
    print("PASS: feature validation succeeded with no nulls.")

    _print_heading("Step 4 — Negative Validation Test")
    corrupted = features_df.copy()
    corrupted.loc[0, "of__imbalance_ratio"] = 2.0
    try:
        validate_orderflow_tick_features(corrupted)
    except FeatureValidationError:
        print("PASS: validator rejected corrupted data as expected.")
    else:
        raise AssertionError("Validator did not catch corrupted imbalance ratio")

    print("orderflow_tick base features — SAFE TO FREEZE")


if __name__ == "__main__":
    main()
