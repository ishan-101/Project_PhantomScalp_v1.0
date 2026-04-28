"""End-to-end smoke test for Higher-Order engineered features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_engineering.higher_order.features import FeatureComputationError, HigherOrderConfig, compute_higher_order_features
from feature_engineering.higher_order.validator import FeatureValidationError, validate_higher_order_features


def _build_base_features(rows: int = 16) -> pd.DataFrame:
    """Construct a deterministic synthetic base-feature frame."""

    index = pd.RangeIndex(rows)
    price_last = pd.Series(np.linspace(100.0, 101.5, rows), index=index, dtype="float32")
    tick_return = pd.Series(np.linspace(0.0005, 0.0015, rows), index=index, dtype="float32")
    volume_tick = pd.Series(np.linspace(1000, 1400, rows), index=index, dtype="float32")
    spread = pd.Series(np.linspace(0.01, 0.02, rows), index=index, dtype="float32")
    ob_imbalance = pd.Series(np.linspace(-0.2, 0.2, rows), index=index, dtype="float32")
    base = pd.DataFrame(
        {
            "price__last": price_last,
            "price__mid": price_last + 0.01,
            "volume__tick": volume_tick,
            "spread__l1": spread,
            "ob__imbalance": ob_imbalance,
            "ob__top_level_size_bid": pd.Series(500 + np.arange(rows), index=index, dtype="float32"),
            "ob__top_level_size_ask": pd.Series(480 + np.arange(rows), index=index, dtype="float32"),
            "ob__total_depth_bid": pd.Series(5000 + np.arange(rows) * 2, index=index, dtype="float32"),
            "ob__total_depth_ask": pd.Series(5050 + np.arange(rows) * 2, index=index, dtype="float32"),
            "tick_return": tick_return,
            "of__signed_volume": pd.Series(np.linspace(50, -40, rows), index=index, dtype="float32"),
            "of__price_impact_per_unit_volume": pd.Series(np.linspace(0.01, 0.04, rows), index=index, dtype="float32"),
            "regime__liquidity_score": pd.Series(np.linspace(0.2, 0.8, rows), index=index, dtype="float32"),
        }
    )
    return base


def _build_config() -> HigherOrderConfig:
    return HigherOrderConfig(
        baseline_price_mean=100.5,
        baseline_price_std=0.75,
        baseline_volume_mean=1200.0,
        baseline_volume_std=120.0,
        baseline_ob_imbalance_mean=0.0,
        baseline_ob_imbalance_std=0.25,
        pca_book_loadings=(0.4, 0.4, 0.1, 0.1),
        ehlers_coefficients=(
            0.020083365564211235,
            0.04016673112842247,
            0.020083365564211235,
            -1.5610180758007182,
            0.6413515380575631,
        ),
        signed_volume_alpha=0.35,
    )


def main() -> None:
    base_df = _build_base_features()
    config = _build_config()

    print("Step 1 — compute higher-order features")
    features_df = compute_higher_order_features(base_df, config=config)
    if set(features_df.columns) != {
        "ho__z_price",
        "ho__log_return",
        "ho__price_accel",
        "ho__volume_z",
        "ho__ob_imbalance_z",
        "ho__pca1_book",
        "ho__interaction_of_volume_spread",
        "ho__return_signed_volume_interaction",
        "ho__return_over_spread",
        "ho__imbalance_times_return",
        "ho__filtered_mid_price",
        "ho__book_flow_interaction",
        "ho__normalized_impact",
        "ho__signed_volume_ema",
    }:
        raise FeatureComputationError("Unexpected feature columns produced.")

    print("Step 2 — validate higher-order features")
    validation_result = validate_higher_order_features(features_df, base_df=base_df)
    if not validation_result.get("validated"):
        raise FeatureValidationError("Validation did not affirm success.")
    if features_df.isna().any().any():
        raise FeatureValidationError("Null values detected post-validation.")

    print("Step 3 — corrupt one value and ensure validator fails")
    corrupted = features_df.copy()
    corrupted.loc[0, "ho__z_price"] = 10_000.0
    try:
        validate_higher_order_features(corrupted, base_df=base_df)
    except FeatureValidationError:
        print("PASS: validator rejected corrupted payload.")
    else:
        raise AssertionError("Validator did not fail on corrupted payload.")

    print("higher_order base features — SAFE TO FREEZE")


if __name__ == "__main__":
    main()
