import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_engineering.cross_asset_funding.features import (  # noqa: E402
    FeatureComputationError,
    compute_cross_asset_funding_features,
)
from feature_engineering.cross_asset_funding.validator import (  # noqa: E402
    FeatureValidationError,
    validate_cross_asset_funding_features,
)


def _print_heading(title: str) -> None:
    print(f"\n=== {title} ===")


def _generate_synthetic_market_data() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=40, freq="h")
    steps = np.arange(len(timestamps))
    btc_spot = 40000 + steps * 5 + np.sin(steps / 3) * 50
    eth_spot = 2200 + steps * 1.5 + np.cos(steps / 4) * 10
    perp_basis = 30 + np.sin(steps / 5) * 5
    btc_perpetual = btc_spot + perp_basis
    funding_rate = 0.0005 + np.sin(steps / 6) * 0.0001
    dxy_index = 101 + np.cos(steps / 7) * 0.2
    return pd.DataFrame(
        {
            "ts": timestamps,
            "btc_spot": btc_spot,
            "btc_perpetual": btc_perpetual,
            "eth_spot": eth_spot,
            "funding_rate": funding_rate,
            "dxy_index": dxy_index,
        }
    )


def _run_core_flow() -> pd.DataFrame:
    _print_heading("Step 1 — Generate Synthetic Cross-Asset Inputs")
    raw_df = _generate_synthetic_market_data()
    print(f"PASS: generated synthetic inputs with shape {raw_df.shape}.")

    _print_heading("Step 2 — Compute Cross-Asset / Funding Features")
    features_df = compute_cross_asset_funding_features(raw_df)
    cross_cols = [col for col in features_df.columns if col.startswith("cross__")]
    if len(cross_cols) != 9:
        raise FeatureComputationError(f"Expected 9 cross__ features, found {len(cross_cols)}")
    if features_df[cross_cols].isna().any().any():
        raise FeatureComputationError("NaNs present after feature computation")
    print("PASS: computed 9 cross-asset features with no nulls.")

    _print_heading("Step 3 — Validate Feature Schema and Ranges")
    validate_cross_asset_funding_features(features_df[cross_cols])
    if (features_df["cross__btc_dxy_corr_proxy"].abs() > 1.0).any():
        raise AssertionError("Correlation proxy exceeded [-1, 1] bounds.")
    if features_df["cross__risk_on_off_flag"].dtype != bool:
        raise AssertionError("Risk on/off flag is not boolean.")
    print("PASS: validator accepted features and manual range checks succeeded.")

    return features_df


def _run_corruption_check(features_df: pd.DataFrame) -> None:
    _print_heading("Step 4 — Corruption Check")
    corrupted = features_df.copy()
    corrupted.loc[corrupted.index[0], "cross__funding_rate"] = np.inf
    try:
        validate_cross_asset_funding_features(corrupted[[col for col in corrupted.columns if col.startswith("cross__")]])
    except FeatureValidationError:
        print("PASS: validator correctly rejected corrupted funding_rate.")
        return
    raise AssertionError("Validator did not reject corrupted funding_rate spike")


if __name__ == "__main__":
    try:
        outputs: Dict[str, pd.DataFrame] = {}
        features = _run_core_flow()
        outputs["features"] = features
        _run_corruption_check(features)
        print("\ncross_asset_funding base features — SAFE TO FREEZE")
    except Exception as exc:  # pragma: no cover - smoke test guard
        print(f"\ncross_asset_funding base features — DO NOT FREEZE: {exc}")
        raise
