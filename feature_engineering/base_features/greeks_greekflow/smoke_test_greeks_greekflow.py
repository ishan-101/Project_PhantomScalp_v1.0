import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_engineering.base_features.greeks_greekflow.features import (  # noqa: E402
    FeatureComputationError,
    compute_greeks_greekflow_features,
)
from feature_engineering.base_features.greeks_greekflow.validator import (  # noqa: E402
    FeatureValidationError,
    validate_greeks_greekflow_features,
)


def _print_heading(title: str) -> None:
    print(f"\n=== {title} ===")


def _generate_synthetic_option_chain() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=5, freq="T")
    symbols = ["SPX"]
    strikes = [4400, 4450, 4500]
    option_types = ["call", "put"]

    records = []
    for ts in timestamps:
        spot = 4450 + np.sin(ts.minute) * 5
        ttes = [1.0, 2.0]
        for tte in ttes:
            for strike in strikes:
                intrinsic = max(spot - strike, 0.0) if strike <= spot else max(strike - spot, 0.0)
                for opt_type in option_types:
                    option_price = intrinsic + 10 + 0.1 * tte
                    implied_volatility = 0.2 + abs(strike - spot) / 10000 + tte * 0.01
                    records.append(
                        {
                            "ts": ts,
                            "symbol": symbols[0],
                            "spot": float(spot),
                            "option_type": opt_type,
                            "strike": float(strike),
                            "open_interest": float(1000 + (strike - 4450) * 0.2),
                            "implied_volatility": float(implied_volatility),
                            "option_price": float(option_price),
                            "time_to_expiry": float(tte),
                        }
                    )
    return pd.DataFrame.from_records(records)


def _assert_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(missing)}")


def _run_core_flow() -> Dict[str, pd.DataFrame]:
    _print_heading("Step 1 — Generate Synthetic Option Chain Data")
    raw_df = _generate_synthetic_option_chain()
    _assert_columns(
        raw_df,
        [
            "ts",
            "symbol",
            "spot",
            "option_type",
            "strike",
            "open_interest",
            "implied_volatility",
            "option_price",
            "time_to_expiry",
        ],
    )
    print(f"PASS: generated synthetic option records: {len(raw_df)} rows.")

    _print_heading("Step 2 — Compute Greeks & Greek-Flow Base Features")
    features_df = compute_greeks_greekflow_features(raw_df)
    greek_cols = [col for col in features_df.columns if col.startswith("greek__")]
    if len(greek_cols) != 15:
        raise FeatureComputationError(f"Expected 15 features, found {len(greek_cols)}")
    if features_df[greek_cols].isna().any().any():
        raise FeatureComputationError("NaNs present after feature computation")
    print("PASS: computed 15 Greeks & Greek-Flow features with no nulls.")

    _print_heading("Step 3 — Validate Greeks & Greek-Flow Features")
    validate_greeks_greekflow_features(features_df[greek_cols])
    print("PASS: validator accepted feature set and enforced schema.")

    if (features_df["greek__implied_vol_surface_flag"].abs() > 1).any():
        raise AssertionError("implied_vol_surface_flag out of ternary bounds")
    if features_df["greek__gamma_shock_indicator"].dtype != bool:
        raise AssertionError("gamma_shock_indicator is not boolean")
    print("PASS: manual invariant checks succeeded.")

    return {"raw": raw_df, "features": features_df}


def _run_corruption_check(features_df: pd.DataFrame) -> None:
    _print_heading("Step 4 — Corruption Check")
    corrupted = features_df.copy()
    corrupted.loc[0, "greek__gamma_flow"] = np.inf
    try:
        validate_greeks_greekflow_features(corrupted[[col for col in corrupted.columns if col.startswith("greek__")]])
    except FeatureValidationError:
        print("PASS: validator correctly rejected corrupted gamma_flow.")
        return
    raise AssertionError("Validator did not reject corrupted gamma_flow spike")


if __name__ == "__main__":
    try:
        outputs = _run_core_flow()
        _run_corruption_check(outputs["features"])
        print("\ngreeks_greekflow base features — SAFE TO FREEZE")
    except Exception as exc:  # pragma: no cover - smoke test guard
        print(f"\ngreeks_greekflow base features — DO NOT FREEZE: {exc}")
        raise
