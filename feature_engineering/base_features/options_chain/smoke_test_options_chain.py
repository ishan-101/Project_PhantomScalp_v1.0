"""Smoke test for Options Chain base features (schema -> features -> validator)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_engineering.base_features.options_chain.features import (  # noqa: E402
    FeatureComputationError,
    compute_options_chain_features,
)
from feature_engineering.base_features.options_chain.validator import (  # noqa: E402
    FeatureValidationError,
    validate_options_chain_features,
)


def _print_heading(title: str) -> None:
    print(f"\n=== {title} ===")


def _generate_synthetic_option_chain() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=4, freq="T")
    symbols = ["SPX"]
    strikes = [4400, 4450, 4500]
    option_types = ["call", "put"]

    records = []
    for ts in timestamps:
        spot = 4450 + np.sin(ts.minute) * 5
        time_to_expiry_options = [1, 2]
        for tte in time_to_expiry_options:
            for strike in strikes:
                for opt_type in option_types:
                    open_interest = 1000 + (strike - 4450) * 0.5
                    volume = 50 + (tte * 5) + (1 if opt_type == "call" else 2)
                    implied_volatility = 0.2 + abs(strike - spot) / 10000 + tte * 0.005
                    bid_iv = implied_volatility - 0.01
                    ask_iv = implied_volatility + 0.01
                    records.append(
                        {
                            "ts": ts,
                            "symbol": symbols[0],
                            "spot": spot,
                            "option_type": opt_type,
                            "strike": float(strike),
                            "open_interest": float(open_interest),
                            "volume": float(volume),
                            "implied_volatility": float(implied_volatility),
                            "bid_iv": float(bid_iv),
                            "ask_iv": float(ask_iv),
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
            "volume",
            "implied_volatility",
            "bid_iv",
            "ask_iv",
            "time_to_expiry",
        ],
    )
    print(f"PASS: generated synthetic option records: {len(raw_df)} rows.")

    _print_heading("Step 2 — Compute Options Chain Base Features")
    features_df = compute_options_chain_features(raw_df)
    opt_features = [col for col in features_df.columns if col.startswith("opt__")]
    if len(opt_features) != 15:
        raise FeatureComputationError(f"Expected 15 features, found {len(opt_features)}")
    if features_df[opt_features].isna().any().any():
        raise FeatureComputationError("NaNs present after feature computation")
    print("PASS: computed 15 Options Chain base features with no nulls.")

    _print_heading("Step 3 — Validate Options Chain Base Features")
    validate_options_chain_features(features_df[opt_features])
    print("PASS: validator accepted feature set and enforced schema.")

    ratio = features_df["opt__call_put_oi_ratio"]
    if ((ratio < 0) | (ratio > 1)).any():
        raise AssertionError("Call/put OI ratio out of [0,1] bounds")
    if features_df["opt__iv_crush_detector"].dtype != bool:
        raise AssertionError("iv_crush_detector is not boolean")
    print("PASS: manual invariant checks succeeded.")

    return {"raw": raw_df, "features": features_df}


def _run_corruption_check(features_df: pd.DataFrame) -> None:
    _print_heading("Step 4 — Corruption Check")
    corrupted = features_df.copy()
    corrupted.loc[0, "opt__call_put_oi_ratio"] = 1.5
    try:
        validate_options_chain_features(corrupted[[col for col in corrupted.columns if col.startswith("opt__")]])
    except FeatureValidationError:
        print("PASS: validator correctly rejected corrupted ratio.")
        return
    raise AssertionError("Validator did not reject corrupted ratio > 1")


if __name__ == "__main__":
    try:
        outputs = _run_core_flow()
        _run_corruption_check(outputs["features"])
        print("\noptions_chain base features — SAFE TO FREEZE")
    except Exception as exc:  # pragma: no cover - smoke test guard
        print(f"\noptions_chain base features — DO NOT FREEZE: {exc}")
        raise
