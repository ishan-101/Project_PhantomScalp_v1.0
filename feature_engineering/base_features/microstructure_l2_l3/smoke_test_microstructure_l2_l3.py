"""Smoke test for microstructure L2/L3 base features."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_access import parquet_loader
from feature_engineering.base_features.microstructure_l2_l3.features import (
    FeatureComputationError,
    compute_microstructure_l2_l3_features,
)
from feature_engineering.base_features.microstructure_l2_l3.validator import (
    validate_microstructure_l2_l3_features,
)
from feature_engineering.utils.validation_helpers import ValidationError


OUTPUT_BASE = REPO_ROOT / "synthetic_data_generator" / "outputs" / "orderbook" / "l2"
ORCHESTRATOR = REPO_ROOT / "synthetic_data_generator" / "engine" / "orchestrator.py"


def _latest_parquet(path: Path, fragment: str) -> Path:
    matches = sorted(path.rglob(f"*{fragment}.parquet"))
    if not matches:
        raise FileNotFoundError(f"No parquet files matching '*{fragment}.parquet' under {path}")
    return matches[-1]


def _print_stage(name: str, status: str) -> None:
    print(f"{name}: {status}")


def main() -> None:
    try:
        subprocess.run(["python", str(ORCHESTRATOR)], check=True)
        _print_stage("Synthetic generation", "PASS")

        parquet_path = _latest_parquet(OUTPUT_BASE, "orderbook_l2")
        base_df = parquet_loader.load_parquet(parquet_path)
        base_df = base_df.sort_values("meta__timestamp").reset_index(drop=True)
        _print_stage("Load order book", "PASS")

        features_df = compute_microstructure_l2_l3_features(base_df.head(200))
        new_cols = [c for c in features_df.columns if c not in base_df.columns]
        assert len(new_cols) == 28, "Expected exactly 28 new feature columns"
        if features_df[new_cols].isna().any().any():
            raise FeatureComputationError("NaNs detected in computed features")
        if not ((features_df["ob__imbalance"] <= 1).all() and (features_df["ob__imbalance"] >= -1).all()):
            raise FeatureComputationError("ob__imbalance outside [-1,1]")
        if (features_df["ob__book_entropy"] < 0).any():
            raise FeatureComputationError("ob__book_entropy must be non-negative")
        if not features_df[["ob__hidden_liquidity_indicator", "ob__liquidity_void_flag"]].applymap(
            lambda v: isinstance(v, (bool, pd.BooleanDtype().type))
        ).all().all():
            raise FeatureComputationError("Boolean flags contain non-boolean values")
        _print_stage("Feature computation", "PASS")

        validate_microstructure_l2_l3_features(features_df)
        _print_stage("Validator", "PASS")

        corrupted = features_df.copy()
        corrupted.loc[0, "ob__book_entropy"] = -1.0
        try:
            validate_microstructure_l2_l3_features(corrupted)
        except ValidationError:
            _print_stage("Validator corruption check", "PASS")
        else:
            raise AssertionError("Validator did not fail on corrupted entropy")

        print("microstructure_l2_l3 base features — SAFE TO FREEZE")
    except Exception as exc:  # pylint: disable=broad-except
        print(exc)
        print("microstructure_l2_l3 base features — DO NOT FREEZE")


if __name__ == "__main__":
    main()

