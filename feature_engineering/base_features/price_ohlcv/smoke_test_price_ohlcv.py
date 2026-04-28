"""Linear smoke test for price_ohlcv base features.

Execution follows the prescribed steps:
1) run synthetic data orchestrator,
2) load parquet outputs via data_access (loader -> dtype_enforcer -> aligner -> validator),
3) compute price_ohlcv base features,
4) validate features,
5) assert manual invariants,
6) intentionally corrupt one value to ensure validation fails.

This script does not modify production code; it only exercises the existing stack.
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import Iterable, Tuple
import json

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_access import aligner, dtype_enforcer, parquet_loader, raw_schema, validator
from feature_engineering.base_features.price_ohlcv.features import (
    FeatureComputationError,
    compute_price_ohlcv_features,
)
from feature_engineering.base_features.price_ohlcv.validator import (
    FeatureValidationError,
    validate_price_ohlcv_features,
)


OUTPUT_BASE = REPO_ROOT / "synthetic_data_generator" / "outputs"
ORCHESTRATOR = REPO_ROOT / "synthetic_data_generator" / "engine" / "orchestrator.py"


def _print_heading(title: str) -> None:
    print(f"\n=== {title} ===")


def _latest_parquet(path: Path, name_fragment: str) -> Path:
    candidates = sorted(path.rglob(f"*{name_fragment}*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet files matching '*{name_fragment}*.parquet' under {path}")
    return candidates[-1]


def _assert_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(missing)}")


def _run_orchestrator() -> None:
    _print_heading("Step 1 — Generate Synthetic Data")
    result = subprocess.run([sys.executable, str(ORCHESTRATOR)], cwd=REPO_ROOT, check=True)
    if result.returncode != 0:
        raise SystemExit("orchestrator exited with non-zero status")

    parquet_files = list(OUTPUT_BASE.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("No parquet outputs found after orchestrator run.")
    print(f"PASS: orchestrator completed with {len(parquet_files)} parquet files.")


def _load_trades_frame() -> Tuple[pd.DataFrame, Path]:
    trades_dir = OUTPUT_BASE / "ticks_and_orderflow" / "trades"
    parquet_path = _latest_parquet(trades_dir, "ticks_trades")
    raw_df = parquet_loader.load_parquet(parquet_path)
    renamed = raw_df.rename(columns={"meta__timestamp": "ts"})
    required_cols = ["ts", "symbol", "price", "size"]
    _assert_columns(renamed, required_cols)
    trimmed = renamed[required_cols + [col for col in ["side"] if col in renamed.columns]].copy()

    schema = raw_schema.TRADES_SCHEMA
    coerced = dtype_enforcer.enforce_dtypes(trimmed, schema)
    aligned = aligner.sort_and_validate_timestamp(coerced, schema, remove_exact_duplicates=True)
    validator.validate(aligned, schema)
    print("PASS: trades feed loaded, dtype-enforced, aligned, and validated.")
    return aligned.reset_index(drop=True), parquet_path


def _load_orderbook_frame() -> Tuple[pd.DataFrame, Path]:
    ob_dir = OUTPUT_BASE / "orderbook" / "l1"
    parquet_path = _latest_parquet(ob_dir, "orderbook_l1")
    raw_df = parquet_loader.load_parquet(parquet_path)
    renamed = raw_df.rename(
        columns={
            "meta__timestamp": "ts",
            "top_bid": "bid_price",
            "top_ask": "ask_price",
            "bid_size_0": "bid_size",
            "ask_size_0": "ask_size",
        }
    )
    required_cols = ["ts", "symbol", "bid_price", "bid_size", "ask_price", "ask_size"]
    _assert_columns(renamed, required_cols)
    keep_cols = required_cols + [col for col in ["midpoint"] if col in renamed.columns]
    trimmed = renamed[keep_cols].copy()

    schema = raw_schema.ORDERBOOK_SNAPSHOT_SCHEMA
    coerced = dtype_enforcer.enforce_dtypes(trimmed, schema)
    aligned = aligner.sort_and_validate_timestamp(coerced, schema, remove_exact_duplicates=True)
    validator.validate(aligned, schema)
    print("PASS: orderbook L1 feed loaded, dtype-enforced, aligned, and validated.")
    return aligned.reset_index(drop=True), parquet_path


def _prepare_feature_inputs(trades: pd.DataFrame, orderbook: pd.DataFrame) -> pd.DataFrame:
    _print_heading("Step 3 — Compute Price / OHLCV Base Features")
    merged = pd.merge_asof(
        trades.sort_values("ts"),
        orderbook.sort_values("ts"),
        on="ts",
        direction="nearest",
        suffixes=("_trade", "_book"),
    )
    working = merged.head(500).copy()

    side_col = working["side"] if "side" in working.columns else None
    buy_volume = working["size"] if side_col is None else working["size"].where(side_col == "buy", 0.0)
    sell_volume = np.zeros(len(working)) if side_col is None else working["size"].where(side_col == "sell", 0.0)

    base = pd.DataFrame(
        {
            "price__last": working["price"],
            "price__bid": working["bid_price"],
            "price__ask": working["ask_price"],
            "bid_size": working["bid_size"],
            "ask_size": working["ask_size"],
            "ohlcv__open": working["price"],
            "ohlcv__high": working["price"],
            "ohlcv__low": working["price"],
            "ohlcv__close": working["price"],
            "volume__tick": working["size"],
            "trade_count": pd.Series(1, index=working.index, dtype="int32"),
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "vwap__in_record": working["price"],
        }
    )

    current_mid = (base["price__bid"] + base["price__ask"]) / 2
    base["prev_price__last"] = base["price__last"].shift(1)
    base["prev_price__bid"] = base["price__bid"].shift(1)
    base["prev_price__ask"] = base["price__ask"].shift(1)
    base["prev_price__mid"] = current_mid.shift(1)

    sweep_volume_threshold = float(max(base["volume__tick"].max(), 1.0))
    spread_series = (base["price__ask"] - base["price__bid"]).abs()
    sweep_spread_threshold = float(max(spread_series.median(), 1e-6))

    features_df = compute_price_ohlcv_features(
        base,
        jump_threshold=0.05,
        sweep_volume_threshold=sweep_volume_threshold,
        sweep_spread_threshold=sweep_spread_threshold,
    )

    feature_columns = {
        "price__last",
        "price__bid",
        "price__ask",
        "price__mid",
        "ohlcv__open",
        "ohlcv__high",
        "ohlcv__low",
        "ohlcv__close",
        "volume__tick",
        "trade_count",
        "volume__buy_sell_imbalance",
        "vwap__in_record",
        "spread__l1",
        "tick_return",
        "price__tick_direction",
        "return__bid_change",
        "return__ask_change",
        "return__mid_change",
        "price__micro_volatility",
        "price__jump_flag",
        "exec__slippage_estimate",
        "price__micro_fair",
        "price__near_term_return_volatility",
        "price__imbalance_adjusted_return",
        "price__tick_sweep_flag",
    }

    missing_features = feature_columns - set(features_df.columns)
    if missing_features:
        raise FeatureComputationError(f"Missing computed features: {sorted(missing_features)}")
    if len(feature_columns) != 25:
        raise FeatureComputationError("Feature set definition is not size 25")
    feature_list = list(feature_columns)
    if features_df[feature_list].isna().any().any():
        raise FeatureComputationError("NaNs detected in computed feature columns")

    print("PASS: feature computation produced all 25 features with no NaNs.")
    return features_df


def _validate_features(features_df: pd.DataFrame) -> None:
    _print_heading("Step 4 — Validate Price / OHLCV Features")
    schema_path = REPO_ROOT / "feature_engineering" / "base_features" / "price_ohlcv" / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    dtype_map = {feature["name"]: feature["dtype"] for feature in schema["features"]}

    coerced = features_df.copy()
    for col, dtype in dtype_map.items():
        if col in coerced.columns:
            coerced[col] = coerced[col].astype(dtype)

    diagnostics = validate_price_ohlcv_features(coerced)
    print("PASS: feature validation succeeded.")
    print(f"Null counts summary: {diagnostics['null_counts']}")


def _manual_assertions(features_df: pd.DataFrame) -> None:
    _print_heading("Step 5 — Manual Sanity Assertions")
    sample_indices = features_df.sample(n=min(10, len(features_df)), random_state=42).index
    mid = (features_df.loc[sample_indices, "price__bid"] + features_df.loc[sample_indices, "price__ask"]) / 2
    assert np.allclose(features_df.loc[sample_indices, "price__mid"], mid, atol=1e-9)

    spread = features_df.loc[sample_indices, "price__ask"] - features_df.loc[sample_indices, "price__bid"]
    assert np.allclose(features_df.loc[sample_indices, "spread__l1"], spread, atol=1e-9)

    first_tick = features_df.iloc[0]["tick_return"]
    assert np.isclose(first_tick, 0.0)

    assert set(features_df.loc[sample_indices, "price__tick_direction"].unique()).issubset({-1, 0, 1})

    feature_cols = [col for col in features_df.columns if col.startswith("price__") or col.startswith("return__") or col.startswith("volume__") or col.startswith("spread__") or col.startswith("vwap__") or col.startswith("tick_return") or col.startswith("exec__")]
    assert not features_df[feature_cols].isna().any().any()
    print("PASS: manual assertions succeeded on sampled rows.")


def _corruption_test(features_df: pd.DataFrame) -> None:
    _print_heading("Step 6 — Intentional Corruption Test")
    corrupted = features_df.copy()
    corrupted.loc[corrupted.index[0], "price__tick_direction"] = 2
    try:
        validate_price_ohlcv_features(corrupted)
    except FeatureValidationError:
        print("PASS: validator correctly rejected corrupted data.")
        return
    raise AssertionError("Validator did not reject corrupted data")


def main() -> None:
    success = False
    try:
        _run_orchestrator()
        trades_df, trades_path = _load_trades_frame()
        orderbook_df, ob_path = _load_orderbook_frame()

        print(f"Loaded trades from {trades_path}")
        print(f"Loaded orderbook L1 from {ob_path}")

        features_df = _prepare_feature_inputs(trades_df, orderbook_df)
        _validate_features(features_df)
        _manual_assertions(features_df)
        _corruption_test(features_df)
        success = True
    except Exception as exc:
        print(f"FAIL: {exc}")
        print("price_ohlcv base features — DO NOT FREEZE")
        raise
    finally:
        if success:
            print("\nprice_ohlcv base features — SAFE TO FREEZE")


if __name__ == "__main__":
    main()
