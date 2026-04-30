"""Linear validation checks for the data_access layer using synthetic parquet outputs.

Run this script from the project root after ensuring dependencies are installed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_access import aligner, dtype_enforcer, parquet_loader, raw_schema, validator


OUTPUT_BASE = PROJECT_ROOT / "synthetic_data_generator" / "outputs"
ORDERBOOK_ORCHESTRATOR = PROJECT_ROOT / "synthetic_data_generator" / "engine" / "orderbook" / "orderbook_orchestrator.py"


def _print_heading(title: str) -> None:
    print(f"\n=== {title} ===")


def _require_files_exist(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Expected parquet files missing: {', '.join(missing)}")


def _latest_parquet(path: Path, name_fragment: str) -> Path:
    matches = sorted(path.rglob(f"*{name_fragment}.parquet"))
    if not matches:
        raise FileNotFoundError(f"No parquet files matching '*{name_fragment}.parquet' under {path}")
    return matches[-1]


def _load_trades_frame() -> Tuple[pd.DataFrame, Path]:
    trades_dir = OUTPUT_BASE / "ticks_and_orderflow" / "trades"
    try:
        parquet_path = _latest_parquet(trades_dir, "ticks_trades")
    except FileNotFoundError:
        runner = PROJECT_ROOT / "synthetic_data_generator" / "engine" / "ticks_and_orderflow" / "run_ticks_trades.py"
        subprocess.run([sys.executable, str(runner)], check=True)
        parquet_path = _latest_parquet(trades_dir, "ticks_trades")

    raw_df = parquet_loader.load_parquet(parquet_path)
    renamed = raw_df.rename(columns={"meta__timestamp": "ts"})
    required_cols = ["ts", "symbol", "price", "size"]
    missing = [col for col in required_cols if col not in renamed.columns]
    if missing:
        raise ValueError(f"Renamed frame missing required columns: {', '.join(missing)}")
    trimmed = renamed[required_cols].copy()
    return trimmed, parquet_path


def test_0_synthetic_generation() -> None:
    _print_heading("Test 0 — Synthetic Generator Smoke Test")
    result = subprocess.run([sys.executable, str(ORDERBOOK_ORCHESTRATOR)], check=True)
    if result.returncode != 0:
        raise SystemExit("orchestrator exited with non-zero status")

    l2_file = _latest_parquet(OUTPUT_BASE / "orderbook" / "l2", "orderbook_l2")
    l3_file = _latest_parquet(OUTPUT_BASE / "orderbook" / "l3", "orderbook_l3")
    _require_files_exist([l2_file, l3_file])
    print("PASS: synthetic data generation")


def test_1_happy_path(base_df: pd.DataFrame, schema: raw_schema.FeedSchema) -> None:
    _print_heading("Test 1 — Happy Path (Valid Data)")
    coerced = dtype_enforcer.enforce_dtypes(base_df, schema)
    aligned = aligner.sort_and_validate_timestamp(
        coerced, schema, remove_exact_duplicates=True
    )
    validator.validate(aligned, schema)
    print("PASS: data_access happy path")


def test_2_missing_required_column(base_df: pd.DataFrame, schema: raw_schema.FeedSchema) -> None:
    _print_heading("Test 2 — Missing Required Column")
    broken = base_df.drop(columns=[schema.required_fields[0].name])
    try:
        validator.validate(broken, schema)
    except validator.ValidationError:
        print("PASS: missing column correctly rejected")
        return
    raise AssertionError("Missing column did not raise ValidationError")


def test_3_invalid_dtype(base_df: pd.DataFrame, schema: raw_schema.FeedSchema) -> None:
    _print_heading("Test 3 — Invalid Dtype")
    broken = base_df.copy()
    broken[schema.required_fields[2].name] = "not-a-number"
    try:
        dtype_enforcer.enforce_dtypes(broken, schema)
    except dtype_enforcer.DtypeEnforcementError:
        print("PASS: invalid dtype correctly rejected")
        return
    raise AssertionError("Invalid dtype did not raise DtypeEnforcementError")


def test_4_non_monotonic_timestamps(base_df: pd.DataFrame, schema: raw_schema.FeedSchema) -> None:
    _print_heading("Test 4 — Non-Monotonic Timestamps")
    broken = base_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    broken.loc[0, schema.timestamp_column] = broken.loc[1, schema.timestamp_column]
    try:
        aligner.sort_and_validate_timestamp(broken, schema)
    except aligner.AlignmentError:
        print("PASS: non-monotonic timestamps rejected")
        return
    raise AssertionError("Non-monotonic timestamps did not raise AlignmentError")


def test_5_duplicate_rows(base_df: pd.DataFrame, schema: raw_schema.FeedSchema) -> None:
    _print_heading("Test 5 — Duplicate Rows")
    duplicated = pd.concat([base_df, base_df.iloc[[0]]], ignore_index=True)
    try:
        validator.validate(duplicated, schema)
    except validator.ValidationError:
        print("PASS: duplicate rows rejected")
        return
    raise AssertionError("Duplicate rows did not raise ValidationError")


if __name__ == "__main__":
    test_0_synthetic_generation()
    trades_schema = raw_schema.TRADES_SCHEMA
    trades_df, parquet_path = _load_trades_frame()
    limited = trades_df.head(1000).copy()

    test_1_happy_path(limited, trades_schema)
    test_2_missing_required_column(limited, trades_schema)
    test_3_invalid_dtype(limited, trades_schema)
    test_4_non_monotonic_timestamps(limited, trades_schema)
    test_5_duplicate_rows(limited, trades_schema)

    print("\nAll tests completed. Parquet source:", parquet_path)
