#!/usr/bin/env python3
"""
Unified validator for options datasets (options_chain, options_iv_surface, options_oi).
Follows router pattern with one validator per dataset and shared output/summary.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthetic_data_generator.engine.config.loader import load_config

DATASETS = [
    "options_chain",
    "options_iv_surface",
    "options_oi",
]

REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "options_chain": [
        "meta__timestamp",
        "meta__sequence_id",
        "exchange",
        "symbol",
        "expiry_ts",
        "expiry_days",
        "strike",
        "option_type",
        "bid",
        "ask",
        "mid",
        "last",
        "volume",
        "open_interest",
        "date",
    ],
    "options_iv_surface": [
        "meta__timestamp",
        "meta__sequence_id",
        "exchange",
        "symbol",
        "expiry_ts",
        "expiry_days",
        "strike",
        "option_type",
        "implied_vol",
        "moneyness",
        "skew",
        "term_structure",
        "date",
    ],
    "options_oi": [
        "meta__timestamp",
        "meta__sequence_id",
        "exchange",
        "symbol",
        "expiry_ts",
        "expiry_days",
        "strike",
        "option_type",
        "open_interest",
        "oi_change",
        "volume",
        "date",
    ],
}

FILE_NAMES = {
    "options_chain": "options_chain.parquet",
    "options_iv_surface": "options_iv_surface.parquet",
    "options_oi": "options_oi.parquet",
}

PRINT_DIVIDER = "-" * 60


def _latest_partition_path(base: Path, relative: str, file_name: str) -> Optional[Path]:
    base_dir = base / relative
    partitions = sorted((p for p in base_dir.glob("date=*") if p.is_dir()), reverse=True)
    for part in partitions:
        candidate = part / file_name
        if candidate.exists():
            return candidate
    return None


def _required_columns_check(df: pd.DataFrame, required: Iterable[str]) -> Tuple[bool, List[str]]:
    missing = [col for col in required if col not in df.columns]
    return len(missing) == 0, missing


def _nulls_ok(df: pd.DataFrame, cols: Iterable[str]) -> Tuple[bool, Dict[str, int]]:
    counts = {col: int(df[col].isnull().sum()) for col in cols if col in df.columns}
    return all(count == 0 for count in counts.values()), counts


def _timestamp_utc_ok(series: pd.Series) -> Tuple[bool, Optional[str]]:
    if series.empty:
        return False, None
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = pd.to_datetime(series, errors="coerce")
    tz_ok = pd.api.types.is_datetime64tz_dtype(series) and str(series.dt.tz) == "UTC"
    if not tz_ok:
        return False, None
    try:
        max_date = series.dt.tz_convert("UTC").max().date()
        return True, max_date.isoformat()
    except Exception:
        return False, None


def _sequence_checks(series: pd.Series) -> Tuple[bool, bool]:
    dtype_ok = pd.api.types.is_integer_dtype(series)
    monotonic_ok = bool(series.is_monotonic_increasing)
    return dtype_ok, monotonic_ok


def _partition_declared(path: Path) -> Optional[str]:
    parent = path.parent
    if parent.name.startswith("date="):
        return parent.name.split("=", 1)[1]
    return None


def _business_chain(df: pd.DataFrame) -> bool:
    positive_strike = (df["strike"] > 0).all()
    bid_ask_order = (df["bid"] <= df["ask"]).all()
    option_type_ok = df["option_type"].isin(["C", "P"]).all()
    expiry_after_ts = (df["expiry_ts"] > df["meta__timestamp"]).all()
    non_negative = (
        (df["bid"] >= 0).all()
        and (df["ask"] >= 0).all()
        and (df["mid"] >= 0).all()
        and (df["last"] >= 0).all()
        and (df["volume"] >= 0).all()
        and (df["open_interest"] >= 0).all()
    )
    return bool(positive_strike and bid_ask_order and option_type_ok and expiry_after_ts and non_negative)


def _business_iv_surface(df: pd.DataFrame) -> bool:
    vol_ok = (df["implied_vol"] > 0).all()
    moneyness_ok = (df["moneyness"] > 0).all()
    term_structure_ok = (df["term_structure"] > 0).all()
    option_type_ok = df["option_type"].isin(["C", "P"]).all()
    expiry_after_ts = (df["expiry_ts"] > df["meta__timestamp"]).all()
    all_finite = np.isfinite(df[["implied_vol", "moneyness", "skew", "term_structure"]].to_numpy()).all()
    return bool(vol_ok and moneyness_ok and term_structure_ok and option_type_ok and expiry_after_ts and all_finite)


def _business_oi(df: pd.DataFrame) -> bool:
    oi_ok = (df["open_interest"] >= 0).all()
    volume_ok = (df["volume"] >= 0).all()
    volume_covers_change = (df["volume"] >= df["oi_change"].abs()).all()
    option_type_ok = df["option_type"].isin(["C", "P"]).all()
    expiry_after_ts = (df["expiry_ts"] > df["meta__timestamp"]).all()
    return bool(oi_ok and volume_ok and volume_covers_change and option_type_ok and expiry_after_ts)


BUSINESS_RULES: Dict[str, Callable[[pd.DataFrame], bool]] = {
    "options_chain": _business_chain,
    "options_iv_surface": _business_iv_surface,
    "options_oi": _business_oi,
}


def _print_checks(
    dataset: str,
    path: Path,
    rows: int,
    required_ok: bool,
    nulls_ok: bool,
    ts_ok: bool,
    seq_dtype_ok: bool,
    seq_mono_ok: bool,
    business_ok: bool,
    rows_ok: bool,
    partition_match: bool,
) -> None:
    print(f"Engine: {dataset}")
    print(f"Path: {path}")
    print(f"Rows: {rows}")
    print(PRINT_DIVIDER)
    print(f"[{'OK ' if required_ok else 'ERR'}] required_columns")
    print(f"[{'OK ' if nulls_ok else 'ERR'}] nulls")
    print(f"[{'OK ' if ts_ok else 'ERR'}] meta__timestamp_tz")
    print(f"[{'OK ' if seq_dtype_ok else 'ERR'}] meta__sequence_id_dtype")
    print(f"[{'OK ' if seq_mono_ok else 'ERR'}] meta__sequence_id_monotonic")
    print(f"[{'OK ' if business_ok else 'ERR'}] business_rules")
    print(f"[{'OK ' if rows_ok else 'ERR'}] rows_limit")
    print(f"[{'OK ' if partition_match else 'ERR'}] partition_match")


def _print_missing(dataset: str, base_path: Path, rel_path: Optional[str], file_name: str) -> None:
    print(
        f"Validating {dataset} -> {base_path / (rel_path or '') / 'date=YYYY-MM-DD' / file_name}"
    )
    print(f"Engine: {dataset}")
    print(f"Path: {base_path / (rel_path or '')}")
    print("Rows: 0")
    print(PRINT_DIVIDER)
    print("[ERR] path not found")
    print("\nSUMMARY:")
    print("PASSED: False\n")


def _validate_single(dataset: str, cfg: dict) -> bool:
    base_path = Path(cfg.get("paths", {}).get("base", ""))
    rel_path = cfg.get("paths", {}).get(dataset)
    expected_rows = int(cfg.get("rows", {}).get(dataset, 0))

    target_path = _latest_partition_path(base_path, rel_path, FILE_NAMES[dataset]) if rel_path else None
    if target_path is None:
        _print_missing(dataset, base_path, rel_path, FILE_NAMES[dataset])
        return False

    print(f"Validating {dataset} -> {target_path}")
    try:
        df = pd.read_parquet(target_path)
    except Exception as exc:
        print(f"Engine: {dataset}")
        print(f"Path: {target_path}")
        print("Rows: 0")
        print(PRINT_DIVIDER)
        print(f"[ERR] failed_to_read: {exc}")
        print("\nSUMMARY:")
        print("PASSED: False\n")
        return False

    rows = len(df)
    required_ok, _ = _required_columns_check(df, REQUIRED_COLUMNS[dataset])
    nulls_ok, _ = _nulls_ok(df, REQUIRED_COLUMNS[dataset])
    ts_ok, derived_date = _timestamp_utc_ok(df["meta__timestamp"]) if "meta__timestamp" in df.columns else (False, None)
    seq_dtype_ok, seq_mono_ok = _sequence_checks(df["meta__sequence_id"]) if "meta__sequence_id" in df.columns else (False, False)
    business_ok = BUSINESS_RULES[dataset](df) if required_ok else False
    rows_ok = rows == expected_rows
    declared_date = _partition_declared(target_path)
    partition_match = declared_date == derived_date if declared_date and derived_date else False

    _print_checks(
        dataset,
        target_path,
        rows,
        required_ok,
        nulls_ok,
        ts_ok,
        seq_dtype_ok,
        seq_mono_ok,
        business_ok,
        rows_ok,
        partition_match,
    )

    passed = all(
        [
            required_ok,
            nulls_ok,
            ts_ok,
            seq_dtype_ok,
            seq_mono_ok,
            business_ok,
            rows_ok,
            partition_match,
        ]
    )

    print("\nSUMMARY:")
    print(f"PASSED: {str(passed)}\n")
    return passed


def main() -> None:
    cfg = load_config()
    all_passed = True
    for dataset in DATASETS:
        result = _validate_single(dataset, cfg)
        all_passed = all_passed and result

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
