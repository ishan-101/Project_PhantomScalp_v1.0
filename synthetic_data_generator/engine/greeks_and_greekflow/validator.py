#!/usr/bin/env python3
"""
Unified validator for greeks_primary and greeks_flow datasets.
Follows router-style dispatch with shared printing and summaries.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthetic_data_generator.engine.config.loader import load_config

DATASETS = ["greeks_primary", "greeks_flow"]

REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "greeks_primary": [
        "meta__timestamp",
        "meta__sequence_id",
        "exchange",
        "symbol",
        "option_type",
        "strike",
        "expiry",
        "underlying_price",
        "delta",
        "gamma",
        "theta",
        "vega",
        "rho",
        "implied_volatility",
        "date",
    ],
    "greeks_flow": [
        "meta__timestamp",
        "meta__sequence_id",
        "exchange",
        "symbol",
        "date",
        "option_type",
        "strike",
        "expiry",
        "delta",
        "gamma",
        "theta",
        "vega",
        "rho",
        "delta_flow",
        "gamma_flow",
        "vega_flow",
        "oi_change",
        "iv_change",
        "price_change",
    ],
}

FILE_NAMES = {
    "greeks_primary": "greeks_primary.parquet",
    "greeks_flow": "greeks_flow.parquet",
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


def _nulls_ok(df: pd.DataFrame, cols: Iterable[str], missing: Iterable[str]) -> Tuple[bool, Dict[str, int]]:
    counts = {col: int(df[col].isnull().sum()) for col in cols if col in df.columns}
    if missing:
        return False, counts
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


def _sequence_dtype_ok(series: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(series)


def _sequence_monotonic(series: pd.Series) -> bool:
    return bool(series.is_monotonic_increasing)


def _partition_declared(path: Path) -> Optional[str]:
    parent = path.parent
    if parent.name.startswith("date="):
        return parent.name.split("=", 1)[1]
    return None


def _finite_ok(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().any():
            return False
        if series.isin([float("inf"), float("-inf")]).any():
            return False
    return True


def _business_primary(df: pd.DataFrame) -> bool:
    option_type_ok = df["option_type"].isin(["C", "P"]).all()
    strike_ok = pd.to_numeric(df["strike"], errors="coerce").gt(0).all()
    iv_ok = pd.to_numeric(df["implied_volatility"], errors="coerce").ge(0).all()
    greeks_finite = _finite_ok(df, ["delta", "gamma", "vega", "theta", "rho", "implied_volatility"])
    expiry_after_ts = (
        pd.to_datetime(df["expiry"], utc=True, errors="coerce")
        > pd.to_datetime(df["meta__timestamp"], utc=True, errors="coerce")
    ).all()
    return bool(option_type_ok and strike_ok and iv_ok and greeks_finite and expiry_after_ts)


def _business_flow(df: pd.DataFrame) -> bool:
    option_type_ok = df["option_type"].isin(["call", "put"]).all()
    numeric_cols = [
        "delta",
        "gamma",
        "theta",
        "vega",
        "rho",
        "delta_flow",
        "gamma_flow",
        "vega_flow",
        "oi_change",
        "iv_change",
        "price_change",
    ]
    flows_finite = _finite_ok(df, numeric_cols)
    expiry_after_ts = (
        pd.to_datetime(df["expiry"], utc=True, errors="coerce")
        > pd.to_datetime(df["meta__timestamp"], utc=True, errors="coerce")
    ).all()
    return bool(option_type_ok and flows_finite and expiry_after_ts)


BUSINESS_RULES: Dict[str, Callable[[pd.DataFrame], bool]] = {
    "greeks_primary": _business_primary,
    "greeks_flow": _business_flow,
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
    print(f"[{'OK ' if seq_dtype_ok else 'ERR'}] meta__sequence_id_dtype_int")
    print(f"[{'OK ' if seq_mono_ok else 'ERR'}] meta__sequence_id_monotonic")
    print(f"[{'OK ' if business_ok else 'ERR'}] business_rules")
    print(f"[{'OK ' if rows_ok else 'ERR'}] rows_limit")
    print(f"[{'OK ' if partition_match else 'ERR'}] partition_match")


def _print_missing(dataset: str, base_path: Path, rel_path: Optional[str], file_name: str) -> None:
    print(f"Validating {dataset} -> {base_path / (rel_path or '') / 'date=YYYY-MM-DD' / file_name}")
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
    required_ok, missing = _required_columns_check(df, REQUIRED_COLUMNS[dataset])
    nulls_ok, _ = _nulls_ok(df, REQUIRED_COLUMNS[dataset], missing)
    ts_ok, derived_date = (
        _timestamp_utc_ok(df["meta__timestamp"]) if "meta__timestamp" in df.columns else (False, None)
    )
    seq_dtype_ok = _sequence_dtype_ok(df["meta__sequence_id"]) if "meta__sequence_id" in df.columns else False
    seq_mono_ok = _sequence_monotonic(df["meta__sequence_id"]) if "meta__sequence_id" in df.columns else False
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