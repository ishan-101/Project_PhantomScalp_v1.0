#!/usr/bin/env python3
"""
Unified validator for crossasset datasets (correlation and funding).
Follows router-style dispatch with sequential validation and independent summaries.
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

DATASETS = ["crossasset_correlation", "crossasset_funding"]

REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "crossasset_correlation": [
        "meta__timestamp",
        "meta__sequence_id",
        "date",
        "asset_x",
        "asset_y",
        "exchange",
        "returns_corr_1m",
        "returns_corr_5m",
        "returns_corr_15m",
        "returns_corr_1h",
        "returns_corr_4h",
        "returns_corr_1d",
        "rolling_covariance",
        "beta_xy",
        "volatility_ratio",
        "lead_lag_score",
        "correlation_zscore",
        "correlation_regime",
    ],
    "crossasset_funding": [
        "meta__timestamp",
        "meta__sequence_id",
        "date",
        "base_symbol",
        "quote_symbol",
        "funding_rate_base",
        "funding_rate_quote",
        "funding_diff",
        "funding_diff_zscore",
        "funding_volatility",
        "funding_regime",
    ],
}

FILE_NAMES = {
    "crossasset_correlation": "crossasset_correlation.parquet",
    "crossasset_funding": "crossasset_funding.parquet",
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


def _nulls_ok(
    df: pd.DataFrame, cols: Iterable[str], missing: Iterable[str]
) -> Tuple[bool, Dict[str, int]]:
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
        if not np.isfinite(series.to_numpy()).all():
            return False
    return True


def _business_correlation(df: pd.DataFrame) -> bool:
    asset_diff_ok = (df["asset_x"] != df["asset_y"]).all()
    corr_cols = [
        "returns_corr_1m",
        "returns_corr_5m",
        "returns_corr_15m",
        "returns_corr_1h",
        "returns_corr_4h",
        "returns_corr_1d",
    ]
    corr_range_ok = all(df[col].between(-1.0, 1.0).all() for col in corr_cols)
    numeric_cols = corr_cols + [
        "rolling_covariance",
        "beta_xy",
        "volatility_ratio",
        "lead_lag_score",
        "correlation_zscore",
    ]
    finite_ok = _finite_ok(df, numeric_cols)
    vol_ratio_ok = (df["volatility_ratio"] > 0).all()
    beta_ok = pd.to_numeric(df["beta_xy"], errors="coerce").notna().all()
    regime_ok = df["correlation_regime"].notna().all()
    return bool(asset_diff_ok and corr_range_ok and finite_ok and vol_ratio_ok and beta_ok and regime_ok)


def _business_funding(df: pd.DataFrame) -> bool:
    symbol_ok = (df["base_symbol"] != df["quote_symbol"]).all()
    numeric_cols = [
        "funding_rate_base",
        "funding_rate_quote",
        "funding_diff",
        "funding_diff_zscore",
        "funding_volatility",
    ]
    finite_ok = _finite_ok(df, numeric_cols)
    vol_ok = (df["funding_volatility"] >= 0).all()
    regime_ok = df["funding_regime"].notna().all()
    return bool(symbol_ok and finite_ok and vol_ok and regime_ok)


BUSINESS_RULES: Dict[str, Callable[[pd.DataFrame], bool]] = {
    "crossasset_correlation": _business_correlation,
    "crossasset_funding": _business_funding,
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


def _date_partition_match(df: pd.DataFrame, declared: Optional[str]) -> bool:
    if declared is None or "date" not in df.columns:
        return False
    try:
        unique_dates = pd.to_datetime(df["date"], errors="coerce").dt.date.dropna().unique()
        return len(unique_dates) == 1 and unique_dates[0].isoformat() == declared
    except Exception:
        return False


def _validate_single(dataset: str, cfg: dict) -> bool:
    base_path = Path(cfg.get("paths", {}).get("base", ""))
    rel_path = cfg.get("paths", {}).get(dataset if dataset != "crossasset_correlation" else "crossasset_corr")
    if dataset == "crossasset_correlation":
        rel_path = cfg.get("paths", {}).get("crossasset_corr")
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
    ts_ok, _ = (_timestamp_utc_ok(df["meta__timestamp"]) if "meta__timestamp" in df.columns else (False, None))
    seq_dtype_ok = _sequence_dtype_ok(df["meta__sequence_id"]) if "meta__sequence_id" in df.columns else False
    seq_mono_ok = _sequence_monotonic(df["meta__sequence_id"]) if "meta__sequence_id" in df.columns else False
    business_ok = BUSINESS_RULES[dataset](df) if required_ok else False
    rows_ok = rows == expected_rows
    declared_date = _partition_declared(target_path)
    partition_match = _date_partition_match(df, declared_date)

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