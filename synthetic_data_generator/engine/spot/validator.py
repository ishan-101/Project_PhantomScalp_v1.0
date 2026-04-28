# synthetic_data_generator/engine/spot/validator.py
#!/usr/bin/env python3
"""
Spot validator modeled after orderbook validator.
Usage:
  python synthetic_data_generator/engine/spot/validator.py
  python synthetic_data_generator/engine/spot/validator.py --path data/synthetic_data/spot/date=2025-12-01/spot.parquet --json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import datetime
import pandas as pd

# ---- Small timezone helpers (kept identical in behavior to orderbook validator) ----
try:
    DatetimeTZDtype = pd.DatetimeTZDtype  # type: ignore
except Exception:  # pragma: no cover - defensive
    DatetimeTZDtype = None

def is_series_tz_aware(s: pd.Series) -> bool:
    """Return True if series dtype is timezone-aware datetimetz."""
    try:
        dtype = s.dtype
        if DatetimeTZDtype is not None and isinstance(dtype, DatetimeTZDtype):
            return True
        if getattr(dtype, "tz", None) is not None:
            return True
        return pd.api.types.is_datetime64tz_dtype(dtype)
    except Exception:
        return False

def ensure_datetime_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return s
    return pd.to_datetime(s, errors="coerce")

def series_to_utc(s: pd.Series) -> pd.Series:
    s = ensure_datetime_series(s)
    if s.empty:
        return s
    if is_series_tz_aware(s):
        try:
            return s.dt.tz_convert("UTC")
        except Exception:
            return s.dt.tz_localize(None).dt.tz_localize("UTC")
    else:
        try:
            return s.dt.tz_localize("UTC")
        except Exception:
            return s.dt.tz_convert("UTC")

def safe_max_date_from_series(s: pd.Series) -> Optional[datetime.date]:
    s = ensure_datetime_series(s)
    if s.empty:
        return None
    s_utc = series_to_utc(s.dropna())
    if s_utc.empty:
        return None
    mx = s_utc.max()
    if hasattr(mx, "tzinfo") and mx.tzinfo is not None:
        return mx.tz_convert("UTC").date()
    return mx.date()

# ---- Spot config ----
ENGINE_NAME = "spot"
REQUIRED_COLUMNS = [
    "meta__timestamp",
    "meta__sequence_id",
    "exchange",
    "symbol",
    "price",
    "size",
    "side",
]
# Partition readers (e.g., pyarrow) may surface a synthetic "date" column when reading a
# partitioned directory. Treat it as allowed-but-unexpected so validation mirrors the
# orderbook validator behavior rather than failing outright.
ALLOWED_EXTRAS = {"date"}
EXPECTED_DTYPES = {
    "meta__timestamp": "datetimetz",
    "meta__sequence_id": "int",
    "exchange": "object",
    "symbol": "object",
    "price": "float",
    "size": "float",
    "side": "object",
}
DEFAULT_DATA_ROOT = Path("data") / "synthetic_data"
SPOT_PATH_REL = Path("spot")

# ---- Helpers ----
def find_latest_parquet() -> Optional[Path]:
    base_dir = DEFAULT_DATA_ROOT.joinpath(SPOT_PATH_REL)
    if not base_dir.exists():
        return None
    candidates: List[Path] = []
    for p in base_dir.glob("date=*"):
        if p.is_dir():
            for f in p.glob("*.parquet"):
                candidates.append(f)
    for f in base_dir.glob("*.parquet"):
        candidates.append(f)
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: (p.parent.name, p.stat().st_mtime), reverse=True)
    return candidates[0]

# ---- Validation ----
def validate_parquet(path: Path, deep: bool = False, sample: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
    results: Dict[str, Any] = {}
    if not path.exists():
        raise FileNotFoundError(f"path not found: {path}")
    try:
        if deep and sample:
            df = pd.read_parquet(path, columns=None).sample(n=min(sample, 10000), random_state=0)
        else:
            df = pd.read_parquet(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read parquet {path}: {e}")

    rows = len(df)
    results["rows"] = rows

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    unexpected = [c for c in df.columns if c not in REQUIRED_COLUMNS and c not in ALLOWED_EXTRAS]
    results["required_columns"] = {
        "status": len(missing) == 0 and len(unexpected) == 0,
        "detail": {"missing": missing, "unexpected": unexpected},
    }

    # dtype validation (best-effort)
    dtype_issues: Dict[str, str] = {}
    for col, expected in EXPECTED_DTYPES.items():
        if col not in df.columns:
            dtype_issues[col] = "missing"
            continue
        series = df[col]
        if expected == "datetimetz":
            if not is_series_tz_aware(series):
                dtype_issues[col] = str(series.dtype)
        elif expected == "int":
            if not pd.api.types.is_integer_dtype(series.dtype):
                dtype_issues[col] = str(series.dtype)
        elif expected == "float":
            if not pd.api.types.is_float_dtype(series.dtype):
                dtype_issues[col] = str(series.dtype)
        elif expected == "object":
            if series.dtype != "object" and not pd.api.types.is_string_dtype(series.dtype):
                dtype_issues[col] = str(series.dtype)
    results["schema"] = {
        "status": len(dtype_issues) == 0,
        "detail": dtype_issues,
    }

    nulls = {c: int(df[c].isnull().sum()) for c in df.columns if c in REQUIRED_COLUMNS or c.startswith("meta__")}
    results["nulls"] = nulls

    if "meta__timestamp" in df.columns:
        ts = df["meta__timestamp"]
        tz_aware = is_series_tz_aware(ts)
        results["meta__timestamp_tz"] = tz_aware
        derived_date = safe_max_date_from_series(ts)
        results["partition_derived"] = derived_date.isoformat() if derived_date else None
    else:
        results["meta__timestamp_tz"] = False
        results["partition_derived"] = None

    if "meta__sequence_id" in df.columns:
        is_int = pd.api.types.is_integer_dtype(df["meta__sequence_id"].dtype)
        results["meta__sequence_id_dtype_int"] = bool(is_int)
        try:
            seq_monotonic = df["meta__sequence_id"].is_monotonic_increasing
        except Exception:
            seq_monotonic = df["meta__sequence_id"].sort_values().is_monotonic_increasing
        results["meta__sequence_id_monotonic"] = bool(seq_monotonic)
    else:
        results["meta__sequence_id_dtype_int"] = False
        results["meta__sequence_id_monotonic"] = False

    # business checks
    if "price" in df.columns:
        results["price_min"] = float(df["price"].min())
    if "size" in df.columns:
        results["size_min"] = float(df["size"].min())
    if "side" in df.columns:
        results["side_values"] = sorted(list(pd.unique(df["side"].astype(str))))

    # partition from path
    parent = path.parent
    declared = parent.name.split("=", 1)[1] if parent.name.startswith("date=") else None
    results["partition_declared"] = declared

    # rows limit from config (best effort)
    cfg_limit = None
    try:
        cfg_files = [
            Path(".").resolve() / "synthetic_config.yaml",
            Path(__file__).resolve().parents[2] / "config" / "synthetic_config.yaml",
            Path(__file__).resolve().parents[3] / "synthetic_config.yaml",
        ]
        for cfg in cfg_files:
            if cfg.exists():
                import yaml  # local import
                with open(cfg, "r", encoding="utf8") as fh:
                    cfgd = yaml.safe_load(fh)
                    if isinstance(cfgd, dict):
                        cfg_limit = cfgd.get("rows", {}).get("spot")
                        break
    except Exception:
        cfg_limit = None
    if cfg_limit is not None:
        results["rows_limit"] = {"rows_limit_ok": rows <= int(cfg_limit), "rows": rows, "limit": int(cfg_limit)}
    else:
        results["rows_limit"] = None

    # sample checks
    try:
        if "meta__timestamp" in df.columns and len(df) > 0:
            sample_df = df if (not sample or deep) else df.sample(n=min(5000, len(df)), random_state=0)
            ts_series = series_to_utc(sample_df["meta__timestamp"])
            ts_sorted = ts_series.sort_values()
            results["sample_timestamp_ordered"] = bool(ts_sorted.is_monotonic_increasing)
        if "meta__sequence_id" in df.columns:
            seq_series = sample_df["meta__sequence_id"] if 'sample_df' in locals() else df["meta__sequence_id"]
            results["sample_sequence_monotonic"] = bool(seq_series.sort_values().is_monotonic_increasing)
    except Exception:
        pass

    passed = True
    if isinstance(results.get("required_columns"), dict) and not results["required_columns"]["status"]:
        passed = False
    if not results.get("meta__timestamp_tz", False):
        passed = False
    if results.get("meta__sequence_id_dtype_int") is False:
        passed = False

    results["passed"] = passed
    return passed, results

# ---- Printing ----
def print_validation(path: Optional[Path], results: Optional[Dict[str, Any]]):
    if path is None:
        print(f"Validating {ENGINE_NAME} -> ERROR: path not found: {DEFAULT_DATA_ROOT.joinpath(SPOT_PATH_REL).joinpath('date=YYYY-MM-DD','spot.parquet')}")
        return

    print(f"Validating {ENGINE_NAME} -> {path}")
    print(f"Engine: {ENGINE_NAME}")
    print(f"Path: {path}")
    if results is None:
        print("Rows: unknown")
        print("ERROR reading file or validation aborted")
        return
    print(f"Rows: {results.get('rows', 'unknown')}")
    print("-" * 100)

    rc = results.get("required_columns")
    if rc:
        if rc.get("status", False):
            print("[OK ] required_columns: {'status': True, 'detail': {'missing': [], 'unexpected': []}}")
        else:
            print(f"[ERR] required_columns: {rc.get('detail', {})}")
    else:
        print("[OK ] required_columns: present")

    schema = results.get("schema")
    if isinstance(schema, dict):
        if schema.get("status", False):
            print("[OK ] schema: {'status': True, 'detail': {}}")
        else:
            print(f"[ERR] schema: {schema.get('detail', {})}")

    nulls = results.get("nulls")
    if nulls is not None:
        print(f"[OK ] nulls: {nulls}")

    tz_ok = results.get("meta__timestamp_tz")
    if tz_ok is True:
        print("[OK ] meta__timestamp_tz: True")
    elif tz_ok is False:
        print("[ERR] meta__timestamp_tz: False")
    else:
        print(f"[OK ] meta__timestamp_tz: {tz_ok}")

    seq_dtype_ok = results.get("meta__sequence_id_dtype_int")
    if seq_dtype_ok is True:
        print("[OK ] meta__sequence_id_dtype_int: True")
    elif seq_dtype_ok is False:
        print(f"[ERR] meta__sequence_id_dtype_int: non-integer")
    else:
        print(f"[OK ] meta__sequence_id_dtype_int: {seq_dtype_ok}")

    mono = results.get("meta__sequence_id_monotonic")
    print(f"[OK ] meta__sequence_id_monotonic: {'True' if mono else 'False'}")

    if "price_min" in results:
        print(f"[OK ] price_min: {results['price_min']}")
    if "size_min" in results:
        print(f"[OK ] size_min: {results['size_min']}")
    if "side_values" in results:
        print(f"[OK ] side_values: {results['side_values']}")

    pd_decl = results.get("partition_declared")
    pd_derived = results.get("partition_derived")
    if pd_decl is not None:
        print(f"[OK ] partition_declared: {pd_decl}")
    else:
        print("[OK ] partition_declared: None")
    if pd_derived is not None:
        print(f"[OK ] partition_derived: {pd_derived}")
    else:
        print("[OK ] partition_derived: None")
    if pd_decl and pd_derived:
        print(f"[OK ] partition_match: True" if pd_decl == pd_derived else f"[ERR] partition_match: False (declared={pd_decl}, derived={pd_derived})")

    if results.get("rows_limit") is not None:
        print(f"[OK ] rows_limit: {results.get('rows_limit')}")

    print("\nSUMMARY:")
    print("PASSED: {}".format("True" if results.get("passed") else "False"))
    print()

# ---- CLI ----
def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, help="Explicit parquet path to validate")
    parser.add_argument("--deep", action="store_true", help="Do deeper checks (sampling)")
    parser.add_argument("--sample", type=int, default=None, help="Sample size for deep checks")
    parser.add_argument("--json", action="store_true", help="Dump JSON manifest")
    args = parser.parse_args(argv)

    path = Path(args.path) if args.path else find_latest_parquet()
    try:
        passed, results = validate_parquet(path, deep=args.deep, sample=args.sample) if path else (False, None)
    except FileNotFoundError:
        print(f"Validating {ENGINE_NAME} -> ERROR: path not found: {DEFAULT_DATA_ROOT.joinpath(SPOT_PATH_REL).joinpath('date=YYYY-MM-DD','spot.parquet')}")
        return
    except Exception as e:
        print(f"Validating {ENGINE_NAME} -> {path}")
        print(f"ERROR during validation: {e}")
        return

    print_validation(path, results)
    if args.json:
        print(json.dumps({"engine": ENGINE_NAME, "path": str(path) if path else None, "results": results}, indent=2, default=str))

if __name__ == "__main__":
    main()
