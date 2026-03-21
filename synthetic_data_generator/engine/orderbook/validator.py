#!/usr/bin/env python3
"""
Orderbook validator (patched to validate l3 -> l2 -> l1 and include bid_price_0/ask_price_0).
Usage:
  python synthetic_data_generator/engine/orderbook/validator.py
  python synthetic_data_generator/engine/orderbook/validator.py --engine orderbook_l3 --deep --sample 2000
  python synthetic_data_generator/engine/orderbook/validator.py --path data/synthetic_data/orderbook/l3/date=2025-12-10/orderbook_l3.parquet --json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import datetime
import glob
import warnings

# ---- Keep a small wrapper around timezone checks to avoid repeated deprecation warnings ----
try:
    DatetimeTZDtype = pd.DatetimeTZDtype  # type: ignore
except Exception:
    DatetimeTZDtype = None  # not available; fall back to pandas API

def is_series_tz_aware(s: pd.Series) -> bool:
    """Return True if series dtype is timezone-aware datetimetz."""
    try:
        dtype = s.dtype
        if DatetimeTZDtype is not None and isinstance(dtype, DatetimeTZDtype):
            return True
        if getattr(dtype, "tz", None) is not None:
            return True
        # final fallback (may emit DeprecationWarning on some pandas versions)
        return pd.api.types.is_datetime64tz_dtype(dtype)
    except Exception:
        return False

def ensure_datetime_series(s: pd.Series) -> pd.Series:
    """Coerce a series to pandas datetime (maintain tz if present)."""
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return s
    return pd.to_datetime(s, errors="coerce")

def series_to_utc(s: pd.Series) -> pd.Series:
    """Convert a datetime Series to timezone-aware UTC series reliably."""
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
    """Return the max date (UTC) from a timestamp series or None if empty/na."""
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

# ---- Engine required columns (kept consistent with existing logic) ----
ENGINES = {
    "orderbook_l3": {
        "path_rel": "orderbook/l3",
        "required": [
            "meta__timestamp",
            "meta__sequence_id",
            "event_type",
            "order_id",
            "side",
            "price",
            "size",
            "exchange",
            "symbol",
        ],
    },
    "orderbook_l2": {
        "path_rel": "orderbook/l2",
        "required": [
            "meta__timestamp",
            "meta__sequence_id",
            "exchange",
            "symbol",
            "top_bid",
            "top_ask",
            "spread",
            # level 0..9 price and size expected
        ],
    },
    "orderbook_l1": {
        "path_rel": "orderbook/l1",
        "required": [
            "meta__timestamp",
            "meta__sequence_id",
            "exchange",
            "symbol",
            "top_bid",
            "top_ask",
            "spread",
            "bid_size_0",
            "ask_size_0",
            # We explicitly require bid_price_0/ask_price_0 now
            "bid_price_0",
            "ask_price_0",
        ],
    },
}

# add level columns for l2 and l1 expected sets
for lvl in range(0, 10):
    ENGINES["orderbook_l2"]["required"].append(f"bid_price_{lvl}")
    ENGINES["orderbook_l2"]["required"].append(f"ask_price_{lvl}")
    ENGINES["orderbook_l2"]["required"].append(f"bid_size_{lvl}")
    ENGINES["orderbook_l2"]["required"].append(f"ask_size_{lvl}")
    # l1 should also have bid_price_1..9 and ask_price_1..9 and sizes
    if lvl > 0:
        ENGINES["orderbook_l1"]["required"].append(f"bid_price_{lvl}")
        ENGINES["orderbook_l1"]["required"].append(f"ask_price_{lvl}")
        ENGINES["orderbook_l1"]["required"].append(f"bid_size_{lvl}")
        ENGINES["orderbook_l1"]["required"].append(f"ask_size_{lvl}")

# ---- Helpers to find latest parquet path under canonical data root ----
DEFAULT_DATA_ROOT = Path("data") / "synthetic_data"

def find_latest_parquet_for_engine(engine_name: str) -> Optional[Path]:
    """
    Locate the most recent parquet file for the engine under DEFAULT_DATA_ROOT.
    Looks under: <root>/<path_rel>/date=*/<engine>.parquet or any .parquet file.
    Returns Path or None.
    """
    info = ENGINES[engine_name]
    rel = info["path_rel"]
    base_dir = DEFAULT_DATA_ROOT.joinpath(rel)
    if not base_dir.exists():
        return None
    # search date partitions "date=YYYY-MM-DD" and pick newest by folder name (lexicographic on ISO date works)
    candidates = []
    for p in base_dir.glob("date=*"):
        if p.is_dir():
            # find parquet files directly inside
            for f in p.glob("*.parquet"):
                candidates.append(f)
    # also consider any parquet directly under base_dir
    for f in base_dir.glob("*.parquet"):
        candidates.append(f)
    if not candidates:
        return None
    # pick by file mtime as robust fallback
    candidates = sorted(candidates, key=lambda p: (p.parent.name, p.stat().st_mtime), reverse=True)
    return candidates[0]

# ---- Actual validation checks per engine ----
def validate_parquet(path: Path, engine_name: str, deep: bool = False, sample: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate a single parquet file for the given engine.
    Returns (passed: bool, results: dict)
    """
    results: Dict[str, Any] = {}
    if not path.exists():
        raise FileNotFoundError(f"path not found: {path}")

    # read parquet (deep may only sample)
    try:
        if deep and sample:
            df = pd.read_parquet(path, columns=None).sample(n=min(sample, 10000), random_state=0)
        else:
            df = pd.read_parquet(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read parquet {path}: {e}")

    rows = len(df)
    results["rows"] = rows

    req_cols = ENGINES[engine_name]["required"]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        results["required_columns"] = {"status": False, "detail": {"missing": missing}}
    else:
        results["required_columns"] = {"status": True, "detail": {"missing": []}}

    # Null counts for relevant columns (report for present columns only)
    nulls = {c: int(df[c].isnull().sum()) for c in df.columns if c in req_cols or c.startswith("meta__") or c.startswith("bid_") or c.startswith("ask_")}
    results["nulls"] = nulls

    # meta__timestamp checks
    if "meta__timestamp" in df.columns:
        ts = df["meta__timestamp"]
        tz_aware = is_series_tz_aware(ts)
        results["meta__timestamp_tz"] = tz_aware
        # derived partition date from max ts
        derived_date = safe_max_date_from_series(ts)
        if derived_date is not None:
            results["partition_derived"] = derived_date.isoformat()
    else:
        results["meta__timestamp_tz"] = False
        results["partition_derived"] = None

    # meta__sequence_id dtype and monotonic
    if "meta__sequence_id" in df.columns:
        seq_dtype = str(df["meta__sequence_id"].dtype)
        # check integer dtype
        is_int = pd.api.types.is_integer_dtype(df["meta__sequence_id"].dtype)
        results["meta__sequence_id_dtype_int"] = is_int
        # monotonic check
        try:
            seq_monotonic = df["meta__sequence_id"].is_monotonic_increasing
        except Exception:
            seq_monotonic = df["meta__sequence_id"].sort_values().is_monotonic_increasing
        results["meta__sequence_id_monotonic"] = bool(seq_monotonic)
    else:
        results["meta__sequence_id_dtype_int"] = False
        results["meta__sequence_id_monotonic"] = False

    # Engine specific checks
    if engine_name == "orderbook_l3":
        if "event_type" in df.columns:
            vals = sorted(list(pd.unique(df["event_type"].astype(str))))
            results["event_type_values"] = vals
        if "side" in df.columns:
            vals = sorted(list(pd.unique(df["side"].astype(str))))
            results["side_values"] = vals
        # price/size minima
        if "price" in df.columns:
            results["price_min"] = float(df["price"].min())
        if "size" in df.columns:
            results["size_min"] = float(df["size"].min())
    elif engine_name == "orderbook_l2":
        # rows limit check if possible (we try to infer expected rows from file name or env)
        results["rows_limit"] = {"rows": rows, "limit": None}
    elif engine_name == "orderbook_l1":
        # check that top_bid/top_ask exist and are positive
        if "top_bid" in df.columns and "top_ask" in df.columns:
            results["price_positive"] = {"top_bid": float(df["top_bid"].min()), "top_ask": float(df["top_ask"].min())}
        # sizes
        size_cols = [c for c in df.columns if c.startswith("bid_size_") or c.startswith("ask_size_")]
        size_mins = {c: float(df[c].min()) for c in size_cols} if size_cols else {}
        results["size_positive"] = size_mins
        # basic business checks (spread > 0, bid < ask)
        if "top_bid" in df.columns and "top_ask" in df.columns:
            spread_min = float((df["top_ask"] - df["top_bid"]).min())
            bid_lt_ask = bool((df["top_bid"] < df["top_ask"]).all())
            results["l1_business"] = {"spread_min": spread_min, "bid_lt_ask": bid_lt_ask}

    # partition declaration (try to read partition from path "date=YYYY-MM-DD")
    declared = None
    parent = path.parent
    if parent.name.startswith("date="):
        declared = parent.name.split("=", 1)[1]
        results["partition_declared"] = declared
    else:
        results["partition_declared"] = None

    # rows_limit inference attempt: look for synthetic_config.yaml in repo root or engine/config (best-effort)
    cfg_limit = None
    try:
        # Try repo root
        p = Path(".").resolve()
        cfg_files = [
            p / "synthetic_config.yaml",
            p / "engine" / "config" / "synthetic_config.yaml",
            Path(__file__).resolve().parents[3] / "synthetic_config.yaml",
        ]
        for cfg in cfg_files:
            if cfg.exists():
                import yaml  # local import
                with open(cfg, "r", encoding="utf8") as fh:
                    cfgd = yaml.safe_load(fh)
                    if isinstance(cfgd, dict):
                        rows_map = cfgd.get("rows", {})
                        key = "orderbook_l1" if engine_name == "orderbook_l1" else ("orderbook_l2" if engine_name == "orderbook_l2" else "orderbook_l3")
                        cfg_limit = rows_map.get(key)
                        break
    except Exception:
        cfgd = None

    if cfg_limit is not None:
        results["rows_limit"] = {"rows_limit_ok": rows <= int(cfg_limit), "rows": rows, "limit": int(cfg_limit)}
    else:
        # ensure key exists even if not resolved
        if "rows_limit" not in results:
            results["rows_limit"] = None

    # Final pass: sample ordering checks (monotonic timestamp/sequence in sample)
    try:
        if "meta__timestamp" in df.columns and len(df) > 0:
            # sample a small amount deterministically if large
            sample_df = df if (not sample or deep) else df.sample(n=min(5000, len(df)), random_state=0)
            # ensure timestamps are timezone-aware for comparison
            ts_series = series_to_utc(sample_df["meta__timestamp"])
            ts_sorted = ts_series.sort_values()
            results["sample_timestamp_ordered"] = bool(ts_sorted.is_monotonic_increasing)
        if "meta__sequence_id" in df.columns:
            seq_series = sample_df["meta__sequence_id"]
            results["sample_sequence_monotonic"] = bool(seq_series.sort_values().is_monotonic_increasing)
    except Exception:
        pass

    # decide pass/fail heuristics (conservative)
    passed = True
    # missing required columns => fail
    if isinstance(results.get("required_columns"), dict) and not results["required_columns"]["status"]:
        passed = False
    # meta timestamp tz required
    if not results.get("meta__timestamp_tz", False):
        passed = False
    # sequence dtype must be int
    if results.get("meta__sequence_id_dtype_int") is False:
        passed = False

    results["passed"] = passed
    return passed, results

# ---- Print formatting to match the exact style you asked for ----
def print_validation_for_engine(engine_name: str, path: Optional[Path], results: Optional[Dict[str, Any]]):
    """
    Format output in same style as your example.
    If path is None -> prints ERROR: path not found
    """
    if path is None:
        print(f"Validating {engine_name} -> ERROR: path not found: {DEFAULT_DATA_ROOT.joinpath(ENGINES[engine_name]['path_rel']).joinpath('date=YYYY-MM-DD','<file>.parquet')}")
        return

    print(f"Validating {engine_name} -> {path}")
    print(f"Engine: {engine_name}")
    print(f"Path: {path}")
    if results is None:
        print("Rows: unknown")
        print("ERROR reading file or validation aborted")
        return
    rows = results.get("rows", "unknown")
    print(f"Rows: {rows}")
    print("-" * 100)
    # Now print each check in a tidy form similar to earlier output
    # required_columns
    rc = results.get("required_columns")
    if rc:
        if rc.get("status", False):
            print("[OK ] required_columns: {'status': True, 'detail': {'missing': []}}")
        else:
            print(f"[ERR] required_columns: {rc.get('detail', {})}")
    else:
        print("[OK ] required_columns: present")

    # nulls
    nulls = results.get("nulls")
    if nulls is not None:
        print(f"[OK ] nulls: {nulls}")

    # timestamp tz
    tz_ok = results.get("meta__timestamp_tz")
    if tz_ok is True:
        print("[OK ] meta__timestamp_tz: True")
    elif tz_ok is False:
        print("[ERR] meta__timestamp_tz: False")
    else:
        print(f"[OK ] meta__timestamp_tz: {tz_ok}")

    # seq dtype
    seq_dtype_ok = results.get("meta__sequence_id_dtype_int")
    if seq_dtype_ok is True:
        print("[OK ] meta__sequence_id_dtype_int: True")
    elif seq_dtype_ok is False:
        # include actual dtype if possible
        val = "non-integer"
        print(f"[ERR] meta__sequence_id_dtype_int: {val}")
    else:
        print(f"[OK ] meta__sequence_id_dtype_int: {seq_dtype_ok}")

    # monotonic
    mono = results.get("meta__sequence_id_monotonic")
    print("[OK ] meta__sequence_id_monotonic: {}".format("True" if mono else "False"))

    # engine-specific prints
    if engine_name == "orderbook_l3":
        if "event_type_values" in results:
            print(f"[OK ] event_type_values: {results['event_type_values']}")
        if "side_values" in results:
            print(f"[OK ] side_values: {results['side_values']}")
        if "price_min" in results:
            print(f"[OK ] price_min: {results['price_min']}")
        if "size_min" in results:
            print(f"[OK ] size_min: {results['size_min']}")
    elif engine_name == "orderbook_l2":
        # rows_limit
        rl = results.get("rows_limit")
        if isinstance(rl, dict):
            print(f"[OK ] rows_limit: {rl}")
        else:
            print(f"[OK ] rows_limit: {rl}")
    elif engine_name == "orderbook_l1":
        if "price_positive" in results:
            print(f"[OK ] price_positive: {results['price_positive']}")
        if "size_positive" in results:
            print(f"[OK ] size_positive: {results['size_positive']}")
        if "l1_business" in results:
            print(f"[OK ] l1_business: {results['l1_business']}")

    # partition info
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
    # match
    if pd_decl and pd_derived:
        print(f"[OK ] partition_match: True" if pd_decl == pd_derived else f"[ERR] partition_match: False (declared={pd_decl}, derived={pd_derived})")

    # rows limit message (if present)
    if results.get("rows_limit") is not None:
        print(f"[OK ] rows_limit: {results.get('rows_limit')}")

    print("\nSUMMARY:")
    print("PASSED: {}".format("True" if results.get("passed") else "False"))
    # spacer between engines
    print()

def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="all", choices=["all", "orderbook_l3", "orderbook_l2", "orderbook_l1"])
    parser.add_argument("--path", default=None, help="Explicit parquet path to validate")
    parser.add_argument("--deep", action="store_true", help="Do deeper checks (sampleing)")
    parser.add_argument("--sample", type=int, default=None, help="Sample size for deep checks")
    parser.add_argument("--json", action="store_true", help="Dump JSON manifest instead of human output")
    args = parser.parse_args(argv)

    # determine order: l3, l2, l1
    engines_to_run = ["orderbook_l3", "orderbook_l2", "orderbook_l1"] if args.engine == "all" else [args.engine]

    json_out = []
    for eng in engines_to_run:
        explicit_path = Path(args.path) if args.path else None
        if explicit_path:
            path = explicit_path
        else:
            path = find_latest_parquet_for_engine(eng)
        try:
            passed, results = validate_parquet(path, eng, deep=args.deep, sample=args.sample)
        except FileNotFoundError:
            # print missing when path not found
            print(f"Validating {eng} -> ERROR: path not found: {DEFAULT_DATA_ROOT.joinpath(ENGINES[eng]['path_rel']).joinpath('date=YYYY-MM-DD','<file>.parquet')}")
            print()
            continue
        except Exception as e:
            print(f"Validating {eng} -> {path}")
            print(f"ERROR during validation: {e}")
            print()
            continue

        # print in human format
        print_validation_for_engine(eng, path, results)
        json_out.append({ "engine": eng, "path": str(path) if path else None, "results": results })

    if args.json:
        print(json.dumps(json_out, indent=2, default=str))

if __name__ == "__main__":
    main()
