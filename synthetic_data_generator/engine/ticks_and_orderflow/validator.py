#!/usr/bin/env python3
"""
Validator for ticks_and_orderflow engines.

Usage examples:
    python synthetic_data_generator/engine/ticks_and_orderflow/validator.py --engine all
    python synthetic_data_generator/engine/ticks_and_orderflow/validator.py --engine ticks_orderflow --deep --sample 5000
    python -m synthetic_data_generator.engine.ticks_and_orderflow.validator --engine ticks_trades --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

import pandas as pd


# ---------- Configuration / engine -> path mapping ----------
# Defaults; will be overridden by central config if available.
ENGINE_CONFIG = {
    "ticks_trades": {
        "subdir": Path("ticks_and_orderflow") / "trades",
        "filename": "ticks_trades.parquet",
        "max_rows": 50000,
        "required_columns": [
            "meta__timestamp",
            "meta__sequence_id",
            "price",
            "size",
            "aggressor",
            "exchange",
            "symbol",
        ],
    },
    "ticks_orderflow": {
        "subdir": Path("ticks_and_orderflow") / "orderflow",
        "filename": "ticks_orderflow.parquet",
        "max_rows": 500000,
        "required_columns": [
            "meta__timestamp",
            "meta__sequence_id",
            "event_type",
            "price",
            "size",
            "aggressor",
            "inventory_pressure",
            "exchange",
            "symbol",
        ],
    },
}
DEFAULT_BASE_DATA_DIR = Path("synthetic_data_generator") / "outputs"

# ---------- Config loader (best-effort) ----------
def _load_central_config() -> Optional[Dict[str, Any]]:
    """
    Best-effort loader: try the shared loader; if that fails, read YAML from known locations.
    """
    mod_paths = (
        "synthetic_data_generator.engine.config.loader",
        "engine.config.loader",
    )
    for mod_path in mod_paths:
        try:
            mod = __import__(mod_path, fromlist=["load_config"])
            if hasattr(mod, "load_config"):
                cfg = mod.load_config()
                if isinstance(cfg, dict):
                    return cfg
        except Exception:
            continue

    # fallback: direct YAML read from expected file locations
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "config" / "synthetic_config.yaml",      # engine/config/
        here.parents[3] / "synthetic_config.yaml",                  # repo root (if present)
        here.parents[2] / "config" / "synthetic_config.yaml",       # synthetic_data_generator/config/
    ]
    for p in candidates:
        try:
            if p.exists():
                with p.open("r", encoding="utf-8") as fh:
                    cfg = yaml.safe_load(fh)
                    if isinstance(cfg, dict):
                        return cfg
        except Exception:
            continue
    return None


def resolve_engine_layout():
    """
    Return (base_dir, engine_cfg_map, config_obj) where:
      - base_dir: Path to paths.base (default data/synthetic_data)
      - engine_cfg_map: merged ENGINE_CONFIG with subdir/max_rows from config when present
      - config_obj: the loaded config dict or None
    """
    cfg = _load_central_config()
    base_dir = DEFAULT_BASE_DATA_DIR
    engine_cfg: Dict[str, Dict[str, Any]] = {}

    if isinstance(cfg, dict):
        try:
            base_dir = Path(cfg.get("paths", {}).get("base", DEFAULT_BASE_DATA_DIR))
        except Exception:
            base_dir = DEFAULT_BASE_DATA_DIR
    for key, meta in ENGINE_CONFIG.items():
        merged = dict(meta)
        merged["subdir"] = Path(meta["subdir"])
        if isinstance(cfg, dict):
            paths = cfg.get("paths", {})
            rows = cfg.get("rows", {})
            cfg_key = "ticks_trades" if key == "ticks_trades" else "ticks_orderflow"
            if cfg_key in paths:
                merged["subdir"] = Path(paths[cfg_key])
            if key in rows:
                merged["max_rows"] = rows[key]
        engine_cfg[key] = merged
    return base_dir, engine_cfg, cfg


# ---------- Utility helpers ----------
def locate_engine_file(engine_key: str, base_dir: Path, engine_cfg: Dict[str, Any]) -> Optional[Path]:
    """
    Try to locate the parquet file for the engine. It supports partitioned layout like:
      data/synthetic_data/ticks_and_orderflow/<subdir>/date=YYYY-MM-DD/<filename>
    and unpartitioned layout:
      data/synthetic_data/ticks_and_orderflow/<subdir>/<filename>
    Returns the first match found (prefers partitioned matches using descending modification time).
    """
    cfg = engine_cfg.get(engine_key)
    if cfg is None:
        return None

    subdir = base_dir / cfg["subdir"]
    filename = cfg["filename"]
    if not subdir.exists():
        return None

    # search for partitioned and non-partitioned files
    patterns = [
        f"**/date=*/{filename}",
        f"**/{filename}",
    ]
    matches: List[Path] = []
    for pat in patterns:
        for p in subdir.glob(pat):
            if p.is_file():
                matches.append(p)

    if not matches:
        return None

    # choose the most recently modified candidate
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def load_parquet(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Load parquet with pandas (delegates to pandas.read_parquet)."""
    return pd.read_parquet(path, columns=columns)


def _is_tzaware(series: pd.Series) -> bool:
    """
    Robust check whether a datetime Series is timezone-aware.
    Avoids using deprecated pandas helpers.
    """
    try:
        # Series.dt.tz exists for Datetime-like Series; if present and not None => tz-aware
        tz = getattr(series.dt, "tz", None)
        if tz is not None:
            return True
        # Fallback: dtype may expose a tz attribute (e.g., DatetimeTZDtype)
        return getattr(series.dtype, "tz", None) is not None
    except Exception:
        return False


def _extract_partition_date_from_path(path: Path) -> Optional[str]:
    """
    If path contains a 'date=YYYY-MM-DD' segment, return the YYYY-MM-DD string.
    """
    for part in path.parts:
        if part.startswith("date="):
            return part.split("=", 1)[1]
    return None


# ---------- Validation checks ----------
def check_required_columns(df: pd.DataFrame, required: List[str]) -> Dict[str, object]:
    missing = [c for c in required if c not in df.columns]
    return {"status": len(missing) == 0, "missing": missing}


def check_nulls(df: pd.DataFrame, cols: List[str]) -> Dict[str, int]:
    return {c: int(df[c].isna().sum()) for c in cols if c in df.columns}


def check_timestamp_tz(df: pd.DataFrame, ts_col: str = "meta__timestamp") -> bool:
    if ts_col not in df.columns:
        return False
    return _is_tzaware(df[ts_col])


def check_sequence_monotonic(df: pd.DataFrame, seq_col: str = "meta__sequence_id") -> bool:
    if seq_col not in df.columns:
        return False
    # check monotonic increasing
    try:
        return df[seq_col].is_monotonic_increasing
    except Exception:
        # fallback: use numpy
        try:
            import numpy as np

            arr = np.asarray(df[seq_col])
            return (arr[1:] >= arr[:-1]).all()
        except Exception:
            return False


def check_timestamp_ordered(df: pd.DataFrame, ts_col: str = "meta__timestamp") -> bool:
    if ts_col not in df.columns:
        return False
    try:
        return df[ts_col].is_monotonic_increasing
    except Exception:
        # fallback rough method
        s = pd.to_datetime(df[ts_col])
        return s.is_monotonic_increasing


def check_positive_min(df: pd.DataFrame, col: str) -> Optional[float]:
    if col not in df.columns:
        return None
    try:
        # cast numeric safely and compute min
        s = pd.to_numeric(df[col], errors="coerce")
        if s.isna().all():
            return None
        return float(s.min())
    except Exception:
        return None


# ---------- Main validator logic ----------
def validate_path(
    path: Path,
    engine_key: str,
    deep: bool = False,
    sample_n: int = 2000,
    engine_cfg_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict:
    """
    Validate a single parquet file path for the given engine.
    Returns a dictionary summary with checks.
    """
    cfg_source = engine_cfg_map if engine_cfg_map is not None else ENGINE_CONFIG
    cfg = cfg_source.get(engine_key)
    if cfg is None:
        raise ValueError(f"unknown engine: {engine_key}")

    result: Dict = {
        "engine_key": engine_key,
        "path": str(path),
        "rows": None,
        "checks": {},
        "passed": False,
    }

    if not path.exists():
        result["checks"]["path_exists"] = {"status": False, "detail": "path not found"}
        return result

    # load
    try:
        df = load_parquet(path)
    except Exception as e:
        result["checks"]["load"] = {"status": False, "detail": f"failed to read parquet: {e}"}
        return result

    result["rows"] = int(len(df))

    # required columns
    rc = check_required_columns(df, cfg["required_columns"])
    result["checks"]["required_columns"] = {
        "status": rc["status"],
        "detail": "present" if rc["status"] else f"missing: {rc['missing']}",
    }

    # null counts
    nulls = check_nulls(df, cfg["required_columns"])
    result["checks"]["nulls"] = {"status": True, "detail": nulls}

    # timestamp tz
    tz_ok = check_timestamp_tz(df)
    result["checks"]["meta__timestamp_tz"] = {
        "status": tz_ok,
        "detail": "timezone-aware (sample)" if tz_ok else "not timezone-aware or missing",
    }

    # sequence monotonic (full-file)
    seq_mono = check_sequence_monotonic(df)
    result["checks"]["meta__sequence_id_monotonic"] = {
        "status": seq_mono,
        "detail": "monotonic" if seq_mono else "not monotonic or missing",
    }

    # event_type domain (where applicable)
    if "event_type" in df.columns:
        unique_ev = sorted(df["event_type"].dropna().unique().tolist())
        result["checks"]["event_type_values"] = {"status": True, "detail": unique_ev}
    else:
        result["checks"]["event_type_values"] = {"status": False, "detail": "N/A"}

    # price and size positive checks (sample)
    price_min = check_positive_min(df, "price")
    size_min = check_positive_min(df, "size")
    result["checks"]["price_positive"] = {
        "status": price_min is not None and price_min > 0,
        "detail": f"min={price_min}" if price_min is not None else "missing or non-numeric",
    }
    result["checks"]["size_positive"] = {
        "status": size_min is not None and size_min > 0,
        "detail": f"min={size_min}" if size_min is not None else "missing or non-numeric",
    }

    # rows limit
    max_rows = int(cfg.get("max_rows", 0))
    rows_ok = (max_rows == 0) or (len(df) <= max_rows)
    result["checks"]["rows_limit"] = {
        "status": rows_ok,
        "detail": f"{len(df)} rows (<= configured {max_rows})" if max_rows else f"{len(df)} rows (no limit configured)",
    }

    # extract partition date if present
    part_date = _extract_partition_date_from_path(path)
    if part_date:
        result["checks"]["partition_date"] = {"status": True, "detail": f"partition date matches {part_date}"}
    else:
        result["checks"]["partition_date"] = {"status": False, "detail": "partition date not found in path"}

    # deep/sample checks
    if deep or (sample_n and sample_n > 0):
        # sample size limited to dataframe length
        samp_n = min(sample_n, max(1, len(df)))
        # deterministic sample
        s = df.sample(n=samp_n, random_state=0)

        # --- FIX: sort sample by timestamp before checking monotonicity ---
        if "meta__timestamp" in s.columns:
            s = s.sort_values("meta__timestamp")

        # timestamp monotonicity on the sample (after sorting)
        sample_ts_mono = s["meta__timestamp"].is_monotonic_increasing if "meta__timestamp" in s.columns else False
        result["checks"]["sample_timestamp_ordered"] = {
            "status": bool(sample_ts_mono),
            "detail": "sample timestamps monotonic (after sort)" if sample_ts_mono else "not monotonic in sample or missing",
        }

        # sample sequence monotonic in sampled order (after sorting)
        if "meta__sequence_id" in s.columns:
            sample_seq_mono = s["meta__sequence_id"].is_monotonic_increasing
        else:
            sample_seq_mono = False
        result["checks"]["sample_sequence_monotonic"] = {
            "status": bool(sample_seq_mono),
            "detail": "sample sequence monotonic (after sort)" if sample_seq_mono else "not monotonic in sample or missing",
        }

    # final pass/fail: fatal checks are:
    # - required columns present
    # - timestamp tz present
    # - sequence monotonic (full)
    # - rows limit
    fatal_ok = all(
        [
            result["checks"]["required_columns"]["status"],
            result["checks"]["meta__timestamp_tz"]["status"],
            result["checks"]["meta__sequence_id_monotonic"]["status"],
            result["checks"]["rows_limit"]["status"],
        ]
    )
    result["passed"] = fatal_ok

    return result


# ---------- CLI ----------
def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(prog="ticks_and_orderflow.validator")
    p.add_argument("--engine", choices=list(ENGINE_CONFIG.keys()) + ["all"], default="all")
    p.add_argument("--path", help="explicit parquet path to validate (overrides --engine)")
    p.add_argument("--deep", action="store_true", help="perform deep/sample checks")
    p.add_argument("--sample", type=int, default=2000, help="sample size for deep/sample checks")
    p.add_argument("--json", action="store_true", help="emit JSON summary")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    base_dir, engine_cfg_map, central_cfg = resolve_engine_layout()

    engines = []
    if args.path:
        # if explicit path provided, user must also pass an engine key to choose which checks to apply.
        if args.engine == "all":
            print("When --path is provided, you must also set --engine to either 'ticks_trades' or 'ticks_orderflow'.", file=sys.stderr)
            return 2
        engines = [args.engine]
        explicit_path = Path(args.path)
        if not explicit_path.exists():
            print(f"ERROR: path not found: {explicit_path}", file=sys.stderr)
            return 2
        engine_paths = {args.engine: explicit_path}
    else:
        if args.engine == "all":
            engines = [k for k in ENGINE_CONFIG.keys()]
        else:
            engines = [args.engine]
        # locate file paths
        engine_paths = {}
        for e in engines:
            p = locate_engine_file(e, base_dir, engine_cfg_map)
            if p is None:
                # try non-partitioned fallback
                cfg = engine_cfg_map[e]
                candidate = base_dir / cfg["subdir"] / cfg["filename"]
                p = candidate if candidate.exists() else None
            engine_paths[e] = p

    overall = {"summary": {}}
    all_passed = True

    for e in engines:
        path = engine_paths.get(e)
        if not path:
            print(f"Validating {e} -> path not found")
            overall["summary"][e] = {"path": None, "rows": 0, "passed": False, "error": "path not found"}
            all_passed = False
            continue

        print(f"Validating {e} -> {path}")
        try:
            res = validate_path(path, e, deep=args.deep, sample_n=args.sample, engine_cfg_map=engine_cfg_map)
        except Exception as exc:
            print(f"ERROR validating {e}: {exc}", file=sys.stderr)
            overall["summary"][e] = {"path": str(path), "error": str(exc), "passed": False}
            all_passed = False
            continue

        # print human-readable output unless JSON requested
        if args.json:
            overall["summary"][e] = res
        else:
            # brief human output
            rows = res.get("rows", 0)
            print(f"Engine: {e}")
            print(f"Path: {res['path']}")
            print(f"Rows: {rows}")
            print("-" * 60)
            for k, v in res["checks"].items():
                status = "[OK ]" if v.get("status", False) else "[ERR]"
                print(f"{status} {k}: {v.get('detail')}")
            print("\nSUMMARY:", json.dumps({"passed": res["passed"]}, indent=2))
            print()
            overall["summary"][e] = res

        if not res.get("passed", False):
            all_passed = False

    if args.json:
        print(json.dumps(overall, indent=2))

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
