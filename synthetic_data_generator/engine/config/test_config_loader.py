#!/usr/bin/env python3
# synthetic_data_generator/engine/config/test_config_loader.py
"""
Smoke test for synthetic_config.yaml + loader.py

Run from repository root:
    python synthetic_data_generator/engine/config/test_config_loader.py
"""

import sys
from pathlib import Path

# Import loader (must run from repo root so package imports resolve)
try:
    from synthetic_data_generator.engine.config.loader import load_config, ConfigError
except Exception as e:
    print("[ERROR] Failed to import config loader:", e)
    sys.exit(1)


def fail(msg: str):
    print("[FAIL]", msg)
    sys.exit(1)


def ok(msg: str):
    print("[OK ]", msg)


def main():
    try:
        cfg = load_config()
    except Exception as e:
        print("[ERROR] load_config() raised an exception:")
        raise

    # Basic prints
    print("Loaded config hash:", cfg.get("_config_hash", "<missing>"))

    # 1) top-level required keys
    required_top = [
        "meta",
        "global",
        "rows",
        "paths",
        "partitioning",
        "writer",
        "sharder",
        "validation",
        "engines",
        "manifest",
        "logging",
    ]
    for k in required_top:
        if k not in cfg:
            fail(f"Missing top-level key: {k}")
    ok("All required top-level keys present")

    # 2) rows are positive ints
    rows = cfg["rows"]
    if not isinstance(rows, dict):
        fail("cfg['rows'] must be a mapping")
    for engine, v in rows.items():
        if not isinstance(v, int) or v <= 0:
            fail(f"rows.{engine} must be a positive int (got {v!r})")
    ok("rows: all engines have positive integer values")

    # 3) paths.base exists (if not, create it)
    base = cfg["paths"].get("base")
    if not isinstance(base, str):
        fail("paths.base must be a string")
    base_path = Path(base)
    if not base_path.exists():
        print(f"[INFO] paths.base does not exist, creating: {base_path}")
        try:
            base_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            fail(f"Could not create paths.base '{base_path}': {e}")
    ok(f"paths.base exists: {base_path.resolve()}")

    # 4) engine relative paths are strings and resolve under base
    engine_keys = [
        "spot", "orderbook_l2", "orderbook_l3",
        "ticks_trades", "ticks_orderflow",
        "options_chain", "options_iv_surface", "options_oi",
        "greeks_primary", "greeks_flow",
        "crossasset_corr", "crossasset_funding",
        "meta"
    ]
    for k in engine_keys:
        val = cfg["paths"].get(k)
        if not isinstance(val, str):
            fail(f"paths.{k} must be a string (got {val!r})")
        p = base_path / val
        # not creating all engine folders, but check path is a valid relative path
        if ".." in Path(val).parts:
            fail(f"paths.{k} should not contain parent traversal ('..'): {val}")
    ok("paths: engine relative paths validated")

    # 5) partitioning is date-only
    partition_cols = cfg["partitioning"].get("columns", [])
    if not isinstance(partition_cols, list):
        fail("partitioning.columns must be a list")
    if "date" not in partition_cols:
        fail("partitioning.columns must include 'date' (date-based partitioning locked)")
    ok("partitioning: date-based partitioning present")

    # 6) sharder enabled and max_rows_per_file check
    sharding = cfg["partitioning"].get("sharding", {})
    if not isinstance(sharding, dict):
        fail("partitioning.sharding must be a mapping")
    if not sharding.get("enabled", False):
        print("[WARN] partitioning.sharding.enabled is False (OK if intentional)")
    max_rows = sharding.get("max_rows_per_file")
    if not isinstance(max_rows, int) or max_rows <= 0:
        fail("partitioning.sharding.max_rows_per_file must be a positive integer")
    ok(f"sharding: max_rows_per_file = {max_rows}")

    # 7) validation flags
    if not cfg["validation"].get("timestamp_tz_required", False):
        print("[WARN] validation.timestamp_tz_required is not enabled (recommended true)")

    # 8) quick sanity summary
    total_rows = sum(rows.values())
    print(f"Summary: {len(rows)} engines declared, total target rows = {total_rows:,}")

    # Done
    print("\nSMOKE TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
