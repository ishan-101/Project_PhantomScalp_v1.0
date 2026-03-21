# run_orderbook_l3.py
"""
Runner script to execute orderbook_l3 engine with post-write validation, manifest, and provenance.
Usage:
    python synthetic_data_generator/engine/orderbook/run_orderbook_l3.py
"""
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Ensure project root on sys.path when run as script
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ENGINE_MODULE_CANDIDATES = [
    "synthetic_data_generator.engine.orderbook.orderbook_l3_engine",
    "engine.orderbook.orderbook_l3_engine",
]

for _module_path in ENGINE_MODULE_CANDIDATES:
    try:
        engine_module = importlib.import_module(_module_path)
        run_engine = getattr(engine_module, "run_engine")
        break
    except Exception:
        continue
else:
    raise ImportError("Unable to import orderbook_l3_engine.run_engine")


def _resolve_loader():
    candidates = [
        "synthetic_data_generator.engine.config.loader",
        "engine.config.loader",
    ]
    for module_path in candidates:
        spec = importlib.util.find_spec(module_path)
        if spec is not None:
            module = importlib.import_module(module_path)
            load_config = getattr(module, "load_config", None)
            if callable(load_config):
                return load_config
    file_path = Path(__file__).resolve().parents[1] / "config" / "loader.py"
    spec = importlib.util.spec_from_file_location("config_loader_direct", file_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to resolve config loader module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    load_config = getattr(module, "load_config", None)
    if callable(load_config):
        return load_config
    raise ImportError("load_config not found in config loader module.")


def _resolve_meta_module(module_suffix: str):
    candidates = [
        f"synthetic_data_generator.engine.meta_provenance.{module_suffix}",
        f"engine.meta_provenance.{module_suffix}",
    ]
    for module_path in candidates:
        spec = importlib.util.find_spec(module_path)
        if spec is not None:
            return importlib.import_module(module_path)
    raise ImportError(f"Unable to import meta_provenance module: {module_suffix}")


ORDERBOOK_L3_SCHEMA_SPEC: Dict[str, Any] = {
    "required_columns": {
        "meta__timestamp": "datetime64[ns, UTC]",
        "meta__sequence_id": "int64",
    },
    "optional_columns": {},
}


def _build_engine_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    base_path = Path(cfg.get("paths", {}).get("base", "data/synthetic_data"))
    rel_path = cfg.get("paths", {}).get("orderbook_l3", "orderbook/l3")
    out_path = base_path / rel_path / "orderbook_l3.parquet"

    compression = cfg.get("writer", {}).get(
        "compression", cfg.get("partitioning", {}).get("compression", "snappy")
    )

    sharder = cfg.get("partitioning", {}).get("sharding", {}) or {}
    default_part_rows = sharder.get("max_rows_per_file") or sharder.get("min_rows_per_file")

    engine_defaults = getattr(engine_module, "DEFAULT_CONFIG", {})

    return {
        **engine_defaults,
        "rows": int(cfg.get("rows", {}).get("orderbook_l3", engine_defaults.get("rows", 0))),
        "part_rows": int(default_part_rows or engine_defaults.get("part_rows", 20000)),
        "out_path": str(out_path),
        "seed": int(cfg.get("global", {}).get("seed", engine_defaults.get("seed", 0))),
        "compression": compression,
    }


if __name__ == "__main__":
    print("[runner] Starting orderbook_l3 engine...")

    load_config = _resolve_loader()
    manifest_mod = _resolve_meta_module("manifest")
    prov = _resolve_meta_module("provenance_helper")
    schema_validator = _resolve_meta_module("schema_validator")

    cfg = load_config()
    engine_cfg = _build_engine_config(cfg)
    res = run_engine(engine_cfg)

    output_path = Path(res.get("manifest", {}).get("path", "")).resolve() if isinstance(res, dict) else None
    if output_path is None or not output_path.exists():
        raise FileNotFoundError(f"Expected output parquet not found at {output_path}")

    df = pd.read_parquet(output_path)

    validation_cfg = cfg.get("validation", {})
    schema_spec = dict(ORDERBOOK_L3_SCHEMA_SPEC)
    if "max_null_ratio" in validation_cfg:
        schema_spec["max_null_ratio"] = validation_cfg.get("max_null_ratio")
    validation_result = {"passed": True, "errors": {}, "warnings": {}}
    if validation_cfg.get("enable_schema_validation", False):
        validation_result = schema_validator.validate_schema(df, schema_spec)
        if not validation_result.get("passed", False):
            print("[runner] schema validation failed:")
            print(json.dumps(validation_result, indent=2, default=str))
            if validation_cfg.get("dtype_strict", True):
                raise SystemExit("[runner] blocking validation failure; aborting.")
            else:
                print("[runner] continuing despite validation errors (non-blocking).")

    base_path = Path(cfg.get("paths", {}).get("base", "synthetic_data_generator/outputs")).resolve()
    try:
        relative_output = output_path.relative_to(base_path)
    except ValueError:
        relative_output = output_path

    partition_date: Optional[str] = None
    for parent in output_path.parents:
        if parent.name.startswith("date="):
            partition_date = parent.name.split("=", 1)[-1]
            break
    if partition_date is None and "meta__timestamp" in df:
        partition_date = str(pd.to_datetime(df["meta__timestamp"]).dt.date.iloc[0])

    manifest_entry = manifest_mod.build_manifest_entry(
        engine="orderbook_l3",
        dataset_type="orderbook_l3",
        symbol=str(res.get("symbol", "SYM")) if isinstance(res, dict) else "SYM",
        exchange=str(res.get("exchange", "EX")) if isinstance(res, dict) else "EX",
        row_count=int(res.get("rows", len(df)) if isinstance(res, dict) else len(df)),
        columns=list(df.columns),
        partition_date=partition_date or "",
        file_path=output_path,
        config=cfg,
    )
    manifest_entry["file_path"] = str(relative_output)

    manifest_cfg = cfg.get("manifest", {})
    manifest_name = manifest_cfg.get("manifest_name", "_manifest.json")
    manifest_dir = Path(cfg.get("paths", {}).get("meta", "meta"))
    manifest_path = base_path / manifest_dir / manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_mod.append_manifest(manifest_entry, manifest_path)

    config_hash = cfg.get("_config_hash") or prov.hash_config(cfg)
    global_cfg = cfg.get("global", {})
    provenance_record = prov.build_provenance_record(
        dataset_name="orderbook_l3",
        engine_name="orderbook_l3_engine",
        engine_version=cfg.get("meta", {}).get("config_version", "unknown"),
        config_version=cfg.get("meta", {}).get("config_version", "unknown"),
        config_hash=config_hash,
        time_range_start=global_cfg.get("default_start_ts", ""),
        time_range_end=global_cfg.get("default_end_ts", ""),
        rows=int(res.get("rows", len(df)) if isinstance(res, dict) else len(df)),
        symbol=str(res.get("symbol", "SYM")) if isinstance(res, dict) else "SYM",
        exchange=str(res.get("exchange", "EX")) if isinstance(res, dict) else "EX",
        environment=global_cfg.get("environment", "dev"),
        seed=global_cfg.get("seed"),
        notes=json.dumps(
            {
                "output_path": str(relative_output),
                "validation_passed": validation_result.get("passed", True),
            }
        ),
    )

    runner_result: Dict[str, Any] = {
        "engine": "orderbook_l3",
        "dataset_type": "orderbook_l3",
        "rows": manifest_entry["row_count"],
        "output_files": [str(relative_output)],
        "partition_dates": [partition_date] if partition_date else [],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "validation": validation_result,
        "manifest_entry": manifest_entry,
        "provenance": provenance_record,
    }

    print("[runner] Done.")
    print(json.dumps(runner_result, indent=2, default=str))