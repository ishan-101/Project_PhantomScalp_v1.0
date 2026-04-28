"""Runner for cross-asset funding engine with validation, manifest, and provenance."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _add_repo_root_to_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _import_engine():
    try:
        from synthetic_data_generator.engine.crossasset.crossasset_funding_engine import (
            FundingEngineError,
            generate_crossasset_funding,
        )
        return generate_crossasset_funding, FundingEngineError
    except Exception:
        repo_root = _add_repo_root_to_path()
        from synthetic_data_generator.engine.crossasset.crossasset_funding_engine import (
            FundingEngineError,
            generate_crossasset_funding,
        )
        return generate_crossasset_funding, FundingEngineError


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


CROSSASSET_FUNDING_SCHEMA_SPEC: Dict[str, Any] = {
    "required_columns": {
        "meta__timestamp": "datetime64[ns, UTC]",
        "meta__sequence_id": "int64",
    },
    "optional_columns": {},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-asset funding engine")
    parser.add_argument("--rows", type=int, required=False, help="Override row count")
    parser.add_argument(
        "--start-ts",
        dest="start_ts",
        type=str,
        required=False,
        help="Override start timestamp (e.g., 2025-12-01T00:00:00Z)",
    )
    return parser.parse_args()


def main() -> int:
    generate_crossasset_funding, FundingEngineError = _import_engine()
    load_config = _resolve_loader()
    manifest_mod = _resolve_meta_module("manifest")
    prov = _resolve_meta_module("provenance_helper")
    schema_validator = _resolve_meta_module("schema_validator")

    cfg = load_config()
    args = parse_args()

    print("[runner] starting cross-asset funding generation")
    try:
        result = generate_crossasset_funding(rows_override=args.rows, start_ts=args.start_ts)
    except FundingEngineError as exc:
        print(f"[runner] failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected
        print(f"[runner] unexpected failure: {exc}", file=sys.stderr)
        return 1

    manifest_info = result.get("manifest", {}) if isinstance(result, dict) else {}
    output_path = Path(manifest_info.get("path", "")).resolve()
    if not output_path.exists():
        raise FileNotFoundError(f"Expected output parquet not found at {output_path}")

    df = pd.read_parquet(output_path)

    validation_cfg = cfg.get("validation", {})
    schema_spec = dict(CROSSASSET_FUNDING_SCHEMA_SPEC)
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
        engine="crossasset_funding",
        dataset_type="crossasset_funding",
        symbol=str(result.get("base_symbol", "BASE")) if isinstance(result, dict) else "BASE",
        exchange=str(result.get("exchange", "EX")) if isinstance(result, dict) else "EX",
        row_count=int(result.get("rows", len(df)) if isinstance(result, dict) else len(df)),
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
        dataset_name="crossasset_funding",
        engine_name="crossasset_funding_engine",
        engine_version=cfg.get("meta", {}).get("config_version", "unknown"),
        config_version=cfg.get("meta", {}).get("config_version", "unknown"),
        config_hash=config_hash,
        time_range_start=global_cfg.get("default_start_ts", ""),
        time_range_end=global_cfg.get("default_end_ts", ""),
        rows=int(result.get("rows", len(df)) if isinstance(result, dict) else len(df)),
        symbol=str(result.get("base_symbol", "BASE")) if isinstance(result, dict) else "BASE",
        exchange=str(result.get("exchange", "EX")) if isinstance(result, dict) else "EX",
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
        "engine": "crossasset_funding",
        "dataset_type": "crossasset_funding",
        "rows": manifest_entry["row_count"],
        "output_files": [str(relative_output)],
        "partition_dates": [partition_date] if partition_date else [],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "validation": validation_result,
        "manifest_entry": manifest_entry,
        "provenance": provenance_record,
    }

    print(json.dumps(runner_result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())