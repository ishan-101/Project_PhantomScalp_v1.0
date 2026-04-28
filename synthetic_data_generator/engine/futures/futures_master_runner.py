"""Institutional deterministic master runner for the futures engine pipeline."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

_here = os.path.dirname(__file__)
root = os.path.abspath(os.path.join(_here, "../../.."))
if root not in sys.path:
    sys.path.insert(0, root)

from synthetic_data_generator.engine.config.loader import load_config
from synthetic_data_generator.engine.meta_provenance import manifest as manifest_mod
from synthetic_data_generator.engine.meta_provenance import provenance_helper as prov

from synthetic_data_generator.engine.futures.open_interest.runner import run_engine as run_open_interest
from synthetic_data_generator.engine.futures.funding_rate.runner import run_engine as run_funding_rate
from synthetic_data_generator.engine.futures.basis_structure.runner import run_engine as run_basis_structure
from synthetic_data_generator.engine.futures.positioning.runner import run_engine as run_positioning
from synthetic_data_generator.engine.futures.liquidation_pressure.runner import run_engine as run_liquidation_pressure
from synthetic_data_generator.engine.futures.volume_flow.runner import run_engine as run_volume_flow
from synthetic_data_generator.engine.futures.leverage_metrics.runner import run_engine as run_leverage_metrics
from synthetic_data_generator.engine.futures.derivatives_regime.runner import run_engine as run_derivatives_regime


PIPELINE_NAME = "futures_master_runner"
PROVENANCE_FILE_NAME = "futures_master_runner_provenance.jsonl"


@dataclass(frozen=True)
class StepSpec:
    step_name: str
    engine_name: str
    path_key: str
    output_file_name: str
    runner: Callable[..., Dict[str, Any]]


PIPELINE_STEPS: List[StepSpec] = [
    StepSpec("open_interest", "fut_open_interest", "fut_open_interest", "fut_open_interest.parquet", run_open_interest),
    StepSpec("funding_rate", "fut_funding_rate", "fut_funding_rate", "fut_funding_rate.parquet", run_funding_rate),
    StepSpec("basis_structure", "fut_basis_structure", "fut_basis_structure", "fut_basis_structure.parquet", run_basis_structure),
    StepSpec("positioning", "fut_positioning", "fut_positioning", "fut_positioning.parquet", run_positioning),
    StepSpec("liquidation_pressure", "fut_liquidation_pressure", "fut_liquidation_pressure", "fut_liquidation_pressure.parquet", run_liquidation_pressure),
    StepSpec("volume_flow", "fut_volume_flow", "fut_volume_flow", "fut_volume_flow.parquet", run_volume_flow),
    StepSpec("leverage_metrics", "fut_leverage_metrics", "fut_leverage_metrics", "fut_leverage_metrics.parquet", run_leverage_metrics),
    StepSpec("derivatives_regime", "fut_derivatives_regime", "fut_derivatives_regime", "fut_derivatives_regime.parquet", run_derivatives_regime),
]


def _resolve_partition_date(cfg: Dict[str, Any], explicit_partition_date: Optional[str]) -> str:
    if explicit_partition_date:
        return str(explicit_partition_date)

    base = Path(cfg["paths"]["base"])
    trades_base = base / Path(cfg["paths"]["ticks_trades"])
    orderflow_base = base / Path(cfg["paths"]["ticks_orderflow"])

    trades_dates = {p.name.split("=", 1)[-1] for p in trades_base.glob("date=*") if p.is_dir()}
    orderflow_dates = {p.name.split("=", 1)[-1] for p in orderflow_base.glob("date=*") if p.is_dir()}
    common_dates = sorted(trades_dates.intersection(orderflow_dates))

    if not common_dates:
        raise RuntimeError(
            "Unable to resolve a common futures partition_date from ticks_trades and ticks_orderflow. "
            f"trades_base={trades_base}, orderflow_base={orderflow_base}"
        )

    return common_dates[-1]


def _step_output_path(cfg: Dict[str, Any], step: StepSpec, partition_date: str) -> Path:
    return Path(cfg["paths"]["base"]) / Path(cfg["paths"][step.path_key]) / f"date={partition_date}" / step.output_file_name


def _manifest_path(cfg: Dict[str, Any]) -> Path:
    base = Path(cfg.get("paths", {}).get("base", "synthetic_data_generator/outputs"))
    manifest_name = cfg.get("manifest", {}).get("manifest_name", "_manifest.json")
    meta_dir = Path(cfg.get("paths", {}).get("meta", "meta"))
    return base / meta_dir / manifest_name


def _master_provenance_path(cfg: Dict[str, Any]) -> Path:
    base = Path(cfg.get("paths", {}).get("base", "synthetic_data_generator/outputs"))
    meta_dir = Path(cfg.get("paths", {}).get("meta", "meta"))
    return base / meta_dir / PROVENANCE_FILE_NAME


def _load_master_provenance(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    p = _master_provenance_path(cfg)
    if not p.exists():
        return []

    records: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(json.loads(stripped))
    return records


def _append_master_provenance(cfg: Dict[str, Any], record: Dict[str, Any]) -> None:
    p = _master_provenance_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, default=str))
        fh.write("\n")


def _manifest_has_entry_for_step(cfg: Dict[str, Any], step: StepSpec, partition_date: str, output_path: Path) -> bool:
    manifest_path = _manifest_path(cfg)
    if not manifest_path.exists():
        return False

    entries = manifest_mod.read_manifest(manifest_path)
    base = Path(cfg["paths"]["base"]).resolve()
    output_resolved = output_path.resolve()
    try:
        output_relative = str(output_resolved.relative_to(base))
    except ValueError:
        output_relative = str(output_resolved)

    for entry in entries:
        if entry.get("engine") != step.engine_name:
            continue
        if str(entry.get("partition_date")) != str(partition_date):
            continue
        file_path = str(entry.get("file_path", ""))
        if file_path in {output_relative, str(output_resolved), str(output_path)}:
            return True

    return False


def _has_prior_step_provenance(cfg: Dict[str, Any], step_name: str, partition_date: str) -> bool:
    for record in reversed(_load_master_provenance(cfg)):
        if str(record.get("partition_date")) != str(partition_date):
            continue
        step_records = record.get("step_results", [])
        for step_rec in step_records:
            if step_rec.get("step") != step_name:
                continue
            if step_rec.get("status") in {"completed", "skipped_existing_valid_output"}:
                return True
    return False


def _validate_output_integrity(cfg: Dict[str, Any], step: StepSpec, partition_date: str) -> Tuple[bool, Dict[str, Any]]:
    output_path = _step_output_path(cfg, step, partition_date)
    diagnostics: Dict[str, Any] = {
        "output_path": str(output_path),
        "parquet_exists": output_path.exists(),
        "parquet_non_empty": False,
        "row_count": 0,
        "manifest_entry_exists": False,
        "provenance_exists": False,
        "corruption_check_passed": False,
    }

    if not output_path.exists():
        return False, diagnostics

    try:
        row_count = int(len(pd.read_parquet(output_path, columns=["meta__timestamp"])))
    except Exception:
        try:
            row_count = int(len(pd.read_parquet(output_path)))
        except Exception:
            return False, diagnostics

    diagnostics["row_count"] = row_count
    diagnostics["parquet_non_empty"] = row_count > 0
    diagnostics["corruption_check_passed"] = row_count > 0
    diagnostics["manifest_entry_exists"] = _manifest_has_entry_for_step(cfg, step, partition_date, output_path)
    diagnostics["provenance_exists"] = _has_prior_step_provenance(cfg, step.step_name, partition_date)

    passed = all(
        [
            diagnostics["parquet_exists"],
            diagnostics["parquet_non_empty"],
            diagnostics["manifest_entry_exists"],
            diagnostics["corruption_check_passed"],
            diagnostics["provenance_exists"],
            diagnostics["row_count"] > 0,
        ]
    )
    return passed, diagnostics


def run_engine(partition_date: Optional[str] = None, fail_step: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_config()
    resolved_partition = _resolve_partition_date(cfg, partition_date)

    injected_fail_step = fail_step or os.getenv("PHANTOMSCALP_FAIL_STEP")

    t0 = time.perf_counter()
    completed_steps: List[str] = []
    skipped_steps: List[str] = []
    step_results: List[Dict[str, Any]] = []
    final_output: Dict[str, Any] = {}

    for step in PIPELINE_STEPS:
        print(f"[MASTER] starting {step.step_name}")

        output_is_valid, output_diag = _validate_output_integrity(cfg, step, resolved_partition)
        if output_is_valid:
            skipped_steps.append(step.step_name)
            print(f"[MASTER] skipped {step.step_name}")
            step_results.append(
                {
                    "step": step.step_name,
                    "engine": step.engine_name,
                    "status": "skipped_existing_valid_output",
                    "partition_date": resolved_partition,
                    "output": output_diag,
                }
            )
            if step.step_name == "derivatives_regime":
                final_output[step.step_name] = output_diag["output_path"]
            continue

        if injected_fail_step and injected_fail_step == step.step_name:
            exc = RuntimeError(f"Injected failure for step '{step.step_name}'")
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            failure_payload = {
                "pipeline": PIPELINE_NAME,
                "status": "failed",
                "completed_steps": completed_steps,
                "skipped_steps": skipped_steps,
                "failed_step": step.step_name,
                "failure": {
                    "engine": step.engine_name,
                    "step": step.step_name,
                    "dependency_stage": step.step_name,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "prior_completed_steps": completed_steps,
                },
                "total_runtime_ms": elapsed_ms,
                "partition_date": resolved_partition,
                "final_output": final_output,
                "step_results": step_results,
            }
            _append_master_provenance(
                cfg,
                {
                    "pipeline": PIPELINE_NAME,
                    "status": "failed",
                    "partition_date": resolved_partition,
                    "config_version": cfg.get("meta", {}).get("config_version", "unknown"),
                    "config_hash": cfg.get("_config_hash") or prov.hash_config(cfg),
                    "seed": cfg.get("global", {}).get("seed"),
                    "generated_at_utc": prov.utc_now_iso(),
                    "runtime_ms": elapsed_ms,
                    "completed_steps": completed_steps,
                    "skipped_steps": skipped_steps,
                    "failed_step": step.step_name,
                    "step_results": step_results,
                },
            )
            print(f"[MASTER] failure at {step.step_name}")
            return failure_payload

        try:
            step_result = step.runner(partition_date=resolved_partition)
            actual_partition = str(step_result.get("partition_date"))
            if actual_partition != resolved_partition:
                raise RuntimeError(
                    f"Partition drift detected for {step.step_name}: "
                    f"expected={resolved_partition}, got={actual_partition}"
                )

            output_ok, output_diag_after = _validate_output_integrity(cfg, step, resolved_partition)
            if not output_ok:
                raise RuntimeError(f"Post-run validation failed for {step.step_name}: {output_diag_after}")

            provenance_record = step_result.get("provenance")
            if not isinstance(provenance_record, dict) or not prov.validate_provenance_schema(provenance_record):
                raise RuntimeError(f"Invalid provenance record for {step.step_name}")

            completed_steps.append(step.step_name)
            print(f"[MASTER] completed {step.step_name}")

            step_result_payload = {
                "step": step.step_name,
                "engine": step.engine_name,
                "status": "completed",
                "partition_date": resolved_partition,
                "timing_ms": step_result.get("timing_ms"),
                "rows": step_result.get("rows"),
                "output_files": step_result.get("output_files", []),
                "output": output_diag_after,
                "provenance": provenance_record,
            }
            step_results.append(step_result_payload)

            if step.step_name == "derivatives_regime":
                if step_result.get("output_files"):
                    final_output[step.step_name] = step_result["output_files"][0]
                else:
                    final_output[step.step_name] = output_diag_after["output_path"]

        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            failure_payload = {
                "pipeline": PIPELINE_NAME,
                "status": "failed",
                "completed_steps": completed_steps,
                "skipped_steps": skipped_steps,
                "failed_step": step.step_name,
                "failure": {
                    "engine": step.engine_name,
                    "step": step.step_name,
                    "dependency_stage": step.step_name,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "prior_completed_steps": completed_steps,
                },
                "total_runtime_ms": elapsed_ms,
                "partition_date": resolved_partition,
                "final_output": final_output,
                "step_results": step_results,
            }
            _append_master_provenance(
                cfg,
                {
                    "pipeline": PIPELINE_NAME,
                    "status": "failed",
                    "partition_date": resolved_partition,
                    "config_version": cfg.get("meta", {}).get("config_version", "unknown"),
                    "config_hash": cfg.get("_config_hash") or prov.hash_config(cfg),
                    "seed": cfg.get("global", {}).get("seed"),
                    "generated_at_utc": prov.utc_now_iso(),
                    "runtime_ms": elapsed_ms,
                    "completed_steps": completed_steps,
                    "skipped_steps": skipped_steps,
                    "failed_step": step.step_name,
                    "step_results": step_results,
                },
            )
            print(f"[MASTER] failure at {step.step_name}")
            return failure_payload

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    summary = {
        "pipeline": PIPELINE_NAME,
        "status": "success",
        "completed_steps": completed_steps,
        "skipped_steps": skipped_steps,
        "failed_step": None,
        "total_runtime_ms": elapsed_ms,
        "partition_date": resolved_partition,
        "final_output": final_output,
        "step_results": step_results,
    }

    _append_master_provenance(
        cfg,
        {
            "pipeline": PIPELINE_NAME,
            "status": "success",
            "partition_date": resolved_partition,
            "config_version": cfg.get("meta", {}).get("config_version", "unknown"),
            "config_hash": cfg.get("_config_hash") or prov.hash_config(cfg),
            "seed": cfg.get("global", {}).get("seed"),
            "generated_at_utc": prov.utc_now_iso(),
            "runtime_ms": elapsed_ms,
            "completed_steps": completed_steps,
            "skipped_steps": skipped_steps,
            "failed_step": None,
            "step_results": step_results,
        },
    )

    return summary


if __name__ == "__main__":
    print(json.dumps(run_engine(), indent=2, default=str))
