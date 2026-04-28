"""Semantic guardrails for the Meta / QC / Provenance base feature family.

This script MUST be run before introducing the remaining artifacts in this
family. It statically inspects the Meta / QC / Provenance modules (when
present) to ensure they:

* Depend only on schemas, timestamps, null counts, range checks, and config
  thresholds.
* Never import or call feature computation logic from other base features.
* Never access rolling buffers or other feature internals.
* Never assign to non-meta columns.

If any violation is detected, an exception is raised and execution stops.
When all checks pass, the script prints:

    Meta / QC / Provenance semantic validation — PASSED
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable, List, Optional, Sequence, Set

import pandas as pd


META_COLUMNS: Sequence[str] = (
    "meta__timestamp",
    "meta__sequence_id",
    "meta__source_feed",
    "meta__data_latency_ms",
    "meta__feature_confidence",
    "meta__staleness_flag",
    "meta__ingest_quality_score",
    "meta__feature_mask_count",
)

BANNED_IMPORT_PREFIXES: Sequence[str] = ("feature_engineering.base_features.",)
ALLOWED_ANALYTICS_DEPENDENCIES: Sequence[str] = (
    "typing",
    "datetime",
    "json",
    "math",
    "pathlib",
    "pandas",
    "numpy",
    "feature_engineering.utils",
)
BROADLY_ALLOWED_MODULES: Set[str] = {"__future__", "dataclasses", "sys"}
BANNED_BUFFER_METHODS: Set[str] = {"rolling", "ewm", "expanding"}
META_MODULE_NAMES: Sequence[str] = ("features.py", "validator.py", "smoke_test_meta_qc_provenance.py")


class SemanticViolation(RuntimeError):
    """Raised when the Meta / QC / Provenance semantic contract is broken."""


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise SemanticViolation(f"Unable to load module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _string_key(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Index):  # type: ignore[attr-defined]
        return _string_key(node.value)  # pragma: no cover
    return None


def _check_imports(tree: ast.AST, module_path: Path) -> None:
    forbidden: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                if mod in BROADLY_ALLOWED_MODULES:
                    continue
                if _is_forbidden_import(mod):
                    forbidden.append(mod)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module
            if mod in BROADLY_ALLOWED_MODULES:
                continue
            if mod and _is_forbidden_import(mod):
                forbidden.append(mod)
            elif mod and not any(mod.startswith(prefix) for prefix in ALLOWED_ANALYTICS_DEPENDENCIES):
                if mod.startswith("feature_engineering.base_features.meta_qc_provenance"):
                    continue
                # Unknown imports are considered suspicious unless clearly analytic/utility.
                forbidden.append(mod)
    if forbidden:
        raise SemanticViolation(
            f"{module_path.name} imports disallowed modules: {sorted(set(forbidden))}"
        )


def _is_forbidden_import(module_name: str) -> bool:
    if module_name.startswith("feature_engineering.base_features.meta_qc_provenance"):
        return False
    return any(module_name.startswith(prefix) for prefix in BANNED_IMPORT_PREFIXES)


def _check_buffer_usage(tree: ast.AST, module_path: Path) -> None:
    banned_hits: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in BANNED_BUFFER_METHODS:
            banned_hits.append(node.attr)
    if banned_hits:
        raise SemanticViolation(
            f"{module_path.name} references buffer methods {sorted(set(banned_hits))}, "
            "which are forbidden for Meta / QC / Provenance."
        )


def _check_assignments(tree: ast.AST, module_path: Path) -> None:
    if module_path.name not in {"features.py"}:
        return
    invalid_targets: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Subscript):
                key = _string_key(target.slice)  # type: ignore[arg-type]
                if key is None:
                    continue
                if not key.startswith("meta__"):
                    invalid_targets.append(key)
    if invalid_targets:
        raise SemanticViolation(
            f"{module_path.name} attempts to assign non-meta columns: {sorted(set(invalid_targets))}"
        )


def _check_runtime_contract(module_path: Path) -> None:
    if module_path.name != "features.py":
        return
    module = _load_module(module_path)
    if not hasattr(module, "compute_meta_qc_provenance_features"):
        return
    compute_fn = getattr(module, "compute_meta_qc_provenance_features")
    if not callable(compute_fn):
        raise SemanticViolation("compute_meta_qc_provenance_features must be callable.")

    input_df = pd.DataFrame(
        {
            "price__last": [100.0, 101.0],
            "ohlcv__close": [100.0, 101.0],
            "ingest_timestamp": pd.to_datetime(["2024-01-01T00:00:00Z"] * 2),
            "source_timestamp": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z"]),
        }
    )
    original = input_df.copy(deep=True)
    result = compute_fn(input_df, config=None)

    if not original.equals(input_df):
        raise SemanticViolation("Input DataFrame was mutated; Meta / QC must be read-only for inputs.")

    extra_columns = set(result.columns) - set(original.columns)
    non_meta = [col for col in extra_columns if col not in META_COLUMNS]
    if non_meta:
        raise SemanticViolation(f"Unexpected non-meta columns added: {sorted(non_meta)}")


def _assert_semantics() -> None:
    base_dir = Path(__file__).resolve().parent
    for module_name in META_MODULE_NAMES:
        module_path = base_dir / module_name
        if not module_path.exists():
            # Files may not exist yet; absence is not a violation during scaffold creation.
            continue
        tree = ast.parse(module_path.read_text())
        _check_imports(tree, module_path)
        _check_buffer_usage(tree, module_path)
        _check_assignments(tree, module_path)
        _check_runtime_contract(module_path)


def main() -> None:
    _assert_semantics()
    print("Meta / QC / Provenance semantic validation — PASSED")


if __name__ == "__main__":
    main()
