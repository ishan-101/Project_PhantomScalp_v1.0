"""
Shared schema validation utilities for synthetic parquet datasets.

Philosophy (declarative, engine-agnostic):
* Required columns: must be present; absence is an error.
* Optional columns: validated when present; absence is permitted.
* Type expectations: per-column dtype declarations enforced deterministically.
* Allowed null ratios: global cap with per-column overrides for flexibility.
* Engine-specific extensions: injected via schema specs/custom checks, never hardcoded.

Schema specification (example shape, composable and extensible):
schema_spec = {
    "required_columns": {"col": "int64", ...},
    "optional_columns": {"maybe": "float64", ...},
    "dtypes": {"override": "string", ...},  # optional alias/override map
    "max_null_ratio": 0.0,  # global default
    "per_column_null_ratio": {"col": 0.1, ...},  # per-column overrides
    "custom_checks": [callable | {"name": str, "fn": callable}],
    "metadata": {"schema_version": "v1", "description": "..."},
}

Validation output contract:
{
    "passed": bool,
    "errors": {<section>: [messages]},
    "warnings": {<section>: [messages]},
    "checks": {<section>: {"passed": bool, "details": ...}},
}
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import pandas as pd

ValidationResult = Dict[str, Any]


def validate_schema(df: pd.DataFrame, schema_spec: Mapping[str, Any]) -> ValidationResult:
    """Run all configured schema checks against the provided DataFrame.

    The DataFrame is never mutated. Only validates what is described in
    ``schema_spec``; missing sections are skipped gracefully.
    """

    required = set(schema_spec.get("required_columns", {}) or {})
    optional = schema_spec.get("optional_columns", {}) or {}
    dtype_map = _merge_dtype_maps(
        schema_spec.get("required_columns", {}),
        optional,
        schema_spec.get("dtypes", {}),
    )
    max_null_ratio = schema_spec.get("max_null_ratio")
    per_column_null_ratio = schema_spec.get("per_column_null_ratio") or {}

    results: List[ValidationResult] = []

    if required:
        results.append(check_required_columns(df, required))

    if dtype_map:
        results.append(check_dtypes(df, dtype_map))

    if max_null_ratio is not None:
        results.append(check_null_ratios(df, max_null_ratio, per_column_null_ratio))

    custom_checks = schema_spec.get("custom_checks") or []
    for check in custom_checks:
        results.append(_run_custom_check(df, check))

    return merge_validation_results(*results)


def check_required_columns(df: pd.DataFrame, required: Iterable[str]) -> ValidationResult:
    """Validate presence of required columns."""

    required_set = set(required)
    missing = sorted(col for col in required_set if col not in df.columns)

    passed = not missing
    errors = {"required_columns": [f"Missing required columns: {', '.join(missing)}"]} if missing else {}

    return {
        "passed": passed,
        "errors": errors,
        "warnings": {},
        "checks": {
            "required_columns": {
                "passed": passed,
                "details": {
                    "missing": missing,
                    "present": sorted(col for col in required_set if col in df.columns),
                },
            }
        },
    }


def check_dtypes(df: pd.DataFrame, dtype_map: Mapping[str, Any]) -> ValidationResult:
    """Validate column dtypes using pandas dtype resolution."""

    mismatches = {}
    details = {}

    for col, expected in dtype_map.items():
        if col not in df.columns:
            continue  # Column presence handled elsewhere.

        expected_dtype = _coerce_dtype(expected)
        actual_dtype = df[col].dtype
        match = pd.api.types.is_dtype_equal(actual_dtype, expected_dtype)

        details[col] = {
            "expected": str(expected_dtype),
            "actual": str(actual_dtype),
            "passed": match,
        }

        if not match:
            mismatches[col] = {
                "expected": str(expected_dtype),
                "actual": str(actual_dtype),
            }

    passed = not mismatches
    errors = {"dtypes": [f"Type mismatches: {mismatches}"]} if mismatches else {}

    return {
        "passed": passed,
        "errors": errors,
        "warnings": {},
        "checks": {"dtypes": {"passed": passed, "details": details}},
    }


def check_null_ratios(
    df: pd.DataFrame,
    max_ratio: float,
    per_column: Optional[Mapping[str, float]] = None,
) -> ValidationResult:
    """Validate null ratios against global and per-column thresholds."""

    per_column = per_column or {}
    violations = {}
    details = {}

    for col in df.columns:
        col_ratio = float(df[col].isna().mean()) if len(df) else 0.0
        limit = per_column.get(col, max_ratio)
        passed_col = col_ratio <= limit

        details[col] = {
            "null_ratio": col_ratio,
            "limit": limit,
            "passed": passed_col,
        }

        if not passed_col:
            violations[col] = {"null_ratio": col_ratio, "limit": limit}

    passed = not violations
    errors = {"null_ratios": [f"Null ratio violations: {violations}"]} if violations else {}

    return {
        "passed": passed,
        "errors": errors,
        "warnings": {},
        "checks": {"null_ratios": {"passed": passed, "details": details}},
    }


def merge_validation_results(*results: ValidationResult) -> ValidationResult:
    """Merge individual validation results into a single aggregate report."""

    aggregate_errors: Dict[str, List[str]] = {}
    aggregate_warnings: Dict[str, List[str]] = {}
    aggregate_checks: Dict[str, Any] = {}
    overall_passed = True

    for result in results:
        if not result:
            continue

        overall_passed = overall_passed and result.get("passed", False)

        for section, messages in result.get("errors", {}).items():
            aggregate_errors.setdefault(section, []).extend(messages)

        for section, messages in result.get("warnings", {}).items():
            aggregate_warnings.setdefault(section, []).extend(messages)

        for section, check_detail in result.get("checks", {}).items():
            aggregate_checks[section] = check_detail

    return {
        "passed": overall_passed and not aggregate_errors,
        "errors": aggregate_errors,
        "warnings": aggregate_warnings,
        "checks": aggregate_checks,
    }


def format_validation_report(result: ValidationResult) -> str:
    """Human-readable formatting for validation results."""

    lines = ["Validation Report", f"Overall passed: {result.get('passed', False)}"]

    if result.get("errors"):
        lines.append("Errors:")
        for section, messages in result["errors"].items():
            for msg in messages:
                lines.append(f"  - [{section}] {msg}")

    if result.get("warnings"):
        lines.append("Warnings:")
        for section, messages in result["warnings"].items():
            for msg in messages:
                lines.append(f"  - [{section}] {msg}")

    if result.get("checks"):
        lines.append("Checks:")
        for section, detail in result["checks"].items():
            lines.append(f"  - {section}: {detail}")

    return "\n".join(lines)


def _merge_dtype_maps(
    required: Mapping[str, Any],
    optional: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> Dict[str, Any]:
    """Merge dtype maps with precedence: overrides > required > optional."""

    merged = dict(optional)
    merged.update(required)
    merged.update(overrides)
    return merged


def _coerce_dtype(dtype_spec: Any):
    """Convert a dtype specification into a pandas-compatible dtype."""

    try:
        return pd.api.types.pandas_dtype(dtype_spec)
    except TypeError:
        return pd.api.types.pandas_dtype(str(dtype_spec))


def _run_custom_check(df: pd.DataFrame, check: Any) -> ValidationResult:
    """Execute a custom check callable and normalize its output."""

    if isinstance(check, Mapping):
        fn = check.get("fn")
        name = check.get("name", getattr(fn, "__name__", "custom_check"))
    else:
        fn = check
        name = getattr(check, "__name__", "custom_check")

    if not callable(fn):
        return {
            "passed": False,
            "errors": {"custom_checks": [f"Invalid custom check: {check}"]},
            "warnings": {},
            "checks": {name: {"passed": False, "details": "not callable"}},
        }

    result = fn(df)
    if not isinstance(result, Mapping):
        return {
            "passed": False,
            "errors": {"custom_checks": [f"Custom check {name} returned invalid result"]},
            "warnings": {},
            "checks": {name: {"passed": False, "details": "invalid result type"}},
        }

    normalized = {
        "passed": bool(result.get("passed", False)),
        "errors": result.get("errors", {}),
        "warnings": result.get("warnings", {}),
        "checks": {name: result.get("checks", result)},
    }

    return normalized
