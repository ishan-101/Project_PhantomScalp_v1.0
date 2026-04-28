"""Validator for Options Chain base features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from feature_engineering.utils.dtype_enforcement import DtypeValidationError, validate_dtypes
from feature_engineering.utils.validation_helpers import ValidationError, check_value_range


class FeatureValidationError(ValueError):
    """Raised when validation fails for Options Chain features."""


def _load_schema(schema_path: Path | None = None) -> List[Dict[str, Any]]:
    path = schema_path or Path(__file__).with_name("schema.json")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["features"]


def _convert_bound(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value == "inf":
            return np.inf
        if value == "-inf":
            return -np.inf
    return float(value)


def _validate_ranges(df: pd.DataFrame, features: List[Dict[str, Any]]) -> None:
    for feature in features:
        name = feature["name"]
        lower_raw, upper_raw = feature.get("expected_range", (None, None))
        lower = _convert_bound(lower_raw)
        upper = _convert_bound(upper_raw)
        check_value_range(df[name], min_value=lower, max_value=upper, inclusive=True)


def _validate_boolean(df: pd.DataFrame, boolean_features: List[str]) -> None:
    for col in boolean_features:
        if df[col].isna().any():
            raise ValidationError(f"Boolean flag column '{col}' contains nulls.")
        if df[col].dtype != bool:
            raise DtypeValidationError(
                f"Boolean flag column '{col}' must have dtype bool; found {df[col].dtype}."
            )


def validate_options_chain_features(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate Options Chain base features against schema constraints."""

    schema_features = _load_schema()
    expected_columns = [feature["name"] for feature in schema_features]
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise FeatureValidationError(f"Missing expected feature columns: {missing}")

    dtype_expectations = {feature["name"]: feature["dtype"] for feature in schema_features}
    try:
        validate_dtypes(df, dtype_expectations)
    except DtypeValidationError as exc:
        raise FeatureValidationError(str(exc)) from exc

    try:
        _validate_ranges(df, schema_features)
        boolean_cols = [feature["name"] for feature in schema_features if feature["dtype"] == "bool"]
        _validate_boolean(df, boolean_cols)
    except (ValidationError, DtypeValidationError) as exc:
        raise FeatureValidationError(str(exc)) from exc

    null_counts = {col: int(df[col].isna().sum()) for col in expected_columns}
    if any(count > 0 for count in null_counts.values()):
        raise FeatureValidationError(
            "Nulls present in validated features: "
            + ", ".join(f"{col}={count}" for col, count in null_counts.items() if count > 0)
        )

    default_fill_counts = {}
    for feature in schema_features:
        col = feature["name"]
        default_value = feature.get("default_fill")
        default_fill_counts[col] = int((df[col] == default_value).sum())

    return {
        "validated": True,
        "null_counts": null_counts,
        "default_fill_counts": default_fill_counts,
    }


__all__ = ["validate_options_chain_features", "FeatureValidationError"]
