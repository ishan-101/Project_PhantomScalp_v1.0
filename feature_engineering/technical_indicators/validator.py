"""Validator for Technical Indicator base features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from feature_engineering.utils.dtype_enforcement import DtypeValidationError, validate_dtypes
from feature_engineering.utils.validation_helpers import ValidationError, check_value_range


class FeatureValidationError(ValueError):
    """Raised when validation fails for technical indicator features."""


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


def _validate_finite(df: pd.DataFrame, columns: List[str]) -> None:
    for col in columns:
        series = df[col]
        if series.isna().any():
            raise FeatureValidationError(f"Nulls detected in feature '{col}'.")
        if not np.isfinite(series.to_numpy()).all():
            raise FeatureValidationError(f"Non-finite values detected in feature '{col}'.")


def validate_technical_indicator_features(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate technical indicator features against the frozen schema."""

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
        _validate_finite(df, expected_columns)
        _validate_ranges(df, schema_features)
    except (ValidationError, FeatureValidationError) as exc:
        raise FeatureValidationError(str(exc)) from exc

    diagnostics = {
        "validated": True,
        "null_counts": {col: int(df[col].isna().sum()) for col in expected_columns},
        "default_fill_counts": {},
    }
    for feature in schema_features:
        col = feature["name"]
        default_value = feature.get("default_fill")
        diagnostics["default_fill_counts"][col] = int((df[col] == default_value).sum())

    return diagnostics


__all__ = ["validate_technical_indicator_features", "FeatureValidationError"]
