"""Validator for Cross-Asset / Funding base features."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from feature_engineering.utils.dtype_enforcement import DtypeValidationError, validate_dtypes
from feature_engineering.utils.validation_helpers import ValidationError, check_value_range


class FeatureValidationError(ValueError):
    """Raised when validation fails for Cross-Asset / Funding features."""


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


def validate_cross_asset_funding_features(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate Cross-Asset / Funding features against the frozen schema."""
    schema_features = _load_schema()
    expected_columns = [feature["name"] for feature in schema_features]

    present_cross_columns = [col for col in df.columns if col.startswith("cross__")]
    if len(present_cross_columns) != len(expected_columns):
        raise FeatureValidationError(f"Expected exactly {len(expected_columns)} cross__ columns, found {len(present_cross_columns)}.")

    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise FeatureValidationError(f"Missing expected feature columns: {missing}")

    dtype_expectations = {feature["name"]: feature["dtype"] for feature in schema_features}
    try:
        validate_dtypes(df[expected_columns], dtype_expectations)
    except DtypeValidationError as exc:
        raise FeatureValidationError(str(exc)) from exc

    try:
        _validate_ranges(df, schema_features)
    except ValidationError as exc:
        raise FeatureValidationError(str(exc)) from exc

    # Explicitly reject NaNs and infinities.
    nulls = df[expected_columns].isna().sum()
    if nulls.any():
        raise FeatureValidationError(f"Nulls present in validated features: {nulls[nulls > 0].to_dict()}")

    for col in expected_columns:
        if not np.isfinite(df[col]).all():
            raise FeatureValidationError(f"Non-finite values detected in column '{col}'.")

    boolean_columns = [feature["name"] for feature in schema_features if feature["dtype"] == "bool"]
    for col in boolean_columns:
        if df[col].dtype != bool:
            raise FeatureValidationError(f"Boolean feature '{col}' must have dtype bool; found {df[col].dtype}.")

    return {"validated": True, "validated_columns": expected_columns}


__all__ = ["validate_cross_asset_funding_features", "FeatureValidationError"]
