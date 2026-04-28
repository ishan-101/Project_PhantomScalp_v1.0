"""Validator for Price / OHLCV base features prior to downstream usage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from feature_engineering.utils.dtype_enforcement import DtypeValidationError, validate_dtypes
from feature_engineering.utils.validation_helpers import ValidationError, check_value_range


class FeatureValidationError(ValueError):
    """Raised when feature validation fails for Price / OHLCV."""


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


def _validate_direction_flags(df: pd.DataFrame) -> None:
    direction = df["price__tick_direction"]
    allowed = {-1, 0, 1}
    invalid_mask = ~direction.isin(allowed)
    if invalid_mask.any():
        raise ValidationError(
            "price__tick_direction contains invalid values outside {-1, 0, 1}: "
            f"indices {list(direction.index[invalid_mask])}"
        )


def _validate_boolean_flags(df: pd.DataFrame) -> None:
    for col in ("price__jump_flag", "price__tick_sweep_flag"):
        if df[col].isna().any():
            raise ValidationError(f"Boolean flag column '{col}' contains nulls.")
        if df[col].dtype != bool:
            raise DtypeValidationError(
                f"Boolean flag column '{col}' must have dtype bool; found {df[col].dtype}."
            )


def validate_price_ohlcv_features(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate that all Price / OHLCV base features satisfy the schema contract.

    Args:
        df: DataFrame expected to contain all 25 Price / OHLCV features.

    Returns:
        Diagnostics summarizing null counts and default fill usage.

    Raises:
        FeatureValidationError: If validation fails for any rule.
    """

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
        _validate_direction_flags(df)
        _validate_boolean_flags(df)
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


__all__ = ["validate_price_ohlcv_features", "FeatureValidationError"]
