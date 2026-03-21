"""Validator for Higher-Order engineered features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from feature_engineering.utils.dtype_enforcement import DtypeValidationError, validate_dtypes
from feature_engineering.utils.validation_helpers import ValidationError, check_value_range


class FeatureValidationError(ValueError):
    """Raised when higher-order feature validation fails."""


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


def _assert_no_nulls(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if df[col].isna().any():
            raise FeatureValidationError(f"Null values detected in '{col}'.")


def _assert_finite(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if not np.isfinite(df[col]).all():
            raise FeatureValidationError(f"Non-finite values detected in '{col}'.")


def _assert_dependencies(base_df: pd.DataFrame | None) -> None:
    if base_df is None:
        return
    required = [
        "price__last",
        "price__mid",
        "volume__tick",
        "spread__l1",
        "ob__imbalance",
        "ob__top_level_size_bid",
        "ob__top_level_size_ask",
        "ob__total_depth_bid",
        "ob__total_depth_ask",
        "tick_return",
        "of__signed_volume",
        "of__price_impact_per_unit_volume",
        "regime__liquidity_score",
    ]
    missing = [col for col in required if col not in base_df.columns]
    if missing:
        raise FeatureValidationError(f"Missing required base feature dependencies: {missing}")
    _assert_no_nulls(base_df, required)
    _assert_finite(base_df, required)


def validate_higher_order_features(
    df: pd.DataFrame,
    *,
    base_df: pd.DataFrame | None = None,
    schema_path: Path | None = None,
) -> Dict[str, Any]:
    """Validate higher-order engineered features against the authoritative schema."""

    schema_features = _load_schema(schema_path)
    expected_columns = [feature["name"] for feature in schema_features]
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise FeatureValidationError(f"Missing higher-order feature columns: {missing}")
    if len(expected_columns) != 14:
        raise FeatureValidationError("Schema must define exactly 14 higher-order features.")

    dtype_expectations = {feature["name"]: feature["dtype"] for feature in schema_features}
    try:
        validate_dtypes(df, dtype_expectations)
    except DtypeValidationError as exc:
        raise FeatureValidationError(str(exc)) from exc

    try:
        _validate_ranges(df, schema_features)
    except (ValidationError, DtypeValidationError) as exc:
        raise FeatureValidationError(str(exc)) from exc

    _assert_no_nulls(df, expected_columns)
    _assert_finite(df, expected_columns)
    _assert_dependencies(base_df)

    return {
        "validated": True,
        "null_counts": {col: int(df[col].isna().sum()) for col in expected_columns},
    }


__all__ = ["validate_higher_order_features", "FeatureValidationError"]
