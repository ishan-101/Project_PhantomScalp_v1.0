"""Validation gate for microstructure L2/L3 base features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from feature_engineering.utils.dtype_enforcement import validate_dtypes
from feature_engineering.utils.validation_helpers import ValidationError, check_value_range

SCHEMA_PATH = Path(__file__).with_name("schema.json")


def _load_schema() -> Dict:
    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _expected_dtype_mapping(schema: Dict) -> Dict[str, str]:
    return {feature["name"]: feature["dtype"] for feature in schema.get("features", [])}


def _expected_ranges(schema: Dict) -> Dict[str, Tuple[float | None, float | None]]:
    ranges: Dict[str, Tuple[float | None, float | None]] = {}
    for feature in schema.get("features", []):
        raw_min, raw_max = feature.get("expected_range", [None, None])
        min_val = None if raw_min in (None, "-inf") else float(raw_min)
        max_val = None if raw_max in (None, "inf") else float(raw_max)
        ranges[feature["name"]] = (min_val, max_val)
    return ranges


def validate_microstructure_l2_l3_features(df: pd.DataFrame) -> None:
    """Validate presence, dtype, and value ranges for all microstructure base features."""

    schema = _load_schema()
    feature_names = [feature["name"] for feature in schema.get("features", [])]

    missing = [col for col in feature_names if col not in df.columns]
    if missing:
        raise ValidationError(f"Missing microstructure features: {missing}")

    expected_dtypes = _expected_dtype_mapping(schema)
    validate_dtypes(df, expected_dtypes)

    ranges = _expected_ranges(schema)
    for col, (min_val, max_val) in ranges.items():
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            if series.isna().any():
                raise ValidationError(f"Boolean feature '{col}' contains nulls.")
            continue
        check_value_range(series, min_value=min_val, max_value=max_val, inclusive=True)

    # Entropy bounded between 0 and log(number of depth bins).
    level_count = len([c for c in df.columns if c.startswith("bid_size_")])
    entropy_max = np.log(max(1, level_count * 2))
    check_value_range(df["ob__book_entropy"], min_value=0.0, max_value=entropy_max, inclusive=True)

    nulls = {col: int(df[col].isna().sum()) for col in feature_names}
    lingering = {col: count for col, count in nulls.items() if count > 0}
    if lingering:
        raise ValidationError(f"Nulls present in validated features: {lingering}")

