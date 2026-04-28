"""Strict validator for Futures Basis Structure dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


class BasisStructureValidationError(Exception):
    pass


REQUIRED_COLUMNS: List[str] = [
    "meta__timestamp",
    "meta__sequence_id",
    "fut__perp_spot_basis",
    "fut__basis_change",
    "fut__basis_velocity",
    "fut__basis_zscore",
    "fut__basis_regime_flag",
    "fut__basis_compression_ratio",
    "fut__basis_mean_reversion_score",
]

FLOAT_FEATURES: List[str] = [
    "fut__perp_spot_basis",
    "fut__basis_change",
    "fut__basis_velocity",
    "fut__basis_zscore",
    "fut__basis_compression_ratio",
    "fut__basis_mean_reversion_score",
]


def _load_futures_base_schema() -> Dict:
    schema_path = Path(__file__).resolve().parents[1] / "schema" / "Futures Base Schema V1 0 · json"
    with schema_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_basis_structure_df(df: pd.DataFrame) -> bool:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise BasisStructureValidationError(f"Missing required columns: {missing}")

    ts = df["meta__timestamp"]
    seq = df["meta__sequence_id"]
    if str(ts.dtype) != "datetime64[ns, UTC]":
        raise BasisStructureValidationError(f"meta__timestamp must be datetime64[ns, UTC], got {ts.dtype}")
    if str(seq.dtype) != "int64":
        raise BasisStructureValidationError(f"meta__sequence_id must be int64, got {seq.dtype}")

    if not ts.is_monotonic_increasing:
        raise BasisStructureValidationError("meta__timestamp must be monotonic increasing")
    if not seq.is_monotonic_increasing:
        raise BasisStructureValidationError("meta__sequence_id must be monotonic increasing")

    null_counts = df[REQUIRED_COLUMNS].isna().sum()
    nulls = {k: int(v) for k, v in null_counts.items() if int(v) > 0}
    if nulls:
        raise BasisStructureValidationError(f"Zero-null policy violated: {nulls}")

    for col in FLOAT_FEATURES:
        if str(df[col].dtype) != "float32":
            raise BasisStructureValidationError(f"{col} must be float32, got {df[col].dtype}")

    if str(df["fut__basis_regime_flag"].dtype) != "bool":
        raise BasisStructureValidationError(f"fut__basis_regime_flag must be bool, got {df['fut__basis_regime_flag'].dtype}")

    schema = _load_futures_base_schema()
    rules = schema.get("validation_rules", {})
    if not rules.get("strict_dtype_enforcement", False):
        raise BasisStructureValidationError("Futures base schema does not enforce strict dtypes")
    if not rules.get("fail_on_null", False):
        raise BasisStructureValidationError("Futures base schema does not enforce null-fail")
    if not rules.get("fail_on_non_monotonic_sequence", False):
        raise BasisStructureValidationError("Futures base schema does not enforce monotonic sequence")

    return True
