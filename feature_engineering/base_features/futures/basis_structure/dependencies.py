# basis_structure/dependencies.py (FINAL CORRECTED)

from __future__ import annotations

from typing import List


REQUIRES_RAW: List[str] = [
    "perp_price",
    "spot_price",
]


REQUIRES_FEATURES: List[str] = []


PROVIDES_FEATURES: List[str] = [
    "fut__basis__mtf-none__strike-none__maturity-none",
    "fut__basis_change__mtf-none__strike-none__maturity-none",
    "fut__basis_zscore__mtf-none__strike-none__maturity-none",
    "fut__basis_trend__mtf-none__strike-none__maturity-none",
    "fut__basis_volatility__mtf-none__strike-none__maturity-none",
    "fut__basis_regime_flag__mtf-none__strike-none__maturity-none",
]


FEATURE_FAMILY: str = "futures"
SUBFAMILY: str = "basis_structure"
VERSION: str = "1.0"


def validate_raw_inputs(columns: List[str]) -> None:
    missing = [col for col in REQUIRES_RAW if col not in columns]
    if missing:
        raise ValueError(f"[basis_structure] Missing raw inputs: {missing}")


def validate_feature_outputs(features: List[str]) -> None:
    missing = [f for f in PROVIDES_FEATURES if f not in features]
    if missing:
        raise ValueError(f"[basis_structure] Missing features: {missing}")
