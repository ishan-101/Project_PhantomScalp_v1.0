# open_interest/dependencies.py (FINAL CORRECTED)

from __future__ import annotations

from typing import List


REQUIRES_RAW: List[str] = [
    "open_interest",
    "price",
    "volume",
]


REQUIRES_FEATURES: List[str] = []


PROVIDES_FEATURES: List[str] = [
    "fut__open_interest__mtf-none__strike-none__maturity-none",
    "fut__oi_change__mtf-none__strike-none__maturity-none",
    "fut__oi_velocity__mtf-none__strike-none__maturity-none",
    "fut__oi_acceleration__mtf-none__strike-none__maturity-none",
    "fut__oi_zscore__mtf-none__strike-none__maturity-none",
    "fut__oi_price_divergence__mtf-none__strike-none__maturity-none",
    "fut__oi_price_divergence_strength__mtf-none__strike-none__maturity-none",
    "fut__oi_turnover__mtf-none__strike-none__maturity-none",
    "fut__oi_open_close_ratio__mtf-none__strike-none__maturity-none",
]


FEATURE_FAMILY: str = "futures"
SUBFAMILY: str = "open_interest"
VERSION: str = "1.0"


def validate_raw_inputs(columns: List[str]) -> None:
    missing = [col for col in REQUIRES_RAW if col not in columns]
    if missing:
        raise ValueError(f"[open_interest] Missing raw inputs: {missing}")


def validate_feature_outputs(features: List[str]) -> None:
    missing = [f for f in PROVIDES_FEATURES if f not in features]
    if missing:
        raise ValueError(f"[open_interest] Missing features: {missing}")
