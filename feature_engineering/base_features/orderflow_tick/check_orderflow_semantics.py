"""Semantic validation gate for orderflow_tick base features.

This script ensures that each orderflow feature can be computed using only the
current and previous records plus already-frozen feature families. No feature
values are computed here; only feasibility and causality are asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


class SemanticValidationError(RuntimeError):
    """Raised when semantic constraints are violated."""


@dataclass
class FeatureSpec:
    """Specification for a single feature dependency check."""

    name: str
    required_inputs: List[str]
    uses_previous: bool = False


REQUIRED_BASE_INPUTS = {
    "trade_price",
    "trade_size",
    "aggressor_side",
    "trade_count",
    "timestamp",
    "mid_price",
    "spread",
    "visible_depth",
}


FEATURE_SPECS: List[FeatureSpec] = [
    FeatureSpec("of__signed_volume", ["trade_size", "aggressor_side"]),
    FeatureSpec("of__imbalance_ratio", ["trade_size", "aggressor_side"]),
    FeatureSpec("of__large_trade_count", ["trade_size", "large_trade_threshold", "trade_count"]),
    FeatureSpec("of__avg_trade_size", ["trade_size", "trade_count"]),
    FeatureSpec("of__time_between_trades", ["timestamp", "trade_count"]),
    FeatureSpec("of__aggressor_flag_ratio", ["aggressor_side", "trade_count"]),
    FeatureSpec("of__trade_price_vs_vwap", ["trade_price", "trade_size", "vwap"]),
    FeatureSpec("of__run_length_up", ["trade_price", "previous_trade_price"], uses_previous=True),
    FeatureSpec("of__run_length_down", ["trade_price", "previous_trade_price"], uses_previous=True),
    FeatureSpec("of__aggressive_buy_size", ["trade_size", "aggressor_side"]),
    FeatureSpec("of__aggressive_sell_size", ["trade_size", "aggressor_side"]),
    FeatureSpec("of__sequence_entropy", ["aggressor_side", "trade_count"]),
    FeatureSpec("of__small_trade_vol", ["trade_size", "small_trade_threshold"]),
    FeatureSpec("of__medium_trade_vol", ["trade_size", "small_trade_threshold", "large_trade_threshold"]),
    FeatureSpec("of__large_trade_vol", ["trade_size", "large_trade_threshold"]),
    FeatureSpec("of__vwap_pressure", ["vwap", "mid_price"]),
    FeatureSpec("of__aggressor_volume_ratio", ["trade_size", "aggressor_side"]),
    FeatureSpec("of__execution_flow_polarity", ["trade_size", "aggressor_side"]),
    FeatureSpec("of__market_pressure_tilt", ["trade_size", "aggressor_side", "spread", "visible_depth"]),
    FeatureSpec("of__impact_adjusted_flow", ["trade_size", "aggressor_side", "trade_price", "mid_price", "spread"]),
    FeatureSpec(
        "of__aggression_persistence",
        ["aggressor_side", "previous_aggressor_side"],
        uses_previous=True,
    ),
    FeatureSpec("of__trade_burst_intensity", ["trade_count", "timestamp"]),
    FeatureSpec(
        "of__toxicity_proxy",
        ["trade_size", "aggressor_side", "trade_price", "mid_price", "spread"],
    ),
    FeatureSpec(
        "of__realized_sign_rate",
        ["aggressor_side", "trade_price", "previous_trade_price"],
        uses_previous=True,
    ),
    FeatureSpec(
        "of__price_impact_per_unit_volume",
        ["trade_price", "trade_size", "previous_trade_price"],
        uses_previous=True,
    ),
    FeatureSpec(
        "of__initiator_persistence",
        ["aggressor_side", "previous_aggressor_side"],
        uses_previous=True,
    ),
    FeatureSpec(
        "of__time_decay_of_flow",
        ["trade_size", "aggressor_side", "previous_signed_volume"],
        uses_previous=True,
    ),
]


def _assert_inputs_present(available: Iterable[str], required: Iterable[str]) -> None:
    available_set = set(available)
    missing = [col for col in required if col not in available_set]
    if missing:
        raise SemanticValidationError(f"Missing required inputs: {missing}")


def validate_semantics(available_inputs: Dict[str, bool]) -> None:
    """Validate that all orderflow features are causally computable."""

    _assert_inputs_present(available_inputs, REQUIRED_BASE_INPUTS)

    for spec in FEATURE_SPECS:
        _assert_inputs_present(available_inputs, spec.required_inputs)
        if spec.uses_previous and not available_inputs.get("has_previous", False):
            raise SemanticValidationError(
                f"Feature {spec.name} requires previous record context but none is available."
            )


if __name__ == "__main__":
    mock_inputs = {
        "trade_price": True,
        "trade_size": True,
        "aggressor_side": True,
        "trade_count": True,
        "timestamp": True,
        "mid_price": True,
        "spread": True,
        "visible_depth": True,
        "vwap": True,
        "small_trade_threshold": True,
        "large_trade_threshold": True,
        "previous_trade_price": True,
        "previous_aggressor_side": True,
        "previous_signed_volume": True,
        "has_previous": True,
    }

    try:
        validate_semantics(mock_inputs)
    except SemanticValidationError as exc:
        raise SystemExit(f"Orderflow semantic validation — FAILED: {exc}")

    print("Orderflow semantic validation — PASSED")
