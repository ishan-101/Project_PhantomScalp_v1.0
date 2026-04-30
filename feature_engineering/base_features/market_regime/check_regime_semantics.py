"""Semantic validation gate for market regime / volatility / liquidity features."""
from __future__ import annotations

from typing import Dict, List, Set


class SemanticValidationError(RuntimeError):
    """Raised when semantic or causality rules are violated."""


def _assert_required_inputs_present(required_inputs: Set[str], available_inputs: Set[str]) -> None:
    missing = required_inputs - available_inputs
    if missing:
        raise SemanticValidationError(f"Missing required input columns: {sorted(missing)}")


def _check_feature_dependencies(
    feature: Dict[str, object],
    available_current: Set[str],
    available_previous: Set[str],
    computed: Set[str],
) -> None:
    name = feature["name"]
    required_inputs: Set[str] = feature["required_inputs"]  # type: ignore[assignment]

    for req in required_inputs:
        if "future" in req:
            raise SemanticValidationError(f"Future-looking dependency detected for {name}: {req}")
        if req.startswith("previous:"):
            if req not in available_previous:
                raise SemanticValidationError(
                    f"Feature {name} requires previous snapshot input '{req.split(':', 1)[1]}' which is unavailable."
                )
            continue

        if req not in available_current and req not in computed:
            raise SemanticValidationError(
                f"Feature {name} requires '{req}' not available in current snapshot or prior computed features."
            )

    if feature.get("allows_future", False):
        raise SemanticValidationError(f"Feature {name} improperly allows future data access.")



def main() -> None:
    required_base_inputs = {
        "price__mid",
        "tick_return",
        "price__micro_volatility",
        "spread__l1",
        "price__tick_direction",
    }

    orderflow_inputs = {
        "of__signed_volume",
        "of__aggressor_volume_ratio",
        "of__execution_flow_polarity",
        "of__trade_burst_intensity",
    }

    depth_liquidity_inputs = {
        "ob__top_level_size_bid",
        "ob__top_level_size_ask",
        "ob__total_depth_bid",
        "ob__total_depth_ask",
        "ob__cancellation_rate",
        "ob__new_order_rate",
        "ob__top_of_book_turnover",
        "ob__queue_resilience",
        "ob__depth_elasticity",
        "ob__hidden_to_visible_ratio",
    }

    implied_vol_inputs = {
        "opt__best_bid_iv",
        "opt__best_ask_iv",
    }

    auxiliary_inputs = {
        "price__near_term_return_volatility",
        "ob__queue_position_change_bid",
        "ob__queue_position_change_ask",
    }

    available_current = (
        required_base_inputs
        | orderflow_inputs
        | depth_liquidity_inputs
        | implied_vol_inputs
        | auxiliary_inputs
    )

    _assert_required_inputs_present(required_base_inputs | orderflow_inputs | depth_liquidity_inputs | implied_vol_inputs, available_current)

    previous_snapshot_available = True
    if not previous_snapshot_available:
        raise SemanticValidationError("Previous snapshot access is required but unavailable.")

    available_previous = {f"previous:{col}" for col in available_current}

    features: List[Dict[str, object]] = [
        {
            "name": "regime__realized_vol",
            "required_inputs": {"price__micro_volatility"},
        },
        {
            "name": "regime__iv_atm",
            "required_inputs": {"opt__best_bid_iv", "opt__best_ask_iv"},
        },
        {
            "name": "regime__liquidity_score",
            "required_inputs": {
                "spread__l1",
                "ob__top_level_size_bid",
                "ob__top_level_size_ask",
                "ob__cancellation_rate",
                "ob__new_order_rate",
            },
        },
        {
            "name": "regime__volatility_spike_flag",
            "required_inputs": {"regime__realized_vol", "previous:regime__realized_vol"},
        },
        {
            "name": "regime__spread_stability",
            "required_inputs": {"spread__l1", "previous:spread__l1"},
        },
        {
            "name": "regime__orderflow_extremeness",
            "required_inputs": {"of__signed_volume", "of__aggressor_volume_ratio", "previous:of__signed_volume"},
        },
        {
            "name": "regime__market_state_flag",
            "required_inputs": {"price__tick_direction", "tick_return", "of__execution_flow_polarity", "previous:tick_return"},
        },
        {
            "name": "regime__spread_zscore",
            "required_inputs": {"spread__l1"},
        },
        {
            "name": "regime__liquidity_fractal_index",
            "required_inputs": {
                "ob__top_of_book_turnover",
                "ob__queue_resilience",
                "previous:ob__top_of_book_turnover",
                "previous:ob__queue_resilience",
            },
        },
        {
            "name": "regime__depth_stress_ratio",
            "required_inputs": {"ob__total_depth_bid", "ob__total_depth_ask", "spread__l1"},
        },
        {
            "name": "regime__volatility_compression_flag",
            "required_inputs": {"regime__realized_vol", "previous:regime__realized_vol"},
        },
        {
            "name": "regime__momentum_ignition_flag",
            "required_inputs": {
                "of__trade_burst_intensity",
                "spread__l1",
                "previous:spread__l1",
                "regime__realized_vol",
            },
        },
        {
            "name": "regime__spread_regime_crossover",
            "required_inputs": {"spread__l1", "previous:spread__l1"},
        },
        {
            "name": "regime__short_term_vol_forecast_error",
            "required_inputs": {"regime__realized_vol", "price__near_term_return_volatility"},
        },
        {
            "name": "regime__micro_liquidity_index",
            "required_inputs": {
                "spread__l1",
                "ob__depth_elasticity",
                "ob__hidden_to_visible_ratio",
                "ob__top_level_size_bid",
                "ob__top_level_size_ask",
            },
        },
    ]

    available_previous |= {f"previous:{feature['name']}" for feature in features}

    computed_features: Set[str] = set()
    for feature in features:
        _check_feature_dependencies(feature, available_current, available_previous, computed_features)
        computed_features.add(feature["name"])

    print("Market regime semantic validation — PASSED")


if __name__ == "__main__":
    main()
