"""Design-time semantic validation for Greeks & Greek-Flow base features.

This script ensures that each required feature can be computed using only the
current and previous option chain snapshots along with existing base features.
It must be executed before implementing downstream computation modules.
"""
from __future__ import annotations

from typing import Dict, List, Set


class SemanticValidationError(RuntimeError):
    """Raised when a semantic rule is violated for Greeks features."""


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
        if "rolling" in req:
            raise SemanticValidationError(f"Rolling dependency detected for {name}: {req}")
        if req.startswith("previous:"):
            if req not in available_previous:
                raise SemanticValidationError(
                    f"Feature {name} requires previous snapshot input '{req.split(':', 1)[1]}' which is unavailable."
                )
            continue

        if req not in available_current and req not in computed:
            raise SemanticValidationError(
                f"Feature {name} requires '{req}' not available in current snapshot or prior features."
            )

    if feature.get("allows_future", False):
        raise SemanticValidationError(f"Feature {name} improperly allows future data access.")



def main() -> None:
    # Base required inputs explicitly mandated by specification.
    required_base_inputs = {
        "spot",
        "option_price",
        "implied_volatility",
        "option_type",
        "open_interest",
    }

    # Additional deterministic inputs typically present in the option chain rows.
    auxiliary_inputs = {
        "strike",
        "time_to_expiry",
        "volume",
        "bid_iv",
        "ask_iv",
    }

    # Outputs from Options Chain base features that are permissible dependencies.
    options_chain_features = {
        "opt__spot",
        "opt__nearest_oi_call",
        "opt__nearest_oi_put",
        "opt__total_oi_calls",
        "opt__total_oi_puts",
        "opt__call_put_oi_ratio",
        "opt__best_bid_iv",
        "opt__best_ask_iv",
        "opt__oi_change",
        "opt__volume_calls",
        "opt__volume_puts",
        "opt__iv_crush_detector",
        "opt__trade_size_by_moneyness_proxy",
        "opt__implied_vol_slope",
        "opt__option_flow_imbalance",
    }

    available_current = required_base_inputs | auxiliary_inputs | options_chain_features
    _assert_required_inputs_present(required_base_inputs, available_current)

    previous_snapshot_available = True
    if not previous_snapshot_available:
        raise SemanticValidationError("Previous snapshot access is required but unavailable.")

    available_previous = {f"previous:{col}" for col in available_current | options_chain_features}

    features: List[Dict[str, object]] = [
        {
            "name": "greek__delta_net",
            "required_inputs": {"spot", "option_price", "implied_volatility", "option_type", "open_interest"},
        },
        {
            "name": "greek__gamma_net",
            "required_inputs": {"spot", "option_price", "implied_volatility", "option_type", "open_interest"},
        },
        {
            "name": "greek__vega_net",
            "required_inputs": {"spot", "option_price", "implied_volatility", "option_type", "open_interest"},
        },
        {
            "name": "greek__theta_net",
            "required_inputs": {"spot", "option_price", "implied_volatility", "time_to_expiry", "option_type", "open_interest"},
        },
        {
            "name": "greek__delta_flow",
            "required_inputs": {"greek__delta_net", "previous:greek__delta_net"},
        },
        {
            "name": "greek__gamma_flow",
            "required_inputs": {"greek__gamma_net", "previous:greek__gamma_net"},
        },
        {
            "name": "greek__implied_vol_surface_flag",
            "required_inputs": {"implied_volatility", "previous:implied_volatility"},
        },
        {
            "name": "greek__vanna_net",
            "required_inputs": {"spot", "implied_volatility", "option_type", "open_interest", "strike"},
        },
        {
            "name": "greek__charm_net",
            "required_inputs": {"spot", "implied_volatility", "time_to_expiry", "option_type", "open_interest"},
        },
        {
            "name": "greek__skew_slope",
            "required_inputs": {"implied_volatility", "strike", "spot"},
        },
        {
            "name": "greek__vol_of_vol_proxy",
            "required_inputs": {"implied_volatility", "strike"},
        },
        {
            "name": "greek__gamma_shock_indicator",
            "required_inputs": {"greek__gamma_flow", "previous:greek__gamma_flow"},
        },
        {
            "name": "greek__sticky_delta_indicator",
            "required_inputs": {"implied_volatility", "spot", "previous:implied_volatility", "previous:spot"},
        },
        {
            "name": "greek__delta_hedge_pressure",
            "required_inputs": {"greek__delta_net", "spot", "open_interest"},
        },
        {
            "name": "greek__gamma_per_notional",
            "required_inputs": {"greek__gamma_net", "option_price", "open_interest"},
        },
    ]

    available_previous |= {f"previous:{feature['name']}" for feature in features}

    computed_features: Set[str] = set()
    for feature in features:
        _check_feature_dependencies(feature, available_current, available_previous, computed_features)
        computed_features.add(feature["name"])

    print("Greeks semantic validation — PASSED")


if __name__ == "__main__":
    main()
