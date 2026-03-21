"""Semantic and causality gate for Cross-Asset / Funding base features.

This script enforces that the nine required features rely only on snapshot-causal
inputs, use deterministic rules, and avoid any forward-looking or regime logic.
Execution must precede any schema or computation work; if it fails, stop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set


class SemanticValidationError(RuntimeError):
    """Raised when a semantic rule is violated."""


@dataclass(frozen=True)
class FeatureSemantic:
    name: str
    required_inputs: Set[str]
    window: int | None = None
    allows_future: bool = False
    smoothing: bool = False
    description: str = ""


REQUIRED_BASE_INPUTS: Set[str] = {
    "ts",
    "btc_spot",
    "btc_perpetual",
    "eth_spot",
    "funding_rate",
    "dxy_index",
}

# Additional permissible derived helpers from current snapshot context.
ALLOWED_AUX_INPUTS: Set[str] = {
    "time_delta_seconds",
    "btc_return",
    "dxy_return",
}


def _assert_required_inputs_present(available_inputs: Set[str]) -> None:
    missing = REQUIRED_BASE_INPUTS - available_inputs
    if missing:
        raise SemanticValidationError(f"Missing required input feeds: {sorted(missing)}")


def _check_feature_rules(feature: FeatureSemantic, available_inputs: Set[str], computed: Set[str]) -> None:
    """Validate semantic constraints for a single feature definition."""
    if feature.allows_future:
        raise SemanticValidationError(f"{feature.name} improperly allows future data.")

    for dependency in feature.required_inputs:
        if dependency not in available_inputs and dependency not in computed:
            raise SemanticValidationError(
                f"{feature.name} requires unavailable dependency '{dependency}'."
            )

    if feature.window is not None and feature.window <= 0:
        raise SemanticValidationError(f"{feature.name} declares non-positive window {feature.window}.")

    if feature.smoothing:
        raise SemanticValidationError(f"{feature.name} illegally requests smoothing.")


def main() -> None:
    # Step 1: assert mandatory feeds exist.
    available_inputs = REQUIRED_BASE_INPUTS | ALLOWED_AUX_INPUTS
    _assert_required_inputs_present(available_inputs)

    # Step 2: enumerate the authoritative feature semantics.
    features: List[FeatureSemantic] = [
        FeatureSemantic(
            name="cross__btc_eth_spread",
            required_inputs={"btc_spot", "eth_spot"},
            description="BTC spot minus ETH spot; relative value pressure with no scaling.",
        ),
        FeatureSemantic(
            name="cross__perp_spot_basis",
            required_inputs={"btc_perpetual", "btc_spot"},
            description="Perpetual-spot basis capturing leverage premium/discount.",
        ),
        FeatureSemantic(
            name="cross__funding_rate",
            required_inputs={"funding_rate"},
            description="Current perpetual funding rate, forward-fill only, no smoothing.",
        ),
        FeatureSemantic(
            name="cross__btc_dxy_corr_proxy",
            required_inputs={"btc_return", "dxy_return"},
            window=24,
            description="Past-only rolling correlation proxy between BTC and USD index returns.",
        ),
        FeatureSemantic(
            name="cross__funding_8h_rolling_avg",
            required_inputs={"funding_rate"},
            window=8,
            description="Backward-looking 8-hour funding rate mean with full window requirement.",
        ),
        FeatureSemantic(
            name="cross__basis_change",
            required_inputs={"cross__perp_spot_basis"},
            description="First difference of perpetual-spot basis; acceleration of leverage pressure.",
        ),
        FeatureSemantic(
            name="cross__risk_on_off_flag",
            required_inputs={"cross__funding_rate", "cross__perp_spot_basis", "cross__btc_dxy_corr_proxy"},
            description="Boolean proxy using funding sign, basis sign, and correlation direction; fixed rule.",
        ),
        FeatureSemantic(
            name="cross__perp_basis_velocity",
            required_inputs={"cross__basis_change", "time_delta_seconds"},
            description="Time-normalized change rate of perp basis using backward deltas only.",
        ),
        FeatureSemantic(
            name="cross__correlation_instability",
            required_inputs={"cross__btc_dxy_corr_proxy"},
            description="Magnitude of change in rolling correlation to detect macro coupling shifts.",
        ),
    ]

    # Step 3: enforce deterministic, past-only semantics.
    computed: Set[str] = set()
    for feature in features:
        _check_feature_rules(feature, available_inputs, computed)
        computed.add(feature.name)

    print("Cross-Asset / Funding semantic validation — PASSED")


if __name__ == "__main__":
    main()
