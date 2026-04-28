"""Semantic and causality gate for higher-order engineered features.

This script must pass before any higher-order feature artifacts are generated.
It asserts that every higher-order feature can be computed solely from
already-frozen base features (plus explicit, immutable constants), without
future leakage, labeling logic, or implicit model fitting.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_FEATURE_DIR = REPO_ROOT / "feature_engineering" / "base_features"


class SemanticValidationError(RuntimeError):
    """Raised when semantic or dependency checks fail."""


@dataclass(frozen=True)
class FeatureSpec:
    """Contract for a single higher-order engineered feature."""

    name: str
    base_dependencies: Sequence[str]
    constant_dependencies: Sequence[str] = field(default_factory=tuple)
    allows_future_data: bool = False
    allows_label_logic: bool = False
    allows_model_fitting: bool = False

    def validate(self, available_features: set[str], allowed_constants: set[str]) -> None:
        """Validate dependency availability and rule adherence."""

        missing = [dep for dep in self.base_dependencies if dep not in available_features]
        if missing:
            raise SemanticValidationError(f"{self.name}: missing base dependencies {missing}")

        unknown_constants = [c for c in self.constant_dependencies if c not in allowed_constants]
        if unknown_constants:
            raise SemanticValidationError(f"{self.name}: unknown constants {unknown_constants}")

        if self.allows_future_data:
            raise SemanticValidationError(f"{self.name}: future data access is not permitted.")
        if self.allows_label_logic:
            raise SemanticValidationError(f"{self.name}: label/target leakage is not permitted.")
        if self.allows_model_fitting:
            raise SemanticValidationError(f"{self.name}: model fitting is not permitted.")


def _load_base_features(schema_path: Path) -> set[str]:
    """Load feature names from a base feature schema file."""

    if not schema_path.exists():
        raise SemanticValidationError(f"Missing base feature schema: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as f:
        content = json.load(f)
    feature_names = {item["name"] for item in content.get("features", [])}
    if not feature_names:
        raise SemanticValidationError(f"No features declared in schema: {schema_path}")
    return feature_names


def _assert_required_families_exist(families: Mapping[str, Path]) -> None:
    """Ensure all required base feature families are present on disk."""

    missing = [name for name, path in families.items() if not path.exists()]
    if missing:
        raise SemanticValidationError(f"Missing required base feature families: {missing}")


def _available_base_features() -> set[str]:
    """Union of all base feature names across required families."""

    schema_map = {
        "price_ohlcv": BASE_FEATURE_DIR / "price_ohlcv" / "schema.json",
        "microstructure_l2_l3": BASE_FEATURE_DIR / "microstructure_l2_l3" / "schema.json",
        "orderflow_tick": BASE_FEATURE_DIR / "orderflow_tick" / "schema.json",
        "options_chain": BASE_FEATURE_DIR / "options_chain" / "schema.json",
        "greeks_greekflow": BASE_FEATURE_DIR / "greeks_greekflow" / "schema.json",
        "market_regime": BASE_FEATURE_DIR / "market_regime" / "schema.json",
    }

    _assert_required_families_exist(schema_map)

    feature_sets = [_load_base_features(path) for path in schema_map.values()]
    available = set().union(*feature_sets)
    expected_families = set(schema_map.keys())
    if len(feature_sets) != len(expected_families):
        raise SemanticValidationError("Unexpected discrepancy in loaded base feature families.")
    return available


def _feature_specs() -> list[FeatureSpec]:
    """Authoritative list of higher-order engineered feature contracts."""

    # Immutable constants permitted for z-scores and filters.
    allowed_constants = {
        "baseline__price_mean",
        "baseline__price_std",
        "baseline__volume_mean",
        "baseline__volume_std",
        "baseline__ob_imbalance_mean",
        "baseline__ob_imbalance_std",
        "pca__book_loadings",
        "filter__ehlers_coefficients",
        "ema__signed_volume_alpha",
    }

    specs = [
        FeatureSpec(
            name="ho__z_price",
            base_dependencies=("price__last",),
            constant_dependencies=("baseline__price_mean", "baseline__price_std"),
        ),
        FeatureSpec(
            name="ho__log_return",
            base_dependencies=("tick_return",),
        ),
        FeatureSpec(
            name="ho__price_accel",
            base_dependencies=("tick_return",),
        ),
        FeatureSpec(
            name="ho__volume_z",
            base_dependencies=("volume__tick",),
            constant_dependencies=("baseline__volume_mean", "baseline__volume_std"),
        ),
        FeatureSpec(
            name="ho__ob_imbalance_z",
            base_dependencies=("ob__imbalance",),
            constant_dependencies=("baseline__ob_imbalance_mean", "baseline__ob_imbalance_std"),
        ),
        FeatureSpec(
            name="ho__pca1_book",
            base_dependencies=("ob__top_level_size_bid", "ob__top_level_size_ask", "ob__total_depth_bid", "ob__total_depth_ask"),
            constant_dependencies=("pca__book_loadings",),
        ),
        FeatureSpec(
            name="ho__interaction_of_volume_spread",
            base_dependencies=("volume__tick", "spread__l1"),
        ),
        FeatureSpec(
            name="ho__return_signed_volume_interaction",
            base_dependencies=("tick_return", "of__signed_volume"),
        ),
        FeatureSpec(
            name="ho__return_over_spread",
            base_dependencies=("tick_return", "spread__l1"),
        ),
        FeatureSpec(
            name="ho__imbalance_times_return",
            base_dependencies=("ob__imbalance", "tick_return"),
        ),
        FeatureSpec(
            name="ho__filtered_mid_price",
            base_dependencies=("price__mid",),
            constant_dependencies=("filter__ehlers_coefficients",),
        ),
        FeatureSpec(
            name="ho__book_flow_interaction",
            base_dependencies=("ob__imbalance", "of__signed_volume"),
        ),
        FeatureSpec(
            name="ho__normalized_impact",
            base_dependencies=("of__price_impact_per_unit_volume", "regime__liquidity_score"),
        ),
        FeatureSpec(
            name="ho__signed_volume_ema",
            base_dependencies=("of__signed_volume",),
            constant_dependencies=("ema__signed_volume_alpha",),
        ),
    ]

    if len(specs) != 14:
        raise SemanticValidationError("Feature spec list must contain exactly 14 entries.")

    seen = set()
    for spec in specs:
        if spec.name in seen:
            raise SemanticValidationError(f"Duplicate feature spec detected: {spec.name}")
        seen.add(spec.name)

    # Validate constants referenced across specs.
    for spec in specs:
        extra_constants = [c for c in spec.constant_dependencies if c not in allowed_constants]
        if extra_constants:
            raise SemanticValidationError(f"{spec.name}: constants not whitelisted {extra_constants}")
    return specs


def _run_semantic_checks() -> None:
    """Run all semantic validations and print pass message on success."""

    available_features = _available_base_features()
    specs = _feature_specs()
    allowed_constants = {
        "baseline__price_mean",
        "baseline__price_std",
        "baseline__volume_mean",
        "baseline__volume_std",
        "baseline__ob_imbalance_mean",
        "baseline__ob_imbalance_std",
        "pca__book_loadings",
        "filter__ehlers_coefficients",
        "ema__signed_volume_alpha",
    }

    missing_families: list[str] = []
    required_sources = {
        "Price / OHLCV": {"price__last", "price__mid", "volume__tick", "spread__l1"},
        "Microstructure": {"ob__imbalance", "ob__top_level_size_bid", "ob__top_level_size_ask"},
        "Orderflow": {"of__signed_volume", "of__price_impact_per_unit_volume"},
        "Options Chain": {"opt__best_bid_iv", "opt__best_ask_iv"},
        "Greeks": {"greek__gamma_per_notional", "greek__delta_net"},
        "Market Regime": {"regime__liquidity_score"},
    }
    for source, deps in required_sources.items():
        if not deps.issubset(available_features):
            missing_families.append(source)
    if missing_families:
        raise SemanticValidationError(f"Missing required dependency families: {missing_families}")

    for spec in specs:
        spec.validate(available_features, allowed_constants)


def main(argv: Iterable[str] | None = None) -> int:
    try:
        _run_semantic_checks()
    except SemanticValidationError as exc:
        print(f"Higher-Order semantic validation — FAILED: {exc}")
        return 1

    print("Higher-Order semantic validation — PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
