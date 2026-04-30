"""Volume delta feature engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


class VolumeDeltaEngineError(Exception):
    pass


def _normalize_aggressor(aggressor: pd.Series) -> pd.Series:
    side = aggressor.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1,
        "buy": 1,
        "b": 1,
        "buyer": 1,
        "buy_aggressor": 1,
        "-1": -1,
        "sell": -1,
        "s": -1,
        "seller": -1,
        "sell_aggressor": -1,
    }
    mapped = side.map(mapping)
    if mapped.isna().any():
        bad = sorted(set(side[mapped.isna()].tolist()))
        raise VolumeDeltaEngineError(f"Unsupported aggressor_side values: {bad}")
    return mapped.astype("int8")


def add_volume_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative aggressive buy-minus-sell volume delta."""
    needed = ["trade_size", "aggressor_side"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise VolumeDeltaEngineError(f"Missing columns: {missing}")

    out = df.copy(deep=True)
    size = pd.to_numeric(out["trade_size"], errors="coerce")
    if size.isna().any() or (size <= 0).any():
        raise VolumeDeltaEngineError("trade_size contains invalid values for aggressor delta computation")

    side = _normalize_aggressor(out["aggressor_side"])

    # Aggressor-validated signed flow; excludes unknown sides by hard fail above.
    signed = np.where(side.to_numpy() > 0, size.to_numpy(), -size.to_numpy())
    out["fut__volume_delta"] = pd.Series(signed, index=out.index).cumsum().astype("float32")
    return out
