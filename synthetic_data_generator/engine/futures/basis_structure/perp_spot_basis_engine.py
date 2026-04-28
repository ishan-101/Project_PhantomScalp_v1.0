"""Perp-vs-spot basis state engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PerpSpotBasisEngineError(Exception):
    pass


def _safe_div(num: pd.Series, den: pd.Series, eps: float = 1e-9) -> pd.Series:
    return num / den.abs().clip(lower=eps)


def _build_timeline(trades_df: pd.DataFrame, orderflow_df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    trades_req = ["meta__timestamp", "meta__sequence_id", "price", "size", "aggressor"]
    flow_req = ["meta__timestamp", "meta__sequence_id", "event_type", "size", "aggressor", "inventory_pressure"]
    funding_req = [
        "meta__timestamp",
        "meta__sequence_id",
        "fut__funding_rate",
        "fut__funding_rate_zscore",
        "fut__funding_pressure_index",
        "fut__funding_oi_stress",
    ]

    for req, name, frame in ((trades_req, "trades", trades_df), (flow_req, "orderflow", orderflow_df), (funding_req, "funding", funding_df)):
        missing = [c for c in req if c not in frame.columns]
        if missing:
            raise PerpSpotBasisEngineError(f"{name} missing required columns: {missing}")

    tr = trades_df[trades_req].copy()
    tr["meta__timestamp"] = pd.to_datetime(tr["meta__timestamp"], utc=True)
    tr["meta__sequence_id"] = pd.to_numeric(tr["meta__sequence_id"], errors="coerce").fillna(0).astype("int64")
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce").fillna(0.0)
    tr["size"] = pd.to_numeric(tr["size"], errors="coerce").fillna(0.0)
    tr["aggressor"] = pd.to_numeric(tr["aggressor"], errors="coerce").fillna(0.0)

    of = orderflow_df[flow_req].copy()
    of["meta__timestamp"] = pd.to_datetime(of["meta__timestamp"], utc=True)
    of["meta__sequence_id"] = pd.to_numeric(of["meta__sequence_id"], errors="coerce").fillna(0).astype("int64")
    of["size"] = pd.to_numeric(of["size"], errors="coerce").fillna(0.0)
    of["aggressor"] = pd.to_numeric(of["aggressor"], errors="coerce").fillna(0.0)
    of["inventory_pressure"] = pd.to_numeric(of["inventory_pressure"], errors="coerce").fillna(0.0)

    etype = of["event_type"].astype(str).str.lower()
    of_trade_impulse = np.where(etype.eq("trade"), of["aggressor"] * of["size"], 0.0)
    of_queue_impulse = np.where(etype.eq("add"), of["size"], np.where(etype.eq("cancel"), -of["size"], 0.0))
    of["flow_impulse"] = of_trade_impulse + 0.30 * of_queue_impulse + 0.15 * of["inventory_pressure"]

    timeline = tr[["meta__timestamp", "meta__sequence_id", "price", "size", "aggressor"]].copy()
    timeline = timeline.sort_values(["meta__timestamp", "meta__sequence_id"], kind="mergesort").reset_index(drop=True)

    agg_flow = of.groupby("meta__timestamp", sort=True)["flow_impulse"].mean().rename("flow_impulse")
    timeline = timeline.merge(agg_flow, on="meta__timestamp", how="left")

    funding = funding_df[funding_req].copy()
    funding["meta__timestamp"] = pd.to_datetime(funding["meta__timestamp"], utc=True)
    funding = funding.sort_values(["meta__timestamp", "meta__sequence_id"], kind="mergesort").reset_index(drop=True)

    timeline = pd.merge_asof(
        timeline.sort_values("meta__timestamp"),
        funding[[
            "meta__timestamp",
            "fut__funding_rate",
            "fut__funding_rate_zscore",
            "fut__funding_pressure_index",
            "fut__funding_oi_stress",
        ]].sort_values("meta__timestamp"),
        on="meta__timestamp",
        direction="backward",
        allow_exact_matches=True,
    )

    fill_cols = [
        "flow_impulse",
        "fut__funding_rate",
        "fut__funding_rate_zscore",
        "fut__funding_pressure_index",
        "fut__funding_oi_stress",
    ]
    timeline[fill_cols] = timeline[fill_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)

    return timeline


def add_perp_spot_basis(trades_df: pd.DataFrame, orderflow_df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    out = _build_timeline(trades_df, orderflow_df, funding_df)

    perp_price = out["price"].clip(lower=1e-9)
    size = out["size"].clip(lower=0.0)
    notional = perp_price * size

    # Spot fair proxy: trailing VWAP adjusted by inventory and funding pressure crowding.
    trailing_vwap = _safe_div(
        (perp_price * size).rolling(240, min_periods=5).sum(),
        size.rolling(240, min_periods=5).sum().replace(0.0, np.nan),
    ).bfill().fillna(perp_price)

    flow_bias = np.tanh(out["flow_impulse"].rolling(120, min_periods=3).mean().fillna(0.0))
    funding_bias = np.tanh(
        0.70 * out["fut__funding_pressure_index"] + 0.20 * out["fut__funding_rate_zscore"] + 0.10 * out["fut__funding_oi_stress"]
    )

    # Remove futures-specific crowding component to estimate spot fair value baseline.
    crowding_adjustment = (0.0015 * flow_bias + 0.0018 * funding_bias).clip(-0.0075, 0.0075)
    spot_proxy = trailing_vwap * (1.0 - crowding_adjustment)

    raw_basis = _safe_div(perp_price - spot_proxy, spot_proxy.replace(0.0, np.nan))

    # Liquidity normalization: same premium under low notional should have smaller structural weight.
    liquidity_scale = notional.rolling(180, min_periods=5).median().replace(0.0, np.nan)
    liquidity_weight = np.tanh(_safe_div(notional, liquidity_scale).fillna(0.0))

    aligned_pressure = np.sign(raw_basis.fillna(0.0)) * np.tanh(funding_bias.abs() + flow_bias.abs())
    structural_basis = raw_basis * (0.65 + 0.35 * liquidity_weight) * (1.0 + 0.25 * aligned_pressure)

    out["fut__perp_spot_basis"] = (
        structural_basis.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.25, 0.25).astype("float32")
    )

    # Internal columns used by downstream basis engines.
    out["__basis_funding_pressure_alignment"] = (
        (np.sign(out["fut__perp_spot_basis"]) * funding_bias).replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32")
    )
    out["__basis_flow_pressure"] = flow_bias.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32")

    return out
