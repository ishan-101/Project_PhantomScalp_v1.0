"""Base Open Interest state engine."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


class OpenInterestEngineError(Exception):
    """Raised when OI base-state inputs are invalid."""


def _build_trades_events(trades_df: pd.DataFrame) -> pd.DataFrame:
    req = ["meta__timestamp", "meta__sequence_id", "price", "size", "aggressor"]
    missing = [c for c in req if c not in trades_df.columns]
    if missing:
        raise OpenInterestEngineError(f"trades_df missing columns: {missing}")

    events = trades_df[req].copy()
    size = pd.to_numeric(events["size"], errors="coerce").fillna(0.0)
    aggressor = pd.to_numeric(events["aggressor"], errors="coerce").fillna(0.0)

    opening_flow = np.where(aggressor > 0, size, 0.0)
    closing_flow = np.where(aggressor < 0, size, 0.0)

    events["opening_flow"] = opening_flow.astype("float32")
    events["closing_flow"] = closing_flow.astype("float32")
    events["price"] = pd.to_numeric(events["price"], errors="coerce").fillna(0.0).astype("float32")
    return events


def _build_orderflow_events(orderflow_df: pd.DataFrame) -> pd.DataFrame:
    req = ["meta__timestamp", "meta__sequence_id", "event_type", "price", "size", "aggressor"]
    missing = [c for c in req if c not in orderflow_df.columns]
    if missing:
        raise OpenInterestEngineError(f"orderflow_df missing columns: {missing}")

    events = orderflow_df[req].copy()
    etype = events["event_type"].astype(str).str.lower()
    size = pd.to_numeric(events["size"], errors="coerce").fillna(0.0)
    aggressor = pd.to_numeric(events["aggressor"], errors="coerce").fillna(0.0)

    opening_flow = np.where(etype.eq("add"), size, 0.0)
    opening_flow = np.where(etype.eq("trade") & (aggressor > 0), size * 0.5, opening_flow)

    closing_flow = np.where(etype.eq("cancel"), size, 0.0)
    closing_flow = np.where(etype.eq("trade") & (aggressor < 0), size * 0.5, closing_flow)

    events["opening_flow"] = opening_flow.astype("float32")
    events["closing_flow"] = closing_flow.astype("float32")
    events["price"] = pd.to_numeric(events["price"], errors="coerce").fillna(0.0).astype("float32")
    return events


def build_open_interest_state(trades_df: pd.DataFrame, orderflow_df: pd.DataFrame) -> pd.DataFrame:
    """Build canonical event timeline and fut__open_interest state."""

    tr_events = _build_trades_events(trades_df)
    of_events = _build_orderflow_events(orderflow_df)

    timeline = pd.concat([tr_events, of_events], axis=0, ignore_index=True)
    timeline = timeline.sort_values(
        by=["meta__timestamp", "meta__sequence_id"], kind="mergesort"
    ).reset_index(drop=True)

    net = (
        pd.to_numeric(timeline["opening_flow"], errors="coerce").fillna(0.0)
        - pd.to_numeric(timeline["closing_flow"], errors="coerce").fillna(0.0)
    ).to_numpy(dtype="float64")

    oi = np.empty(len(timeline), dtype="float64")
    running = 0.0
    for i, delta in enumerate(net):
        running = max(0.0, running + float(delta))
        oi[i] = running

    out = timeline[["meta__timestamp", "meta__sequence_id"]].copy()
    out["open_flow"] = pd.to_numeric(timeline["opening_flow"], errors="coerce").fillna(0.0).astype("float32")
    out["close_flow"] = pd.to_numeric(timeline["closing_flow"], errors="coerce").fillna(0.0).astype("float32")
    out["event_price"] = pd.to_numeric(timeline["price"], errors="coerce").fillna(0.0).astype("float32")
    out["fut__open_interest"] = oi.astype("float32")

    if out.isna().any().any():
        raise OpenInterestEngineError("fut__open_interest state contains null values")

    return out
