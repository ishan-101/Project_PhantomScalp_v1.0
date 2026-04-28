# app/schemas.py
# Project_PhantomScalp v1.0 (v0.2)
# Centralized, stable I/O contracts for data, features, labels, signals, trades, KPIs, and reports.

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Literal, Any

from pydantic import BaseModel, Field, validator


# -----------------------------
# Enums & literals
# -----------------------------

class Venue(str, Enum):
    DELTA = "DELTA"
    BINANCE = "BINANCE"


class OptionType(str, Enum):
    CALL = "CALL"
    PUT = "PUT"


class Action(str, Enum):
    FLAT = "FLAT"
    CALL_LONG = "CALL_LONG"
    PUT_LONG = "PUT_LONG"
    CALL_SHORT = "CALL_SHORT"
    PUT_SHORT = "PUT_SHORT"


SideLiteral = Literal["CALL_LONG", "PUT_LONG", "CALL_SHORT", "PUT_SHORT"]
TimeframeLiteral = Literal["15s", "1m", "3m", "5m", "15m", "30m", "1h"]


# -----------------------------
# Helpers
# -----------------------------

def _ensure_ts_iso(ts: Any) -> str:
    """
    Accepts datetime | str. Returns ISO8601 string (no tz conversion).
    """
    if isinstance(ts, datetime):
        return ts.isoformat()
    if isinstance(ts, str):
        # quick sanity (not a full parser to keep pydantic free of extras)
        # allow both "YYYY-MM-DD HH:MM:SS" and ISO
        s = ts.strip().replace(" ", "T") if " " in ts else ts.strip()
        return s
    raise ValueError(f"Unsupported timestamp type: {type(ts)}")


# -----------------------------
# Market data
# -----------------------------

class OHLCV(BaseModel):
    ts: str = Field(..., description="Timestamp (ISO8601)")
    symbol: str
    frame: TimeframeLiteral
    open: float
    high: float
    low: float
    close: float
    volume: float

    _ts_iso = validator("ts", allow_reuse=True)(_ensure_ts_iso)


class OptionQuote(BaseModel):
    ts: str = Field(..., description="Timestamp (ISO8601)")
    symbol: str = Field(..., description="Underlying symbol, e.g., BTCUSD")
    venue: Venue = Field(..., description="Exchange/venue name")
    expiry: str = Field(..., description="ISO date string of option expiry, e.g., 2025-09-27")
    strike: float
    option_type: OptionType
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    iv: Optional[float] = Field(None, description="Implied vol (decimal, e.g., 0.65 for 65%)")
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None
    oi: Optional[float] = Field(None, description="Open interest (contracts)")
    volume: Optional[float] = Field(None, description="Volume (contracts)")

    _ts_iso = validator("ts", allow_reuse=True)(_ensure_ts_iso)


# -----------------------------
# Feature / label / signal rows
# -----------------------------

class FeatureRow(BaseModel):
    """
    Flexible feature carrier.
    Keep cols flat: key -> float (or small int). Namespaced keys are fine (e.g., micro.imbalance).
    """
    ts: str
    values: Dict[str, float] = Field(default_factory=dict)

    _ts_iso = validator("ts", allow_reuse=True)(_ensure_ts_iso)


class RegimeLabel(BaseModel):
    """
    Example scheme:
    0 = chop, 1 = uptrend, 2 = downtrend
    """
    ts: str
    label: int = Field(..., ge=0, description="Discrete regime class id")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    _ts_iso = validator("ts", allow_reuse=True)(_ensure_ts_iso)


class ReversalLabel(BaseModel):
    """
    Example scheme:
    0 = none, 1 = local-top, 2 = local-bottom
    """
    ts: str
    label: int = Field(..., ge=0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    _ts_iso = validator("ts", allow_reuse=True)(_ensure_ts_iso)


class CycleLabel(BaseModel):
    """
    Example scheme:
    0 = early, 1 = mid, 2 = late (or any discrete phase id)
    """
    ts: str
    label: int = Field(..., ge=0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    _ts_iso = validator("ts", allow_reuse=True)(_ensure_ts_iso)


class SignalRow(BaseModel):
    ts: str
    regime_sig: int
    reversal_sig: int
    cycle_sig: int
    action: Action = Action.FLAT
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    _ts_iso = validator("ts", allow_reuse=True)(_ensure_ts_iso)


# -----------------------------
# Orders, fills, trades
# -----------------------------

class OrderRow(BaseModel):
    ts: str
    order_id: str
    side: SideLiteral
    qty: float
    limit_price: Optional[float] = None
    meta: Optional[dict] = None

    _ts_iso = validator("ts", allow_reuse=True)(_ensure_ts_iso)


class FillRow(BaseModel):
    ts: str
    order_id: str
    side: SideLiteral
    qty: float
    price: float
    fee: float = 0.0
    slippage: float = 0.0
    meta: Optional[dict] = None

    _ts_iso = validator("ts", allow_reuse=True)(_ensure_ts_iso)


class TradeRow(BaseModel):
    time_in: str
    time_out: str
    side: Literal[
        "CALL_LONG", "PUT_LONG", "CALL_SHORT", "PUT_SHORT",
        "BUY_CALL", "BUY_PUT", "SELL_CALL", "SELL_PUT"  # keep wide for router variants
    ]
    qty: float
    entry: float
    exit: float
    pnl: float
    ret: float
    win: bool
    meta: Optional[dict] = None

    _tin_iso = validator("time_in", allow_reuse=True)(_ensure_ts_iso)
    _tout_iso = validator("time_out", allow_reuse=True)(_ensure_ts_iso)


# -----------------------------
# KPIs / Reporting
# -----------------------------

class BacktestKPI(BaseModel):
    total_return: float
    cagr: Optional[float] = None
    max_dd: float
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    hit_rate: float
    avg_rr: Optional[float] = None
    n_trades: Optional[int] = None
    exposure_mean: Optional[float] = None


class ReportPaths(BaseModel):
    summary_json: Optional[str] = None
    report_html: Optional[str] = None
    images: Dict[str, str] = Field(default_factory=dict)   # name -> path
    csvs: Dict[str, str] = Field(default_factory=dict)     # name -> path


class ReportSummary(BaseModel):
    config: Dict[str, Any] = Field(default_factory=dict)
    kpis: Dict[str, Any] = Field(default_factory=dict)
    paths: Optional[ReportPaths] = None


# -----------------------------
# Minimal config (optional; use your own if you already have one)
# -----------------------------

class BacktestConfig(BaseModel):
    symbol: str = "BTCUSD"
    start: str
    end: str
    frame: TimeframeLiteral = "1m"
    output_dir: str = "./out/v02"
    models: Dict[str, Any] = Field(default_factory=dict)  # model_name -> path or spec
    policy: Dict[str, Any] = Field(default_factory=dict)  # risk knobs, etc.

    _start_iso = validator("start", allow_reuse=True)(_ensure_ts_iso)
    _end_iso = validator("end", allow_reuse=True)(_ensure_ts_iso)


# -----------------------------
# Convenience containers
# -----------------------------

class TradesTable(BaseModel):
    rows: List[TradeRow]


class SignalsTable(BaseModel):
    rows: List[SignalRow]


class OHLCVTable(BaseModel):
    rows: List[OHLCV]


class OptionQuotesTable(BaseModel):
    rows: List[OptionQuote]
