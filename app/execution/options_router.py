# app/execution/options_router.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List

from app.dataio.schemas import MarketState, OptionQuote, OptionSelection, Side


@dataclass
class RouterConfig:
    expiries: int = 3
    strike_span: int = 6            # ±6 ATM
    target_delta_band: Tuple[float, float] = (0.25, 0.35)
    min_oi: int = 50
    min_vol: int = 10
    max_spread_pct: float = 3.0
    iv_percentile_bounds: Tuple[int, int] = (5, 95)
    limit_price_slippage_pct: float = 0.005  # 0.5% around mid


def _spread_pct(bid: float, ask: float) -> float:
    if ask <= 0.0:
        return 999.0
    return max(0.0, 100.0 * (ask - bid) / max(1e-9, ask))


def _choose_leg(cands: List[OptionQuote], side: Side, cfg: RouterConfig) -> Optional[OptionQuote]:
    # Filter by liquidity and spread
    filtered = []
    for q in cands:
        if q.oi < cfg.min_oi or q.vol < cfg.min_vol:
            continue
        if _spread_pct(q.bid, q.ask) > cfg.max_spread_pct:
            continue
        if q.delta is None:
            continue
        d = abs(q.delta)
        if not (cfg.target_delta_band[0] <= d <= cfg.target_delta_band[1]):
            continue
        filtered.append(q)
    if not filtered:
        return None
    # Choose the tightest spread, then closest to mid delta in band
    filtered.sort(key=lambda x: (_spread_pct(x.bid, x.ask), abs(abs(x.delta) - sum(cfg.target_delta_band) / 2)))
    return filtered[0]


def route(market: MarketState, side: Side, cfg: Optional[RouterConfig] = None) -> Optional[OptionSelection]:
    """
    Given market snapshot and desired side, pick the best option.
    Assumes market.options is already limited to nearest 1–3 expiries and ±6 ATM.
    """
    cfg = cfg or RouterConfig()
    if not market.options:
        return None

    # Partition by expiry, favor nearer expiries
    expiries = sorted({q.expiry for q in market.options})
    expiries = expiries[: cfg.expiries]

    # Candidate selection by expiry
    for expiry in expiries:
        cands = [q for q in market.options if q.expiry == expiry]
        leg = _choose_leg(cands, side, cfg)
        if leg:
            # Build limit price around mid (buy: ask/min(mid*(1+e), ask) ; sell: bid/max(mid*(1-e), bid))
            e = cfg.limit_price_slippage_pct
            mid = leg.mid if leg.mid > 0 else (leg.bid + leg.ask) / 2.0
            if "BUY" in side:
                limit = min(leg.ask, mid * (1.0 + e))
            else:
                limit = max(leg.bid, mid * (1.0 - e))
            return OptionSelection(
                expiry=leg.expiry,
                strike=leg.strike,
                side=side,
                qty=0.0,  # qty decided by risk manager upstream
                limit_price=round(float(limit), 8),
                meta={
                    "delta": leg.delta,
                    "spread_pct": _spread_pct(leg.bid, leg.ask),
                    "mid": mid,
                    "symbol": leg.symbol,
                },
            )
    return None
