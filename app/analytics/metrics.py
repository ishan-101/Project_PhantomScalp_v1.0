# app/analytics/metrics.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from collections import defaultdict
from datetime import datetime


@dataclass
class RunMetrics:
    initial_capital: float
    final_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    loss_rate_pct: float
    avg_profit_percentage_per_win: float
    avg_loss_percentage_per_loss: float
    winning_days: int
    losing_days: int
    avg_profit_percentage_per_winning_days: float
    avg_loss_percentage_per_losing_days: float
    avg_trades_per_day: float
    avg_winning_trades_per_day: float
    avg_losing_trades_per_day: float
    total_scalp_trades: int
    total_runner_trades: int
    total_trend_trades: int
    avg_scalp_trades_per_day: float
    avg_runner_trades_per_day: float
    avg_trend_trades_per_day: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Metrics:
    """
    Backward compatible with v0.1:
      - You can still call update(signal, row) with row['future_return_5m'].
    v0.2:
      - Tracks full ledger and computes the complete metrics dict.
      - Preferred usage: add_trade(day, profile, pnl_pct, win).
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = float(initial_capital)
        self.ledger: List[Dict[str, Any]] = []
        self._trades = 0
        self._wins = 0

    # ===== v0.1 compatibility layer =====
    def update(self, signal, row: Dict[str, Any]):
        """
        v0.1 placeholder compatibility:
        Counts a win if signal.direction matches sign of future_return_5m.
        Also records a minimal trade into the v0.2 ledger.
        """
        direction = getattr(signal, "direction", None)  # 'long' | 'short'
        fut = float(row.get("future_return_5m", 0.0))
        if direction in ("long", "short"):
            self._trades += 1
            win = (direction == "long" and fut > 0.0) or (direction == "short" and fut < 0.0)
            if win:
                self._wins += 1
            # Record a simple ledger entry (assume scalp profile by default)
            day = row.get("day") or self._infer_day(row.get("ts"))
            self.ledger.append({
                "day": day,
                "profile": row.get("profile", "scalp"),
                "pnl_pct": abs(fut) if win else -abs(fut),
                "win": win
            })

    def summary(self) -> Dict[str, Any]:
        """v0.1 summary — still available."""
        win_rate = (self._wins / self._trades * 100.0) if self._trades else 0.0
        return {
            "total_trades": self._trades,
            "winning_trades": self._wins,
            "win_rate_pct": round(win_rate, 2)
        }

    # ===== v0.2 preferred API =====
    def add_trade(self, *, day: Optional[str] = None, ts: Optional[int] = None,
                  profile: str, pnl_pct: float, win: bool):
        """
        Add a trade to the ledger explicitly.
        - day: 'YYYY-MM-DD' (preferred). If None, inferred from ts.
        - profile: 'scalp' | 'runner' | 'trend'
        - pnl_pct: percentage PnL for the trade (e.g., +12.3 or -4.7)
        - win: True/False
        """
        d = day or self._infer_day(ts)
        pr = profile if profile in ("scalp", "runner", "trend") else "scalp"
        self.ledger.append({"day": d, "profile": pr, "pnl_pct": float(pnl_pct), "win": bool(win)})

    def finalize(self) -> RunMetrics:
        """
        Compute full v0.2 metrics from the ledger.
        """
        init = self.initial_capital
        n = len(self.ledger)
        wins = sum(1 for r in self.ledger if r["win"])
        losses = n - wins
        win_rate = (wins / n * 100.0) if n else 0.0
        loss_rate = 100.0 - win_rate

        win_pcts = [abs(r["pnl_pct"]) for r in self.ledger if r["win"]]
        loss_pcts = [abs(r["pnl_pct"]) for r in self.ledger if not r["win"]]
        avg_win = (sum(win_pcts) / len(win_pcts)) if win_pcts else 0.0
        avg_loss = (sum(loss_pcts) / len(loss_pcts)) if loss_pcts else 0.0

        # per-day aggregates
        by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in self.ledger:
            by_day[r["day"]].append(r)
        days = sorted(by_day.keys())
        day_pnls = {d: sum(x["pnl_pct"] for x in by_day[d]) for d in days}
        winning_days = sum(1 for d in days if day_pnls[d] > 0.0)
        losing_days = sum(1 for d in days if day_pnls[d] <= 0.0)
        avg_win_day = (sum(day_pnls[d] for d in days if day_pnls[d] > 0.0) / winning_days) if winning_days else 0.0
        avg_loss_day = (abs(sum(day_pnls[d] for d in days if day_pnls[d] <= 0.0)) / losing_days) if losing_days else 0.0

        # trades per day
        tpd = (n / len(days)) if days else 0.0
        wtpd = (wins / len(days)) if days else 0.0
        ltpd = (losses / len(days)) if days else 0.0

        # profile counts
        scalp = sum(1 for r in self.ledger if r["profile"] == "scalp")
        runner = sum(1 for r in self.ledger if r["profile"] == "runner")
        trend = sum(1 for r in self.ledger if r["profile"] == "trend")

        # profile rates per day
        aspd = (scalp / len(days)) if days else 0.0
        arpd = (runner / len(days)) if days else 0.0
        atpd = (trend / len(days)) if days else 0.0

        # final capital (compound by trade PnL%)
        capital = init
        for r in self.ledger:
            capital *= (1.0 + r["pnl_pct"] / 100.0)

        return RunMetrics(
            initial_capital=init,
            final_capital=capital,
            total_trades=n,
            winning_trades=wins,
            losing_trades=losses,
            win_rate_pct=win_rate,
            loss_rate_pct=loss_rate,
            avg_profit_percentage_per_win=avg_win,
            avg_loss_percentage_per_loss=avg_loss,
            winning_days=winning_days,
            losing_days=losing_days,
            avg_profit_percentage_per_winning_days=avg_win_day,
            avg_loss_percentage_per_losing_days=avg_loss_day,
            avg_trades_per_day=tpd,
            avg_winning_trades_per_day=wtpd,
            avg_losing_trades_per_day=ltpd,
            total_scalp_trades=scalp,
            total_runner_trades=runner,
            total_trend_trades=trend,
            avg_scalp_trades_per_day=aspd,
            avg_runner_trades_per_day=arpd,
            avg_trend_trades_per_day=atpd,
        )

    # ===== helpers =====
    @staticmethod
    def _infer_day(ts: Optional[int]) -> str:
        if ts is None:
            # fallback to today (UTC) if timestamp absent
            return datetime.utcnow().strftime("%Y-%m-%d")
        # ts in ms → convert
        sec = ts / 1000.0 if ts > 10_000_000_000 else ts
        return datetime.utcfromtimestamp(sec).strftime("%Y-%m-%d")