# clock.py
"""CanonicalClock - deterministic timestamp supplier for synthetic data.
Provides strictly increasing, timezone-aware UTC pandas.Timestamp values.
Configurable via 'start_ts' (string or pd.Timestamp), and an 'inter_event_us'
microsecond increment (default 1000 us = 1 ms).
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import pandas as pd


class CanonicalClock:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        start_ts = cfg.get("start_ts")
        if start_ts is None:
            # default: now UTC
            ts = pd.Timestamp.now(tz="UTC")
        else:
            ts = pd.to_datetime(start_ts, utc=True)
        self._current = ts
        # microseconds step between successive ticks; default 1ms
        self._inter_event_us = int(cfg.get("inter_event_us", 1000))
        # optional jitter disabled by default
        self._jitter_us = int(cfg.get("jitter_us", 0))

    def next(self) -> pd.Timestamp:
        """Return next timestamp (tz-aware UTC) and advance internal pointer."""
        ts = self._current
        # advance
        delta = pd.Timedelta(microseconds=self._inter_event_us)
        self._current = ts + delta
        # apply optional jitter deterministically (not random)
        if self._jitter_us:
            # simple alternating jitter pattern to keep deterministic behavior
            j = self._jitter_us if (int(ts.value) % 2 == 0) else -self._jitter_us
            self._current = self._current + pd.Timedelta(microseconds=j)
        # ensure tz-aware UTC
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts

    def now(self) -> pd.Timestamp:
        return self._current

    def reset(self, start_ts: Optional[str] = None):
        if start_ts is None:
            self._current = pd.Timestamp.now(tz="UTC")
        else:
            self._current = pd.to_datetime(start_ts, utc=True)
