# app/dataio/binance_spot.py
from __future__ import annotations

import os
import time
import math
import json
import gzip
import hashlib
import logging
import pathlib
import datetime as dt
from typing import Iterable, Optional, Dict, Any, List, Tuple

import urllib.parse
import urllib.request
import urllib.error

try:
    import pandas as pd
except ImportError as e:
    raise ImportError("pandas is required for DataIO modules") from e

logger = logging.getLogger(__name__)
ISO = "%Y-%m-%dT%H:%M:%SZ"


def _utc_ms(x: dt.datetime | int | float) -> int:
    if isinstance(x, (int, float)):
        return int(x)
    if x.tzinfo is None:
        x = x.replace(tzinfo=dt.timezone.utc)
    return int(x.timestamp() * 1000)


def _to_utc(ms: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc)


class RateLimiter:
    """Simple token-bucket-ish limiter: N requests per window seconds."""
    def __init__(self, max_per_window: int, window_seconds: float):
        self.max = max_per_window
        self.window = window_seconds
        self.bucket: List[float] = []

    def wait(self):
        now = time.time()
        self.bucket = [t for t in self.bucket if (now - t) < self.window]
        if len(self.bucket) >= self.max:
            sleep_for = self.window - (now - self.bucket[0]) + 0.01
            if sleep_for > 0:
                time.sleep(sleep_for)
        self.bucket.append(time.time())


class DiskCache:
    """Simple GET cache with optional gzip. Keyed by URL + query string hash."""
    def __init__(self, root: str = "data/cache/binance", gzip_enabled: bool = True):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.gzip_enabled = gzip_enabled

    def _path(self, key: str) -> pathlib.Path:
        h = hashlib.sha256(key.encode()).hexdigest()
        ext = ".json.gz" if self.gzip_enabled else ".json"
        return self.root / f"{h}{ext}"

    def get(self, key: str) -> Optional[bytes]:
        p = self._path(key)
        if not p.exists():
            return None
        data = p.read_bytes()
        if self.gzip_enabled:
            return gzip.decompress(data)
        return data

    def set(self, key: str, payload: bytes):
        p = self._path(key)
        if self.gzip_enabled:
            payload = gzip.compress(payload, compresslevel=6)
        p.write_bytes(payload)


class HttpClient:
    """Minimal stdlib HTTP client with retries + jitter."""
    def __init__(self, rate: Optional[RateLimiter] = None, timeout: int = 20, retries: int = 3):
        self.rate = rate or RateLimiter(10, 1.0)  # default ~10 rps
        self.timeout = timeout
        self.retries = retries

    def get(self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> bytes:
        headers = headers or {}
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        key_for_cache = url
        # No ETags here; disk cache handled upstream if desired.
        for attempt in range(self.retries):
            try:
                self.rate.wait()
                req = urllib.request.Request(url, headers=headers, method="GET")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.read()
            except urllib.error.HTTPError as e:
                if e.code in (418, 429, 418):  # binance may return 418/429 for limits
                    sleep_s = (1.5 ** attempt) + (0.05 * attempt)
                    logger.warning("HTTP %s from %s. Backing off %.2fs", e.code, url, sleep_s)
                    time.sleep(sleep_s)
                    continue
                raise
            except urllib.error.URLError as e:
                sleep_s = (1.5 ** attempt) + (0.05 * attempt)
                logger.warning("URLError %s. Retrying in %.2fs", e, sleep_s)
                time.sleep(sleep_s)
        raise RuntimeError(f"Failed GET after retries: {url}")


class BinanceSpotClient:
    """
    Spot OHLCV from Binance (public). Also supports 'file' mode for offline backtests.

    Modes:
      - live: hits REST endpoints
      - file: reads pre-dumped klines CSV/NDJSON for repeatable backtests

    Output:
      pandas.DataFrame with columns:
        ['datetime','timestamp_ms','symbol','interval','open','high','low','close','volume','trades','quote_volume','taker_base_vol','taker_quote_vol']
      All UTC; 'datetime' is timezone-aware.
    """
    def __init__(
        self,
        base_url: Optional[str] = None,
        cache: Optional[DiskCache] = None,
        http: Optional[HttpClient] = None,
        mode: str = "live",
        file_path: Optional[str] = None,
    ):
        self.base_url = base_url or os.environ.get("BINANCE_REST_BASE", "https://api.binance.com")
        self.cache = cache or DiskCache()
        self.http = http or HttpClient()
        self.mode = mode
        self.file_path = file_path  # used in file mode

    # ---- Public API ---------------------------------------------------------

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start: dt.datetime | int,
        end: dt.datetime | int,
        limit_per_call: int = 1000,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV klines in [start, end).

        interval examples: '15s', '1m', '3m', '5m', '15m', '1h', '4h'
        """
        if self.mode == "file":
            return self._load_klines_from_file(symbol, interval, start, end)

        start_ms = _utc_ms(start)
        end_ms = _utc_ms(end)

        out: List[List[Any]] = []
        next_from = start_ms

        endpoint = f"{self.base_url}/api/v3/klines"

        while next_from < end_ms:
            params = dict(symbol=symbol.upper(), interval=interval, startTime=next_from, endTime=end_ms, limit=limit_per_call)
            url_with_qs = f"{endpoint}?{urllib.parse.urlencode(params)}"
            payload = None
            if use_cache:
                payload = self.cache.get(url_with_qs)
            if payload is None:
                payload = self.http.get(endpoint, params=params)
                if use_cache:
                    self.cache.set(url_with_qs, payload)
            chunk = json.loads(payload.decode("utf-8"))
            if not chunk:
                break
            out.extend(chunk)
            # Binance returns [openTime, open, high, low, close, volume, closeTime, ...]
            last_open = chunk[-1][0]
            # Advance by 1 ms after last open to avoid duplicate
            next_from = last_open + 1
            # Safety: stop infinite loops
            if len(chunk) < limit_per_call:
                break

        if not out:
            return self._empty_ohlcv(symbol, interval)

        return self._klines_to_df(symbol, interval, out)

    # ---- Internals ----------------------------------------------------------

    def _klines_to_df(self, symbol: str, interval: str, rows: List[List[Any]]):
        # See: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
        # row indices:
        # 0 openTime, 1 open, 2 high, 3 low, 4 close, 5 volume, 6 closeTime,
        # 7 quote_asset_volume, 8 number_of_trades, 9 taker_buy_base,
        # 10 taker_buy_quote, 11 ignore
        import pandas as pd
        data = []
        for r in rows:
            open_ms = int(r[0])
            data.append(
                dict(
                    datetime=_to_utc(open_ms),
                    timestamp_ms=open_ms,
                    symbol=symbol.upper(),
                    interval=interval,
                    open=float(r[1]),
                    high=float(r[2]),
                    low=float(r[3]),
                    close=float(r[4]),
                    volume=float(r[5]),
                    trades=int(r[8]),
                    quote_volume=float(r[7]),
                    taker_base_vol=float(r[9]),
                    taker_quote_vol=float(r[10]),
                )
            )
        df = pd.DataFrame(data).sort_values("timestamp_ms").reset_index(drop=True)
        return df

    def _load_klines_from_file(
        self, symbol: str, interval: str, start: dt.datetime | int, end: dt.datetime | int
    ):
        if not self.file_path:
            raise ValueError("file_path must be provided in file mode")
        import pandas as pd
        df = pd.read_csv(self.file_path)
        # expected columns: datetime (ISO or epoch ms), open, high, low, close, volume
        if "timestamp_ms" not in df.columns:
            if "datetime" in df.columns:
                # try parse
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                df["timestamp_ms"] = (df["datetime"].astype("int64") // 10**6).astype("int64")
            else:
                raise ValueError("file must contain either 'timestamp_ms' or 'datetime' column")
        df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df["symbol"] = symbol.upper()
        df["interval"] = interval
        # Normalize optional columns
        for c in ("trades", "quote_volume", "taker_base_vol", "taker_quote_vol"):
            if c not in df.columns:
                df[c] = 0.0 if "vol" in c or "volume" in c else 0
        start_ms, end_ms = _utc_ms(start), _utc_ms(end)
        df = df[(df["timestamp_ms"] >= start_ms) & (df["timestamp_ms"] < end_ms)].copy()
        df = df[["datetime","timestamp_ms","symbol","interval","open","high","low","close","volume","trades","quote_volume","taker_base_vol","taker_quote_vol"]]
        return df.reset_index(drop=True)

    @staticmethod
    def _empty_ohlcv(symbol: str, interval: str):
        import pandas as pd
        return pd.DataFrame(
            columns=["datetime","timestamp_ms","symbol","interval","open","high","low","close","volume","trades","quote_volume","taker_base_vol","taker_quote_vol"]
        )
