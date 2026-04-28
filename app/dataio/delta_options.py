# app/dataio/delta_options.py
from __future__ import annotations

import os
import time
import json
import gzip
import math
import hashlib
import logging
import pathlib
import datetime as dt
from typing import Optional, Dict, Any, List, Tuple

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
    def __init__(self, root: str = "data/cache/delta", gzip_enabled: bool = True):
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
    def __init__(self, rate: Optional[RateLimiter] = None, timeout: int = 20, retries: int = 3):
        self.rate = rate or RateLimiter(8, 1.0)  # default ~8 rps
        self.timeout = timeout
        self.retries = retries

    def get(self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> bytes:
        headers = headers or {}
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        for attempt in range(self.retries):
            try:
                self.rate.wait()
                req = urllib.request.Request(url, headers=headers, method="GET")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.read()
            except urllib.error.HTTPError as e:
                if e.code in (418, 429, 503):
                    sleep_s = (1.6 ** attempt) + (0.05 * attempt)
                    logger.warning("HTTP %s from %s. Backing off %.2fs", e.code, url, sleep_s)
                    time.sleep(sleep_s)
                    continue
                raise
            except urllib.error.URLError as e:
                sleep_s = (1.6 ** attempt) + (0.05 * attempt)
                logger.warning("URLError %s. Retrying in %.2fs", e, sleep_s)
                time.sleep(sleep_s)
        raise RuntimeError(f"Failed GET after retries: {url}")


class DeltaOptionsClient:
    """
    Public market data client for Delta Exchange options (Delta Exchange India compatible).
    Also supports 'file' mode for offline/recorded data.

    Key endpoints (base_url configurable via env DELTA_REST_BASE):
      - /v2/public/instruments  -> list instruments (incl. options)
      - /v2/public/ticker       -> mark/last/iv per instrument
      - /v2/public/ohlc         -> candles for instrument
      - /v2/public/trades       -> recent trades per instrument

    NOTE: Exact paths can vary by environment; keep base_url/env configurable.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        cache: Optional[DiskCache] = None,
        http: Optional[HttpClient] = None,
        mode: str = "live",
        file_map: Optional[Dict[str, str]] = None,
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url or os.environ.get("DELTA_REST_BASE", "https://api.delta.exchange")
        self.cache = cache or DiskCache()
        self.http = http or HttpClient()
        self.mode = mode
        self.file_map = file_map or {}   # e.g., {"instruments": "data/drops/delta_instruments.json", "ohlc:BTC-30AUG24-60000-C:1m": "..."}
        self.api_key = api_key or os.environ.get("DELTA_API_KEY", "")

    # ---------------------- Public API --------------------------------------

    def list_options_instruments(self, underlying: str = "BTC") -> pd.DataFrame:
        """
        Returns instruments with columns:
         ['symbol','underlying','expiry','strike','option_type','tick_size','lot_size','active']
        """
        if self.mode == "file" and "instruments" in self.file_map:
            with open(self.file_map["instruments"], "rb") as f:
                payload = f.read()
        else:
            url = f"{self.base_url}/v2/public/instruments"
            key = url
            payload = self.cache.get(key)
            if payload is None:
                payload = self.http.get(url)
                self.cache.set(key, payload)

        data = json.loads(payload.decode("utf-8"))
        items = data.get("result") or data.get("instruments") or data  # flexible
        rows = []
        for it in items:
            try:
                if str(it.get("underlying_asset","")).upper().startswith(underlying.upper()):
                    rows.append(dict(
                        symbol=it.get("symbol") or it.get("name"),
                        underlying=it.get("underlying_asset") or it.get("underlying"),
                        expiry=it.get("expiry_date") or it.get("settlement_time") or it.get("expiry"),
                        strike=float(it.get("strike_price") or it.get("strike") or 0.0),
                        option_type=(it.get("option_type") or it.get("type","")).upper(),
                        tick_size=float(it.get("tick_size") or it.get("min_tick_size") or 0.5),
                        lot_size=float(it.get("contract_value") or it.get("lot_size") or 1.0),
                        active=bool(it.get("is_active", True)),
                    ))
            except Exception:
                continue
        import pandas as pd
        df = pd.DataFrame(rows)
        if not df.empty:
            # Normalize expiry -> UTC datetime if parseable
            df["expiry"] = pd.to_datetime(df["expiry"], utc=True, errors="coerce")
        return df

    def fetch_option_ohlc(
        self,
        instrument: str,
        interval: str,
        start: dt.datetime | int,
        end: dt.datetime | int,
        limit_per_call: int = 1000,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Candles for a single option instrument.
        Returns DataFrame with:
         ['datetime','timestamp_ms','instrument','interval','open','high','low','close','volume']
        """
        if self.mode == "file":
            key = f"ohlc:{instrument}:{interval}"
            path = self.file_map.get(key)
            if not path:
                raise ValueError(f"file mode requires file_map['{key}']")
            import pandas as pd
            df = pd.read_csv(path)
            if "timestamp_ms" not in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                df["timestamp_ms"] = (df["datetime"].astype("int64") // 10**6).astype("int64")
            df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
            df["instrument"] = instrument
            df["interval"] = interval
            cols = ["datetime","timestamp_ms","instrument","interval","open","high","low","close","volume"]
            return df[cols].sort_values("timestamp_ms").reset_index(drop=True)

        start_ms = _utc_ms(start)
        end_ms = _utc_ms(end)
        out: List[Dict[str, Any]] = []

        endpoint = f"{self.base_url}/v2/public/ohlc"
        # Many exchanges accept params like instrument_name, resolution, start, end.
        # Keep names flexible via multiple keys.
        next_from = start_ms
        while next_from < end_ms:
            params = dict(
                instrument_name=instrument,
                symbol=instrument,
                resolution=interval,
                interval=interval,
                start=next_from,
                end=end_ms,
                limit=limit_per_call,
            )
            url_with_qs = f"{endpoint}?{urllib.parse.urlencode(params)}"
            payload = None
            if use_cache:
                payload = self.cache.get(url_with_qs)
            if payload is None:
                payload = self.http.get(endpoint, params=params)
                if use_cache:
                    self.cache.set(url_with_qs, payload)
            data = json.loads(payload.decode("utf-8"))
            candles = data.get("result") or data.get("candles") or data.get("data") or []
            if not candles:
                break
            out.extend(candles)
            # Infer next_from progression
            last = candles[-1]
            last_ts = int(last.get("time") or last.get("t") or last.get("timestamp") or last.get("open_time"))
            next_from = last_ts + 1
            if len(candles) < limit_per_call:
                break

        if not out:
            return self._empty_ohlc(instrument, interval)

        # Normalize to columns
        rows = []
        for c in out:
            # accept multiple possible field keys
            open_ms = int(c.get("time") or c.get("t") or c.get("timestamp") or c.get("open_time"))
            rows.append(dict(
                datetime=_to_utc(open_ms),
                timestamp_ms=open_ms,
                instrument=instrument,
                interval=interval,
                open=float(c.get("open") or c.get("o")),
                high=float(c.get("high") or c.get("h")),
                low=float(c.get("low") or c.get("l")),
                close=float(c.get("close") or c.get("c")),
                volume=float(c.get("volume") or c.get("v") or 0.0),
            ))
        import pandas as pd
        df = pd.DataFrame(rows).sort_values("timestamp_ms").reset_index(drop=True)
        return df

    def fetch_option_trades(
        self,
        instrument: str,
        start: dt.datetime | int,
        end: dt.datetime | int,
        page_size: int = 1000,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Tick trades (best effort). Returns:
          ['datetime','timestamp_ms','instrument','price','size','side','trade_id']
        """
        if self.mode == "file":
            key = f"trades:{instrument}"
            path = self.file_map.get(key)
            if not path:
                raise ValueError(f"file mode requires file_map['{key}']")
            import pandas as pd
            df = pd.read_csv(path)
            if "timestamp_ms" not in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                df["timestamp_ms"] = (df["datetime"].astype("int64") // 10**6).astype("int64")
            df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
            df["instrument"] = instrument
            cols = ["datetime","timestamp_ms","instrument","price","size","side","trade_id"]
            return df[cols].sort_values("timestamp_ms").reset_index(drop=True)

        start_ms = _utc_ms(start)
        end_ms = _utc_ms(end)
        out: List[Dict[str, Any]] = []

        endpoint = f"{self.base_url}/v2/public/trades"
        next_from = start_ms
        while next_from < end_ms:
            params = dict(
                instrument_name=instrument,
                symbol=instrument,
                start=next_from,
                end=end_ms,
                limit=page_size,
            )
            url_with_qs = f"{endpoint}?{urllib.parse.urlencode(params)}"
            payload = None
            if use_cache:
                payload = self.cache.get(url_with_qs)
            if payload is None:
                payload = self.http.get(endpoint, params=params)
                if use_cache:
                    self.cache.set(url_with_qs, payload)
            data = json.loads(payload.decode("utf-8"))
            trades = data.get("result") or data.get("trades") or data.get("data") or []
            if not trades:
                break
            out.extend(trades)
            last = trades[-1]
            last_ts = int(last.get("timestamp") or last.get("t") or last.get("time"))
            next_from = last_ts + 1
            if len(trades) < page_size:
                break

        rows = []
        for t in out:
            ts = int(t.get("timestamp") or t.get("t") or t.get("time"))
            rows.append(dict(
                datetime=_to_utc(ts),
                timestamp_ms=ts,
                instrument=instrument,
                price=float(t.get("price") or t.get("p")),
                size=float(t.get("size") or t.get("qty") or t.get("q") or 0.0),
                side=(t.get("side") or t.get("s") or "").upper(),
                trade_id=str(t.get("id") or t.get("trade_id") or ""),
            ))
        import pandas as pd
        df = pd.DataFrame(rows).sort_values("timestamp_ms").reset_index(drop=True)
        return df

    # ---------------------- Helpers -----------------------------------------

    @staticmethod
    def _empty_ohlc(instrument: str, interval: str):
        import pandas as pd
        return pd.DataFrame(
            columns=["datetime","timestamp_ms","instrument","interval","open","high","low","close","volume"]
        )
