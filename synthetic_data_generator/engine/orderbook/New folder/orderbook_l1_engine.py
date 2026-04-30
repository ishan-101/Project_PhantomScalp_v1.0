# orderbook_l1_engine.py
"""
Orderbook L1 engine (patched).
- Defensive imports for utils/config regardless of package vs direct script run.
- Preserves existing L1 generation logic (timestamps, sequence ids, top-of-book).
- Writes into: <paths.base>/<paths.orderbook_l1>/date=YYYY-MM-DD/orderbook_l1.parquet
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ---------------------------
# Utility import helpers
# ---------------------------
def _try_import_utils():
    """
    Try to import utility modules from a few candidate locations.
    Returns tuple (loader_mod, io_writer_mod, sequence_id_mod, rng_mod, clock_mod)
    Raises ImportError with diagnostic text on failure.
    """
    tried = []

    candidates = [
        "synthetic_data_generator.engine.utils",
        "engine.utils",
    ]

    for base in candidates:
        try:
            base_mod = __import__(base, fromlist=["loader", "io_writer", "sequence_id", "rng", "clock"])
            # try to get submodules either as attributes or import them explicitly
            def _get(sub):
                return getattr(base_mod, sub, __import__(f"{base}.{sub}", fromlist=[sub]))
            loader_mod = _get("loader")
            io_writer_mod = _get("io_writer")
            seq_mod = _get("sequence_id")
            rng_mod = _get("rng")
            clock_mod = _get("clock")
            return loader_mod, io_writer_mod, seq_mod, rng_mod, clock_mod
        except Exception as e:
            tried.append((base, str(e)))
            continue

    # fallback: loader might live under engine/config
    config_loader_paths = [
        "synthetic_data_generator.engine.config.loader",
        "engine.config.loader",
    ]
    for loader_path in config_loader_paths:
        try:
            loader_mod = __import__(loader_path, fromlist=["load_config"])
            # get other utils from the utils candidate paths
            for base in ("synthetic_data_generator.engine.utils", "engine.utils"):
                try:
                    utils_mod = __import__(base, fromlist=["io_writer", "sequence_id", "rng", "clock"])
                    io_writer_mod = getattr(utils_mod, "io_writer", __import__(f"{base}.io_writer", fromlist=["ParquetWriter"]))
                    seq_mod = getattr(utils_mod, "sequence_id", __import__(f"{base}.sequence_id", fromlist=["SequenceID"]))
                    rng_mod = getattr(utils_mod, "rng", __import__(f"{base}.rng", fromlist=["RNG"]))
                    clock_mod = getattr(utils_mod, "clock", __import__(f"{base}.clock", fromlist=["CanonicalClock"]))
                    return loader_mod, io_writer_mod, seq_mod, rng_mod, clock_mod
                except Exception as e:
                    tried.append((f"{loader_path}+{base}", str(e)))
                    continue
        except Exception as e:
            tried.append((loader_path, str(e)))
            continue

    # last resort: try relative imports via repository layout (repo root detection)
    # If running as a script, repo root is assumed to be three parents up from this file
    here = Path(__file__).resolve()
    repo_root = here.parents[3]  # .../synthetic_data_generator/engine/orderbook/... -> repo root
    sys.path.insert(0, str(repo_root))
    try:
        import synthetic_data_generator.engine.utils as utils_pkg
        loader_mod = getattr(utils_pkg, "loader", __import__("synthetic_data_generator.engine.utils.loader", fromlist=["loader"]))
        io_writer_mod = getattr(utils_pkg, "io_writer", __import__("synthetic_data_generator.engine.utils.io_writer", fromlist=["ParquetWriter"]))
        seq_mod = getattr(utils_pkg, "sequence_id", __import__("synthetic_data_generator.engine.utils.sequence_id", fromlist=["SequenceID"]))
        rng_mod = getattr(utils_pkg, "rng", __import__("synthetic_data_generator.engine.utils.rng", fromlist=["RNG"]))
        clock_mod = getattr(utils_pkg, "clock", __import__("synthetic_data_generator.engine.utils.clock", fromlist=["CanonicalClock"]))
        return loader_mod, io_writer_mod, seq_mod, rng_mod, clock_mod
    except Exception as e:
        tried.append(("local import after adding repo_root", str(e)))

    msg = "Failed to import utilities from expected locations. Tried:\n"
    for base, err in tried:
        msg += f" - {base}: {err}\n"
    raise ImportError(msg)


# ---------------------------
# Config loader (central config)
# ---------------------------
def _load_central_config(loader_mod: Optional[Any] = None) -> Dict[str, Any]:
    """
    Attempts to load central synthetic_config.yaml using:
      1) loader_mod.load_config() if provided
      2) engine/config/synthetic_config.yaml relative to this file
      3) repo_root/synthetic_config.yaml
    """
    if loader_mod is not None:
        try:
            if hasattr(loader_mod, "load_config"):
                cfg = loader_mod.load_config()
                if isinstance(cfg, dict):
                    return cfg
        except Exception:
            pass

    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "synthetic_config.yaml",               # repo root
        here.parents[1] / "config" / "synthetic_config.yaml",   # engine/config
        here.parents[3] / "engine" / "config" / "synthetic_config.yaml",
    ]
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf8") as fh:
                cfg = yaml.safe_load(fh)
                if isinstance(cfg, dict):
                    return cfg

    raise FileNotFoundError(
        "Central config not found. Put synthetic_config.yaml in repo root or engine/config/ or provide config argument."
    )


# ---------------------------
# Adapters for minor API differences
# ---------------------------
class SequenceIDAdapter:
    def __init__(self, seq_mod):
        cls = None
        for name in ("SequenceID", "SequenceIDGenerator", "SequenceGenerator"):
            cls = getattr(seq_mod, name, None)
            if cls is not None:
                break
        if cls is None:
            # maybe module exports a callable instance or class
            if callable(seq_mod):
                cls = seq_mod
            else:
                raise ImportError("Could not find SequenceID class in sequence_id module.")
        self._inst = cls() if callable(cls) else cls

    def next_id(self):
        inst = self._inst
        if hasattr(inst, "next"):
            return inst.next()
        if hasattr(inst, "next_id"):
            return inst.next_id()
        if callable(inst):
            return inst()
        if hasattr(inst, "nextval"):
            return inst.nextval()
        raise RuntimeError("SequenceID instance has no known 'next' method.")


class RNGAdapter:
    def __init__(self, rng_mod):
        rng_cls = getattr(rng_mod, "RNG", None)
        if rng_cls is None and callable(rng_mod):
            rng_cls = rng_mod
        if rng_cls is None:
            rng_inst = getattr(rng_mod, "rng", None)
            if rng_inst is not None:
                self._rng = rng_inst
            else:
                self._rng = None
        else:
            self._rng = rng_cls()

    def choice(self, *args, **kwargs):
        # try wrapped RNG first
        if self._rng is not None and hasattr(self._rng, "choice"):
            try:
                return self._rng.choice(*args, **kwargs)
            except TypeError:
                # drop leading key if present
                try:
                    if len(args) >= 2 and isinstance(args[0], str) and hasattr(args[1], "__iter__"):
                        return self._rng.choice(args[1], **kwargs)
                except Exception:
                    pass
        # fallback to numpy
        items = None
        size = kwargs.get("size", None)
        p = kwargs.get("p", None)
        if len(args) == 1:
            items = args[0]
        elif len(args) >= 2:
            if isinstance(args[0], str) and hasattr(args[1], "__iter__"):
                items = args[1]
            else:
                items = args[0]
        if items is None:
            raise TypeError("Could not resolve items for choice() fallback.")
        return np.random.choice(list(items), size=size, p=p)


# ---------------------------
# Output path helpers
# ---------------------------
def _make_out_paths(cfg: Dict[str, Any], start_ts: pd.Timestamp) -> Tuple[Path, str]:
    base = Path(cfg.get("paths", {}).get("base", "data/synthetic_data"))
    rel = cfg.get("paths", {}).get("orderbook_l1", "orderbook/l1")
    # ensure timezone-aware in UTC
    start_ts = pd.to_datetime(start_ts)
    if start_ts.tz is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    date_str = start_ts.strftime("%Y-%m-%d")
    out_dir = base.joinpath(rel).joinpath(f"date={date_str}")
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = "orderbook_l1.parquet"
    return out_dir, filename


# ---------------------------
# L1 generation logic
# ---------------------------
def _generate_l1_snapshots(
    rng_adapter: RNGAdapter,
    seq_adapter: SequenceIDAdapter,
    n: int,
    start_ts: pd.Timestamp,
    seed: Optional[int] = None,
    exchange: str = "EX",
    symbol: str = "SYM",
) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(int(seed))

    start_ts = pd.to_datetime(start_ts)
    if start_ts.tz is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")

    # generate exponential inter-arrivals (seconds)
    avg_interval_s = 0.05
    inter_arrival = np.random.exponential(scale=avg_interval_s, size=n)
    cum_seconds = np.cumsum(inter_arrival)

    # timestamps creation: keep them timezone-aware
    base_ns = start_ts.value
    offsets_ns = (cum_seconds * 1e9).astype("int64")
    timestamps = pd.to_datetime(base_ns + offsets_ns, unit="ns").tz_localize("UTC") if pd.to_datetime(base_ns).tz is None else pd.to_datetime(base_ns + offsets_ns, unit="ns").tz_convert("UTC")

    seq_ids = [int(seq_adapter.next_id()) for _ in range(n)]

    mid_price = 30000.0 + np.random.normal(scale=200.0)
    price_noise = np.cumsum(np.random.normal(scale=0.5, size=n))
    mid_prices = mid_price + price_noise

    spreads = np.abs(np.random.lognormal(mean=-3.0, sigma=0.5, size=n))
    top_bid = mid_prices - spreads / 2.0
    top_ask = mid_prices + spreads / 2.0

    top_bid_size = np.random.lognormal(mean=-3.0, sigma=1.0, size=n)
    top_ask_size = np.random.lognormal(mean=-3.0, sigma=1.0, size=n)

    df = pd.DataFrame(
        {
            "meta__timestamp": timestamps,
            "meta__sequence_id": np.array(seq_ids, dtype="int64"),
            "exchange": [exchange] * n,
            "symbol": [symbol] * n,
            "top_bid": top_bid,
            "top_ask": top_ask,
            "spread": top_ask - top_bid,
            "bid_size_0": top_bid_size,
            "ask_size_0": top_ask_size,
        }
    )

    # add price/size levels 0..9 (include bid_price_0/ask_price_0)
    for lvl in range(0, 10):
        if lvl == 0:
            df[f"bid_price_{lvl}"] = df["top_bid"]
            df[f"ask_price_{lvl}"] = df["top_ask"]
        else:
            df[f"bid_price_{lvl}"] = df["top_bid"] - lvl * 0.1
            df[f"ask_price_{lvl}"] = df["top_ask"] + lvl * 0.1
        df[f"bid_size_{lvl}"] = df["bid_size_0"] * np.random.uniform(0.1, 0.9, size=n)
        df[f"ask_size_{lvl}"] = df["ask_size_0"] * np.random.uniform(0.1, 0.9, size=n)

    df["meta__sequence_id"] = df["meta__sequence_id"].astype("int64")
    numeric_cols = [c for c in df.columns if c not in ("meta__timestamp", "exchange", "symbol")]
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df


# ---------------------------
# Engine entrypoint
# ---------------------------
def run_engine(
    config: Optional[Dict[str, Any]] = None,
    start_ts: Optional[pd.Timestamp] = None,
    rows: Optional[int] = None,
    exchange: str = "EX",
    symbol: str = "SYM",
    chunk_size: Optional[int] = None,
) -> Dict[str, Any]:
    t0 = time.time()

    loader_mod, io_writer_mod, seq_mod, rng_mod, clock_mod = _try_import_utils()

    Seq = SequenceIDAdapter(seq_mod)
    RNG = RNGAdapter(rng_mod)

    cfg = config or _load_central_config(loader_mod)

    n_total = rows or int(cfg.get("rows", {}).get("orderbook_l1", 300000))
    sharder = cfg.get("partitioning", {}).get("sharding", {}) or {}
    default_chunk = int(sharder.get("min_rows_per_file", 20000))
    chunk_size = int(chunk_size or sharder.get("max_rows_per_file", default_chunk) or default_chunk)

    if start_ts is None:
        g = cfg.get("global", {})
        default_ts = g.get("default_start_ts")
        if default_ts:
            start_ts = pd.to_datetime(default_ts)
            if start_ts.tz is None:
                start_ts = start_ts.tz_localize("UTC")
            else:
                start_ts = start_ts.tz_convert("UTC")
        else:
            start_ts = pd.Timestamp.utcnow().tz_localize("UTC")
    else:
        start_ts = pd.to_datetime(start_ts)
        if start_ts.tz is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")

    out_dir, filename = _make_out_paths(cfg, start_ts)
    out_path = out_dir / filename

    compression = cfg.get("writer", {}).get("compression", cfg.get("partitioning", {}).get("compression", "snappy"))
    PartWriter = getattr(io_writer_mod, "ParquetWriter", None)
    if PartWriter is None:
        if hasattr(io_writer_mod, "writer"):
            ParquetWriter = io_writer_mod.writer
        else:
            raise ImportError("io_writer.ParquetWriter not found in utils.io_writer module.")
    else:
        ParquetWriter = PartWriter

    # create writer with fallback signatures
    writer = None
    try:
        writer = ParquetWriter(str(out_path), compression=compression, part_rows=int(chunk_size))
    except TypeError:
        try:
            writer = ParquetWriter(out_dir=out_dir, filename=filename, part_rows=int(chunk_size), compression=compression)
        except Exception:
            writer = ParquetWriter(str(out_path), compression=compression)

    produced = 0
    part_idx = 0
    while produced < n_total:
        this_n = min(chunk_size, n_total - produced)
        df = _generate_l1_snapshots(RNG, Seq, this_n, start_ts + pd.to_timedelta(produced, unit="s"), seed=cfg.get("global", {}).get("seed"), exchange=exchange, symbol=symbol)

        part_name = f"orderbook_l1.part-{part_idx:05d}.parquet"
        try:
            if hasattr(writer, "write"):
                try:
                    writer.write(df)
                except TypeError:
                    writer.write(part_name, df)
            else:
                writer(df, part_name)
        except Exception as e:
            raise RuntimeError(f"Failed to write part via io_writer: {e}")

        produced += this_n
        part_idx += 1
        elapsed = time.time() - t0
        print(f"[orderbook_l1] wrote part {part_idx} — total {produced}/{n_total} snapshots — elapsed {elapsed:.1f}s")

    try:
        if hasattr(writer, "finalize"):
            writer.finalize()
        elif hasattr(io_writer_mod, "finalize_writer"):
            io_writer_mod.finalize_writer(str(out_path))
    except Exception as e:
        raise RuntimeError(f"Failed to finalize/merge parts: {e}")

    elapsed_total = time.time() - t0
    bytes_written = out_path.stat().st_size if out_path.exists() else 0
    manifest = {"path": str(out_path), "rows": n_total, "bytes": bytes_written}
    result = {"rows": n_total, "manifest": manifest, "timing": {"seconds": elapsed_total}}
    return result


if __name__ == "__main__":
    import json
    try:
        res = run_engine()
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"[runner ERROR] Engine raised exception: {e}")
        raise
