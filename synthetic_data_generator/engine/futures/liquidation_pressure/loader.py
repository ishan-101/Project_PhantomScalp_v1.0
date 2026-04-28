"""Loader for Futures Liquidation Pressure dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from synthetic_data_generator.engine.config.loader import load_config
from synthetic_data_generator.engine.utils.schema_validator import validate_basic_tick_schema


class LiquidationPressureLoaderError(Exception):
    """Raised when liquidation source dependencies are missing or invalid."""


@dataclass(frozen=True)
class SourceLoadResult:
    partition_date: str
    trades_path: str
    orderflow_path: str
    oi_path: str
    funding_path: str
    trades_df: pd.DataFrame
    orderflow_df: pd.DataFrame
    oi_df: pd.DataFrame
    funding_df: pd.DataFrame

    def to_dict(self) -> Dict[str, object]:
        return {
            "partition_date": self.partition_date,
            "trades_path": self.trades_path,
            "orderflow_path": self.orderflow_path,
            "oi_path": self.oi_path,
            "funding_path": self.funding_path,
            "trades_df": self.trades_df,
            "orderflow_df": self.orderflow_df,
            "oi_df": self.oi_df,
            "funding_df": self.funding_df,
        }


def _resolve_partition_dir(base_dir: Path, explicit_date: Optional[str], dataset_name: str) -> tuple[str, Path]:
    if explicit_date:
        partition_dir = base_dir / f"date={explicit_date}"
        if not partition_dir.exists():
            raise LiquidationPressureLoaderError(
                f"Requested {dataset_name} partition does not exist: {partition_dir}"
            )
        return explicit_date, partition_dir

    partitions = sorted([p for p in base_dir.glob("date=*") if p.is_dir()], key=lambda p: p.name)
    if not partitions:
        raise LiquidationPressureLoaderError(f"No date partitions found for {dataset_name} under: {base_dir}")

    # Deterministic latest valid partition (lexicographically largest date=YYYY-MM-DD)
    latest = partitions[-1]
    return latest.name.split("=", 1)[-1], latest


def _read_partition_df(partition_dir: Path, dataset_name: str) -> pd.DataFrame:
    if not partition_dir.exists():
        raise LiquidationPressureLoaderError(f"Missing parquet partition for {dataset_name}: {partition_dir}")

    try:
        df = pd.read_parquet(partition_dir)
    except Exception as exc:
        raise LiquidationPressureLoaderError(
            f"Failed reading parquet for {dataset_name} at {partition_dir}: {exc}"
        ) from exc

    if df.empty:
        raise LiquidationPressureLoaderError(f"Parquet for {dataset_name} is empty at {partition_dir}")

    return df


def _validate_tick_like_df(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    required_cols = ["meta__timestamp", "meta__sequence_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise LiquidationPressureLoaderError(
            f"Invalid schema for {dataset_name}: missing columns {missing}"
        )

    validate_basic_tick_schema(df, required_cols)

    out = df.copy(deep=True)
    out["meta__timestamp"] = pd.to_datetime(out["meta__timestamp"], utc=True).astype("datetime64[ns, UTC]")
    out["meta__sequence_id"] = pd.to_numeric(out["meta__sequence_id"], errors="coerce")
    if out["meta__sequence_id"].isna().any():
        raise LiquidationPressureLoaderError(f"Invalid schema for {dataset_name}: non-numeric meta__sequence_id")
    out["meta__sequence_id"] = out["meta__sequence_id"].astype("int64")

    tuple_idx = pd.MultiIndex.from_arrays([out["meta__timestamp"], out["meta__sequence_id"]])
    if not tuple_idx.is_monotonic_increasing:
        raise LiquidationPressureLoaderError(
            f"Invalid ordering for {dataset_name}: (timestamp, sequence_id) not monotonic"
        )

    return out


def _validate_oi_df(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "meta__timestamp",
        "meta__sequence_id",
        "fut__open_interest",
        "fut__oi_velocity",
        "fut__oi_zscore",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise LiquidationPressureLoaderError(f"Invalid schema for open_interest: missing columns {missing}")

    out = _validate_tick_like_df(df, "open_interest")

    for col in ["fut__open_interest", "fut__oi_velocity", "fut__oi_zscore"]:
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.isna().any():
            raise LiquidationPressureLoaderError(f"Invalid schema for open_interest: {col} has non-numeric/null values")

    return out


def _validate_funding_df(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "meta__timestamp",
        "meta__sequence_id",
        "fut__funding_rate_zscore",
        "fut__funding_oi_stress",
        "fut__funding_pressure_index",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise LiquidationPressureLoaderError(f"Invalid schema for funding_rate: missing columns {missing}")

    out = _validate_tick_like_df(df, "funding_rate")

    for col in ["fut__funding_rate_zscore", "fut__funding_oi_stress", "fut__funding_pressure_index"]:
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.isna().any():
            raise LiquidationPressureLoaderError(f"Invalid schema for funding_rate: {col} has non-numeric/null values")

    return out


def load_source_parquets(partition_date: Optional[str] = None) -> Dict[str, object]:
    """Load trades/orderflow/OI/funding source parquet dependencies."""

    cfg = load_config()
    paths_cfg = cfg.get("paths", {})

    base_path = Path(paths_cfg["base"])
    trades_base = base_path / Path(paths_cfg["ticks_trades"])
    orderflow_base = base_path / Path(paths_cfg["ticks_orderflow"])
    oi_base = base_path / Path(paths_cfg["fut_open_interest"])
    funding_base = base_path / Path(paths_cfg["fut_funding_rate"])

    trades_date, trades_partition = _resolve_partition_dir(trades_base, partition_date, "trades")
    orderflow_date, orderflow_partition = _resolve_partition_dir(orderflow_base, partition_date, "orderflow")
    oi_date, oi_partition = _resolve_partition_dir(oi_base, partition_date, "open_interest")
    funding_date, funding_partition = _resolve_partition_dir(funding_base, partition_date, "funding_rate")

    if not (trades_date == orderflow_date == oi_date == funding_date):
        raise LiquidationPressureLoaderError(
            "Partition mismatch between sources: "
            f"trades={trades_date}, orderflow={orderflow_date}, open_interest={oi_date}, funding_rate={funding_date}"
        )

    trades_df = _validate_tick_like_df(_read_partition_df(trades_partition, "trades"), "trades")
    orderflow_df = _validate_tick_like_df(_read_partition_df(orderflow_partition, "orderflow"), "orderflow")
    oi_df = _validate_oi_df(_read_partition_df(oi_partition, "open_interest"))
    funding_df = _validate_funding_df(_read_partition_df(funding_partition, "funding_rate"))

    return SourceLoadResult(
        partition_date=trades_date,
        trades_path=str(trades_partition),
        orderflow_path=str(orderflow_partition),
        oi_path=str(oi_partition),
        funding_path=str(funding_partition),
        trades_df=trades_df,
        orderflow_df=orderflow_df,
        oi_df=oi_df,
        funding_df=funding_df,
    ).to_dict()
