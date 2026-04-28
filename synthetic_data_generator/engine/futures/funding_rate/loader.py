"""Loader for Futures Funding Rate dependencies.

Loads trades, orderflow, and futures open_interest parquet outputs using
configuration-resolved paths only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from synthetic_data_generator.engine.config.loader import load_config
from synthetic_data_generator.engine.utils.schema_validator import validate_basic_tick_schema


class FundingRateLoaderError(Exception):
    """Raised when required source parquet dependencies are missing or invalid."""


@dataclass(frozen=True)
class SourceLoadResult:
    partition_date: str
    trades_path: str
    orderflow_path: str
    oi_path: str
    trades_df: pd.DataFrame
    orderflow_df: pd.DataFrame
    oi_df: pd.DataFrame

    def to_dict(self) -> Dict[str, object]:
        return {
            "partition_date": self.partition_date,
            "trades_path": self.trades_path,
            "orderflow_path": self.orderflow_path,
            "oi_path": self.oi_path,
            "trades_df": self.trades_df,
            "orderflow_df": self.orderflow_df,
            "oi_df": self.oi_df,
        }


def _resolve_partition_dir(base_dir: Path, explicit_date: Optional[str], dataset_name: str) -> tuple[str, Path]:
    if explicit_date:
        partition_dir = base_dir / f"date={explicit_date}"
        if not partition_dir.exists():
            raise FundingRateLoaderError(
                f"Requested {dataset_name} partition does not exist: {partition_dir}"
            )
        return explicit_date, partition_dir

    partitions = sorted([p for p in base_dir.glob("date=*") if p.is_dir()], key=lambda p: p.name)
    if not partitions:
        raise FundingRateLoaderError(f"No date partitions found for {dataset_name} under: {base_dir}")

    latest = partitions[-1]
    return latest.name.split("=", 1)[-1], latest


def _read_partition_df(partition_dir: Path, dataset_name: str) -> pd.DataFrame:
    if not partition_dir.exists():
        raise FundingRateLoaderError(f"Missing parquet partition for {dataset_name}: {partition_dir}")

    try:
        df = pd.read_parquet(partition_dir)
    except Exception as exc:
        raise FundingRateLoaderError(
            f"Failed reading parquet for {dataset_name} at {partition_dir}: {exc}"
        ) from exc

    if df.empty:
        raise FundingRateLoaderError(f"Parquet for {dataset_name} is empty at {partition_dir}")

    return df


def _validate_tick_df(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    required_cols = ["meta__timestamp", "meta__sequence_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise FundingRateLoaderError(
            f"Invalid schema for {dataset_name}: missing columns {missing}"
        )

    validate_basic_tick_schema(df, required_cols)

    validated = df.copy(deep=True)
    validated["meta__timestamp"] = pd.to_datetime(validated["meta__timestamp"], utc=True)

    if not validated["meta__timestamp"].is_monotonic_increasing:
        raise FundingRateLoaderError(
            f"Invalid schema for {dataset_name}: meta__timestamp is non-monotonic"
        )

    tuple_idx = pd.MultiIndex.from_arrays(
        [validated["meta__timestamp"], validated["meta__sequence_id"]]
    )
    if not tuple_idx.is_monotonic_increasing:
        raise FundingRateLoaderError(
            f"Invalid ordering for {dataset_name}: (timestamp, sequence_id) not monotonic"
        )

    return validated


def _validate_oi_df(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "meta__timestamp",
        "meta__sequence_id",
        "fut__open_interest",
        "fut__oi_change",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise FundingRateLoaderError(f"Invalid schema for open_interest: missing columns {missing}")

    validated = df.copy(deep=True)
    validated["meta__timestamp"] = pd.to_datetime(validated["meta__timestamp"], utc=True)
    try:
        validated["meta__sequence_id"] = validated["meta__sequence_id"].astype("int64")
    except Exception as exc:
        raise FundingRateLoaderError("open_interest meta__sequence_id cannot be coerced to int64") from exc

    if not validated["meta__timestamp"].is_monotonic_increasing:
        raise FundingRateLoaderError("Invalid schema for open_interest: meta__timestamp is non-monotonic")
    if not validated["meta__sequence_id"].is_monotonic_increasing:
        raise FundingRateLoaderError("Invalid schema for open_interest: meta__sequence_id is non-monotonic")

    if validated[required_cols].isna().any().any():
        counts = {
            c: int(validated[c].isna().sum())
            for c in required_cols
            if int(validated[c].isna().sum()) > 0
        }
        raise FundingRateLoaderError(f"Invalid schema for open_interest: required nulls detected {counts}")

    tuple_idx = pd.MultiIndex.from_arrays(
        [validated["meta__timestamp"], validated["meta__sequence_id"]]
    )
    if not tuple_idx.is_monotonic_increasing:
        raise FundingRateLoaderError(
            "Invalid ordering for open_interest: (timestamp, sequence_id) not monotonic"
        )

    return validated


def load_source_parquets(partition_date: Optional[str] = None) -> Dict[str, object]:
    """Load and validate source dependencies for funding-rate feature generation."""

    cfg = load_config()
    paths_cfg = cfg.get("paths", {})

    base_path = Path(paths_cfg["base"])
    trades_base = base_path / Path(paths_cfg["ticks_trades"])
    orderflow_base = base_path / Path(paths_cfg["ticks_orderflow"])
    oi_base = base_path / Path(paths_cfg["fut_open_interest"])

    trades_date, trades_partition = _resolve_partition_dir(trades_base, partition_date, "trades")
    orderflow_date, orderflow_partition = _resolve_partition_dir(orderflow_base, partition_date, "orderflow")
    oi_date, oi_partition = _resolve_partition_dir(oi_base, partition_date, "open_interest")

    if not (trades_date == orderflow_date == oi_date):
        raise FundingRateLoaderError(
            "Partition mismatch between sources: "
            f"trades={trades_date}, orderflow={orderflow_date}, open_interest={oi_date}"
        )

    trades_df = _validate_tick_df(_read_partition_df(trades_partition, "trades"), "trades")
    orderflow_df = _validate_tick_df(_read_partition_df(orderflow_partition, "orderflow"), "orderflow")
    oi_df = _validate_oi_df(_read_partition_df(oi_partition, "open_interest"))

    return SourceLoadResult(
        partition_date=trades_date,
        trades_path=str(trades_partition),
        orderflow_path=str(orderflow_partition),
        oi_path=str(oi_partition),
        trades_df=trades_df,
        orderflow_df=orderflow_df,
        oi_df=oi_df,
    ).to_dict()
