"""Loader for Futures Volume Flow dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from synthetic_data_generator.engine.config.loader import load_config
from synthetic_data_generator.engine.utils.schema_validator import validate_basic_tick_schema


class VolumeFlowLoaderError(Exception):
    """Raised when required source parquet dependencies are missing or invalid."""


@dataclass(frozen=True)
class SourceLoadResult:
    partition_date: str
    trades_path: str
    orderflow_path: str
    trades_df: pd.DataFrame
    orderflow_df: pd.DataFrame

    def to_dict(self) -> Dict[str, object]:
        return {
            "partition_date": self.partition_date,
            "trades_path": self.trades_path,
            "orderflow_path": self.orderflow_path,
            "trades_df": self.trades_df,
            "orderflow_df": self.orderflow_df,
        }


def _resolve_partition_dir(base_dir: Path, explicit_date: Optional[str], dataset_name: str) -> Tuple[str, Path]:
    if explicit_date:
        partition_dir = base_dir / f"date={explicit_date}"
        if not partition_dir.exists():
            raise VolumeFlowLoaderError(
                f"Requested {dataset_name} partition does not exist: {partition_dir}"
            )
        return explicit_date, partition_dir

    candidates = sorted([p for p in base_dir.glob("date=*") if p.is_dir()], key=lambda p: p.name)
    if not candidates:
        raise VolumeFlowLoaderError(f"No date partitions found for {dataset_name} under: {base_dir}")

    # Deterministic latest valid partition only.
    for partition in reversed(candidates):
        try:
            df = pd.read_parquet(partition)
        except Exception:
            continue
        if not df.empty:
            return partition.name.split("=", 1)[-1], partition

    raise VolumeFlowLoaderError(
        f"No readable non-empty date partitions found for {dataset_name} under: {base_dir}"
    )


def _read_partition_df(partition_dir: Path, dataset_name: str) -> pd.DataFrame:
    if not partition_dir.exists():
        raise VolumeFlowLoaderError(f"Missing parquet partition for {dataset_name}: {partition_dir}")

    try:
        df = pd.read_parquet(partition_dir)
    except Exception as exc:
        raise VolumeFlowLoaderError(
            f"Failed reading parquet for {dataset_name} at {partition_dir}: {exc}"
        ) from exc

    if df.empty:
        raise VolumeFlowLoaderError(f"Parquet for {dataset_name} is empty at {partition_dir}")

    return df


def _ensure_required_with_aliases(df: pd.DataFrame, required_to_aliases: Dict[str, tuple[str, ...]], dataset: str) -> pd.DataFrame:
    out = df.copy(deep=True)
    for canonical, aliases in required_to_aliases.items():
        if canonical in out.columns:
            continue

        matched = next((a for a in aliases if a in out.columns), None)
        if matched is None:
            options = ", ".join((canonical, *aliases))
            raise VolumeFlowLoaderError(
                f"Invalid schema for {dataset}: missing required field '{canonical}' (accepted columns: {options})"
            )
        out[canonical] = out[matched]

    return out


def _validate_tick_df(df: pd.DataFrame, dataset_name: str, required_extra: Dict[str, tuple[str, ...]]) -> pd.DataFrame:
    required_cols = ["meta__timestamp", "meta__sequence_id"]
    validate_basic_tick_schema(df, required_cols)

    validated = _ensure_required_with_aliases(df, required_extra, dataset_name)
    validated["meta__timestamp"] = pd.to_datetime(validated["meta__timestamp"], utc=True)
    validated["meta__sequence_id"] = pd.to_numeric(validated["meta__sequence_id"], errors="raise").astype("int64")

    if not validated["meta__timestamp"].is_monotonic_increasing:
        raise VolumeFlowLoaderError(
            f"Invalid schema for {dataset_name}: meta__timestamp is non-monotonic"
        )

    tuple_idx = pd.MultiIndex.from_arrays(
        [validated["meta__timestamp"], validated["meta__sequence_id"]]
    )
    if not tuple_idx.is_monotonic_increasing:
        raise VolumeFlowLoaderError(
            f"Invalid ordering for {dataset_name}: (timestamp, sequence_id) not monotonic"
        )

    return validated


def load_source_parquets(partition_date: Optional[str] = None) -> Dict[str, object]:
    """Load and validate trades and orderflow parquets used by volume-flow features."""

    cfg = load_config()
    paths_cfg = cfg.get("paths", {})

    base_path = Path(paths_cfg["base"])
    trades_base = base_path / Path(paths_cfg["ticks_trades"])
    orderflow_base = base_path / Path(paths_cfg["ticks_orderflow"])

    trades_date, trades_partition = _resolve_partition_dir(trades_base, partition_date, "trades")
    orderflow_date, orderflow_partition = _resolve_partition_dir(orderflow_base, partition_date, "orderflow")

    if trades_date != orderflow_date:
        raise VolumeFlowLoaderError(
            "Partition mismatch between sources: "
            f"trades={trades_date}, orderflow={orderflow_date}"
        )

    trades_required = {
        "trade_size": ("size",),
        "trade_price": ("price",),
        "aggressor_side": ("aggressor",),
    }
    orderflow_required = {
        "event_type": tuple(),
        "event_size": ("size",),
        "aggressor_side": ("aggressor",),
    }

    trades_df = _validate_tick_df(_read_partition_df(trades_partition, "trades"), "trades", trades_required)
    orderflow_df = _validate_tick_df(_read_partition_df(orderflow_partition, "orderflow"), "orderflow", orderflow_required)

    return SourceLoadResult(
        partition_date=trades_date,
        trades_path=str(trades_partition),
        orderflow_path=str(orderflow_partition),
        trades_df=trades_df,
        orderflow_df=orderflow_df,
    ).to_dict()
