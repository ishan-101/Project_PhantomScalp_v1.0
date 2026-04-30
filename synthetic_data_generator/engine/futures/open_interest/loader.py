"""Loader for Futures Open Interest Engine source dependencies.

Loads trades and orderflow parquet outputs from ticks_and_orderflow partitions,
using shared configuration only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from synthetic_data_generator.engine.config.loader import load_config
from synthetic_data_generator.engine.utils.schema_validator import validate_basic_tick_schema


class OpenInterestLoaderError(Exception):
    """Raised when source parquet dependencies are missing or invalid."""


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


def _resolve_partition_dir(base_dir: Path, explicit_date: Optional[str]) -> tuple[str, Path]:
    if explicit_date:
        partition_dir = base_dir / f"date={explicit_date}"
        if not partition_dir.exists():
            raise OpenInterestLoaderError(
                f"Requested partition does not exist: {partition_dir}"
            )
        return explicit_date, partition_dir

    partitions = sorted(
        [p for p in base_dir.glob("date=*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if not partitions:
        raise OpenInterestLoaderError(f"No date partitions found under: {base_dir}")

    latest = partitions[-1]
    date_value = latest.name.split("=", 1)[-1]
    return date_value, latest


def _read_partition_df(partition_dir: Path, dataset_name: str) -> pd.DataFrame:
    if not partition_dir.exists():
        raise OpenInterestLoaderError(
            f"Missing parquet partition for {dataset_name}: {partition_dir}"
        )

    try:
        df = pd.read_parquet(partition_dir)
    except Exception as exc:
        raise OpenInterestLoaderError(
            f"Failed reading parquet for {dataset_name} at {partition_dir}: {exc}"
        ) from exc

    if df.empty:
        raise OpenInterestLoaderError(
            f"Parquet for {dataset_name} is empty at {partition_dir}"
        )

    return df


def _validate_source_df(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    required_cols = ["meta__timestamp", "meta__sequence_id"]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise OpenInterestLoaderError(
            f"Invalid schema for {dataset_name}: missing columns {missing}"
        )

    # Shared validator enforces base tick schema expectations.
    validate_basic_tick_schema(df, required_cols)

    # UTC-safe timestamp normalization to preserve canonical event clock assumptions.
    ts = pd.to_datetime(df["meta__timestamp"], utc=True)
    if not ts.is_monotonic_increasing:
        raise OpenInterestLoaderError(
            f"Invalid schema for {dataset_name}: meta__timestamp is non-monotonic"
        )

    validated = df.copy()
    validated["meta__timestamp"] = ts

    # Deterministic load order guard: verify (timestamp, sequence) tuple is stable monotonic.
    tuple_idx = pd.MultiIndex.from_arrays(
        [validated["meta__timestamp"], validated["meta__sequence_id"]]
    )
    if not tuple_idx.is_monotonic_increasing:
        raise OpenInterestLoaderError(
            f"Invalid ordering for {dataset_name}: (timestamp, sequence_id) not monotonic"
        )

    return validated


def load_source_parquets(partition_date: Optional[str] = None) -> Dict[str, object]:
    """Load and validate trades + orderflow source parquet datasets.

    Args:
        partition_date: optional YYYY-MM-DD partition selector. If omitted,
            latest available partition is selected deterministically.

    Returns:
        Dict containing partition date, resolved paths, and loaded dataframes.
    """

    cfg = load_config()
    paths_cfg = cfg.get("paths", {})

    base_path = Path(paths_cfg["base"])
    trades_base = base_path / Path(paths_cfg["ticks_trades"])
    orderflow_base = base_path / Path(paths_cfg["ticks_orderflow"])

    selected_date, trades_partition = _resolve_partition_dir(trades_base, partition_date)
    orderflow_date, orderflow_partition = _resolve_partition_dir(orderflow_base, partition_date)

    if selected_date != orderflow_date:
        raise OpenInterestLoaderError(
            "Partition mismatch between trades and orderflow: "
            f"trades={selected_date}, orderflow={orderflow_date}"
        )

    trades_df = _validate_source_df(
        _read_partition_df(trades_partition, "trades"),
        "trades",
    )
    orderflow_df = _validate_source_df(
        _read_partition_df(orderflow_partition, "orderflow"),
        "orderflow",
    )

    result = SourceLoadResult(
        partition_date=selected_date,
        trades_path=str(trades_partition),
        orderflow_path=str(orderflow_partition),
        trades_df=trades_df,
        orderflow_df=orderflow_df,
    )
    return result.to_dict()
