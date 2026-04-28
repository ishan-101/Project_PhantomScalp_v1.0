"""Loader for Futures Basis Structure source dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from synthetic_data_generator.engine.config.loader import load_config
from synthetic_data_generator.engine.utils.schema_validator import validate_basic_tick_schema


class BasisStructureLoaderError(Exception):
    """Raised when basis source parquet dependencies are missing or invalid."""


@dataclass(frozen=True)
class SourceLoadResult:
    partition_date: str
    trades_path: str
    orderflow_path: str
    funding_path: str
    trades_df: pd.DataFrame
    orderflow_df: pd.DataFrame
    funding_df: pd.DataFrame

    def to_dict(self) -> Dict[str, object]:
        return {
            "partition_date": self.partition_date,
            "trades_path": self.trades_path,
            "orderflow_path": self.orderflow_path,
            "funding_path": self.funding_path,
            "trades_df": self.trades_df,
            "orderflow_df": self.orderflow_df,
            "funding_df": self.funding_df,
        }


def _resolve_partition_dir(base_dir: Path, explicit_date: Optional[str], dataset_name: str) -> tuple[str, Path]:
    if explicit_date:
        part = base_dir / f"date={explicit_date}"
        if not part.exists():
            raise BasisStructureLoaderError(f"Requested {dataset_name} partition does not exist: {part}")
        return explicit_date, part

    partitions = sorted([p for p in base_dir.glob("date=*") if p.is_dir()], key=lambda p: p.name)
    if not partitions:
        raise BasisStructureLoaderError(f"No date partitions found for {dataset_name} under: {base_dir}")
    latest = partitions[-1]
    return latest.name.split("=", 1)[-1], latest


def _read_non_empty_parquet(partition_dir: Path, dataset_name: str) -> pd.DataFrame:
    try:
        df = pd.read_parquet(partition_dir)
    except Exception as exc:
        raise BasisStructureLoaderError(
            f"Failed reading parquet for {dataset_name} at {partition_dir}: {exc}"
        ) from exc

    if df.empty:
        raise BasisStructureLoaderError(f"Parquet for {dataset_name} is empty at {partition_dir}")
    return df


def _validate_tick_df(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    required = ["meta__timestamp", "meta__sequence_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise BasisStructureLoaderError(f"Invalid schema for {dataset_name}: missing columns {missing}")

    validate_basic_tick_schema(df, required)

    out = df.copy(deep=True)
    out["meta__timestamp"] = pd.to_datetime(out["meta__timestamp"], utc=True).astype("datetime64[ns, UTC]")
    out["meta__sequence_id"] = pd.to_numeric(out["meta__sequence_id"], errors="coerce").astype("int64")
    return out


def _validate_funding_df(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "meta__timestamp",
        "meta__sequence_id",
        "fut__funding_rate",
        "fut__funding_rate_zscore",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise BasisStructureLoaderError(f"Invalid schema for funding_rate: missing columns {missing}")

    out = df.copy(deep=True)
    out["meta__timestamp"] = pd.to_datetime(out["meta__timestamp"], utc=True).astype("datetime64[ns, UTC]")
    out["meta__sequence_id"] = pd.to_numeric(out["meta__sequence_id"], errors="coerce").astype("int64")

    # Canonical aliases expected by basis structure design; fail fast if neither canonical nor deterministic fallback exists.
    if "fut__funding_pressure_index" not in out.columns:
        if "fut__predicted_funding_shift" in out.columns:
            out["fut__funding_pressure_index"] = pd.to_numeric(out["fut__predicted_funding_shift"], errors="coerce")
        else:
            raise BasisStructureLoaderError(
                "Funding dependency missing fut__funding_pressure_index (or fallback fut__predicted_funding_shift)"
            )

    if "fut__funding_oi_stress" not in out.columns:
        if "fut__funding_stress_score" in out.columns:
            out["fut__funding_oi_stress"] = pd.to_numeric(out["fut__funding_stress_score"], errors="coerce")
        else:
            raise BasisStructureLoaderError(
                "Funding dependency missing fut__funding_oi_stress (or fallback fut__funding_stress_score)"
            )

    needed = [
        "meta__timestamp",
        "meta__sequence_id",
        "fut__funding_rate",
        "fut__funding_rate_zscore",
        "fut__funding_pressure_index",
        "fut__funding_oi_stress",
    ]
    if out[needed].isna().any().any():
        bad = {c: int(out[c].isna().sum()) for c in needed if int(out[c].isna().sum()) > 0}
        raise BasisStructureLoaderError(f"Funding schema has critical nulls: {bad}")

    return out


def load_source_parquets(partition_date: Optional[str] = None) -> Dict[str, object]:
    cfg = load_config()
    paths_cfg = cfg.get("paths", {})

    base_path = Path(paths_cfg["base"])
    trades_base = base_path / Path(paths_cfg["ticks_trades"])
    orderflow_base = base_path / Path(paths_cfg["ticks_orderflow"])
    funding_base = base_path / Path(paths_cfg["fut_funding_rate"])

    trades_date, trades_part = _resolve_partition_dir(trades_base, partition_date, "trades")
    orderflow_date, orderflow_part = _resolve_partition_dir(orderflow_base, partition_date, "orderflow")
    funding_date, funding_part = _resolve_partition_dir(funding_base, partition_date, "funding_rate")

    if not (trades_date == orderflow_date == funding_date):
        raise BasisStructureLoaderError(
            "Partition mismatch between sources: "
            f"trades={trades_date}, orderflow={orderflow_date}, funding={funding_date}"
        )

    return SourceLoadResult(
        partition_date=trades_date,
        trades_path=str(trades_part),
        orderflow_path=str(orderflow_part),
        funding_path=str(funding_part),
        trades_df=_validate_tick_df(_read_non_empty_parquet(trades_part, "trades"), "trades"),
        orderflow_df=_validate_tick_df(_read_non_empty_parquet(orderflow_part, "orderflow"), "orderflow"),
        funding_df=_validate_funding_df(_read_non_empty_parquet(funding_part, "funding_rate")),
    ).to_dict()
