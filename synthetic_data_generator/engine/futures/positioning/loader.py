"""Loader for Futures Positioning dependencies (open interest + funding rate)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from synthetic_data_generator.engine.config.loader import load_config


class PositioningLoaderError(Exception):
    """Raised when required parquet dependencies are missing or invalid."""


@dataclass(frozen=True)
class SourceLoadResult:
    partition_date: str
    oi_path: str
    funding_path: str
    oi_df: pd.DataFrame
    funding_df: pd.DataFrame

    def to_dict(self) -> Dict[str, object]:
        return {
            "partition_date": self.partition_date,
            "oi_path": self.oi_path,
            "funding_path": self.funding_path,
            "oi_df": self.oi_df,
            "funding_df": self.funding_df,
        }


def _resolve_partition_dir(base_dir: Path, explicit_date: Optional[str], dataset_name: str) -> tuple[str, Path]:
    if explicit_date:
        partition_dir = base_dir / f"date={explicit_date}"
        if not partition_dir.exists():
            raise PositioningLoaderError(f"Requested {dataset_name} partition not found: {partition_dir}")
        return explicit_date, partition_dir

    partitions = sorted([p for p in base_dir.glob("date=*") if p.is_dir()], key=lambda p: p.name)
    if not partitions:
        raise PositioningLoaderError(f"No date partitions found for {dataset_name} under: {base_dir}")

    latest = partitions[-1]
    return latest.name.split("=", 1)[-1], latest


def _read_partition_df(partition_dir: Path, dataset_name: str) -> pd.DataFrame:
    try:
        df = pd.read_parquet(partition_dir)
    except Exception as exc:
        raise PositioningLoaderError(f"Failed reading {dataset_name} parquet at {partition_dir}: {exc}") from exc

    if df.empty:
        raise PositioningLoaderError(f"{dataset_name} parquet is empty at {partition_dir}")

    return df


def _validate_common(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    required_meta = ["meta__timestamp", "meta__sequence_id"]
    missing = [c for c in required_meta if c not in df.columns]
    if missing:
        raise PositioningLoaderError(f"{dataset_name} missing required meta columns: {missing}")

    validated = df.copy(deep=True)
    validated["meta__timestamp"] = pd.to_datetime(validated["meta__timestamp"], utc=True)

    try:
        validated["meta__sequence_id"] = validated["meta__sequence_id"].astype("int64")
    except Exception as exc:
        raise PositioningLoaderError(f"{dataset_name} meta__sequence_id cannot coerce to int64") from exc

    if validated[required_meta].isna().any().any():
        raise PositioningLoaderError(f"{dataset_name} contains nulls in critical meta columns")

    key_index = pd.MultiIndex.from_arrays([validated["meta__timestamp"], validated["meta__sequence_id"]])
    if not key_index.is_monotonic_increasing:
        validated = validated.sort_values(["meta__timestamp", "meta__sequence_id"], kind="mergesort").reset_index(drop=True)

    return validated


def _validate_oi_df(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "meta__timestamp",
        "meta__sequence_id",
        "fut__open_interest",
        "fut__oi_zscore",
        "fut__oi_change",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise PositioningLoaderError(f"open_interest missing required columns: {missing}")

    validated = _validate_common(df, "open_interest")
    if validated[required].isna().any().any():
        raise PositioningLoaderError("open_interest contains nulls in required positioning dependencies")
    return validated


def _validate_funding_df(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "meta__timestamp",
        "meta__sequence_id",
        "fut__funding_rate",
        "fut__funding_rate_zscore",
        "fut__funding_oi_stress",
        "fut__funding_rate_regime_flag",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise PositioningLoaderError(f"funding_rate missing required columns: {missing}")

    validated = _validate_common(df, "funding_rate")
    if validated[required].isna().any().any():
        raise PositioningLoaderError("funding_rate contains nulls in required positioning dependencies")
    return validated


def load_source_parquets(partition_date: Optional[str] = None) -> Dict[str, object]:
    """Load and validate open-interest + funding-rate dependencies."""

    cfg = load_config()
    paths_cfg = cfg.get("paths", {})
    base_path = Path(paths_cfg["base"])

    oi_base = base_path / Path(paths_cfg["fut_open_interest"])
    funding_base = base_path / Path(paths_cfg["fut_funding_rate"])

    oi_date, oi_part = _resolve_partition_dir(oi_base, partition_date, "open_interest")
    funding_date, funding_part = _resolve_partition_dir(funding_base, partition_date, "funding_rate")

    if oi_date != funding_date:
        raise PositioningLoaderError(
            f"Partition mismatch between open_interest ({oi_date}) and funding_rate ({funding_date})"
        )

    oi_df = _validate_oi_df(_read_partition_df(oi_part, "open_interest"))
    funding_df = _validate_funding_df(_read_partition_df(funding_part, "funding_rate"))

    return SourceLoadResult(
        partition_date=oi_date,
        oi_path=str(oi_part),
        funding_path=str(funding_part),
        oi_df=oi_df,
        funding_df=funding_df,
    ).to_dict()
