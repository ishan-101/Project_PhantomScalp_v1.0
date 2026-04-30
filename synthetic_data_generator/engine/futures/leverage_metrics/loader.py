"""Loader for Futures Leverage Metrics dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from synthetic_data_generator.engine.config.loader import load_config


class LeverageMetricsLoaderError(Exception):
    """Raised when leverage-metrics source dependencies are missing or invalid."""


@dataclass(frozen=True)
class SourceLoadResult:
    partition_date: str
    oi_path: str
    funding_path: str
    basis_path: str
    volume_flow_path: str
    oi_df: pd.DataFrame
    funding_df: pd.DataFrame
    basis_df: pd.DataFrame
    volume_flow_df: pd.DataFrame

    def to_dict(self) -> Dict[str, object]:
        return {
            "partition_date": self.partition_date,
            "oi_path": self.oi_path,
            "funding_path": self.funding_path,
            "basis_path": self.basis_path,
            "volume_flow_path": self.volume_flow_path,
            "oi_df": self.oi_df,
            "funding_df": self.funding_df,
            "basis_df": self.basis_df,
            "volume_flow_df": self.volume_flow_df,
        }


def _resolve_partition_dir(base_dir: Path, explicit_date: Optional[str], dataset_name: str) -> tuple[str, Path]:
    if explicit_date:
        partition_dir = base_dir / f"date={explicit_date}"
        if not partition_dir.exists():
            raise LeverageMetricsLoaderError(
                f"Requested {dataset_name} partition does not exist: {partition_dir}"
            )
        return explicit_date, partition_dir

    partitions = sorted([p for p in base_dir.glob("date=*") if p.is_dir()], key=lambda p: p.name)
    if not partitions:
        raise LeverageMetricsLoaderError(f"No date partitions found for {dataset_name} under: {base_dir}")

    latest = partitions[-1]
    return latest.name.split("=", 1)[-1], latest


def _read_non_empty_parquet(partition_dir: Path, dataset_name: str) -> pd.DataFrame:
    if not partition_dir.exists():
        raise LeverageMetricsLoaderError(f"Missing parquet partition for {dataset_name}: {partition_dir}")

    try:
        df = pd.read_parquet(partition_dir)
    except Exception as exc:
        raise LeverageMetricsLoaderError(
            f"Failed reading parquet for {dataset_name} at {partition_dir}: {exc}"
        ) from exc

    if df.empty:
        raise LeverageMetricsLoaderError(f"Parquet for {dataset_name} is empty at {partition_dir}")

    return df


def _validate_base(df: pd.DataFrame, dataset_name: str, required_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise LeverageMetricsLoaderError(f"Invalid schema for {dataset_name}: missing columns {missing}")

    validated = df[required_cols].copy(deep=True)
    validated["meta__timestamp"] = pd.to_datetime(validated["meta__timestamp"], utc=True, errors="coerce")
    validated["meta__sequence_id"] = pd.to_numeric(validated["meta__sequence_id"], errors="coerce")

    if validated["meta__timestamp"].isna().any():
        raise LeverageMetricsLoaderError(f"{dataset_name}: invalid meta__timestamp values detected")
    if validated["meta__sequence_id"].isna().any():
        raise LeverageMetricsLoaderError(f"{dataset_name}: invalid meta__sequence_id values detected")

    validated["meta__timestamp"] = validated["meta__timestamp"].astype("datetime64[ns, UTC]")
    validated["meta__sequence_id"] = validated["meta__sequence_id"].astype("int64")

    if not validated["meta__timestamp"].is_monotonic_increasing:
        raise LeverageMetricsLoaderError(f"{dataset_name}: meta__timestamp is non-monotonic")
    if not validated["meta__sequence_id"].is_monotonic_increasing:
        raise LeverageMetricsLoaderError(f"{dataset_name}: meta__sequence_id is non-monotonic")

    tuple_idx = pd.MultiIndex.from_arrays([validated["meta__timestamp"], validated["meta__sequence_id"]])
    if not tuple_idx.is_monotonic_increasing:
        raise LeverageMetricsLoaderError(
            f"Invalid ordering for {dataset_name}: (meta__timestamp, meta__sequence_id) not monotonic"
        )

    return validated


def load_source_parquets(partition_date: Optional[str] = None) -> Dict[str, object]:
    """Load and validate open-interest, funding, basis, and volume-flow dependencies."""

    cfg = load_config()
    paths_cfg = cfg.get("paths", {})

    base_path = Path(paths_cfg["base"])
    oi_base = base_path / Path(paths_cfg["fut_open_interest"])
    funding_base = base_path / Path(paths_cfg["fut_funding_rate"])
    basis_base = base_path / Path(paths_cfg["fut_basis_structure"])
    volume_flow_base = base_path / Path(paths_cfg["fut_volume_flow"])

    oi_date, oi_partition = _resolve_partition_dir(oi_base, partition_date, "open_interest")
    funding_date, funding_partition = _resolve_partition_dir(funding_base, partition_date, "funding_rate")
    basis_date, basis_partition = _resolve_partition_dir(basis_base, partition_date, "basis_structure")
    volume_flow_date, volume_flow_partition = _resolve_partition_dir(volume_flow_base, partition_date, "volume_flow")

    if not (oi_date == funding_date == basis_date == volume_flow_date):
        raise LeverageMetricsLoaderError(
            "Partition mismatch between sources: "
            f"open_interest={oi_date}, funding_rate={funding_date}, basis_structure={basis_date}, volume_flow={volume_flow_date}"
        )

    oi_df = _validate_base(
        _read_non_empty_parquet(oi_partition, "open_interest"),
        "open_interest",
        ["meta__timestamp", "meta__sequence_id", "fut__oi_zscore"],
    )
    funding_df = _validate_base(
        _read_non_empty_parquet(funding_partition, "funding_rate"),
        "funding_rate",
        ["meta__timestamp", "meta__sequence_id", "fut__funding_rate_zscore", "fut__funding_oi_stress"],
    )
    basis_df = _validate_base(
        _read_non_empty_parquet(basis_partition, "basis_structure"),
        "basis_structure",
        ["meta__timestamp", "meta__sequence_id", "fut__basis_zscore"],
    )
    volume_flow_df = _validate_base(
        _read_non_empty_parquet(volume_flow_partition, "volume_flow"),
        "volume_flow",
        ["meta__timestamp", "meta__sequence_id", "fut__volume_delta_ratio"],
    )

    return SourceLoadResult(
        partition_date=oi_date,
        oi_path=str(oi_partition),
        funding_path=str(funding_partition),
        basis_path=str(basis_partition),
        volume_flow_path=str(volume_flow_partition),
        oi_df=oi_df,
        funding_df=funding_df,
        basis_df=basis_df,
        volume_flow_df=volume_flow_df,
    ).to_dict()
