"""Computation of Meta / QC / Provenance base features.

This family is strictly additive: it reads the fully computed feature
DataFrame and appends meta__* observability fields without mutating any
existing feature columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


META_FEATURE_NAMES: Sequence[str] = (
    "meta__timestamp",
    "meta__sequence_id",
    "meta__source_feed",
    "meta__data_latency_ms",
    "meta__feature_confidence",
    "meta__staleness_flag",
    "meta__ingest_quality_score",
    "meta__feature_mask_count",
)

ALLOWED_SOURCE_FEEDS = {"L1", "L2", "L3", "OPTIONS", "OTHER", "UNKNOWN"}


@dataclass(frozen=True)
class MetaQCConfig:
    """Configuration for meta / QC computation thresholds."""

    ingest_timestamp_column: str = "ingest_timestamp"
    source_timestamp_column: str = "source_timestamp"
    latency_large_ms: float = 1_000_000_000.0
    staleness_threshold_ms: float = 1_000.0
    null_ratio_penalty_weight: float = 0.5
    range_violation_penalty_weight: float = 0.3
    staleness_penalty_weight: float = 0.2
    mask_count_columns: Optional[Sequence[str]] = None


def _select_timestamp(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            series = pd.to_datetime(df[column], errors="coerce", utc=True)
            if series.notna().any():
                return series
    return pd.Series(pd.NaT, index=df.index)


def _compute_latency_ms(df: pd.DataFrame, config: MetaQCConfig) -> pd.Series:
    source_ts = _select_timestamp(
        df,
        (
            config.source_timestamp_column,
            "meta__timestamp",
            "timestamp",
            "ts",
        ),
    )
    ingest_ts = _select_timestamp(df, (config.ingest_timestamp_column,))
    latency = (ingest_ts - source_ts).dt.total_seconds() * 1000
    latency = latency.clip(lower=0)
    return latency.fillna(config.latency_large_ms).astype("float32")


def _compute_sequence_id(df: pd.DataFrame) -> pd.Series:
    if "meta__sequence_id" in df.columns:
        existing = pd.to_numeric(df["meta__sequence_id"], errors="coerce")
        return existing.fillna(method="ffill").fillna(0).astype("int32")
    return pd.Series(range(len(df)), index=df.index, dtype="int32")


def _normalize_source_feed(df: pd.DataFrame) -> pd.Series:
    if "meta__source_feed" in df.columns:
        raw = df["meta__source_feed"].astype(str)
    elif "source_feed" in df.columns:
        raw = df["source_feed"].astype(str)
    else:
        raw = pd.Series(["UNKNOWN"] * len(df), index=df.index)
    normalized = raw.where(raw.isin(ALLOWED_SOURCE_FEEDS), other=raw)
    normalized = normalized.fillna("UNKNOWN")
    return normalized.astype("string")


def _compute_mask_count(df: pd.DataFrame, config: MetaQCConfig) -> pd.Series:
    columns = list(config.mask_count_columns) if config.mask_count_columns else [
        col for col in df.columns if not col.startswith("meta__")
    ]
    mask_df = df.loc[:, columns] if columns else df
    return mask_df.isna().sum(axis=1).astype("int32")


def _range_violation_ratio(df: pd.DataFrame) -> pd.Series:
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return pd.Series(0.0, index=df.index, dtype="float32")
    non_finite = ~np.isfinite(numeric)
    extreme = numeric.abs() > 1e12
    violations = (non_finite | extreme).sum(axis=1)
    ratio = violations / max(1, numeric.shape[1])
    return ratio.astype("float32")


def _compute_confidence_scores(
    df: pd.DataFrame,
    latency_ms: pd.Series,
    config: MetaQCConfig,
) -> tuple[pd.Series, pd.Series]:
    column_count = max(1, df.shape[1])
    null_ratio = df.isna().sum(axis=1) / column_count
    range_ratio = _range_violation_ratio(df)
    staleness_penalty = (latency_ms > config.staleness_threshold_ms).astype(float)

    raw_confidence = (
        1.0
        - config.null_ratio_penalty_weight * null_ratio
        - config.range_violation_penalty_weight * range_ratio
        - config.staleness_penalty_weight * staleness_penalty
    )
    feature_confidence = raw_confidence.clip(0.0, 1.0).astype("float32")

    quality_penalty = (null_ratio + range_ratio).clip(0.0, 1.0)
    ingest_quality_score = (1.0 - quality_penalty).clip(0.0, 1.0).astype("float32")

    return feature_confidence, ingest_quality_score


def compute_meta_qc_provenance_features(
    df: pd.DataFrame,
    *,
    config: Optional[MetaQCConfig] = None,
) -> pd.DataFrame:
    """Append the eight Meta / QC / Provenance features to the provided frame.

    Args:
        df: Input DataFrame containing fully computed base features.
        config: Optional configuration overriding defaults.

    Returns:
        A new DataFrame containing the original columns plus meta__* fields.
    """

    cfg = config or MetaQCConfig()
    output = df.copy()

    latency_ms = _compute_latency_ms(output, cfg)
    feature_confidence, ingest_quality_score = _compute_confidence_scores(output, latency_ms, cfg)

    timestamp_series = _select_timestamp(
        output,
        (
            "meta__timestamp",
            cfg.source_timestamp_column,
            "timestamp",
            "ts",
        ),
    )
    timestamp_strings = timestamp_series.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    output["meta__timestamp"] = timestamp_strings.where(timestamp_series.notna(), other=pd.NA).astype("string")

    output["meta__sequence_id"] = _compute_sequence_id(output)
    output["meta__source_feed"] = _normalize_source_feed(output)
    output["meta__data_latency_ms"] = latency_ms
    output["meta__feature_confidence"] = feature_confidence
    output["meta__staleness_flag"] = (latency_ms > cfg.staleness_threshold_ms).fillna(False).astype(bool)
    output["meta__ingest_quality_score"] = ingest_quality_score
    output["meta__feature_mask_count"] = _compute_mask_count(output, cfg)

    return output


__all__ = ["compute_meta_qc_provenance_features", "MetaQCConfig", "META_FEATURE_NAMES"]
