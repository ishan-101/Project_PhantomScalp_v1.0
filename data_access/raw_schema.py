"""Single source of truth for raw feed schemas.

This module defines the expected shape of upstream parquet deliveries. It does
not perform IO, validation, or casting. Downstream components use these
contracts to enforce determinism and fail fast when data deviates from the
specification.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FieldSpec:
    """Defines a single column in a feed.

    Attributes:
        name: Canonical column name as produced by upstream.
        dtype: Pandas-readable dtype string representing the expected in-memory
            dtype after loading.
        required: Whether the column must be present and non-null at load time.
        description: Optional human-readable intent for operators.
    """

    name: str
    dtype: str
    required: bool
    description: Optional[str] = None


@dataclass(frozen=True)
class FeedSchema:
    """Defines the full schema for a feed."""

    feed_name: str
    timestamp_column: str
    fields: List[FieldSpec]
    allow_exact_duplicates: bool = False
    description: Optional[str] = None

    @property
    def required_fields(self) -> List[FieldSpec]:
        """Returns the subset of required fields.

        Note: This is simple attribute access; no validation or IO is performed
        in this module.
        """

        return [field for field in self.fields if field.required]

    @property
    def optional_fields(self) -> List[FieldSpec]:
        """Returns the subset of optional fields."""

        return [field for field in self.fields if not field.required]


# Feed: Trades (tick-level executions)
TRADES_SCHEMA = FeedSchema(
    feed_name="trades",
    timestamp_column="ts",
    description="Tick-level executions emitted by upstream capture systems.",
    allow_exact_duplicates=False,
    fields=[
        FieldSpec(
            name="ts",
            dtype="datetime64[ns]",
            required=True,
            description="Event timestamp in nanoseconds UTC.",
        ),
        FieldSpec(
            name="symbol",
            dtype="string",
            required=True,
            description="Canonical instrument identifier (e.g., ticker).",
        ),
        FieldSpec(
            name="price",
            dtype="float64",
            required=True,
            description="Execution price in quote currency units.",
        ),
        FieldSpec(
            name="size",
            dtype="float64",
            required=True,
            description="Executed quantity in base units.",
        ),
        FieldSpec(
            name="side",
            dtype="string",
            required=False,
            description="Aggressor side when provided by upstream (e.g., 'buy'/'sell').",
        ),
        FieldSpec(
            name="venue",
            dtype="string",
            required=False,
            description="Execution venue or source identifier when available.",
        ),
    ],
)

# Feed: Order book depth snapshots
ORDERBOOK_SNAPSHOT_SCHEMA = FeedSchema(
    feed_name="orderbook_snapshots",
    timestamp_column="ts",
    description="Top-of-book snapshots emitted at fixed cadence by upstream.",
    allow_exact_duplicates=False,
    fields=[
        FieldSpec(
            name="ts",
            dtype="datetime64[ns]",
            required=True,
            description="Snapshot timestamp in nanoseconds UTC.",
        ),
        FieldSpec(
            name="symbol",
            dtype="string",
            required=True,
            description="Canonical instrument identifier (e.g., ticker).",
        ),
        FieldSpec(
            name="bid_price",
            dtype="float64",
            required=True,
            description="Best bid price.",
        ),
        FieldSpec(
            name="bid_size",
            dtype="float64",
            required=True,
            description="Aggregate size at best bid.",
        ),
        FieldSpec(
            name="ask_price",
            dtype="float64",
            required=True,
            description="Best ask price.",
        ),
        FieldSpec(
            name="ask_size",
            dtype="float64",
            required=True,
            description="Aggregate size at best ask.",
        ),
        FieldSpec(
            name="midpoint",
            dtype="float64",
            required=False,
            description="Optional midpoint provided by upstream if pre-computed.",
        ),
    ],
)

# Mapping of feed name to schema for easy lookup by consumers.
RAW_SCHEMAS: Dict[str, FeedSchema] = {
    TRADES_SCHEMA.feed_name: TRADES_SCHEMA,
    ORDERBOOK_SNAPSHOT_SCHEMA.feed_name: ORDERBOOK_SNAPSHOT_SCHEMA,
}
