# Data Access Layer (v0.1)

## Purpose
The Data Access Layer is a narrow, deterministic boundary between raw parquet feeds and downstream feature engineering. Its sole responsibility is to load parquet files and produce in-memory tables that strictly match documented schemas. It is infrastructure glue, not analytics logic.

## What this layer does
- Define canonical raw schemas for every feed, including column names, expected dtypes, and required/optional fields.
- Load parquet files from disk into pandas DataFrames without mutation.
- Enforce exact dtypes specified by the raw schema without silent coercion.
- Align tables on timestamp order and detect duplicate rows when requested.
- Validate structure and basic sanity constraints before any feature code runs.
- Fail fast with explicit errors when input data violates expectations.

## What this layer does **not** do
- No feature engineering, labeling, or model-aware logic.
- No rolling windows, smoothing, interpolation, resampling, or data repair.
- No missing-value filling or inference of absent fields.
- No exchange adapters, real-time ingestion, or streaming concerns.

## Strict assumptions
- Upstream producers deliver parquet files that already contain all required fields with correct logical meaning.
- Timestamps are authoritative and must be monotonic when sorted; non-monotonicity is an error.
- Any deviation from the documented schema is treated as a hard failure, not something to fix implicitly.

## Failure philosophy
- Prefer explicit, human-readable errors over silent fixes.
- Reject incompatible dtypes instead of coercing them.
- Stop processing immediately when violations are detected; downstream consumers should never guess.

## Module boundaries
The Data Access Layer never references feature names, labels, or ML models. It only outputs clean, schema-aligned DataFrames for later stages to consume.
