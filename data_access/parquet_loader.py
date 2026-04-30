"""Parquet loading utilities.

This module only reads parquet files from disk into pandas DataFrames. It does
not mutate, validate, or align data. Any missing file or IO issue is surfaced as
an explicit error.
"""

from pathlib import Path
from typing import Union

import pandas as pd


PathLike = Union[str, Path]


def load_parquet(path: PathLike) -> pd.DataFrame:
    """Load a parquet file from disk into a DataFrame.

    Args:
        path: File system path to the parquet file.

    Returns:
        DataFrame containing the file contents.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        OSError: For underlying IO issues raised by pandas.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")

    # No mutation or schema logic occurs here; consumers are responsible for
    # downstream enforcement.
    return pd.read_parquet(file_path)
