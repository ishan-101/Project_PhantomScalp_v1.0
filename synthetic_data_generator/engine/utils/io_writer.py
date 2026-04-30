"""
engine/utils/io_writer.py (FIXED VERSION)

Robust ParquetWriter that supports:
 - chunked writes to .part-XXXXX.parquet files
 - finalize(): merge parts into a single canonical parquet file and remove parts
 - streaming merge using pyarrow (low memory)
 - portable pandas fallback
"""

from __future__ import annotations
import os
import glob
from pathlib import Path
from typing import Dict
import logging

import pandas as pd

log = logging.getLogger("engine.utils.io_writer")
log.setLevel(logging.INFO)
if not log.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[io_writer] %(message)s"))
    log.addHandler(ch)

# Try PyArrow
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except Exception:
    _HAS_PYARROW = False
    log.warning("pyarrow not available — fallback to pandas concat (slower & higher memory).")


class ParquetWriter:
    """
    Writes chunked part files until finalize() merges them into one clean parquet.
    """

    def __init__(self, base_path: str | Path, compression: str = "snappy"):
        self.base_path = Path(base_path)
        self._compression = compression

        self._parts_dir = self.base_path.parent
        self._part_prefix = f"{self.base_path.stem}.part-"
        self._part_counter = 0

        self._rows_written = 0
        self._bytes_written = 0

        # ensure directory exists
        self._parts_dir.mkdir(parents=True, exist_ok=True)

    def _part_path(self) -> Path:
        name = f"{self._part_prefix}{self._part_counter:05d}.parquet"
        self._part_counter += 1
        return self._parts_dir / name

    def write(self, df: pd.DataFrame, append: bool = True) -> None:
        """Write a dataframe either as a part-file or canonical file."""
        if append:
            p = self._part_path()
            df.to_parquet(p, index=False, compression=self._compression)
            self._rows_written += len(df)
            self._bytes_written += p.stat().st_size
            log.info(f"wrote part {p.name} — {len(df)} rows")
        else:
            tmp = self.base_path.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, index=False, compression=self._compression)
            tmp.replace(self.base_path)
            self._rows_written = len(df)
            self._bytes_written = self.base_path.stat().st_size
            log.info(f"wrote canonical file {self.base_path} — {self._rows_written} rows")

    def _find_parts(self) -> list[Path]:
        pattern = str(self._parts_dir / f"{self._part_prefix}*.parquet")
        return sorted([Path(p) for p in glob.glob(pattern)])

    def finalize(self) -> None:
        """
        Merge part files into a single file.

        FIXED VERSION:
            - Reads first part → extracts schema
            - Creates ParquetWriter(schema=first.schema)
            - Writes all parts as row groups
            - Atomically replaces final file
        """
        parts = self._find_parts()
        if not parts:
            # nothing to merge
            if self.base_path.exists():
                self._bytes_written = self.base_path.stat().st_size
            return

        log.info(f"finalize: {len(parts)} part(s) → {self.base_path.name}")
        parts = sorted(parts, key=lambda x: x.name)

        tmp_path = self.base_path.with_suffix(".merged.tmp")

        try:
            if _HAS_PYARROW:
                # LOAD FIRST PART → GET SCHEMA
                first_tbl = pq.read_table(parts[0])
                schema = first_tbl.schema

                # open final writer with extracted schema
                writer = pq.ParquetWriter(tmp_path, schema=schema, compression=self._compression)

                # write first part
                writer.write_table(first_tbl)

                # append remaining parts
                for p in parts[1:]:
                    tbl = pq.read_table(p)
                    writer.write_table(tbl)

                writer.close()

                self._rows_written = sum(len(pq.read_table(p)) for p in parts)

            else:
                # fallback concat (slow)
                dfs = [pd.read_parquet(p) for p in parts]
                df = pd.concat(dfs, ignore_index=True)
                df.to_parquet(tmp_path, index=False, compression=self._compression)
                self._rows_written = len(df)

            # atomic swap
            tmp_path.replace(self.base_path)
            self._bytes_written = self.base_path.stat().st_size

            # cleanup parts
            for p in parts:
                try:
                    p.unlink()
                except:
                    pass

            log.info(f"finalize: wrote {self.base_path.name} — rows={self._rows_written}")

        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except:
                    pass

    def get_manifest(self) -> Dict:
        return {
            "path": str(self.base_path),
            "rows": int(self._rows_written),
            "bytes": int(self._bytes_written),
        }
