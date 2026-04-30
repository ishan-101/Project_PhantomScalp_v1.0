# sequence_id.py
"""Deterministic sequence id generator with optional seed and batching support.
Produces strictly increasing integer IDs starting from an offset derived
from the seed to be reproducible across runs.
"""

from __future__ import annotations
from typing import Optional, Iterable, List
import threading


class SequenceID:
    """
    Simple, deterministic sequence id generator.

    Usage:
        seq = SequenceID(seed=42, start=0)
        seq.next() -> 0
        seq.next() -> 1
        seq.next_batch(5) -> [2,3,4,5,6]
    """

    def __init__(self, seed: Optional[int] = None, start: Optional[int] = None):
        self._lock = threading.Lock()
        if start is not None:
            self._current = int(start)
        else:
            # deterministic mapping from seed to start offset
            self._current = int(seed or 0) * 1000
        # _current points to next value to be returned
        self._seed = seed

    @property
    def current(self) -> int:
        # last-assigned + 1 actually; return the next to be issued
        with self._lock:
            return int(self._current)

    def next(self) -> int:
        with self._lock:
            val = int(self._current)
            self._current += 1
            return val

    def next_batch(self, n: int) -> List[int]:
        if n <= 0:
            return []
        with self._lock:
            start = int(self._current)
            end = start + int(n)
            self._current = end
            return list(range(start, end))
