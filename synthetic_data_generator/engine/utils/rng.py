# synthetic_data_generator/engine/utils/rng.py
"""
Deterministic RNG wrapper for Project_PhantomScalp synthetic data engines.

This wrapper provides a small, forgiving API surface over numpy's Generator so that
engine code can call functions using either:
    rng.choice(choices, size=n, p=...)
or
    rng.choice("tag", choices, size=n, p=...)

It also protects against duplicate keyword/positional arguments that previously
raised TypeError in the engines.

Ensure numpy >= 1.17 is installed.
"""

from __future__ import annotations
from typing import Any, Iterable, Optional, Tuple, Union, Sequence
import numpy as np

# Expose types for callers
ArrayLike = Union[Iterable[Any], np.ndarray, Sequence[Any]]

class RNG:
    """
    Lightweight wrapper around numpy.random.Generator with forgiving signatures.

    Usage patterns supported (examples):
      rng = RNG(seed=123)
      rng.choice(['A','B'], size=10, p=[0.5,0.5])
      rng.choice('tag', ['A','B'], size=10)      # tag is ignored by RNG but accepted
      rng.poisson(lam=5, size=100)
      rng.poisson('arrivals', 5, size=100)
      rng.integers(low=0, high=10, size=5)
    """

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    # --- seeding / state ---
    def seed(self, seed: Optional[int]) -> None:
        """Reseed the internal Generator deterministically."""
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def random_state(self) -> np.random.Generator:
        """Return the underlying numpy Generator."""
        return self._rng

    # --- helpers to interpret flexible argument order ---
    @staticmethod
    def _strip_tag_and_extract_args(args: Tuple[Any, ...]) -> Tuple[Tuple[Any, ...], Optional[str]]:
        """
        If the first arg is a string, treat it as a 'tag' and remove it.
        Return (remaining_args, tag_or_none)
        """
        if len(args) >= 1 and isinstance(args[0], str):
            return args[1:], args[0]
        return args, None

    @staticmethod
    def _pop_first_if_iterable(args: Tuple[Any, ...]) -> Tuple[Optional[Any], Tuple[Any, ...]]:
        """
        If the first positional arg looks like choices/values (iterable but not str),
        pop and return it and the remaining args.
        """
        if not args:
            return None, args
        first = args[0]
        # treat numpy arrays and sequences (but not strings) as choices
        if isinstance(first, (np.ndarray, list, tuple, range)) or hasattr(first, "__iter__") and not isinstance(first, (str, bytes)):
            return first, args[1:]
        return None, args

    # --- public RNG methods with forgiving signatures ---
    def choice(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Flexible wrapper around Generator.choice.

        Supported call forms:
          choice(choices, size=...)
          choice(tag, choices, size=...)
          choice(choices, k)  [k interpreted as size]
        """
        args, tag = self._strip_tag_and_extract_args(args)

        # If first positional arg is choices-like, consume it
        choices, args = self._pop_first_if_iterable(args)

        # If choices provided positionally, kwargs may include 'p', 'replace', 'size'
        if choices is None:
            # if no positional choices, expect choices in kwargs as 'a' or 'choices'
            choices = kwargs.pop("a", None) or kwargs.pop("choices", None)

        # If sizes passed positionally (e.g. choice(choices, 10)), handle it
        if args and (isinstance(args[0], (int, tuple))):
            # convert to size keyword if not already present
            if "size" not in kwargs:
                kwargs["size"] = args[0]
            args = args[1:]

        if choices is None:
            raise TypeError("choice() missing required 'choices' argument.")

        # Final safety: avoid duplicate keyword for size if user passed both positional and size kw
        # (numpy would have complained earlier)
        size = kwargs.pop("size", None)
        replace = kwargs.pop("replace", True)
        p = kwargs.pop("p", None)

        # Now call numpy's choice with only safe keyword args
        if p is not None:
            return self._rng.choice(choices, size=size, replace=replace, p=p)
        return self._rng.choice(choices, size=size, replace=replace)

    def poisson(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Wrapper for poisson. Supports:
          poisson(lam=..., size=...)
          poisson(lam, size=...)            # positional lam
          poisson('tag', lam, size=...)     # optional tag
          poisson('tag', lam=..., size=...)
        """
        args, tag = self._strip_tag_and_extract_args(args)

        # If first positional arg exists and is numeric -> lam
        lam = None
        if args:
            candidate = args[0]
            if isinstance(candidate, (int, float, np.generic)):
                lam = float(candidate)
                args = args[1:]

        # If lam not found yet, check kwargs
        lam = float(kwargs.pop("lam", lam)) if ("lam" in kwargs or lam is not None) else kwargs.pop("lambda", None)  # accept 'lambda' if used
        if lam is None:
            raise TypeError("poisson() missing required 'lam' parameter")

        # size: accept positional or kw
        size = kwargs.pop("size", None)
        if not size and args and isinstance(args[0], (int, tuple)):
            size = args[0]

        # ensure size is None or int/tuple
        return self._rng.poisson(lam=lam, size=size)

    def normal(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        normal(loc=0.0, scale=1.0, size=...)
        Supports normal(size=...), normal(n) (n -> size), normal('tag', size=...)
        """
        args, tag = self._strip_tag_and_extract_args(args)
        # positional size
        if args and isinstance(args[0], (int, tuple)):
            if "size" not in kwargs:
                kwargs["size"] = args[0]
        # delegate
        loc = kwargs.pop("loc", 0.0)
        scale = kwargs.pop("scale", 1.0)
        size = kwargs.pop("size", None)
        return self._rng.normal(loc=loc, scale=scale, size=size)

    def integers(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        integers(low, high=None, size=...)
        Supports integers(high) -> returns [0, high)
                 integers(low, high, size)
                 integers('tag', low, high, size)
        """
        args, tag = self._strip_tag_and_extract_args(args)

        # positional resolution
        if args:
            if len(args) == 1:
                # integers(high)
                low = 0
                high = int(args[0])
            elif len(args) >= 2:
                low = int(args[0])
                high = int(args[1])
            else:
                low = kwargs.get("low", 0)
                high = kwargs.get("high", None)
        else:
            low = kwargs.get("low", 0)
            high = kwargs.get("high", None)

        if high is None:
            raise TypeError("integers() missing 'high' argument")

        size = kwargs.pop("size", None)
        return self._rng.integers(low=low, high=high, size=size)

    def uniform(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        uniform(low=0.0, high=1.0, size=...)
        Accepts uniform('tag', size=...), uniform(size)
        """
        args, tag = self._strip_tag_and_extract_args(args)
        if args and isinstance(args[0], (int, float, tuple)):
            if "size" not in kwargs and isinstance(args[0], (int, tuple)):
                kwargs["size"] = args[0]
        low = kwargs.pop("low", 0.0)
        high = kwargs.pop("high", 1.0)
        size = kwargs.pop("size", None)
        return self._rng.uniform(low=low, high=high, size=size)

    def exponential(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        exponential(scale=1.0, size=...)
        """
        args, tag = self._strip_tag_and_extract_args(args)
        if args and isinstance(args[0], (int, float, tuple)):
            if "size" not in kwargs:
                kwargs["size"] = args[0]
        scale = kwargs.pop("scale", 1.0)
        size = kwargs.pop("size", None)
        return self._rng.exponential(scale=scale, size=size)

    def binomial(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        binomial(n, p, size=...)
        supports binomial('tag', n, p, size=...)
        """
        args, tag = self._strip_tag_and_extract_args(args)
        if args and len(args) >= 2:
            n = int(args[0])
            p = float(args[1])
        else:
            n = kwargs.pop("n", None)
            p = kwargs.pop("p", None)
        if n is None or p is None:
            raise TypeError("binomial() requires n and p")
        size = kwargs.pop("size", None)
        return self._rng.binomial(n=n, p=p, size=size)

    # convenience alias for backwards compatibility
    def rand(self, *shape: int) -> np.ndarray:
        return self._rng.random(size=shape)

    # other wrappers may be added as required


# Backwards compatibility alias expected in some engine codebases
class RNGManager(RNG):
    """Alias for compatibility with older engine code that imports RNGManager."""
    pass


# Provide a module-level default RNG instance and accessor
_default_rng = RNG()

def get_default_rng() -> RNG:
    """Return a shared default RNG instance (deterministic if seed set)."""
    return _default_rng


# Module-level convenience API that some files may call as `from rng import RNG, get_default_rng`
__all__ = ["RNG", "RNGManager", "get_default_rng", "ArrayLike"]
