"""
Mini Xiangqi - Reproducibility Utility
=======================================

Provides helpers to seed all relevant random number generators for
reproducible experiments.  Only the standard library ``random`` module
is required; numpy and torch are seeded when available.

Usage:
    from utils.seed import set_seed, reproducible

    # One-time global seeding
    set_seed(42)

    # Context manager for a reproducible block (restores state after)
    with reproducible(123):
        result = some_stochastic_function()

    # Decorator form
    @reproducible(seed=7)
    def train_epoch():
        ...
"""

from __future__ import annotations

import functools
import random
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional, TypeVar, Union, overload

F = TypeVar("F", bound=Callable[..., Any])


def set_seed(seed: int) -> None:
    """Seed all available random number generators.

    Sets seeds for:
      - Python's built-in ``random`` module (always).
      - NumPy (``numpy.random.seed``) if numpy is importable.
      - PyTorch (``torch.manual_seed`` + CUDA) if torch is importable.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)

    try:
        import numpy as np  # noqa: F401

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Deterministic algorithms (may reduce performance)
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except ImportError:
        pass


def _save_rng_state() -> dict[str, Any]:
    """Capture the current RNG states."""
    state: dict[str, Any] = {"random": random.getstate()}

    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass

    try:
        import torch

        state["torch"] = torch.random.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass

    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    """Restore previously captured RNG states."""
    if "random" in state:
        random.setstate(state["random"])

    try:
        import numpy as np

        if "numpy" in state:
            np.random.set_state(state["numpy"])
    except ImportError:
        pass

    try:
        import torch

        if "torch" in state:
            torch.random.set_rng_state(state["torch"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except ImportError:
        pass


@contextmanager
def _reproducible_context(seed: int) -> Generator[None, None, None]:
    """Context manager: seed RNGs, yield, then restore previous state."""
    saved = _save_rng_state()
    set_seed(seed)
    try:
        yield
    finally:
        _restore_rng_state(saved)


class reproducible:
    """Decorator **and** context manager for reproducible code blocks.

    As a context manager::

        with reproducible(42):
            ...

    As a decorator (with or without arguments)::

        @reproducible(seed=42)
        def experiment():
            ...

        @reproducible
        def experiment():
            ...

    When used as a decorator the RNG state is saved before and restored
    after the wrapped function, so surrounding code is unaffected.
    """

    def __init__(self, seed_or_func: Union[int, F, None] = None, *, seed: Optional[int] = None) -> None:
        # Determine the actual seed and whether we wrap a function directly
        self._seed: Optional[int] = None
        self._func: Optional[F] = None

        if callable(seed_or_func):
            # Used as @reproducible without parentheses — need a seed kwarg
            # In this case we default to seed=0 unless provided via seed kwarg
            self._func = seed_or_func  # type: ignore[assignment]
            self._seed = seed if seed is not None else 0
        elif isinstance(seed_or_func, int):
            self._seed = seed_or_func
        elif seed is not None:
            self._seed = seed

    # -- Context manager protocol ------------------------------------------

    def __enter__(self) -> "reproducible":
        self._saved_state = _save_rng_state()
        if self._seed is not None:
            set_seed(self._seed)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        _restore_rng_state(self._saved_state)
        return None

    # -- Decorator protocol -------------------------------------------------

    def __call__(self, func: F) -> F:
        """Wrap *func* so it runs under a seeded RNG, restoring state after."""
        seed = self._seed if self._seed is not None else 0

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with _reproducible_context(seed):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]
