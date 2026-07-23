"""
Mini Xiangqi - Timing Utility
==============================

Provides a lightweight Timer context manager and a TimeTracker for
accumulating and reporting named timing sections.

Usage:
    from utils.timer import Timer, TimeTracker

    # Simple one-off timing
    with Timer("mcts_search") as t:
        run_search()
    print(t.elapsed)  # seconds

    # Accumulating multiple timings
    tracker = TimeTracker()
    with tracker.track("search"):
        run_search()
    with tracker.track("eval"):
        run_eval()
    tracker.report()  # prints a summary table
"""

from __future__ import annotations

import time
from typing import Optional


class Timer:
    """Context manager that records wall-clock elapsed time in seconds.

    Attributes:
        name: A human-readable label for the timed block.
        elapsed: Elapsed time in seconds (available after the block exits).

    Example:
        >>> with Timer("search") as t:
        ...     time.sleep(0.1)
        >>> assert t.elapsed >= 0.1
    """

    def __init__(self, name: str = "unnamed") -> None:
        self.name = name
        self.elapsed: float = 0.0
        self._start: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        if self._start is not None:
            self.elapsed = time.perf_counter() - self._start
        return None

    def __repr__(self) -> str:
        return f"Timer(name={self.name!r}, elapsed={self.elapsed:.4f}s)"


class TimeTracker:
    """Accumulates named timers and prints a summary report.

    Each call to :meth:`track` returns a context manager whose elapsed
    time is added to the running total for that name.

    Example:
        >>> tracker = TimeTracker()
        >>> with tracker.track("search"):
        ...     time.sleep(0.05)
        >>> with tracker.track("search"):
        ...     time.sleep(0.05)
        >>> tracker.total("search") >= 0.1
        True
    """

    def __init__(self) -> None:
        # name -> (total_seconds, call_count)
        self._records: dict[str, list[float]] = {}

    def track(self, name: str) -> "_TrackedTimer":
        """Return a context manager that records elapsed time under *name*.

        Args:
            name: Label for the timing section.

        Returns:
            A context manager; elapsed time is accumulated on exit.
        """
        return _TrackedTimer(self, name)

    def record(self, name: str, elapsed: float) -> None:
        """Manually record an elapsed time for *name*.

        Args:
            name: Label for the timing section.
            elapsed: Elapsed seconds to add.
        """
        if name not in self._records:
            self._records[name] = [0.0, 0.0]  # [total, count]
        self._records[name][0] += elapsed
        self._records[name][1] += 1

    def total(self, name: str) -> float:
        """Return total accumulated seconds for *name* (0 if unknown)."""
        entry = self._records.get(name)
        return entry[0] if entry else 0.0

    def count(self, name: str) -> int:
        """Return the number of recorded calls for *name*."""
        entry = self._records.get(name)
        return int(entry[1]) if entry else 0

    def reset(self) -> None:
        """Clear all accumulated records."""
        self._records.clear()

    def summary(self) -> str:
        """Build a human-readable summary table as a string."""
        if not self._records:
            return "TimeTracker: no records."

        header = f"{'Name':<24} {'Total (s)':>10} {'Calls':>7} {'Avg (s)':>10}"
        sep = "-" * len(header)
        lines = [sep, header, sep]

        for name, (total, count) in sorted(
            self._records.items(), key=lambda kv: -kv[1][0]
        ):
            avg = total / count if count > 0 else 0.0
            lines.append(f"{name:<24} {total:>10.4f} {int(count):>7} {avg:>10.4f}")

        lines.append(sep)
        return "\n".join(lines)

    def report(self) -> None:
        """Print the summary table to stdout."""
        print(self.summary())

    def __repr__(self) -> str:
        names = list(self._records.keys())
        return f"TimeTracker(sections={names})"


class _TrackedTimer:
    """Internal context manager used by TimeTracker.track()."""

    def __init__(self, tracker: TimeTracker, name: str) -> None:
        self._tracker = tracker
        self._name = name
        self.elapsed: float = 0.0
        self._start: Optional[float] = None

    def __enter__(self) -> "_TrackedTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        if self._start is not None:
            self.elapsed = time.perf_counter() - self._start
            self._tracker.record(self._name, self.elapsed)
        return None
