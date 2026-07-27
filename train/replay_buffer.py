"""train/replay_buffer.py — Experience replay buffer for AlphaZero training.

Holds :class:`nn.dataset.SelfPlaySample` objects produced by self-play and
serves random mini-batches to the trainer. Samples are kept in a bounded
:class:`collections.deque` so the most recent experience gradually pushes out
the oldest (a standard "sliding window" replay buffer).

The buffer can be persisted to / restored from a pickle file, and it can also
be (re)populated in bulk from a :class:`nn.dataset.SelfPlayDataset` JSONL file
via :meth:`ReplayBuffer.load_from_dataset`.

Example::

    from nn.dataset import SelfPlayDataset
    from train.replay_buffer import ReplayBuffer

    buf = ReplayBuffer(max_size=50_000)
    buf.load_from_dataset(SelfPlayDataset("data/games/games.jsonl"), max_games=200)
    batch = buf.sample(64)
"""

from __future__ import annotations

import os
import pickle
import random
from collections import deque
from typing import List, Optional

from nn.dataset import SelfPlayDataset, SelfPlaySample


class ReplayBuffer:
    """A bounded experience-replay buffer over self-play samples.

    Samples from many games are stored flat (one entry per visited position).
    :meth:`sample` draws a uniformly random mini-batch without replacement.

    Attributes:
        max_size: Maximum number of samples retained. When exceeded, the oldest
            samples are discarded (FIFO).
    """

    def __init__(self, max_size: int = 100_000) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.max_size = int(max_size)
        self._buffer: "deque[SelfPlaySample]" = deque(maxlen=self.max_size)

    # ------------------------------------------------------------------ #
    # Insertion
    # ------------------------------------------------------------------ #
    def add_game(self, samples: List[SelfPlaySample]) -> None:
        """Add every sample from one finished game to the buffer.

        Args:
            samples: The per-position samples of a single self-play game.
        """
        for sample in samples:
            self._buffer.append(sample)

    def add(self, sample: SelfPlaySample) -> None:
        """Add a single sample (convenience wrapper around :meth:`add_game`)."""
        self._buffer.append(sample)

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def sample(self, batch_size: int) -> List[SelfPlaySample]:
        """Return a uniformly random mini-batch of ``batch_size`` samples.

        If the buffer holds fewer than ``batch_size`` samples, all of them are
        returned (shuffled). Sampling is without replacement.

        Args:
            batch_size: Desired number of samples.

        Returns:
            A list of :class:`SelfPlaySample` of length
            ``min(batch_size, len(self))``.
        """
        if batch_size <= 0:
            return []
        n = len(self._buffer)
        if n == 0:
            return []
        k = min(batch_size, n)
        return random.sample(list(self._buffer), k)

    # ------------------------------------------------------------------ #
    # Bulk loading from a dataset
    # ------------------------------------------------------------------ #
    def load_from_dataset(
        self,
        dataset: SelfPlayDataset,
        max_games: Optional[int] = None,
    ) -> int:
        """Populate the buffer from a :class:`SelfPlayDataset` JSONL store.

        Streams the dataset once via :meth:`SelfPlayDataset.iter_games` (a
        single O(n) pass). If ``max_games`` is given, only the most recent
        ``max_games`` games are loaded (a rolling window, useful to keep the
        buffer focused on fresh experience); otherwise every game is loaded.
        The buffer's own ``max_size`` still caps the total samples retained.

        Args:
            dataset: The on-disk self-play dataset to read from.
            max_games: Optional cap on the number of (most recent) games to load.

        Returns:
            The number of games actually loaded.
        """
        if max_games is None:
            loaded = 0
            for game in dataset.iter_games():
                self.add_game(game)
                loaded += 1
            return loaded

        # Keep only the most recent ``max_games`` games via a rolling window.
        recent: "deque[List[SelfPlaySample]]" = deque(maxlen=max_games)
        for game in dataset.iter_games():
            recent.append(game)
        for game in recent:
            self.add_game(game)
        return len(recent)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Persist the buffer contents (and ``max_size``) to a pickle file.

        Args:
            path: Destination file path. Parent directories are created.
        """
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        state = {"max_size": self.max_size, "samples": list(self._buffer)}
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        """Restore buffer contents from a pickle file written by :meth:`save`.

        The current contents are replaced. The saved ``max_size`` is honored.

        Args:
            path: Path to a pickle file produced by :meth:`save`.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.max_size = int(state.get("max_size", self.max_size))
        self._buffer = deque(state.get("samples", []), maxlen=self.max_size)

    # ------------------------------------------------------------------ #
    # Composition diagnostics
    # ------------------------------------------------------------------ #
    def value_composition(self) -> dict:
        """Count win/loss/draw samples currently in the buffer by value sign.

        Returns:
            ``{"win", "loss", "draw", "total"}`` integer counts.
        """
        win = loss = draw = 0
        for s in self._buffer:
            if s.value > 0:
                win += 1
            elif s.value < 0:
                loss += 1
            else:
                draw += 1
        return {"win": win, "loss": loss, "draw": draw,
                "total": win + loss + draw}

    # ------------------------------------------------------------------ #
    # Dunder helpers
    # ------------------------------------------------------------------ #
    def clear(self) -> None:
        """Remove all samples from the buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        """Current number of samples in the buffer."""
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self._buffer)}, max_size={self.max_size})"
