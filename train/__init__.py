"""train package — AlphaZero-style training pipeline for Mini Xiangqi.

This package houses the training side of the project (the data-generation and
self-play logic now live in :mod:`nn.dataset` and :mod:`selfplay` respectively):

  * :mod:`train.replay_buffer` — :class:`ReplayBuffer`, a bounded experience
    replay buffer over :class:`nn.dataset.SelfPlaySample`.
  * :mod:`train.checkpoint` — :class:`CheckpointManager` for saving / loading
    model checkpoints and the current best model.
  * :mod:`train.trainer` — :class:`Trainer`, the self-play -> train -> evaluate
    -> checkpoint loop, plus a ``python -m train.trainer`` CLI.

Convenient re-exports::

    from train import Trainer, ReplayBuffer, CheckpointManager

PyTorch is optional at import time: :class:`ReplayBuffer` and
:class:`CheckpointManager` are always available, while :class:`Trainer` is
re-exported only when torch is installed (constructing it without torch raises
a clear ``RuntimeError`` regardless).
"""

from __future__ import annotations

from .replay_buffer import ReplayBuffer
from .checkpoint import CheckpointManager

__all__ = [
    "ReplayBuffer",
    "CheckpointManager",
]

# Trainer pulls in the torch-only network/loss stack; import it lazily so the
# package still imports cleanly in environments without PyTorch.
try:  # pragma: no cover - depends on environment
    from .trainer import Trainer

    __all__.append("Trainer")
except ImportError:
    pass
