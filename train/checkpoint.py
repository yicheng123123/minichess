"""train/checkpoint.py — Model checkpoint management for AlphaZero training.

Persists and restores :class:`nn.network.TorchPolicyValueNet` weights alongside
training metadata (iteration, timestamp, config). Checkpoints are ``.pt`` files
named ``checkpoint_<iteration>.pt`` inside a checkpoint directory; the current
best model (as judged by arena evaluation) is additionally stored as ``best.pt``.

A checkpoint file is a dict::

    {
        "state_dict": <model state_dict>,
        "iteration":  int,
        "timestamp":  float,   # time.time()
        "config":     dict | None,
        "metadata":   dict | None,   # caller-supplied extras
    }

PyTorch is required to use this module; importing it without torch still works
(constructing a :class:`CheckpointManager` is fine), but any save/load call
raises a clear ``RuntimeError``.
"""

from __future__ import annotations

import glob
import os
import re
import time
from typing import Any, Dict, List, Optional

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError(
            "PyTorch is required for train.checkpoint; install torch to "
            "save or load model checkpoints."
        )


# Filename convention: checkpoint_<iteration>.pt
_CKPT_PATTERN = re.compile(r"checkpoint_(\d+)\.pt$")
_BEST_FILENAME = "best.pt"


class CheckpointManager:
    """Manage model checkpoints within a directory.

    Responsibilities:
      * save numbered iteration checkpoints (``checkpoint_<iter>.pt``),
      * save / locate the single "best" model (``best.pt``),
      * load the latest (or a specific) checkpoint back into a network,
      * list the iterations that have checkpoints on disk.

    Args:
        checkpoint_dir: Directory in which checkpoint files are stored. Created
            on demand when a checkpoint is saved.
    """

    def __init__(self, checkpoint_dir: str = "models") -> None:
        self.checkpoint_dir = checkpoint_dir

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #
    def _ensure_dir(self) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _ckpt_path(self, iteration: int) -> str:
        return os.path.join(self.checkpoint_dir, f"checkpoint_{iteration}.pt")

    @property
    def best_model_path(self) -> str:
        """Path to the best-model checkpoint (``best.pt``)."""
        return os.path.join(self.checkpoint_dir, _BEST_FILENAME)

    # ------------------------------------------------------------------ #
    # Saving
    # ------------------------------------------------------------------ #
    def save(
        self,
        net: Any,
        iteration: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save ``net``'s weights and metadata as ``checkpoint_<iteration>.pt``.

        Args:
            net: A ``torch.nn.Module`` (e.g. the policy-value network).
            iteration: Training iteration number used in the filename.
            metadata: Optional extra info. Recognized top-level keys:
                ``config`` (dict) is stored under its own field; everything is
                also retained under ``metadata``.

        Returns:
            The path of the written checkpoint file.
        """
        _require_torch()
        self._ensure_dir()

        metadata = dict(metadata or {})
        config = metadata.get("config")

        payload = {
            "state_dict": net.state_dict(),
            "iteration": int(iteration),
            "timestamp": time.time(),
            "config": config,
            "metadata": metadata,
        }
        path = self._ckpt_path(iteration)
        torch.save(payload, path)
        return path

    def save_best(self, net: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save ``net`` as the current best model (``best.pt``).

        Args:
            net: A ``torch.nn.Module``.
            metadata: Optional metadata (same conventions as :meth:`save`).

        Returns:
            The path of the written best-model file.
        """
        _require_torch()
        self._ensure_dir()

        metadata = dict(metadata or {})
        payload = {
            "state_dict": net.state_dict(),
            "iteration": int(metadata.get("iteration", -1)),
            "timestamp": time.time(),
            "config": metadata.get("config"),
            "metadata": metadata,
        }
        path = self.best_model_path
        torch.save(payload, path)
        return path

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load(self, net: Any, iteration: int, map_location: Optional[str] = None) -> Dict[str, Any]:
        """Load a specific iteration's checkpoint into ``net``.

        Args:
            net: A ``torch.nn.Module`` whose ``state_dict`` will be replaced.
            iteration: The iteration number to load.
            map_location: Optional ``torch.load`` map_location (e.g. ``"cpu"``).

        Returns:
            The full checkpoint payload dict (so callers can read metadata).

        Raises:
            FileNotFoundError: If no checkpoint exists for ``iteration``.
        """
        _require_torch()
        path = self._ckpt_path(iteration)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint for iteration {iteration}: {path}")
        payload = torch.load(path, map_location=map_location)
        net.load_state_dict(payload["state_dict"])
        return payload

    def load_latest(self, net: Any, map_location: Optional[str] = None) -> Optional[int]:
        """Load the most recent checkpoint into ``net``.

        Args:
            net: A ``torch.nn.Module`` whose ``state_dict`` will be replaced.
            map_location: Optional ``torch.load`` map_location.

        Returns:
            The iteration number that was loaded, or ``None`` if no checkpoint
            exists (in which case ``net`` is left unchanged).
        """
        iterations = self.list_checkpoints()
        if not iterations:
            return None
        latest = max(iterations)
        self.load(net, latest, map_location=map_location)
        return latest

    def load_best(self, net: Any, map_location: Optional[str] = None) -> bool:
        """Load the best-model checkpoint (``best.pt``) into ``net`` if present.

        Returns:
            ``True`` if a best model was loaded, ``False`` if none exists.
        """
        _require_torch()
        path = self.best_model_path
        if not os.path.exists(path):
            return False
        payload = torch.load(path, map_location=map_location)
        net.load_state_dict(payload["state_dict"])
        return True

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #
    def list_checkpoints(self) -> List[int]:
        """Return the sorted list of iteration numbers that have checkpoints."""
        if not os.path.isdir(self.checkpoint_dir):
            return []
        iterations: List[int] = []
        for path in glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.pt")):
            match = _CKPT_PATTERN.search(os.path.basename(path))
            if match:
                iterations.append(int(match.group(1)))
        return sorted(iterations)

    def has_best(self) -> bool:
        """Whether a ``best.pt`` checkpoint exists on disk."""
        return os.path.exists(self.best_model_path)

    def __repr__(self) -> str:
        return f"CheckpointManager(dir={self.checkpoint_dir!r})"
