"""nn/dataset.py — Self-play dataset storage for Mini Xiangqi.

Self-play produces one *sample* per position visited in a finished game::

    sample = SelfPlaySample(
        planes = np.ndarray (C, 7, 7),          # Board.to_planes() snapshot
        policy = np.ndarray (NUM_MOVE_ACTIONS,), # MCTS visit-count target
        value  = float,                          # final outcome z in {-1, 0, +1}
                                                 #   from the mover's perspective
        move   = str,                            # the move actually played (UCI)
    )

Games are written as JSON-lines to a ``.jsonl`` file, with the binary planes
base64-encoded to keep the file compact. This module turns those files back
into a PyTorch-style dataset (or a plain iterator for non-Torch callers).

The on-disk format is one JSON object per line, each object wrapping a game's
worth of samples under a ``"samples"`` list. This is an in-memory / single-file
design; a later version can shard across many files and add shuffling/windowing.

numpy is required for (de)serialization of the plane arrays. PyTorch is optional
and only needed for :class:`TorchSelfPlayDataset` and :func:`collate_fn`.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Iterator, List

from engine.board import BOARD_SIZE
from nn.network import NUM_MOVE_ACTIONS

# Numpy is needed to (de)serialize the plane arrays.
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False


def encode_planes(planes) -> str:
    """Base64-encode a float32 plane array for JSON storage."""
    if not _HAS_NUMPY:
        raise RuntimeError("numpy required for encode_planes")
    return base64.b64encode(
        np.asarray(planes, dtype=np.float32).tobytes()
    ).decode("ascii")


def decode_planes(b64: str):
    """Inverse of :func:`encode_planes`. Restores the ``(C, 7, 7)`` shape."""
    if not _HAS_NUMPY:
        raise RuntimeError("numpy required for decode_planes")
    buf = base64.b64decode(b64)
    arr = np.frombuffer(buf, dtype=np.float32).copy()  # copy: frombuffer is read-only
    # encode_planes stored the flattened (C, 7, 7) array; reshape back. We infer
    # C from the total length so the channel count stays in sync with Board.
    per_plane = BOARD_SIZE * BOARD_SIZE  # 49
    n_channels = arr.size // per_plane
    return arr.reshape(n_channels, BOARD_SIZE, BOARD_SIZE)


@dataclass
class SelfPlaySample:
    """One training sample: planes, policy target, value target, played move."""

    planes: Any  # np.ndarray of shape (C, 7, 7)
    policy: Any  # np.ndarray of shape (NUM_MOVE_ACTIONS,)
    value: float
    move: str

    def to_dict(self) -> dict:
        # Accept either a numpy array (.tolist()) or a plain list.
        policy = self.policy
        if hasattr(policy, "tolist"):
            policy = policy.tolist()
        return {
            "planes": encode_planes(self.planes),
            "policy": list(policy),
            "value": float(self.value),
            "move": self.move,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SelfPlaySample":
        policy = (
            np.asarray(d["policy"], dtype=np.float32) if _HAS_NUMPY else d["policy"]
        )
        return cls(decode_planes(d["planes"]), policy, d["value"], d["move"])


class SelfPlayDataset:
    """Append-only JSONL store of self-play games and an in-memory view.

    Usage::

        ds = SelfPlayDataset("data/games/games.jsonl")
        ds.append_game(samples)      # persist one finished game
        for s in ds.iter_samples():  # stream all samples
            ...

    The on-disk format is one JSON object per line, each object wrapping a
    game's worth of samples under a ``"samples"`` list.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    def append_game(self, samples: List[SelfPlaySample]) -> None:
        """Append one finished game's samples to the JSONL file."""
        record = {"samples": [s.to_dict() for s in samples]}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def iter_samples(self) -> Iterator[SelfPlaySample]:
        """Yield every sample across every game in the file."""
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                for s in record.get("samples", []):
                    yield SelfPlaySample.from_dict(s)

    def num_games(self) -> int:
        """Count of stored games (lines in the file)."""
        if not os.path.exists(self.path):
            return 0
        with open(self.path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def get_game(self, index: int) -> List[SelfPlaySample]:
        """Return the samples of the ``index``-th game (0-based).

        Negative indices count from the end, like a Python list. Raises
        ``IndexError`` if out of range. Each line is one game, so this scans to
        the requested line; fine for the dataset sizes a base pipeline produces.
        """
        if not os.path.exists(self.path):
            raise IndexError(f"dataset not found: {self.path}")
        target = index if index >= 0 else index + self.num_games()
        if target < 0:
            raise IndexError(f"game index {index} out of range")
        with open(self.path, "r", encoding="utf-8") as f:
            seen = -1
            for line in f:
                line = line.strip()
                if not line:
                    continue
                seen += 1
                if seen == target:
                    record = json.loads(line)
                    return [
                        SelfPlaySample.from_dict(s)
                        for s in record.get("samples", [])
                    ]
        raise IndexError(f"game index {index} out of range ({seen + 1} games)")


# Optional: PyTorch Dataset wrapper + collate helper, only if torch is installed.
try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


if _HAS_TORCH:

    class TorchSelfPlayDataset(_TorchDataset):
        """In-memory ``torch.utils.data.Dataset`` over a JSONL file.

        Loads the whole file into memory on construction. For large-scale
        training, replace this with a sharded, memory-mapped loader.
        """

        def __init__(self, path: str) -> None:
            self._samples: List[SelfPlaySample] = list(
                SelfPlayDataset(path).iter_samples()
            )

        def __len__(self) -> int:
            return len(self._samples)

        def __getitem__(self, idx: int):
            s = self._samples[idx]
            planes = torch.from_numpy(np.asarray(s.planes)).float()
            policy = torch.from_numpy(np.asarray(s.policy)).float()
            value = torch.tensor(s.value, dtype=torch.float32)
            return planes, policy, value

    def collate_fn(batch):
        """Stack a list of ``(planes, policy, value)`` tuples into batched tensors.

        Suitable for ``torch.utils.data.DataLoader(..., collate_fn=collate_fn)``.
        Returns ``(planes, policy, value)`` with shapes ``[B, C, H, W]``,
        ``[B, NUM_MOVE_ACTIONS]`` and ``[B]`` respectively.
        """
        planes, policy, value = zip(*batch)
        planes = torch.stack(planes, dim=0)
        policy = torch.stack(policy, dim=0)
        value = torch.stack(value, dim=0)
        return planes, policy, value
