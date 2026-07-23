"""nn/network.py — Residual policy-value network for Mini Xiangqi.

This is an improved version of the network that originally lived in
``network/model.py``. It keeps the same public contract (the
:class:`PolicyValueNet` interface, the move<->index helpers and the
``NUM_MOVE_ACTIONS`` constant) but upgrades the architecture:

  * a stem convolution + batch normalization,
  * a configurable stack of **residual blocks** (conv-bn-relu-conv-bn + skip),
  * a **policy head** producing ``NUM_MOVE_ACTIONS`` (7^4 = 2401) logits, and
  * a **value head** producing a scalar in [-1, 1] (via ``tanh``).

The board is encoded by :meth:`engine.board.Board.to_planes` (11 channels by
default: 10 piece-occupancy planes + 1 side-to-move plane).

PyTorch is optional. If it is not installed, this module still imports and
exposes :class:`RandomPolicyValueNet` — a pure-Python fallback returning a
uniform policy and zero value — so the engine, search, self-play and tests can
run without a deep-learning stack. The real training pipeline requires PyTorch.
"""

from __future__ import annotations

import random
from typing import Dict, Tuple

from engine.board import Board, BOARD_SIZE
from engine.move import Move

# Flat move-space size: every (from_row, from_col, to_row, to_col) tuple.
NUM_MOVE_ACTIONS = BOARD_SIZE ** 4  # 7^4 = 2401


def move_to_index(move: Move) -> int:
    """Flatten a move to its policy-logit index in ``[0, NUM_MOVE_ACTIONS)``."""
    fr, fc, tr, tc = move.from_row, move.from_col, move.to_row, move.to_col
    return ((fr * BOARD_SIZE + fc) * BOARD_SIZE + tr) * BOARD_SIZE + tc


def index_to_move(idx: int) -> Move:
    """Inverse of :func:`move_to_index`."""
    tc = idx % BOARD_SIZE
    tr = (idx // BOARD_SIZE) % BOARD_SIZE
    fc = (idx // (BOARD_SIZE * BOARD_SIZE)) % BOARD_SIZE
    fr = idx // (BOARD_SIZE * BOARD_SIZE * BOARD_SIZE)
    return Move((fr, fc), (tr, tc))


class PolicyValueNet:
    """Abstract interface every network implementation satisfies.

    Callers (MCTS, self-play, training) should depend on this interface, not on
    the concrete PyTorch module, so the pure-Python fallback can be swapped in
    for testing.
    """

    def predict(self, board: Board) -> Tuple[Dict[int, float], float]:
        """Return ``(policy_logits_by_action_index, value)``.

        ``policy_logits_by_action_index`` only needs entries for legal moves;
        callers are responsible for masking. ``value`` is in [-1, 1].
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Pure-Python fallback (always available, no PyTorch needed)
# --------------------------------------------------------------------------- #
class RandomPolicyValueNet(PolicyValueNet):
    """Uniform policy + zero value. Useful for smoke tests of the pipeline."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def predict(self, board: Board) -> Tuple[Dict[int, float], float]:
        from engine.move_generator import legal_moves

        moves = legal_moves(board)
        # Uniform probability over legal moves, returned as "logits" that the
        # caller will softmax-mask. Equal logits -> uniform after softmax.
        logits = {move_to_index(m): 0.0 for m in moves}
        value = 0.0
        return logits, value


# --------------------------------------------------------------------------- #
# PyTorch implementation (defined only if torch is importable)
# --------------------------------------------------------------------------- #
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


if _HAS_TORCH:

    class ResidualBlock(nn.Module):
        """A pre-activation-free residual block: conv-bn-relu-conv-bn + skip.

        Two 3x3 convolutions (stride 1, padding 1) preserve the spatial size so
        the input can be added back as a skip connection. Batch normalization
        follows each convolution; a ReLU is applied after the first conv and
        after the residual sum.
        """

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            residual = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + residual  # skip connection
            return F.relu(out)

    class TorchPolicyValueNet(nn.Module, PolicyValueNet):
        """A residual CNN policy-value network.

        Architecture:
          * stem: 3x3 conv -> batch norm -> ReLU,
          * ``num_res_blocks`` residual blocks (each keeps ``hidden`` channels),
          * policy head: 1x1 conv -> bn -> ReLU -> flatten -> linear to
            ``NUM_MOVE_ACTIONS`` logits,
          * value head: 1x1 conv -> bn -> ReLU -> flatten -> linear -> ReLU ->
            linear(1) -> tanh (scalar in [-1, 1]).

        Input planes come from ``Board.to_planes()`` (11 channels by default).
        """

        def __init__(
            self,
            in_channels: int = 11,
            hidden: int = 128,
            num_res_blocks: int = 4,
        ) -> None:
            super().__init__()
            self.in_channels = in_channels
            self.hidden = hidden
            self.num_res_blocks = num_res_blocks

            # Stem: lift the input planes up to ``hidden`` channels.
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(),
            )

            # Residual tower.
            self.res_blocks = nn.Sequential(
                *(ResidualBlock(hidden) for _ in range(num_res_blocks))
            )

            # Policy head -> NUM_MOVE_ACTIONS logits.
            self.policy_conv = nn.Sequential(
                nn.Conv2d(hidden, 2, kernel_size=1, bias=False),
                nn.BatchNorm2d(2),
                nn.ReLU(),
            )
            self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, NUM_MOVE_ACTIONS)

            # Value head -> scalar in [-1, 1].
            self.value_conv = nn.Sequential(
                nn.Conv2d(hidden, 1, kernel_size=1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(),
            )
            self.value_fc = nn.Sequential(
                nn.Linear(BOARD_SIZE * BOARD_SIZE, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Tanh(),
            )

        def forward(self, planes: "torch.Tensor"):
            """Return ``(policy_logits, value)`` for a batched input ``[B, C, H, W]``.

            ``policy_logits`` has shape ``[B, NUM_MOVE_ACTIONS]`` and ``value``
            has shape ``[B]``.
            """
            h = self.stem(planes)
            h = self.res_blocks(h)

            p = self.policy_conv(h).flatten(1)
            policy_logits = self.policy_fc(p)

            v = self.value_conv(h).flatten(1)
            value = self.value_fc(v).squeeze(-1)
            return policy_logits, value

        def predict(self, board: Board) -> Tuple[Dict[int, float], float]:
            """Single-position inference, returning the same format as the fallback."""
            from engine.move_generator import legal_moves

            self.eval()
            planes = torch.from_numpy(board.to_planes()).float().unsqueeze(0)
            with torch.no_grad():
                logits, value = self.forward(planes)
            logits = logits.squeeze(0)
            value = float(value.item())

            # Only return logits for legal moves (caller masks anyway, but
            # returning a compact dict keeps MCTS allocations small).
            legal_idx = [move_to_index(m) for m in legal_moves(board)]
            out: Dict[int, float] = {i: float(logits[i].item()) for i in legal_idx}
            return out, value

        def save(self, path: str) -> None:
            torch.save(self.state_dict(), path)

        def load(self, path: str, map_location=None) -> None:
            self.load_state_dict(torch.load(path, map_location=map_location))

    def create_network(
        in_channels: int = 11,
        hidden: int = 128,
        num_res_blocks: int = 4,
    ) -> "TorchPolicyValueNet":
        """Factory for a residual :class:`TorchPolicyValueNet`.

        Args:
            in_channels: Number of input planes (``Board.to_planes()`` yields 11
                by default: 10 piece planes + 1 side-to-move plane).
            hidden: Number of channels throughout the residual tower and heads.
            num_res_blocks: How many residual blocks to stack.

        Returns:
            A freshly initialized network ready for training or inference.
        """
        return TorchPolicyValueNet(
            in_channels=in_channels,
            hidden=hidden,
            num_res_blocks=num_res_blocks,
        )


def default_net() -> PolicyValueNet:
    """Return a usable network: the PyTorch model if available, else the fallback."""
    if _HAS_TORCH:
        return create_network()
    return RandomPolicyValueNet()
