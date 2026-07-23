"""nn package — Neural network, dataset and losses for Mini Xiangqi.

This package houses the deep-learning side of the project:

  * :mod:`nn.network` — the residual policy-value network, the
    :class:`PolicyValueNet` interface, the move<->index helpers and the
    ``NUM_MOVE_ACTIONS`` constant.
  * :mod:`nn.dataset` — self-play sample storage (JSONL) and PyTorch dataset
    wrappers.
  * :mod:`nn.loss` — policy / value / combined loss functions and an L2 helper.

PyTorch is optional. The package imports cleanly without torch; the pure-Python
:class:`RandomPolicyValueNet` fallback and the JSONL dataset utilities remain
available, while the torch-only symbols (``TorchPolicyValueNet``,
``create_network``, ``TorchSelfPlayDataset``, ``collate_fn`` and the loss
functions) are present only when torch is installed.

Convenient re-exports::

    from nn import NUM_MOVE_ACTIONS, move_to_index, index_to_move
    from nn import PolicyValueNet, RandomPolicyValueNet, default_net
    from nn import SelfPlaySample, SelfPlayDataset
"""

from __future__ import annotations

from .network import (
    NUM_MOVE_ACTIONS,
    move_to_index,
    index_to_move,
    PolicyValueNet,
    RandomPolicyValueNet,
    default_net,
)
from .dataset import (
    SelfPlaySample,
    SelfPlayDataset,
    encode_planes,
    decode_planes,
)

__all__ = [
    # network
    "NUM_MOVE_ACTIONS",
    "move_to_index",
    "index_to_move",
    "PolicyValueNet",
    "RandomPolicyValueNet",
    "default_net",
    # dataset
    "SelfPlaySample",
    "SelfPlayDataset",
    "encode_planes",
    "decode_planes",
]

# Torch-only symbols are re-exported lazily so the package still imports when
# torch is absent. Accessing them without torch raises a clear AttributeError.
try:  # pragma: no cover - depends on environment
    from .network import TorchPolicyValueNet, create_network, ResidualBlock
    from .dataset import TorchSelfPlayDataset, collate_fn
    from . import loss

    __all__ += [
        "TorchPolicyValueNet",
        "create_network",
        "ResidualBlock",
        "TorchSelfPlayDataset",
        "collate_fn",
        "loss",
    ]
except ImportError:
    pass
