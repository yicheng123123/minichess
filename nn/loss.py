"""nn/loss.py — Loss functions for training the Mini Xiangqi policy-value net.

The network produces two outputs per position:

  * ``logits`` — raw policy logits over the ``NUM_MOVE_ACTIONS`` move space, and
  * ``value``  — a scalar in [-1, 1] estimating the game outcome.

Training matches these against self-play targets:

  * a **soft** policy target (the MCTS visit-count distribution), and
  * a scalar value target ``z`` in {-1, 0, +1} from the mover's perspective.

All functions here operate on PyTorch tensors. PyTorch is optional at import
time so the rest of the ``nn`` package stays importable without a deep-learning
stack; calling any function in this module without torch raises ``RuntimeError``.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError(
            "PyTorch is required for nn.loss; install torch to train the network."
        )


def policy_loss(logits: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """Cross-entropy between predicted logits and a *soft* policy target.

    Unlike ``F.cross_entropy`` with an integer class index, the target here is a
    full probability distribution (the MCTS visit counts normalized to sum to 1).
    The loss is the mean over the batch of::

        -sum_a target[a] * log_softmax(logits)[a]

    Args:
        logits: ``[B, NUM_MOVE_ACTIONS]`` raw, unnormalized policy logits.
        target: ``[B, NUM_MOVE_ACTIONS]`` target distribution (non-negative; it
            is renormalized to sum to 1 along the action axis for safety).

    Returns:
        A scalar tensor: the mean soft cross-entropy over the batch.
    """
    _require_torch()
    log_probs = F.log_softmax(logits, dim=-1)
    # Renormalize the target so slightly-off distributions (e.g. unnormalized
    # visit counts) still define a valid soft label.
    target = target.to(log_probs.dtype)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    per_example = -(target * log_probs).sum(dim=-1)
    return per_example.mean()


def value_loss(predicted: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """Mean squared error between the predicted value and the target outcome.

    Args:
        predicted: ``[B]`` network value head output (in [-1, 1]).
        target: ``[B]`` ground-truth outcome ``z`` in {-1, 0, +1}.

    Returns:
        A scalar tensor: the mean squared error over the batch.
    """
    _require_torch()
    return F.mse_loss(predicted.reshape(-1), target.reshape(-1).to(predicted.dtype))


def combined_loss(
    logits: "torch.Tensor",
    value: "torch.Tensor",
    policy_target: "torch.Tensor",
    value_target: "torch.Tensor",
    value_weight: float = 1.0,
) -> "torch.Tensor":
    """Weighted sum of the (soft) policy cross-entropy and the value MSE.

    ``total = policy_loss + value_weight * value_loss``

    The policy term is kept at unit weight; ``value_weight`` scales the value
    term so the two heads can be balanced during training.

    Args:
        logits: ``[B, NUM_MOVE_ACTIONS]`` policy logits.
        value: ``[B]`` predicted value.
        policy_target: ``[B, NUM_MOVE_ACTIONS]`` soft policy target.
        value_target: ``[B]`` value target ``z``.
        value_weight: Scalar weight applied to the value loss.

    Returns:
        A scalar tensor combining both losses.
    """
    _require_torch()
    p_loss = policy_loss(logits, policy_target)
    v_loss = value_loss(value, value_target)
    return p_loss + value_weight * v_loss


def l2_regularization(model, weight_decay: float = 1e-4) -> "torch.Tensor":
    """Sum of squared L2 norms of the model's (non-batch-norm) parameters.

    Intended to be added to the training loss as ``weight_decay * reg`` when the
    optimizer is not already applying weight decay. Batch-norm parameters are
    excluded by convention (they are not regularized).

    Args:
        model: A ``torch.nn.Module`` (e.g. the policy-value net).
        weight_decay: Multiplicative coefficient applied to the squared norms.

    Returns:
        A scalar tensor ``weight_decay * sum(p.pow(2).sum())`` over regularized
        parameters. Returns a zero tensor (on CPU) if there are no parameters.
    """
    _require_torch()
    reg = None
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Skip batch-norm weights/biases by name convention.
        if "bn" in name or "batch_norm" in name:
            continue
        term = p.pow(2).sum()
        reg = term if reg is None else reg + term
    if reg is None:
        return torch.zeros((), dtype=torch.float32)
    return weight_decay * reg
