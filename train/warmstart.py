"""train/warmstart.py — Generic warm-start (supervised pretraining).

Loads an "expert" JSONL dataset (see :mod:`selfplay.expert` for the format)
and pretrains a policy-value network on it to break the cold-start spiral
where a value head that has only ever seen draws keeps outputting zero:

  * the **value** loss is applied to *every* position, so the network learns
    what winning and losing actually look like; and
  * the **policy** loss is applied only to positions flagged ``teacher``, so
    the network imitates the strong side's moves and ignores the weak side's
    blunders.

The module is teacher-agnostic: anything that emits the expert JSONL format
(alpha-beta search, human games, an older network) can warm-start training
through this single interface.

Example::

    from train.warmstart import pretrain_from_expert
    pretrain_from_expert(net, "data/expert/expert.jsonl", epochs=2)
"""

from __future__ import annotations

import json
import random
from typing import List, Optional

from nn.dataset import decode_planes, SelfPlaySample
from utils.logger import logger

try:
    import numpy as np
    import torch
    from nn import loss as nn_loss

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError(
            "PyTorch is required for warm-start pretraining; install torch."
        )


def load_expert_samples(path: str) -> List[dict]:
    """Flatten every sample (with its teacher flag) out of an expert JSONL file."""
    samples: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            samples.extend(record.get("samples", []))
    return samples


def expert_to_selfplay_samples(raw_samples: List[dict]) -> List[SelfPlaySample]:
    """Convert raw expert dicts into :class:`SelfPlaySample` (drops the teacher
    flag) so they can be seeded into the normal replay buffer."""
    out: List[SelfPlaySample] = []
    for s in raw_samples:
        out.append(SelfPlaySample(
            planes=decode_planes(s["planes"]),
            policy=np.asarray(s["policy"], dtype=np.float32),
            value=float(s["value"]),
            move=s["move"],
        ))
    return out


def pretrain_from_expert(
    net,
    expert_path: str,
    epochs: int = 2,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: Optional["torch.device"] = None,
) -> dict:
    """Pretrain ``net`` in place on expert data.

    Args:
        net: A policy-value ``torch.nn.Module`` with ``forward(planes) ->
            (logits, value)``.
        expert_path: Path to an expert JSONL file.
        epochs: Number of passes over the expert data (keep small — 2-3 — to
            kick-start without slipping into full imitation learning).
        batch_size: Mini-batch size.
        lr: Learning rate for the pretraining optimizer.
        device: Torch device; auto-detected when omitted.

    Returns:
        ``{"policy_loss", "value_loss", "n_samples"}`` averaged over training.
    """
    _require_torch()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = load_expert_samples(expert_path)
    if not raw:
        logger.warning(f"No expert samples found in {expert_path}")
        return {"policy_loss": 0.0, "value_loss": 0.0, "n_samples": 0}

    # Decode everything once into tensors on the target device.
    planes = torch.stack(
        [torch.from_numpy(np.asarray(decode_planes(s["planes"]))).float()
         for s in raw]
    ).to(device)
    policy_target = torch.stack(
        [torch.from_numpy(np.asarray(s["policy"], dtype=np.float32)).float()
         for s in raw]
    ).to(device)
    value_target = torch.tensor(
        [float(s["value"]) for s in raw], dtype=torch.float32
    ).to(device)
    teacher_mask = torch.tensor(
        [bool(s.get("teacher", True)) for s in raw], dtype=torch.bool
    ).to(device)

    n = len(raw)
    n_teacher = int(teacher_mask.sum().item())
    logger.info(f"Warm-start: {n} expert samples ({n_teacher} teacher positions) "
                f"from {expert_path}; {epochs} epochs on {device}")

    net.to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    stats: List[tuple] = []
    indices = list(range(n))
    for epoch in range(epochs):
        random.shuffle(indices)
        for start in range(0, n, batch_size):
            batch_idx = torch.tensor(indices[start:start + batch_size], device=device)
            b_planes = planes[batch_idx]
            b_policy = policy_target[batch_idx]
            b_value = value_target[batch_idx]
            b_teacher = teacher_mask[batch_idx]

            optimizer.zero_grad()
            logits, value = net(b_planes)

            # Value loss on ALL positions.
            v_loss = nn_loss.value_loss(value, b_value)
            # Policy loss only on teacher positions.
            if b_teacher.any():
                p_loss = nn_loss.policy_loss(logits[b_teacher], b_policy[b_teacher])
            else:
                p_loss = torch.zeros((), device=device)

            (p_loss + v_loss).backward()
            optimizer.step()
            stats.append((float(p_loss.item()), float(v_loss.item())))

        avg_p = sum(s[0] for s in stats) / len(stats)
        avg_v = sum(s[1] for s in stats) / len(stats)
        logger.info(f"[warm-start] epoch {epoch + 1}/{epochs} "
                    f"policy_loss={avg_p:.4f} value_loss={avg_v:.4f}")

    avg_p = sum(s[0] for s in stats) / len(stats)
    avg_v = sum(s[1] for s in stats) / len(stats)
    return {"policy_loss": avg_p, "value_loss": avg_v, "n_samples": n}
