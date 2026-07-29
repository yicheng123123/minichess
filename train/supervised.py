"""train/supervised.py — Full supervised pretraining (AlphaGo-style SL stage).

This is the advisor's Phase 1 for the engineering route to a playable 7x7 AI:
instead of a light warm-start (train/warmstart.py, which applies the policy loss
only on ``teacher`` positions for 2-3 epochs to *avoid* imitation learning), this
module does the OPPOSITE on purpose — it imitates a strong alpha-beta teacher on
EVERY position until the network reproduces the teacher's move with high accuracy
(>90%). The goal is to give the network real board sense (basic tactics, captures,
progress, endgames) BEFORE self-play begins, so self-play starts from a competent
policy instead of a near-random one that only ever produces draws.

Differences from warmstart.py:
  * policy loss on ALL positions (not just ``teacher``) — imitate the move played;
  * uses ``F.cross_entropy`` against the teacher's move index (equivalent to soft
    cross-entropy on the one-hot target, but cheaper and lower memory);
  * reports **policy accuracy** (top-1 match with the teacher move) each epoch,
    on both a training and a held-out validation split, to track the >90% milestone
    and catch overfitting;
  * keeps data on CPU and streams mini-batches to the device so hundreds of
    thousands of positions fit without exhausting VRAM;
  * saves the final network and the best-validation-accuracy snapshot.

The teacher data is the standard expert JSONL (see selfplay/expert.py): each
sample's ``policy`` field is a one-hot over the move that was actually played, so
for a dataset where both sides play strong alpha-beta, every position's target is a
strong move.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

from nn.dataset import decode_planes
from utils.logger import logger

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from nn import loss as nn_loss

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError(
            "PyTorch is required for supervised pretraining; install torch."
        )


def _load_tensors(paths: Sequence[str]):
    """Flatten expert JSONL files into CPU tensors.

    Returns ``(planes, target_idx, value)``:
      * planes:     float32 ``[N, C, 7, 7]``
      * target_idx: int64   ``[N]`` — index of the teacher's move (argmax of the
                    stored one-hot policy target)
      * value:      float32 ``[N]`` — outcome from the mover's perspective
    """
    import json

    planes_list: List[np.ndarray] = []
    idx_list: List[int] = []
    val_list: List[float] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                for s in record.get("samples", []):
                    planes_list.append(np.asarray(decode_planes(s["planes"]),
                                                  dtype=np.float32))
                    pol = np.asarray(s["policy"], dtype=np.float32)
                    idx_list.append(int(pol.argmax()))
                    val_list.append(float(s["value"]))

    planes = torch.from_numpy(np.stack(planes_list))
    target_idx = torch.tensor(idx_list, dtype=torch.long)
    value = torch.tensor(val_list, dtype=torch.float32)
    return planes, target_idx, value


def supervised_pretrain(
    net,
    data_paths,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    value_weight: float = 1.0,
    val_fraction: float = 0.05,
    save_path: Optional[str] = None,
    device: Optional["torch.device"] = None,
    seed: int = 0,
) -> dict:
    """Supervisedly pretrain ``net`` in place to imitate the expert teacher.

    Args:
        net: A policy-value ``torch.nn.Module`` (``forward(planes) -> (logits,
            value)``).
        data_paths: One expert JSONL path or a list of them (combined).
        epochs: Number of passes over the data.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        weight_decay: L2 regularization coefficient (Adam weight_decay); counters
            the memorization/overfitting that a single-move imitation target tends
            to produce.
        value_weight: Weight on the value MSE term relative to the policy CE.
        val_fraction: Fraction of positions held out for validation accuracy.
        save_path: If given, save the final network here; the best-validation
            snapshot is saved alongside as ``<save_path>.best``.
        device: Torch device; auto-detected when omitted.
        seed: Shuffle seed for the train/val split.

    Returns:
        ``{"n_samples", "best_val_acc", "history"}`` where ``history`` is a list
        of per-epoch ``{train_policy_loss, train_value_loss, train_acc, val_acc}``.
    """
    _require_torch()
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    planes, target_idx, value = _load_tensors(data_paths)
    n = planes.size(0)
    if n == 0:
        logger.warning(f"No supervised samples found in {data_paths}")
        return {"n_samples": 0, "best_val_acc": 0.0, "history": []}

    # Deterministic train/val split.
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    n_val = int(n * val_fraction)
    val_set = torch.tensor(order[:n_val], dtype=torch.long)
    train_set = torch.tensor(order[n_val:], dtype=torch.long)
    logger.info(f"Supervised pretraining: {n} positions from {len(data_paths)} "
                f"file(s) | train={train_set.numel()} val={val_set.numel()} | "
                f"{epochs} epochs bs={batch_size} lr={lr} on {device}")

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    def _eval_accuracy(indices) -> float:
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for start in range(0, indices.numel(), batch_size):
                bi = indices[start:start + batch_size]
                bp = planes[bi].to(device)
                bt = target_idx[bi].to(device)
                logits, _v = net(bp)
                correct += int((logits.argmax(dim=-1) == bt).sum().item())
                total += bt.numel()
        net.train()
        return correct / total if total else 0.0

    history: List[dict] = []
    best_val_acc = 0.0
    indices = train_set.clone()
    for epoch in range(epochs):
        net.train()
        # Shuffle the training order each epoch.
        perm = torch.randperm(indices.numel())
        indices = train_set[perm]
        p_losses, v_losses, correct, total = [], [], 0, 0
        for start in range(0, indices.numel(), batch_size):
            bi = indices[start:start + batch_size]
            bp = planes[bi].to(device)
            bt = target_idx[bi].to(device)
            bv = value[bi].to(device)

            optimizer.zero_grad()
            logits, val_pred = net(bp)
            p_loss = F.cross_entropy(logits, bt)
            v_loss = nn_loss.value_loss(val_pred, bv)
            (p_loss + value_weight * v_loss).backward()
            optimizer.step()

            p_losses.append(float(p_loss.item()))
            v_losses.append(float(v_loss.item()))
            correct += int((logits.argmax(dim=-1) == bt).sum().item())
            total += bt.numel()

        train_acc = correct / total if total else 0.0
        val_acc = _eval_accuracy(val_set)
        row = {
            "train_policy_loss": sum(p_losses) / len(p_losses),
            "train_value_loss": sum(v_losses) / len(v_losses),
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        history.append(row)
        logger.info(f"[supervised] epoch {epoch + 1}/{epochs} "
                    f"p_loss={row['train_policy_loss']:.4f} "
                    f"v_loss={row['train_value_loss']:.4f} "
                    f"train_acc={100*train_acc:.2f}% val_acc={100*val_acc:.2f}%")

        if save_path:
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                torch.save(net.state_dict(), save_path + ".best")
            torch.save(net.state_dict(), save_path)

    if save_path:
        logger.info(f"Saved supervised net -> {save_path} "
                    f"(best val_acc={100*best_val_acc:.2f}% -> {save_path}.best)")
    return {"n_samples": n, "best_val_acc": best_val_acc, "history": history}
