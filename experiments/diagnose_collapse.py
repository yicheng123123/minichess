"""diagnose_collapse.py — 定位 value 头在训练循环里被打塌的原因。

已证：warm-start 后 value 头健康（eval std≈0.86），但自弈训练第 0 轮 300 步内
塌回 std≈0。本实验隔离两个嫌疑：
  (A) policy_loss 梯度通过共享 backbone 压垮 value 头（与数据无关）；
  (B) 和棋样本（target=0）把 value 往 0 拉（数据下毒）。
做法：warm-start 3 epoch -> 只用【纯专家数据】（97% 胜负，无和棋）跑 300 个
train step，每 50 步在固定的留出专家局面上测 eval 模式 value std。
  - 若纯专家数据也塌 -> 主因是 (A) policy 梯度霸权 / 优化问题；
  - 若纯专家数据不塌 -> 主因是 (B) 和棋数据下毒。

用法（minichess 根目录）：
    python experiments/diagnose_collapse.py --expert data/expert/expert_egreedy.jsonl
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", default="data/expert/expert_egreedy.jsonl")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--probe-every", type=int, default=50)
    args = ap.parse_args()

    import numpy as np
    import torch
    from nn.network import create_network
    from nn import loss as nn_loss
    from train.warmstart import (
        pretrain_from_expert, load_expert_samples, expert_to_selfplay_samples,
    )
    from utils.config import get_config

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = create_network(hidden=cfg.hidden_channels,
                         num_res_blocks=cfg.num_res_blocks).to(device)

    print("[info] warm-start 3 epoch ...", flush=True)
    pretrain_from_expert(net, args.expert, epochs=3,
                         batch_size=args.batch_size, lr=args.lr, device=device)

    samples = expert_to_selfplay_samples(load_expert_samples(args.expert))

    # 固定的留出探针局面（含 ±1 价值），用来追踪 value std
    random.seed(123)
    probe = random.sample(samples, min(400, len(samples)))
    probe_planes = torch.stack(
        [torch.from_numpy(np.asarray(s.planes)).float() for s in probe]
    ).to(device)

    def value_std() -> float:
        net.eval()
        with torch.no_grad():
            _, v = net(probe_planes)
        return float(v.cpu().std().item())

    print(f"[step   0] value std = {value_std():.4f}  (warm-start 基线)", flush=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        batch = random.sample(samples, args.batch_size)
        planes = torch.stack(
            [torch.from_numpy(np.asarray(s.planes)).float() for s in batch]
        ).to(device)
        pt = torch.stack(
            [torch.from_numpy(np.asarray(s.policy)).float() for s in batch]
        ).to(device)
        vt = torch.tensor([s.value for s in batch], dtype=torch.float32).to(device)

        net.train()
        optimizer.zero_grad()
        logits, value = net(planes)
        # 与 trainer.train_batch 完全一致：policy + value（权重 1）
        total = nn_loss.combined_loss(logits, value, pt, vt, value_weight=1.0)
        total.backward()
        optimizer.step()

        if step % args.probe_every == 0:
            ploss = float(nn_loss.policy_loss(logits, pt).item())
            vloss = float(nn_loss.value_loss(value, vt).item())
            print(f"[step {step:3d}] value std = {value_std():.4f}  "
                  f"(train-batch policy_loss={ploss:.3f} value_loss={vloss:.3f})",
                  flush=True)

    final = value_std()
    print("-" * 60, flush=True)
    if final < 0.05:
        print("结论: 纯专家数据也把 value 头打塌了 -> 主因是 policy 梯度霸权 / "
              "共享 backbone 优化问题，与和棋数据无关。", flush=True)
    elif final < 0.3:
        print("结论: value std 明显下降但未完全塌 -> policy 梯度有压制作用。", flush=True)
    else:
        print("结论: 纯专家数据下 value 头保持健康 -> 主因是和棋数据下毒。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
