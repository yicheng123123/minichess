"""diagnose_bn.py — 验证 value 头是否存在 BatchNorm train/eval 不一致。

假设：value_loss 在 train() 模式（用 batch 统计）能拟合到 0.25，但 value_health
和 MCTS 自弈用的 eval() 模式（用 running stats）输出塌成常数 std≈0。若成立，
则 value 头的"学习"对推理完全不可见——MCTS 永远拿不到价值信号。

做法：新建网络 -> 照 Exp3 热启动 3 epoch -> 立刻在同一批专家局面上分别测
eval/train 模式的 value 输出分布，并打印 value_conv 里 BatchNorm 的 running 统计。

用法（minichess 根目录）：
    python experiments/diagnose_bn.py --expert data/expert/expert_egreedy.jsonl
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
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()

    import numpy as np
    import torch
    from nn.network import create_network
    from train.warmstart import (
        pretrain_from_expert, load_expert_samples, expert_to_selfplay_samples,
    )
    from utils.config import get_config

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = create_network(hidden=cfg.hidden_channels,
                         num_res_blocks=cfg.num_res_blocks).to(device)

    print(f"[info] 热启动 {args.epochs} epoch（复刻 Exp3）...", flush=True)
    pretrain_from_expert(net, args.expert, epochs=args.epochs,
                         batch_size=256, lr=1e-3, device=device)

    # 取一批多样化的专家局面（含 ±1 价值）
    raw = load_expert_samples(args.expert)
    samples = expert_to_selfplay_samples(raw)
    random.seed(0)
    sel = random.sample(samples, min(args.n, len(samples)))
    planes = torch.stack(
        [torch.from_numpy(np.asarray(s.planes)).float() for s in sel]
    ).to(device)
    targets = torch.tensor([s.value for s in sel], dtype=torch.float32).to(device)

    # EVAL 模式（MCTS / value_health 用的）
    net.eval()
    with torch.no_grad():
        _, v_eval = net(planes)
    v_eval = v_eval.cpu()

    # TRAIN 模式（训练算 value_loss 用的）
    net.train()
    _, v_train = net(planes)
    v_train = v_train.detach().cpu()

    print("-" * 60, flush=True)
    print(f"targets     : mean={targets.mean():+.3f} std={targets.std():.3f}", flush=True)
    print(f"EVAL  value : mean={v_eval.mean():+.4f} std={v_eval.std():.4f} "
          f"min={v_eval.min():+.3f} max={v_eval.max():+.3f}", flush=True)
    print(f"TRAIN value : mean={v_train.mean():+.4f} std={v_train.std():.4f} "
          f"min={v_train.min():+.3f} max={v_train.max():+.3f}", flush=True)

    # 两种模式的 MSE 对比
    mse_eval = float(((v_eval - targets.cpu()) ** 2).mean())
    mse_train = float(((v_train - targets.cpu()) ** 2).mean())
    print(f"MSE  eval={mse_eval:.4f}   train={mse_train:.4f}", flush=True)

    # 检查 value_conv 的 BatchNorm running 统计
    bn = net.value_conv[1]
    print("-" * 60, flush=True)
    print(f"value_conv BN running_mean={bn.running_mean.cpu().numpy()}", flush=True)
    print(f"value_conv BN running_var ={bn.running_var.cpu().numpy()}", flush=True)
    print(f"value_conv BN weight(gamma)={bn.weight.detach().cpu().numpy()} "
          f"bias(beta)={bn.bias.detach().cpu().numpy()}", flush=True)

    print("-" * 60, flush=True)
    if v_eval.std() < 0.05 and v_train.std() > 0.1:
        print("结论: 确认 BatchNorm train/eval 不一致 —— eval 模式 value 塌缩，"
              "train 模式正常。MCTS 自弈拿到的价值信号恒为常数。", flush=True)
    elif v_eval.std() < 0.05 and v_train.std() < 0.05:
        print("结论: 两种模式都塌缩 —— value 头整体没学起来（非 BN 模式问题）。", flush=True)
    else:
        print("结论: eval 模式 value 有分化，BN 不一致假设不成立。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
