"""diagnose_draws_parallel.py — 并行版：诊断自弈和棋的真实成因（标签质量检查）。

背景：player.play_game 把两种情况都记成 outcome=0：
  (a) 规则和棋 repetition
  (b) 撞 max_plies=200 上限被强制截断（局面可能一方已大优）
本脚本并行下一批棋，每局打上终止原因 + 终局子力差，用来回答：
到底有多少 "和棋" 其实是 "撞上限"，以及这些局是否存在 "多子却判和" 的假标签。

用法（minichess 根目录）：
    python experiments/diagnose_draws_parallel.py --games 24 --simulations 100
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 子力价值（粗略，仅用于诊断终局是否一方明显多子）
_PIECE_VALUE = {"K": 0, "R": 9, "N": 4, "C": 4.5, "P": 1}


def _play_one(args: dict) -> dict:
    """Worker：下一局，返回终止原因 + 终局信息。在子进程里运行。"""
    import random
    import numpy as np
    import torch

    from engine.board import Board
    from engine.piece import Color
    from engine.rules import game_result
    from search.mcts import MCTS
    from nn.network import create_network

    net_kwargs = args["net_kwargs"]
    state_dict = args["state_dict"]
    mcts_kwargs = args["mcts_kwargs"]
    max_plies = args["max_plies"]
    temperature = args["temperature"]
    temp_drop_after = args["temp_drop_after"]
    seed = args["seed"]
    device_str = args["device"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(device_str)
    net = create_network(**net_kwargs).to(device)
    net.load_state_dict(state_dict)
    net.eval()
    mcts = MCTS(**mcts_kwargs)

    board = Board()
    ply = 0
    while ply < max_plies:
        if game_result(board) is not None:
            break
        temp = temperature if ply < temp_drop_after else 0.0
        if temp > 0:
            _vc, mv = mcts.search_with_temperature(board, net, temp)
        else:
            _vc, mv = mcts.search(board, net)
        board.make_move(mv)
        ply += 1

    result = game_result(board)
    if result is None:
        reason, outcome = "max_plies", 0
    else:
        reason = result.reason
        ov = result.outcome.value
        outcome = 1 if ov == "red_wins" else -1 if ov == "black_wins" else 0

    # 终局子力差（红 - 黑）
    mat = 0.0
    for _sq, piece in board.pieces():
        v = _PIECE_VALUE.get(piece.ptype.value, 0)
        mat += v if piece.color is Color.RED else -v

    return {"reason": reason, "outcome": outcome, "plies": ply, "material": mat}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--simulations", type=int, default=100)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--temp-drop-after", type=int, default=30)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    import torch
    from nn.network import create_network
    from train.checkpoint import CheckpointManager
    from utils.config import get_config

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = create_network(hidden=cfg.hidden_channels,
                         num_res_blocks=cfg.num_res_blocks).to(device)
    ckpt = CheckpointManager(checkpoint_dir=cfg.checkpoint_dir)
    if ckpt.has_best():
        ckpt.load_best(net)
        src = "best.pt"
    else:
        src = f"latest iter={ckpt.load_latest(net)}"
    net.eval()
    state_dict = net.state_dict()
    # 把 state_dict 搬到 CPU 再分发，避免每个 worker 反序列化 GPU 张量
    state_dict = {k: v.cpu() for k, v in state_dict.items()}

    net_kwargs = {"hidden": cfg.hidden_channels, "num_res_blocks": cfg.num_res_blocks}
    mcts_kwargs = {"num_simulations": args.simulations, "c_puct": cfg.c_puct,
                   "dirichlet_alpha": cfg.dirichlet_alpha}

    workers = args.workers or min(os.cpu_count() or 4, args.games)
    print(f"[info] source={src} device={device} sims={args.simulations} "
          f"games={args.games} workers={workers}", flush=True)

    tasks = [{
        "net_kwargs": net_kwargs, "state_dict": state_dict,
        "mcts_kwargs": mcts_kwargs, "max_plies": args.max_plies,
        "temperature": args.temperature, "temp_drop_after": args.temp_drop_after,
        "seed": args.seed + i, "device": str(device),
    } for i in range(args.games)]

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_play_one, t): i for i, t in enumerate(tasks)}
        done = 0
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            rows.append(r)
            print(f"[{done:2d}/{args.games}] reason={r['reason']:<14} "
                  f"outcome={r['outcome']:+d} plies={r['plies']:3d} "
                  f"material(红-黑)={r['material']:+.1f}", flush=True)

    n = len(rows)
    counts: dict = {}
    for r in rows:
        counts[r["reason"]] = counts.get(r["reason"], 0) + 1
    print("-" * 60, flush=True)
    print(f"=== 终止原因分布 (共 {n} 局) ===", flush=True)
    for reason, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<16} {c:3d}  ({100.0*c/n:.1f}%)", flush=True)

    draws = [r for r in rows if r["outcome"] == 0]
    maxp = [r for r in draws if r["reason"] == "max_plies"]
    rep = [r for r in draws if r["reason"] == "repetition"]
    print("-" * 60, flush=True)
    print(f"=== 和棋细分 (共 {len(draws)} 和) ===", flush=True)
    print(f"  max_plies 撞上限   : {len(maxp)}", flush=True)
    print(f"  repetition 规则和棋: {len(rep)}", flush=True)

    if maxp:
        mats = [abs(r["material"]) for r in maxp]
        decisive = [m for m in mats if m >= 2.0]
        print("-" * 60, flush=True)
        print(f"=== 撞上限局终局子力差 |红-黑| (n={len(maxp)}) ===", flush=True)
        print(f"  平均 |子力差| = {sum(mats)/len(mats):.2f}", flush=True)
        print(f"  最大 |子力差| = {max(mats):.1f}", flush=True)
        print(f"  子力差>=2 (一方明显多子却判和): {len(decisive)} "
              f"({100.0*len(decisive)/len(maxp):.0f}% of max_plies)", flush=True)
        print("  逐局子力差:", ", ".join(f"{r['material']:+.1f}" for r in maxp),
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
