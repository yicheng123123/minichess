"""eval_sl_strength.py — 实战棋力评估：SL 网络 + MCTS 对打 alpha-beta 搜索。

顾问路线转向后的核心指标：不再看 SL 的"模仿准确率"，而看**实战棋力**——SL 网络
配上 MCTS 能不能逼平甚至赢过它的老师（alpha-beta 搜索）。若能，则"SL + MCTS"
本身就已经是一个可用的 Mini-AlphaGo，self-play 只负责后续微调。

每局让 SL+MCTS 执一方、AB(depth) 执另一方，双向执色各下若干局以消除先后手偏差。
并行（MCTS 用 GPU、AB 用 CPU；为避开单笔记本 GPU 多进程争用，默认少量 worker）。

用法（minichess 根目录）：
    python experiments/eval_sl_strength.py --model models/sl_net.pt.best \
        --games 20 --sims 100 --ab-depth 3
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _greedy_policy_move(board, net, device):
    """Pick the legal move with the highest raw policy logit (no MCTS)."""
    import numpy as np
    import torch
    from engine.move_generator import legal_moves
    from nn.network import move_to_index

    moves = legal_moves(board)
    if not moves:
        return None
    planes = torch.from_numpy(board.to_planes()).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _value = net(planes)
    logits = logits.squeeze(0).cpu().numpy()
    idxs = [move_to_index(m) for m in moves]
    best = int(np.argmax([logits[i] for i in idxs]))
    return moves[best]


def _play_one(args: dict) -> dict:
    """Worker: 下一局 SL+MCTS vs AB，返回相对 SL 视角的结果。"""
    import random as _random
    import torch
    from engine.board import Board
    from engine.piece import Color
    from engine.rules import game_result
    from search.mcts import MCTS
    from search.alphabeta import alphabeta
    from nn.network import create_network

    _random.seed(args["seed"])
    device = torch.device(args["device"])
    net = create_network(**args["net_kwargs"]).to(device)
    net.load_state_dict(args["state_dict"])
    net.eval()
    mcts = MCTS(num_simulations=args["sims"], c_puct=args["c_puct"],
                add_noise=False)  # 评估用干净着法，不加 Dirichlet 噪声

    board = Board()
    sl_color = Color.RED if args["sl_plays_red"] else Color.BLACK
    max_plies = args["max_plies"]
    ab_depth = args["ab_depth"]
    policy_only = args.get("policy_only", False)

    ply = 0
    while ply < max_plies:
        if game_result(board) is not None:
            break
        mover = board.side_to_move
        if mover is sl_color:
            if policy_only:
                mv = _greedy_policy_move(board, net, device)
            else:
                _probs, mv = mcts.search(board, net)
        else:
            _score, mv = alphabeta(board, depth=ab_depth)
        if mv is None:
            break
        board.make_move(mv)
        ply += 1

    result = game_result(board)
    if result is None:
        outcome, reason = 0, "max_plies"
    else:
        reason = result.reason
        ov = result.outcome.value
        outcome = 1 if ov == "red_wins" else -1 if ov == "black_wins" else 0
    if sl_color is Color.BLACK:   # 统一换算成 SL 视角
        outcome = -outcome
    return {"outcome": outcome, "reason": reason, "plies": ply,
            "sl_red": args["sl_plays_red"]}


def _summarize(name: str, rows: list) -> None:
    n = len(rows)
    if n == 0:
        print(f"  {name}: (无结果)", flush=True)
        return
    w = sum(1 for r in rows if r["outcome"] > 0)
    l = sum(1 for r in rows if r["outcome"] < 0)
    d = n - w - l
    mate = sum(1 for r in rows if r["outcome"] > 0 and r["reason"] == "checkmate")
    lost = sum(1 for r in rows if r["outcome"] < 0 and r["reason"] == "checkmate")
    rep = sum(1 for r in rows if r["reason"] == "repetition")
    mp = sum(1 for r in rows if r["reason"] == "max_plies")
    plies = [r["plies"] for r in rows]
    score = (w + 0.5 * d) / n
    print(f"  {name:<26} n={n:3d} | SL胜={w:3d} 和={d:3d} 负={l:3d} "
          f"score={score:.2f} | SL杀={mate} 被杀={lost} "
          f"repetition={rep} maxplies={mp} | avgplies={sum(plies)/n:.0f}",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="SL 网络 .pt 路径")
    ap.add_argument("--games", type=int, default=20, help="每方执色局数")
    ap.add_argument("--sims", type=int, default=100, help="SL 方 MCTS 模拟数")
    ap.add_argument("--ab-depth", type=int, default=3, help="AB 对手搜索深度")
    ap.add_argument("--policy-only", action="store_true",
                    help="SL 方只用 greedy policy（argmax），不跑 MCTS——"
                         "用于诊断 policy 单独棋力 vs MCTS 增益")
    ap.add_argument("--c-puct", type=float, default=None)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import torch
    from nn.network import create_network
    from utils.config import get_config

    cfg = get_config()
    c_puct = args.c_puct if args.c_puct is not None else cfg.c_puct
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = create_network(hidden=cfg.hidden_channels,
                         num_res_blocks=cfg.num_res_blocks).to(device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.eval()
    state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
    net_kwargs = {"hidden": cfg.hidden_channels, "num_res_blocks": cfg.num_res_blocks}
    workers = min(args.workers, os.cpu_count() or 4)

    print(f"[info] SL net from {args.model} | device={device} | "
          f"MCTS sims={args.sims} c_puct={c_puct} | AB opponent depth={args.ab_depth} "
          f"| {args.games} games/color | workers={workers}", flush=True)

    tasks = []
    for i in range(args.games):
        for sl_red in (True, False):
            tasks.append({
                "state_dict": state_dict, "net_kwargs": net_kwargs, "device": device,
                "sims": args.sims, "c_puct": c_puct, "ab_depth": args.ab_depth,
                "max_plies": args.max_plies, "sl_plays_red": sl_red,
                "policy_only": args.policy_only,
                "seed": args.seed + i + (10000 if not sl_red else 0),
            })

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_play_one, t) for t in tasks]
        done = 0
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  progress: {done}/{len(tasks)} games", flush=True)

    print("\n" + "=" * 70, flush=True)
    _summarize(f"SL+MCTS vs AB(d{args.ab_depth}) 全部", rows)
    _summarize("  SL 执红", [r for r in rows if r["sl_red"]])
    _summarize("  SL 执黑", [r for r in rows if not r["sl_red"]])
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
