"""diagnose_draws.py — 诊断自弈和棋的真实成因（标签质量检查）。

背景：训练中自弈几乎 100% 和棋，value target 全是 0。但 player.play_game 把
两种完全不同的情况都记成 outcome=0：
  (a) 规则和棋（repetition，三次重复局面）
  (b) 撞到 max_plies=200 上限被强制截断（此时局面可能一方已大优）
本脚本复刻训练用的自弈循环，但给每局打上终止原因，并统计终局子力差，
用来回答：到底有多少 "和棋" 其实是 "撞上限"，以及这些被截断的局面是否
存在 "多子却判和" 的假标签。

用法（在 minichess 根目录）：
    python experiments/diagnose_draws.py --games 20 --simulations 100
"""

from __future__ import annotations

import argparse
import os
import sys

# 让脚本能从 experiments/ 子目录 import 项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from engine.board import Board
from engine.piece import Color, PieceType
from engine.rules import game_result
from search.mcts import MCTS
from nn.network import create_network
from train.checkpoint import CheckpointManager
from utils.config import get_config

# 子力价值（粗略，仅用于诊断"终局是否一方明显多子"）
PIECE_VALUE = {
    PieceType.KING: 0,
    PieceType.ROOK: 9,
    PieceType.HORSE: 4,
    PieceType.CANNON: 4.5,
    PieceType.SOLDIER: 1,
}


def material_score(board: Board) -> float:
    """红方子力 - 黑方子力（正数=红优）。"""
    score = 0.0
    for _sq, piece in board.pieces():
        v = PIECE_VALUE.get(piece.ptype, 0)
        score += v if piece.color is Color.RED else -v
    return score


def play_one_diagnostic(net, mcts, max_plies: int, temperature: float,
                        temp_drop_after: int) -> dict:
    """下一局，返回终止原因 + 终局信息。"""
    board = Board()
    ply = 0
    while ply < max_plies:
        result = game_result(board)
        if result is not None:
            break
        current_temp = temperature if ply < temp_drop_after else 0.0
        if current_temp > 0:
            _vc, best_move = mcts.search_with_temperature(board, net, current_temp)
        else:
            _vc, best_move = mcts.search(board, net)
        board.make_move(best_move)
        ply += 1

    result = game_result(board)
    if result is None:
        reason = "max_plies"          # 撞上限截断
        outcome = 0
    else:
        reason = result.reason        # repetition / checkmate / no_legal_moves / king_captured
        if result.outcome.value == "red_wins":
            outcome = 1
        elif result.outcome.value == "black_wins":
            outcome = -1
        else:
            outcome = 0

    return {
        "reason": reason,
        "outcome": outcome,
        "plies": ply,
        "material": material_score(board),   # 红 - 黑
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--simulations", type=int, default=100)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--temp-drop-after", type=int, default=30)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = create_network(hidden=cfg.hidden_channels,
                         num_res_blocks=cfg.num_res_blocks).to(device)

    ckpt = CheckpointManager(checkpoint_dir=cfg.checkpoint_dir)
    if ckpt.has_best():
        ckpt.load_best(net)
        print(f"[info] loaded best.pt from {cfg.checkpoint_dir}")
    else:
        it = ckpt.load_latest(net)
        print(f"[info] no best.pt; loaded latest checkpoint iter={it}")
    net.eval()

    mcts = MCTS(num_simulations=args.simulations, c_puct=cfg.c_puct,
                dirichlet_alpha=cfg.dirichlet_alpha)

    print(f"[info] device={device} sims={args.simulations} "
          f"max_plies={args.max_plies} games={args.games}")
    print("-" * 60)

    reason_counts: dict = {}
    rows = []
    for i in range(args.games):
        r = play_one_diagnostic(net, mcts, args.max_plies,
                                args.temperature, args.temp_drop_after)
        reason_counts[r["reason"]] = reason_counts.get(r["reason"], 0) + 1
        rows.append(r)
        print(f"game {i+1:2d}: reason={r['reason']:<14} "
              f"outcome={r['outcome']:+d} plies={r['plies']:3d} "
              f"material(红-黑)={r['material']:+.1f}")

    print("-" * 60)
    n = len(rows)
    print(f"=== 终止原因分布 (共 {n} 局) ===")
    for reason, c in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<16} {c:3d}  ({100.0*c/n:.1f}%)")

    draws = [r for r in rows if r["outcome"] == 0]
    maxp = [r for r in draws if r["reason"] == "max_plies"]
    rep = [r for r in draws if r["reason"] == "repetition"]
    print("-" * 60)
    print(f"=== 和棋细分 (共 {len(draws)} 和) ===")
    print(f"  max_plies 撞上限   : {len(maxp)}")
    print(f"  repetition 规则和棋: {len(rep)}")

    if maxp:
        mats = [abs(r["material"]) for r in maxp]
        decisive = [m for m in mats if m >= 2.0]   # 终局子力差>=2 视为"其实分出优劣"
        print("-" * 60)
        print(f"=== 撞上限局的终局子力差 |红-黑| (n={len(maxp)}) ===")
        print(f"  平均 |子力差| = {sum(mats)/len(mats):.2f}")
        print(f"  最大 |子力差| = {max(mats):.1f}")
        print(f"  子力差>=2 (一方明显多子却判和) 的局数: {len(decisive)} "
              f"({100.0*len(decisive)/len(maxp):.0f}% of max_plies)")
        print("  逐局子力差:", ", ".join(f"{r['material']:+.1f}" for r in maxp))

    return 0


if __name__ == "__main__":
    sys.exit(main())
