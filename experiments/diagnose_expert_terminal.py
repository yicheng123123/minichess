"""diagnose_expert_terminal.py — 直接重新生成 expert 局，记录真实终局方式。

目的：验证顾问路线第一阶段的前提——AB 教师到底会不会"杀"。
不调用重放（重放受镜像样本污染），而是现场用 alpha-beta 下棋，每局结束时
读取 game_result 的 reason（checkmate / king_captured / no_legal_moves /
repetition / max_plies）与终局子力差。

用法（minichess 根目录）：
    python experiments/diagnose_expert_terminal.py --games 40 --depth-high 3 --depth-low 2
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.board import Board
from engine.move_generator import legal_moves
from engine.piece import Color
from engine.rules import game_result, GameOutcome
from search.alphabeta import alphabeta

_PIECE_VALUE = {"K": 0, "R": 9, "N": 4, "C": 4.5, "P": 1}


def material_score(board: Board) -> float:
    s = 0.0
    for _sq, piece in board.pieces():
        v = _PIECE_VALUE.get(piece.ptype.value, 0)
        s += v if piece.color is Color.RED else -v
    return s


def play_one(depth_high: int, depth_low: int, high_plays_red: bool,
             max_plies: int, seed: int, opening_plies: int) -> dict:
    random.seed(seed)
    board = Board()
    high_color = Color.RED if high_plays_red else Color.BLACK

    opening = 0
    while opening < opening_plies:
        if game_result(board) is not None:
            break
        mvs = legal_moves(board)
        if not mvs:
            break
        board.make_move(random.choice(mvs))
        opening += 1

    ply = opening
    while ply < max_plies:
        if game_result(board) is not None:
            break
        mover = board.side_to_move
        depth = depth_high if mover is high_color else depth_low
        _score, move = alphabeta(board, depth=depth)
        if move is None:
            break
        board.make_move(move)
        ply += 1

    result = game_result(board)
    if result is None:
        reason, outcome = "max_plies", 0
    else:
        reason = result.reason
        if result.outcome is GameOutcome.RED_WINS:
            outcome = 1
        elif result.outcome is GameOutcome.BLACK_WINS:
            outcome = -1
        else:
            outcome = 0
    return {"reason": reason, "outcome": outcome, "plies": ply,
            "material": material_score(board)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--depth-high", type=int, default=3)
    ap.add_argument("--depth-low", type=int, default=2)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--opening-plies", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[info] AB depth_high={args.depth_high} vs depth_low={args.depth_low} "
          f"games={args.games} opening={args.opening_plies}", flush=True)
    rows = []
    for i in range(args.games):
        r = play_one(args.depth_high, args.depth_low,
                     high_plays_red=(i % 2 == 0),
                     max_plies=args.max_plies, seed=args.seed + i,
                     opening_plies=args.opening_plies)
        rows.append(r)
        print(f"[{i+1:2d}] reason={r['reason']:<14} outcome={r['outcome']:+d} "
              f"plies={r['plies']:3d} material(红-黑)={r['material']:+.1f}",
              flush=True)

    n = len(rows)
    counts: dict = {}
    for r in rows:
        counts[r["reason"]] = counts.get(r["reason"], 0) + 1
    decisive = [r for r in rows if r["outcome"] != 0]
    print("-" * 60, flush=True)
    print(f"=== 终局方式分布 (共 {n} 局) ===", flush=True)
    for reason, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<16} {c:3d}  ({100.0*c/n:.1f}%)", flush=True)
    print(f"胜负局 {len(decisive)} ({100*len(decisive)/n:.0f}%)  "
          f"和棋 {n-len(decisive)}", flush=True)
    if decisive:
        # 胜负局里按终局方式细分（关键：多少是 checkmate）
        sub: dict = {}
        for r in decisive:
            sub[r["reason"]] = sub.get(r["reason"], 0) + 1
        print("  胜负局终局方式:", ", ".join(f"{k}={v}" for k, v in sub.items()),
              flush=True)
        mats = [abs(r["material"]) for r in decisive]
        print(f"  胜负局终局 |子力差|: 平均={sum(mats)/len(mats):.1f} "
              f"最大={max(mats):.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
