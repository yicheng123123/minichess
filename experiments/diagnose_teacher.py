"""diagnose_teacher.py — 验证 "AB(depth) vs Random" 教师是否产生大量将杀。

顾问路线第一阶段：用 AB vs Random 热启动教 policy "优势之后怎么将死"。
前提假设：Random 疯狂送子，AB 会打出短促干净的多步将杀。本脚本实测：
让一方用 alpha-beta(depth)，另一方完全随机，统计终局方式
（checkmate / king_captured / no_legal_moves / repetition / max_plies）、
胜负率、终局步数与子力差。若 checkmate 占比高、局短，则假设成立。

用法（minichess 根目录）：
    python experiments/diagnose_teacher.py --games 16 --depth 3
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


def play_one(depth: int, ab_plays_red: bool, max_plies: int, seed: int) -> dict:
    random.seed(seed)
    board = Board()
    ab_color = Color.RED if ab_plays_red else Color.BLACK
    ply = 0
    while ply < max_plies:
        if game_result(board) is not None:
            break
        mover = board.side_to_move
        mvs = legal_moves(board)
        if not mvs:
            break
        if mover is ab_color:
            _score, move = alphabeta(board, depth=depth)
            if move is None:
                move = random.choice(mvs)
        else:
            move = random.choice(mvs)
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
    # AB 视角的胜负
    ab_won = (outcome == 1) == ab_plays_red if outcome != 0 else None
    return {"reason": reason, "outcome": outcome, "plies": ply,
            "material": material_score(board), "ab_won": ab_won}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    print(f"[info] AB(depth={args.depth}) vs Random, {args.games} games "
          f"(AB 红黑交替)", flush=True)
    rows = []
    for i in range(args.games):
        r = play_one(args.depth, ab_plays_red=(i % 2 == 0),
                     max_plies=args.max_plies, seed=args.seed + i)
        rows.append(r)
        ab = "win" if r["ab_won"] is True else "loss" if r["ab_won"] is False else "draw"
        print(f"[{i+1:2d}] reason={r['reason']:<14} plies={r['plies']:3d} "
              f"material={r['material']:+.1f}  AB={ab}", flush=True)

    n = len(rows)
    counts: dict = {}
    for r in rows:
        counts[r["reason"]] = counts.get(r["reason"], 0) + 1
    print("-" * 60, flush=True)
    print(f"=== 终局方式分布 (共 {n} 局) ===", flush=True)
    for reason, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<16} {c:3d}  ({100.0*c/n:.1f}%)", flush=True)
    ab_wins = sum(1 for r in rows if r["ab_won"] is True)
    ab_loss = sum(1 for r in rows if r["ab_won"] is False)
    decisive = [r for r in rows if r["outcome"] != 0]
    plies_list = [r["plies"] for r in decisive] or [0]
    print(f"AB 胜={ab_wins} AB 负={ab_loss} 和={n-ab_wins-ab_loss}  "
          f"胜负局平均步数={sum(plies_list)/len(plies_list):.0f}", flush=True)
    cm = counts.get("checkmate", 0)
    print(f"将杀(checkmate)占比: {cm}/{n} = {100*cm/n:.0f}%", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
