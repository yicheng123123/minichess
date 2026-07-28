"""analyze_expert.py — 审计 expert 数据的终局构成（验证热启动教师是否会"杀"）。

顾问路线第一阶段：用 100 局 AB 热启动教 policy "优势之后怎么结束"。
前提：expert 数据里得有真正的终结样本。本脚本重放 expert.jsonl 里每局的
着法序列，精确还原终局方式（checkmate / king_captured / no_legal_moves /
repetition / max_plies），并统计终局子力差，回答：
  - expert 的胜负局是怎么赢的？
  - 有多少是和棋（value 全 0 的"废局"）？
  - 终局子力差分布如何？

用法（minichess 根目录）：
    python experiments/analyze_expert.py --path data/expert/expert.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.board import Board
from engine.move import Move
from engine.move_generator import legal_moves
from engine.piece import Color
from engine.rules import game_result

_PIECE_VALUE = {"K": 0, "R": 9, "N": 4, "C": 4.5, "P": 1}


def material_score(board: Board) -> float:
    s = 0.0
    for _sq, piece in board.pieces():
        v = _PIECE_VALUE.get(piece.ptype.value, 0)
        s += v if piece.color is Color.RED else -v
    return s


def analyze_game(record: dict) -> dict:
    """重放一局 expert 着法，精确还原终局并读出终局原因。

    expert.py 生成时 ``samples = 原始 + 镜像``（镜像追加在后），且记录了真实
    步数 ``record["plies"]``。因此前 ``plies`` 个样本恰好就是原始着法序列，
    逐步重放即可还原终局局面，镜像样本根本不进入循环。
    """
    samples = record["samples"]
    board = Board()
    plies = 0
    for s in samples:
        if game_result(board) is not None:
            break
        try:
            mv = Move.from_uci(s["move"])
        except Exception:
            continue
        # 只走当前局面下合法的着法；镜像/错位样本通常非法，自然被跳过。
        if mv not in legal_moves(board):
            continue
        board.make_move(mv)
        plies += 1
    result = game_result(board)
    reason = result.reason if result is not None else "max_plies"
    return {
        "recorded_outcome": record.get("outcome"),
        "reason": reason,
        "plies": plies,
        "material": material_score(board),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/expert/expert.jsonl")
    ap.add_argument("--show", type=int, default=0,
                    help="逐局打印前 N 局（默认 0 不打印）")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"[error] 找不到 {args.path}")
        return 1

    rows = []
    with open(args.path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append(analyze_game(rec))

    n = len(rows)
    print(f"=== expert 数据审计: {args.path} (共 {n} 局) ===")
    print("-" * 60)

    # 按记录的 outcome 分类
    rec_red = sum(1 for r in rows if r["recorded_outcome"] == 1)
    rec_black = sum(1 for r in rows if r["recorded_outcome"] == -1)
    rec_draw = sum(1 for r in rows if r["recorded_outcome"] == 0)
    print(f"记录 outcome: 红胜={rec_red} 黑胜={rec_black} 和={rec_draw} "
          f"(胜负局 {rec_red+rec_black}, {100*(rec_red+rec_black)/n:.0f}%)")
    print("-" * 60)

    # 按重放还原的终局原因分类
    counts: dict = {}
    for r in rows:
        counts[r["reason"]] = counts.get(r["reason"], 0) + 1
    print("重放还原的终局方式:")
    for reason, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<16} {c:3d}  ({100.0*c/n:.1f}%)")
    print("-" * 60)

    # 胜负局的终局子力差
    decisive = [r for r in rows if r["recorded_outcome"] != 0]
    if decisive:
        mats = [abs(r["material"]) for r in decisive]
        print(f"胜负局 (n={len(decisive)}) 终局 |子力差|: "
              f"平均={sum(mats)/len(mats):.1f} 最大={max(mats):.1f}")
    draws = [r for r in rows if r["recorded_outcome"] == 0]
    if draws:
        dmats = [abs(r["material"]) for r in draws]
        print(f"和棋局 (n={len(draws)}) 终局 |子力差|: "
              f"平均={sum(dmats)/len(dmats):.1f} 最大={max(dmats):.1f}")

    if args.show > 0:
        print("-" * 60)
        for i, r in enumerate(rows[:args.show]):
            print(f"  game {i+1}: outcome={r['recorded_outcome']:+d} "
                  f"reason={r['reason']:<14} plies={r['plies']:3d} "
                  f"material={r['material']:+.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
