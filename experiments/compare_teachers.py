"""compare_teachers.py — 对比三种专家教师的终局分布，为课程学习定结构。

顾问建议：不要直接拿 AB vs Random 当方案，而是实证对比三种教师：
  - ab      : AB(depth_high) vs AB(depth_low)        —— 高质量决策，可能偏长偏和
  - random  : AB(depth_high) vs Random               —— 大量短促将杀（终结课程候选）
  - egreedy : AB(depth_high) vs ε-greedy AB(depth_low)—— 会防守但偶尔犯错，最接近自弈
对每种教师统计：平均步数、将杀率、重复率、终局方式分布、终局子力差。
据此决定课程顺序（如 Warmup -> AB-Random 终结 -> AB-AB 中盘 -> Selfplay）。

纯引擎 + alpha-beta，不依赖 torch；多进程并行加速。

用法（minichess 根目录）：
    python experiments/compare_teachers.py --games 20 --depth-high 3 --depth-low 2 --epsilon 0.2
"""

from __future__ import annotations

import argparse
import os
import random as _random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PIECE_VALUE = {"K": 0, "R": 9, "N": 4, "C": 4.5, "P": 1}


def _play_one(args: dict) -> dict:
    """Worker：下一局 AB(high) vs opponent，返回终局信息。子进程运行。"""
    from engine.board import Board
    from engine.move_generator import legal_moves
    from engine.piece import Color
    from engine.rules import game_result, GameOutcome
    from search.alphabeta import alphabeta

    depth_high = args["depth_high"]
    depth_low = args["depth_low"]
    opponent = args["opponent"]          # "ab" | "random" | "egreedy"
    epsilon = args["epsilon"]
    high_plays_red = args["high_plays_red"]
    max_plies = args["max_plies"]
    seed = args["seed"]

    _random.seed(seed)
    board = Board()
    high_color = Color.RED if high_plays_red else Color.BLACK

    ply = 0
    while ply < max_plies:
        if game_result(board) is not None:
            break
        mover = board.side_to_move
        mvs = legal_moves(board)
        if not mvs:
            break
        if mover is high_color:
            _score, move = alphabeta(board, depth=depth_high)
            if move is None:
                move = _random.choice(mvs)
        else:
            # 对手着法
            if opponent == "random":
                move = _random.choice(mvs)
            elif opponent == "egreedy" and _random.random() < epsilon:
                move = _random.choice(mvs)
            else:  # "ab"，或 egreedy 的 (1-epsilon) 分支
                _s2, m2 = alphabeta(board, depth=depth_low)
                move = m2 if m2 is not None else _random.choice(mvs)
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

    mat = 0.0
    for _sq, piece in board.pieces():
        v = _PIECE_VALUE.get(piece.ptype.value, 0)
        mat += v if piece.color is Color.RED else -v

    ab_won = ((outcome == 1) == high_plays_red) if outcome != 0 else None
    return {"reason": reason, "outcome": outcome, "plies": ply,
            "material": mat, "ab_won": ab_won}


def run_config(opponent: str, n_games: int, depth_high: int, depth_low: int,
               epsilon: float, max_plies: int, seed: int, workers: int) -> list:
    tasks = [{
        "depth_high": depth_high, "depth_low": depth_low, "opponent": opponent,
        "epsilon": epsilon, "high_plays_red": (i % 2 == 0),
        "max_plies": max_plies, "seed": seed + i,
    } for i in range(n_games)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_play_one, t) for t in tasks]
        for fut in as_completed(futs):
            rows.append(fut.result())
    return rows


def summarize(name: str, rows: list) -> None:
    n = len(rows)
    counts: dict = {}
    for r in rows:
        counts[r["reason"]] = counts.get(r["reason"], 0) + 1
    decisive = [r for r in rows if r["outcome"] != 0]
    plies = [r["plies"] for r in decisive] or [0]
    ab_wins = sum(1 for r in rows if r["ab_won"] is True)
    cm = counts.get("checkmate", 0)
    rep = counts.get("repetition", 0)
    print(f"\n=== 教师: {name}  (n={n}) ===", flush=True)
    print(f"  终局方式: " + ", ".join(
        f"{k}={v}({100*v/n:.0f}%)" for k, v in
        sorted(counts.items(), key=lambda kv: -kv[1])), flush=True)
    print(f"  将杀率={100*cm/n:.0f}%  重复率={100*rep/n:.0f}%  "
          f"AB胜率={100*ab_wins/n:.0f}%", flush=True)
    print(f"  胜负局平均步数={sum(plies)/len(plies):.0f}  "
          f"最短={min(plies)} 最长={max(plies)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--depth-high", type=int, default=3)
    ap.add_argument("--depth-low", type=int, default=2)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    workers = args.workers or min(os.cpu_count() or 4, args.games)
    print(f"[info] depth_high={args.depth_high} depth_low={args.depth_low} "
          f"games/config={args.games} workers={workers} epsilon={args.epsilon}",
          flush=True)

    # 先跑快的（random / egreedy），最后跑可能很慢的 ab，避免超时丢结果。
    configs = [
        (f"AB{args.depth_high} vs Random", "random"),
        (f"AB{args.depth_high} vs ε-greedy AB{args.depth_low}(ε={args.epsilon})",
         "egreedy"),
        (f"AB{args.depth_high} vs AB{args.depth_low}", "ab"),
    ]
    for name, opp in configs:
        rows = run_config(opp, args.games, args.depth_high, args.depth_low,
                          args.epsilon, args.max_plies, args.seed, workers)
        summarize(name, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
