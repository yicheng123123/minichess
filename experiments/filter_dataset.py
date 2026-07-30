"""experiments/filter_dataset.py — 从 DAgger 生成池切出消融实验数据集。

顾问的对照实验设计：只改变 draw 过滤这一项，其它完全一致。
从同一个 pool 文件出发：

  * Exp A: 保留全部和棋（draw_frac = 1.0）
  * Exp B: 只保留一部分和棋（如 draw_frac = 0.2）

两个版本的"分出胜负"局完全相同，唯一区别是和棋数量。这样如果棋力
出现差异，就能归因到 draw 过滤，而不是混合教师或其它变量。

和棋下采样使用固定种子，可复现；优先保留 student 执黑守和的局
（这是 sl_dagger1 最值钱的能力）。

用法：
    python experiments/filter_dataset.py \
        --in data/expert/dagger_pool.jsonl \
        --out data/expert/dagger_expB.jsonl \
        --draw-frac 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random


def classify(record: dict) -> str:
    """从 student 视角分类：'win' / 'loss' / 'draw'。"""
    outcome = record["outcome"]
    if record.get("student_color") == "black":
        outcome = -outcome
    if record["reason"] in ("repetition", "stalemate", "max_plies") or outcome == 0:
        return "draw"
    return "win" if outcome > 0 else "loss"


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter DAgger pool for ablations")
    ap.add_argument("--in", dest="inp", required=True, help="输入 pool 文件")
    ap.add_argument("--out", required=True, help="输出过滤后文件")
    ap.add_argument("--draw-frac", type=float, default=1.0,
                    help="保留的和棋比例（1.0=全保留, 0.2=只留20%%）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    wins, losses, draws = [], [], []
    with open(args.inp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            c = classify(rec)
            if c == "win":
                wins.append(rec)
            elif c == "loss":
                losses.append(rec)
            else:
                draws.append(rec)

    # 和棋下采样：优先保留 student 执黑的守和局
    n_draw_keep = int(round(len(draws) * args.draw_frac))
    if n_draw_keep < len(draws):
        # 执黑和棋排前面（更稳定地保留防守知识），再按种子打乱同色内部
        black_draws = [r for r in draws if r.get("student_color") == "black"]
        red_draws = [r for r in draws if r.get("student_color") != "black"]
        random.shuffle(black_draws)
        random.shuffle(red_draws)
        ordered = black_draws + red_draws
        draws_kept = ordered[:n_draw_keep]
    else:
        draws_kept = draws

    # 输出：胜负局全保留 + 采样后的和棋，按原始顺序无关，打乱混合
    out_records = wins + losses + draws_kept
    random.shuffle(out_records)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_samples = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")
            n_samples += len(rec["samples"])

    print(f"[filter] {args.inp} -> {args.out}")
    print(f"  wins={len(wins)} losses={len(losses)} "
          f"draws={len(draws)}->kept {len(draws_kept)} (frac={args.draw_frac})")
    print(f"  total games={len(out_records)}, samples={n_samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
