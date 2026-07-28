"""probe_policy.py — Policy 意图探针：优势局面下 policy 到底想干什么。

顾问建议（比课程更重要）：不看 mate 率，直接看 policy 在优势局面里的着法偏好——
是想 将军 / 吃子 / 收紧对方王，还是 原地来回走。这能直接区分"policy 不会推进"
和"value 问题"。

对每个优势局面：
  * 计算网络 policy 分布（legal moves 上 softmax）；
  * 列出 top 着法及其概率，分类为 check / capture / quiet；
  * 统计概率质量在 check/capture/quiet 上的占比（progress vs 原地踏步）。
并对选定局面画一张"落点概率热力图"+ top 着法箭头（PNG）。

用法（minichess 根目录）：
    python experiments/probe_policy.py --top 8
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (名称, FEN)。残局 + 两个"多子但仍是复杂中盘"的优势局（从起始局面对手少一子）。
POSITIONS = [
    ("KRR vs K (+18)", "4k2/R6/7/7/7/6R/3K3 r 0 1"),
    ("KRC vs K (+13)", "4k2/R6/7/3C3/7/7/3K3 r 0 1"),
    ("KR  vs K (+9)",  "4k2/R6/7/7/7/7/3K3 r 0 1"),
    ("中盘多一车 (+9)", "1cnkncr/p1ppp1p/7/7/7/P1PPP1P/RCNKNCR r 0 1"),
    ("中盘多一马 (+4)", "rc1kncr/p1ppp1p/7/7/7/P1PPP1P/RCNKNCR r 0 1"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--sims", type=int, default=100,
                    help="MCTS 模拟数，用于对比 visit 分布与网络原始 logits")
    ap.add_argument("--heatmap", default="experiments/policy_heatmap.png",
                    help="热力图输出 PNG（对第一个局面）；置空则不画")
    args = ap.parse_args()

    import numpy as np
    import torch
    from engine.board import Board
    from engine.move_generator import legal_moves, in_check
    from engine.piece import Color
    from nn.network import create_network, move_to_index
    from search.mcts import MCTS
    from train.checkpoint import CheckpointManager
    from utils.config import get_config

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = create_network(hidden=cfg.hidden_channels,
                         num_res_blocks=cfg.num_res_blocks).to(device)
    ckpt = CheckpointManager(checkpoint_dir=cfg.checkpoint_dir)
    src = "best.pt" if ckpt.load_best(net) else f"latest iter={ckpt.load_latest(net)}"
    net.eval()
    mcts = MCTS(num_simulations=args.sims, c_puct=cfg.c_puct,
                dirichlet_alpha=cfg.dirichlet_alpha)
    print(f"[info] frozen net from {src} | device={device} | MCTS sims={args.sims}",
          flush=True)

    def policy_dist(board):
        planes = torch.from_numpy(board.to_planes()).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits, value = net(planes)
        logits = logits.squeeze(0).cpu().numpy()
        lm = legal_moves(board)
        idxs = [move_to_index(m) for m in lm]
        raw = np.array([logits[i] for i in idxs], dtype=np.float64)
        raw = raw - raw.max()
        probs = np.exp(raw)
        probs /= probs.sum()
        return lm, probs, float(value.item())

    def categorize(board, move) -> str:
        b2 = board.clone()
        b2.make_move(move)
        if in_check(b2, board.side_to_move.opponent):
            return "check"
        if board.piece_at(move.to_row, move.to_col) is not None:
            return "capture"
        return "quiet"

    def category_mass(board, move_weights) -> dict:
        """move_weights: [(Move, weight>=0)]。返回归一化后的 check/capture/quiet 质量。"""
        mass = {"check": 0.0, "capture": 0.0, "quiet": 0.0}
        total = sum(w for _m, w in move_weights)
        if total <= 0:
            return mass
        for mv, w in move_weights:
            mass[categorize(board, mv)] += w / total
        return mass

    def mcts_dist(board):
        """运行 MCTS，返回 [(Move, 访问量)]（未归一化，category_mass 内部会归一）。"""
        visit_counts, _best = mcts.search(board, net)
        return list(visit_counts.items())

    agg_raw = {"check": 0.0, "capture": 0.0, "quiet": 0.0}
    agg_mcts = {"check": 0.0, "capture": 0.0, "quiet": 0.0}
    first_dist = None
    for name, fen in POSITIONS:
        board = Board.from_fen(fen)
        lm, probs, value = policy_dist(board)
        if first_dist is None:
            first_dist = (board, lm, probs)
        order = np.argsort(-probs)
        print(f"\n=== {name} | 网络 value={value:+.3f} | {len(lm)} 合法着 ===",
              flush=True)
        print("  [网络原始 logits] top:", flush=True)
        for rank, i in enumerate(order[:args.top]):
            mv, p = lm[i], probs[i]
            print(f"    {rank+1}. {mv.uci():<6} p={p:5.1%}  [{categorize(board, mv)}]",
                  flush=True)

        mass_raw = category_mass(board, list(zip(lm, probs)))
        mass_mcts = category_mass(board, mcts_dist(board))
        for k in agg_raw:
            agg_raw[k] += mass_raw[k]
            agg_mcts[k] += mass_mcts[k]
        print(f"  -> 网络logits : 将军={mass_raw['check']:.0%} "
              f"吃子={mass_raw['capture']:.0%} 闲走={mass_raw['quiet']:.0%}", flush=True)
        print(f"  -> MCTS visit : 将军={mass_mcts['check']:.0%} "
              f"吃子={mass_mcts['capture']:.0%} 闲走={mass_mcts['quiet']:.0%}", flush=True)

    n = len(POSITIONS)
    pr = (agg_raw['check'] + agg_raw['capture']) / n
    pm = (agg_mcts['check'] + agg_mcts['capture']) / n
    print("\n" + "=" * 60, flush=True)
    print(f"=== 跨 {n} 个优势局面平均（推进性 = 将军+吃子）===", flush=True)
    print(f"  网络原始 logits: 将军={agg_raw['check']/n:.0%} "
          f"吃子={agg_raw['capture']/n:.0%} 闲走={agg_raw['quiet']/n:.0%} | 推进={pr:.0%}",
          flush=True)
    print(f"  MCTS visit 分布: 将军={agg_mcts['check']/n:.0%} "
          f"吃子={agg_mcts['capture']/n:.0%} 闲走={agg_mcts['quiet']/n:.0%} | 推进={pm:.0%}",
          flush=True)
    print("-" * 60, flush=True)
    if pr < 0.3 and pm < 0.3:
        print("  结论: 网络本身与 MCTS 都偏好闲走 —— 瓶颈在网络原始 policy"
              "（搜索无力纠正，value 失效）。", flush=True)
    elif pr > 0.5 > pm:
        print("  结论: 网络原始 policy 会推进，但 MCTS 把它搜成了闲走 —— "
              "瓶颈在搜索（value 头失效拖垮了 MCTS）。", flush=True)
    elif pr < 0.3 < pm:
        print("  结论: 网络原始偏闲走，但 MCTS 搜索能纠正成推进 —— 搜索在帮忙。",
              flush=True)
    else:
        print("  结论: 网络与 MCTS 都偏推进。", flush=True)

    # 热力图（第一个局面）
    if args.heatmap and first_dist is not None:
        try:
            _render_heatmap(args.heatmap, *first_dist, cfg)
            print(f"\n[info] 热力图已保存 -> {args.heatmap}", flush=True)
        except Exception as e:
            print(f"\n[warn] 热力图渲染失败: {e}", flush=True)
    return 0


def _render_heatmap(save_path, board, lm, probs, cfg):
    """画棋盘 + 落点概率热力 + top3 着法箭头。"""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gui.matplotlib_viz import _draw_grid, _draw_piece, _board_to_xy
    from engine.board import BOARD_SIZE

    # 落点概率质量（每个目标格被多少概率质量指向）
    dest_mass = np.zeros((BOARD_SIZE, BOARD_SIZE))
    for mv, p in zip(lm, probs):
        dest_mass[mv.to_row, mv.to_col] += p

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_facecolor("#2b2b30")
    fig.patch.set_facecolor("#2b2b30")
    _draw_grid(ax)

    # 热力（目标格）
    vmax = dest_mass.max() if dest_mass.max() > 0 else 1.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            m = dest_mass[r, c]
            if m > 1e-4:
                x, y = _board_to_xy(r, c)
                ax.scatter([x], [y], s=1500, c=[[1.0, 0.85, 0.2]],
                           alpha=0.15 + 0.6 * (m / vmax), zorder=1.5)

    # 棋子
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece = board.piece_at(r, c)
            if piece is not None:
                _draw_piece(ax, r, c, piece)

    # top3 着法箭头
    order = np.argsort(-probs)
    for rank, i in enumerate(order[:3]):
        mv, p = lm[i], probs[i]
        x0, y0 = _board_to_xy(mv.from_row, mv.from_col)
        x1, y1 = _board_to_xy(mv.to_row, mv.to_col)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=["#4fc3f7", "#81c784", "#ffb74d"][rank],
                                    lw=1.5 + 4 * p, alpha=0.9),
                    zorder=4)
        ax.text(x0, y0 + 0.18, f"{p:.0%}", color="white", fontsize=8,
                ha="center", zorder=5)

    ax.set_xticks(range(BOARD_SIZE))
    ax.set_xticklabels([chr(ord("a") + c) for c in range(BOARD_SIZE)], fontsize=9)
    ax.set_yticks(range(BOARD_SIZE))
    ax.set_yticklabels([str(r + 1) for r in range(BOARD_SIZE - 1, -1, -1)], fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.7, BOARD_SIZE - 0.3)
    ax.set_ylim(-0.7, BOARD_SIZE - 0.3)
    ax.set_aspect("equal")
    ax.set_title("Policy 落点概率热力 + Top3 着法（黄=概率质量，箭头=首选着法）",
                 fontsize=10, color="white")
    fig.savefig(save_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
