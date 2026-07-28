"""probe_frozen.py — 冻结网络探针：在不训练的前提下诊断"为什么不会终结"。

顾问路线：先别动训练标签（Teacher Value 会引入混杂变量），用冻结的 best.pt
做三组单变量实验，确认到底是 搜索深度 / 探索参数 / policy 本身 哪个让优势局
转化不成将杀。

三种模式（均并行、纯推理、不更新权重）：
  --mode endgame : 从必胜残局(KRR/KRC/KR vs K)出发，让 MCTS 执红攻杀，
                   黑方分别用 random 与 mcts，统计能否将杀及所需步数。
                   —— 最决定性：若连 KRR-vs-K 都杀不掉 random，policy 根本没学会终结。
  --mode sims    : 从起始局面自弈，扫描 simulation(25/50/100/200/400)，
                   统计 checkmate/repetition/max_plies 占比 —— 搜索越深是否越能杀。
  --mode params  : 从起始局面自弈，扫描 Dirichlet α(0.03/0.15/0.30/0.50)，
                   统计终结率 —— 是否只是探索不足。

用法（minichess 根目录）：
    python experiments/probe_frozen.py --mode endgame --sims 100 --trials 6
    python experiments/probe_frozen.py --mode sims --games 16
    python experiments/probe_frozen.py --mode params --games 16 --sims 100
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PIECE_VALUE = {"K": 0, "R": 9, "N": 4, "C": 4.5, "P": 1}

# 必胜残局（红方大优，红先）。王错开不同列以避免"将帅照面"。
ENDGAMES = [
    ("KRR vs K", "4k2/R6/7/7/7/6R/3K3 r 0 1"),
    ("KRC vs K", "4k2/R6/7/3C3/7/7/3K3 r 0 1"),
    ("KR  vs K", "4k2/R6/7/7/7/7/3K3 r 0 1"),
]


def _material(board) -> float:
    s = 0.0
    from engine.piece import Color
    for _sq, piece in board.pieces():
        v = _PIECE_VALUE.get(piece.ptype.value, 0)
        s += v if piece.color is Color.RED else -v
    return s


def _probe_game(args: dict) -> dict:
    """Worker：下一局（可自定义起始 FEN、MCTS 参数、黑方策略），返回终局信息。"""
    import random as _random
    import torch
    from engine.board import Board
    from engine.move_generator import legal_moves
    from engine.piece import Color
    from engine.rules import game_result, GameOutcome
    from search.mcts import MCTS
    from nn.network import create_network

    _random.seed(args["seed"])
    device = torch.device(args["device"])
    net = create_network(**args["net_kwargs"]).to(device)
    net.load_state_dict(args["state_dict"])
    net.eval()

    mcts = MCTS(num_simulations=args["sims"], c_puct=args["c_puct"],
                dirichlet_alpha=args["alpha"],
                dirichlet_epsilon=args["dirichlet_eps"], add_noise=args["add_noise"])

    board = Board.from_fen(args["start_fen"]) if args["start_fen"] else Board()
    black_policy = args["black_policy"]   # "mcts" | "random"
    temperature = args["temperature"]
    temp_drop_after = args["temp_drop_after"]
    max_plies = args["max_plies"]

    ply = 0
    first_rep = None
    while ply < max_plies:
        if game_result(board) is not None:
            break
        mover = board.side_to_move
        use_mcts = (black_policy == "mcts") or (mover is Color.RED)
        if use_mcts:
            temp = temperature if ply < temp_drop_after else 0.0
            if temp > 0:
                _vc, mv = mcts.search_with_temperature(board, net, temp)
            else:
                _vc, mv = mcts.search(board, net)
        else:
            mv = _random.choice(legal_moves(board))
        board.make_move(mv)
        ply += 1
        if first_rep is None and board.repetition_count() >= 2:
            first_rep = ply

    result = game_result(board)
    if result is None:
        reason, outcome = "max_plies", 0
    else:
        reason = result.reason
        ov = result.outcome.value
        outcome = 1 if ov == "red_wins" else -1 if ov == "black_wins" else 0
    return {"reason": reason, "outcome": outcome, "plies": ply,
            "material": _material(board), "first_rep": first_rep}


def _summarize(name: str, rows: list) -> None:
    n = len(rows)
    if n == 0:
        print(f"  {name}: (无结果)", flush=True)
        return
    counts: dict = {}
    for r in rows:
        counts[r["reason"]] = counts.get(r["reason"], 0) + 1
    cm = counts.get("checkmate", 0)
    rep = counts.get("repetition", 0)
    mp = counts.get("max_plies", 0)
    nlm = counts.get("no_legal_moves", 0)
    plies = [r["plies"] for r in rows]
    print(f"  {name:<22} n={n:3d} | mate={100*cm/n:3.0f}% rep={100*rep/n:3.0f}% "
          f"maxplies={100*mp/n:3.0f}% nolegal={100*nlm/n:3.0f}% | "
          f"avgplies={sum(plies)/n:.0f}", flush=True)


def _run_pool(tasks: list, workers: int) -> list:
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_probe_game, t) for t in tasks]
        for fut in as_completed(futs):
            rows.append(fut.result())
    return rows


def _base_task(args, state_dict, net_kwargs, device) -> dict:
    return {
        "state_dict": state_dict, "net_kwargs": net_kwargs, "device": device,
        "sims": args.sims, "c_puct": args.c_puct, "alpha": args.alpha,
        "dirichlet_eps": args.dirichlet_eps, "add_noise": not args.no_noise,
        "temperature": args.temperature, "temp_drop_after": args.temp_drop_after,
        "max_plies": args.max_plies, "start_fen": None, "black_policy": "mcts",
        "seed": args.seed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["endgame", "sims", "params"], required=True)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--games", type=int, default=16, help="sims/params 模式每配置局数")
    ap.add_argument("--trials", type=int, default=6, help="endgame 模式每局面尝试次数")
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--temp-drop-after", type=int, default=30)
    ap.add_argument("--c-puct", type=float, default=2.5)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--dirichlet-eps", type=float, default=0.25)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    import torch
    from nn.network import create_network
    from train.checkpoint import CheckpointManager
    from utils.config import get_config
    from engine.board import Board
    from engine.move_generator import legal_moves
    from engine.rules import game_result

    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = create_network(hidden=cfg.hidden_channels,
                         num_res_blocks=cfg.num_res_blocks).to(device)
    ckpt = CheckpointManager(checkpoint_dir=cfg.checkpoint_dir)
    src = "best.pt" if ckpt.load_best(net) else f"latest iter={ckpt.load_latest(net)}"
    net.eval()
    state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
    net_kwargs = {"hidden": cfg.hidden_channels, "num_res_blocks": cfg.num_res_blocks}
    workers = args.workers or min(os.cpu_count() or 4, 8)
    print(f"[info] frozen net from {src} | device={device} workers={workers} "
          f"mode={args.mode}", flush=True)

    if args.mode == "endgame":
        print(f"[endgame] sims={args.sims} trials={args.trials}/局面/对手", flush=True)
        for name, fen in ENDGAMES:
            # 验证局面合法（非终局、走子方有合法着法）
            b = Board.from_fen(fen)
            if game_result(b) is not None or not legal_moves(b):
                print(f"  {name}: 跳过（局面非法或已终局）", flush=True)
                continue
            for opp in ("random", "mcts"):
                tasks = []
                for t in range(args.trials):
                    tk = _base_task(args, state_dict, net_kwargs, device)
                    tk["start_fen"] = fen
                    tk["black_policy"] = opp
                    tk["seed"] = args.seed + t + (1000 if opp == "mcts" else 0)
                    tasks.append(tk)
                rows = _run_pool(tasks, workers)
                _summarize(f"{name} | 黑={opp}", rows)

    elif args.mode == "sims":
        for sims in (25, 50, 100, 200, 400):
            tasks = []
            for i in range(args.games):
                tk = _base_task(args, state_dict, net_kwargs, device)
                tk["sims"] = sims
                tk["seed"] = args.seed + i
                tasks.append(tk)
            rows = _run_pool(tasks, workers)
            _summarize(f"sims={sims}", rows)

    elif args.mode == "params":
        for alpha in (0.03, 0.15, 0.30, 0.50):
            tasks = []
            for i in range(args.games):
                tk = _base_task(args, state_dict, net_kwargs, device)
                tk["alpha"] = alpha
                tk["seed"] = args.seed + i
                tasks.append(tk)
            rows = _run_pool(tasks, workers)
            _summarize(f"alpha={alpha}", rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
