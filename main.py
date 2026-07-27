"""main.py — Unified entry point for the Mini Xiangqi project.

Subcommands:

    python main.py play        # open the pygame GUI (human vs AI by default)
    python main.py search      # let alpha-beta pick a move from the start
    python main.py mcts        # let MCTS pick a move from the start
    python main.py selfplay    # generate self-play games (MCTS + network)
    python main.py expert      # generate expert (AB vs AB) games for warm-start
    python main.py train       # run the full AlphaZero training loop
    python main.py api         # start the FastAPI server
    python main.py viz         # render a board position to PNG
    python main.py compare     # play two checkpoints head-to-head
    python main.py test        # run the unittest suite

The CLI has no required third-party dependencies for ``search`` / ``test``;
``play`` needs pygame, ``train`` needs torch, ``api`` needs fastapi.
"""

from __future__ import annotations

import argparse
import sys
import unittest


def cmd_play(args) -> int:
    from gui.pygame_gui import run, GameMode, Difficulty
    from engine.piece import Color

    mode = GameMode.HUMAN_VS_HUMAN if args.mode == "hvh" else GameMode.HUMAN_VS_AI
    difficulty_map = {
        "easy": Difficulty.EASY,
        "medium": Difficulty.MEDIUM,
        "hard": Difficulty.HARD,
        "expert": Difficulty.EXPERT,
    }
    difficulty = difficulty_map.get(args.difficulty, Difficulty.MEDIUM)

    # If --color is explicitly given, skip the selection screen.
    if args.color is not None:
        human_color = Color.RED if args.color == "red" else Color.BLACK
        run(mode=mode, difficulty=difficulty, human_color=human_color, choose_color=False)
    else:
        # Let the GUI show the color selection screen (HvAI mode).
        run(mode=mode, difficulty=difficulty, choose_color=True)
    return 0


def cmd_search(args) -> int:
    from engine.board import Board
    from search.alphabeta import alphabeta

    board = Board()
    if args.fen:
        board = Board.from_fen(args.fen)
    print(board)
    score, move = alphabeta(board, depth=args.depth)
    print(f"\nalpha-beta (depth {args.depth}) -> {move}  score={score:+.2f}")
    return 0


def cmd_mcts(args) -> int:
    from engine.board import Board
    from search.mcts import MCTS
    from nn.network import default_net

    board = Board()
    if args.fen:
        board = Board.from_fen(args.fen)
    print(board)

    net = default_net()
    mcts = MCTS(num_simulations=args.simulations)
    probs, best = mcts.search(board, net)

    print(f"\nMCTS ({args.simulations} sims) -> {best}")
    print("Top moves by visit count:")
    sorted_moves = sorted(probs.items(), key=lambda x: -x[1])[:5]
    for mv, prob in sorted_moves:
        print(f"  {mv.uci():6s}  {prob:.3f}")
    return 0


def cmd_selfplay(args) -> int:
    from nn.network import default_net, RandomPolicyValueNet
    from nn.dataset import SelfPlayDataset
    from selfplay.player import generate_games
    from search.mcts import MCTS

    net = default_net() if args.use_net else RandomPolicyValueNet(seed=args.seed)
    mcts = MCTS(num_simulations=args.simulations) if args.simulations > 0 else None

    dataset = SelfPlayDataset(args.data)
    outcomes = generate_games(
        dataset,
        n_games=args.games,
        net=net,
        mcts=mcts,
        seed=args.seed,
        max_plies=args.max_plies,
        temperature=args.temperature,
    )
    red = sum(1 for o in outcomes if o > 0)
    black = sum(1 for o in outcomes if o < 0)
    draws = len(outcomes) - red - black
    print(f"played {len(outcomes)} games: red={red} black={black} draw={draws}")
    print(f"stored {dataset.num_games()} games in {args.data}")
    return 0


def cmd_expert(args) -> int:
    from selfplay.expert import generate_expert_games

    outcomes = generate_expert_games(
        n_games=args.games,
        out_path=args.out,
        depth_high=args.depth_high,
        depth_low=args.depth_low,
        max_plies=args.max_plies,
        seed=args.seed,
        augment=not args.no_augment,
        random_opening_plies=args.opening_plies,
    )
    decisive = sum(1 for o in outcomes if o != 0)
    print(f"generated {len(outcomes)} expert games ({decisive} decisive) -> {args.out}")
    print(f"use them with: python main.py train --warm-start {args.out}")
    return 0


def cmd_train(args) -> int:
    from train.trainer import Trainer
    from utils.config import get_config

    cfg = get_config()
    # Allow overriding MCTS simulations for faster CPU training.
    if args.simulations is not None:
        cfg.num_simulations = args.simulations
    # Allow overriding the root Dirichlet noise concentration (A/B testing).
    if args.dirichlet_alpha is not None:
        cfg.dirichlet_alpha = args.dirichlet_alpha

    trainer = Trainer(
        accept_threshold=args.accept_threshold,
        replay_buffer_size=args.buffer_size,
    )
    trainer.train(
        iterations=args.iterations,
        games_per_iter=args.games,
        epochs_per_iter=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.workers,
        evaluate_games=args.eval_games,
        eval_every=args.eval_every,
        warm_start=args.warm_start,
        warm_start_epochs=args.warm_start_epochs,
        fresh_buffer=args.fresh_buffer,
        resume=not args.no_resume,
    )
    return 0


def cmd_api(args) -> int:
    from api.server import run as run_server
    run_server(host=args.host, port=args.port)
    return 0


def cmd_viz(args) -> int:
    from engine.board import Board
    from engine.move import Move
    from gui.matplotlib_viz import render

    board = Board()
    last_move = None
    if args.fen:
        board = Board.from_fen(args.fen)
    elif args.moves:
        for uci in args.moves:
            mv = Move.from_uci(uci)
            last_move = mv
            board.make_move(mv)

    title = args.title or (
        f"FEN: {args.fen}" if args.fen else
        (f"after moves: {' '.join(args.moves)}" if args.moves else "start position")
    )
    render(board, last_move=last_move, title=title, save_path=args.out)
    print(f"rendered -> {args.out}")
    return 0


def cmd_compare(args) -> int:
    """Compare two checkpoints by playing them against each other."""
    import torch
    from nn.network import create_network
    from search.mcts import MCTS
    from engine.board import Board
    from engine.rules import game_result, GameOutcome
    from utils.config import get_config

    cfg = get_config()

    def load_net(path):
        net = create_network(hidden=cfg.hidden_channels, num_res_blocks=cfg.num_res_blocks)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in payload:
            net.load_state_dict(payload["state_dict"])
        else:
            net.load_state_dict(payload)
        net.eval()
        return net

    net_a = load_net(args.model_a)
    net_b = load_net(args.model_b)
    mcts_a = MCTS(num_simulations=args.simulations, add_noise=False)
    mcts_b = MCTS(num_simulations=args.simulations, add_noise=False)

    print(f"Comparing: {args.model_a} (Red) vs {args.model_b} (Black)")
    print(f"Games: {args.games}, Simulations: {args.simulations}")
    print("-" * 50)

    wins_a, wins_b, draws = 0, 0, 0
    for i in range(args.games):
        board = Board()
        ply = 0
        while ply < 150:
            result = game_result(board)
            if result is not None:
                break
            if board.side_to_move.value == "red":
                _, move = mcts_a.search(board, net_a)
            else:
                _, move = mcts_b.search(board, net_b)
            board.make_move(move)
            ply += 1

        result = game_result(board)
        if result is None:
            draws += 1
            outcome = "Draw"
        elif result.outcome == GameOutcome.RED_WINS:
            wins_a += 1
            outcome = "A wins"
        else:
            wins_b += 1
            outcome = "B wins"
        print(f"  Game {i+1}/{args.games}: {outcome} ({ply} plies)")

        # Swap colors for fairness (alternate who plays Red).
        if (i + 1) % 2 == 0:
            net_a, net_b = net_b, net_a
            mcts_a, mcts_b = mcts_b, mcts_a

    print("-" * 50)
    print(f"Result: A={wins_a} wins, B={wins_b} wins, Draw={draws}")
    rate_a = (wins_a + 0.5 * draws) / args.games
    print(f"A score rate: {rate_a:.1%}")
    return 0


def cmd_test(_args) -> int:
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mini Xiangqi (迷你象棋) — 7x7 Chinese Chess AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- play ---
    p_play = sub.add_parser("play", help="open the pygame GUI")
    p_play.add_argument("--mode", choices=["hvh", "hvai"], default="hvai",
                        help="hvh = human vs human, hvai = human vs AI")
    p_play.add_argument("--difficulty", choices=["easy", "medium", "hard", "expert"],
                        default="medium")
    p_play.add_argument("--color", choices=["red", "black"], default=None,
                        help="which side the human plays (vs AI mode); omit to choose in GUI")

    # --- search ---
    p_search = sub.add_parser("search", help="alpha-beta move from a position")
    p_search.add_argument("--depth", type=int, default=3)
    p_search.add_argument("--fen", default=None, help="FEN (default: start)")

    # --- mcts ---
    p_mcts = sub.add_parser("mcts", help="MCTS move from a position")
    p_mcts.add_argument("--simulations", type=int, default=400)
    p_mcts.add_argument("--fen", default=None, help="FEN (default: start)")

    # --- selfplay ---
    p_sp = sub.add_parser("selfplay", help="generate self-play games")
    p_sp.add_argument("--games", type=int, default=5)
    p_sp.add_argument("--max-plies", type=int, default=200)
    p_sp.add_argument("--data", default="data/games/games.jsonl")
    p_sp.add_argument("--seed", type=int, default=0)
    p_sp.add_argument("--simulations", type=int, default=0,
                      help="MCTS simulations per move (0 = pure network policy)")
    p_sp.add_argument("--temperature", type=float, default=1.0)
    p_sp.add_argument("--use-net", action="store_true",
                      help="use the trained network (if available)")

    # --- expert ---
    p_exp = sub.add_parser("expert",
                           help="generate expert (AB vs AB) games for warm-start")
    p_exp.add_argument("--games", type=int, default=100,
                       help="number of expert games to generate (default: 100)")
    p_exp.add_argument("--depth-high", type=int, default=3,
                       help="search depth of the stronger/teacher side (default: 3)")
    p_exp.add_argument("--depth-low", type=int, default=2,
                       help="search depth of the weaker side (default: 2)")
    p_exp.add_argument("--max-plies", type=int, default=200)
    p_exp.add_argument("--out", default="data/expert/expert.jsonl",
                       help="output expert JSONL path")
    p_exp.add_argument("--seed", type=int, default=0)
    p_exp.add_argument("--no-augment", action="store_true",
                       help="disable horizontal-flip augmentation")
    p_exp.add_argument("--opening-plies", type=int, default=8,
                       help="random opening half-moves before alpha-beta plays, "
                            "to make each game unique (default: 8)")

    # --- train ---
    p_train = sub.add_parser("train", help="run the AlphaZero training loop")
    p_train.add_argument("--iterations", type=int, default=10)
    p_train.add_argument("--games", type=int, default=20,
                         help="self-play games per iteration")
    p_train.add_argument("--epochs", type=int, default=5,
                         help="training epochs per iteration")
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--simulations", type=int, default=None,
                         help="MCTS simulations per move (default: 400; use 50-100 for quick tests)")
    p_train.add_argument("--dirichlet-alpha", type=float, default=None,
                         help="root Dirichlet noise concentration for exploration "
                              "(default: config value 0.15; try 0.1-0.3 to break "
                              "repetitive play)")
    p_train.add_argument("--workers", type=int, default=None,
                         help="parallel self-play workers (default: CPU count)")
    p_train.add_argument("--eval-games", type=int, default=10,
                         help="arena evaluation games per iteration (default: 10)")
    p_train.add_argument("--eval-every", type=int, default=1,
                         help="run arena evaluation once every N iterations "
                              "(default: 1 = every iteration; use 3-5 to speed up)")
    p_train.add_argument("--accept-threshold", type=float, default=0.55,
                         help="arena score rate needed to promote a new best model "
                              "(default: 0.55; lower to 0.5 to let drawn matches "
                              "promote too)")
    p_train.add_argument("--warm-start", default=None,
                         help="path to an expert JSONL file (from `main.py expert`) "
                              "to pretrain on before self-play")
    p_train.add_argument("--warm-start-epochs", type=int, default=2,
                         help="pretraining passes over the expert data (default: 2)")
    p_train.add_argument("--buffer-size", type=int, default=100_000,
                         help="replay buffer capacity (default: 100000; lower to "
                              "~10000-20000 to drop stale draw-heavy data faster)")
    p_train.add_argument("--no-resume", action="store_true",
                         help="start from scratch instead of resuming from the "
                              "latest checkpoint")
    p_train.add_argument("--fresh-buffer", action="store_true",
                         help="resume the network but start the replay buffer "
                              "empty (skip re-reading the on-disk dataset); ideal "
                              "for a warm-start run")

    # --- api ---
    p_api = sub.add_parser("api", help="start the FastAPI server")
    p_api.add_argument("--host", default="127.0.0.1")
    p_api.add_argument("--port", type=int, default=8000)

    # --- viz ---
    p_viz = sub.add_parser("viz", help="render a board position to PNG")
    src = p_viz.add_mutually_exclusive_group()
    src.add_argument("--fen", help="FEN string to render")
    src.add_argument("--moves", nargs="+", help="UCI moves from start")
    p_viz.add_argument("--out", default="board.png", help="output PNG path")
    p_viz.add_argument("--title", default=None)

    # --- compare ---
    p_cmp = sub.add_parser("compare", help="compare two checkpoints head-to-head")
    p_cmp.add_argument("model_a", help="path to first checkpoint (.pt)")
    p_cmp.add_argument("model_b", help="path to second checkpoint (.pt)")
    p_cmp.add_argument("--games", type=int, default=10,
                       help="number of games to play (default: 10)")
    p_cmp.add_argument("--simulations", type=int, default=100,
                       help="MCTS simulations per move (default: 100)")

    # --- test ---
    sub.add_parser("test", help="run the unittest suite")

    args = parser.parse_args()
    handlers = {
        "play": cmd_play,
        "search": cmd_search,
        "mcts": cmd_mcts,
        "selfplay": cmd_selfplay,
        "expert": cmd_expert,
        "train": cmd_train,
        "api": cmd_api,
        "viz": cmd_viz,
        "compare": cmd_compare,
        "test": cmd_test,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
