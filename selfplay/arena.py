"""Arena module for comparing two neural network models.

Used by the training loop to gate model updates: a new model must
demonstrate sufficient strength against the current champion before
being accepted (Elo-like evaluation gate).
"""

import random
from typing import Optional

from engine.board import Board
from engine.move import Move
from engine.move_generator import legal_moves
from engine.rules import game_result, GameOutcome
from engine.piece import Color
from search.mcts import MCTS
from nn.network import PolicyValueNet
from utils.logger import logger


def _play_one_game(
    net_red: PolicyValueNet,
    net_black: PolicyValueNet,
    mcts_simulations: int = 100,
    seed: Optional[int] = None,
    max_plies: int = 200,
) -> int:
    """Play a single game between two networks.

    Args:
        net_red: Network playing as Red (moves first).
        net_black: Network playing as Black.
        mcts_simulations: Number of MCTS simulations per move.
        seed: Optional random seed.
        max_plies: Maximum half-moves before draw.

    Returns:
        +1 if Red wins, -1 if Black wins, 0 for draw.
    """
    if seed is not None:
        random.seed(seed)

    board = Board()
    mcts = MCTS(num_simulations=mcts_simulations)

    ply = 0
    while ply < max_plies:
        result = game_result(board)
        if result is not None:
            break

        # Select the appropriate network for the side to move
        if board.side_to_move == Color.RED:
            net = net_red
        else:
            net = net_black

        # Greedy play (temperature=0) for evaluation — no exploration noise
        visit_counts, best_move = mcts.search(board, net)

        board.make_move(best_move)
        ply += 1

    # Determine outcome
    result = game_result(board)
    if result is None:
        return 0
    elif result.outcome == GameOutcome.RED_WINS:
        return 1
    elif result.outcome == GameOutcome.BLACK_WINS:
        return -1
    else:
        return 0


def evaluate_match(
    net_a: PolicyValueNet,
    net_b: PolicyValueNet,
    n_games: int = 20,
    mcts_simulations: int = 100,
    seed: Optional[int] = None,
    max_plies: int = 200,
) -> dict:
    """Evaluate two networks by playing a match of n_games.

    Colors are alternated each game to eliminate first-move bias:
    - Even-indexed games: net_a plays Red, net_b plays Black.
    - Odd-indexed games: net_b plays Red, net_a plays Black.

    Args:
        net_a: First network (typically the challenger).
        net_b: Second network (typically the current champion).
        n_games: Total number of games to play (should be even for
                 perfect color balance, but odd is handled gracefully).
        mcts_simulations: MCTS simulations per move for both players.
        seed: Base random seed for reproducibility.
        max_plies: Maximum half-moves per game before declaring draw.

    Returns:
        Dictionary with keys:
            - "wins_a": Number of games won by net_a.
            - "wins_b": Number of games won by net_b.
            - "draws": Number of drawn games.
    """
    wins_a = 0
    wins_b = 0
    draws = 0

    for i in range(n_games):
        game_seed = (seed + i) if seed is not None else None

        # Alternate colors
        if i % 2 == 0:
            net_red, net_black = net_a, net_b
            a_is_red = True
        else:
            net_red, net_black = net_b, net_a
            a_is_red = False

        outcome = _play_one_game(
            net_red=net_red,
            net_black=net_black,
            mcts_simulations=mcts_simulations,
            seed=game_seed,
            max_plies=max_plies,
        )

        # Map outcome back to net_a / net_b perspective
        if outcome == 0:
            draws += 1
        elif a_is_red:
            # net_a was Red
            if outcome == 1:
                wins_a += 1
            else:
                wins_b += 1
        else:
            # net_a was Black
            if outcome == -1:
                wins_a += 1
            else:
                wins_b += 1

        if (i + 1) % 5 == 0 or (i + 1) == n_games:
            logger.info(
                f"Arena progress: {i + 1}/{n_games} — "
                f"A={wins_a} B={wins_b} D={draws}"
            )

    logger.info(
        f"Match complete ({n_games} games): "
        f"net_a={wins_a}, net_b={wins_b}, draws={draws}"
    )

    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
    }


def should_accept_new_model(
    match_result: dict,
    win_rate_threshold: float = 0.55,
) -> bool:
    """Decide whether to accept a new model based on match results.

    The new model (net_a) is accepted if its score (wins + 0.5*draws)
    divided by total games meets or exceeds the threshold.

    Args:
        match_result: Output of evaluate_match().
        win_rate_threshold: Minimum score rate to accept (default 55%).

    Returns:
        True if the new model should replace the current champion.
    """
    total = match_result["wins_a"] + match_result["wins_b"] + match_result["draws"]
    if total == 0:
        return False

    score = match_result["wins_a"] + 0.5 * match_result["draws"]
    rate = score / total

    logger.info(
        f"Model gate: score_rate={rate:.3f} "
        f"(threshold={win_rate_threshold:.3f}) -> "
        f"{'ACCEPT' if rate >= win_rate_threshold else 'REJECT'}"
    )

    return rate >= win_rate_threshold
