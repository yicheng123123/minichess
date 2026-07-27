"""selfplay/expert.py — Generate expert games for warm-start pretraining.

Plays alpha-beta at two different depths against itself to produce decisive
games carrying real win/loss value signals (the cold-start cure for a value
head that has only ever seen draws).

The output is a *generic* "expert" JSONL format consumed by
:mod:`train.warmstart`. Any process that emits this format (alpha-beta, human
games, an older network) can warm-start training through that one interface —
nothing here is hard-wired to alpha-beta beyond this generator.

On-disk format (one game per line)::

    {"samples": [
        {"planes": <base64>, "policy": [...], "value": float,
         "move": "uci", "teacher": true},
        ...
    ],
    "outcome": 1, "plies": 87}

Per sample:

* ``value``  — game outcome from the mover's perspective (+1 win / -1 loss / 0 draw).
* ``policy`` — one-hot vector on the move actually played.
* ``teacher``— whether this position's mover is the side to imitate (the winner
  of a decisive game, or the stronger side when the game is drawn). The
  warm-start trainer applies the policy loss only on teacher positions and the
  value loss on every position, so the network imitates strong play while still
  learning what winning/losing looks like from all positions.

CLI::

    python main.py expert --games 100 --depth-high 3 --depth-low 2 \
        --out data/expert/expert.jsonl
"""

from __future__ import annotations

import json
import os
import random
from typing import List, Optional

import numpy as np

from engine.board import Board
from engine.move_generator import legal_moves
from engine.piece import Color
from engine.rules import game_result, GameOutcome
from search.alphabeta import alphabeta
from nn.network import move_to_index, NUM_MOVE_ACTIONS
from nn.dataset import encode_planes, decode_planes
from selfplay.player import _mirror_planes, _mirror_policy
from utils.logger import logger


def _one_hot_policy(move) -> np.ndarray:
    """A NUM_MOVE_ACTIONS vector with 1.0 at the move's index, 0 elsewhere."""
    policy = np.zeros(NUM_MOVE_ACTIONS, dtype=np.float32)
    idx = move_to_index(move)
    if 0 <= idx < NUM_MOVE_ACTIONS:
        policy[idx] = 1.0
    return policy


def play_expert_game(
    depth_high: int = 3,
    depth_low: int = 2,
    high_plays_red: bool = True,
    max_plies: int = 200,
    seed: Optional[int] = None,
    random_opening_plies: int = 8,
) -> dict:
    """Play one AB(depth_high) vs AB(depth_low) game and collect raw samples.

    Args:
        depth_high: Search depth of the stronger ("teacher") side.
        depth_low: Search depth of the weaker side.
        high_plays_red: Whether the stronger side plays Red (alternate this
            across games so the teacher covers both colors).
        max_plies: Cap on half-moves before the game is scored a draw.
        seed: Optional random seed (alpha-beta is otherwise deterministic).
        random_opening_plies: Number of half-moves played with uniformly random
            legal moves before alpha-beta takes over. Alpha-beta is fully
            deterministic, so without this every game at the same depths is
            identical; a randomized (but seeded) opening gives each game a
            distinct trajectory. These opening plies are not recorded as
            training data — recording starts once alpha-beta plays.

    Returns:
        ``{"samples": [...], "outcome": int, "plies": int}`` where each sample
        is a dict with planes/policy/value/move/teacher (see module docstring).
    """
    if seed is not None:
        random.seed(seed)

    board = Board()
    high_color = Color.RED if high_plays_red else Color.BLACK

    # Randomized opening (not recorded) to diversify the games.
    opening = 0
    while opening < random_opening_plies:
        if game_result(board) is not None:
            break
        moves = legal_moves(board)
        if not moves:
            break
        board.make_move(random.choice(moves))
        opening += 1

    history: List[tuple] = []  # (planes, move, mover)
    ply = opening
    while ply < max_plies:
        if game_result(board) is not None:
            break
        mover = board.side_to_move
        depth = depth_high if mover is high_color else depth_low
        _score, move = alphabeta(board, depth=depth)
        if move is None:
            break
        history.append((board.to_planes(), move, mover))
        board.make_move(move)
        ply += 1

    # Score the game.
    result = game_result(board)
    if result is None or result.outcome is GameOutcome.DRAW:
        outcome = 0
        winner = None
    elif result.outcome is GameOutcome.RED_WINS:
        outcome = 1
        winner = Color.RED
    else:
        outcome = -1
        winner = Color.BLACK

    # Teacher = winner of a decisive game; otherwise the stronger side.
    teacher_side = winner if winner is not None else high_color

    samples = []
    for planes, move, mover in history:
        if outcome == 0:
            value = 0.0
        else:
            value = float(outcome) if mover is Color.RED else float(-outcome)
        samples.append({
            "planes": encode_planes(planes),
            "policy": _one_hot_policy(move).tolist(),
            "value": value,
            "move": move.uci(),
            "teacher": (mover is teacher_side),
        })

    return {"samples": samples, "outcome": outcome, "plies": ply}


def _mirror_expert_sample(sample: dict) -> dict:
    """Horizontally mirror an expert sample (planes + policy); value/teacher
    are invariant under a left-right board flip."""
    planes = decode_planes(sample["planes"])
    policy = np.asarray(sample["policy"], dtype=np.float32)
    return {
        "planes": encode_planes(_mirror_planes(planes)),
        "policy": _mirror_policy(policy).tolist(),
        "value": sample["value"],
        "move": sample["move"],
        "teacher": sample["teacher"],
    }


def generate_expert_games(
    n_games: int,
    out_path: str,
    depth_high: int = 3,
    depth_low: int = 2,
    max_plies: int = 200,
    seed: int = 0,
    augment: bool = True,
    random_opening_plies: int = 8,
) -> List[int]:
    """Play ``n_games`` expert games and append them to ``out_path`` (JSONL).

    The stronger side alternates color each game so the teacher covers both
    Red and Black. With ``augment`` each game also contributes a mirrored
    copy, roughly doubling the data.

    Returns:
        List of game outcomes (+1 Red wins, -1 Black wins, 0 draw).
    """
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    outcomes: List[int] = []
    decisive = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(n_games):
            high_plays_red = (i % 2 == 0)
            game = play_expert_game(
                depth_high=depth_high,
                depth_low=depth_low,
                high_plays_red=high_plays_red,
                max_plies=max_plies,
                seed=seed + i,
                random_opening_plies=random_opening_plies,
            )
            samples = game["samples"]
            if augment:
                samples = samples + [_mirror_expert_sample(s) for s in samples]
            record = {"samples": samples, "outcome": game["outcome"],
                      "plies": game["plies"]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            outcomes.append(game["outcome"])
            if game["outcome"] != 0:
                decisive += 1
            logger.info(f"  expert game {i+1}/{n_games}: "
                        f"{'red' if game['outcome'] > 0 else 'black' if game['outcome'] < 0 else 'draw'} "
                        f"({game['plies']} plies, {len(samples)} samples)")

    red = outcomes.count(1)
    black = outcomes.count(-1)
    draws = outcomes.count(0)
    logger.info(f"Generated {n_games} expert games: "
                f"Red={red} Black={black} Draw={draws} "
                f"({decisive} decisive) -> {out_path}")
    return outcomes
