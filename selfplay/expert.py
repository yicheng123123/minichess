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
from engine.move_generator import legal_moves, in_check
from engine.piece import Color
from engine.rules import game_result, GameOutcome
from search.alphabeta import alphabeta
from search.evaluation import evaluate
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
    opponent: str = "ab",
    epsilon: float = 0.0,
) -> dict:
    """Play one AB(depth_high) vs <opponent> game and collect raw samples.

    Args:
        depth_high: Search depth of the stronger ("teacher") side.
        depth_low: Search depth of the weaker side (used when ``opponent`` is
            ``"ab"``, or as the non-random branch of ``"egreedy"``).
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
        opponent: How the weaker (non-teacher) side plays. ``"ab"`` (default)
            uses alpha-beta at ``depth_low``; ``"random"`` plays uniformly random
            legal moves (yields many short, clean mates — a "finishing" course);
            ``"egreedy"`` plays a random move with probability ``epsilon`` and
            otherwise alpha-beta(depth_low) — a defender that also blunders,
            whose distribution is closest to future self-play.
        epsilon: Random-move probability for the ``"egreedy"`` opponent.

    Returns:
        ``{"samples": [...], "outcome": int, "plies": int, "reason": str}``
        where each sample is a dict with planes/policy/value/move/teacher (see
        module docstring) and ``reason`` is the terminal reason
        (``"checkmate"``/``"no_legal_moves"``/``"repetition"``/``"max_plies"``).
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
        if mover is high_color:
            # Teacher side: always full-strength alpha-beta.
            _score, move = alphabeta(board, depth=depth_high)
        elif opponent == "random":
            move = random.choice(legal_moves(board))
        elif opponent == "egreedy" and random.random() < epsilon:
            move = random.choice(legal_moves(board))
        else:
            # "ab" opponent, or the (1-epsilon) branch of "egreedy".
            _score, move = alphabeta(board, depth=depth_low)
        if move is None:
            break
        history.append((board.to_planes(), move, mover))
        board.make_move(move)
        ply += 1

    # Score the game.
    result = game_result(board)
    if result is None:
        reason = "max_plies"
    else:
        reason = result.reason
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

    return {"samples": samples, "outcome": outcome, "plies": ply, "reason": reason}


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
    opponent: str = "ab",
    epsilon: float = 0.2,
    epsilon_jitter: bool = False,
) -> List[int]:
    """Play ``n_games`` expert games and append them to ``out_path`` (JSONL).

    The stronger side alternates color each game so the teacher covers both
    Red and Black. With ``augment`` each game also contributes a mirrored
    copy, roughly doubling the data.

    Args:
        opponent: Weaker-side policy forwarded to :func:`play_expert_game`
            (``"ab"``/``"random"``/``"egreedy"``).
        epsilon: Base random-move probability for the ``"egreedy"`` opponent.
        epsilon_jitter: If True (and ``opponent == "egreedy"``), draw each game's
            epsilon uniformly from ``{0.1, 0.2, 0.3}`` instead of using the fixed
            ``epsilon``. A spread of opponent strengths keeps the teacher data
            from being too narrow (future self-play opponents also vary).

    Returns:
        List of game outcomes (+1 Red wins, -1 Black wins, 0 draw).
    """
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    _JITTER = random.Random(seed)
    outcomes: List[int] = []
    reason_counts: dict = {}
    decisive = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(n_games):
            high_plays_red = (i % 2 == 0)
            eps = epsilon
            if epsilon_jitter and opponent == "egreedy":
                eps = _JITTER.choice((0.1, 0.2, 0.3))
            game = play_expert_game(
                depth_high=depth_high,
                depth_low=depth_low,
                high_plays_red=high_plays_red,
                max_plies=max_plies,
                seed=seed + i,
                random_opening_plies=random_opening_plies,
                opponent=opponent,
                epsilon=eps,
            )
            samples = game["samples"]
            if augment:
                samples = samples + [_mirror_expert_sample(s) for s in samples]
            record = {"samples": samples, "outcome": game["outcome"],
                      "plies": game["plies"], "reason": game["reason"]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            outcomes.append(game["outcome"])
            reason_counts[game["reason"]] = reason_counts.get(game["reason"], 0) + 1
            if game["outcome"] != 0:
                decisive += 1
            logger.info(f"  expert game {i+1}/{n_games}: "
                        f"{'red' if game['outcome'] > 0 else 'black' if game['outcome'] < 0 else 'draw'} "
                        f"({game['plies']} plies, {len(samples)} samples, {game['reason']})")

    red = outcomes.count(1)
    black = outcomes.count(-1)
    draws = outcomes.count(0)
    reason_summary = ", ".join(f"{k}={v}" for k, v in
                               sorted(reason_counts.items(), key=lambda kv: -kv[1]))
    logger.info(f"Generated {n_games} expert games: "
                f"Red={red} Black={black} Draw={draws} "
                f"({decisive} decisive) terminal[{reason_summary}] -> {out_path}")
    return outcomes


# --------------------------------------------------------------------------- #
# Advantage Curriculum (Tier 2): teach the policy to CONVERT an advantage.
# --------------------------------------------------------------------------- #
def _eval_for(board: Board, color: Color) -> float:
    """Static eval from ``color``'s perspective (evaluate() is side-to-move)."""
    e = evaluate(board)
    return e if board.side_to_move is color else -e


def _is_forcing(board: Board, move) -> bool:
    """Whether ``move`` is a forcing move: a capture or a check."""
    if board.piece_at(move.to_row, move.to_col) is not None:
        return True
    b2 = board.clone()
    b2.make_move(move)
    return in_check(b2, board.side_to_move.opponent)


def play_advantage_game(
    depth_high: int = 3,
    depth_low: int = 2,
    high_plays_red: bool = True,
    max_plies: int = 200,
    seed: Optional[int] = None,
    random_opening_plies: int = 8,
    opponent: str = "egreedy",
    epsilon: float = 0.2,
    advantage_threshold: float = 2.0,
) -> dict:
    """Play one AB game and extract "advantage conversion" training samples.

    The full trajectory is recorded with a running static eval (Red-positive)
    and a per-move ``forcing`` flag (capture or check). Once the game is known
    to be decisive with winner ``W``, every position where ``W``'s advantage
    (eval from ``W``'s perspective) is at least ``advantage_threshold`` is an
    *advantage-phase* position. For ``W``'s moves in those positions we emit a
    sample whose ``teacher`` flag is set only on **forcing** moves, so the
    warm-start policy loss imitates checks/captures (making progress) and never
    the quiet shuffling moves. Value targets are the game outcome (+/-1) from
    the mover's perspective; because only decisive games contribute, there are
    no draw-0 labels to collapse the value head.

    Returns ``{"samples", "outcome", "plies", "reason", "n_teacher"}``; an
    drawn/unfinished game yields an empty ``samples`` list.
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
        mvs = legal_moves(board)
        if not mvs:
            break
        board.make_move(random.choice(mvs))
        opening += 1

    # Record (planes, move, mover, eval_red, forcing) for the whole game.
    history: List[tuple] = []
    ply = opening
    while ply < max_plies:
        if game_result(board) is not None:
            break
        mover = board.side_to_move
        eval_red = _eval_for(board, Color.RED)
        if mover is high_color:
            _score, move = alphabeta(board, depth=depth_high)
        elif opponent == "random":
            move = random.choice(legal_moves(board))
        elif opponent == "egreedy" and random.random() < epsilon:
            move = random.choice(legal_moves(board))
        else:
            _score, move = alphabeta(board, depth=depth_low)
        if move is None:
            break
        forcing = _is_forcing(board, move)
        history.append((board.to_planes(), move, mover, eval_red, forcing))
        board.make_move(move)
        ply += 1

    result = game_result(board)
    reason = "max_plies" if result is None else result.reason
    if result is None or result.outcome is GameOutcome.DRAW:
        return {"samples": [], "outcome": 0, "plies": ply, "reason": reason,
                "n_teacher": 0}
    if result.outcome is GameOutcome.RED_WINS:
        outcome, winner = 1, Color.RED
    else:
        outcome, winner = -1, Color.BLACK

    samples = []
    n_teacher = 0
    for planes, move, mover, eval_red, forcing in history:
        adv = eval_red if winner is Color.RED else -eval_red
        if adv < advantage_threshold:
            continue                      # not yet in an advantage phase
        if mover is not winner:
            continue                      # only teach the converter's choices
        value = float(outcome) if mover is Color.RED else float(-outcome)
        teacher = forcing                 # policy learns only forcing moves
        if teacher:
            n_teacher += 1
        samples.append({
            "planes": encode_planes(planes),
            "policy": _one_hot_policy(move).tolist(),
            "value": value,
            "move": move.uci(),
            "teacher": teacher,
        })
    return {"samples": samples, "outcome": outcome, "plies": ply,
            "reason": reason, "n_teacher": n_teacher}


def generate_advantage_games(
    n_games: int,
    out_path: str,
    depth_high: int = 3,
    depth_low: int = 2,
    max_plies: int = 200,
    seed: int = 0,
    augment: bool = True,
    random_opening_plies: int = 8,
    opponent: str = "egreedy",
    epsilon: float = 0.2,
    epsilon_jitter: bool = True,
    advantage_threshold: float = 2.0,
) -> List[int]:
    """Generate Advantage-Curriculum games (Tier 2) and append to ``out_path``.

    Each decisive game contributes the winner's advantage-phase positions; the
    policy is taught only the forcing moves among them. See
    :func:`play_advantage_game` for the labeling rule.
    """
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    _JITTER = random.Random(seed)
    outcomes: List[int] = []
    reason_counts: dict = {}
    total_samples = 0
    total_teacher = 0
    decisive = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(n_games):
            high_plays_red = (i % 2 == 0)
            eps = epsilon
            if epsilon_jitter and opponent == "egreedy":
                eps = _JITTER.choice((0.1, 0.2, 0.3))
            game = play_advantage_game(
                depth_high=depth_high, depth_low=depth_low,
                high_plays_red=high_plays_red, max_plies=max_plies,
                seed=seed + i, random_opening_plies=random_opening_plies,
                opponent=opponent, epsilon=eps,
                advantage_threshold=advantage_threshold,
            )
            samples = game["samples"]
            outcomes.append(game["outcome"])
            reason_counts[game["reason"]] = reason_counts.get(game["reason"], 0) + 1
            if not samples:
                res = {1: "red", -1: "black"}.get(game["outcome"], "draw")
                logger.info(f"  advantage game {i+1}/{n_games}: {res} "
                            f"({game['plies']} plies, {game['reason']}) — no advantage "
                            f"samples (decided before an advantage phase), skipped")
                continue
            if augment:
                samples = samples + [_mirror_expert_sample(s) for s in samples]
            record = {"samples": samples, "outcome": game["outcome"],
                      "plies": game["plies"], "reason": game["reason"]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            decisive += 1
            total_samples += len(samples)
            total_teacher += game["n_teacher"] * (2 if augment else 1)
            logger.info(f"  advantage game {i+1}/{n_games}: "
                        f"{'red' if game['outcome'] > 0 else 'black'} "
                        f"({game['plies']} plies, {len(samples)} samples, "
                        f"{game['n_teacher']*(2 if augment else 1)} forcing teacher, "
                        f"{game['reason']})")

    red = outcomes.count(1)
    black = outcomes.count(-1)
    draws = outcomes.count(0)
    reason_summary = ", ".join(f"{k}={v}" for k, v in
                               sorted(reason_counts.items(), key=lambda kv: -kv[1]))
    logger.info(f"Generated {n_games} advantage games: "
                f"Red={red} Black={black} Draw={draws} "
                f"({decisive} decisive w/ samples) terminal[{reason_summary}] "
                f"samples={total_samples} forcing_teacher={total_teacher} "
                f"-> {out_path}")
    return outcomes
