"""search/minimax.py — Plain minimax search for Mini Xiangqi.

This is the didactic, un-pruned reference searcher. It's correct but slow; use
:mod:`search.alphabeta` for anything beyond depth 3-4. Both share the same
negamax-style return convention: the score is from the perspective of the side
to move at the node being evaluated.

The interface is deliberately simple so the trainer / GUI can call it the same
way as the alpha-beta and (eventually) MCTS searchers::

    score, move = minimax(board, depth=3)

It uses :func:`engine.rules.game_result` to short-circuit terminal nodes, so
checkmate / repetition are scored correctly without searching deeper.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from engine.board import Board, Move
from engine.move_generator import legal_moves
from engine.rules import GameOutcome, game_result, DEFAULT_REPETITION_THRESHOLD
from engine.piece import Color
from search.evaluation import evaluate, KING_MISSING_BONUS

# Large magnitude for mate scores, kept below the king-capture bonus so that
# a forced mate ranks above merely winning a king in deeper search.
MATE_SCORE = KING_MISSING_BONUS * 0.5


def _terminal_value(board: Board) -> Optional[float]:
    """Return a side-to-move-relative terminal score, or None if not terminal."""
    result = game_result(board, DEFAULT_REPETITION_THRESHOLD)
    if result is None:
        return None
    if result.outcome is GameOutcome.DRAW:
        return 0.0
    # A win for the side NOT to move means the side to move is mated/lost.
    if result.winner is board.side_to_move:
        return MATE_SCORE
    return -MATE_SCORE


def minimax(board: Board, depth: int) -> Tuple[float, Optional[Move]]:
    """Full-width minimax. Returns ``(score, best_move)``.

    ``score`` is relative to the side to move at ``board``. At depth 0 the
    static evaluation is returned with ``move=None``. Terminal positions return
    a mate/draw score. When the node has no moves and isn't otherwise terminal,
    it's treated as a loss for the side to move (consistent with xiangqi).
    """
    terminal = _terminal_value(board)
    if terminal is not None:
        return terminal, None

    if depth <= 0:
        return evaluate(board), None

    moves = legal_moves(board)
    if not moves:
        # No legal move and not caught above -> the side to move loses.
        return -MATE_SCORE, None

    best_value = -math.inf
    best_move: Optional[Move] = None
    for move in moves:
        board.make_move(move)
        # After make_move the side flips; negamax negates the child's value.
        child_value, _ = minimax(board, depth - 1)
        value = -child_value
        board.undo_move()
        if value > best_value:
            best_value = value
            best_move = move
    return best_value, best_move


def best_move(board: Board, depth: int = 3) -> Optional[Move]:
    """Convenience: just the best move at a given depth."""
    _score, move = minimax(board, depth)
    return move
