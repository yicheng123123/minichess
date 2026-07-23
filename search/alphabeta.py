"""search/alphabeta.py — Alpha-beta negamax search for Mini Xiangqi.

The pruned counterpart to :mod:`search.minimax`. Same negamax convention
(score relative to the side to move) and same terminal handling, but with an
``[alpha, beta]`` window that prunes branches that cannot affect the result.

Basic version: fixed depth, no move ordering, no transposition table, no
quiescence. These are the obvious next optimizations; the structure here is
intentionally minimal so they can be layered on without rewriting the search.

Usage::

    score, move = alphabeta(board, depth=4)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from engine.board import Board, Move
from engine.move_generator import legal_moves
from engine.rules import GameOutcome, game_result, DEFAULT_REPETITION_THRESHOLD
from search.evaluation import evaluate, KING_MISSING_BONUS
from search.minimax import _terminal_value  # reuse the terminal-value helper

MATE_SCORE = KING_MISSING_BONUS * 0.5


def _alphabeta(board: Board, depth: int, alpha: float, beta: float) -> float:
    """Internal recursive negamax with alpha-beta pruning (no move returned)."""
    terminal = _terminal_value(board)
    if terminal is not None:
        return terminal

    if depth <= 0:
        return evaluate(board)

    moves = legal_moves(board)
    if not moves:
        return -MATE_SCORE  # side to move is stuck -> loses

    value = -math.inf
    for move in moves:
        board.make_move(move)
        child = -_alphabeta(board, depth - 1, -beta, -alpha)
        board.undo_move()
        if child > value:
            value = child
        if value > alpha:
            alpha = value
        if alpha >= beta:
            break  # beta cutoff
    return value


def alphabeta(board: Board, depth: int) -> Tuple[float, Optional[Move]]:
    """Root search: returns ``(score, best_move)`` relative to side to move."""
    terminal = _terminal_value(board)
    if terminal is not None:
        return terminal, None

    moves = legal_moves(board)
    if not moves:
        return -MATE_SCORE, None

    alpha = -math.inf
    beta = math.inf
    best_value = -math.inf
    best_move: Optional[Move] = None

    for move in moves:
        board.make_move(move)
        # Root call: full window search of each child.
        child = -_alphabeta(board, depth - 1, -beta, -alpha)
        board.undo_move()
        if child > best_value:
            best_value = child
            best_move = move
        if best_value > alpha:
            alpha = best_value
        # No beta cutoff at the root: beta is +inf, so alpha < beta always.

    return best_value, best_move


def best_move(board: Board, depth: int = 4) -> Optional[Move]:
    """Convenience: just the best move at a given depth."""
    _score, move = alphabeta(board, depth)
    return move
