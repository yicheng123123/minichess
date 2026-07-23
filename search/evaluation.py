"""search/evaluation.py — Static evaluation for Mini Xiangqi.

A handcrafted evaluation function used by the classical searchers
(:mod:`search.minimax`, :mod:`search.alphabeta`) and as a baseline before the
neural network is trained. It is deliberately simple and fast.

Evaluation is from the perspective of the side to move (positive = good for
the side to move), which is the convention the negamax-style searchers use.

Components:
  * Material balance, using standard xiangqi-relative piece values.
  * A small positional bonus table per piece type (e.g. soldier advances,
    rook on an open file, centralization of horse).
  * A large terminal bonus when the opponent's king is gone / will be gone.

Piece values are tuned relative to a Soldier = 1.0 baseline; they're easy to
re-tune without touching the search code.
"""

from __future__ import annotations

from typing import Dict

from engine.board import Board, BOARD_SIZE
from engine.piece import Color, PieceType

# Material values, Soldier = 1.0 baseline. These are starting heuristics;
# re-tune after a few self-play rounds.
PIECE_VALUE: Dict[PieceType, float] = {
    PieceType.KING: 1000.0,
    PieceType.ROOK: 9.0,
    PieceType.CANNON: 4.5,
    PieceType.HORSE: 4.0,
    PieceType.SOLDIER: 1.0,
}

# Win/loss bonus applied when a king is missing (redundant with search's
# terminal handling, but keeps the evaluator safe to call on any position).
KING_MISSING_BONUS = 10000.0


def _material(board: Board) -> float:
    """Signed material sum from Red's perspective (Red +, Black -)."""
    score = 0.0
    for _sq, piece in board.pieces():
        v = PIECE_VALUE[piece.ptype]
        score += v if piece.color is Color.RED else -v
    return score


def _positional(board: Board) -> float:
    """Small positional terms, Red-positive.

    Kept lightweight so the evaluator stays fast in the inner search loop.
    The main terms:
      * soldiers gain value as they advance toward the enemy back rank;
      * horses and cannons get a tiny centralization bonus.
    """
    score = 0.0
    for (row, col), piece in board.pieces():
        sign = 1.0 if piece.color is Color.RED else -1.0
        if piece.ptype is PieceType.SOLDIER:
            # Red soldiers advance by increasing row; Black by decreasing.
            advanced = row if piece.color is Color.RED else (BOARD_SIZE - 1 - row)
            score += sign * 0.1 * advanced
        elif piece.ptype is PieceType.HORSE:
            # Mild centralization: distance from center column 3.
            score += sign * 0.05 * (3 - abs(col - 3))
    return score


def evaluate(board: Board) -> float:
    """Static evaluation from the perspective of the side to move.

    Positive means the side to move is better; negative means worse. For a
    Red-to-move position this returns (red_material + red_pos - black_*), and
    for Black to move it returns the negation.
    """
    # Terminal: a side whose king is gone has lost catastrophically.
    if board.king_is_missing(Color.BLACK):
        base = KING_MISSING_BONUS
    elif board.king_is_missing(Color.RED):
        base = -KING_MISSING_BONUS
    else:
        base = _material(board) + _positional(board)

    # base is Red-positive; flip sign when it's Black's turn so the value is
    # always relative to the side to move (negamax convention).
    return base if board.side_to_move is Color.RED else -base
