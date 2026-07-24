"""engine/rules.py — Game-phase and terminal rules for Mini Xiangqi.

This module builds on :mod:`engine.move_generator` (for legal moves and check
detection) and answers the higher-level questions a game loop, search, or
self-play trainer needs:

  * Is the game over, and who won (:func:`game_result`)?
  * Is the side to move in checkmate / stalemate (:func:`is_checkmate`,
    :func:`is_stalemate`)?
  * Is the position a draw by repetition (:func:`is_repetition_draw`)?

Draw / repetition policy
------------------------
Xiangqi rules treat perpetual check and repeated positions specially, but the
exact policy varies by rule set. This module takes a *simple, configurable*
stance suitable for a training pipeline:

  * A position repeated ``repetition_threshold`` times (default 3) is a draw.
  * A side with no legal move loses (there is no stalemate draw in xiangqi):
    if it's that side's turn and it's in check -> checkmate (it loses); if
    it's not in check -> it still loses (xiangqi "dead" position), since
    xiangqi has no stalemate draw.

The repetition threshold can be tuned without touching the callers.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from .board import Board
from .move_generator import in_check, legal_moves
from .piece import Color

# Default: a position occurring 3 times is a draw (configurable).
DEFAULT_REPETITION_THRESHOLD = 3


class GameOutcome(Enum):
    """Coarse outcome of a finished game, from Red's perspective.

    ``RED_WINS`` / ``BLACK_WINS`` mean that side delivered a won position;
    ``DRAW`` covers repetition draws. Unfinished games are represented by a
    ``None`` :class:`GameResult` rather than an outcome value.
    """

    RED_WINS = "red_wins"
    BLACK_WINS = "black_wins"
    DRAW = "draw"



class GameResult:
    """Terminal-state descriptor.

    Attributes
    ----------
    outcome : GameOutcome
        Who won, or a draw.
    winner : Optional[Color]
        The winning :class:`Color`, or ``None`` for a draw. Convenience alias
        that self-play reward code can use directly.
    reason : str
        Human-readable reason, e.g. ``"checkmate"``, ``"no_legal_moves"``,
        ``"king_captured"``, ``"repetition"``.
    """

    __slots__ = ("outcome", "winner", "reason")

    def __init__(self, outcome: GameOutcome, reason: str) -> None:
        self.outcome = outcome
        self.winner: Optional[Color] = (
            Color.RED if outcome is GameOutcome.RED_WINS
            else Color.BLACK if outcome is GameOutcome.BLACK_WINS
            else None
        )
        self.reason = reason

    @property
    def is_terminal(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"GameResult({self.outcome.value}, reason={self.reason!r})"


def has_no_legal_moves(board: Board) -> bool:
    """Whether the side to move has zero legal moves."""
    return len(legal_moves(board)) == 0


def is_checkmate(board: Board) -> bool:
    """Whether the side to move is checkmated (in check, with no legal move)."""
    color = board.side_to_move
    return in_check(board, color) and has_no_legal_moves(board)


def is_stalemate(board: Board) -> bool:
    """Not-in-check but no legal move.

    In strict xiangqi this still loses for the stuck side, but we expose it as
    a distinct predicate for diagnostics and for callers that want chess-like
    semantics.
    """
    color = board.side_to_move
    return not in_check(board, color) and has_no_legal_moves(board)


def is_repetition_draw(board: Board, threshold: int = DEFAULT_REPETITION_THRESHOLD) -> bool:
    """Whether the current position has recurred ``threshold`` times."""
    return board.repetition_count() >= threshold


def game_result(
    board: Board,
    repetition_threshold: int = DEFAULT_REPETITION_THRESHOLD,
) -> Optional[GameResult]:
    """Classify the position as terminal or non-terminal.

    Precedence (first match wins):

    1. A missing King -> that side lost (king captured). This is the raw
       terminal signal surfaced by :class:`Board` itself.
    2. Repetition draw (>= ``repetition_threshold`` occurrences).
    3. The side to move has no legal move:
         * in check  -> checkmate, that side loses
         * not in check -> still a loss for that side (xiangqi has no
           stalemate draw); reported with reason ``"no_legal_moves"``.

    Returns ``None`` if the game is still in progress.
    """
    # 1. King captured.
    winner = board.winner()
    if winner is not None:
        outcome = GameOutcome.RED_WINS if winner is Color.RED else GameOutcome.BLACK_WINS
        return GameResult(outcome, reason="king_captured")

    # 2. Repetition draw.
    if is_repetition_draw(board, repetition_threshold):
        return GameResult(GameOutcome.DRAW, reason="repetition")

    # 3. No legal moves.
    if has_no_legal_moves(board):
        loser = board.side_to_move
        if in_check(board, loser):
            outcome = (
                GameOutcome.BLACK_WINS if loser is Color.RED else GameOutcome.RED_WINS
            )
            return GameResult(outcome, reason="checkmate")
        # Xiangqi: no stalemate draw — the immobilized side loses.
        outcome = (
            GameOutcome.BLACK_WINS if loser is Color.RED else GameOutcome.RED_WINS
        )
        return GameResult(outcome, reason="no_legal_moves")

    return None


def is_game_over(
    board: Board,
    repetition_threshold: int = DEFAULT_REPETITION_THRESHOLD,
) -> bool:
    """Boolean convenience wrapper around :func:`game_result`."""
    return game_result(board, repetition_threshold) is not None
