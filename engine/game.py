"""engine/game.py — Game controller for Mini Xiangqi.

This module ties together the board, move generator, and rules into a
high-level game loop that external modules (GUI, self-play, API) can drive
without knowing the internal layering.

Responsibilities:
  * Track whose turn it is (delegated to Board).
  * Accept a move, validate legality, apply it.
  * Detect game-over conditions after each move.
  * Provide undo (take-back) support.
  * Expose the full move history for replay / serialization.

Typical usage::

    game = Game()
    game.play("d2d3")       # Red advances a soldier
    game.play("d6d5")       # Black responds
    print(game.is_over)     # False
    game.undo()             # take back Black's move
    print(game.move_history)  # ["d2d3"]
"""

from __future__ import annotations

from typing import List, Optional

from .board import Board
from .move import Move
from .move_generator import legal_moves, in_check
from .piece import Color
from .rules import GameResult, game_result, DEFAULT_REPETITION_THRESHOLD


class Game:
    """High-level game controller wrapping Board + rules.

    Attributes
    ----------
    board : Board
        The current board state.
    result : Optional[GameResult]
        Non-None once the game has ended.
    """

    def __init__(
        self,
        board: Optional[Board] = None,
        repetition_threshold: int = DEFAULT_REPETITION_THRESHOLD,
    ) -> None:
        self.board = board or Board()
        self.repetition_threshold = repetition_threshold
        self._history: List[Move] = []
        self._result: Optional[GameResult] = None
        # Check if the starting position is already terminal (e.g. from FEN).
        self._update_result()

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def side_to_move(self) -> Color:
        return self.board.side_to_move

    @property
    def is_over(self) -> bool:
        return self._result is not None

    @property
    def result(self) -> Optional[GameResult]:
        return self._result

    @property
    def move_history(self) -> List[str]:
        """UCI strings of all moves played so far."""
        return [m.uci() for m in self._history]

    @property
    def ply(self) -> int:
        """Number of half-moves (plies) played."""
        return len(self._history)

    # ------------------------------------------------------------------ #
    # Move interface
    # ------------------------------------------------------------------ #
    def legal_moves(self) -> List[Move]:
        """All legal moves for the side to move."""
        if self.is_over:
            return []
        return legal_moves(self.board)

    def is_legal(self, move: Move) -> bool:
        """Whether ``move`` is legal in the current position."""
        return move in self.legal_moves()

    def play(self, move) -> Optional[GameResult]:
        """Apply a move and return the game result if the game ended.

        Parameters
        ----------
        move : Move or str
            A Move object or a UCI string like ``"d2d3"``.

        Raises
        ------
        ValueError
            If the game is already over or the move is illegal.
        """
        if self.is_over:
            raise ValueError("game is already over")

        if isinstance(move, str):
            move = Move.from_uci(move)

        if move not in self.legal_moves():
            raise ValueError(f"illegal move: {move.uci()}")

        self.board.make_move(move)
        self._history.append(move)
        self._update_result()
        return self._result

    def undo(self) -> Optional[Move]:
        """Take back the last move. Returns the undone Move, or None."""
        if not self._history:
            return None
        self.board.undo_move()
        move = self._history.pop()
        self._result = None  # game is no longer over after undo
        # Re-check: the position before the undone move might itself be terminal
        # (e.g. undoing a move out of a checkmate position).
        self._update_result()
        return move

    # ------------------------------------------------------------------ #
    # Query helpers
    # ------------------------------------------------------------------ #
    def in_check(self) -> bool:
        """Whether the side to move is currently in check."""
        return in_check(self.board, self.board.side_to_move)

    def winner(self) -> Optional[Color]:
        """The winning color if the game is over with a winner, else None."""
        if self._result is None:
            return None
        return self._result.winner

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_fen(self) -> str:
        return self.board.to_fen()

    @classmethod
    def from_fen(cls, fen: str, **kwargs) -> "Game":
        """Create a Game from a FEN string."""
        board = Board.from_fen(fen)
        return cls(board=board, **kwargs)

    @classmethod
    def from_moves(cls, moves: List[str], **kwargs) -> "Game":
        """Replay a list of UCI moves from the starting position."""
        game = cls(**kwargs)
        for uci in moves:
            game.play(uci)
        return game

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _update_result(self) -> None:
        self._result = game_result(self.board, self.repetition_threshold)

    def __repr__(self) -> str:
        status = "over" if self.is_over else "in progress"
        return f"Game(ply={self.ply}, {status}, side={self.side_to_move.value})"
