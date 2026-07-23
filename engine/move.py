"""engine/move.py — Move representation for Mini Xiangqi.

A Move is a directed edge between two squares on the 7x7 board. It carries no
legality information — that is the responsibility of ``move_generator.py``.

All higher-level modules (search, MCTS, GUI, self-play) operate on Move objects,
making this the universal "action" type across the entire project.

Coordinates use ``(row, col)`` with row 0 = Red's back rank, col 0 = file 'a'.
UCI-style algebraic notation: ``"a1a4"`` means from (0,0) to (3,0).
"""

from __future__ import annotations

from typing import Tuple

Square = Tuple[int, int]

BOARD_SIZE = 7


def in_bounds(row: int, col: int) -> bool:
    """Whether (row, col) is a valid square on the 7x7 board."""
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def square_to_alg(row: int, col: int) -> str:
    """``(row, col)`` -> algebraic, e.g. (0, 0) -> 'a1', (6, 3) -> 'd7'."""
    if not in_bounds(row, col):
        raise ValueError(f"square out of bounds: ({row}, {col})")
    return f"{chr(ord('a') + col)}{row + 1}"


def alg_to_square(alg: str) -> Square:
    """Algebraic -> ``(row, col)``, e.g. 'a1' -> (0, 0)."""
    alg = alg.strip()
    if len(alg) != 2 or alg[0] not in "abcdefg" or alg[1] not in "1234567":
        raise ValueError(f"invalid algebraic square: {alg!r}")
    col = ord(alg[0]) - ord("a")
    row = int(alg[1]) - 1
    return (row, col)


class Move:
    """A move between two squares. Legality is decided elsewhere.

    Stores both ``(row, col)`` pairs so the engine never has to recompute
    them; the algebraic form is available via :meth:`uci`.
    """

    __slots__ = ("from_row", "from_col", "to_row", "to_col")

    def __init__(self, from_sq: Square, to_sq: Square) -> None:
        self.from_row, self.from_col = from_sq
        self.to_row, self.to_col = to_sq

    @classmethod
    def from_uci(cls, uci: str) -> "Move":
        """Build a move from a 4-char UCI string, e.g. ``'a1a4'``."""
        uci = uci.strip()
        if len(uci) != 4:
            raise ValueError(f"invalid UCI move: {uci!r}")
        return cls(alg_to_square(uci[:2]), alg_to_square(uci[2:]))

    @property
    def from_sq(self) -> Square:
        return (self.from_row, self.from_col)

    @property
    def to_sq(self) -> Square:
        return (self.to_row, self.to_col)

    def uci(self) -> str:
        """4-character UCI notation, e.g. 'a1a4'."""
        return square_to_alg(self.from_row, self.from_col) + square_to_alg(
            self.to_row, self.to_col
        )

    def is_capture(self, board) -> bool:
        """Whether this move captures a piece on the target square.

        Requires a board reference to check occupancy.
        """
        return board.piece_at(self.to_row, self.to_col) is not None

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Move)
            and self.from_row == other.from_row
            and self.from_col == other.from_col
            and self.to_row == other.to_row
            and self.to_col == other.to_col
        )

    def __hash__(self) -> int:
        return hash((self.from_row, self.from_col, self.to_row, self.to_col))

    def __repr__(self) -> str:
        return f"Move({self.uci()})"

    def __str__(self) -> str:
        return self.uci()
