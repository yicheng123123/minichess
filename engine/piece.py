"""engine/piece.py — Piece, Color, and PieceType definitions for Mini Xiangqi.

This module holds *only* the piece-level types, with no dependency on the
board. That keeps it importable from ``board.py``, ``move_generator.py`` and
``rules.py`` without creating a circular import.

Piece set (no advisors / elephants):
    K = King    (帥 / 將)
    R = Rook    (車)
    N = Horse   (馬)
    C = Cannon  (炮)
    P = Soldier (兵 / 卒)

Character code is the FEN letter: uppercase for Red, lowercase for Black,
e.g. ``'K'`` -> red King, ``'n'`` -> black Horse.
"""

from __future__ import annotations

from enum import Enum


class Color(Enum):
    """The two sides. Red moves first and starts at the bottom of the board."""

    RED = "r"
    BLACK = "b"

    @property
    def opponent(self) -> "Color":
        return Color.BLACK if self is Color.RED else Color.RED

    @property
    def back_rank_row(self) -> int:
        """Row index of this side's back rank (where the heavy pieces start)."""
        return 0 if self is Color.RED else 6

    @property
    def soldier_rank_row(self) -> int:
        """Row index of this side's initial soldier rank."""
        return 1 if self is Color.RED else 5

    @property
    def forward(self) -> int:
        """Direction soldiers move in (increasing row for Red, decreasing for Black)."""
        return 1 if self is Color.RED else -1


class PieceType(Enum):
    KING = "K"
    ROOK = "R"
    HORSE = "N"
    CANNON = "C"
    SOLDIER = "P"

    @classmethod
    def from_char(cls, ch: str) -> "PieceType":
        try:
            return cls(ch.upper())
        except ValueError as exc:
            raise ValueError(f"unknown piece char: {ch!r}") from exc


class Piece:
    """An immutable piece: a type paired with a color."""

    __slots__ = ("ptype", "color")

    def __init__(self, ptype: PieceType, color: Color) -> None:
        self.ptype = ptype
        self.color = color

    @property
    def char(self) -> str:
        """Single-letter code: uppercase for Red, lowercase for Black."""
        base = self.ptype.value
        return base if self.color is Color.RED else base.lower()

    @classmethod
    def from_char(cls, ch: str) -> "Piece":
        """Parse a FEN letter, e.g. ``'K'`` -> red King, ``'n'`` -> black Horse."""
        if not ch or len(ch) != 1:
            raise ValueError(f"invalid piece char: {ch!r}")
        color = Color.RED if ch.isupper() else Color.BLACK
        return cls(PieceType.from_char(ch), color)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Piece)
            and self.ptype is other.ptype
            and self.color is other.color
        )

    def __hash__(self) -> int:
        return hash((self.ptype, self.color))

    def __repr__(self) -> str:
        return f"Piece({self.ptype.name}, {self.color.name})"
