"""engine/board.py — Board-state representation for Mini Xiangqi (迷你象棋).

This is the *pure state* layer of the engine. It deliberately knows nothing
about which moves are legal (that lives in ``move_generator.py``) nor about
check / checkmate / perpetual-check rules (that lives in ``rules.py``). Its
only responsibilities are:

  * store the piece placement, side to move and position history,
  * apply and revert moves (``make_move`` / ``undo_move``),
  * answer geometric questions (palace bounds, king location),
  * serialize to / from a FEN-like string and to a neural-network friendly
    tensor (``to_fen`` / ``from_fen`` / ``to_planes``).

The piece-level types (``Color``, ``PieceType``, ``Piece``) live in
``engine/piece.py``; this module imports them to avoid duplication and to
keep a clean import graph (piece <- board <- move_generator <- rules).

Variant specification
---------------------
* Board: 7 ranks (rows) x 7 files (columns) of intersections -> 49 squares.
* Sides: Red at the bottom (rows 0-2), Black at the top (rows 4-6).
* Piece set (no advisors / elephants):
      K = King    (帥 / 將)   confined to the 3x3 palace
      R = Rook    (車)
      N = Horse   (馬)
      C = Cannon  (炮)
      P = Soldier (兵 / 卒)
* Starting setup (row 0 = Red's back rank, at the bottom):
      row 0:   R C N K N C R        <- Red
      row 1:   . P P P P P .        <- Red soldiers (central five files)
      rows 2-4: empty
      row 5:   . p p p p p .        <- Black soldiers
      row 6:   r c n k n c r        <- Black
* Palace (九宫): Red   -> rows 0-2, cols 2-4
                 Black -> rows 4-6, cols 2-4
* No river in this variant.
* Terminal signal: a side whose King is no longer on the board has lost (its
  King was captured). Finer win/loss rules (checkmate, stalemate, perpetuals)
  belong to ``rules.py``.

Coordinates
-----------
Squares are addressed as ``(row, col)`` with ``row`` growing from Red's side
(row 0) up to Black's side (row 6) and ``col`` from left (0) to right (6).
Algebraic notation uses files ``a..g`` (col 0..6) and ranks ``1..7``
(row 0..6), so Red's left rook starts on ``a1`` and Black's king on ``d7``.
"""

from __future__ import annotations

from collections import Counter
import random
from typing import Iterator, List, Optional, Tuple

from .piece import Color, Piece, PieceType
from .move import Move, Square, in_bounds, square_to_alg, alg_to_square

# Numpy is only required for the neural-network tensor view. Import it lazily
# so the engine stays runnable in a bare Python without scientific packages.
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BOARD_SIZE = 7                            # 7x7 intersections
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE     # 49 squares

# Palace bounds per side, as (row_min, row_max, col_min, col_max) inclusive.
PALACE_RED: Tuple[int, int, int, int] = (0, 2, 2, 4)
PALACE_BLACK: Tuple[int, int, int, int] = (4, 6, 2, 4)

# Soldiers start on files a, c, d, e, g (cols 0, 2, 3, 4, 6) of the second
# rank. Edge files have soldiers (matching standard Xiangqi's alternating
# pattern) and the center file blocks the two kings from facing each other.
# (row 1 for Red, row 5 for Black).
_SOLDIER_COLS: Tuple[int, ...] = (0, 2, 3, 4, 6)

# Back-rank layout, col 0..6 (symmetric around the king on col 3).
_BACK_RANK: Tuple[str, ...] = ("R", "C", "N", "K", "N", "C", "R")

# --------------------------------------------------------------------------- #
# Zobrist hashing (incremental position hashing for fast repetition detection)
# --------------------------------------------------------------------------- #
# A position's hash is the XOR of a random bitstring per (square, piece-kind)
# present, plus a side-to-move key. make_move/undo_move update it in O(1)
# instead of rebuilding a board string, which profiling showed cost ~16s/game.

# Piece-kind index: color (0=Red, 1=Black) * 5 + piece-type index.
_PIECE_TYPE_INDEX = {
    PieceType.KING: 0, PieceType.ROOK: 1, PieceType.HORSE: 2,
    PieceType.CANNON: 3, PieceType.SOLDIER: 4,
}
_NUM_PIECE_KINDS = 10


def _piece_kind(piece: Piece) -> int:
    color_idx = 0 if piece.color is Color.RED else 1
    return color_idx * 5 + _PIECE_TYPE_INDEX[piece.ptype]


def _sq_index(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


# Random bitstrings, generated once with a fixed seed for determinism.
_zrng = random.Random(0xC0FFEE)
ZOBRIST_PIECE = [
    [_zrng.getrandbits(64) for _ in range(_NUM_PIECE_KINDS)]
    for _ in range(NUM_SQUARES)
]
ZOBRIST_SIDE = _zrng.getrandbits(64)  # XORed in when Black is to move


# --------------------------------------------------------------------------- #
# Board
# --------------------------------------------------------------------------- #
class _UndoRecord:
    """Snapshot of everything :meth:`Board.make_move` changes, for undo."""

    __slots__ = ("move", "mover", "captured", "side", "halfmove_before")

    def __init__(
        self,
        move: Move,
        mover: Piece,
        captured: Optional[Piece],
        side: Color,
        halfmove_before: int,
    ) -> None:
        self.move = move
        self.mover = mover
        self.captured = captured
        self.side = side  # side that was to move *before* the move
        self.halfmove_before = halfmove_before  # clock value before this move


class Board:
    """Mini Xiangqi board state.

    The board is a 7x7 grid of optional :class:`Piece`. Red is to move first.
    Moves are applied with :meth:`make_move` and reverted with :meth:`undo_move`
    so that search / self-play can explore lines cheaply.
    """

    def __init__(self) -> None:
        self._grid: List[List[Optional[Piece]]] = [
            [None] * BOARD_SIZE for _ in range(BOARD_SIZE)
        ]
        self._side: Color = Color.RED
        self._fullmove: int = 1          # increments after Black moves
        self._halfmove: int = 0          # plies since last capture or soldier advance
        # Zobrist-hash history, one entry per position reached (incl. start).
        # `_zcounts` mirrors it for O(1) repetition queries.
        self._zstack: List[int] = []
        self._zcounts: Counter = Counter()
        self._history: List[_UndoRecord] = []
        self.reset()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Restore the standard starting position."""
        self._grid = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self._side = Color.RED
        self._fullmove = 1
        self._halfmove = 0
        self._zstack = []
        self._zcounts = Counter()
        self._history = []

        # Red back rank (row 0) and Black back rank (row 6).
        for col, letter in enumerate(_BACK_RANK):
            self._grid[0][col] = Piece.from_char(letter)
            self._grid[BOARD_SIZE - 1][col] = Piece.from_char(letter.lower())
        # Soldiers on the second rank of each side.
        for col in _SOLDIER_COLS:
            self._grid[1][col] = Piece(PieceType.SOLDIER, Color.RED)
            self._grid[BOARD_SIZE - 2][col] = Piece(PieceType.SOLDIER, Color.BLACK)

        # Cache king positions for O(1) find_king. Kings start on file d (col 3):
        # Red at row 0, Black at the last row.
        self._king_pos = {
            Color.RED: (0, 3),
            Color.BLACK: (BOARD_SIZE - 1, 3),
        }

        z = self._compute_zhash()
        self._zstack = [z]
        self._zcounts = Counter({z: 1})

    # ------------------------------------------------------------------ #
    # Basic queries
    # ------------------------------------------------------------------ #
    @property
    def side_to_move(self) -> Color:
        return self._side

    @property
    def fullmove_number(self) -> int:
        return self._fullmove

    @property
    def halfmove_clock(self) -> int:
        return self._halfmove

    def piece_at(self, row: int, col: int) -> Optional[Piece]:
        """Return the piece on a square, or ``None`` if empty / off-board."""
        if not in_bounds(row, col):
            return None
        return self._grid[row][col]

    def is_empty(self, row: int, col: int) -> bool:
        return in_bounds(row, col) and self._grid[row][col] is None

    def is_in_palace(self, row: int, col: int, color: Color) -> bool:
        """Whether ``(row, col)`` lies inside the given side's 3x3 palace."""
        if not in_bounds(row, col):
            return False
        r0, r1, c0, c1 = PALACE_RED if color is Color.RED else PALACE_BLACK
        return r0 <= row <= r1 and c0 <= col <= c1

    def find_king(self, color: Color) -> Optional[Square]:
        """Location of ``color``'s King, or ``None`` if it has been captured.

        O(1) lookup into a cache maintained by make_move/undo_move (profiling
        showed the old 49-square scan was called ~1.75M times per game).
        """
        return self._king_pos[color]

    def pieces(self, color: Optional[Color] = None) -> Iterator[Tuple[Square, Piece]]:
        """Iterate ``(square, piece)``; optionally filtered by color."""
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self._grid[row][col]
                if piece is not None and (color is None or piece.color is color):
                    yield ((row, col), piece)

    # ------------------------------------------------------------------ #
    # Terminal state
    # ------------------------------------------------------------------ #
    def king_is_missing(self, color: Color) -> bool:
        return self.find_king(color) is None

    def is_terminal(self) -> bool:
        """True once a King has been captured (the game-deciding event)."""
        return self.king_is_missing(Color.RED) or self.king_is_missing(Color.BLACK)

    def winner(self) -> Optional[Color]:
        """The side whose King still stands, if the opponent's is gone.

        Returns ``None`` for non-terminal positions. This is the signal the
        self-play loop uses to assign terminal game rewards.
        """
        red_alive = not self.king_is_missing(Color.RED)
        black_alive = not self.king_is_missing(Color.BLACK)
        if red_alive and not black_alive:
            return Color.RED
        if black_alive and not red_alive:
            return Color.BLACK
        return None

    # ------------------------------------------------------------------ #
    # Move application
    # ------------------------------------------------------------------ #
    def make_move(self, move: Move) -> Optional[Piece]:
        """Apply ``move`` to the board and return the captured piece, if any.

        The move is applied unconditionally; legality is the caller's concern
        (typically ``move_generator.py``). Revert with :meth:`undo_move`.
        """
        mover = self._grid[move.from_row][move.from_col]
        if mover is None:
            raise ValueError(f"no piece to move from {move.from_sq}")
        captured = self._grid[move.to_row][move.to_col]

        # Snapshot the pre-move clock *before* mutating it.
        self._history.append(_UndoRecord(move, mover, captured, self._side, self._halfmove))

        self._grid[move.to_row][move.to_col] = mover
        self._grid[move.from_row][move.from_col] = None

        # Keep the king-position cache in sync.
        if mover.ptype is PieceType.KING:
            self._king_pos[mover.color] = (move.to_row, move.to_col)
        if captured is not None and captured.ptype is PieceType.KING:
            self._king_pos[captured.color] = None

        # Halfmove clock resets on captures and soldier advances, else grows.
        if captured is not None or mover.ptype is PieceType.SOLDIER:
            self._halfmove = 0
        else:
            self._halfmove += 1
        if self._side is Color.BLACK:
            self._fullmove += 1

        self._side = self._side.opponent

        # Update the Zobrist hash incrementally for repetition tracking (O(1);
        # XOR is its own inverse, so removing/adding a piece is just an XOR).
        z = self._zstack[-1]
        from_idx = _sq_index(move.from_row, move.from_col)
        to_idx = _sq_index(move.to_row, move.to_col)
        z ^= ZOBRIST_PIECE[from_idx][_piece_kind(mover)]
        z ^= ZOBRIST_PIECE[to_idx][_piece_kind(mover)]
        if captured is not None:
            z ^= ZOBRIST_PIECE[to_idx][_piece_kind(captured)]
        z ^= ZOBRIST_SIDE  # side to move flipped
        self._zstack.append(z)
        self._zcounts[z] += 1

        return captured

    def undo_move(self) -> Optional[_UndoRecord]:
        """Revert the last :meth:`make_move`. Returns the undone record."""
        if not self._history:
            return None

        # Drop the position reached by the move being undone.
        self._zcounts[self._zstack.pop()] -= 1
        # (Counter entries are left at 0; harmless and avoids key churn.)

        record = self._history.pop()
        self._side = record.side
        if record.side is Color.BLACK:
            self._fullmove -= 1
        # Restore the clock to its value *before* the undone move.
        self._halfmove = record.halfmove_before

        self._grid[record.move.from_row][record.move.from_col] = record.mover
        self._grid[record.move.to_row][record.move.to_col] = record.captured

        # Restore the king-position cache.
        if record.mover.ptype is PieceType.KING:
            self._king_pos[record.mover.color] = (record.move.from_row, record.move.from_col)
        if record.captured is not None and record.captured.ptype is PieceType.KING:
            self._king_pos[record.captured.color] = (record.move.to_row, record.move.to_col)

        return record

    # ------------------------------------------------------------------ #
    # Repetition / hashing
    # ------------------------------------------------------------------ #
    def _compute_zhash(self) -> int:
        """Compute the full Zobrist hash from scratch (XOR of all pieces + side).

        Used to (re)initialize the incremental hash in reset/from_fen/clone;
        make_move/undo_move then maintain it in O(1).
        """
        h = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                pc = self._grid[row][col]
                if pc is not None:
                    h ^= ZOBRIST_PIECE[_sq_index(row, col)][_piece_kind(pc)]
        if self._side is Color.BLACK:
            h ^= ZOBRIST_SIDE
        return h

    def position_key(self) -> str:
        """Compact, hashable description of placement + side to move.

        Two positions with the same key are identical for repetition purposes.
        """
        rows = []
        for row in range(BOARD_SIZE):
            rows.append(
                "".join(
                    self._grid[row][col].char if self._grid[row][col] else "."
                    for col in range(BOARD_SIZE)
                )
            )
        return "/".join(rows) + " " + self._side.value

    def repetition_count(self) -> int:
        """How many times the current position has occurred in this game."""
        return self._zcounts[self._zstack[-1]] if self._zstack else 1

    def __hash__(self) -> int:
        return hash(self.position_key())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Board) and self.position_key() == other.position_key()

    # ------------------------------------------------------------------ #
    # Copying
    # ------------------------------------------------------------------ #
    def clone(self) -> "Board":
        """Deep copy of the board, for parallel search branches."""
        new = Board.__new__(Board)
        new._grid = [
            [None if c is None else Piece(c.ptype, c.color) for c in row]
            for row in self._grid
        ]
        new._side = self._side
        new._fullmove = self._fullmove
        new._halfmove = self._halfmove
        new._king_pos = dict(self._king_pos)
        new._zstack = list(self._zstack)
        new._zcounts = Counter(self._zcounts)
        new._history = list(self._history)
        return new

    # ------------------------------------------------------------------ #
    # FEN (de)serialization
    # ------------------------------------------------------------------ #
    def to_fen(self) -> str:
        """Serialize placement + side + clocks to a FEN-like string.

        Ranks are listed from Black's side (row 6) down to Red's (row 0),
        matching standard xiangqi FEN ordering. Empty runs become digits.
        """
        rank_strs: List[str] = []
        for row in range(BOARD_SIZE - 1, -1, -1):
            run = 0
            chars: List[str] = []
            for col in range(BOARD_SIZE):
                piece = self._grid[row][col]
                if piece is None:
                    run += 1
                else:
                    if run:
                        chars.append(str(run))
                        run = 0
                    chars.append(piece.char)
            if run:
                chars.append(str(run))
            rank_strs.append("".join(chars))
        return (
            f"{'/'.join(rank_strs)} {self._side.value} "
            f"{self._halfmove} {self._fullmove}"
        )

    @classmethod
    def from_fen(cls, fen: str) -> "Board":
        """Build a board from a FEN string produced by :meth:`to_fen`.

        A single FEN with just placement + side (e.g. the starting placement)
        is also accepted; missing clocks default to 0 / 1.
        """
        parts = fen.strip().split()
        if not parts:
            raise ValueError("empty FEN")
        placement = parts[0]
        side = Color(parts[1]) if len(parts) > 1 else Color.RED
        halfmove = int(parts[2]) if len(parts) > 2 else 0
        fullmove = int(parts[3]) if len(parts) > 3 else 1

        board = cls.__new__(cls)
        board._grid = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        ranks = placement.split("/")
        if len(ranks) != BOARD_SIZE:
            raise ValueError(f"FEN must have {BOARD_SIZE} ranks, got {len(ranks)}")
        # ranks[0] is the top rank (row 6); map downward.
        for i, rank in enumerate(ranks):
            row = BOARD_SIZE - 1 - i
            col = 0
            for ch in rank:
                if ch.isdigit():
                    col += int(ch)
                else:
                    if col >= BOARD_SIZE:
                        raise ValueError(f"rank too long in FEN: {rank!r}")
                    board._grid[row][col] = Piece.from_char(ch)
                    col += 1
            if col != BOARD_SIZE:
                push_msg = f"rank too short in FEN: {rank!r}"
                raise ValueError(push_msg)

        # Populate the king-position cache from the parsed grid.
        board._king_pos = {Color.RED: None, Color.BLACK: None}
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                pc = board._grid[r][c]
                if pc is not None and pc.ptype is PieceType.KING:
                    board._king_pos[pc.color] = (r, c)

        board._side = side
        board._halfmove = halfmove
        board._fullmove = fullmove
        z = board._compute_zhash()
        board._zstack = [z]
        board._zcounts = Counter({z: 1})
        board._history = []
        return board

    # ------------------------------------------------------------------ #
    # Neural-network view
    # ------------------------------------------------------------------ #
    def to_planes(self, include_side: bool = True):
        """Encode the position as a ``(C, 7, 7)`` float32 tensor.

        The first 10 channels are binary piece-occupancy planes, ordered as
        ``[<color><piece>]`` with color Red then Black, and piece order
        King, Rook, Horse, Cannon, Soldier (5 per color -> 10 planes).

        If ``include_side`` is True, an 11th plane is all 1.0 when Red is to
        move and all 0.0 when Black is to move, letting the network condition
        on side-to-move.

        Requires numpy. The training pipeline guarantees numpy is installed.
        """
        if not _HAS_NUMPY:
            raise RuntimeError(
                "numpy is required for Board.to_planes(); install numpy or "
                "run the engine in a pure-Python context."
            )
        piece_order = [
            PieceType.KING, PieceType.ROOK, PieceType.HORSE,
            PieceType.CANNON, PieceType.SOLDIER,
        ]
        n_planes = 11 if include_side else 10
        planes = np.zeros((n_planes, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self._grid[row][col]
                if piece is None:
                    continue
                color_idx = 0 if piece.color is Color.RED else 1
                piece_idx = piece_order.index(piece.ptype)
                planes[color_idx * 5 + piece_idx][row][col] = 1.0
        if include_side:
            planes[10][:, :] = 1.0 if self._side is Color.RED else 0.0
        return planes

    # ------------------------------------------------------------------ #
    # Pretty printing
    # ------------------------------------------------------------------ #
    def __str__(self) -> str:
        lines: List[str] = []
        # Files header.
        lines.append("   " + " ".join(chr(ord("a") + c) for c in range(BOARD_SIZE)))
        for row in range(BOARD_SIZE - 1, -1, -1):
            cells = []
            for col in range(BOARD_SIZE):
                piece = self._grid[row][col]
                cells.append("." if piece is None else piece.char)
            lines.append(f"{row + 1}  " + " ".join(cells))
        lines.append(f"   side to move: {self._side.name.lower()}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Board(side={self._side.value}, fen={self.to_fen()!r})"
