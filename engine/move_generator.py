"""engine/move_generator.py — Move generation for Mini Xiangqi.

This module produces *pseudo-legal* and *legal* moves from a
:class:`engine.board.Board`. Pseudo-legal moves respect each piece's movement
geometry (including the King's palace confinement) and the rule that you may
not capture your own piece, but they ignore "leaving your King in check".
Legal moves additionally filter out any move that would leave the mover's own
King in check or in a face-to-face "flying generals" situation.

It also exposes the attack primitives that :mod:`engine.rules` needs:
:func:`square_attacked_by` (is a square attacked by a color?) and
:func:`in_check` (is a color's king attacked?).

Movement rules (this variant)
-----------------------------
* **Rook (R):** any number of empty squares orthogonally; stops at the first
  piece (capturing it if it's an enemy).
* **Horse (N):** one step orthogonally then one step diagonally outward. The
  orthogonal "leg" square must be empty (the classic horse-leg block). 8
  candidate destinations.
* **Cannon (C):** moves like a rook *to an empty square*; to capture it must
  jump exactly one piece (the "screen") of either color and land on an enemy
  beyond it.
* **King (K):** one step orthogonally, and must stay inside its 3x3 palace.
  Two kings may not face each other on an empty file with nothing between them
  ("flying generals") — enforced at the legality stage.
* **Soldier (P):** moves one step forward (toward the enemy back rank). This
  variant has no river, so soldiers do *not* gain sideways movement; the one
  concession is that a soldier that has reached the enemy back rank may step
  sideways (otherwise it would be stuck). Captures follow the same forward
  direction.
"""

from __future__ import annotations

from typing import List

from .board import Board, Move, in_bounds
from .piece import Color, Piece, PieceType

# Orthogonal directions for Rook / Cannon / King.
_ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _sign(x: int) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0

# Horse moves as (leg_dr, leg_dc, dest_dr, dest_dc). The leg is the adjacent
# orthogonal square that must be empty; dest is the final landing square.
_HORSE = (
    (1, 0, 2, 1), (1, 0, 2, -1),     # leg down, then diag
    (-1, 0, -2, 1), (-1, 0, -2, -1),  # leg up
    (0, 1, 1, 2), (0, 1, -1, 2),      # leg right
    (0, -1, 1, -2), (0, -1, -1, -2),  # leg left
)


def _add(piece_moves: List[Move], board: Board, fr_row: int, fr_col: int,
         to_row: int, to_col: int, color: Color) -> None:
    """Append the move to ``piece_moves`` if the target is on-board and not friendly.

    This is the shared "can I land here?" check: off-board and own-piece
    squares are skipped; empty and enemy squares are allowed.
    """
    if not in_bounds(to_row, to_col):
        return
    target = board.piece_at(to_row, to_col)
    if target is not None and target.color is color:
        return  # can't capture own piece
    piece_moves.append(Move((fr_row, fr_col), (to_row, to_col)))


# --------------------------------------------------------------------------- #
# Per-piece pseudo-legal generators
# --------------------------------------------------------------------------- #
def _gen_rook(board: Board, row: int, col: int, color: Color, out: List[Move]) -> None:
    for dr, dc in _ORTHO:
        r, c = row + dr, col + dc
        while in_bounds(r, c):
            target = board.piece_at(r, c)
            if target is None:
                out.append(Move((row, col), (r, c)))
            else:
                if target.color is not color:
                    out.append(Move((row, col), (r, c)))  # capture, then stop
                break
            r += dr
            c += dc


def _gen_horse(board: Board, row: int, col: int, color: Color, out: List[Move]) -> None:
    for leg_dr, leg_dc, dest_dr, dest_dc in _HORSE:
        leg_r, leg_c = row + leg_dr, col + leg_dc
        if not in_bounds(leg_r, leg_c):
            continue
        if board.piece_at(leg_r, leg_c) is not None:
            continue  # horse leg is blocked
        _add(out, board, row, col, row + dest_dr, col + dest_dc, color)


def _gen_cannon(board: Board, row: int, col: int, color: Color, out: List[Move]) -> None:
    for dr, dc in _ORTHO:
        r, c = row + dr, col + dc
        # Phase 1: slide to empty squares until we hit any piece (the screen).
        screen_found = False
        while in_bounds(r, c):
            target = board.piece_at(r, c)
            if not screen_found:
                if target is None:
                    out.append(Move((row, col), (r, c)))
                else:
                    screen_found = True  # first piece is the screen
            else:
                # Phase 2: look for an enemy beyond the screen to capture.
                if target is not None:
                    if target.color is not color:
                        out.append(Move((row, col), (r, c)))
                    break  # screen + next piece ends this ray
            r += dr
            c += dc


def _gen_king(board: Board, row: int, col: int, color: Color, out: List[Move]) -> None:
    for dr, dc in _ORTHO:
        nr, nc = row + dr, col + dc
        # King must remain within its own palace.
        if not board.is_in_palace(nr, nc, color):
            continue
        _add(out, board, row, col, nr, nc, color)


def _gen_soldier(board: Board, row: int, col: int, color: Color, out: List[Move]) -> None:
    fwd = color.forward
    # Forward step is always available.
    _add(out, board, row, col, row + fwd, col, color)
    # Sideways step is always available (no river restriction in this variant).
    _add(out, board, row, col, row, col - 1, color)
    _add(out, board, row, col, row, col + 1, color)


_GENERATORS = {
    PieceType.ROOK: _gen_rook,
    PieceType.HORSE: _gen_horse,
    PieceType.CANNON: _gen_cannon,
    PieceType.KING: _gen_king,
    PieceType.SOLDIER: _gen_soldier,
}


def pseudo_legal_moves(board: Board, color: Color | None = None) -> List[Move]:
    """All pseudo-legal moves for ``color`` (defaults to the side to move).

    Pseudo-legal means piece geometry + no friendly capture is respected, but
    moves that leave one's own King in check are *not* filtered out.
    """
    if color is None:
        color = board.side_to_move
    moves: List[Move] = []
    for (row, col), piece in board.pieces(color):
        _GENERATORS[piece.ptype](board, row, col, color, moves)
    return moves


def piece_moves(board: Board, row: int, col: int) -> List[Move]:
    """Pseudo-legal moves for the single piece on ``(row, col)``."""
    piece = board.piece_at(row, col)
    if piece is None:
        return []
    moves: List[Move] = []
    _GENERATORS[piece.ptype](board, row, col, piece.color, moves)
    return moves


# --------------------------------------------------------------------------- #
# Attack detection (used by legality + rules)
# --------------------------------------------------------------------------- #
def square_attacked_by(board: Board, row: int, col: int, by: Color) -> bool:
    """Whether square ``(row, col)`` is attacked by any piece of color ``by``.

    This is the core primitive for check detection. It tests each attack
    pattern directly rather than enumerating moves, so it's O(board) per call.
    """
    # --- Rook / King / Soldier attacks (orthogonal & forward) -------------- #
    for dr, dc in _ORTHO:
        # The first adjacent piece along the ray could be a King (1 step) or a
        # Rook (any distance) attacking this square.
        r, c = row + dr, col + dc
        if in_bounds(r, c):
            p = board.piece_at(r, c)
            if p is not None and p.color is by:
                if p.ptype is PieceType.KING or p.ptype is PieceType.ROOK:
                    return True
        # Rook attacks from further out along the same ray.
        r, c = row + dr, col + dc
        while in_bounds(r, c):
            p = board.piece_at(r, c)
            if p is not None:
                if p.color is by and p.ptype is PieceType.ROOK:
                    return True
                break
            r += dr
            c += dc

    # --- Soldier attacks --------------------------------------------------- #
    # A soldier attacks the square directly *in front of* it from our point of
    # view, i.e. a `by`-soldier sits at (row - by.forward, col).
    sr = row - by.forward
    if in_bounds(sr, col):
        p = board.piece_at(sr, col)
        if p is not None and p.color is by and p.ptype is PieceType.SOLDIER:
            return True
    # Sideways soldier attack: soldiers can always step sideways, so a `by`
    # soldier adjacent in file on the same rank attacks this square.
    for dc in (-1, 1):
        sc = col + dc
        if in_bounds(row, sc):
            p = board.piece_at(row, sc)
            if (p is not None and p.color is by
                    and p.ptype is PieceType.SOLDIER):
                return True

    # --- Horse attacks ----------------------------------------------------- #
    # A horse attacks (row, col) if a `by`-horse sits on any of the 8 horse
    # squares around it AND the "leg" (the orthogonal square one step from the
    # horse toward the target along the long axis) is empty. We find each
    # candidate horse square via the _HORSE table, then compute the leg
    # geometrically from horse -> target so it's always the right square.
    for _leg_dr, _leg_dc, dest_dr, dest_dc in _HORSE:
        hr, hc = row + dest_dr, col + dest_dc  # where the horse stands
        if not in_bounds(hr, hc):
            continue
        p = board.piece_at(hr, hc)
        if p is None or p.color is not by or p.ptype is not PieceType.HORSE:
            continue
        # The two-square axis of the jump is the larger of |dest_dr|/|dest_dc|.
        # The leg is one step from the horse along that axis toward the target.
        if abs(dest_dr) > abs(dest_dc):
            leg_r, leg_c = hr - _sign(dest_dr), hc
        else:
            leg_r, leg_c = hr, hc - _sign(dest_dc)
        if in_bounds(leg_r, leg_c) and board.piece_at(leg_r, leg_c) is None:
            return True

    # --- Cannon attacks ---------------------------------------------------- #
    # A cannon attacks if, along an orthogonal ray, there is exactly one
    # screen and then this square holds... anything (cannon attacks the square
    # by virtue of being able to capture whatever is on it). For "is the king
    # in check" the target is the king; for a general square we say the square
    # is attacked if a cannon could capture a piece sitting on it. To keep the
    # semantics simple and correct for check detection (the main caller), we
    # require a piece on (row, col) for the cannon to "attack" it.
    target_piece = board.piece_at(row, col)
    for dr, dc in _ORTHO:
        r, c = row + dr, col + dc
        screen_found = False
        while in_bounds(r, c):
            p = board.piece_at(r, c)
            if not screen_found:
                if p is not None:
                    screen_found = True
            else:
                if p is not None:
                    if (p.color is by and p.ptype is PieceType.CANNON
                            and target_piece is not None):
                        return True
                    break
            r += dr
            c += dc

    return False


def kings_face_each_other(board: Board) -> bool:
    """True if the two Kings share a file with no pieces between them.

    This is the "flying generals" (飞将) rule: such a position is illegal, so
    any move that creates it is illegal. Used by :func:`legal_moves`.
    """
    rk = board.find_king(Color.RED)
    bk = board.find_king(Color.BLACK)
    if rk is None or bk is None:
        return False
    if rk[1] != bk[1]:  # different files
        return False
    col = rk[1]
    lo, hi = sorted((rk[0], bk[0]))
    for r in range(lo + 1, hi):
        if board.piece_at(r, col) is not None:
            return False
    return True


def in_check(board: Board, color: Color) -> bool:
    """Whether ``color``'s King is currently attacked by the opponent."""
    king_sq = board.find_king(color)
    if king_sq is None:
        return False  # king already captured; rules.py handles terminality
    if square_attacked_by(board, king_sq[0], king_sq[1], color.opponent):
        return True
    # Flying generals also counts as the king being "exposed".
    return kings_face_each_other(board)


# --------------------------------------------------------------------------- #
# Legal move generation
# --------------------------------------------------------------------------- #
def legal_moves(board: Board, color: Color | None = None) -> List[Move]:
    """All fully legal moves for ``color`` (defaults to the side to move).

    A move is legal if it is pseudo-legal and, after making it, the mover's own
    King is not in check and the two kings are not facing each other.
    Implemented by make/undo so no board copying is needed.
    """
    if color is None:
        color = board.side_to_move
    result: List[Move] = []
    for move in pseudo_legal_moves(board, color):
        board.make_move(move)
        # After the move it's the opponent's turn; we check whether *our* king
        # is left in check. make_move flips side_to_move, so we test `color`.
        legal = not in_check(board, color) and not kings_face_each_other(board)
        board.undo_move()
        if legal:
            result.append(move)
    return result


def is_legal(board: Board, move: Move) -> bool:
    """Fast single-move legality test via make/undo."""
    color = board.side_to_move
    # Reject immediately if the move isn't even pseudo-legal.
    if move not in pseudo_legal_moves(board, color):
        return False
    board.make_move(move)
    ok = not in_check(board, color) and not kings_face_each_other(board)
    board.undo_move()
    return ok
