"""tests/test_move_generator.py — Tests for engine.move_generator and engine.rules.

Run with: ``python -m unittest tests.test_move_generator``.

Covers: pseudo-legal vs legal move counts, the "flying generals" rule, check
detection on constructed positions, and checkmate / repetition classification
from engine.rules.
"""

from __future__ import annotations

import unittest

from engine.board import Board, BOARD_SIZE
from engine.move import Move
from engine.move_generator import (
    in_check,
    kings_face_each_other,
    legal_moves,
    piece_moves,
    pseudo_legal_moves,
    square_attacked_by,
)
from engine.piece import Color, PieceType
from engine.rules import (
    GameOutcome,
    is_checkmate,
    is_game_over,
    game_result,
)


class TestStartingPositionMoves(unittest.TestCase):
    def test_pseudo_equals_legal_at_start(self) -> None:
        for color in (Color.RED, Color.BLACK):
            b = Board()
            b._side = color
            self.assertEqual(
                len(pseudo_legal_moves(b, color)),
                len(legal_moves(b, color)),
            )

    def test_red_starting_move_count(self) -> None:
        b = Board()
        moves = legal_moves(b, Color.RED)
        # Soldiers can move forward AND sideways from the start.
        # Rooks blocked by own soldiers on a2/g2, cannons have no captures.
        self.assertEqual(len(moves), 19)

    def test_no_captures_at_start(self) -> None:
        # The new soldier layout blocks the rooks and removes cannon screens,
        # so Red has zero captures on move 1 (unlike the old open-file layout).
        b = Board()
        uci_captures = {m.uci() for m in legal_moves(b) if b.piece_at(*m.to_sq)}
        self.assertEqual(uci_captures, set())


def _fen(fen: str) -> Board:
    """Wrap Board.from_fen with a rank-width check so FEN typos fail loudly."""
    for rank in fen.split()[0].split("/"):
        width = sum(int(c) if c.isdigit() else 1 for c in rank)
        assert width == BOARD_SIZE, f"rank {rank!r} has width {width}, want {BOARD_SIZE}"
    return Board.from_fen(fen)


class TestFlyingGenerals(unittest.TestCase):
    def test_empty_file_kings_face(self) -> None:
        b = _fen("3k3/7/7/7/7/7/3K3 r 0 1")
        self.assertTrue(kings_face_each_other(b))

    def test_blocked_file_kings_dont_face(self) -> None:
        b = _fen("3k3/7/7/3p3/7/7/3K3 r 0 1")
        self.assertFalse(kings_face_each_other(b))

    def test_king_cannot_move_to_create_flying_generals(self) -> None:
        b = _fen("3k3/7/7/7/7/7/3K3 r 0 1")
        moves = {m.uci() for m in legal_moves(b)}
        self.assertNotIn("d1d2", moves)


class TestCheckDetection(unittest.TestCase):
    def test_rook_gives_check(self) -> None:
        b = _fen("3k3/7/7/7/7/3R3/4K2 b 0 1")
        self.assertTrue(in_check(b, Color.BLACK))

    def test_no_check_when_blocked(self) -> None:
        b = _fen("3k3/7/7/3p3/7/3R3/4K2 b 0 1")
        self.assertFalse(in_check(b, Color.BLACK))

    def test_horse_check_with_leg_free(self) -> None:
        b = _fen("3k3/7/2N4/7/7/7/4K2 b 0 1")
        self.assertIs(b.piece_at(4, 2).ptype, PieceType.HORSE)
        self.assertTrue(in_check(b, Color.BLACK))

    def test_horse_check_blocked_by_leg(self) -> None:
        b = _fen("3k3/2p4/2N4/7/7/7/4K2 b 0 1")
        self.assertIsNotNone(b.piece_at(5, 2))
        self.assertFalse(in_check(b, Color.BLACK))


class TestCheckmate(unittest.TestCase):
    def test_back_rank_mate(self) -> None:
        b = _fen("2pkp2/7/7/7/7/7/3RK2 b 0 1")
        self.assertIs(b.piece_at(6, 3).ptype, PieceType.KING)
        self.assertIsNotNone(b.piece_at(6, 2))
        self.assertIsNotNone(b.piece_at(6, 4))
        self.assertIs(b.piece_at(0, 3).ptype, PieceType.ROOK)
        self.assertIs(b.piece_at(0, 4).ptype, PieceType.KING)
        self.assertTrue(in_check(b, Color.BLACK))
        self.assertTrue(is_checkmate(b))


class TestSquareAttacked(unittest.TestCase):
    def test_rook_attacks_along_file(self) -> None:
        b = _fen("3k3/7/7/7/7/3R3/4K2 r 0 1")
        self.assertTrue(square_attacked_by(b, 6, 3, Color.RED))

    def test_cannon_needs_screen(self) -> None:
        b = _fen("3k3/7/3p3/7/3C3/7/4K2 r 0 1")
        self.assertIs(b.piece_at(2, 3).ptype, PieceType.CANNON)
        self.assertTrue(square_attacked_by(b, 6, 3, Color.RED))

        b2 = _fen("3k3/7/7/7/3C3/7/4K2 r 0 1")
        self.assertFalse(square_attacked_by(b2, 6, 3, Color.RED))


class TestRulesResult(unittest.TestCase):
    def test_non_terminal_at_start(self) -> None:
        self.assertFalse(is_game_over(Board()))
        self.assertIsNone(game_result(Board()))

    def test_king_capture_is_terminal(self) -> None:
        b = _fen("3k3/7/7/7/7/7/7 r 0 1")
        result = game_result(b)
        self.assertIsNotNone(result)
        self.assertIs(result.winner, Color.BLACK)
        self.assertEqual(result.outcome, GameOutcome.BLACK_WINS)


if __name__ == "__main__":
    unittest.main()
