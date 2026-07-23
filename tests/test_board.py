"""tests/test_board.py — Unit tests for engine (board, piece, move, game).

Run with: ``python -m unittest tests.test_board`` or ``python -m pytest``.

Covers: starting position, piece parsing, coordinates, make/undo invariance,
FEN round-trip, palace bounds, king lookup, repetition counting, the NN
plane shape (when numpy is available), and the Game controller.
"""

from __future__ import annotations

import unittest

from engine.board import BOARD_SIZE, Board
from engine.move import Move, alg_to_square, square_to_alg
from engine.piece import Color, Piece, PieceType
from engine.game import Game


class TestPiece(unittest.TestCase):
    def test_char_roundtrip(self) -> None:
        for ch in "RrCcNnPpKk":
            self.assertEqual(Piece.from_char(ch).char, ch)

    def test_color_case(self) -> None:
        self.assertIs(Piece.from_char("K").color, Color.RED)
        self.assertIs(Piece.from_char("k").color, Color.BLACK)

    def test_bad_char_raises(self) -> None:
        with self.assertRaises(ValueError):
            Piece.from_char("X")
        with self.assertRaises(ValueError):
            Piece.from_char("")


class TestCoordinates(unittest.TestCase):
    def test_alg_roundtrip(self) -> None:
        for alg in ["a1", "d4", "g7", "a7", "g1"]:
            r, c = alg_to_square(alg)
            self.assertEqual(square_to_alg(r, c), alg)

    def test_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            square_to_alg(7, 0)
        with self.assertRaises(ValueError):
            alg_to_square("h1")


class TestMove(unittest.TestCase):
    def test_uci_roundtrip(self) -> None:
        for uci in ["a1a4", "d2d3", "g7g1"]:
            m = Move.from_uci(uci)
            self.assertEqual(m.uci(), uci)

    def test_equality_and_hash(self) -> None:
        m1 = Move.from_uci("a1a4")
        m2 = Move((0, 0), (3, 0))
        self.assertEqual(m1, m2)
        self.assertEqual(hash(m1), hash(m2))

    def test_invalid_uci(self) -> None:
        with self.assertRaises(ValueError):
            Move.from_uci("abc")
        with self.assertRaises(ValueError):
            Move.from_uci("a1a8")


class TestStartingPosition(unittest.TestCase):
    def setUp(self) -> None:
        self.b = Board()

    def test_dimensions(self) -> None:
        self.assertEqual(BOARD_SIZE, 7)

    def test_back_ranks(self) -> None:
        expected = [PieceType.ROOK, PieceType.CANNON, PieceType.HORSE,
                    PieceType.KING, PieceType.HORSE, PieceType.CANNON,
                    PieceType.ROOK]
        for col, ptype in enumerate(expected):
            p = self.b.piece_at(0, col)
            self.assertIsNotNone(p)
            self.assertIs(p.ptype, ptype)
            self.assertIs(p.color, Color.RED)
        for col, ptype in enumerate(expected):
            p = self.b.piece_at(6, col)
            self.assertIsNotNone(p)
            self.assertIs(p.ptype, ptype)
            self.assertIs(p.color, Color.BLACK)

    def test_soldiers(self) -> None:
        # Soldiers on alternating files a, c, d, e, g (cols 0, 2, 3, 4, 6).
        for col in (0, 2, 3, 4, 6):
            self.assertIs(self.b.piece_at(1, col).ptype, PieceType.SOLDIER)
            self.assertIs(self.b.piece_at(1, col).color, Color.RED)
            self.assertIs(self.b.piece_at(5, col).ptype, PieceType.SOLDIER)
            self.assertIs(self.b.piece_at(5, col).color, Color.BLACK)
        # Files b and f (cols 1, 5) have no soldiers.
        for col in (1, 5):
            self.assertIsNone(self.b.piece_at(1, col))
            self.assertIsNone(self.b.piece_at(5, col))

    def test_empty_middle(self) -> None:
        for row in (2, 3, 4):
            for col in range(BOARD_SIZE):
                self.assertIsNone(self.b.piece_at(row, col))

    def test_red_to_move_first(self) -> None:
        self.assertIs(self.b.side_to_move, Color.RED)
        self.assertEqual(self.b.fullmove_number, 1)


class TestPalace(unittest.TestCase):
    def test_red_palace_bounds(self) -> None:
        b = Board()
        self.assertTrue(b.is_in_palace(0, 3, Color.RED))
        self.assertTrue(b.is_in_palace(2, 4, Color.RED))
        self.assertFalse(b.is_in_palace(3, 3, Color.RED))
        self.assertFalse(b.is_in_palace(0, 1, Color.RED))

    def test_black_palace_bounds(self) -> None:
        b = Board()
        self.assertTrue(b.is_in_palace(6, 3, Color.BLACK))
        self.assertTrue(b.is_in_palace(4, 2, Color.BLACK))
        self.assertFalse(b.is_in_palace(3, 3, Color.BLACK))


class TestKingLookup(unittest.TestCase):
    def test_find_kings(self) -> None:
        b = Board()
        self.assertEqual(b.find_king(Color.RED), (0, 3))
        self.assertEqual(b.find_king(Color.BLACK), (6, 3))

    def test_missing_king(self) -> None:
        b = Board()
        fen = b.to_fen().replace("K", "1", 1)
        b2 = Board.from_fen(fen)
        self.assertTrue(b2.king_is_missing(Color.RED))
        self.assertFalse(b2.king_is_missing(Color.BLACK))
        self.assertTrue(b2.is_terminal())
        self.assertIs(b2.winner(), Color.BLACK)


class TestMakeUndo(unittest.TestCase):
    def test_undo_restores_exactly(self) -> None:
        b = Board()
        before = b.to_fen()
        key_before = b.position_key()
        moves = [Move.from_uci("d2d3"), Move.from_uci("d6d5"), Move.from_uci("a1a2")]
        for m in moves:
            b.make_move(m)
        for _ in moves:
            b.undo_move()
        self.assertEqual(b.to_fen(), before)
        self.assertEqual(b.position_key(), key_before)
        self.assertIs(b.side_to_move, Color.RED)

    def test_fullmove_increments_after_black(self) -> None:
        b = Board()
        self.assertEqual(b.fullmove_number, 1)
        b.make_move(Move.from_uci("d2d3"))
        self.assertEqual(b.fullmove_number, 1)
        b.make_move(Move.from_uci("d6d5"))
        self.assertEqual(b.fullmove_number, 2)

    def test_undo_with_no_history_returns_none(self) -> None:
        self.assertIsNone(Board().undo_move())

    def test_make_move_on_empty_raises(self) -> None:
        b = Board()
        with self.assertRaises(ValueError):
            b.make_move(Move((3, 3), (3, 4)))

    def test_halfmove_clock(self) -> None:
        b = Board()
        self.assertEqual(b.halfmove_clock, 0)
        b.make_move(Move.from_uci("d2d3"))
        self.assertEqual(b.halfmove_clock, 0)
        b.make_move(Move.from_uci("d6d5"))
        self.assertEqual(b.halfmove_clock, 0)
        # Cannon move (non-capture, non-soldier) increments the clock.
        b.make_move(Move.from_uci("b1b2"))
        self.assertEqual(b.halfmove_clock, 1)


class TestFenRoundTrip(unittest.TestCase):
    def test_roundtrip(self) -> None:
        b = Board()
        self.assertEqual(Board.from_fen(b.to_fen()).to_fen(), b.to_fen())
        b.make_move(Move.from_uci("d2d3"))
        b.make_move(Move.from_uci("d6d5"))
        self.assertEqual(Board.from_fen(b.to_fen()).to_fen(), b.to_fen())

    def test_start_fen(self) -> None:
        self.assertEqual(
            Board().to_fen(),
            "rcnkncr/p1ppp1p/7/7/7/P1PPP1P/RCNKNCR r 0 1",
        )


class TestRepetition(unittest.TestCase):
    def test_repetition_after_repeating_line(self) -> None:
        b = Board()
        self.assertEqual(b.repetition_count(), 1)
        # Shuffle cannons on the b-file back and forth to recreate the start.
        for _ in range(2):
            b.make_move(Move.from_uci("b1b2"))
            b.make_move(Move.from_uci("b7b6"))
            b.make_move(Move.from_uci("b2b1"))
            b.make_move(Move.from_uci("b6b7"))
        # The starting position has now occurred 3 times.
        self.assertGreaterEqual(b.repetition_count(), 3)


class TestClone(unittest.TestCase):
    def test_clone_is_independent(self) -> None:
        b = Board()
        c = b.clone()
        b.make_move(Move.from_uci("d2d3"))
        self.assertEqual(c.position_key(), Board().position_key())
        self.assertIs(c.side_to_move, Color.RED)


class TestPlanes(unittest.TestCase):
    def test_plane_shape_or_skip(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        planes = Board().to_planes()
        self.assertEqual(planes.shape, (11, BOARD_SIZE, BOARD_SIZE))
        self.assertEqual(int(planes[:10].sum()), 24)


class TestGame(unittest.TestCase):
    """Tests for the Game controller."""

    def test_new_game_not_over(self) -> None:
        g = Game()
        self.assertFalse(g.is_over)
        self.assertIsNone(g.result)
        self.assertEqual(g.ply, 0)

    def test_play_and_history(self) -> None:
        g = Game()
        g.play("d2d3")
        self.assertEqual(g.move_history, ["d2d3"])
        self.assertEqual(g.ply, 1)
        self.assertIs(g.side_to_move, Color.BLACK)

    def test_play_string_and_move_object(self) -> None:
        g = Game()
        g.play("d2d3")
        g.play(Move.from_uci("d6d5"))
        self.assertEqual(g.move_history, ["d2d3", "d6d5"])

    def test_illegal_move_raises(self) -> None:
        g = Game()
        with self.assertRaises(ValueError):
            g.play("d2d5")  # not a legal move

    def test_undo(self) -> None:
        g = Game()
        g.play("d2d3")
        g.play("d6d5")
        undone = g.undo()
        self.assertEqual(undone.uci(), "d6d5")
        self.assertEqual(g.move_history, ["d2d3"])
        self.assertIs(g.side_to_move, Color.BLACK)

    def test_undo_empty(self) -> None:
        g = Game()
        self.assertIsNone(g.undo())

    def test_legal_moves_available(self) -> None:
        g = Game()
        moves = g.legal_moves()
        self.assertEqual(len(moves), 19)

    def test_from_moves(self) -> None:
        g = Game.from_moves(["d2d3", "d6d5"])
        self.assertEqual(g.ply, 2)
        self.assertIs(g.side_to_move, Color.RED)

    def test_from_fen(self) -> None:
        fen = "rcnkncr/1ppppp1/7/7/7/1PPPPP1/RCNKNCR b 0 1"
        g = Game.from_fen(fen)
        self.assertIs(g.side_to_move, Color.BLACK)

    def test_in_check(self) -> None:
        # Position where black king is in check from a rook
        g = Game.from_fen("3k3/7/7/7/7/3R3/4K2 b 0 1")
        self.assertTrue(g.in_check())


if __name__ == "__main__":
    unittest.main()
