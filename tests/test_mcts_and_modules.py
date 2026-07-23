"""tests/test_mcts_and_modules.py — Tests for MCTS, selfplay, and utils.

Run with: ``python -m unittest tests.test_mcts_and_modules``.
"""

from __future__ import annotations

import unittest

from engine.board import Board
from engine.move import Move
from engine.game import Game
from engine.piece import Color


class TestMCTS(unittest.TestCase):
    """Basic smoke tests for the MCTS module."""

    def test_import(self) -> None:
        from search.mcts import MCTS, MCTSNode
        self.assertIsNotNone(MCTS)
        self.assertIsNotNone(MCTSNode)

    def test_search_returns_legal_move(self) -> None:
        from search.mcts import MCTS
        from nn.network import RandomPolicyValueNet
        from engine.move_generator import legal_moves

        board = Board()
        net = RandomPolicyValueNet(seed=42)
        mcts = MCTS(num_simulations=20)  # small for speed
        probs, best = mcts.search(board, net)

        # best move must be legal
        legal = legal_moves(board)
        self.assertIn(best, legal)

        # probs should be a dict of Move -> float summing to ~1
        self.assertGreater(len(probs), 0)
        total = sum(probs.values())
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_search_with_temperature(self) -> None:
        from search.mcts import MCTS
        from nn.network import RandomPolicyValueNet

        board = Board()
        net = RandomPolicyValueNet(seed=0)
        mcts = MCTS(num_simulations=20)
        probs, move = mcts.search_with_temperature(board, net, temperature=1.0)
        self.assertIsInstance(move, Move)


class TestSelfplay(unittest.TestCase):
    """Smoke tests for the selfplay module."""

    def test_import(self) -> None:
        from selfplay.player import play_game, generate_games
        self.assertIsNotNone(play_game)

    def test_play_game_random(self) -> None:
        from selfplay.player import play_game
        from nn.network import RandomPolicyValueNet

        net = RandomPolicyValueNet(seed=7)
        result = play_game(net=net, max_plies=40, seed=7)
        self.assertIn("samples", result)
        self.assertIn("outcome", result)
        self.assertIn(result["outcome"], (-1, 0, 1))
        self.assertGreater(result["plies"], 0)


class TestUtils(unittest.TestCase):
    """Tests for the utils module."""

    def test_config(self) -> None:
        from utils.config import Config, get_config
        cfg = get_config()
        self.assertEqual(cfg.board_size, 7)
        self.assertEqual(cfg.num_simulations, 400)

    def test_timer(self) -> None:
        from utils.timer import Timer
        with Timer("test") as t:
            _ = sum(range(1000))
        self.assertGreaterEqual(t.elapsed, 0.0)

    def test_seed(self) -> None:
        import random
        from utils.seed import set_seed
        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        self.assertEqual(a, b)


class TestGameIntegration(unittest.TestCase):
    """Integration tests for the Game controller with search."""

    def test_alphabeta_finds_move(self) -> None:
        from search.alphabeta import alphabeta
        board = Board()
        score, move = alphabeta(board, depth=2)
        self.assertIsNotNone(move)
        self.assertIsInstance(move, Move)

    def test_game_with_ai_moves(self) -> None:
        from search.alphabeta import best_move
        game = Game()
        for _ in range(6):
            if game.is_over:
                break
            mv = best_move(game.board, depth=2)
            if mv is None:
                break
            game.play(mv)
        self.assertGreater(game.ply, 0)


if __name__ == "__main__":
    unittest.main()
