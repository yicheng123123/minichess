"""Self-play module for Mini Xiangqi AlphaZero training.

Provides game generation, data augmentation, and model evaluation
for the self-play training loop.

Key functions:
    play_game: Play a single self-play game and collect training samples.
    generate_games: Play multiple games and persist to a dataset.
    evaluate_match: Compare two networks in a head-to-head match.
    should_accept_new_model: Elo-like gate for model acceptance.
"""

from selfplay.player import play_game, generate_games
from selfplay.arena import evaluate_match, should_accept_new_model

__all__ = [
    "play_game",
    "generate_games",
    "evaluate_match",
    "should_accept_new_model",
]
