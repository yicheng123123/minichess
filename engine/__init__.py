"""engine package — Rules engine for Mini Xiangqi (迷你象棋).

Public API re-exported here for convenient access::

    from engine import Board, Move, Game, Color, PieceType
    from engine import legal_moves, in_check, game_result
"""

from .piece import Color, Piece, PieceType
from .move import Move, Square, in_bounds, square_to_alg, alg_to_square
from .board import Board, BOARD_SIZE, NUM_SQUARES, PALACE_RED, PALACE_BLACK
from .move_generator import (
    legal_moves,
    pseudo_legal_moves,
    piece_moves,
    in_check,
    kings_face_each_other,
    square_attacked_by,
)
from .rules import (
    GameOutcome,
    GameResult,
    game_result,
    is_game_over,
    is_checkmate,
    is_stalemate,
    is_repetition_draw,
    DEFAULT_REPETITION_THRESHOLD,
)
from .game import Game

__all__ = [
    # piece
    "Color", "Piece", "PieceType",
    # move
    "Move", "Square", "in_bounds", "square_to_alg", "alg_to_square",
    # board
    "Board", "BOARD_SIZE", "NUM_SQUARES", "PALACE_RED", "PALACE_BLACK",
    # move_generator
    "legal_moves", "pseudo_legal_moves", "piece_moves",
    "in_check", "kings_face_each_other", "square_attacked_by",
    # rules
    "GameOutcome", "GameResult", "game_result", "is_game_over",
    "is_checkmate", "is_stalemate", "is_repetition_draw",
    "DEFAULT_REPETITION_THRESHOLD",
    # game
    "Game",
]
