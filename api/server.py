"""api/server.py — FastAPI server for Mini Xiangqi.

Provides HTTP endpoints for creating/playing games, querying board state, and
requesting AI moves (alpha-beta or MCTS). Games are stored in an in-memory
dictionary keyed by UUID.

Run directly::

    python -m api.server --port 8000
    python -m api.server --host 0.0.0.0 --port 9000

Or import the app factory::

    from api.server import create_app
    app = create_app()
"""

from __future__ import annotations

import argparse
import uuid
from typing import Dict, Optional

# --------------------------------------------------------------------------- #
# Guarded imports — give a clear error if FastAPI / uvicorn are missing.
# --------------------------------------------------------------------------- #
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'fastapi' package is required to run the API server.\n"
        "Install it with:  pip install fastapi uvicorn"
    ) from exc

try:
    import uvicorn  # noqa: F401 — used in run()
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'uvicorn' package is required to run the API server.\n"
        "Install it with:  pip install uvicorn"
    ) from exc

# --------------------------------------------------------------------------- #
# Engine / search imports (always available in this project).
# --------------------------------------------------------------------------- #
from engine.board import Board
from engine.game import Game
from engine.move import Move
from engine.piece import Color

from .schemas import (
    AIMoveRequest,
    AIMoveResponse,
    GameState,
    HealthResponse,
    MoveRequest,
    NewGameResponse,
)

# --------------------------------------------------------------------------- #
# In-memory game store
# --------------------------------------------------------------------------- #
_games: Dict[str, Game] = {}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _game_or_404(game_id: str) -> Game:
    """Retrieve a game by ID or raise a 404 HTTPException."""
    game = _games.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Game '{game_id}' not found.")
    return game


def _result_string(game: Game) -> Optional[str]:
    """Build a human-readable result string, or None if the game is ongoing."""
    if game.result is None:
        return None
    outcome = game.result.outcome.value  # e.g. "red_wins"
    reason = game.result.reason  # e.g. "checkmate"
    return f"{outcome} by {reason}"


def _game_state(game: Game) -> GameState:
    """Serialize a Game into a GameState response model."""
    return GameState(
        fen=game.board.to_fen(),
        side_to_move=game.side_to_move.value,
        is_over=game.is_over,
        result=_result_string(game),
        legal_moves=[m.uci() for m in game.legal_moves()],
        move_history=game.move_history,
        in_check=game.in_check(),
    )


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns a fully wired app instance with CORS middleware and all endpoints
    registered. Callers can further customize the app before serving.
    """
    app = FastAPI(
        title="Mini Xiangqi API",
        description=(
            "HTTP API for the Mini Xiangqi (7x7 Chinese chess) engine. "
            "Create games, play moves, undo, and request AI suggestions."
        ),
        version="1.0.0",
    )

    # -- CORS: allow browser-based clients (dev convenience) ---------------- #
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --------------------------------------------------------------------- #
    # Endpoints
    # --------------------------------------------------------------------- #

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        """Health-check endpoint. Returns 200 with status 'ok'."""
        return HealthResponse(status="ok")

    @app.post("/game/new", response_model=NewGameResponse, tags=["game"])
    def new_game() -> NewGameResponse:
        """Create a new game from the standard starting position.

        Returns the game ID, initial FEN, and side to move.
        """
        game = Game()
        game_id = str(uuid.uuid4())
        _games[game_id] = game
        return NewGameResponse(
            game_id=game_id,
            fen=game.board.to_fen(),
            side_to_move=game.side_to_move.value,
        )

    @app.get("/game/{game_id}", response_model=GameState, tags=["game"])
    def get_game(game_id: str) -> GameState:
        """Retrieve the full state of an existing game."""
        game = _game_or_404(game_id)
        return _game_state(game)

    @app.post("/game/{game_id}/move", response_model=GameState, tags=["game"])
    def play_move(game_id: str, body: MoveRequest) -> GameState:
        """Play a move in an existing game.

        The move must be a legal UCI string (e.g. 'd2d3'). Returns the
        updated game state after the move is applied.

        Raises 400 if the move is illegal or the game is already over.
        """
        game = _game_or_404(game_id)
        try:
            game.play(body.move)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _game_state(game)

    @app.post("/game/{game_id}/undo", response_model=GameState, tags=["game"])
    def undo_move(game_id: str) -> GameState:
        """Undo the last move in a game.

        If there are no moves to undo, the state is returned unchanged.
        """
        game = _game_or_404(game_id)
        game.undo()
        return _game_state(game)

    @app.post("/ai/move", response_model=AIMoveResponse, tags=["ai"])
    def ai_move(body: AIMoveRequest) -> AIMoveResponse:
        """Ask the AI engine for the best move in a given position.

        Supports two engines:
          - **alphabeta**: fixed-depth alpha-beta search. Returns a score.
          - **mcts**: Monte Carlo Tree Search guided by a neural network.
            Returns the selected move (score is None).

        Raises 400 for an invalid FEN or unknown engine name.
        """
        # Parse the board from FEN.
        try:
            board = Board.from_fen(body.fen)
        except (ValueError, IndexError) as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid FEN: {exc}"
            ) from exc

        engine_name = body.engine.lower().strip()

        if engine_name == "alphabeta":
            from search.alphabeta import alphabeta

            score, best_move = alphabeta(board, body.depth)
            if best_move is None:
                raise HTTPException(
                    status_code=400,
                    detail="No legal moves available in the given position.",
                )
            return AIMoveResponse(move=best_move.uci(), score=score)

        elif engine_name == "mcts":
            from nn.network import default_net
            from search.mcts import MCTS

            net = default_net()
            searcher = MCTS(num_simulations=body.simulations)
            _move_probs, best_move = searcher.search(board, net)
            if best_move is None:
                raise HTTPException(
                    status_code=400,
                    detail="No legal moves available in the given position.",
                )
            return AIMoveResponse(move=best_move.uci(), score=None)

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown engine '{body.engine}'. Use 'alphabeta' or 'mcts'.",
            )

    @app.get("/board", tags=["board"])
    def board_view(fen: str = Query(..., description="FEN of the position to display.")) -> Dict[str, str]:
        """Return a text rendering of the board for a given FEN.

        The text representation matches the engine's ``str(board)`` output.
        """
        try:
            board = Board.from_fen(fen)
        except (ValueError, IndexError) as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid FEN: {exc}"
            ) from exc
        return {"board": str(board)}

    return app


# --------------------------------------------------------------------------- #
# Convenience runner
# --------------------------------------------------------------------------- #


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the API server using uvicorn.

    Parameters
    ----------
    host : str
        Bind address. Default ``127.0.0.1`` (localhost only).
    port : int
        TCP port. Default ``8000``.
    """
    import uvicorn as _uvicorn

    app = create_app()
    _uvicorn.run(app, host=host, port=port)


# --------------------------------------------------------------------------- #
# CLI entry-point:  python -m api.server [--host H] [--port P]
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mini Xiangqi API server",
        prog="python -m api.server",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port (default: 8000)",
    )
    args = parser.parse_args()
    run(host=args.host, port=args.port)
