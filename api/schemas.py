"""api/schemas.py — Pydantic models for the Mini Xiangqi HTTP API.

These models define the request and response shapes for every endpoint in
:mod:`api.server`. They are intentionally decoupled from the engine internals
so that the public API contract can evolve independently.

If Pydantic is not installed, importing this module raises a clear error
directing the user to install the dependency.
"""

from __future__ import annotations

try:
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'pydantic' package is required for the API module.\n"
        "Install it with:  pip install pydantic\n"
        "(or:  pip install fastapi  — which bundles pydantic)"
    ) from exc

from typing import List, Optional


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    """Response for the health-check endpoint."""

    status: str = "ok"


class NewGameResponse(BaseModel):
    """Response returned when a new game is created."""

    game_id: str = Field(..., description="Unique identifier for the game session.")
    fen: str = Field(..., description="FEN of the initial position.")
    side_to_move: str = Field(..., description="Side to move ('r' for Red, 'b' for Black).")


class GameState(BaseModel):
    """Full snapshot of a game's current state."""

    fen: str = Field(..., description="Current board FEN.")
    side_to_move: str = Field(..., description="Side to move ('r' or 'b').")
    is_over: bool = Field(..., description="Whether the game has ended.")
    result: Optional[str] = Field(
        None,
        description=(
            "Human-readable result string if the game is over, "
            "e.g. 'red_wins by checkmate'. None if still in progress."
        ),
    )
    legal_moves: List[str] = Field(
        default_factory=list,
        description="UCI strings of all legal moves for the side to move.",
    )
    move_history: List[str] = Field(
        default_factory=list,
        description="UCI strings of all moves played so far.",
    )
    in_check: bool = Field(
        ..., description="Whether the side to move is currently in check."
    )


class AIMoveResponse(BaseModel):
    """Response from the AI move endpoint."""

    move: str = Field(..., description="Best move in UCI notation, e.g. 'd2d3'.")
    score: Optional[float] = Field(
        None,
        description=(
            "Evaluation score from the engine's perspective. "
            "May be None for MCTS (which returns visit-based selection)."
        ),
    )


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #


class MoveRequest(BaseModel):
    """Request body for playing a move in an existing game."""

    move: str = Field(
        ...,
        description="UCI move string, e.g. 'd2d3' (from-square + to-square).",
        examples=["d2d3", "a1a4"],
    )


class AIMoveRequest(BaseModel):
    """Request body for asking the AI to suggest a move."""

    fen: str = Field(..., description="Board position in FEN notation.")
    engine: str = Field(
        default="alphabeta",
        description="Search engine to use: 'alphabeta' or 'mcts'.",
    )
    depth: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Search depth for alpha-beta (ignored for MCTS).",
    )
    simulations: int = Field(
        default=400,
        ge=1,
        le=100000,
        description="Number of MCTS simulations (ignored for alpha-beta).",
    )
