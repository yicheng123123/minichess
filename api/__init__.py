"""api — HTTP API module for Mini Xiangqi.

Exposes the engine, search, and neural-network components through a FastAPI
server so that external clients (web GUIs, bots, test harnesses) can interact
with the game over HTTP.

Quick start::

    python -m api.server --port 8000

Submodules:
    schemas  — Pydantic request/response models.
    server   — FastAPI application factory and endpoint definitions.
"""
