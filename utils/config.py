"""
Mini Xiangqi - Configuration Management
========================================

Provides a dataclass-based configuration with sensible defaults for the
Mini Xiangqi engine (MCTS search, neural network training, etc.).

Usage:
    from utils.config import get_config, set_config, load_config, save_config

    cfg = get_config()
    print(cfg.num_simulations)  # 400

    cfg.num_simulations = 800
    set_config(cfg)

    save_config(cfg, "my_config.json")
    cfg2 = load_config("my_config.json")
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional


@dataclass
class Config:
    """Central configuration for the Mini Xiangqi project.

    Attributes:
        board_size: Board dimension (7x7 for Mini Xiangqi).
        num_simulations: Number of MCTS simulations per move.
        c_puct: PUCT exploration constant for MCTS.
        dirichlet_alpha: Concentration of the root Dirichlet exploration noise
            (0.15 suits a chess-like game with ~20-40 legal moves).
        temperature: Move selection temperature (1.0 = proportional to visit count).
        max_plies: Maximum game length before declaring a draw.
        learning_rate: Initial learning rate for training.
        batch_size: Training mini-batch size.
        hidden_channels: Number of hidden channels in the residual network.
        num_res_blocks: Number of residual blocks in the network.
        checkpoint_dir: Directory for saving model checkpoints.
        data_dir: Directory for training data / self-play games.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """

    board_size: int = 7
    num_simulations: int = 400
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.15
    temperature: float = 1.0
    max_plies: int = 200
    learning_rate: float = 1e-3
    batch_size: int = 64
    hidden_channels: int = 128
    num_res_blocks: int = 4
    checkpoint_dir: str = "models"
    data_dir: str = "data"
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create a Config from a dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def update(self, **kwargs: Any) -> "Config":
        """Update fields in-place and return self for chaining."""
        valid_keys = {f.name for f in fields(self)}
        for key, value in kwargs.items():
            if key not in valid_keys:
                raise KeyError(f"Unknown config key: {key!r}")
            setattr(self, key, value)
        return self


# ----------------------------------------------------------------------
# JSON persistence
# ----------------------------------------------------------------------


def save_config(config: Config, path: str) -> None:
    """Save a Config instance to a JSON file.

    Args:
        config: The configuration to persist.
        path: Destination file path (parent directories are created).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)


def load_config(path: str) -> Config:
    """Load a Config from a JSON file.

    Missing keys fall back to defaults; unknown keys are ignored.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        A Config instance populated from the file.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Config.from_dict(data)


# ----------------------------------------------------------------------
# Global singleton pattern
# ----------------------------------------------------------------------

_global_config: Optional[Config] = None


def get_config() -> Config:
    """Return the global Config, creating one with defaults if needed."""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_config(config: Config) -> None:
    """Replace the global Config instance.

    Args:
        config: The new configuration to use globally.
    """
    global _global_config
    _global_config = config
