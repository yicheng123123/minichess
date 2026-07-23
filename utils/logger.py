"""
Mini Xiangqi - Logging Utility
===============================

Provides a pre-configured logger and a factory function for creating
additional loggers with console and optional file output.

Usage:
    from utils.logger import logger, setup_logger

    logger.info("Game started")

    # Create a custom logger with file output
    train_log = setup_logger("training", level="DEBUG", log_file="train.log")
    train_log.debug("Epoch 1 complete")
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

# Default format: [time] [level] [module] message
_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create (or retrieve) a logger with console and optional file handler.

    If the logger already has handlers, it is returned as-is to avoid
    duplicate output on repeated calls.

    Args:
        name: Logger name (e.g. "mini_xiangqi", "training").
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to a log file. Parent directories are created.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    _logger = logging.getLogger(name)
    _logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid adding duplicate handlers on repeated calls
    if _logger.handlers:
        return _logger

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoids double output)
    _logger.propagate = False

    return _logger


# Pre-configured project-wide logger
logger: logging.Logger = setup_logger("mini_xiangqi", level="INFO")
