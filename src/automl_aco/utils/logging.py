"""Logging helpers."""
from __future__ import annotations

import logging
from typing import Optional


_DEF_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=_DEF_FORMAT)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger
