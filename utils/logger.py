from __future__ import annotations

import logging
from typing import Optional


_CONFIGURED = False


def _configure_root(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    _configure_root(level=logging.INFO if level is None else level)
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger
