"""Logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from utils.path_utils import ensure_dir



def get_logger(name: str, debug: bool = False, log_path: Path | None = None) -> logging.Logger:
    """Configure and return a logger."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_path is not None:
        ensure_dir(log_path.parent)
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
