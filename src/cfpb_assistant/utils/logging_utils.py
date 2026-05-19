"""
Logging utility — sets up a consistent logger for every module in the project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

from cfpb_assistant.utils.config import ensure_dir, get_project_root


def get_logger(name: str, log_to_file: bool = False) -> logging.Logger:
    """
    Return a configured logger.

    Args:
        name: logger name, typically __name__ of the calling module
        log_to_file: if True, also write to outputs/logs/<date>.log

    Returns:
        logging.Logger instance
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_to_file:
        log_dir = ensure_dir(get_project_root() / "outputs" / "logs")
        date_str = datetime.now().strftime("%Y%m%d")
        file_handler = logging.FileHandler(log_dir / f"{date_str}.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
