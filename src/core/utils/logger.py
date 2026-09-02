import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str = __name__,
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Creates and configures a standardized logger.

    Args:
        name: Name of the logger (typically __name__).
        log_file: Optional path to save log outputs to a file.
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if get_logger is called repeatedly
    if logger.handlers:
        return logger

    # Log format: 2026-09-02 23:45:00 | INFO | homography | Video loaded successfully
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (Optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Default application-wide logger
logger = get_logger("SportsEngine")

