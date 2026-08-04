import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=None)
def get_logger(name: str = "rag_project", level: Optional[int] = None) -> logging.Logger:
    """Create or retrieve a configured logger for the project."""
    logger = logging.getLogger(name)
    logger.setLevel(level or logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def log_step(message: str, logger: Optional[logging.Logger] = None) -> None:
    """Log a workflow step message."""
    target_logger = logger or get_logger()
    target_logger.info(message)
