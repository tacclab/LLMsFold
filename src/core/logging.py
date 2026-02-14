"""Centralized logging setup and logger factory."""

import logging

from src.core.constants import DEFAULT_LOG_FORMAT, DEFAULT_LOG_LEVEL


def _resolve_level(level_name: str) -> int:
    """Converts a level string into a stdlib logging level."""

    resolved = getattr(logging, level_name.upper(), None)
    return resolved if isinstance(resolved, int) else logging.INFO


def configure_logging(level: str = DEFAULT_LOG_LEVEL, fmt: str = DEFAULT_LOG_FORMAT) -> None:
    """Configures root logging once and keeps level synchronized."""

    root = logging.getLogger()
    resolved_level = _resolve_level(level)

    if not root.handlers:
        logging.basicConfig(level=resolved_level, format=fmt)
        return

    root.setLevel(resolved_level)


def get_logger(name: str) -> logging.Logger:
    """Returns a module-scoped logger."""

    return logging.getLogger(name)
