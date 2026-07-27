"""Centralized logging setup and logger factory."""

import logging
from pathlib import Path

from tqdm.auto import tqdm

from src.core.constants import (
    DEFAULT_LOG_DATE_FORMAT,
    DEFAULT_LOG_FILE_DATE_FORMAT,
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
)


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that keeps tqdm progress bars intact."""

    def emit(self, record: logging.LogRecord) -> None:
        """Writes formatted log records via `tqdm.write`."""

        try:
            tqdm.write(self.format(record))
        except Exception:  # noqa: BLE001
            self.handleError(record)


def _resolve_level(level_name: str) -> int:
    """Converts a level string into a stdlib logging level."""

    resolved = getattr(logging, level_name.upper(), None)
    return resolved if isinstance(resolved, int) else logging.INFO


def _configure_third_party_loggers(resolved_level: int) -> None:
    """Keeps verbose transport loggers from drowning application logs."""

    transport_level = resolved_level if resolved_level > logging.WARNING else logging.WARNING
    logging.getLogger("httpx").setLevel(transport_level)
    logging.getLogger("httpcore").setLevel(transport_level)


def _get_or_create_tqdm_handler(root: logging.Logger) -> TqdmLoggingHandler:
    """Returns the shared tqdm-aware root handler."""

    for handler in root.handlers:
        if isinstance(handler, TqdmLoggingHandler):
            return handler

    handler = TqdmLoggingHandler()
    root.addHandler(handler)
    return handler


def _get_or_create_file_handler(root: logging.Logger, log_file: Path) -> logging.FileHandler:
    """Returns the shared run-log file handler, recreating it if the path changed."""

    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            if Path(handler.baseFilename) == log_file:
                return handler
            root.removeHandler(handler)
            handler.close()

    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    root.addHandler(handler)
    return handler


def configure_logging(
    level: str = DEFAULT_LOG_LEVEL,
    fmt: str = DEFAULT_LOG_FORMAT,
    datefmt: str = DEFAULT_LOG_DATE_FORMAT,
    log_file: Path | str | None = None,
    file_fmt: str = DEFAULT_LOG_FORMAT,
    file_datefmt: str = DEFAULT_LOG_FILE_DATE_FORMAT,
) -> None:
    """Configures root logging once and keeps level synchronized.

    When `log_file` is given, every log record is additionally written to
    that file (created with parent directories as needed) so a run's full
    detail -- including per-step and per-iteration timings -- survives after
    the console output has scrolled away.
    """

    root = logging.getLogger()
    resolved_level = _resolve_level(level)
    root.setLevel(resolved_level)
    handler = _get_or_create_tqdm_handler(root)
    handler.setLevel(resolved_level)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    _configure_third_party_loggers(resolved_level)

    if log_file is not None:
        file_handler = _get_or_create_file_handler(root, Path(log_file))
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(logging.Formatter(fmt=file_fmt, datefmt=file_datefmt))


def get_logger(name: str) -> logging.Logger:
    """Returns a module-scoped logger."""

    return logging.getLogger(name)
