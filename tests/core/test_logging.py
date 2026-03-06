"""Tests for centralized logging configuration."""

import logging

from src.core.logging import TqdmLoggingHandler, configure_logging


def test_configure_logging_quiets_http_transport_loggers() -> None:
    """HTTP transport loggers should stay at WARNING or above in normal runs."""

    httpx_logger = logging.getLogger("httpx")
    httpcore_logger = logging.getLogger("httpcore")
    original_httpx_level = httpx_logger.level
    original_httpcore_level = httpcore_logger.level

    try:
        configure_logging(level="INFO")
        assert httpx_logger.level == logging.WARNING
        assert httpcore_logger.level == logging.WARNING

        configure_logging(level="ERROR")
        assert httpx_logger.level == logging.ERROR
        assert httpcore_logger.level == logging.ERROR
    finally:
        httpx_logger.setLevel(original_httpx_level)
        httpcore_logger.setLevel(original_httpcore_level)


def test_configure_logging_reuses_single_tqdm_handler() -> None:
    """Repeated configuration should keep a single tqdm-aware root handler."""

    root = logging.getLogger()
    original_handlers = root.handlers.copy()
    original_level = root.level

    try:
        root.handlers.clear()

        configure_logging(level="INFO")
        configure_logging(level="DEBUG")

        tqdm_handlers = [
            handler for handler in root.handlers if isinstance(handler, TqdmLoggingHandler)
        ]
        assert len(tqdm_handlers) == 1
        assert tqdm_handlers[0].level == logging.DEBUG
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)
