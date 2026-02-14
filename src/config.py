"""Compatibility shim for settings module.

Use `src.core.config` for new imports.
"""

from src.core.config import AppSettings, GeneratorSettings, get_generator_settings, get_settings

__all__ = [
    "AppSettings",
    "GeneratorSettings",
    "get_generator_settings",
    "get_settings",
]
