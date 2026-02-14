"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.constants import (
    DEFAULT_FEW_SHOT_CSV,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PDB_FILE,
)


class GeneratorSettings(BaseSettings):
    """Generation-specific settings loaded from `.env` and OS env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    llm_model: str = Field(default=DEFAULT_LLM_MODEL, alias="LLM_MODEL")
    llm_temperature: float = Field(
        default=DEFAULT_LLM_TEMPERATURE,
        alias="LLM_TEMPERATURE",
        ge=0.0,
        le=2.0,
    )


class AppSettings(GeneratorSettings):
    """Holds runtime configuration loaded from `.env` and OS env vars."""

    groq_api_key: SecretStr = Field(alias="GROQ_API_KEY")
    nvidia_api_key: SecretStr = Field(alias="NVIDIA_API_KEY")
    pdb_file: str = Field(default=DEFAULT_PDB_FILE, alias="PDB_FILE")
    few_shot_csv: str = Field(default=DEFAULT_FEW_SHOT_CSV, alias="FEW_SHOT_CSV")
    output_dir: str = Field(default=DEFAULT_OUTPUT_DIR, alias="OUTPUT_DIR")
    max_iterations: int = Field(default=DEFAULT_MAX_ITERATIONS, alias="MAX_ITERATIONS")
    max_samples: int = Field(default=DEFAULT_MAX_SAMPLES, alias="MAX_SAMPLES")
    log_level: str = Field(default=DEFAULT_LOG_LEVEL, alias="LOG_LEVEL")


@lru_cache(maxsize=1)
def get_generator_settings() -> GeneratorSettings:
    """Returns cached generation-only settings."""

    return GeneratorSettings()  # type: ignore[call-arg]


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Returns a cached settings object to avoid repeated env parsing."""

    return AppSettings()  # type: ignore[call-arg]
