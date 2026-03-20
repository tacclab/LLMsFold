"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.constants import (
    ADJ_AFFINITY_THRESHOLD,
    DEFAULT_BOLTZ_DIFFUSION_SAMPLES,
    DEFAULT_BOLTZ_MAX_WAIT_SECONDS,
    DEFAULT_BOLTZ_POLL_HEADER_SECONDS,
    DEFAULT_BOLTZ_POLL_INTERVAL_SECONDS,
    DEFAULT_BOLTZ_RECYCLING_STEPS,
    DEFAULT_BOLTZ_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_BOLTZ_RETRY_ATTEMPTS,
    DEFAULT_BOLTZ_RETRY_MAX_WAIT_SECONDS,
    DEFAULT_BOLTZ_RETRY_MIN_WAIT_SECONDS,
    DEFAULT_BOLTZ_SAMPLING_STEPS,
    DEFAULT_BOLTZ_STEP_SCALE,
    DEFAULT_BOLTZ_WITHOUT_POTENTIALS,
    DEFAULT_FEW_SHOT_CSV,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PDB_FILE,
    DEFAULT_PUBCHEM_LISTKEY_ATTEMPTS,
    DEFAULT_PUBCHEM_POLL_INTERVAL_SECONDS,
    DEFAULT_PUBCHEM_TIMEOUT_SECONDS,
    SAS_SCORE_MAX,
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
    adj_affinity_threshold: float = Field(
        default=ADJ_AFFINITY_THRESHOLD,
        alias="ADJ_AFFINITY_THRESHOLD",
        ge=0.0,
        le=1.0,
    )
    sas_score_max: float = Field(default=SAS_SCORE_MAX, alias="SAS_SCORE_MAX", ge=1.0, le=10.0)
    best_structure_affinity_threshold: float | None = Field(
        default=None,
        alias="BEST_STRUCTURE_AFFINITY_THRESHOLD",
        ge=0.0,
        le=1.0,
    )
    neoralab_viewer_url: str = Field(
        default="https://neoralab.app/viewer",
        alias="NEORALAB_VIEWER_URL",
    )
    neoralab_viewer_client_id: str | None = Field(
        default=None,
        alias="NEORALAB_VIEWER_CLIENT_ID",
    )
    neoralab_viewer_client_secret: SecretStr | None = Field(
        default=None,
        alias="NEORALAB_VIEWER_CLIENT_SECRET",
    )
    groq_api_key: SecretStr | None = Field(default=None, alias="GROQ_API_KEY")
    boltz_request_timeout_seconds: float = Field(
        default=DEFAULT_BOLTZ_REQUEST_TIMEOUT_SECONDS,
        alias="BOLTZ_REQUEST_TIMEOUT_SECONDS",
        gt=0.0,
    )
    boltz_max_wait_seconds: int = Field(
        default=DEFAULT_BOLTZ_MAX_WAIT_SECONDS,
        alias="BOLTZ_MAX_WAIT_SECONDS",
        ge=1,
    )
    boltz_poll_interval_seconds: float = Field(
        default=DEFAULT_BOLTZ_POLL_INTERVAL_SECONDS,
        alias="BOLTZ_POLL_INTERVAL_SECONDS",
        gt=0.0,
    )
    boltz_poll_header_seconds: int = Field(
        default=DEFAULT_BOLTZ_POLL_HEADER_SECONDS,
        alias="BOLTZ_POLL_HEADER_SECONDS",
        ge=1,
    )
    boltz_retry_attempts: int = Field(
        default=DEFAULT_BOLTZ_RETRY_ATTEMPTS,
        alias="BOLTZ_RETRY_ATTEMPTS",
        ge=1,
    )
    boltz_retry_min_wait_seconds: float = Field(
        default=DEFAULT_BOLTZ_RETRY_MIN_WAIT_SECONDS,
        alias="BOLTZ_RETRY_MIN_WAIT_SECONDS",
        ge=0.0,
    )
    boltz_retry_max_wait_seconds: float = Field(
        default=DEFAULT_BOLTZ_RETRY_MAX_WAIT_SECONDS,
        alias="BOLTZ_RETRY_MAX_WAIT_SECONDS",
        ge=0.0,
    )
    boltz_recycling_steps: int = Field(
        default=DEFAULT_BOLTZ_RECYCLING_STEPS,
        alias="BOLTZ_RECYCLING_STEPS",
        ge=1,
    )
    boltz_sampling_steps: int = Field(
        default=DEFAULT_BOLTZ_SAMPLING_STEPS,
        alias="BOLTZ_SAMPLING_STEPS",
        ge=1,
    )
    boltz_diffusion_samples: int = Field(
        default=DEFAULT_BOLTZ_DIFFUSION_SAMPLES,
        alias="BOLTZ_DIFFUSION_SAMPLES",
        ge=1,
    )
    boltz_step_scale: float = Field(
        default=DEFAULT_BOLTZ_STEP_SCALE,
        alias="BOLTZ_STEP_SCALE",
        gt=0.0,
    )
    boltz_without_potentials: bool = Field(
        default=DEFAULT_BOLTZ_WITHOUT_POTENTIALS,
        alias="BOLTZ_WITHOUT_POTENTIALS",
    )
    pubchem_timeout_seconds: float = Field(
        default=DEFAULT_PUBCHEM_TIMEOUT_SECONDS,
        alias="PUBCHEM_TIMEOUT_SECONDS",
        gt=0.0,
    )
    pubchem_listkey_attempts: int = Field(
        default=DEFAULT_PUBCHEM_LISTKEY_ATTEMPTS,
        alias="PUBCHEM_LISTKEY_ATTEMPTS",
        ge=1,
    )
    pubchem_poll_interval_seconds: float = Field(
        default=DEFAULT_PUBCHEM_POLL_INTERVAL_SECONDS,
        alias="PUBCHEM_POLL_INTERVAL_SECONDS",
        gt=0.0,
    )


class AppSettings(GeneratorSettings):
    """Holds runtime configuration loaded from `.env` and OS env vars."""

    groq_api_key: SecretStr = Field(alias="GROQ_API_KEY")
    nvidia_api_key: SecretStr = Field(alias="NVIDIA_API_KEY")
    pdb_file: Path = Field(default=Path(DEFAULT_PDB_FILE), alias="PDB_FILE")
    few_shot_csv: Path = Field(default=Path(DEFAULT_FEW_SHOT_CSV), alias="FEW_SHOT_CSV")
    output_dir: Path = Field(default=Path(DEFAULT_OUTPUT_DIR), alias="OUTPUT_DIR")
    max_iterations: int = Field(default=DEFAULT_MAX_ITERATIONS, alias="MAX_ITERATIONS", ge=1)
    max_samples: int = Field(default=DEFAULT_MAX_SAMPLES, alias="MAX_SAMPLES", ge=1)
    log_level: str = Field(default=DEFAULT_LOG_LEVEL, alias="LOG_LEVEL")


@lru_cache(maxsize=1)
def get_generator_settings() -> GeneratorSettings:
    """Returns cached generation-only settings."""

    return GeneratorSettings()


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Returns a cached settings object to avoid repeated env parsing."""

    return AppSettings()  # type: ignore[call-arg]
