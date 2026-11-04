"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.constants import (
    ADJ_AFFINITY_THRESHOLD,
    ALREADY_PROPOSED_WINDOW,
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
    DEFAULT_IPTM_THRESHOLD,
    DEFAULT_LEAD_DIVERSITY_MAX_SIM,
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PDB_FILE,
    DEFAULT_PLDDT_THRESHOLD,
    DEFAULT_POCKET_CONTACT_DISTANCE,
    DEFAULT_PUBCHEM_LISTKEY_ATTEMPTS,
    DEFAULT_PUBCHEM_MAX_CONCURRENCY,
    DEFAULT_PUBCHEM_MIN_REQUEST_INTERVAL_SECONDS,
    DEFAULT_PUBCHEM_POLL_INTERVAL_SECONDS,
    DEFAULT_PUBCHEM_TIMEOUT_SECONDS,
    SAS_SCORE_MAX,
    SEED_SMILES_LIMIT,
    SIMILARITY_PENALTY_FRACTION,
    SIMILARITY_PENALTY_THRESHOLD,
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
    llm_max_tokens: int = Field(
        default=DEFAULT_LLM_MAX_TOKENS,
        alias="LLM_MAX_TOKENS",
        ge=1,
    )
    llm_max_retries: int = Field(
        default=DEFAULT_LLM_MAX_RETRIES,
        alias="LLM_MAX_RETRIES",
        ge=0,
    )
    adj_affinity_threshold: float = Field(
        default=ADJ_AFFINITY_THRESHOLD,
        alias="ADJ_AFFINITY_THRESHOLD",
        ge=0.0,
        le=1.0,
    )
    sas_score_max: float = Field(default=SAS_SCORE_MAX, alias="SAS_SCORE_MAX", ge=1.0, le=10.0)
    similarity_penalty_threshold: float = Field(
        default=SIMILARITY_PENALTY_THRESHOLD,
        alias="SIMILARITY_PENALTY_THRESHOLD",
        ge=0.0,
        le=1.0,
    )
    similarity_penalty_fraction: float = Field(
        default=SIMILARITY_PENALTY_FRACTION,
        alias="SIMILARITY_PENALTY_FRACTION",
        ge=0.0,
        le=1.0,
    )
    seed_smiles_limit: int = Field(
        default=SEED_SMILES_LIMIT,
        alias="SEED_SMILES_LIMIT",
        ge=1,
    )
    # Free-text constraint appended to every generation prompt, for
    # exclusions the contrastive negative pools cannot express because they
    # are derived from scoring rather than stated up front.
    extra_prompt_constraints: str | None = Field(default=None, alias="EXTRA_PROMPT_CONSTRAINTS")
    already_proposed_window: int = Field(
        default=ALREADY_PROPOSED_WINDOW,
        alias="ALREADY_PROPOSED_WINDOW",
        ge=0,
    )
    lead_diversity_max_sim: float = Field(
        default=DEFAULT_LEAD_DIVERSITY_MAX_SIM,
        alias="LEAD_DIVERSITY_MAX_SIM",
        ge=0.0,
        le=1.0,
    )
    iptm_threshold: float = Field(
        default=DEFAULT_IPTM_THRESHOLD,
        alias="IPTM_THRESHOLD",
        ge=0.0,
        le=1.0,
    )
    plddt_threshold: float = Field(
        default=DEFAULT_PLDDT_THRESHOLD,
        alias="PLDDT_THRESHOLD",
        ge=0.0,
        le=1.0,
    )
    random_seed: int | None = Field(default=None, alias="RANDOM_SEED")
    # Prepared receptor for Vina rescoring. Left unset by default: preparing a
    # receptor PDBQT involves protonation and cofactor choices that belong to
    # whoever set up the target, not to this pipeline.
    vina_receptor_pdbqt: Path | None = Field(default=None, alias="VINA_RECEPTOR_PDBQT")
    enable_holdout_evaluation: bool = Field(default=True, alias="ENABLE_HOLDOUT_EVALUATION")
    best_structure_affinity_threshold: float | None = Field(
        default=None,
        alias="BEST_STRUCTURE_AFFINITY_THRESHOLD",
        ge=0.0,
        le=1.0,
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
    pubchem_max_concurrency: int = Field(
        default=DEFAULT_PUBCHEM_MAX_CONCURRENCY,
        alias="PUBCHEM_MAX_CONCURRENCY",
        ge=1,
    )
    pubchem_min_request_interval_seconds: float = Field(
        default=DEFAULT_PUBCHEM_MIN_REQUEST_INTERVAL_SECONDS,
        alias="PUBCHEM_MIN_REQUEST_INTERVAL_SECONDS",
        ge=0.0,
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
    # Optional user-specified binding pocket. Unset means the pocket is found
    # by geometry; any one of these pins it instead.
    pocket_residues: str | None = Field(default=None, alias="POCKET_RESIDUES")
    pocket_ligand: str | None = Field(default=None, alias="POCKET_LIGAND")
    pocket_center: str | None = Field(default=None, alias="POCKET_CENTER")
    pocket_index: int | None = Field(default=None, alias="POCKET_INDEX", ge=0)
    pocket_contact_distance: float = Field(
        default=DEFAULT_POCKET_CONTACT_DISTANCE,
        alias="POCKET_CONTACT_DISTANCE",
        gt=0.0,
    )


@lru_cache(maxsize=1)
def get_generator_settings() -> GeneratorSettings:
    """Returns cached generation-only settings."""

    return GeneratorSettings()


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Returns a cached settings object to avoid repeated env parsing."""

    return AppSettings()  # type: ignore[call-arg]
