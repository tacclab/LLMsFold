"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Holds runtime configuration loaded from `.env` and OS env vars.

    Attributes:
        groq_api_key: API key used for Groq chat completions.
        nvidia_api_key: API key used for NVIDIA Boltz requests.
        pdb_file: Default PDB file path.
        few_shot_csv: Default few-shot examples CSV path.
        output_dir: Default output directory for reports.
        max_iterations: Default number of RL-style optimization iterations.
        max_samples: Default number of generated molecules per iteration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    groq_api_key: SecretStr = Field(alias="GROQ_API_KEY")
    nvidia_api_key: SecretStr = Field(alias="NVIDIA_API_KEY")
    pdb_file: str = Field(default="data/target.pdb", alias="PDB_FILE")
    few_shot_csv: str = Field(default="data/few_shot_smiles1.csv", alias="FEW_SHOT_CSV")
    output_dir: str = Field(default="results", alias="OUTPUT_DIR")
    max_iterations: int = Field(default=3, alias="MAX_ITERATIONS")
    max_samples: int = Field(default=5, alias="MAX_SAMPLES")


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Returns a cached settings object to avoid repeated env parsing."""

    return AppSettings()  # type: ignore[call-arg]
