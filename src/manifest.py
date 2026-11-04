"""Run provenance manifest written next to every pipeline report.

Every threshold that decides which molecules reach the final report is
environment-configurable (see `GeneratorSettings`), so a report file on its
own cannot be audited: two runs with identical CSV headers may have applied
completely different affinity, SAS, ipTM, and pLDDT gates. This module
snapshots the fully resolved configuration, the input file digests, the code
version, and the outcome of every iteration into `run_manifest.json` beside
the report, so any table derived from a run can be traced back to the exact
settings that produced it.

The manifest is written twice: once immediately at run start (so a crashed
or interrupted run still leaves its configuration behind) and once at the
end with timings and per-iteration results filled in.
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.core.logging import get_logger

logger = get_logger(__name__)

MANIFEST_SCHEMA_VERSION = 1
REDACTED = "***REDACTED***"

# Any settings key containing one of these substrings is redacted even if its
# value is a plain string rather than a `SecretStr`, so a credential added to
# the settings model later cannot silently leak into a published manifest.
_SECRET_KEY_HINTS = ("api_key", "apikey", "token", "secret", "password", "credential")

# Distribution names whose versions materially affect numeric results. Both
# RDKit distribution spellings are probed because the wheel is published under
# `rdkit` and `rdkit-pypi` depending on the platform.
_TRACKED_PACKAGES = (
    "rdkit",
    "rdkit-pypi",
    "pandas",
    "numpy",
    "deepchem",
    "pydantic",
    "pydantic-settings",
    "groq",
    "httpx",
    "tenacity",
)

_DIGEST_CHUNK_BYTES = 1 << 20
_GIT_TIMEOUT_SECONDS = 5.0


def file_digest(path: str | Path | None) -> str | None:
    """Returns the SHA-256 hex digest of a file, or `None` if unreadable.

    Input files are digested rather than copied so a manifest can prove which
    PDB and few-shot CSV produced a report without duplicating the data.
    """

    if path is None:
        return None

    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while chunk := handle.read(_DIGEST_CHUNK_BYTES):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _git_revision(repo_root: Path) -> dict[str, Any]:
    """Returns the current commit and dirty flag, tolerating non-git checkouts.

    A missing git binary or an unversioned directory is a normal condition
    here, not an error: the manifest simply records that no revision was
    available rather than failing the run.
    """

    def _run(args: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                args,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=_GIT_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    commit = _run(["git", "rev-parse", "HEAD"])
    if commit is None:
        return {"commit": None, "dirty": None, "branch": None}

    status = _run(["git", "status", "--porcelain"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "branch": branch,
    }


def _package_versions() -> dict[str, str]:
    """Returns installed versions for packages that can change numeric output."""

    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def redact_settings(settings: BaseModel) -> dict[str, Any]:
    """Dumps settings to JSON-safe values with every credential removed.

    Redaction is driven by both the declared field type (`SecretStr`) and the
    field name, so neither a renamed secret field nor a credential typed as a
    plain string ends up in a manifest that is meant to be shared alongside a
    publication.
    """

    dumped = settings.model_dump(mode="json")
    for key in list(dumped):
        raw_value = getattr(settings, key, None)
        looks_secret = any(hint in key.lower() for hint in _SECRET_KEY_HINTS)
        if looks_secret or hasattr(raw_value, "get_secret_value"):
            dumped[key] = REDACTED if raw_value is not None else None
    return dumped


class IterationRecord(BaseModel):
    """Outcome of a single generation iteration.

    `MoleculeGenerator.run` catches per-iteration exceptions so one failed
    round does not abort a long run. Without this record that failure is
    invisible in the results: a run reported as "3 iterations" may have
    completed only one, which would misstate both the computational cost and
    the amount of optimization the final candidates actually received.
    """

    model_config = ConfigDict(extra="forbid")

    iteration: int = Field(ge=1)
    status: Literal["succeeded", "failed"]
    error: str | None = None
    proposed: int = 0
    valid_after_filters: int = 0
    boltz_scored: int = 0
    kept: int = 0
    llm_seconds: float = 0.0
    boltz_seconds: float = 0.0


class RunManifest(BaseModel):
    """Full provenance record for one pipeline run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = MANIFEST_SCHEMA_VERSION
    run_id: str
    invocation: list[str] = Field(default_factory=list)
    started_at: str
    finished_at: str | None = None
    elapsed_seconds: float | None = None
    status: Literal["running", "completed", "no_candidates", "failed"] = "running"
    code: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)
    pocket: dict[str, Any] | None = None
    iterations: list[IterationRecord] = Field(default_factory=list)
    diversity: dict[str, Any] = Field(default_factory=dict)
    holdout: dict[str, Any] = Field(default_factory=dict)
    reference_baseline: dict[str, Any] | None = None
    results: dict[str, Any] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)

    def write(self, path: Path) -> None:
        """Writes the manifest as JSON, logging rather than raising on failure.

        A manifest is provenance, not a result: failing to persist it must
        never take down a run that has already spent API budget on scoring.
        """

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Could not write run manifest to %s: %s", path, exc)


def build_run_manifest(
    *,
    run_id: str,
    started_at: datetime,
    settings: BaseModel,
    thresholds: dict[str, Any],
    inputs: dict[str, Any],
    repo_root: Path | None = None,
) -> RunManifest:
    """Assembles the manifest recorded at the start of a run.

    Args:
        run_id: Identifier shared with the run's log file.
        started_at: Wall-clock start time.
        settings: Resolved settings object; credentials are redacted.
        thresholds: The scoring gates applied to this run, surfaced as their
            own block so they can be read without parsing the full settings.
        inputs: Input paths and their digests, plus target metadata.
        repo_root: Directory used for the git revision lookup.

    Returns:
        A manifest with provenance populated and results left empty.
    """

    root = repo_root or Path(__file__).resolve().parents[1]
    return RunManifest(
        run_id=run_id,
        invocation=list(sys.argv),
        started_at=started_at.astimezone(timezone.utc).isoformat(),
        code={
            "git": _git_revision(root),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        inputs=inputs,
        thresholds=thresholds,
        settings=redact_settings(settings),
    )
