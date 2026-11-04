"""Contract for holdout evaluators.

A holdout evaluator judges finished candidates. It is deliberately a
different type from the in-loop scorer: the optimizer is handed to the
generation loop, holdout evaluators are not, so no holdout number can reach
the prompt, the lead pool, the negative-example pools, or the similarity
registry. That separation is what lets the results claim independence --
if a holdout score could steer generation, the evaluation would be measuring
its own influence.

Holdout evaluators are optional and may be unavailable at runtime, since
they wrap third-party tooling this project does not install by default.
`availability` reports that as data rather than raising, so a run records
which checks did and did not happen instead of failing or silently omitting
them.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class HoldoutCandidate:
    """One finished candidate offered for independent evaluation."""

    smiles: str
    candidate_id: str
    # mmCIF complex produced by the in-loop predictor, when one was returned.
    structure: str | None = None


@dataclass(frozen=True)
class HoldoutContext:
    """Target-level inputs shared by every candidate in a run."""

    protein_pdb_path: Path | None = None
    # Pocket box from detection: center_x/y/z and size_x/y/z in Angstroms.
    docking_box: dict[str, float] | None = None


@dataclass(frozen=True)
class Availability:
    """Whether an evaluator can run, and why not when it cannot."""

    available: bool
    reason: str | None = None


class HoldoutEvaluator(Protocol):
    """Evaluates finished candidates without influencing how they were made."""

    name: str

    def availability(self) -> Availability:
        """Reports whether this evaluator's dependencies are usable."""

    def evaluate(
        self,
        candidates: Sequence[HoldoutCandidate],
        context: HoldoutContext,
    ) -> dict[str, dict[str, Any]]:
        """Returns report columns per candidate, keyed by SMILES.

        Candidates that cannot be evaluated are simply absent from the
        result; partial coverage is normal and is reported as such rather
        than being filled in with invented values.
        """
