"""Pose validity checking with PoseBusters.

PoseBusters is the one check in this pipeline that predicts nothing. It
measures whether a pose is physically possible -- bond lengths and angles,
internal clashes, ring flatness, steric overlap with the protein, whether the
ligand actually sits in the pocket -- using geometry and chemistry rules
rather than a learned model.

That is what makes it worth more here than a second co-folding network. Two
neural predictors trained on overlapping structural data can agree with each
other and still both be wrong; agreement between them is cross-method
consistency, not evidence about the true bound state. A geometry check cannot
be circular with the model that produced the pose, because it is not in the
business of predicting poses at all.
"""

import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.core.logging import get_logger
from src.evaluators.base import Availability, HoldoutCandidate, HoldoutContext
from src.structures import split_complex

logger = get_logger(__name__)

# Checks the protein-conditioned config runs. Kept out of the result columns
# individually; the report carries the pass/fail summary and the names of
# whatever failed, so a new check in a future PoseBusters release is picked
# up automatically instead of being silently dropped.
_CONFIG = "dock"


class PoseBustersEvaluator:
    """Validates predicted poses against physical and chemical constraints."""

    name = "posebusters"

    def __init__(self, config: str = _CONFIG) -> None:
        """Stores the PoseBusters config used for every candidate in a run."""

        self._config = config

    def availability(self) -> Availability:
        """Reports whether the optional `posebusters` package is importable."""

        try:
            import posebusters  # noqa: F401
        except ImportError:
            return Availability(
                available=False,
                reason=(
                    "posebusters is not installed; install the 'holdout' extra "
                    "(uv sync --extra holdout) to enable pose validity checks"
                ),
            )
        return Availability(available=True)

    def evaluate(
        self,
        candidates: Sequence[HoldoutCandidate],
        context: HoldoutContext,
    ) -> dict[str, dict[str, Any]]:
        """Runs pose validity checks on each candidate's predicted complex.

        The protein half is taken from the predicted complex itself rather
        than from the input PDB, so clash and pocket-occupancy checks are made
        against the exact conformation the ligand was placed into.
        """

        from posebusters import PoseBusters

        buster = PoseBusters(config=self._config)
        results: dict[str, dict[str, Any]] = {}

        with tempfile.TemporaryDirectory(prefix="llmsfold-posebusters-") as workdir:
            protein_path = Path(workdir) / "protein.pdb"
            for candidate in candidates:
                if not candidate.structure:
                    continue
                parts = split_complex(candidate.structure, candidate.smiles)
                if parts is None:
                    logger.debug("Could not split predicted complex for %s", candidate.candidate_id)
                    continue

                protein_path.write_text(parts.protein_pdb_block, encoding="utf-8")
                try:
                    report = buster.bust(parts.ligand, mol_true=None, mol_cond=protein_path)
                except Exception as exc:  # noqa: BLE001 - third-party failure modes vary
                    logger.debug("PoseBusters failed for %s: %s", candidate.candidate_id, exc)
                    continue

                columns = self._summarize(report)
                if columns is None:
                    continue
                columns["PoseBusters_Bond_Orders_Assigned"] = parts.bond_orders_assigned
                results[candidate.smiles] = columns

        return results

    @staticmethod
    def _summarize(report: Any) -> dict[str, Any] | None:
        """Condenses a PoseBusters report row into report columns.

        Every boolean column is treated as a check, so checks added by a
        future PoseBusters release are counted without needing a list here.
        """

        if report is None or len(report) == 0:
            return None

        row = report.iloc[0]
        # pandas hands back `numpy.bool_`, which is not a Python `bool`, so a
        # bare `isinstance(value, bool)` silently matches nothing and drops
        # every check.
        checks = {
            str(name): bool(value)
            for name, value in row.items()
            if isinstance(value, (bool, np.bool_))
        }
        if not checks:
            return None

        failed = sorted(name for name, passed in checks.items() if not passed)
        return {
            "PoseBusters_Pass": not failed,
            "PoseBusters_Checks_Passed": len(checks) - len(failed),
            "PoseBusters_Checks_Total": len(checks),
            "PoseBusters_Failed_Checks": ", ".join(failed),
        }
