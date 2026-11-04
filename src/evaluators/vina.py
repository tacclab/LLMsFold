"""Scoring predicted poses with AutoDock Vina.

Vina is an empirical scoring function, not a learned co-folding model, so it
brings a genuinely different set of assumptions to the same pose. Its opinion
is weaker than a modern neural predictor's in absolute accuracy, but that is
not what it is for here: the question is whether a candidate selected by one
model still looks reasonable to a method that shares none of its training
data or inductive biases.

The predicted pose is scored where the co-folding model put it, and then
locally minimized. A large gap between the two energies means the pose sits
away from a Vina minimum, which is more informative than either number alone.

Receptor preparation is deliberately not automated. Converting a protein to
PDBQT well needs choices about protonation, flexible residues and cofactors
that belong to whoever set up the target, so this evaluator asks for a
prepared receptor via `VINA_RECEPTOR_PDBQT` and reports itself unavailable
when there is none, rather than silently scoring against a bad receptor.
"""

import shutil
from pathlib import Path
from typing import Any, Sequence

from src.core.logging import get_logger
from src.evaluators.base import Availability, HoldoutCandidate, HoldoutContext
from src.structures import split_complex

logger = get_logger(__name__)


class VinaEvaluator:
    """Rescores co-folded poses with the AutoDock Vina scoring function."""

    name = "vina"

    def __init__(self, receptor_pdbqt: Path | None, scoring_function: str = "vina") -> None:
        """Stores the prepared receptor and scoring function for the run."""

        self._receptor_pdbqt = receptor_pdbqt
        self._scoring_function = scoring_function

    def availability(self) -> Availability:
        """Reports whether Vina, Meeko, and a prepared receptor are all present."""

        missing = []
        try:
            import vina  # noqa: F401
        except ImportError:
            missing.append("vina")
        try:
            import meeko  # noqa: F401
        except ImportError:
            missing.append("meeko")
        if missing:
            return Availability(
                available=False,
                reason=(
                    f"{' and '.join(missing)} not installed; install the 'holdout' extra "
                    "(uv sync --extra holdout) to enable Vina rescoring"
                ),
            )

        if shutil.which("mk_prepare_receptor.py") is None:
            return Availability(
                available=False,
                reason=(
                    "mk_prepare_receptor.py is not on PATH; it ships with meeko and "
                    "is needed to build a receptor from each predicted complex"
                ),
            )
        return Availability(available=True)

    @staticmethod
    def _ligand_pdbqt(ligand: Any) -> str | None:
        """Converts an RDKit ligand with 3D coordinates into a PDBQT string.

        Meeko changed its writer API between releases, so both the current
        `PDBQTWriterLegacy` form and the older in-place form are attempted
        before giving up.
        """

        from rdkit import Chem

        try:
            protonated = Chem.AddHs(ligand, addCoords=True)
        except Exception:  # noqa: BLE001
            protonated = ligand

        from meeko import MoleculePreparation

        try:
            preparator = MoleculePreparation()
            setups = preparator.prepare(protonated)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Meeko ligand preparation failed: %s", exc)
            return None

        try:
            from meeko import PDBQTWriterLegacy

            pdbqt, success, error = PDBQTWriterLegacy.write_string(setups[0])
            if success:
                return str(pdbqt)
            logger.debug("Meeko PDBQT writer reported failure: %s", error)
            return None
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("Meeko PDBQT writing failed: %s", exc)
            return None

        try:
            return preparator.write_pdbqt_string()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Legacy Meeko PDBQT writing failed: %s", exc)
            return None

    @staticmethod
    def _box_around(ligand: Any, margin: float = 8.0) -> tuple[list[float], list[float]]:
        """Returns a Vina grid box enclosing the ligand plus `margin`."""

        import numpy as np

        conformer = ligand.GetConformer()
        points = np.array([list(conformer.GetAtomPosition(i)) for i in range(ligand.GetNumAtoms())])
        centre = points.mean(axis=0)
        size = (points.max(axis=0) - points.min(axis=0)) + 2 * margin
        return [float(v) for v in centre], [float(v) for v in size]

    @staticmethod
    def _receptor_pdbqt_from_block(pdb_block: str, workdir: Path) -> Path | None:
        """Prepares a receptor PDBQT from one complex's own protein atoms."""

        import subprocess

        source = workdir / "receptor.pdb"
        source.write_text(pdb_block, encoding="utf-8")
        target = workdir / "receptor"
        try:
            subprocess.run(
                [
                    "mk_prepare_receptor.py",
                    "--read_pdb",
                    str(source),
                    "-o",
                    str(target),
                    "-p",
                    "-a",
                    "--default_altloc",
                    "A",
                ],
                capture_output=True,
                check=False,
                timeout=300,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("Receptor preparation failed: %s", exc)
            return None
        prepared = target.with_suffix(".pdbqt")
        return prepared if prepared.exists() else None

    def evaluate(
        self,
        candidates: Sequence[HoldoutCandidate],
        context: HoldoutContext,
    ) -> dict[str, dict[str, Any]]:
        """Scores each candidate's predicted pose with Vina.

        Both halves of the complex are taken from the same predicted
        structure. A co-folding model returns its result in its own
        coordinate frame, not the input structure's, so scoring a predicted
        ligand against the input receptor places the two tens of Angstroms
        apart and Vina rejects the pose as outside the grid. The grid is
        likewise built around the ligand's own position rather than from a
        box derived from the input structure.
        """

        import tempfile

        from vina import Vina

        results: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            if not candidate.structure:
                continue
            parts = split_complex(candidate.structure, candidate.smiles)
            if parts is None:
                continue

            pdbqt = self._ligand_pdbqt(parts.ligand)
            if not pdbqt:
                continue

            with tempfile.TemporaryDirectory(prefix="llmsfold-vina-") as workdir:
                receptor = self._receptor_pdbqt_from_block(parts.protein_pdb_block, Path(workdir))
                if receptor is None:
                    logger.debug("No receptor for %s", candidate.candidate_id)
                    continue

                centre, size = self._box_around(parts.ligand)
                try:
                    engine = Vina(sf_name=self._scoring_function, verbosity=0)
                    engine.set_receptor(rigid_pdbqt_filename=str(receptor))
                    engine.compute_vina_maps(center=centre, box_size=size)
                    engine.set_ligand_from_string(pdbqt)
                    in_place = float(engine.score()[0])
                    minimized = float(engine.optimize()[0])
                except Exception as exc:  # noqa: BLE001 - Vina raises on unscorable poses
                    logger.debug("Vina scoring failed for %s: %s", candidate.candidate_id, exc)
                    continue

            results[candidate.smiles] = {
                "Vina_Score_kcal_per_mol": round(in_place, 3),
                "Vina_Minimized_kcal_per_mol": round(minimized, 3),
                # A large drop means the co-folded pose was not at a Vina
                # minimum, which the raw score alone would not reveal.
                "Vina_Minimization_Gain": round(in_place - minimized, 3),
            }

        return results
