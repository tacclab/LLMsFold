"""Scaffold-level diversity and novelty analysis for generated candidates.

Whole-molecule Tanimoto similarity hides the failure mode reviewers look for
first: a run can produce fifteen molecules that all decorate one chemical
core differently, scoring as "distinct" on a fingerprint comparison while
exploring a single scaffold family. Bemis-Murcko scaffolds collapse that
decoration away and expose how many genuinely different cores a run found,
and how many of them were already present in the seed molecules.

Every function here degrades to `None` rather than raising: this analysis
runs at the end of an expensive run, and a molecule RDKit cannot parse must
never destroy results that have already been paid for.
"""

from collections import Counter
from math import log2
from typing import Any, Sequence

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.core.constants import MORGAN_FINGERPRINT_RADIUS
from src.core.logging import get_logger

logger = get_logger(__name__)

# Molecules with no ring system have no Bemis-Murcko scaffold; RDKit returns
# an empty string for them. They are counted separately rather than being
# pooled into one artificial "empty" scaffold family.
ACYCLIC_SCAFFOLD = ""


def _safe_mol(smiles: str) -> Any | None:
    """Parses SMILES, returning `None` for anything that is not an RDKit mol.

    The `isinstance` check matters as much as the exception handling: callers
    may run under test doubles that stand in for RDKit, and a non-molecule
    slipping through here would surface as a confusing failure deep inside a
    scaffold call.
    """

    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:  # noqa: BLE001 - RDKit raises bare exceptions on odd input
        return None
    return mol if isinstance(mol, Chem.Mol) else None


def murcko_scaffold(smiles: str) -> str | None:
    """Returns the Bemis-Murcko scaffold SMILES, or `None` if unparseable.

    Returns `ACYCLIC_SCAFFOLD` (an empty string) for molecules with no ring
    system, which is a valid answer rather than a failure.
    """

    mol = _safe_mol(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:  # noqa: BLE001
        return None


def generic_scaffold(smiles: str) -> str | None:
    """Returns the element- and bond-order-agnostic scaffold framework.

    Two scaffolds that differ only by a heteroatom swap or a saturation
    change collapse to the same generic framework, so this is the stricter
    of the two novelty views: a candidate can carry a "new" Murcko scaffold
    while reusing a seed's framework outright.
    """

    mol = _safe_mol(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaffold))
    except Exception:  # noqa: BLE001
        return None


def _shannon_entropy(counts: Sequence[int]) -> float | None:
    """Returns the Shannon entropy (bits) of a scaffold-occupancy histogram."""

    total = sum(counts)
    if total <= 0 or len(counts) <= 1:
        return 0.0 if total > 0 else None
    return -sum((count / total) * log2(count / total) for count in counts if count > 0)


class ScaffoldAnalyzer:
    """Compares candidate scaffolds against a fixed set of seed molecules.

    Built once per run from the seed molecules, then reused for both
    per-candidate annotation and the run-level summary, so every number
    reported is measured against the same reference set.
    """

    def __init__(self, seed_smiles: Sequence[str]) -> None:
        """Precomputes the seed scaffolds and their fingerprints."""

        self._fingerprint_generator: Any | None
        try:
            self._fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=MORGAN_FINGERPRINT_RADIUS
            )
        except Exception:  # noqa: BLE001
            self._fingerprint_generator = None

        self._seed_scaffolds: set[str] = set()
        self._seed_generic_scaffolds: set[str] = set()
        self._seed_scaffold_fps: list[Any] = []

        for smiles in seed_smiles:
            scaffold = murcko_scaffold(smiles)
            if scaffold:
                self._seed_scaffolds.add(scaffold)
                fingerprint = self._fingerprint(scaffold)
                if fingerprint is not None:
                    self._seed_scaffold_fps.append(fingerprint)
            generic = generic_scaffold(smiles)
            if generic:
                self._seed_generic_scaffolds.add(generic)

    @property
    def seed_scaffold_count(self) -> int:
        """Number of distinct Murcko scaffolds across the seed molecules."""

        return len(self._seed_scaffolds)

    def _fingerprint(self, smiles: str) -> Any | None:
        """Returns a Morgan fingerprint for a SMILES, or `None` on failure."""

        if self._fingerprint_generator is None:
            return None
        mol = _safe_mol(smiles)
        if mol is None:
            return None
        try:
            return self._fingerprint_generator.GetFingerprint(mol)
        except Exception:  # noqa: BLE001
            return None

    def max_scaffold_similarity_to_seeds(self, scaffold: str) -> float | None:
        """Returns the closest Tanimoto match between a scaffold and the seeds."""

        if not scaffold or not self._seed_scaffold_fps:
            return None
        fingerprint = self._fingerprint(scaffold)
        if fingerprint is None:
            return None
        try:
            similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, self._seed_scaffold_fps)
        except Exception:  # noqa: BLE001
            return None
        return max(similarities) if similarities else None

    def annotate(self, smiles: str) -> dict[str, Any]:
        """Returns the per-candidate scaffold columns for one molecule.

        `Scaffold_Is_Novel` answers the question an exact-structure PubChem
        lookup cannot: whether the candidate's core differs from every core
        it was shown, rather than merely differing as a whole molecule.
        """

        scaffold = murcko_scaffold(smiles)
        generic = generic_scaffold(smiles)
        # Novelty is left undefined, not False, when there is no scaffold to
        # judge: an unparseable molecule and a ring-free one both have no
        # core, and reporting False would read as "matches a seed core".
        return {
            "Murcko_Scaffold": scaffold,
            "Generic_Scaffold": generic,
            "Scaffold_Is_Novel": (scaffold not in self._seed_scaffolds if scaffold else None),
            "Generic_Scaffold_Is_Novel": (
                generic not in self._seed_generic_scaffolds if generic else None
            ),
            "Max_Scaffold_Sim_To_Seeds": self.max_scaffold_similarity_to_seeds(scaffold or ""),
        }

    def summarize(self, smiles_list: Sequence[str]) -> dict[str, Any]:
        """Returns run-level scaffold diversity and novelty statistics.

        `scaffold_diversity` is unique scaffolds per molecule: 1.0 means
        every candidate found a different core, and a value near 1/N means
        the run converged on a single family. `internal_diversity` is the
        conventional 1 - mean pairwise Tanimoto over whole molecules, kept
        alongside the scaffold view because the two disagree exactly in the
        case worth reporting -- heavy decoration of one core.
        """

        molecules = len(smiles_list)
        if molecules == 0:
            return {"molecules": 0}

        scaffolds = [murcko_scaffold(smiles) for smiles in smiles_list]
        generics = [generic_scaffold(smiles) for smiles in smiles_list]
        ring_scaffolds = [scaffold for scaffold in scaffolds if scaffold]
        unparseable = sum(1 for scaffold in scaffolds if scaffold is None)
        acyclic = sum(1 for scaffold in scaffolds if scaffold == ACYCLIC_SCAFFOLD)

        counts = Counter(ring_scaffolds)
        unique_scaffolds = len(counts)
        on_seed_scaffolds = sum(
            1 for scaffold in ring_scaffolds if scaffold in self._seed_scaffolds
        )

        summary: dict[str, Any] = {
            "molecules": molecules,
            "unique_murcko_scaffolds": unique_scaffolds,
            "unique_generic_scaffolds": len({generic for generic in generics if generic}),
            "scaffold_diversity": (round(unique_scaffolds / molecules, 4) if molecules else None),
            "largest_scaffold_cluster": max(counts.values()) if counts else 0,
            "scaffold_entropy_bits": (
                round(value, 4)
                if (value := _shannon_entropy(list(counts.values()))) is not None
                else None
            ),
            "seed_scaffolds": len(self._seed_scaffolds),
            "molecules_on_seed_scaffolds": on_seed_scaffolds,
            "novel_scaffold_fraction": (
                round(1.0 - on_seed_scaffolds / len(ring_scaffolds), 4) if ring_scaffolds else None
            ),
            "acyclic_molecules": acyclic,
            "unparseable_molecules": unparseable,
        }

        internal_diversity = self._internal_diversity(smiles_list)
        summary["internal_diversity"] = (
            round(internal_diversity, 4) if internal_diversity is not None else None
        )
        return summary

    def _internal_diversity(self, smiles_list: Sequence[str]) -> float | None:
        """Returns 1 - mean pairwise Tanimoto over whole molecules."""

        fingerprints = [
            fingerprint
            for fingerprint in (self._fingerprint(smiles) for smiles in smiles_list)
            if fingerprint is not None
        ]
        if len(fingerprints) < 2:
            return None

        similarities: list[float] = []
        try:
            for index, fingerprint in enumerate(fingerprints[:-1]):
                similarities.extend(
                    DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints[index + 1 :])
                )
        except Exception:  # noqa: BLE001
            return None
        if not similarities:
            return None
        return 1.0 - sum(similarities) / len(similarities)
