"""Project-local synthetic accessibility score helpers."""

from __future__ import annotations

import gzip
import math
import pickle
from functools import lru_cache
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

_FRAGMENT_SCORES_PATH = Path(__file__).resolve().parents[1] / "fpscores.pkl.gz"
_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2)


@lru_cache(maxsize=1)
def _read_fragment_scores() -> dict[int, float]:
    """Loads fragment scores from the bundled RDKit-compatible score table."""

    if not _FRAGMENT_SCORES_PATH.exists():
        raise FileNotFoundError(f"Missing SA score data file at {_FRAGMENT_SCORES_PATH}")

    with gzip.open(_FRAGMENT_SCORES_PATH, "rb") as handle:
        data = pickle.load(handle)

    scores: dict[int, float] = {}
    for entry in data:
        score = float(entry[0])
        for bit_id in entry[1:]:
            scores[int(bit_id)] = score
    return scores


def _num_bridgeheads_and_spiro(mol: Chem.Mol) -> tuple[int, int]:
    """Returns the number of bridgehead and spiro atoms."""

    return (
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        rdMolDescriptors.CalcNumSpiroAtoms(mol),
    )


def calculate_sa_score(mol: Chem.Mol) -> float:
    """Calculates the canonical RDKit-style synthetic accessibility score."""

    if mol is None:
        raise ValueError("Mol must not be None")
    if not mol.GetNumAtoms():
        raise ValueError("Mol must contain at least one atom")

    fragment_scores = _read_fragment_scores()
    fingerprint = _MORGAN_GENERATOR.GetSparseCountFingerprint(mol)
    nonzero_elements = fingerprint.GetNonzeroElements()

    fragment_score = 0.0
    fragment_count = 0
    for bit_id, count in nonzero_elements.items():
        fragment_count += count
        fragment_score += fragment_scores.get(bit_id, -4.0) * count
    fragment_score /= float(fragment_count)

    num_atoms = mol.GetNumAtoms()
    num_chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    ring_info = mol.GetRingInfo()
    num_bridgeheads, num_spiro = _num_bridgeheads_and_spiro(mol)
    num_macrocycles = sum(1 for ring in ring_info.AtomRings() if len(ring) > 8)

    size_penalty = num_atoms**1.005 - num_atoms
    stereo_penalty = math.log10(num_chiral_centers + 1.0)
    spiro_penalty = math.log10(num_spiro + 1.0)
    bridge_penalty = math.log10(num_bridgeheads + 1.0)
    macrocycle_penalty = math.log10(2.0) if num_macrocycles > 0 else 0.0

    feature_score = (
        0.0 - size_penalty - stereo_penalty - spiro_penalty - bridge_penalty - macrocycle_penalty
    )

    symmetry_correction = 0.0
    num_bits = len(nonzero_elements)
    if num_atoms > num_bits:
        symmetry_correction = 0.5 * math.log(float(num_atoms) / num_bits)

    sa_score = fragment_score + feature_score + symmetry_correction
    sa_score = 11.0 - (sa_score - (-4.0) + 1.0) / (2.5 - (-4.0)) * 9.0

    if sa_score > 8.0:
        sa_score = 8.0 + math.log(sa_score - 8.0)
    if sa_score > 10.0:
        sa_score = 10.0
    elif sa_score < 1.0:
        sa_score = 1.0

    return sa_score


def sa_score(smiles: str) -> float:
    """Returns the synthetic accessibility score for a SMILES string."""

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return calculate_sa_score(mol)


def sa_score_mol(mol: Chem.Mol) -> float:
    """Returns the synthetic accessibility score for an RDKit molecule."""

    return calculate_sa_score(mol)
