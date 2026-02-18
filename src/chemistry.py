"""Chemistry and text parsing utility functions."""

import re
from typing import Any, Sequence

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator


def extract_sequence_from_pdb(pdb_path: str) -> str:
    """Extracts a protein sequence from a PDB file.

    Args:
        pdb_path: Path to the PDB file.

    Returns:
        The primary sequence encoded in the structure.

    Raises:
        ValueError: If the PDB cannot be parsed by RDKit.
    """

    mol = Chem.MolFromPDBFile(pdb_path)
    if not mol:
        raise ValueError(f"Could not parse PDB file at {pdb_path}")
    return Chem.MolToSequence(mol)


def calculate_reward(row: pd.Series) -> float:
    """Computes RL-style score with over-similarity penalty.

    Args:
        row: A dataframe row with `adj_affinity` and `MaxSim`.

    Returns:
        Reward value used to rank candidates.
    """

    affinity = float(row["adj_affinity"])
    penalty = 0.5 * affinity if float(row["MaxSim"]) > 0.9 else 0.0
    return affinity - penalty


def passes_lipinski(mol: Any) -> bool:
    """Checks Lipinski's Rule of Five criteria.

    Args:
        mol: RDKit molecule instance.

    Returns:
        `True` when all Lipinski thresholds pass.
    """

    if mol is None:
        return False
    return all(
        [
            Descriptors.MolWt(mol) <= 500,
            Descriptors.MolLogP(mol) <= 5,
            Descriptors.NumHDonors(mol) <= 5,
            Descriptors.NumHAcceptors(mol) <= 10,
        ]
    )


def get_max_similarity(smiles: str, target_fps: Sequence[Any]) -> float:
    """Calculates max Tanimoto similarity to a target fingerprint set.

    Args:
        smiles: Candidate molecule in SMILES format.
        target_fps: Precomputed RDKit fingerprints for reference molecules.

    Returns:
        Maximum Tanimoto similarity, or `0.0` for invalid input.
    """

    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return 0.0
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
        fingerprint = generator.GetFingerprint(mol)
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, list(target_fps))
        return max(similarities) if similarities else 0.0
    except Exception:
        return 0.0


def parse_smiles_from_text(raw_text: str) -> list[str]:
    """Extracts SMILES-like strings from an LLM Python-list response.

    Args:
        raw_text: Raw model output string.

    Returns:
        List of SMILES candidates parsed from quoted list entries.
    """

    list_match = re.search(r"\[\s*['\"](.*?)['\"]\s*\]", raw_text, re.DOTALL)
    if list_match:
        return re.findall(r"['\"]([a-zA-Z0-9@+\-\[\]\(\)\\\/%=#$]{5,})['\"]", raw_text)
    return []
