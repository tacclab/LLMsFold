"""Chemistry and text parsing utility functions."""

import ast
import json
import re
from typing import Any, Sequence

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from src.core.logging import get_logger

logger = get_logger(__name__)

_PDB_RESIDUE_TO_ONE_LETTER = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


def extract_sequence_from_pdb(pdb_path: str) -> str:
    """Extracts a protein sequence from PDB `SEQRES` records.

    Args:
        pdb_path: Path to the PDB file.

    Returns:
        The primary sequence encoded in the structure.

    Raises:
        ValueError: If no SEQRES records can be parsed.
    """

    chains: dict[str, list[str]] = {}
    try:
        with open(pdb_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("SEQRES"):
                    continue
                chain_id = line[11].strip() or "A"
                residues = line[19:].strip().split()
                chains.setdefault(chain_id, []).extend(residues)
    except OSError as exc:
        raise ValueError(f"Could not open or read PDB file: {pdb_path}") from exc

    if not chains:
        raise ValueError(f"No SEQRES records found in {pdb_path}")

    first_chain = sorted(chains.keys())[0]
    return "".join(_PDB_RESIDUE_TO_ONE_LETTER.get(code, "X") for code in chains[first_chain])


def calculate_heuristic_score(row: pd.Series) -> float:
    """Computes heuristic ranking score with over-similarity penalty.

    Args:
        row: A dataframe row with `adj_affinity` and `MaxSim`.

    Returns:
        Score value used to rank candidates.
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
    """Extracts SMILES candidates from varied LLM response formats.

    Args:
        raw_text: Raw model output string.

    Returns:
        List of SMILES candidates parsed from quoted list entries.
    """

    smiles_pattern = re.compile(r"['\"]([A-Za-z0-9@+\-\[\]\(\)\\\/%=#$\.]{5,})['\"]")
    clean_text = re.sub(r"```[a-zA-Z]*\n?", "", raw_text).replace("```", "").strip()

    list_match = re.search(r"\[.*?\]", clean_text, re.DOTALL)
    if list_match:
        try:
            candidates = ast.literal_eval(list_match.group())
            if isinstance(candidates, list):
                return [item for item in candidates if isinstance(item, str) and len(item) >= 5]
        except (ValueError, SyntaxError):
            pass

    try:
        payload = json.loads(clean_text)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, str) and len(item) >= 5]
    except json.JSONDecodeError:
        pass

    matches = smiles_pattern.findall(clean_text)
    if matches:
        return matches

    logger.warning("Could not parse any SMILES from LLM output: %s", clean_text[:200])
    return []


# Backward-compatible alias for existing imports/tests.
calculate_reward = calculate_heuristic_score
