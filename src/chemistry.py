"""Chemistry and text parsing utility functions."""

import ast
import json
import re
from typing import Any, Sequence

import pandas as pd
from pydantic import ValidationError
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from src.core.exceptions import SequenceExtractionError
from src.core.logging import get_logger
from src.core.messages import no_smiles_parsed, pdb_file_unreadable, pdb_sequence_missing
from src.schemas import ModelOutput

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
    "MSE": "M",
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


def _sequence_from_residue_codes(residue_codes: Sequence[str]) -> str:
    """Converts PDB residue codes into a one-letter sequence."""

    return "".join(_PDB_RESIDUE_TO_ONE_LETTER.get(code, "X") for code in residue_codes)


def _resolve_target_chain(pdb_path: str) -> tuple[str, str]:
    """Resolves the target chain id and its one-letter sequence from a PDB file.

    `SEQRES` is preferred; when absent, falls back to resolved residues in
    `ATOM` records. In both cases the first chain in sorted order is treated
    as the target chain -- the same chain whose sequence gets submitted to
    Boltz, so any residue/pocket logic keyed off this chain id stays
    consistent with what was actually sent to the API.

    Raises:
        SequenceExtractionError: If no sequence-bearing records can be parsed.
    """

    try:
        with open(pdb_path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError as exc:
        raise SequenceExtractionError(pdb_file_unreadable(pdb_path)) from exc

    seqres_chains: dict[str, list[str]] = {}
    for line in lines:
        if not line.startswith("SEQRES"):
            continue
        chain_id = line[11].strip() or "A"
        residues = line[19:].strip().split()
        seqres_chains.setdefault(chain_id, []).extend(residues)

    if seqres_chains:
        first_chain = sorted(seqres_chains.keys())[0]
        return first_chain, _sequence_from_residue_codes(seqres_chains[first_chain])

    atom_chains: dict[str, list[str]] = {}
    seen_residues: set[tuple[str, str, str]] = set()
    saw_model = False
    for line in lines:
        record = line[:6]
        if record.startswith("MODEL"):
            if saw_model:
                break
            saw_model = True
            continue
        if saw_model and record == "ENDMDL":
            break
        if record != "ATOM  ":
            continue

        chain_id = line[21].strip() or "A"
        residue_number = line[22:26].strip()
        insertion_code = line[26].strip()
        residue_key = (chain_id, residue_number, insertion_code)
        if residue_key in seen_residues:
            continue

        seen_residues.add(residue_key)
        residue_name = line[17:20].strip()
        atom_chains.setdefault(chain_id, []).append(residue_name)

    if not atom_chains:
        raise SequenceExtractionError(pdb_sequence_missing(pdb_path))

    first_chain = sorted(atom_chains.keys())[0]
    return first_chain, _sequence_from_residue_codes(atom_chains[first_chain])


def extract_sequence_from_pdb(pdb_path: str) -> str:
    """Extracts a protein sequence from PDB `SEQRES` or `ATOM` records.

    Args:
        pdb_path: Path to the PDB file.

    Returns:
        The primary sequence encoded in the structure. `SEQRES` is preferred;
        when absent, the function falls back to resolved residues in `ATOM`
        records for the first chain.

    Raises:
        SequenceExtractionError: If no sequence-bearing records can be parsed.
    """

    _, sequence = _resolve_target_chain(pdb_path)
    return sequence


def extract_target_chain_id(pdb_path: str) -> str:
    """Returns the chain id whose sequence is submitted to Boltz.

    This is the same chain `extract_sequence_from_pdb` reads from, so pocket
    residue contacts can be filtered to this chain before being remapped onto
    Boltz's single-polymer id ("A") in the request payload.

    Raises:
        SequenceExtractionError: If no sequence-bearing records can be parsed.
    """

    chain_id, _ = _resolve_target_chain(pdb_path)
    return chain_id


def calculate_heuristic_score(row: pd.Series) -> float:
    """Computes heuristic ranking score with over-similarity and synthesizability weighting.

    Args:
        row: A dataframe row with `adj_affinity`, `MaxSim`, and `synth_factor`.

    Returns:
        Score value used to rank candidates: the novelty-adjusted affinity
        scaled by `synth_factor` (1.0 for the easiest-to-synthesize molecules,
        0.0 for molecules at or beyond the SAS ceiling).
    """

    affinity = float(row["adj_affinity"])
    similarity_penalty = 0.5 * affinity if float(row["MaxSim"]) > 0.9 else 0.0
    affinity_component = affinity - similarity_penalty
    return affinity_component * float(row["synth_factor"])


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


def parse_smiles_from_text(raw_text: str) -> tuple[list[str], int]:
    """Extracts SMILES candidates from varied LLM response formats.

    Args:
        raw_text: Raw model output string.

    Returns:
        Tuple of (SMILES candidates parsed from quoted list entries, count of
        proposals discarded for failing SMILES validation before that list
        was produced).
    """

    smiles_pattern = re.compile(r"['\"]([A-Za-z0-9@+\-\[\]\(\)\\\/%=#$\.]+)['\"]")
    clean_text = re.sub(r"```[a-zA-Z]*\n?", "", raw_text).replace("```", "").strip()

    for parser in (ast.literal_eval, json.loads):
        try:
            output = ModelOutput.from_raw_payload(parser(clean_text))
            return output.smiles_list(), output.invalid_count
        except (ValueError, SyntaxError, json.JSONDecodeError, ValidationError):
            continue

    for pattern in (r"\{.*\}", r"\[.*?\]"):
        match = re.search(pattern, clean_text, re.DOTALL)
        if not match:
            continue
        candidate_text = match.group()
        for parser in (ast.literal_eval, json.loads):
            try:
                output = ModelOutput.from_raw_payload(parser(candidate_text))
                return output.smiles_list(), output.invalid_count
            except (ValueError, SyntaxError, json.JSONDecodeError, ValidationError):
                continue

    matches = smiles_pattern.findall(clean_text)
    if matches:
        try:
            output = ModelOutput.from_raw_payload(matches)
            return output.smiles_list(), output.invalid_count
        except ValidationError:
            logger.warning(no_smiles_parsed())
            logger.debug("Unparsed LLM output preview: %s", clean_text[:200])
            return [], len(matches)

    logger.warning(no_smiles_parsed())
    logger.debug("Unparsed LLM output preview: %s", clean_text[:200])
    return [], 0
