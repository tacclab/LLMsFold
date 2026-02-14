"""Backward-compatible utility exports.

This module keeps the original public utility import surface while the
implementation lives in smaller domain modules.
"""

from src.chemistry import (
    calculate_reward,
    extract_sequence_from_pdb,
    get_max_similarity,
    parse_smiles_from_text,
    passes_lipinski,
)
from src.pocket import get_binding_pockets_and_residues, get_p2rank_pocket, setup_p2rank
from src.services.pubchem import check_pubchem_patents

__all__ = [
    "calculate_reward",
    "check_pubchem_patents",
    "extract_sequence_from_pdb",
    "get_binding_pockets_and_residues",
    "get_max_similarity",
    "get_p2rank_pocket",
    "parse_smiles_from_text",
    "passes_lipinski",
    "setup_p2rank",
]
