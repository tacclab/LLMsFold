"""Tests for backward-compatible utility re-exports."""

import pytest

from src import chemistry, pocket, utils
from src.services import pubchem


@pytest.mark.parametrize(
    ("export_name", "expected"),
    [
        ("calculate_reward", chemistry.calculate_reward),
        ("extract_sequence_from_pdb", chemistry.extract_sequence_from_pdb),
        ("get_max_similarity", chemistry.get_max_similarity),
        ("parse_smiles_from_text", chemistry.parse_smiles_from_text),
        ("passes_lipinski", chemistry.passes_lipinski),
        ("get_binding_pockets_and_residues", pocket.get_binding_pockets_and_residues),
        ("get_p2rank_pocket", pocket.get_p2rank_pocket),
        ("setup_p2rank", pocket.setup_p2rank),
        ("check_pubchem_patents", pubchem.check_pubchem_patents),
    ],
)
def test_utils_reexports(export_name: str, expected) -> None:
    """`src.utils` keeps a stable import surface by forwarding implementations."""

    assert getattr(utils, export_name) is expected


def test_utils_all_lists_exported_symbols() -> None:
    """`__all__` contains all published utility helpers."""

    assert sorted(utils.__all__) == [
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
