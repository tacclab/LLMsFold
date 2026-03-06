"""Tests for the project-local synthetic accessibility scorer."""

import pytest
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer as rdkit_sascorer

from src.sa_score import sa_score, sa_score_mol


@pytest.mark.parametrize(
    "smiles",
    [
        "CCO",
        "CC(=O)Oc1ccccc1C(=O)O",
        "COc1cc(F)cc(-c2ncnc(N3CCOCC3)c2C)c1C(=O)N",
    ],
)
def test_sa_score_matches_rdkit_contrib(smiles: str) -> None:
    """Local scorer should stay numerically aligned with RDKit's reference code."""

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None

    assert sa_score(smiles) == pytest.approx(rdkit_sascorer.calculateScore(mol))
    assert sa_score_mol(mol) == pytest.approx(rdkit_sascorer.calculateScore(mol))
