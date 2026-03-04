"""Unit tests for chemistry helpers."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src import chemistry


@pytest.mark.parametrize(
    ("adj_affinity", "max_sim", "expected"),
    [
        (0.8, 0.95, 0.4),
        (0.8, 0.9, 0.8),
        (0.0, 0.99, 0.0),
    ],
)
def test_calculate_reward(adj_affinity: float, max_sim: float, expected: float) -> None:
    """Reward penalizes over-similar compounds when MaxSim exceeds 0.9."""

    row = pd.Series({"adj_affinity": adj_affinity, "MaxSim": max_sim})
    assert chemistry.calculate_reward(row) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("['CCOCC', 'CCNCC']", ["CCOCC", "CCNCC"]),
        ('Result: ["C1=CC=CC=C1"]', ["C1=CC=CC=C1"]),
        ("No list here", []),
        ("['bad']", []),
    ],
)
def test_parse_smiles_from_text(raw_text: str, expected: list[str]) -> None:
    """Only quoted list-like payloads with plausible SMILES are extracted."""

    assert chemistry.parse_smiles_from_text(raw_text) == expected


@pytest.mark.parametrize(
    ("weights", "logp", "donors", "acceptors", "expected"),
    [
        (300.0, 2.0, 1, 3, True),
        (600.0, 2.0, 1, 3, False),
        (300.0, 6.0, 1, 3, False),
        (300.0, 2.0, 6, 3, False),
        (300.0, 2.0, 1, 11, False),
    ],
)
def test_passes_lipinski_with_mocked_descriptors(
    monkeypatch: pytest.MonkeyPatch,
    weights: float,
    logp: float,
    donors: int,
    acceptors: int,
    expected: bool,
) -> None:
    """Lipinski result is based on descriptor thresholds."""

    molecule = MagicMock(name="mol")
    monkeypatch.setattr(chemistry.Descriptors, "MolWt", lambda _mol: weights)
    monkeypatch.setattr(chemistry.Descriptors, "MolLogP", lambda _mol: logp)
    monkeypatch.setattr(chemistry.Descriptors, "NumHDonors", lambda _mol: donors)
    monkeypatch.setattr(chemistry.Descriptors, "NumHAcceptors", lambda _mol: acceptors)

    assert chemistry.passes_lipinski(molecule) is expected


def test_passes_lipinski_rejects_none() -> None:
    """`None` molecules always fail Lipinski checks."""

    assert chemistry.passes_lipinski(None) is False


def test_extract_sequence_from_pdb_success(tmp_path: Path) -> None:
    """Sequence extraction reads canonical residues from SEQRES records."""

    pdb_path = tmp_path / "protein.pdb"
    pdb_path.write_text(
        "SEQRES   1 A    4  MET LYS THR GLY\nSEQRES   1 B    2  ALA CYS\n",
        encoding="utf-8",
    )

    assert chemistry.extract_sequence_from_pdb(str(pdb_path)) == "MKTG"


def test_extract_sequence_from_pdb_raises_on_missing_seqres(tmp_path: Path) -> None:
    """A PDB without SEQRES entries raises a `ValueError`."""

    pdb_path = tmp_path / "bad_file.pdb"
    pdb_path.write_text("HEADER\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No SEQRES records found"):
        chemistry.extract_sequence_from_pdb(str(pdb_path))


def test_get_max_similarity_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Similarity helper returns max score from bulk tanimoto output."""

    fake_generator = MagicMock(name="fingerprint_generator")
    fake_generator.GetFingerprint.return_value = "candidate_fp"

    monkeypatch.setattr(chemistry.Chem, "MolFromSmiles", lambda smiles: object() if smiles == "CCO" else None)
    monkeypatch.setattr(chemistry.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_generator)
    monkeypatch.setattr(chemistry.DataStructs, "BulkTanimotoSimilarity", lambda _fp, _targets: [0.12, 0.78])

    assert chemistry.get_max_similarity("CCO", ["t1", "t2"]) == pytest.approx(0.78)
    assert chemistry.get_max_similarity("invalid", ["t1", "t2"]) == 0.0


def test_get_max_similarity_returns_zero_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected RDKit errors are caught and downgraded to 0.0."""

    fake_generator = MagicMock(name="fingerprint_generator")
    fake_generator.GetFingerprint.return_value = "candidate_fp"

    monkeypatch.setattr(chemistry.Chem, "MolFromSmiles", lambda _smiles: object())
    monkeypatch.setattr(chemistry.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_generator)

    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(chemistry.DataStructs, "BulkTanimotoSimilarity", _raise)

    assert chemistry.get_max_similarity("CCO", ["target"]) == 0.0
