"""Unit tests for chemistry helpers."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src import chemistry
from src.core.exceptions import SequenceExtractionError


@pytest.mark.parametrize(
    ("adj_affinity", "max_sim", "synth_factor", "expected"),
    [
        (0.8, 0.95, 1.0, 0.4),
        (0.8, 0.9, 1.0, 0.8),
        (0.0, 0.99, 1.0, 0.0),
        (0.8, 0.5, 0.5, 0.4),
        (0.8, 0.95, 0.5, 0.2),
    ],
)
def test_calculate_heuristic_score(
    adj_affinity: float, max_sim: float, synth_factor: float, expected: float
) -> None:
    """Heuristic score penalizes over-similar compounds and scales by synthesizability."""

    row = pd.Series({"adj_affinity": adj_affinity, "MaxSim": max_sim, "synth_factor": synth_factor})
    assert chemistry.calculate_heuristic_score(row) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("raw_text", "expected_smiles", "expected_invalid_count"),
    [
        ("['CCOCC', 'CCNCC']", ["CCOCC", "CCNCC"], 0),
        ('Result: ["C1=CC=CC=C1"]', ["c1ccccc1"], 0),  # canonicalized to aromatic form
        ('{"molecules": ["CCO", "invalid", {"SMILES": "CCN"}]}', ["CCO", "CCN"], 1),
        ("No list here", [], 0),
        ("['bad']", [], 1),
    ],
)
def test_parse_smiles_from_text(
    raw_text: str, expected_smiles: list[str], expected_invalid_count: int
) -> None:
    """Only quoted list-like payloads with plausible SMILES are extracted, invalid ones counted."""

    smiles, invalid_count = chemistry.parse_smiles_from_text(raw_text)
    assert smiles == expected_smiles
    assert invalid_count == expected_invalid_count


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


def test_extract_sequence_from_pdb_falls_back_to_atom_records(tmp_path: Path) -> None:
    """Resolved residues are used when `SEQRES` records are unavailable."""

    pdb_path = tmp_path / "resolved_only.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "ATOM      1  N   MET A   1      10.000  11.000  12.000  1.00 20.00           N",
                "ATOM      2  CA  MET A   1      10.500  11.500  12.500  1.00 20.00           C",
                "ATOM      3  N   LYS A   2      11.000  12.000  13.000  1.00 20.00           N",
                "ATOM      4  CA  LYS A   2      11.500  12.500  13.500  1.00 20.00           C",
                "ATOM      5  N   THR B   1      12.000  13.000  14.000  1.00 20.00           N",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert chemistry.extract_sequence_from_pdb(str(pdb_path)) == "MK"


def test_extract_sequence_from_pdb_raises_on_missing_sequence_records(tmp_path: Path) -> None:
    """A PDB without `SEQRES` or `ATOM` entries raises a domain-specific error."""

    pdb_path = tmp_path / "bad_file.pdb"
    pdb_path.write_text("HEADER\n", encoding="utf-8")

    with pytest.raises(SequenceExtractionError, match="No SEQRES or ATOM residue records found"):
        chemistry.extract_sequence_from_pdb(str(pdb_path))


def test_extract_target_chain_id_matches_seqres_chain(tmp_path: Path) -> None:
    """The reported target chain is the same one the sequence was read from."""

    pdb_path = tmp_path / "protein.pdb"
    pdb_path.write_text(
        "SEQRES   1 A    4  MET LYS THR GLY\nSEQRES   1 B    2  ALA CYS\n",
        encoding="utf-8",
    )

    assert chemistry.extract_target_chain_id(str(pdb_path)) == "A"


def test_extract_residue_position_map_handles_atom_gaps(tmp_path: Path) -> None:
    """PDB residue numbers with a gap map onto consecutive sequence positions.

    Regression test for a bug where the raw PDB residue number (which can run
    far higher than the sequence length, e.g. 203-498 for a 294-residue
    construct) was sent to Boltz as if it were already a 1-based sequence
    position. Residues 10 and 11 are followed by a gap (12-14 missing, e.g. a
    disordered loop) before residue 15 resumes -- position 3 must map to PDB
    residue 15, not to some number in the 200s-400s range from an unrelated
    construct.
    """

    pdb_path = tmp_path / "gapped.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "ATOM      1  N   MET A  10      10.000  11.000  12.000  1.00 20.00           N",
                "ATOM      2  N   LYS A  11      11.000  12.000  13.000  1.00 20.00           N",
                "ATOM      3  N   GLY A  15      12.000  13.000  14.000  1.00 20.00           N",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    position_map = chemistry.extract_residue_position_map(str(pdb_path))
    sequence = chemistry.extract_sequence_from_pdb(str(pdb_path))

    assert sequence == "MKG"
    assert position_map == {10: 1, 11: 2, 15: 3}
    # The bug this guards against: no mapped index may fall outside the
    # sequence's own bounds, regardless of how the PDB numbers the residues.
    assert all(1 <= index <= len(sequence) for index in position_map.values())


def test_extract_residue_position_map_seqres_path_matching_counts(tmp_path: Path) -> None:
    """SEQRES-derived sequences map correctly when ATOM residues fully resolve them."""

    pdb_path = tmp_path / "seqres_complete.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "SEQRES   1 A    3  MET LYS GLY",
                "ATOM      1  N   MET A  50      10.000  11.000  12.000  1.00 20.00           N",
                "ATOM      2  N   LYS A  51      11.000  12.000  13.000  1.00 20.00           N",
                "ATOM      3  N   GLY A  52      12.000  13.000  14.000  1.00 20.00           N",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    position_map = chemistry.extract_residue_position_map(str(pdb_path))

    assert position_map == {50: 1, 51: 2, 52: 3}


def test_extract_residue_position_map_seqres_path_unresolved_residues(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """SEQRES longer than resolved ATOM residues yields an empty, safe map."""

    pdb_path = tmp_path / "seqres_gap.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "SEQRES   1 A    3  MET LYS GLY",
                "ATOM      1  N   MET A  50      10.000  11.000  12.000  1.00 20.00           N",
                "ATOM      2  N   GLY A  52      12.000  13.000  14.000  1.00 20.00           N",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with caplog.at_level("WARNING"):
        position_map = chemistry.extract_residue_position_map(str(pdb_path))

    assert position_map == {}
    assert any("Cannot safely map" in record.message for record in caplog.records)


def test_extract_target_chain_id_supports_non_a_only_chain(tmp_path: Path) -> None:
    """A structure whose only chain is not 'A' (e.g. chain C) reports that chain."""

    pdb_path = tmp_path / "chain_c_only.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "ATOM      1  N   MET C   1      10.000  11.000  12.000  1.00 20.00           N",
                "ATOM      2  CA  MET C   1      10.500  11.500  12.500  1.00 20.00           C",
                "ATOM      3  N   LYS C   2      11.000  12.000  13.000  1.00 20.00           N",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert chemistry.extract_target_chain_id(str(pdb_path)) == "C"
    assert chemistry.extract_sequence_from_pdb(str(pdb_path)) == "MK"


def test_get_max_similarity_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Similarity helper returns max score from bulk tanimoto output."""

    fake_generator = MagicMock(name="fingerprint_generator")
    fake_generator.GetFingerprint.return_value = "candidate_fp"

    monkeypatch.setattr(
        chemistry.Chem, "MolFromSmiles", lambda smiles: object() if smiles == "CCO" else None
    )
    monkeypatch.setattr(
        chemistry.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_generator
    )
    monkeypatch.setattr(
        chemistry.DataStructs, "BulkTanimotoSimilarity", lambda _fp, _targets: [0.12, 0.78]
    )

    assert chemistry.get_max_similarity("CCO", ["t1", "t2"]) == pytest.approx(0.78)
    assert chemistry.get_max_similarity("invalid", ["t1", "t2"]) == 0.0


def test_get_max_similarity_returns_zero_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected RDKit errors are caught and downgraded to 0.0."""

    fake_generator = MagicMock(name="fingerprint_generator")
    fake_generator.GetFingerprint.return_value = "candidate_fp"

    monkeypatch.setattr(chemistry.Chem, "MolFromSmiles", lambda _smiles: object())
    monkeypatch.setattr(
        chemistry.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_generator
    )

    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(chemistry.DataStructs, "BulkTanimotoSimilarity", _raise)

    assert chemistry.get_max_similarity("CCO", ["target"]) == 0.0
