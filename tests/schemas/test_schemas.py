"""Tests for pydantic schema models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.schemas import (
    BoltzPrediction,
    ModelOutput,
    MoleculeRecord,
    PatentCheckResult,
    PipelineOptions,
    UnifiedReportRow,
)


@pytest.mark.parametrize(
    ("max_iterations", "max_samples", "is_valid"),
    [
        (1, 1, True),
        (3, 5, True),
        (0, 1, False),
        (1, 0, False),
    ],
)
def test_pipeline_options_constraints(
    max_iterations: int, max_samples: int, is_valid: bool
) -> None:
    """`PipelineOptions` enforces lower bounds for loop/sample controls."""

    payload: dict[str, object] = {
        "pdb_path": Path("protein.pdb"),
        "few_shot_csv": Path("few_shot.csv"),
        "output_dir": Path("results"),
        "protein_sequence": "MKT",
        "max_iterations": max_iterations,
        "max_samples": max_samples,
    }

    if is_valid:
        model = PipelineOptions.model_validate(payload)
        assert model.max_iterations == max_iterations
        assert model.max_samples == max_samples
    else:
        with pytest.raises(ValidationError):
            PipelineOptions.model_validate(payload)


def test_pipeline_options_normalizes_protein_sequence() -> None:
    """Protein sequences are stripped of whitespace and normalized to uppercase."""

    model = PipelineOptions.model_validate(
        {
            "pdb_path": Path("protein.pdb"),
            "few_shot_csv": Path("few_shot.csv"),
            "output_dir": Path("results"),
            "protein_sequence": " mk t \n",
        }
    )

    assert model.protein_sequence == "MKT"


@pytest.mark.parametrize(
    ("cid", "expected_known", "expected_cid_field"),
    [
        (None, "No", "N/A"),
        (0, "No", "N/A"),
        (12345, "Yes", 12345),
    ],
)
def test_patent_check_to_report_row(
    cid: int | None, expected_known: str, expected_cid_field: int | str
) -> None:
    """Patent report rows are normalized for report-friendly fields."""

    result = PatentCheckResult(pubchem_cid=cid, identity_patents=2, substructure_patents=4)
    row = result.to_report_row()

    assert row["PubChem_CID"] == expected_cid_field
    assert row["Identity_Patents"] == 2
    assert row["Substructure_Patents"] == 4
    assert row["PubChem_Known"] == expected_known
    assert "legal novelty" in str(row["PubChem_Novelty_Note"])


def test_boltz_prediction_ignores_extra_fields() -> None:
    """Boltz schema keeps expected fields and ignores unknown keys."""

    parsed = BoltzPrediction.model_validate(
        {
            "ptm_scores": [0.9],
            "iptm_scores": [0.8],
            "confidence_scores": [0.7],
            "complex_plddt_scores": [0.6],
            "affinities": {
                "L1": {
                    "affinity_probability_binary": [0.95],
                    "affinity_pic50": [7.2],
                    "unexpected": "ignored",
                }
            },
            "unknown_root": "ignored",
        }
    )

    assert parsed.ptm_scores == [0.9]
    assert parsed.affinities["L1"].affinity_pic50 == [7.2]


def test_boltz_prediction_requires_score_fields() -> None:
    """Malformed Boltz payloads should fail validation instead of zero-filling."""

    with pytest.raises(ValidationError):
        BoltzPrediction.model_validate({"affinities": {}})


def test_model_output_from_raw_payload_filters_invalid_and_duplicate_smiles() -> None:
    """LLM output schema keeps valid unique molecules and drops invalid proposals."""

    parsed = ModelOutput.from_raw_payload(
        {
            "molecules": [
                " CCO ",
                {"smiles": "CCN"},
                {"SMILES": "CCO"},
                "invalid",
            ]
        }
    )

    assert parsed.smiles_list() == ["CCO", "CCN"]


def test_model_output_requires_at_least_one_valid_molecule() -> None:
    """Purely invalid model payloads should fail validation."""

    with pytest.raises(ValidationError):
        ModelOutput.from_raw_payload({"molecules": ["invalid", "bad"]})


def test_molecule_record_validates_smiles() -> None:
    """Output molecule rows reject invalid SMILES values."""

    with pytest.raises(ValidationError):
        MoleculeRecord.model_validate(
            {
                "SMILES": "bad",
                "pTM": 0.9,
                "ipTM": 0.8,
                "Confidence": 0.7,
                "pLDDT": 0.6,
                "Affinity_Prob": 0.95,
                "pIC50": 7.2,
                "IC50_uM": 0.1,
                "MolWt": 46.07,
                "LogP": 0.2,
                "QED": 0.5,
                "SAS": 2.3,
                "TPSA": 20.0,
                "H_Acceptors": 1,
                "H_Donors": 1,
                "Rotatable_Bonds": 0,
            }
        )


def test_unified_report_row_rejects_inconsistent_pubchem_fields() -> None:
    """Final report rows must keep PubChem status aligned with CID values."""

    with pytest.raises(ValidationError):
        UnifiedReportRow.model_validate(
            {
                "SMILES": "CCO",
                "pTM": 0.9,
                "ipTM": 0.8,
                "Confidence": 0.7,
                "pLDDT": 0.6,
                "Affinity_Prob": 0.95,
                "pIC50": 7.2,
                "IC50_uM": 0.1,
                "MolWt": 46.07,
                "LogP": 0.2,
                "QED": 0.5,
                "SAS": 2.3,
                "TPSA": 20.0,
                "H_Acceptors": 1,
                "H_Donors": 1,
                "Rotatable_Bonds": 0,
                "MaxSim": 0.4,
                "adj_affinity": 0.95,
                "score": 0.95,
                "PubChem_CID": 12345,
                "Identity_Patents": 0,
                "Substructure_Patents": 0,
                "PubChem_Known": "No",
                "PubChem_Novelty_Note": "Absence from PubChem does not establish legal novelty.",
            }
        )
