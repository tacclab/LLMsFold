"""Tests for pydantic schema models."""

import pytest
from pydantic import ValidationError

from src.schemas import BoltzPrediction, PatentCheckResult, PipelineOptions


@pytest.mark.parametrize(
    ("max_iterations", "max_samples", "is_valid"),
    [
        (1, 1, True),
        (3, 5, True),
        (0, 1, False),
        (1, 0, False),
    ],
)
def test_pipeline_options_constraints(max_iterations: int, max_samples: int, is_valid: bool) -> None:
    """`PipelineOptions` enforces lower bounds for loop/sample controls."""

    payload = {
        "pdb_path": "protein.pdb",
        "few_shot_csv": "few_shot.csv",
        "output_dir": "results",
        "protein_sequence": "MKT",
        "max_iterations": max_iterations,
        "max_samples": max_samples,
    }

    if is_valid:
        model = PipelineOptions(**payload)
        assert model.max_iterations == max_iterations
        assert model.max_samples == max_samples
    else:
        with pytest.raises(ValidationError):
            PipelineOptions(**payload)


@pytest.mark.parametrize(
    ("cid", "expected_known", "expected_cid_field"),
    [
        (None, "No", "N/A"),
        (0, "No", "N/A"),
        (12345, "Yes", 12345),
    ],
)
def test_patent_check_to_report_row(cid: int | None, expected_known: str, expected_cid_field: int | str) -> None:
    """Patent report rows are normalized for report-friendly fields."""

    result = PatentCheckResult(pubchem_cid=cid, identity_patents=2, substructure_patents=4)
    row = result.to_report_row()

    assert row["PubChem_CID"] == expected_cid_field
    assert row["Identity_Patents"] == 2
    assert row["Substructure_Patents"] == 4
    assert row["PubChem_Known"] == expected_known
    assert "legal novelty" in row["PubChem_Novelty_Note"]


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
