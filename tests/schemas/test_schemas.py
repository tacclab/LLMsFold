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
    ("cid", "expected_known", "expected_cid_field", "expected_classification"),
    [
        (None, "No", "N/A", "Novel/Not in PubChem"),
        (0, "No", "N/A", "Novel/Not in PubChem"),
        (12345, "Yes", 12345, "Patent-Referenced"),
    ],
)
def test_patent_check_to_report_row(
    cid: int | None,
    expected_known: str,
    expected_cid_field: int | str,
    expected_classification: str,
) -> None:
    """Patent report rows are normalized for report-friendly fields."""

    result = PatentCheckResult(pubchem_cid=cid, identity_patents=2, substructure_patents=4)
    row = result.to_report_row()

    assert row["PubChem_CID"] == expected_cid_field
    assert row["Identity_Patents"] == 2
    assert row["Substructure_Patents"] == 4
    assert row["PubChem_Known"] == expected_known
    assert row["PubChem_Classification"] == expected_classification
    assert "not a formal drug registry lookup" in str(row["PubChem_Novelty_Note"])


def test_patent_check_to_report_row_classifies_known_drug() -> None:
    """A CID with documented drug/medication info is classified as a known drug."""

    result = PatentCheckResult(pubchem_cid=2244, has_drug_info=True)
    row = result.to_report_row()

    assert row["PubChem_Classification"] == "Known Drug/Medication"


def test_patent_check_to_report_row_classifies_unclassified_known_compound() -> None:
    """A CID with no drug info and no patent hits is an unclassified known compound."""

    result = PatentCheckResult(pubchem_cid=999, has_drug_info=False)
    row = result.to_report_row()

    assert row["PubChem_Classification"] == "Known Compound (Unclassified)"


def test_boltz_prediction_ignores_extra_fields() -> None:
    """Boltz schema keeps expected fields and ignores unknown keys."""

    parsed = BoltzPrediction.model_validate(
        {
            "structures": [{"structure": "data_\nloop_", "format": "mmcif"}],
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
    assert parsed.structures[0].structure == "data_\nloop_"


def test_boltz_prediction_rejects_invalid_structure_entry() -> None:
    """Structure entries missing the required 'structure' field should fail validation."""

    with pytest.raises(ValidationError):
        BoltzPrediction.model_validate(
            {
                "structures": [{"format": "mmcif"}],  # missing required 'structure' field
                "confidence_scores": [0.7],
            }
        )


def test_boltz_prediction_affinity_new_fields_optional() -> None:
    """All new affinity sub-fields default to empty lists when absent."""

    parsed = BoltzPrediction.model_validate(
        {
            "structures": [{"structure": "data_", "format": "mmcif"}],
            "confidence_scores": [0.7],
            "affinities": {"L1": {"affinity_probability_binary": [0.8]}},
        }
    )

    aff = parsed.affinities["L1"]
    assert aff.affinity_probability_binary == [0.8]
    assert aff.affinity_pred_value == []
    assert aff.affinity_pic50 == []
    assert aff.model_1_affinity_probability_binary == []
    assert aff.model_2_affinity_probability_binary == []


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


def test_model_output_deduplicates_across_smiles_notations() -> None:
    """The same molecule written two different ways is recognized as one entry.

    Regression test: before canonicalizing in `_validate_smiles`, raw-string
    dedup would treat a Kekule-form and an aromatic-form SMILES for benzene
    as two distinct molecules, double-counting and double-scoring it.
    """

    parsed = ModelOutput.from_raw_payload(
        {"molecules": ["C1=CC=CC=C1", "c1ccccc1"]}  # both are benzene
    )

    assert parsed.smiles_list() == ["c1ccccc1"]


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
