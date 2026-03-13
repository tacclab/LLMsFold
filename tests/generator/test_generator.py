"""Tests for molecule generation orchestration."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src import generator
from src.core.config import GeneratorSettings
from src.core.exceptions import PocketDetectionError
from src.generator import MoleculeGenerator
from src.prompt import build_user_prompt
from src.schemas import PatentCheckResult


@pytest.fixture
def generator_dependencies(monkeypatch: pytest.MonkeyPatch):
    """Builds reusable mocked dependencies for MoleculeGenerator tests."""

    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content="['CCO']"))]

    groq_client = MagicMock()
    groq_client.chat.completions.create.return_value = completion

    filter_catalog = MagicMock()
    filter_catalog.HasMatch.return_value = False

    monkeypatch.setattr(generator, "get_cached_groq_client", lambda _api_key: groq_client)
    monkeypatch.setattr(generator, "_get_filter_catalog", lambda: filter_catalog)

    return {"groq": groq_client, "filter": filter_catalog}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ALA10_A, GLY27_B", [10, 27]),
        (["ASN4_A", "LYS100_B"], [4, 100]),
        ("invalid,residue", []),
        (None, []),
    ],
)
def test_extract_residue_indices(raw, expected: list[int]) -> None:
    """Residue parser extracts numeric indices from multiple input shapes."""

    contacts = MoleculeGenerator._extract_residue_contacts(raw)
    assert [item.residue_index for item in contacts] == expected


@pytest.mark.parametrize(
    ("use_pocket", "residues", "max_samples", "expected_fragment"),
    [
        (True, "ALA10_A", 4, "binding pocket containing"),
        (False, None, 4, "Generate 4 bioisosteres"),
    ],
)
def test_build_user_prompt_variants(
    use_pocket: bool,
    residues: str | None,
    max_samples: int,
    expected_fragment: str,
) -> None:
    """Prompt builder emits the correct template per workflow mode."""

    prompt = build_user_prompt(use_pocket, residues, "CCO", max_samples)
    assert expected_fragment in prompt


def test_post_process_scores_enriches_and_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Post-processing adds ranking columns and filters high-SAS compounds."""

    source = pd.DataFrame(
        [
            {"SMILES": "CCO", "Affinity_Prob": 0.7, "SAS": 4.0},
            {"SMILES": "CCN", "Affinity_Prob": 0.5, "SAS": 7.0},
        ]
    )

    monkeypatch.setattr(generator, "get_max_similarity", lambda _smiles, _target_fps: 0.8)
    monkeypatch.setattr(
        generator, "calculate_heuristic_score", lambda row: row["adj_affinity"] + 0.1
    )

    scored = MoleculeGenerator._post_process_scores(source, target_fps=["fp"])

    assert list(scored["SMILES"]) == ["CCO"]
    assert scored.iloc[0]["adj_affinity"] == pytest.approx(0.7)
    assert scored.iloc[0]["score"] == pytest.approx(0.8)


def test_post_process_scores_accepts_custom_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Score thresholds should be overrideable from runtime settings."""

    source = pd.DataFrame(
        [
            {"SMILES": "CCO", "Affinity_Prob": 0.7, "SAS": 6.5},
            {"SMILES": "CCN", "Affinity_Prob": 0.95, "SAS": 5.5},
        ]
    )

    monkeypatch.setattr(generator, "get_max_similarity", lambda _smiles, _target_fps: 0.8)
    monkeypatch.setattr(generator, "calculate_heuristic_score", lambda row: row["adj_affinity"])

    scored = MoleculeGenerator._post_process_scores(
        source,
        target_fps=["fp"],
        adj_affinity_threshold=0.9,
        sas_score_max=7.0,
    )

    assert list(scored["SMILES"]) == ["CCO", "CCN"]
    assert scored.iloc[0]["adj_affinity"] == pytest.approx(0.0)
    assert scored.iloc[1]["adj_affinity"] == pytest.approx(0.95)


def test_run_full_workflow_generates_report(
    pipeline_options,
    scored_dataframe,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generator run executes mocked full loop and writes unified report."""

    fake_molecule = object()
    monkeypatch.setattr(
        generator.Chem,
        "MolFromSmiles",
        lambda smiles, **_kwargs: fake_molecule if smiles in {"CCO", "CCN", "CCC"} else None,
    )

    fake_fp_generator = MagicMock()
    fake_fp_generator.GetFingerprint.side_effect = lambda mol: f"fp-{id(mol)}"
    monkeypatch.setattr(
        generator.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_fp_generator
    )

    monkeypatch.setattr(
        generator,
        "get_binding_pockets_and_residues",
        lambda *_args, **_kwargs: ("Center", "ALA10_A, LYS12_A"),
    )
    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: ["CCO"])
    monkeypatch.setattr(generator, "passes_lipinski", lambda _mol: True)
    monkeypatch.setattr(generator, "get_max_similarity", lambda _smiles, _targets: 0.7)
    monkeypatch.setattr(
        generator, "calculate_heuristic_score", lambda row: float(row["adj_affinity"])
    )

    boltz_client = MagicMock()
    boltz_client.compute_properties = AsyncMock(return_value=(scored_dataframe, []))

    pubchem_service = MagicMock()
    pubchem_service.check_patents = AsyncMock(
        return_value=PatentCheckResult(pubchem_cid=None, identity_patents=0, substructure_patents=1)
    )

    workflow = MoleculeGenerator(
        groq_api_key="g-key",
        boltz_client=boltz_client,
        pubchem_service=pubchem_service,
    )

    report_path = asyncio.run(workflow.run(pipeline_options))

    assert report_path is not None
    report = Path(report_path)
    assert report.exists()

    output = pd.read_csv(report)
    assert list(output["SMILES"]) == ["CCO"]
    assert "PubChem_Known" in output.columns

    boltz_client.compute_properties.assert_awaited_once()
    kwargs = boltz_client.compute_properties.await_args.kwargs
    assert [c.chain_id for c in kwargs["pocket_residues"]] == ["A", "A"]
    assert [c.residue_index for c in kwargs["pocket_residues"]] == [10, 12]
    pubchem_service.check_patents.assert_awaited_once_with("CCO")


def test_run_returns_none_when_no_molecules(
    pipeline_options,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If no valid candidates survive processing, generator returns `None`."""

    no_pocket_options = pipeline_options.model_copy(update={"use_pocket_data": False})

    fake_molecule = object()
    monkeypatch.setattr(
        generator.Chem,
        "MolFromSmiles",
        lambda smiles, **_kwargs: fake_molecule if smiles in {"CCO", "CCN", "CCC"} else None,
    )

    fake_fp_generator = MagicMock()
    fake_fp_generator.GetFingerprint.side_effect = lambda mol: f"fp-{id(mol)}"
    monkeypatch.setattr(
        generator.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_fp_generator
    )

    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: ["CCO"])
    monkeypatch.setattr(generator, "passes_lipinski", lambda _mol: True)

    boltz_client = MagicMock()
    boltz_client.compute_properties = AsyncMock(return_value=(pd.DataFrame(), []))

    workflow = MoleculeGenerator(
        groq_api_key="g-key", boltz_client=boltz_client, pubchem_service=MagicMock()
    )
    assert asyncio.run(workflow.run(no_pocket_options)) is None


def test_run_logs_rejection_summary_when_candidates_are_filtered(
    pipeline_options,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Iteration logs should explain why scored candidates were discarded."""

    no_pocket_options = pipeline_options.model_copy(update={"use_pocket_data": False})

    fake_molecule = object()
    monkeypatch.setattr(
        generator.Chem,
        "MolFromSmiles",
        lambda smiles, **_kwargs: fake_molecule if smiles in {"CCO", "CCN", "CCC"} else None,
    )

    fake_fp_generator = MagicMock()
    fake_fp_generator.GetFingerprint.side_effect = lambda mol: f"fp-{id(mol)}"
    monkeypatch.setattr(
        generator.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_fp_generator
    )

    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: ["CCO"])
    monkeypatch.setattr(generator, "passes_lipinski", lambda _mol: True)
    monkeypatch.setattr(generator, "get_max_similarity", lambda _smiles, _targets: 0.7)
    monkeypatch.setattr(
        generator, "calculate_heuristic_score", lambda row: float(row["adj_affinity"])
    )

    boltz_client = MagicMock()
    boltz_client.compute_properties = AsyncMock(
        return_value=(pd.DataFrame([{"SMILES": "CCO", "Affinity_Prob": 0.5, "SAS": 8.2}]), [])
    )

    workflow = MoleculeGenerator(
        groq_api_key="g-key",
        boltz_client=boltz_client,
        pubchem_service=MagicMock(),
        settings=GeneratorSettings(ADJ_AFFINITY_THRESHOLD=0.6, SAS_SCORE_MAX=6.0),
    )

    with caplog.at_level("INFO"):
        assert asyncio.run(workflow.run(no_pocket_options)) is None

    assert any(
        "Iteration 1/1 complete" in record.message
        and "rejected_sas=1" in record.message
        and "low_affinity=1" in record.message
        for record in caplog.records
    )


def test_run_uses_configured_llm_model_and_temperature(
    pipeline_options,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generation request uses runtime-configured model and temperature values."""

    no_pocket_options = pipeline_options.model_copy(update={"use_pocket_data": False})

    fake_molecule = object()
    monkeypatch.setattr(
        generator.Chem,
        "MolFromSmiles",
        lambda smiles, **_kwargs: fake_molecule if smiles in {"CCO", "CCN", "CCC"} else None,
    )

    fake_fp_generator = MagicMock()
    fake_fp_generator.GetFingerprint.side_effect = lambda mol: f"fp-{id(mol)}"
    monkeypatch.setattr(
        generator.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_fp_generator
    )

    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: [])

    boltz_client = MagicMock()
    boltz_client.compute_properties = AsyncMock(return_value=(pd.DataFrame(), []))

    workflow = MoleculeGenerator(
        groq_api_key="g-key",
        boltz_client=boltz_client,
        pubchem_service=MagicMock(),
        llm_model="llama-test",
        llm_temperature=0.25,
    )
    assert asyncio.run(workflow.run(no_pocket_options)) is None

    create_kwargs = generator_dependencies["groq"].chat.completions.create.call_args.kwargs
    assert create_kwargs["model"] == "llama-test"
    assert create_kwargs["temperature"] == pytest.approx(0.25)


def test_run_falls_back_when_pocket_detection_is_unavailable(
    pipeline_options,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing P2Rank should downgrade the workflow to few-shot mode."""

    fake_molecule = object()
    monkeypatch.setattr(
        generator.Chem,
        "MolFromSmiles",
        lambda smiles, **_kwargs: fake_molecule if smiles in {"CCO", "CCN", "CCC"} else None,
    )

    fake_fp_generator = MagicMock()
    fake_fp_generator.GetFingerprint.side_effect = lambda mol: f"fp-{id(mol)}"
    monkeypatch.setattr(
        generator.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_fp_generator
    )

    monkeypatch.setattr(
        generator,
        "get_binding_pockets_and_residues",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PocketDetectionError("prank not found")),
    )
    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: ["CCO"])
    monkeypatch.setattr(generator, "passes_lipinski", lambda _mol: True)

    boltz_client = MagicMock()
    boltz_client.compute_properties = AsyncMock(return_value=(pd.DataFrame(), []))

    workflow = MoleculeGenerator(
        groq_api_key="g-key",
        boltz_client=boltz_client,
        pubchem_service=MagicMock(),
    )

    assert asyncio.run(workflow.run(pipeline_options)) is None

    boltz_client.compute_properties.assert_awaited_once()
    assert boltz_client.compute_properties.await_args.kwargs["pocket_residues"] is None

    user_message = generator_dependencies["groq"].chat.completions.create.call_args.kwargs[
        "messages"
    ][1]["content"]
    assert "binding pocket containing" not in user_message
    assert "Generate 2 bioisosteres" in user_message


def test_run_saves_best_structure_payload_above_threshold(
    pipeline_options,
    scored_dataframe,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """High-affinity molecules should persist Boltz docked structure metadata."""

    no_pocket_options = pipeline_options.model_copy(update={"use_pocket_data": False})

    fake_molecule = object()
    monkeypatch.setattr(
        generator.Chem,
        "MolFromSmiles",
        lambda smiles, **_kwargs: fake_molecule if smiles in {"CCO", "CCN", "CCC"} else None,
    )

    fake_fp_generator = MagicMock()
    fake_fp_generator.GetFingerprint.side_effect = lambda mol: f"fp-{id(mol)}"
    monkeypatch.setattr(
        generator.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_fp_generator
    )

    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: ["CCO"])
    monkeypatch.setattr(generator, "passes_lipinski", lambda _mol: True)
    monkeypatch.setattr(generator, "get_max_similarity", lambda _smiles, _targets: 0.7)
    monkeypatch.setattr(
        generator, "calculate_heuristic_score", lambda row: float(row["adj_affinity"])
    )

    boltz_client = MagicMock()
    boltz_client.compute_properties = AsyncMock(
        return_value=(
            scored_dataframe,
            [
                generator.BestStructureRecord(
                    candidate_id="pending",
                    smiles="CCO",
                    affinity_prob=0.95,
                    evaluation={"score": 1.2},
                    pdb="ATOM",
                    structure="MOCK-STRUCTURE",
                )
            ],
        )
    )

    pubchem_service = MagicMock()
    pubchem_service.check_patents = AsyncMock(
        return_value=PatentCheckResult(pubchem_cid=None, identity_patents=0, substructure_patents=1)
    )

    workflow = MoleculeGenerator(
        groq_api_key="g-key",
        boltz_client=boltz_client,
        pubchem_service=pubchem_service,
        settings=GeneratorSettings(BEST_STRUCTURE_AFFINITY_THRESHOLD=0.9),
    )

    report_path = asyncio.run(workflow.run(no_pocket_options))

    assert report_path is not None
    metadata_path = Path(no_pocket_options.output_dir) / "best" / "candidate-0001" / "data" / "metadata.json"
    assert metadata_path.exists()

    metadata = metadata_path.read_text(encoding="utf-8")
    assert '"smiles": "CCO"' in metadata
    assert '"structure": "MOCK-STRUCTURE"' in metadata
