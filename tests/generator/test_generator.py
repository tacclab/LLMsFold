"""Tests for molecule generation orchestration."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from src import generator
from src.core.config import GeneratorSettings
from src.core.constants import REFERENCE_BASELINE_FILENAME, RUN_MANIFEST_FILENAME
from src.core.exceptions import PocketDetectionError
from src.diversity import murcko_scaffold
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

    monkeypatch.setattr(
        generator, "get_cached_groq_client", lambda _api_key, **_kwargs: groq_client
    )
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


def test_build_user_prompt_omits_avoid_clause_when_no_negatives() -> None:
    """No AVOID clause is emitted when neither negative pool has entries."""

    prompt = build_user_prompt(False, None, "CCO", 4)
    assert "AVOID" not in prompt


def test_build_user_prompt_includes_avoid_clause_for_each_negative_pool() -> None:
    """Each populated negative pool contributes its own contrastive clause."""

    prompt = build_user_prompt(
        False,
        None,
        "CCO",
        4,
        avoid_hard_to_synthesize="C1=CC=CC=C1C(=O)O",
        avoid_weak_binders="CC(C)C",
    )
    assert "AVOID" in prompt
    assert "too hard to synthesize" in prompt
    assert "C1=CC=CC=C1C(=O)O" in prompt
    assert "bound too weakly" in prompt
    assert "CC(C)C" in prompt


def test_build_user_prompt_omits_already_proposed_clause_when_empty() -> None:
    """No ALREADY PROPOSED clause is emitted when nothing has been scored yet."""

    prompt = build_user_prompt(False, None, "CCO", 4)
    assert "ALREADY PROPOSED" not in prompt


def test_build_user_prompt_includes_already_proposed_clause() -> None:
    """Exact-repeat history is surfaced as a distinct clause from the AVOID one."""

    prompt = build_user_prompt(
        False,
        None,
        "CCO",
        4,
        already_proposed="CCN, CCC",
    )
    assert "ALREADY PROPOSED" in prompt
    assert "do not repeat these exact structures verbatim" in prompt
    assert "CCN, CCC" in prompt


def test_post_process_scores_enriches_and_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Post-processing adds ranking columns and filters high-SAS compounds."""

    source = pd.DataFrame(
        [
            {"SMILES": "CCO", "Boltz2_Binding_Probability": 0.7, "SAS": 4.0},
            {"SMILES": "CCN", "Boltz2_Binding_Probability": 0.5, "SAS": 7.0},
        ]
    )

    monkeypatch.setattr(generator, "get_max_similarity", lambda _smiles, _target_fps: 0.8)
    monkeypatch.setattr(
        generator,
        "calculate_heuristic_score",
        lambda row, **_kwargs: row["Adj_Binding_Probability"] + 0.1,
    )

    scored = MoleculeGenerator._post_process_scores(source, target_fps=["fp"], sas_score_max=6.0)

    assert list(scored["SMILES"]) == ["CCO"]
    assert scored.iloc[0]["Adj_Binding_Probability"] == pytest.approx(0.7)
    assert scored.iloc[0]["score"] == pytest.approx(0.8)


def test_post_process_scores_accepts_custom_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Score thresholds should be overrideable from runtime settings."""

    source = pd.DataFrame(
        [
            {"SMILES": "CCO", "Boltz2_Binding_Probability": 0.7, "SAS": 6.5},
            {"SMILES": "CCN", "Boltz2_Binding_Probability": 0.95, "SAS": 5.5},
        ]
    )

    monkeypatch.setattr(generator, "get_max_similarity", lambda _smiles, _target_fps: 0.8)
    monkeypatch.setattr(
        generator,
        "calculate_heuristic_score",
        lambda row, **_kwargs: row["Adj_Binding_Probability"],
    )

    scored = MoleculeGenerator._post_process_scores(
        source,
        target_fps=["fp"],
        adj_affinity_threshold=0.9,
        sas_score_max=7.0,
    )

    assert list(scored["SMILES"]) == ["CCO", "CCN"]
    assert scored.iloc[0]["Adj_Binding_Probability"] == pytest.approx(0.0)
    assert scored.iloc[1]["Adj_Binding_Probability"] == pytest.approx(0.95)


def test_select_negative_leads_splits_by_failure_mode() -> None:
    """Strong-binder/hard-to-synthesize and weak-binder/easy-to-synthesize rows separate correctly."""

    enriched = pd.DataFrame(
        [
            {"SMILES": "good-both", "Boltz2_Binding_Probability": 0.9, "SAS": 3.0},
            {"SMILES": "hard-synth", "Boltz2_Binding_Probability": 0.85, "SAS": 8.0},
            {"SMILES": "weak-bind", "Boltz2_Binding_Probability": 0.2, "SAS": 2.0},
        ]
    )

    new_hard, new_weak = MoleculeGenerator._select_negative_leads(
        enriched,
        adj_affinity_threshold=0.6,
        sas_score_max=6.0,
        known_hard_to_synthesize=[],
        known_weak_binders=[],
    )

    assert new_hard == ["hard-synth"]
    assert new_weak == ["weak-bind"]


def test_select_negative_leads_excludes_already_known_smiles() -> None:
    """Previously surfaced negative examples are not re-emitted."""

    enriched = pd.DataFrame(
        [
            {"SMILES": "hard-synth", "Boltz2_Binding_Probability": 0.85, "SAS": 8.0},
            {"SMILES": "weak-bind", "Boltz2_Binding_Probability": 0.2, "SAS": 2.0},
        ]
    )

    new_hard, new_weak = MoleculeGenerator._select_negative_leads(
        enriched,
        adj_affinity_threshold=0.6,
        sas_score_max=6.0,
        known_hard_to_synthesize=["hard-synth"],
        known_weak_binders=[],
    )

    assert new_hard == []
    assert new_weak == ["weak-bind"]


def test_select_negative_leads_handles_empty_dataframe() -> None:
    """Empty input yields no negative examples without raising."""

    new_hard, new_weak = MoleculeGenerator._select_negative_leads(
        pd.DataFrame(),
        adj_affinity_threshold=0.6,
        sas_score_max=6.0,
        known_hard_to_synthesize=[],
        known_weak_binders=[],
    )

    assert new_hard == []
    assert new_weak == []


def test_run_full_workflow_generates_report(
    pipeline_options,
    scored_dataframe,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generator run executes mocked full loop and writes unified report.

    Uses PDB residue numbers (210, 498) that differ from their mapped sequence
    positions (10, 12) to prove pocket contacts are translated through
    `residue_position_map` rather than sent as raw PDB numbers (the A1 bug).
    """

    mapped_options = pipeline_options.model_copy(
        update={"residue_position_map": {210: 10, 498: 12}}
    )

    monkeypatch.setattr(
        generator.Chem,
        "MolFromSmiles",
        lambda smiles, **_kwargs: smiles if smiles in {"CCO", "CCN", "CCC"} else None,
    )
    monkeypatch.setattr(generator.Chem, "MolToSmiles", lambda mol: mol)

    fake_fp_generator = MagicMock()
    fake_fp_generator.GetFingerprint.side_effect = lambda mol: f"fp-{id(mol)}"
    monkeypatch.setattr(
        generator.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_fp_generator
    )

    monkeypatch.setattr(
        generator,
        "get_binding_pockets_and_residues",
        lambda *_args, **_kwargs: ("Center", "ALA210_A, LYS498_A", None),
    )
    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: (["CCO"], 0))
    monkeypatch.setattr(generator, "passes_lipinski", lambda _mol: True)
    monkeypatch.setattr(generator, "get_max_similarity", lambda _smiles, _targets: 0.7)
    monkeypatch.setattr(
        generator,
        "calculate_heuristic_score",
        lambda row, **_kwargs: float(row["Adj_Binding_Probability"]),
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

    report_path = asyncio.run(workflow.run(mapped_options))

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


def test_run_drops_pocket_contacts_outside_target_chain(
    pipeline_options,
    scored_dataframe,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pocket residues detected on a chain other than the target are dropped.

    Regression test for a bug where the target's own chain (e.g. "C") was
    silently discarded because contacts were compared against a hardcoded
    "A" instead of the chain the submitted sequence actually came from.
    """

    chain_c_options = pipeline_options.model_copy(
        update={"target_chain_id": "C", "residue_position_map": {10: 5}}
    )

    monkeypatch.setattr(
        generator.Chem,
        "MolFromSmiles",
        lambda smiles, **_kwargs: smiles if smiles in {"CCO", "CCN", "CCC"} else None,
    )
    monkeypatch.setattr(generator.Chem, "MolToSmiles", lambda mol: mol)

    fake_fp_generator = MagicMock()
    fake_fp_generator.GetFingerprint.side_effect = lambda mol: f"fp-{id(mol)}"
    monkeypatch.setattr(
        generator.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_fp_generator
    )

    monkeypatch.setattr(
        generator,
        "get_binding_pockets_and_residues",
        lambda *_args, **_kwargs: ("Center", "ALA10_C, LYS12_A", None),
    )
    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: (["CCO"], 0))
    monkeypatch.setattr(generator, "passes_lipinski", lambda _mol: True)
    monkeypatch.setattr(generator, "get_max_similarity", lambda _smiles, _targets: 0.7)
    monkeypatch.setattr(
        generator,
        "calculate_heuristic_score",
        lambda row, **_kwargs: float(row["Adj_Binding_Probability"]),
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

    asyncio.run(workflow.run(chain_c_options))

    boltz_client.compute_properties.assert_awaited_once()
    kwargs = boltz_client.compute_properties.await_args.kwargs
    # Only the residue detected on the target chain "C" (PDB number 10, mapped to
    # sequence position 5) survives; LYS12 on chain "A" is dropped since it isn't
    # part of the submitted sequence.
    assert [c.residue_index for c in kwargs["pocket_residues"]] == [5]


def test_run_reuses_cached_boltz_score_across_iterations(
    pipeline_options,
    scored_dataframe,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SMILES already scored in an earlier iteration is not resubmitted to Boltz."""

    two_iteration_options = pipeline_options.model_copy(
        update={"max_iterations": 2, "use_pocket_data": False}
    )

    monkeypatch.setattr(
        generator.Chem,
        "MolFromSmiles",
        lambda smiles, **_kwargs: smiles if smiles == "CCO" else None,
    )
    monkeypatch.setattr(generator.Chem, "MolToSmiles", lambda mol: mol)

    fake_fp_generator = MagicMock()
    fake_fp_generator.GetFingerprint.side_effect = lambda mol: f"fp-{id(mol)}"
    monkeypatch.setattr(
        generator.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_fp_generator
    )

    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: (["CCO"], 0))
    monkeypatch.setattr(generator, "passes_lipinski", lambda _mol: True)
    monkeypatch.setattr(generator, "get_max_similarity", lambda _smiles, _targets: 0.7)
    monkeypatch.setattr(
        generator,
        "calculate_heuristic_score",
        lambda row, **_kwargs: float(row["Adj_Binding_Probability"]),
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

    asyncio.run(workflow.run(two_iteration_options))

    assert boltz_client.compute_properties.await_count == 2
    first_call_smiles = boltz_client.compute_properties.await_args_list[0].args[0]
    second_call_smiles = boltz_client.compute_properties.await_args_list[1].args[0]
    assert first_call_smiles == ["CCO"]
    assert second_call_smiles == []


def test_run_similarity_penalty_pool_grows_with_registry(
    pipeline_options,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The MaxSim/similarity-penalty comparison pool grows each iteration.

    Regression test for the paper's Eq. 3 ("global registry G"): the pool
    passed to `get_max_similarity` must include molecules registered in
    earlier iterations, not stay fixed at the initial seed set.
    """

    two_iteration_options = pipeline_options.model_copy(
        update={"max_iterations": 2, "use_pocket_data": False}
    )

    monkeypatch.setattr(
        generator.Chem,
        "MolFromSmiles",
        lambda smiles, **_kwargs: smiles if smiles in {"MOL1", "MOL2"} else None,
    )
    monkeypatch.setattr(generator.Chem, "MolToSmiles", lambda mol: mol)

    fake_fp_generator = MagicMock()
    fake_fp_generator.GetFingerprint.side_effect = lambda mol: f"fp-{id(mol)}"
    monkeypatch.setattr(
        generator.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_fp_generator
    )

    proposals = iter([(["MOL1"], 0), (["MOL2"], 0)])
    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: next(proposals))
    monkeypatch.setattr(generator, "passes_lipinski", lambda _mol: True)
    monkeypatch.setattr(
        generator,
        "calculate_heuristic_score",
        lambda row, **_kwargs: float(row["Adj_Binding_Probability"]),
    )

    registry_sizes_seen: list[int] = []

    def fake_get_max_similarity(_smiles: str, fps: list) -> float:
        registry_sizes_seen.append(len(fps))
        return 0.5

    monkeypatch.setattr(generator, "get_max_similarity", fake_get_max_similarity)

    async def fake_compute_properties(smiles_list, _sequence, **_kwargs):
        smiles = smiles_list[0]
        df = pd.DataFrame(
            [
                {
                    "SMILES": smiles,
                    "Boltz2_pTM": 0.9,
                    "Boltz2_ipTM": 0.98,
                    "Boltz2_Confidence": 0.7,
                    "Boltz2_Complex_pLDDT": 0.95,
                    "Boltz2_Binding_Probability": 0.95,
                    "Boltz2_Predicted_Affinity_pIC50": 7.0,
                    "Boltz2_Predicted_Affinity_uM": 0.1,
                    "MolWt": 46.07,
                    "LogP": 0.2,
                    "QED": 0.5,
                    "SAS": 3.1,
                    "TPSA": 20.2,
                    "H_Acceptors": 1,
                    "H_Donors": 1,
                    "Rotatable_Bonds": 0,
                }
            ]
        )
        return df, []

    boltz_client = MagicMock()
    boltz_client.compute_properties = fake_compute_properties

    pubchem_service = MagicMock()
    pubchem_service.check_patents = AsyncMock(
        return_value=PatentCheckResult(pubchem_cid=None, identity_patents=0, substructure_patents=1)
    )

    workflow = MoleculeGenerator(
        groq_api_key="g-key",
        boltz_client=boltz_client,
        pubchem_service=pubchem_service,
    )

    asyncio.run(workflow.run(two_iteration_options))

    assert len(registry_sizes_seen) == 2
    # Iteration 2's pool must be larger than iteration 1's: MOL1, registered
    # after iteration 1, is now part of the comparison pool.
    assert registry_sizes_seen[1] == registry_sizes_seen[0] + 1


def test_run_returns_none_when_no_molecules(
    pipeline_options,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If no valid candidates survive processing, generator returns `None`."""

    no_pocket_options = pipeline_options.model_copy(update={"use_pocket_data": False})

    monkeypatch.setattr(
        generator.Chem,
        "MolFromSmiles",
        lambda smiles, **_kwargs: smiles if smiles in {"CCO", "CCN", "CCC"} else None,
    )
    monkeypatch.setattr(generator.Chem, "MolToSmiles", lambda mol: mol)

    fake_fp_generator = MagicMock()
    fake_fp_generator.GetFingerprint.side_effect = lambda mol: f"fp-{id(mol)}"
    monkeypatch.setattr(
        generator.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_fp_generator
    )

    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: (["CCO"], 0))
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

    monkeypatch.setattr(
        generator.Chem,
        "MolFromSmiles",
        lambda smiles, **_kwargs: smiles if smiles in {"CCO", "CCN", "CCC"} else None,
    )
    monkeypatch.setattr(generator.Chem, "MolToSmiles", lambda mol: mol)

    fake_fp_generator = MagicMock()
    fake_fp_generator.GetFingerprint.side_effect = lambda mol: f"fp-{id(mol)}"
    monkeypatch.setattr(
        generator.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_fp_generator
    )

    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: (["CCO"], 0))
    monkeypatch.setattr(generator, "passes_lipinski", lambda _mol: True)
    monkeypatch.setattr(generator, "get_max_similarity", lambda _smiles, _targets: 0.7)
    monkeypatch.setattr(
        generator,
        "calculate_heuristic_score",
        lambda row, **_kwargs: float(row["Adj_Binding_Probability"]),
    )

    boltz_client = MagicMock()
    boltz_client.compute_properties = AsyncMock(
        return_value=(
            pd.DataFrame(
                [
                    {
                        "SMILES": "CCO",
                        "Boltz2_Binding_Probability": 0.5,
                        "SAS": 8.2,
                        "Boltz2_ipTM": 0.8,
                        "Boltz2_Complex_pLDDT": 0.7,
                    }
                ]
            ),
            [],
        )
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


def test_run_filters_candidates_below_iptm_and_plddt_thresholds(
    pipeline_options,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Candidates passing affinity/SAS but failing ipTM or pLDDT are dropped."""

    no_pocket_options = pipeline_options.model_copy(update={"use_pocket_data": False})

    monkeypatch.setattr(
        generator.Chem,
        "MolFromSmiles",
        lambda smiles, **_kwargs: smiles if smiles in {"CCO", "CCN", "CCC"} else None,
    )
    monkeypatch.setattr(generator.Chem, "MolToSmiles", lambda mol: mol)

    fake_fp_generator = MagicMock()
    fake_fp_generator.GetFingerprint.side_effect = lambda mol: f"fp-{id(mol)}"
    monkeypatch.setattr(
        generator.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_fp_generator
    )

    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: (["CCO"], 0))
    monkeypatch.setattr(generator, "passes_lipinski", lambda _mol: True)
    monkeypatch.setattr(generator, "get_max_similarity", lambda _smiles, _targets: 0.7)
    monkeypatch.setattr(
        generator,
        "calculate_heuristic_score",
        lambda row, **_kwargs: float(row["Adj_Binding_Probability"]),
    )

    boltz_client = MagicMock()
    boltz_client.compute_properties = AsyncMock(
        return_value=(
            pd.DataFrame(
                [
                    {
                        "SMILES": "CCO",
                        "Boltz2_Binding_Probability": 0.9,
                        "SAS": 3.0,
                        "Boltz2_ipTM": 0.2,
                        "Boltz2_Complex_pLDDT": 0.1,
                    }
                ]
            ),
            [],
        )
    )

    workflow = MoleculeGenerator(
        groq_api_key="g-key",
        boltz_client=boltz_client,
        pubchem_service=MagicMock(),
        settings=GeneratorSettings(
            ADJ_AFFINITY_THRESHOLD=0.6, SAS_SCORE_MAX=6.0, IPTM_THRESHOLD=0.5, PLDDT_THRESHOLD=0.5
        ),
    )

    with caplog.at_level("INFO"):
        assert asyncio.run(workflow.run(no_pocket_options)) is None

    assert any(
        "Iteration 1/1 complete" in record.message
        and "low_iptm=1" in record.message
        and "low_plddt=1" in record.message
        for record in caplog.records
    )


def test_run_uses_configured_llm_model_and_temperature(
    pipeline_options,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generation request uses runtime-configured model and temperature values."""

    no_pocket_options = pipeline_options.model_copy(update={"use_pocket_data": False})

    monkeypatch.setattr(
        generator.Chem,
        "MolFromSmiles",
        lambda smiles, **_kwargs: smiles if smiles in {"CCO", "CCN", "CCC"} else None,
    )
    monkeypatch.setattr(generator.Chem, "MolToSmiles", lambda mol: mol)

    fake_fp_generator = MagicMock()
    fake_fp_generator.GetFingerprint.side_effect = lambda mol: f"fp-{id(mol)}"
    monkeypatch.setattr(
        generator.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_fp_generator
    )

    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: ([], 0))

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


@pytest.mark.parametrize(
    ("configured_seed", "expect_seed_kwarg"),
    [(None, False), (7, True)],
)
def test_run_passes_configured_seed_to_groq_and_rng(
    pipeline_options,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
    configured_seed: int | None,
    expect_seed_kwarg: bool,
) -> None:
    """A configured RANDOM_SEED reaches both the Groq call and Python/NumPy RNGs."""

    no_pocket_options = pipeline_options.model_copy(update={"use_pocket_data": False})

    monkeypatch.setattr(
        generator.Chem,
        "MolFromSmiles",
        lambda smiles, **_kwargs: smiles if smiles in {"CCO", "CCN", "CCC"} else None,
    )
    monkeypatch.setattr(generator.Chem, "MolToSmiles", lambda mol: mol)

    fake_fp_generator = MagicMock()
    fake_fp_generator.GetFingerprint.side_effect = lambda mol: f"fp-{id(mol)}"
    monkeypatch.setattr(
        generator.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_fp_generator
    )

    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: ([], 0))

    random_seed_mock = MagicMock()
    np_seed_mock = MagicMock()
    monkeypatch.setattr(generator.random, "seed", random_seed_mock)
    monkeypatch.setattr(generator.np.random, "seed", np_seed_mock)

    boltz_client = MagicMock()
    boltz_client.compute_properties = AsyncMock(return_value=(pd.DataFrame(), []))

    workflow = MoleculeGenerator(
        groq_api_key="g-key",
        boltz_client=boltz_client,
        pubchem_service=MagicMock(),
        settings=GeneratorSettings(RANDOM_SEED=configured_seed),
    )
    assert asyncio.run(workflow.run(no_pocket_options)) is None

    create_kwargs = generator_dependencies["groq"].chat.completions.create.call_args.kwargs
    assert ("seed" in create_kwargs) is expect_seed_kwarg
    if expect_seed_kwarg:
        assert create_kwargs["seed"] == configured_seed
        random_seed_mock.assert_called_once_with(configured_seed)
        np_seed_mock.assert_called_once_with(configured_seed)
    else:
        random_seed_mock.assert_not_called()
        np_seed_mock.assert_not_called()


def test_run_falls_back_when_pocket_detection_is_unavailable(
    pipeline_options,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing P2Rank should downgrade the workflow to few-shot mode."""

    monkeypatch.setattr(
        generator.Chem,
        "MolFromSmiles",
        lambda smiles, **_kwargs: smiles if smiles in {"CCO", "CCN", "CCC"} else None,
    )
    monkeypatch.setattr(generator.Chem, "MolToSmiles", lambda mol: mol)

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
    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: (["CCO"], 0))
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

    monkeypatch.setattr(
        generator.Chem,
        "MolFromSmiles",
        lambda smiles, **_kwargs: smiles if smiles in {"CCO", "CCN", "CCC"} else None,
    )
    monkeypatch.setattr(generator.Chem, "MolToSmiles", lambda mol: mol)

    fake_fp_generator = MagicMock()
    fake_fp_generator.GetFingerprint.side_effect = lambda mol: f"fp-{id(mol)}"
    monkeypatch.setattr(
        generator.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_fp_generator
    )

    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: (["CCO"], 0))
    monkeypatch.setattr(generator, "passes_lipinski", lambda _mol: True)
    monkeypatch.setattr(generator, "get_max_similarity", lambda _smiles, _targets: 0.7)
    monkeypatch.setattr(
        generator,
        "calculate_heuristic_score",
        lambda row, **_kwargs: float(row["Adj_Binding_Probability"]),
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
                    structure="MOCK-MMCIF-CONTENT",
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
    data_dir = Path(no_pocket_options.output_dir) / "best" / "candidate-0001" / "data"
    metadata_path = data_dir / "metadata.json"
    cif_path = data_dir / "structure.cif"
    assert metadata_path.exists()
    assert cif_path.exists()

    metadata = metadata_path.read_text(encoding="utf-8")
    assert '"smiles": "CCO"' in metadata
    assert cif_path.read_text(encoding="utf-8") == "MOCK-MMCIF-CONTENT"


# --- Reference inhibitor baseline and run manifest -------------------------


def _boltz_row(smiles: str, **overrides) -> dict:
    """Builds a Boltz result row carrying every `MoleculeRecord` field."""

    row = {
        "SMILES": smiles,
        "Boltz2_pTM": 0.9,
        "Boltz2_ipTM": 0.98,
        "Boltz2_Confidence": 0.7,
        "Boltz2_Complex_pLDDT": 0.95,
        "Boltz2_Binding_Probability": 0.8,
        "Boltz2_Predicted_Affinity_pIC50": 7.0,
        "Boltz2_Predicted_Affinity_uM": 0.1,
        "MolWt": 46.07,
        "LogP": 0.2,
        "QED": 0.5,
        "SAS": 3.1,
        "TPSA": 20.2,
        "H_Acceptors": 1,
        "H_Donors": 1,
        "Rotatable_Bonds": 0,
    }
    row.update(overrides)
    return row


def _fixed_thresholds() -> GeneratorSettings:
    """Pins the scoring gates so tests do not depend on a local `.env`."""

    return GeneratorSettings(
        adj_affinity_threshold=0.6,
        sas_score_max=6.0,
        iptm_threshold=0.95,
        plddt_threshold=0.9,
    )


def test_score_reference_molecules_builds_leave_one_out_baseline(
    pipeline_options,
    generator_dependencies,
) -> None:
    """References are scored on the candidate path with leave-one-out similarity.

    A reference compared against the full seed set would always match itself
    at 1.0, which is not the quantity generated candidates are measured on,
    so the baseline must exclude each reference from its own comparison pool.
    """

    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    reference_smiles = ["CCO", "CCN"]
    reference_fps = [
        fingerprint_generator.GetFingerprint(Chem.MolFromSmiles(smiles))
        for smiles in reference_smiles
    ]

    boltz_client = MagicMock()
    boltz_client.compute_properties = AsyncMock(
        return_value=(
            pd.DataFrame(
                [
                    _boltz_row("CCO"),
                    _boltz_row("CCN", Boltz2_Binding_Probability=0.3, SAS=7.5),
                ]
            ),
            [],
        )
    )

    workflow = MoleculeGenerator(
        groq_api_key="g-key",
        boltz_client=boltz_client,
        pubchem_service=MagicMock(),
        settings=_fixed_thresholds(),
    )

    baseline = asyncio.run(
        workflow._score_reference_molecules(
            reference_smiles,
            ["Ibrutinib", "Venetoclax"],
            reference_fps,
            pipeline_options,
            None,
        )
    )

    assert list(baseline["Reference_Name"]) == ["Ibrutinib", "Venetoclax"]
    assert set(baseline["Role"]) == {"reference"}
    # Self-match excluded, so neither reference reports perfect similarity.
    assert all(0.0 <= value < 1.0 for value in baseline["MaxSim_To_Other_References"])

    passing, failing = baseline.iloc[0], baseline.iloc[1]
    assert bool(passing["Passes_All_Gates"]) is True
    assert bool(failing["Passes_Affinity_Gate"]) is False
    assert bool(failing["Passes_SAS_Gate"]) is False
    assert bool(failing["Passes_All_Gates"]) is False
    # An over-the-ceiling SAS clamps the synthesizability term rather than
    # going negative and dragging the composite score below zero.
    assert failing["synth_factor"] == pytest.approx(0.0)
    assert failing["score"] == pytest.approx(0.0)


def test_score_reference_molecules_returns_empty_when_boltz_returns_nothing(
    pipeline_options,
    generator_dependencies,
) -> None:
    """A failed reference scoring call yields an empty frame, not an exception."""

    boltz_client = MagicMock()
    boltz_client.compute_properties = AsyncMock(return_value=(pd.DataFrame(), []))

    workflow = MoleculeGenerator(
        groq_api_key="g-key",
        boltz_client=boltz_client,
        pubchem_service=MagicMock(),
        settings=_fixed_thresholds(),
    )

    baseline = asyncio.run(
        workflow._score_reference_molecules(["CCO"], ["seed"], [None], pipeline_options, None)
    )

    assert baseline.empty


def _patch_rdkit_for_run(monkeypatch: pytest.MonkeyPatch, accepted: set[str]) -> None:
    """Replaces RDKit lookups with string stand-ins for whole-run tests."""

    monkeypatch.setattr(
        generator.Chem,
        "MolFromSmiles",
        lambda smiles, **_kwargs: smiles if smiles in accepted else None,
    )
    monkeypatch.setattr(generator.Chem, "MolToSmiles", lambda mol: mol)

    fake_fp_generator = MagicMock()
    fake_fp_generator.GetFingerprint.side_effect = lambda mol: f"fp-{mol}"
    monkeypatch.setattr(
        generator.rdFingerprintGenerator, "GetMorganGenerator", lambda radius: fake_fp_generator
    )
    monkeypatch.setattr(
        generator,
        "DataStructs",
        SimpleNamespace(
            BulkTanimotoSimilarity=lambda _fp, others: [0.3] * len(others),
            TanimotoSimilarity=lambda _a, _b: 0.3,
        ),
    )


def test_run_writes_reference_baseline_and_records_it_in_manifest(
    pipeline_options,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--score-references` produces a baseline report the manifest points at."""

    options = pipeline_options.model_copy(
        update={"use_pocket_data": False, "score_references": True}
    )

    _patch_rdkit_for_run(monkeypatch, {"CCO", "CCN", "CCC"})
    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: (["CCO"], 0))
    monkeypatch.setattr(generator, "passes_lipinski", lambda _mol: True)
    monkeypatch.setattr(generator, "get_max_similarity", lambda _smiles, _targets: 0.7)

    async def fake_compute_properties(smiles_list, _sequence, **_kwargs):
        return pd.DataFrame([_boltz_row(smiles) for smiles in smiles_list]), []

    boltz_client = MagicMock()
    boltz_client.compute_properties = fake_compute_properties

    pubchem_service = MagicMock()
    pubchem_service.check_patents = AsyncMock(return_value=PatentCheckResult(pubchem_cid=None))

    workflow = MoleculeGenerator(
        groq_api_key="g-key",
        boltz_client=boltz_client,
        pubchem_service=pubchem_service,
        settings=_fixed_thresholds(),
    )

    asyncio.run(workflow.run(options))

    baseline_path = Path(options.output_dir) / REFERENCE_BASELINE_FILENAME
    assert baseline_path.exists()
    baseline = pd.read_csv(baseline_path)
    assert list(baseline["SMILES"]) == ["CCO", "CCN", "CCC"]
    # No name column in the seed CSV, so positional fallback names are used.
    assert list(baseline["Reference_Name"]) == ["reference-01", "reference-02", "reference-03"]

    manifest = json.loads(
        (Path(options.output_dir) / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["reference_baseline"]["scored"] == 3
    assert manifest["reference_baseline"]["passing_all_gates"] == 3
    assert manifest["reference_baseline"]["report"] == REFERENCE_BASELINE_FILENAME


def test_run_manifest_records_thresholds_inputs_and_outcome(
    pipeline_options,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every gate that decided the report is recorded beside the report.

    Without this, two runs producing identically-shaped CSVs cannot be told
    apart, since all four scoring gates are environment-configurable.
    """

    options = pipeline_options.model_copy(update={"use_pocket_data": False})

    _patch_rdkit_for_run(monkeypatch, {"CCO", "CCN", "CCC"})
    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: (["CCO"], 0))
    monkeypatch.setattr(generator, "passes_lipinski", lambda _mol: True)
    monkeypatch.setattr(generator, "get_max_similarity", lambda _smiles, _targets: 0.7)

    boltz_client = MagicMock()
    boltz_client.compute_properties = AsyncMock(
        return_value=(pd.DataFrame([_boltz_row("CCO")]), [])
    )

    pubchem_service = MagicMock()
    pubchem_service.check_patents = AsyncMock(return_value=PatentCheckResult(pubchem_cid=None))

    workflow = MoleculeGenerator(
        groq_api_key="g-key",
        boltz_client=boltz_client,
        pubchem_service=pubchem_service,
        settings=GeneratorSettings(adj_affinity_threshold=0.42, sas_score_max=6.0),
    )

    asyncio.run(workflow.run(options))

    manifest = json.loads(
        (Path(options.output_dir) / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )

    assert manifest["status"] == "completed"
    assert manifest["thresholds"]["adj_affinity_threshold"] == 0.42
    assert manifest["thresholds"]["similarity_penalty_threshold"] == 0.6
    # The seed CSV exists and is digested; the fixture PDB does not, and that
    # degrades to null rather than failing the run.
    assert manifest["inputs"]["few_shot_csv_sha256"] is not None
    assert manifest["inputs"]["pdb_sha256"] is None
    assert manifest["inputs"]["seed_molecule_count"] == 3
    assert manifest["pocket"]["mode"] == "disabled"
    assert manifest["results"]["iterations_succeeded"] == 1
    assert manifest["results"]["iterations_failed"] == 0
    assert manifest["elapsed_seconds"] is not None


def test_run_manifest_records_failed_iterations(
    pipeline_options,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A swallowed iteration failure is still visible in the manifest.

    `_run_pipeline` catches per-iteration exceptions so one bad round does
    not abort a long run; without this record a run reporting N iterations
    could silently have completed none.
    """

    options = pipeline_options.model_copy(update={"use_pocket_data": False, "max_iterations": 2})

    _patch_rdkit_for_run(monkeypatch, {"CCO", "CCN", "CCC"})
    generator_dependencies["groq"].chat.completions.create.side_effect = RuntimeError("llm down")

    workflow = MoleculeGenerator(
        groq_api_key="g-key",
        boltz_client=MagicMock(),
        pubchem_service=MagicMock(),
        settings=_fixed_thresholds(),
    )

    assert asyncio.run(workflow.run(options)) is None

    manifest = json.loads(
        (Path(options.output_dir) / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )

    assert manifest["status"] == "no_candidates"
    assert [record["status"] for record in manifest["iterations"]] == ["failed", "failed"]
    assert "RuntimeError: llm down" in manifest["iterations"][0]["error"]


def test_run_annotates_scaffolds_and_summarizes_diversity(
    tmp_path: Path,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report rows carry scaffold columns and the manifest carries the summary.

    Runs against real RDKit rather than string stand-ins, so the scaffold
    novelty flags reflect genuine ring-system comparisons: one candidate
    re-decorates a seed core and one introduces a new one.
    """

    seed_csv = tmp_path / "seeds.csv"
    pd.DataFrame({"Smiles": ["c1ccccc1C(=O)O", "c1ccc2[nH]ccc2c1"]}).to_csv(
        seed_csv, sep=";", index=False
    )
    options = generator.PipelineOptions(
        pdb_path=tmp_path / "protein.pdb",
        few_shot_csv=seed_csv,
        output_dir=tmp_path / "results",
        protein_sequence="MKT",
        max_iterations=1,
        max_samples=2,
        use_pocket_data=False,
    )

    reused_core = "c1ccccc1C(=O)N"  # benzamide: same Murcko core as seed 1
    novel_core = "c1ccc2occc2c1"  # benzofuran: a ring system not in the seeds
    monkeypatch.setattr(
        generator, "parse_smiles_from_text", lambda _raw: ([reused_core, novel_core], 0)
    )

    async def fake_compute_properties(smiles_list, _sequence, **_kwargs):
        return pd.DataFrame([_boltz_row(smiles) for smiles in smiles_list]), []

    boltz_client = MagicMock()
    boltz_client.compute_properties = fake_compute_properties

    pubchem_service = MagicMock()
    pubchem_service.check_patents = AsyncMock(return_value=PatentCheckResult(pubchem_cid=None))

    workflow = MoleculeGenerator(
        groq_api_key="g-key",
        boltz_client=boltz_client,
        pubchem_service=pubchem_service,
        settings=_fixed_thresholds(),
    )

    report_path = asyncio.run(workflow.run(options))

    assert report_path is not None
    report = pd.read_csv(report_path).set_index("SMILES")
    # Report rows are keyed by the canonical SMILES the pipeline stores.
    canonical = {
        smiles: Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in (reused_core, novel_core)
    }

    assert not report.loc[canonical[reused_core], "Scaffold_Is_Novel"]
    assert report.loc[canonical[novel_core], "Scaffold_Is_Novel"]
    assert report.loc[canonical[reused_core], "Max_Scaffold_Sim_To_Seeds"] == pytest.approx(1.0)
    assert report.loc[canonical[reused_core], "Murcko_Scaffold"] == murcko_scaffold(
        "c1ccccc1C(=O)O"
    )

    manifest = json.loads(
        (Path(options.output_dir) / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["diversity"]["molecules"] == 2
    assert manifest["diversity"]["unique_murcko_scaffolds"] == 2
    assert manifest["diversity"]["seed_scaffolds"] == 2
    assert manifest["diversity"]["molecules_on_seed_scaffolds"] == 1
    assert manifest["diversity"]["novel_scaffold_fraction"] == pytest.approx(0.5)


# --- Holdout evaluation ----------------------------------------------------


class _StubHoldoutEvaluator:
    """Records when it ran, so call ordering can be asserted."""

    def __init__(self, name: str, columns: dict | None = None, available: bool = True) -> None:
        self.name = name
        self._columns = columns or {}
        self._available = available
        self.calls: list[int] = []

    def availability(self):
        from src.evaluators import Availability

        return Availability(
            available=self._available,
            reason=None if self._available else f"{self.name} deliberately unavailable",
        )

    def evaluate(self, candidates, context):
        self.calls.append(len(candidates))
        return {candidate.smiles: dict(self._columns) for candidate in candidates}


def _holdout_options(tmp_path: Path, seed_csv: Path):
    """Builds pocket-free options pointing at a temporary seed file."""

    return generator.PipelineOptions(
        pdb_path=tmp_path / "protein.pdb",
        few_shot_csv=seed_csv,
        output_dir=tmp_path / "results",
        protein_sequence="MKT",
        max_iterations=1,
        max_samples=2,
        use_pocket_data=False,
    )


def _seed_csv(tmp_path: Path) -> Path:
    """Writes a minimal seed file for holdout tests."""

    path = tmp_path / "seeds.csv"
    pd.DataFrame({"Smiles": ["CCO", "CCN"]}).to_csv(path, sep=";", index=False)
    return path


def _holdout_workflow(monkeypatch, evaluators, proposals=("CCO",)):
    """Wires a generator whose Boltz calls and RDKit lookups are stubbed."""

    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: (list(proposals), 0))

    async def fake_compute_properties(smiles_list, _sequence, **_kwargs):
        return pd.DataFrame([_boltz_row(smiles) for smiles in smiles_list]), []

    boltz_client = MagicMock()
    boltz_client.compute_properties = fake_compute_properties

    pubchem_service = MagicMock()
    pubchem_service.check_patents = AsyncMock(return_value=PatentCheckResult(pubchem_cid=None))

    return MoleculeGenerator(
        groq_api_key="g-key",
        boltz_client=boltz_client,
        pubchem_service=pubchem_service,
        settings=_fixed_thresholds(),
        holdout_evaluators=evaluators,
    )


def test_run_writes_holdout_columns_and_records_evaluators(
    tmp_path: Path,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Holdout columns reach the report and the manifest names what produced them."""

    options = _holdout_options(tmp_path, _seed_csv(tmp_path))
    evaluator = _StubHoldoutEvaluator(
        "posebusters", {"PoseBusters_Pass": True, "PoseBusters_Checks_Passed": 22}
    )
    workflow = _holdout_workflow(monkeypatch, [evaluator])

    report_path = asyncio.run(workflow.run(options))

    report = pd.read_csv(report_path)
    assert bool(report.loc[0, "PoseBusters_Pass"]) is True
    assert int(report.loc[0, "PoseBusters_Checks_Passed"]) == 22

    manifest = json.loads(
        (Path(options.output_dir) / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["holdout"]["enabled"] is True
    assert manifest["holdout"]["evaluators"]["posebusters"]["available"] is True
    assert manifest["holdout"]["evaluators"]["posebusters"]["evaluated"] == 1


def test_run_records_unavailable_holdout_evaluators_with_reasons(
    tmp_path: Path,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing independent check is recorded, not silently skipped.

    A reader must be able to tell "this pose passed validation" from "nobody
    validated this pose", which an absent column alone does not convey.
    """

    options = _holdout_options(tmp_path, _seed_csv(tmp_path))
    evaluator = _StubHoldoutEvaluator("vina", available=False)
    workflow = _holdout_workflow(monkeypatch, [evaluator])

    asyncio.run(workflow.run(options))

    manifest = json.loads(
        (Path(options.output_dir) / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    record = manifest["holdout"]["evaluators"]["vina"]
    assert record["available"] is False
    assert "deliberately unavailable" in record["reason"]
    assert record["evaluated"] == 0
    assert evaluator.calls == []


def test_run_survives_a_failing_holdout_evaluator(
    tmp_path: Path,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crashing evaluator is recorded and the paid-for report still lands."""

    options = _holdout_options(tmp_path, _seed_csv(tmp_path))

    class _Exploding(_StubHoldoutEvaluator):
        def evaluate(self, candidates, context):
            raise RuntimeError("evaluator exploded")

    workflow = _holdout_workflow(monkeypatch, [_Exploding("posebusters")])

    report_path = asyncio.run(workflow.run(options))

    assert report_path is not None and Path(report_path).exists()
    manifest = json.loads(
        (Path(options.output_dir) / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert (
        "RuntimeError: evaluator exploded"
        in manifest["holdout"]["evaluators"]["posebusters"]["error"]
    )


def test_holdout_evaluation_cannot_influence_generation(
    tmp_path: Path,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every prompt is sent before any holdout score exists.

    This is the property the independence claim rests on: if a holdout score
    could reach the prompt, the lead pool or the similarity registry, the
    evaluation would be measuring its own influence on what was generated.
    """

    options = _holdout_options(tmp_path, _seed_csv(tmp_path)).model_copy(
        update={"max_iterations": 3}
    )

    call_order: list[str] = []
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content="['CCO']"))]

    def record_llm_call(**_kwargs):
        call_order.append("llm")
        return completion

    generator_dependencies["groq"].chat.completions.create.side_effect = record_llm_call

    class _Recording(_StubHoldoutEvaluator):
        def evaluate(self, candidates, context):
            call_order.append("holdout")
            return super().evaluate(candidates, context)

    evaluator = _Recording("posebusters", {"PoseBusters_Pass": True})
    workflow = _holdout_workflow(monkeypatch, [evaluator])

    asyncio.run(workflow.run(options))

    assert call_order.count("llm") == 3
    assert call_order.count("holdout") == 1
    # The holdout evaluator runs only once the last prompt has been sent.
    assert call_order.index("holdout") == len(call_order) - 1


def test_holdout_evaluation_can_be_disabled(
    tmp_path: Path,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turning holdout evaluation off is recorded so reports stay interpretable."""

    options = _holdout_options(tmp_path, _seed_csv(tmp_path))
    evaluator = _StubHoldoutEvaluator("posebusters", {"PoseBusters_Pass": True})

    workflow = _holdout_workflow(monkeypatch, [evaluator])
    monkeypatch.setattr(workflow._settings, "enable_holdout_evaluation", False, raising=False)

    asyncio.run(workflow.run(options))

    manifest = json.loads(
        (Path(options.output_dir) / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["holdout"] == {"enabled": False}
    assert evaluator.calls == []


# --- User-specified pockets ------------------------------------------------


def test_user_pocket_is_resolved_instead_of_detection(
    tmp_path: Path,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A named pocket source replaces geometric detection entirely."""

    options = _holdout_options(tmp_path, _seed_csv(tmp_path)).model_copy(
        update={"use_pocket_data": True, "pocket_ligand": "ATP", "residue_position_map": {10: 5}}
    )

    detect = MagicMock()
    monkeypatch.setattr(generator, "get_binding_pockets_and_residues", detect)
    monkeypatch.setattr(
        generator,
        "get_pocket_from_ligand",
        lambda *_args, **_kwargs: ("Ligand ATP center: 1, 2, 3", "ALA10_A", {"size_x": 12.0}),
    )
    _patch_rdkit_for_run(monkeypatch, {"CCO", "CCN"})
    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: (["CCO"], 0))
    monkeypatch.setattr(generator, "passes_lipinski", lambda _mol: True)
    monkeypatch.setattr(generator, "get_max_similarity", lambda _smiles, _targets: 0.7)

    workflow = _holdout_workflow(monkeypatch, [])
    asyncio.run(workflow.run(options))

    detect.assert_not_called()
    manifest = json.loads(
        (Path(options.output_dir) / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["pocket"]["mode"] == "ligand"
    assert manifest["pocket"]["user_specified"] is True
    assert manifest["pocket"]["selection"]["ligand"] == "ATP"
    assert manifest["pocket"]["backend"] is None


def test_pocket_index_still_uses_detection(
    tmp_path: Path,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinning an index selects among detected pockets, so detection runs."""

    options = _holdout_options(tmp_path, _seed_csv(tmp_path)).model_copy(
        update={"use_pocket_data": True, "pocket_index": 2, "residue_position_map": {10: 5}}
    )

    captured: dict[str, object] = {}

    def fake_detect(_pdb, _out, **kwargs):
        captured.update(kwargs)
        return ("Center: 1, 2, 3", "ALA10_A", None)

    monkeypatch.setattr(generator, "get_binding_pockets_and_residues", fake_detect)
    _patch_rdkit_for_run(monkeypatch, {"CCO", "CCN"})
    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: (["CCO"], 0))
    monkeypatch.setattr(generator, "passes_lipinski", lambda _mol: True)
    monkeypatch.setattr(generator, "get_max_similarity", lambda _smiles, _targets: 0.7)

    workflow = _holdout_workflow(monkeypatch, [])
    asyncio.run(workflow.run(options))

    assert captured["pocket_index"] == 2
    manifest = json.loads(
        (Path(options.output_dir) / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["pocket"]["mode"] == "index"
    assert manifest["pocket"]["backend"] == "deepchem"


def test_failed_user_pocket_stops_the_run(
    tmp_path: Path,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pocket the user named must not be silently swapped for no pocket.

    Falling back would spend the whole API budget on a different site while
    the report still looked normal, so the run is stopped instead.
    """

    options = _holdout_options(tmp_path, _seed_csv(tmp_path)).model_copy(
        update={"use_pocket_data": True, "pocket_ligand": "ATP"}
    )

    def explode(*_args, **_kwargs):
        raise generator.PocketDetectionError("No bound ligand matching 'ATP'")

    monkeypatch.setattr(generator, "get_pocket_from_ligand", explode)
    _patch_rdkit_for_run(monkeypatch, {"CCO", "CCN"})

    workflow = _holdout_workflow(monkeypatch, [])

    with pytest.raises(generator.PocketDetectionError, match="could not be resolved"):
        asyncio.run(workflow.run(options))

    manifest = json.loads(
        (Path(options.output_dir) / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"


def test_failed_automatic_detection_still_falls_back(
    tmp_path: Path,
    generator_dependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detection can fail for reasons the user cannot act on, so it degrades."""

    options = _holdout_options(tmp_path, _seed_csv(tmp_path)).model_copy(
        update={"use_pocket_data": True}
    )

    def explode(*_args, **_kwargs):
        raise generator.PocketDetectionError("no pockets found")

    monkeypatch.setattr(generator, "get_binding_pockets_and_residues", explode)
    _patch_rdkit_for_run(monkeypatch, {"CCO", "CCN"})
    monkeypatch.setattr(generator, "parse_smiles_from_text", lambda _raw: (["CCO"], 0))
    monkeypatch.setattr(generator, "passes_lipinski", lambda _mol: True)
    monkeypatch.setattr(generator, "get_max_similarity", lambda _smiles, _targets: 0.7)

    workflow = _holdout_workflow(monkeypatch, [])
    asyncio.run(workflow.run(options))

    manifest = json.loads(
        (Path(options.output_dir) / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["pocket"]["mode"] == "detection_failed"
