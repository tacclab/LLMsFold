"""Tests for holdout evaluators and their availability reporting."""

import sys
from pathlib import Path

import pandas as pd
import pytest

from src.evaluators import HoldoutCandidate, HoldoutContext, PoseBustersEvaluator, VinaEvaluator


def _hide_module(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Makes `import name` fail, simulating a missing optional dependency."""

    monkeypatch.setitem(sys.modules, name, None)


def test_posebusters_reports_unavailable_with_an_actionable_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing optional dependency is data, not an exception.

    The pipeline must finish and say which independent checks did not run,
    rather than failing a run that has already spent its API budget.
    """

    _hide_module(monkeypatch, "posebusters")

    availability = PoseBustersEvaluator().availability()

    assert availability.available is False
    assert "posebusters" in availability.reason
    assert "holdout" in availability.reason


@pytest.mark.parametrize("missing", ["vina", "meeko"])
def test_vina_reports_each_missing_dependency(
    monkeypatch: pytest.MonkeyPatch, missing: str, tmp_path: Path
) -> None:
    """Vina needs both its engine and a ligand preparer to be usable."""

    _hide_module(monkeypatch, missing)
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("", encoding="utf-8")

    availability = VinaEvaluator(receptor_pdbqt=receptor).availability()

    assert availability.available is False
    assert missing in availability.reason


def test_vina_builds_its_own_receptor_from_each_complex() -> None:
    """Vina needs no externally prepared receptor.

    A co-folding model returns its complex in its own coordinate frame, not
    the input structure's, so scoring a predicted ligand against a receptor
    prepared from the input places the two tens of Angstroms apart and Vina
    rejects every pose as outside the grid. The receptor is therefore built
    per complex from that complex's own protein atoms, and availability
    depends only on the tooling being present.
    """

    pytest.importorskip("vina", reason="vina not installed")
    pytest.importorskip("meeko", reason="meeko not installed")

    availability = VinaEvaluator(receptor_pdbqt=None).availability()

    assert availability.available is True


def test_vina_reports_missing_receptor_tooling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without meeko's receptor script the reason says so, rather than failing."""

    pytest.importorskip("vina", reason="vina not installed")
    monkeypatch.setattr("shutil.which", lambda _name: None)

    availability = VinaEvaluator(receptor_pdbqt=None).availability()

    assert availability.available is False
    assert "mk_prepare_receptor" in availability.reason


def test_vina_skips_evaluation_without_a_docking_box(tmp_path: Path) -> None:
    """Rescoring needs the pocket box, which only exists when detection ran."""

    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("", encoding="utf-8")
    evaluator = VinaEvaluator(receptor_pdbqt=receptor)

    results = evaluator.evaluate(
        [HoldoutCandidate(smiles="CCO", candidate_id="c1", structure="mmcif")],
        HoldoutContext(docking_box=None),
    )

    assert results == {}


def test_posebusters_summary_counts_every_boolean_check() -> None:
    """Checks are discovered from the report, not from a hardcoded list.

    A list here would silently ignore any check added by a future PoseBusters
    release, quietly weakening the validation over time.
    """

    report = pd.DataFrame(
        [
            {
                "bond_lengths": True,
                "internal_steric_clash": False,
                "volume_overlap_with_protein": False,
                "molecule": "ignored non-boolean column",
            }
        ]
    )

    summary = PoseBustersEvaluator._summarize(report)

    assert summary["PoseBusters_Pass"] is False
    assert summary["PoseBusters_Checks_Total"] == 3
    assert summary["PoseBusters_Checks_Passed"] == 1
    assert summary["PoseBusters_Failed_Checks"] == (
        "internal_steric_clash, volume_overlap_with_protein"
    )


def test_posebusters_summary_passes_when_every_check_holds() -> None:
    """A fully valid pose reports a clean pass with no failed check names."""

    summary = PoseBustersEvaluator._summarize(
        pd.DataFrame([{"bond_lengths": True, "bond_angles": True}])
    )

    assert summary["PoseBusters_Pass"] is True
    assert summary["PoseBusters_Failed_Checks"] == ""


@pytest.mark.parametrize("report", [None, pd.DataFrame(), pd.DataFrame([{"name": "x"}])])
def test_posebusters_summary_returns_none_without_checks(report) -> None:
    """An empty or check-free report yields no columns rather than a false pass."""

    assert PoseBustersEvaluator._summarize(report) is None


def test_posebusters_skips_candidates_without_a_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidates the predictor returned no pose for are simply not evaluated.

    Partial coverage is reported honestly; it is never filled in with values
    that were not measured.
    """

    pytest.importorskip("posebusters")

    results = PoseBustersEvaluator().evaluate(
        [HoldoutCandidate(smiles="CCO", candidate_id="c1", structure=None)],
        HoldoutContext(),
    )

    assert results == {}
