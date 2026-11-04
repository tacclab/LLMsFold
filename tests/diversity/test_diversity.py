"""Tests for scaffold-level diversity and novelty analysis."""

import pytest

from src.diversity import (
    ACYCLIC_SCAFFOLD,
    ScaffoldAnalyzer,
    generic_scaffold,
    murcko_scaffold,
)

BENZOIC_ACID = "c1ccccc1C(=O)O"
BENZAMIDE = "c1ccccc1C(=O)N"
INDOLE = "c1ccc2[nH]ccc2c1"
BENZOFURAN = "c1ccc2occc2c1"
PYRIDINE = "c1ccncc1"
BENZENE = "c1ccccc1"


def test_murcko_scaffold_strips_decoration() -> None:
    """Two analogues of one core collapse onto the same Murcko scaffold.

    This is the case whole-molecule fingerprint similarity misses: the pair
    differ as molecules while exploring no new chemistry.
    """

    assert murcko_scaffold(BENZOIC_ACID) == murcko_scaffold(BENZAMIDE)
    assert murcko_scaffold(BENZOIC_ACID) != murcko_scaffold(INDOLE)


def test_generic_scaffold_collapses_heteroatoms_and_bond_orders() -> None:
    """Benzene and pyridine share one generic framework but not one scaffold."""

    assert murcko_scaffold(BENZENE) != murcko_scaffold(PYRIDINE)
    assert generic_scaffold(BENZENE) == generic_scaffold(PYRIDINE)


def test_acyclic_molecule_has_an_empty_scaffold() -> None:
    """A ring-free molecule has no Bemis-Murcko core, which is not a failure."""

    assert murcko_scaffold("CCCCO") == ACYCLIC_SCAFFOLD


@pytest.mark.parametrize("bad", ["not-a-smiles", "C1CC", "[[["])
def test_scaffold_helpers_return_none_for_unparseable_input(bad: str) -> None:
    """Unparseable input degrades to `None` rather than raising."""

    assert murcko_scaffold(bad) is None
    assert generic_scaffold(bad) is None


def test_annotate_distinguishes_reused_cores_from_novel_ones() -> None:
    """A decorated seed core is not novel; a different ring system is."""

    analyzer = ScaffoldAnalyzer([BENZOIC_ACID, INDOLE])
    assert analyzer.seed_scaffold_count == 2

    reused = analyzer.annotate(BENZAMIDE)
    assert reused["Scaffold_Is_Novel"] is False
    assert reused["Max_Scaffold_Sim_To_Seeds"] == pytest.approx(1.0)

    novel = analyzer.annotate(BENZOFURAN)
    assert novel["Scaffold_Is_Novel"] is True
    assert novel["Max_Scaffold_Sim_To_Seeds"] < 1.0


def test_annotate_leaves_novelty_undefined_without_a_scaffold() -> None:
    """Ring-free and unparseable molecules report `None`, never `False`.

    `False` would read as "this core matches a seed", which is a different
    claim from "there is no core to compare".
    """

    analyzer = ScaffoldAnalyzer([BENZOIC_ACID])

    acyclic = analyzer.annotate("CCCCO")
    assert acyclic["Murcko_Scaffold"] == ACYCLIC_SCAFFOLD
    assert acyclic["Scaffold_Is_Novel"] is None

    broken = analyzer.annotate("not-a-smiles")
    assert broken["Murcko_Scaffold"] is None
    assert broken["Scaffold_Is_Novel"] is None


def test_summarize_reports_scaffold_convergence() -> None:
    """A run that decorates one core scores far below one that finds three."""

    analyzer = ScaffoldAnalyzer([BENZOIC_ACID])

    converged = analyzer.summarize([BENZOIC_ACID, BENZAMIDE, "c1ccccc1C(=O)NC"])
    assert converged["unique_murcko_scaffolds"] == 1
    # Summary values are rounded to four decimals for the manifest.
    assert converged["scaffold_diversity"] == pytest.approx(1 / 3, abs=1e-4)
    assert converged["largest_scaffold_cluster"] == 3
    assert converged["molecules_on_seed_scaffolds"] == 3
    assert converged["novel_scaffold_fraction"] == pytest.approx(0.0)
    assert converged["scaffold_entropy_bits"] == pytest.approx(0.0)

    diverse = analyzer.summarize([BENZOIC_ACID, INDOLE, BENZOFURAN])
    assert diverse["unique_murcko_scaffolds"] == 3
    assert diverse["scaffold_diversity"] == pytest.approx(1.0)
    assert diverse["largest_scaffold_cluster"] == 1
    assert diverse["novel_scaffold_fraction"] == pytest.approx(2 / 3, abs=1e-4)
    assert diverse["scaffold_entropy_bits"] > converged["scaffold_entropy_bits"]


def test_summarize_counts_acyclic_and_unparseable_separately() -> None:
    """Molecules without a usable core are reported, not silently dropped."""

    analyzer = ScaffoldAnalyzer([BENZOIC_ACID])

    summary = analyzer.summarize([BENZOIC_ACID, "CCCCO", "not-a-smiles"])

    assert summary["molecules"] == 3
    assert summary["unique_murcko_scaffolds"] == 1
    assert summary["acyclic_molecules"] == 1
    assert summary["unparseable_molecules"] == 1


def test_summarize_handles_degenerate_inputs() -> None:
    """Empty and single-molecule runs return without raising."""

    analyzer = ScaffoldAnalyzer([])

    assert analyzer.summarize([]) == {"molecules": 0}

    single = analyzer.summarize([BENZOIC_ACID])
    assert single["molecules"] == 1
    # Pairwise diversity is undefined for one molecule.
    assert single["internal_diversity"] is None


def test_internal_diversity_reflects_whole_molecule_spread() -> None:
    """Identical molecules score 0; unrelated ones score well above 0."""

    analyzer = ScaffoldAnalyzer([])

    assert analyzer.summarize([BENZOIC_ACID, BENZOIC_ACID])["internal_diversity"] == (
        pytest.approx(0.0)
    )
    assert analyzer.summarize([BENZOIC_ACID, INDOLE])["internal_diversity"] > 0.5


def test_analyzer_without_seeds_treats_every_scaffold_as_novel() -> None:
    """With no reference set there is nothing to have reused."""

    analyzer = ScaffoldAnalyzer([])

    annotated = analyzer.annotate(BENZOIC_ACID)
    assert annotated["Scaffold_Is_Novel"] is True
    assert annotated["Max_Scaffold_Sim_To_Seeds"] is None
