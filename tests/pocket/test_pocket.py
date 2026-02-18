"""Tests for pocket discovery helpers."""

import os
import stat
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src import pocket


class DummyPocket:
    """Simple pocket object matching the shape expected by pocket helpers."""

    def __init__(self, center: tuple[float, float, float], spans: tuple[float, float, float]) -> None:
        self._center = center
        sx, sy, sz = spans
        self.x_range = (0.0, sx)
        self.y_range = (0.0, sy)
        self.z_range = (0.0, sz)

    def center(self) -> tuple[float, float, float]:
        return self._center


class FakePosition:
    """Mimics RDKit position container."""

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class FakeResidueInfo:
    """Mimics RDKit PDB residue metadata object."""

    def __init__(self, name: str, number: int, chain: str) -> None:
        self._name = name
        self._number = number
        self._chain = chain

    def GetResidueName(self) -> str:  # noqa: N802
        return self._name

    def GetResidueNumber(self) -> int:  # noqa: N802
        return self._number

    def GetChainId(self) -> str:  # noqa: N802
        return self._chain


class FakeAtom:
    """Mimics RDKit atom object used by residue extraction."""

    def __init__(self, idx: int, residue_info: FakeResidueInfo | None) -> None:
        self._idx = idx
        self._residue_info = residue_info

    def GetIdx(self) -> int:  # noqa: N802
        return self._idx

    def GetPDBResidueInfo(self) -> FakeResidueInfo | None:  # noqa: N802
        return self._residue_info


class FakeConformer:
    """Mimics RDKit conformer object with atom positions."""

    def __init__(self, positions: dict[int, FakePosition]) -> None:
        self._positions = positions

    def GetAtomPosition(self, idx: int) -> FakePosition:  # noqa: N802
        return self._positions[idx]


class FakeMol:
    """Mimics RDKit molecule object required by `get_binding_pockets_and_residues`."""

    def __init__(self, atoms: list[FakeAtom], conformer: FakeConformer) -> None:
        self._atoms = atoms
        self._conformer = conformer

    def GetConformer(self) -> FakeConformer:  # noqa: N802
        return self._conformer

    def GetAtoms(self) -> list[FakeAtom]:  # noqa: N802
        return self._atoms


@pytest.mark.parametrize(
    "spans,expected",
    [
        ((2.0, 3.0, 4.0), 24.0),
        ((1.5, 1.5, 2.0), 4.5),
    ],
)
def test_pocket_volume(spans: tuple[float, float, float], expected: float) -> None:
    """Pocket volume is calculated from axis-aligned span lengths."""

    candidate = DummyPocket(center=(0.0, 0.0, 0.0), spans=spans)
    assert pocket._pocket_volume(candidate) == pytest.approx(expected)


def test_setup_p2rank_uses_project_prank_and_sets_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local `prank/prank` is discovered and made executable."""

    prank_path = tmp_path / "prank" / "prank"
    prank_path.parent.mkdir(parents=True)
    prank_path.write_text("#!/bin/bash\n", encoding="utf-8")
    prank_path.chmod(0o644)

    monkeypatch.chdir(tmp_path)
    resolved = pocket.setup_p2rank()

    assert resolved == str(prank_path)
    assert os.stat(prank_path).st_mode & stat.S_IEXEC


def test_setup_p2rank_falls_back_to_glob_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When `prank/prank` is missing, glob fallback is used."""

    alt = tmp_path / "p2rank-v2.5" / "prank"
    alt.parent.mkdir(parents=True)
    alt.write_text("#!/bin/bash\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    resolved = pocket.setup_p2rank()

    assert resolved == str(alt)


def test_setup_p2rank_raises_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing P2Rank executable raises `FileNotFoundError`."""

    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="prank"):
        pocket.setup_p2rank()


def test_get_p2rank_pocket_reads_residue_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Top-ranked residue ids are returned from P2Rank prediction CSV."""

    pdb_path = tmp_path / "target.pdb"
    pdb_path.write_text("HEADER\n", encoding="utf-8")

    output_dir = tmp_path / "p2rank_output"
    output_dir.mkdir()
    pred_file = output_dir / "target.pdb_predictions.csv"
    pred_file.write_text('residue_ids\n"ALA10_A,LYS12_A"\n', encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pocket, "setup_p2rank", lambda: "/fake/prank")
    run_mock = MagicMock()
    monkeypatch.setattr(pocket.subprocess, "run", run_mock)

    result = pocket.get_p2rank_pocket(str(pdb_path))

    assert result == "ALA10_A,LYS12_A"
    run_mock.assert_called_once()


def test_get_p2rank_pocket_returns_unknown_when_file_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent prediction CSV maps to `Unknown Pocket`."""

    pdb_path = tmp_path / "target.pdb"
    pdb_path.write_text("HEADER\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pocket, "setup_p2rank", lambda: "/fake/prank")
    monkeypatch.setattr(pocket.subprocess, "run", MagicMock())

    assert pocket.get_p2rank_pocket(str(pdb_path)) == "Unknown Pocket"


@pytest.mark.parametrize(
    ("inputs", "expected_index"),
    [
        (["0"], 1),
        (["abc", "5", "1"], 0),
    ],
)
def test_select_pocket_interactively(
    monkeypatch: pytest.MonkeyPatch,
    inputs: list[str],
    expected_index: int,
) -> None:
    """Selection helper handles invalid inputs and supports largest-volume shortcut."""

    pockets = [
        DummyPocket(center=(0.0, 0.0, 0.0), spans=(1.0, 1.0, 1.0)),
        DummyPocket(center=(1.0, 1.0, 1.0), spans=(4.0, 4.0, 4.0)),
    ]
    pocket_rows = [
        {"pocket_id": 1.0, "center_x": 0.0, "center_y": 0.0, "center_z": 0.0, "volume_approx": 1.0},
        {"pocket_id": 2.0, "center_x": 1.0, "center_y": 1.0, "center_z": 1.0, "volume_approx": 64.0},
    ]

    answers = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    selected = pocket._select_pocket_interactively(pockets, pocket_rows)
    assert selected is pockets[expected_index]


def test_get_binding_pockets_and_residues_no_pockets(monkeypatch: pytest.MonkeyPatch) -> None:
    """If DeepChem reports no pockets, function returns no-pocket sentinel values."""

    finder_instance = MagicMock()
    finder_instance.find_pockets.return_value = []
    finder_cls = MagicMock(return_value=finder_instance)
    fake_deepchem = types.SimpleNamespace(
        dock=types.SimpleNamespace(ConvexHullPocketFinder=finder_cls),
    )
    monkeypatch.setitem(__import__("sys").modules, "deepchem", fake_deepchem)

    center, residues = pocket.get_binding_pockets_and_residues("protein.pdb")

    assert center == "No pockets found"
    assert residues == "Unknown"


def test_get_binding_pockets_and_residues_with_mocked_mol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pocket center and nearby residues are derived from selected pocket + PDB atoms."""

    selected = DummyPocket(center=(1.0, 2.0, 3.0), spans=(2.0, 2.0, 2.0))
    finder_instance = MagicMock()
    finder_instance.find_pockets.return_value = [selected]
    finder_cls = MagicMock(return_value=finder_instance)
    fake_deepchem = types.SimpleNamespace(
        dock=types.SimpleNamespace(ConvexHullPocketFinder=finder_cls),
    )
    monkeypatch.setitem(__import__("sys").modules, "deepchem", fake_deepchem)
    monkeypatch.setattr(pocket, "_select_pocket_interactively", lambda pockets, rows: selected)

    atoms = [
        FakeAtom(0, FakeResidueInfo("ALA", 10, "A")),
        FakeAtom(1, FakeResidueInfo("GLY", 20, "A")),
    ]
    conformer = FakeConformer(
        {
            0: FakePosition(1.0, 2.0, 3.0),
            1: FakePosition(40.0, 40.0, 40.0),
        }
    )
    fake_mol = FakeMol(atoms=atoms, conformer=conformer)
    monkeypatch.setattr(pocket.Chem, "MolFromPDBFile", lambda _path: fake_mol)

    output_dir = tmp_path / "results"
    center, residues = pocket.get_binding_pockets_and_residues("protein.pdb", str(output_dir))

    assert center == "Center: 1.00, 2.00, 3.00"
    assert residues == "ALA10_A"
    assert (output_dir / "all_pockets.csv").exists()


def test_get_binding_pockets_and_residues_handles_unreadable_pdb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If RDKit cannot parse PDB coordinates, residue list falls back to `Unknown`."""

    selected = DummyPocket(center=(5.0, 6.0, 7.0), spans=(1.0, 1.0, 1.0))
    finder_instance = MagicMock()
    finder_instance.find_pockets.return_value = [selected]
    finder_cls = MagicMock(return_value=finder_instance)
    fake_deepchem = types.SimpleNamespace(
        dock=types.SimpleNamespace(ConvexHullPocketFinder=finder_cls),
    )
    monkeypatch.setitem(__import__("sys").modules, "deepchem", fake_deepchem)
    monkeypatch.setattr(pocket, "_select_pocket_interactively", lambda pockets, rows: selected)
    monkeypatch.setattr(pocket.Chem, "MolFromPDBFile", lambda _path: None)

    center, residues = pocket.get_binding_pockets_and_residues("protein.pdb", str(tmp_path))

    assert center == "Center: 5.00, 6.00, 7.00"
    assert residues == "Unknown"
