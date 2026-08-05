"""Tests for pocket discovery helpers."""

import os
import stat
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src import pocket
from src.core.exceptions import PocketDetectionError


class DummyPocket:
    """Simple pocket object matching the shape expected by pocket helpers."""

    def __init__(
        self, center: tuple[float, float, float], spans: tuple[float, float, float]
    ) -> None:
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

    def __init__(
        self,
        idx: int,
        residue_info: FakeResidueInfo | None,
        atomic_num: int = 6,
    ) -> None:
        self._idx = idx
        self._residue_info = residue_info
        self._atomic_num = atomic_num

    def GetIdx(self) -> int:  # noqa: N802
        return self._idx

    def GetPDBResidueInfo(self) -> FakeResidueInfo | None:  # noqa: N802
        return self._residue_info

    def GetAtomicNum(self) -> int:  # noqa: N802
        return self._atomic_num


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


def test_setup_p2rank_falls_back_to_glob_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `prank/prank` is missing, glob fallback is used."""

    alt = tmp_path / "p2rank-v2.5" / "prank"
    alt.parent.mkdir(parents=True)
    alt.write_text("#!/bin/bash\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    resolved = pocket.setup_p2rank()

    assert resolved == str(alt)


def test_setup_p2rank_raises_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing P2Rank executable raises a domain-specific pocket error."""

    monkeypatch.chdir(tmp_path)

    with pytest.raises(PocketDetectionError, match="P2Rank executable"):
        pocket.setup_p2rank()


def test_get_p2rank_pocket_reads_residue_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    ("pocket_index", "expected_index"),
    [
        (0, 0),
        (9, 1),
    ],
)
def test_select_pocket(
    pocket_index: int,
    expected_index: int,
) -> None:
    """Selection helper supports deterministic index selection with largest fallback."""

    pockets = [
        DummyPocket(center=(0.0, 0.0, 0.0), spans=(1.0, 1.0, 1.0)),
        DummyPocket(center=(1.0, 1.0, 1.0), spans=(4.0, 4.0, 4.0)),
    ]
    pocket_rows = [
        {"pocket_id": 1.0, "center_x": 0.0, "center_y": 0.0, "center_z": 0.0, "volume_approx": 1.0},
        {
            "pocket_id": 2.0,
            "center_x": 1.0,
            "center_y": 1.0,
            "center_z": 1.0,
            "volume_approx": 64.0,
        },
    ]

    selected = pocket._select_pocket(pockets, pocket_rows, pocket_index=pocket_index)
    assert selected is pockets[expected_index]


def test_select_pocket_excludes_disqualified_pocket_regardless_of_volume() -> None:
    """A pocket failing the minimum dimension is skipped even if its volume is bigger."""

    # Flat/wide pocket: huge volume but fails the per-axis minimum on z.
    oversized_but_shallow = DummyPocket(center=(0.0, 0.0, 0.0), spans=(50.0, 50.0, 1.0))
    # Smaller volume overall, but every axis clears the minimum.
    qualifying = DummyPocket(center=(1.0, 1.0, 1.0), spans=(9.0, 9.0, 9.0))
    pockets = [oversized_but_shallow, qualifying]
    pocket_rows = [
        {"pocket_id": 1.0, "center_x": 0.0, "center_y": 0.0, "center_z": 0.0, "volume_approx": 2500.0},
        {"pocket_id": 2.0, "center_x": 1.0, "center_y": 1.0, "center_z": 1.0, "volume_approx": 729.0},
    ]

    selected = pocket._select_pocket(pockets, pocket_rows, pocket_index=-1)

    assert selected is qualifying


def test_select_pocket_prefers_smallest_among_multiple_qualifying() -> None:
    """Among several qualifying pockets, the smallest by volume is chosen.

    This minimizes the docking search space while still guaranteeing a
    ligand-sized cavity, per the paper's stated selection rule.
    """

    smaller_qualifying = DummyPocket(center=(0.0, 0.0, 0.0), spans=(9.0, 9.0, 9.0))
    larger_qualifying = DummyPocket(center=(1.0, 1.0, 1.0), spans=(10.0, 10.0, 10.0))
    pockets = [larger_qualifying, smaller_qualifying]
    pocket_rows = [
        {"pocket_id": 1.0, "center_x": 1.0, "center_y": 1.0, "center_z": 1.0, "volume_approx": 1000.0},
        {"pocket_id": 2.0, "center_x": 0.0, "center_y": 0.0, "center_z": 0.0, "volume_approx": 729.0},
    ]

    selected = pocket._select_pocket(pockets, pocket_rows, pocket_index=-1)

    assert selected is smaller_qualifying


def test_select_pocket_falls_back_when_none_qualify() -> None:
    """When no pocket meets the minimum dimension, the largest overall is used."""

    small_a = DummyPocket(center=(0.0, 0.0, 0.0), spans=(1.0, 1.0, 1.0))
    small_b = DummyPocket(center=(1.0, 1.0, 1.0), spans=(4.0, 4.0, 4.0))
    pockets = [small_a, small_b]
    pocket_rows = [
        {"pocket_id": 1.0, "center_x": 0.0, "center_y": 0.0, "center_z": 0.0, "volume_approx": 1.0},
        {"pocket_id": 2.0, "center_x": 1.0, "center_y": 1.0, "center_z": 1.0, "volume_approx": 64.0},
    ]

    selected = pocket._select_pocket(pockets, pocket_rows, pocket_index=-1)

    assert selected is small_b


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (2.0, 7.0),    # +5 margin, no cap
        (15.0, 20.0),  # +5 margin, no cap
        (25.0, 30.0),  # +5 margin would exceed max -> capped
        (50.0, 30.0),  # already over max before margin -> capped
    ],
)
def test_expand_box_dimension(size: float, expected: float) -> None:
    """Box dimensions are expanded by an isotropic margin, capped at the maximum."""

    assert pocket._expand_box_dimension(size, 5.0, 30.0) == pytest.approx(expected)


def test_get_binding_pockets_and_residues_uses_p2rank_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default backend uses P2Rank wrapper output."""

    calls: list[str] = []

    def _fake_get_p2rank_pocket(_path: str, output_dir: str = ".") -> str:
        calls.append(output_dir)
        return "ALA10_A"

    monkeypatch.setattr(pocket, "get_p2rank_pocket", _fake_get_p2rank_pocket)
    center, residues, box_dims = pocket.get_binding_pockets_and_residues("protein.pdb")

    assert center == "P2Rank"
    assert residues == "ALA10_A"
    assert box_dims is None
    assert calls == ["results"]


def test_get_binding_pockets_and_residues_no_pockets(monkeypatch: pytest.MonkeyPatch) -> None:
    """If DeepChem reports no pockets, function returns no-pocket sentinel values."""

    finder_instance = MagicMock()
    finder_instance.find_pockets.return_value = []
    finder_cls = MagicMock(return_value=finder_instance)
    fake_deepchem = types.SimpleNamespace(
        dock=types.SimpleNamespace(ConvexHullPocketFinder=finder_cls),
    )
    monkeypatch.setitem(__import__("sys").modules, "deepchem", fake_deepchem)

    center, residues, box_dims = pocket.get_binding_pockets_and_residues("protein.pdb", backend="deepchem")

    assert center == "No pockets found"
    assert residues == "Unknown"
    assert box_dims is None


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
    monkeypatch.setattr(pocket, "_select_pocket", lambda pockets, rows, pocket_index: selected)

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
    center, residues, box_dims = pocket.get_binding_pockets_and_residues(
        "protein.pdb", str(output_dir), backend="deepchem"
    )

    assert center == "Center: 1.00, 2.00, 3.00"
    assert residues == "ALA10_A"
    pockets_csv = output_dir / "all_pockets.csv"
    assert pockets_csv.exists()
    import pandas as pd
    df = pd.read_csv(pockets_csv)
    assert {"center_x", "center_y", "center_z", "size_x", "size_y", "size_z", "volume_approx"}.issubset(df.columns)
    assert box_dims == {
        "center_x": 1.0,
        "center_y": 2.0,
        "center_z": 3.0,
        "size_x": 7.0,
        "size_y": 7.0,
        "size_z": 7.0,
        "raw_size_x": 2.0,
        "raw_size_y": 2.0,
        "raw_size_z": 2.0,
    }


def test_get_binding_pockets_and_residues_box_membership_criterion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A heavy atom outside the contact radius but inside the expanded box
    still qualifies its residue; a hydrogen atom in the same spot does not.

    Regression test for the paper's two complementary residue-selection
    criteria: (i) heavy atom within the expanded box, (ii) any atom within
    the contact radius of the pocket center. Previously only (ii) was
    implemented.
    """

    # Large pocket (20A per axis) so the expanded box half-extent (10A)
    # exceeds the 8A contact radius, letting a box-only atom exist.
    selected = DummyPocket(center=(0.0, 0.0, 0.0), spans=(20.0, 20.0, 20.0))
    finder_instance = MagicMock()
    finder_instance.find_pockets.return_value = [selected]
    finder_cls = MagicMock(return_value=finder_instance)
    fake_deepchem = types.SimpleNamespace(
        dock=types.SimpleNamespace(ConvexHullPocketFinder=finder_cls),
    )
    monkeypatch.setitem(__import__("sys").modules, "deepchem", fake_deepchem)
    monkeypatch.setattr(pocket, "_select_pocket", lambda pockets, rows, pocket_index: selected)

    atoms = [
        # Outside the 8A radius (distance=9) but inside the 20A box (half-extent 10A).
        FakeAtom(0, FakeResidueInfo("ALA", 10, "A"), atomic_num=6),
        # Same position, but a hydrogen -> must not satisfy the box criterion.
        FakeAtom(1, FakeResidueInfo("GLY", 20, "A"), atomic_num=1),
        # Far outside both the radius and the box.
        FakeAtom(2, FakeResidueInfo("SER", 30, "A"), atomic_num=6),
    ]
    conformer = FakeConformer(
        {
            0: FakePosition(9.0, 0.0, 0.0),
            1: FakePosition(9.0, 0.0, 0.0),
            2: FakePosition(40.0, 40.0, 40.0),
        }
    )
    fake_mol = FakeMol(atoms=atoms, conformer=conformer)
    monkeypatch.setattr(pocket.Chem, "MolFromPDBFile", lambda _path: fake_mol)

    _center, residues, box_dims = pocket.get_binding_pockets_and_residues(
        "protein.pdb", str(tmp_path), backend="deepchem"
    )

    assert residues == "ALA10_A"
    assert box_dims["size_x"] == pytest.approx(25.0)


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
    monkeypatch.setattr(pocket, "_select_pocket", lambda pockets, rows, pocket_index: selected)
    monkeypatch.setattr(pocket.Chem, "MolFromPDBFile", lambda _path: None)

    center, residues, box_dims = pocket.get_binding_pockets_and_residues(
        "protein.pdb", str(tmp_path), backend="deepchem"
    )

    assert center == "Center: 5.00, 6.00, 7.00"
    assert residues == "Unknown"
    assert box_dims == {
        "center_x": 5.0,
        "center_y": 6.0,
        "center_z": 7.0,
        "size_x": 6.0,
        "size_y": 6.0,
        "size_z": 6.0,
        "raw_size_x": 1.0,
        "raw_size_y": 1.0,
        "raw_size_z": 1.0,
    }


def test_get_binding_pockets_and_residues_default_pocket_index_is_automatic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: not passing `pocket_index` must trigger automatic
    (dimension-filtered, smallest-by-volume) selection, not silently force
    pocket 0. Previously the default was `0`, which `_select_pocket` treats
    as an explicit "always pick index 0" request, since index 0 is always
    in-range whenever at least one pocket exists. Selection is unconditional
    and non-interactive -- there is no TTY prompt to bypass.
    """

    larger_qualifying = DummyPocket(center=(0.0, 0.0, 0.0), spans=(10.0, 10.0, 10.0))
    smaller_qualifying = DummyPocket(center=(1.0, 1.0, 1.0), spans=(9.0, 9.0, 9.0))
    finder_instance = MagicMock()
    finder_instance.find_pockets.return_value = [larger_qualifying, smaller_qualifying]
    finder_cls = MagicMock(return_value=finder_instance)
    fake_deepchem = types.SimpleNamespace(
        dock=types.SimpleNamespace(ConvexHullPocketFinder=finder_cls),
    )
    monkeypatch.setitem(__import__("sys").modules, "deepchem", fake_deepchem)
    monkeypatch.setattr(pocket.Chem, "MolFromPDBFile", lambda _path: None)

    # Note: no `pocket_index` argument -- exercising the real default.
    center, _residues, _box_dims = pocket.get_binding_pockets_and_residues(
        "protein.pdb",
        backend="deepchem",
    )

    assert center == "Center: 1.00, 1.00, 1.00"
