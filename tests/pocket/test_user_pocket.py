"""Tests for user-specified binding pockets."""

from pathlib import Path

import pytest

from src.core.exceptions import PocketDetectionError
from src.pocket import (
    get_pocket_from_center,
    get_pocket_from_ligand,
    get_pocket_from_residues,
    parse_pocket_residue_tokens,
)


def _atom(serial: int, name: str, residue: str, chain: str, number: int, xyz, hetatm=False):
    """Renders one fixed-column PDB atom record."""

    record = "HETATM" if hetatm else "ATOM  "
    element = name.strip()[0]
    padded = f" {name}" if len(name) < 4 else name
    x, y, z = xyz
    return (
        f"{record}{serial:>5} {padded:<4}{'':1}{residue:>3} "
        f"{chain:1}{number:>4}{'':1}   "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{0.0:>6.2f}          {element:>2}"
    )


@pytest.fixture
def structure(tmp_path: Path) -> Path:
    """Writes a small two-chain structure with one bound ligand.

    Residues sit 5 A apart along x, so a contact radius picks up a known,
    checkable set rather than everything at once.
    """

    lines = []
    serial = 0
    for index, (residue, x) in enumerate(
        [("ALA", 0.0), ("GLY", 5.0), ("SER", 10.0), ("LEU", 40.0)], start=1
    ):
        for name in ("N", "CA", "C"):
            serial += 1
            lines.append(_atom(serial, name, residue, "A", index, (x, 0.0, 0.0)))
    # A second chain, to prove chain qualification actually selects.
    serial += 1
    lines.append(_atom(serial, "CA", "VAL", "B", 2, (5.0, 0.0, 0.0)))
    # A ligand beside residue 2, plus a water that must never be treated as one.
    for offset in (0.0, 1.4):
        serial += 1
        lines.append(_atom(serial, "C1", "LIG", "L", 401, (5.0 + offset, 1.0, 0.0), hetatm=True))
    serial += 1
    lines.append(_atom(serial, "O", "HOH", "W", 501, (5.0, 3.0, 0.0), hetatm=True))

    path = tmp_path / "target.pdb"
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A:203", [("A", 203)]),
        ("ALA203_A", [("A", 203)]),
        ("203", [("X", 203)]),
        ("A:203, A:207", [("A", 203), ("A", 207)]),
        ("203 207", [("X", 203), ("X", 207)]),
    ],
)
def test_parse_residue_tokens_accepts_each_form(text: str, expected) -> None:
    """All three notations parse, and bare numbers take the target chain."""

    assert parse_pocket_residue_tokens(text, default_chain="X") == expected


@pytest.mark.parametrize("bad", ["A:20x", "hello", "::", "A:"])
def test_parse_residue_tokens_rejects_malformed_input(bad: str) -> None:
    """A typo raises rather than silently dropping a requested residue."""

    with pytest.raises(PocketDetectionError):
        parse_pocket_residue_tokens(bad)


def test_parse_residue_tokens_rejects_empty_selection() -> None:
    """An empty selection is an error, not an empty pocket."""

    with pytest.raises(PocketDetectionError):
        parse_pocket_residue_tokens("  ,  ")


def test_residues_mode_returns_labels_in_the_detected_format(structure: Path) -> None:
    """Output matches what detection emits, so downstream code is unchanged."""

    _, residues, box = get_pocket_from_residues(structure, "A:1, A:2", default_chain="A")

    assert sorted(residues.split(", ")) == ["ALA1_A", "GLY2_A"]
    assert box is not None and box["size_x"] > 0


def test_residues_mode_honours_the_chain_qualifier(structure: Path) -> None:
    """Residue 2 exists on both chains; the qualifier picks one."""

    _, chain_a, _ = get_pocket_from_residues(structure, "A:2", default_chain="A")
    _, chain_b, _ = get_pocket_from_residues(structure, "B:2", default_chain="A")

    assert chain_a == "GLY2_A"
    assert chain_b == "VAL2_B"


def test_residues_mode_drops_missing_residues_but_keeps_the_rest(structure: Path) -> None:
    """A residue absent from the file cannot be mapped, so it is dropped."""

    _, residues, _ = get_pocket_from_residues(structure, "A:1, A:9999", default_chain="A")

    assert residues == "ALA1_A"


def test_residues_mode_fails_when_nothing_matches(structure: Path) -> None:
    """If no requested residue exists, the pocket is undefined and we stop."""

    with pytest.raises(PocketDetectionError, match="are present in"):
        get_pocket_from_residues(structure, "A:9999", default_chain="A")


def test_ligand_mode_selects_the_surrounding_residues(structure: Path) -> None:
    """The pocket is read off the bound ligand's neighbourhood."""

    desc, residues, box = get_pocket_from_ligand(structure, "LIG", contact_distance=6.0)

    selected = set(residues.split(", "))
    assert "GLY2_A" in selected
    # 30 A away, so far outside the radius.
    assert "LEU4_A" not in selected
    assert "LIG" in desc and box is not None


def test_ligand_mode_excludes_the_ligand_itself(structure: Path) -> None:
    """The ligand's own atoms are not pocket residues."""

    _, residues, _ = get_pocket_from_ligand(structure, "LIG", contact_distance=6.0)

    assert "LIG" not in residues


def test_ligand_mode_can_be_narrowed_to_one_copy(structure: Path) -> None:
    """A chain-and-number selector addresses a single bound copy."""

    _, residues, _ = get_pocket_from_ligand(structure, "LIG:L:401", contact_distance=6.0)

    assert "GLY2_A" in residues


def test_ligand_mode_refuses_solvent(structure: Path) -> None:
    """Centring a pocket on a water molecule is always a mistake."""

    with pytest.raises(PocketDetectionError, match="solvent"):
        get_pocket_from_ligand(structure, "HOH")


def test_ligand_mode_fails_when_the_ligand_is_absent(structure: Path) -> None:
    """A named ligand that is not in the file stops the run."""

    with pytest.raises(PocketDetectionError, match="No bound ligand"):
        get_pocket_from_ligand(structure, "ATP")


def test_center_mode_selects_nearby_residues(structure: Path) -> None:
    """A point in the structure's frame defines the pocket."""

    _, residues, box = get_pocket_from_center(structure, (5.0, 0.0, 0.0), contact_distance=6.0)

    selected = set(residues.split(", "))
    assert "GLY2_A" in selected
    assert "LEU4_A" not in selected
    assert box is not None


def test_center_mode_fails_when_the_point_is_off_structure(structure: Path) -> None:
    """Coordinates from another frame find nothing, and that is reported."""

    with pytest.raises(PocketDetectionError, match="No residue lies within"):
        get_pocket_from_center(structure, (999.0, 999.0, 999.0), contact_distance=6.0)
