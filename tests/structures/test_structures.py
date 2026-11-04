"""Tests for splitting predicted protein-ligand complexes."""

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from src.structures import ComplexParts, parse_atom_site_records, split_complex

_COLUMNS = (
    "group_PDB",
    "id",
    "type_symbol",
    "label_atom_id",
    "label_comp_id",
    "label_asym_id",
    "label_seq_id",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "auth_asym_id",
    "auth_seq_id",
)


def _embed(smiles: str):
    """Returns a molecule with 3D coordinates, hydrogens removed."""

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    AllChem.MMFFOptimizeMolecule(mol)
    return Chem.RemoveHs(mol)


def build_mmcif(
    ligand_smiles: str,
    *,
    protein_atoms: int = 6,
    ligand_residue: str = "LIG",
    extra_hetatms: tuple[tuple[str, str], ...] = (),
) -> str:
    """Builds a Boltz-style mmCIF holding a protein stub and one ligand."""

    rows = ["loop_"] + [f"_atom_site.{column}" for column in _COLUMNS]
    serial = 0

    for index in range(1, protein_atoms + 1):
        serial += 1
        rows.append(
            f"ATOM {serial} C CA ALA A {index} "
            f"{40.0 + index:.3f} {40.000:.3f} {40.000:.3f} A {index}"
        )

    for residue, element in extra_hetatms:
        serial += 1
        rows.append(
            f"HETATM {serial} {element} {element}1 {residue} S . "
            f"{80.000:.3f} {80.000:.3f} {80.000:.3f} S 1"
        )

    mol = _embed(ligand_smiles)
    conformer = mol.GetConformer()
    for atom in mol.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        serial += 1
        rows.append(
            f"HETATM {serial} {atom.GetSymbol()} {atom.GetSymbol()}{atom.GetIdx() + 1} "
            f"{ligand_residue} L . "
            f"{position.x:.3f} {position.y:.3f} {position.z:.3f} L 1"
        )

    rows.append("#")
    return "\n".join(rows)


def test_parse_atom_site_records_reads_the_declared_columns() -> None:
    """Values are keyed by the loop's own header, not by position guessing."""

    records = parse_atom_site_records(build_mmcif("CCO", protein_atoms=2))

    assert records[0]["group_PDB"] == "ATOM"
    assert records[0]["label_comp_id"] == "ALA"
    assert {record["group_PDB"] for record in records} == {"ATOM", "HETATM"}


def test_parse_atom_site_records_skips_misaligned_rows() -> None:
    """A row with the wrong field count is dropped, never shifted into place.

    Silently absorbing a short row would put coordinates in the wrong columns
    and produce a plausible-looking but wrong structure.
    """

    text = build_mmcif("CCO", protein_atoms=1)
    corrupted = text.replace("#", "ATOM 999 C\n#")

    records = parse_atom_site_records(corrupted)

    assert all(len(record) == len(_COLUMNS) for record in records)
    assert not any(record["id"] == "999" for record in records)


def test_parse_atom_site_records_handles_quoted_values() -> None:
    """Quoted atom names containing spaces do not shift later columns."""

    text = build_mmcif("CCO", protein_atoms=1).replace(" C CA ALA ", " C 'C A' ALA ", 1)

    records = parse_atom_site_records(text)

    assert records[0]["label_atom_id"] == "C A"
    assert records[0]["label_comp_id"] == "ALA"


@pytest.mark.parametrize(
    "smiles",
    [
        "CC(=O)Nc1ccc(O)cc1",
        "c1ccc2[nH]ccc2c1",
        "COc1cc(cc(OC)c1OC)c2cc(cnc2N)c3cccc(O)c3",
    ],
)
def test_split_complex_recovers_bond_orders_from_the_submitted_smiles(smiles: str) -> None:
    """Coordinates carry no bond orders, so they are restored from the SMILES.

    Without this the aromatic rings in a predicted pose come back as single
    bonds, and every downstream geometry check would judge the wrong molecule.
    """

    parts = split_complex(build_mmcif(smiles), smiles)

    assert isinstance(parts, ComplexParts)
    assert parts.bond_orders_assigned
    expected = Chem.MolToSmiles(_embed(smiles))
    assert Chem.MolToSmiles(Chem.RemoveHs(parts.ligand)) == expected


def test_split_complex_preserves_ligand_coordinates() -> None:
    """The ligand keeps the pose it was predicted in, not a re-embedded one."""

    smiles = "CC(=O)Nc1ccc(O)cc1"
    parts = split_complex(build_mmcif(smiles), smiles)

    source = _embed(smiles).GetConformer()
    recovered = parts.ligand.GetConformer()
    expected = np.array([list(source.GetAtomPosition(i)) for i in range(source.GetNumAtoms())])
    actual = np.array([list(recovered.GetAtomPosition(i)) for i in range(recovered.GetNumAtoms())])

    assert np.allclose(np.sort(actual, axis=0), np.sort(expected, axis=0), atol=1e-3)


def test_split_complex_writes_fixed_column_pdb_for_the_protein() -> None:
    """The protein half is emitted as PDB that RDKit can read back."""

    parts = split_complex(build_mmcif("CCO"), "CCO")

    assert parts.protein_pdb_block.startswith("ATOM  ")
    assert parts.protein_atom_count == 6
    # Column 77-78 must carry the element or readers infer it from the name.
    assert parts.protein_pdb_block.splitlines()[0][76:78].strip() == "C"


def test_split_complex_excludes_solvent_and_ions_from_the_ligand() -> None:
    """A stray ion must never be mistaken for the docked ligand."""

    text = build_mmcif("CCO", extra_hetatms=(("HOH", "O"), ("ZN", "ZN")))

    parts = split_complex(text, "CCO")

    assert parts.ligand_atom_count == Chem.MolFromSmiles("CCO").GetNumAtoms()


@pytest.mark.parametrize(
    ("text", "smiles"),
    [
        ("", "CCO"),
        ("loop_\n_atom_site.group_PDB\n#", "CCO"),
    ],
)
def test_split_complex_returns_none_without_usable_atoms(text: str, smiles: str) -> None:
    """A file with no atoms yields `None` rather than an empty structure."""

    assert split_complex(text, smiles) is None


def test_split_complex_returns_none_without_a_ligand() -> None:
    """A protein-only prediction cannot be pose-checked, so it is refused."""

    text = build_mmcif("CCO")
    protein_only = "\n".join(line for line in text.splitlines() if not line.startswith("HETATM"))

    assert split_complex(protein_only, "CCO") is None


def test_split_complex_keeps_the_pose_when_bond_orders_cannot_be_assigned() -> None:
    """A mismatched SMILES still yields a pose, flagged as unassigned.

    An approximate structure is more useful to a geometry check than none,
    provided the report says the bond orders are not trustworthy.
    """

    parts = split_complex(build_mmcif("CCO"), "c1ccccc1C(=O)NCCN")

    assert parts is not None
    assert parts.bond_orders_assigned is False


def test_over_long_residue_name_does_not_shift_pdb_columns() -> None:
    """A four-character ligand name must not corrupt the fixed-column layout.

    Regression test for real Boltz-2 output, which names its ligand "LIG1".
    PDB allows three characters for a residue name, so a fourth pushes the
    chain, residue number and every coordinate one place right. The damage
    is silent: the file still parses, but a coordinate reads "-20.02"
    instead of "-20.028" and the structure becomes unusable.
    """

    smiles = "CC(=O)Nc1ccc(O)cc1"
    parts = split_complex(build_mmcif(smiles, ligand_residue="LIG1"), smiles)

    assert parts is not None
    for block in (parts.protein_pdb_block,):
        for line in block.splitlines():
            if line.startswith(("ATOM", "HETATM")):
                # Coordinates must still begin at column 31.
                assert len(line) >= 54
                float(line[30:38])
                float(line[38:46])
                float(line[46:54])


@pytest.mark.parametrize("residue", ["LIG", "LIG1", "UNKNOWN"])
def test_ligand_is_recovered_whatever_its_residue_name(tmp_path: Path, residue: str) -> None:
    """Bond orders survive regardless of how the predictor names the ligand."""

    smiles = "CC(=O)Nc1ccc(O)cc1"
    text = build_mmcif(smiles, ligand_residue=residue)

    parts = split_complex(text, smiles)

    assert parts is not None and parts.bond_orders_assigned
    assert Chem.MolToSmiles(Chem.RemoveHs(parts.ligand)) == Chem.MolToSmiles(_embed(smiles))
