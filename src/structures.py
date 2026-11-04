"""Parsing and splitting of predicted protein-ligand complexes.

Boltz returns one mmCIF holding the protein and the docked ligand together,
but every physics-based check wants them apart: a pose validator needs the
ligand as a molecule with real bond orders, and the protein separately as the
thing it might be clashing with.

No mmCIF reader is available in this project's RDKit build, so the
`_atom_site` loop is parsed directly here. That loop is the only part of the
file these checks need, and it is a plain whitespace-delimited table whose
column order is declared in its own header.

Bond orders cannot come from the coordinates: mmCIF and PDB record positions
and elements but not bonding. They are recovered by matching the pose against
the SMILES that was submitted to the predictor in the first place, which is
the only trustworthy source for them.
"""

from dataclasses import dataclass
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem

from src.core.logging import get_logger

logger = get_logger(__name__)

_ATOM_SITE_PREFIX = "_atom_site."

# Non-polymer residues that are solvent or crystallisation artefacts rather
# than the docked ligand. Excluded so a stray ion never becomes "the pose".
_NON_LIGAND_RESIDUES = frozenset(
    {
        "HOH",
        "WAT",
        "DOD",
        "SO4",
        "PO4",
        "GOL",
        "EDO",
        "PEG",
        "ACT",
        "DMS",
        "NA",
        "CL",
        "MG",
        "ZN",
        "CA",
        "K",
        "MN",
        "FE",
        "FE2",
        "CU",
        "NI",
        "CO",
        "CD",
        "HG",
        "BR",
        "IOD",
        "F",
        "NH4",
    }
)

# Boltz labels the submitted ligand with this component id; when present it is
# preferred over the general heteroatom sweep.
_BOLTZ_LIGAND_COMP_ID = "LIG"


@dataclass(frozen=True)
class ComplexParts:
    """A predicted complex split into its protein and ligand halves."""

    protein_pdb_block: str
    ligand: Any
    protein_atom_count: int
    ligand_atom_count: int
    bond_orders_assigned: bool


def _split_cif_row(line: str) -> list[str]:
    """Splits one mmCIF data row, honouring quoted values.

    Values such as author atom names may be quoted and contain spaces, so a
    bare `str.split()` would silently shift every later column.
    """

    tokens: list[str] = []
    current: list[str] = []
    quote: str | None = None

    for char in line:
        if quote is not None:
            if char == quote:
                quote = None
            else:
                current.append(char)
            continue
        if char in "'\"":
            quote = char
            continue
        if char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
            continue
        current.append(char)

    if current:
        tokens.append("".join(current))
    return tokens


def parse_atom_site_records(mmcif_text: str) -> list[dict[str, str]]:
    """Returns the `_atom_site` loop as one dict per atom.

    Keys are the column names with the `_atom_site.` prefix stripped. Rows
    whose field count does not match the header are skipped rather than
    misaligned into neighbouring columns.
    """

    lines = mmcif_text.splitlines()
    records: list[dict[str, str]] = []
    index = 0

    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue

        index += 1
        columns: list[str] = []
        while index < len(lines) and lines[index].lstrip().startswith("_"):
            header = lines[index].strip()
            if header.startswith(_ATOM_SITE_PREFIX):
                columns.append(header[len(_ATOM_SITE_PREFIX) :])
            index += 1

        if not columns:
            continue

        while index < len(lines):
            row = lines[index].strip()
            if not row or row.startswith("#") or row.startswith("_") or row == "loop_":
                break
            values = _split_cif_row(row)
            if len(values) == len(columns):
                records.append(dict(zip(columns, values, strict=True)))
            index += 1

    return records


def _record_value(record: dict[str, str], *names: str, default: str = "") -> str:
    """Returns the first present, non-placeholder value among `names`.

    mmCIF writes `.` and `?` for absent values, which must not leak into a
    PDB field as literal text.
    """

    for name in names:
        value = record.get(name, "")
        if value and value not in {".", "?"}:
            return value
    return default


def _pdb_atom_line(record: dict[str, str], serial: int) -> str | None:
    """Renders one parsed mmCIF atom as a fixed-column PDB line."""

    try:
        x = float(record["Cartn_x"])
        y = float(record["Cartn_y"])
        z = float(record["Cartn_z"])
    except (KeyError, ValueError):
        return None

    group = record.get("group_PDB", "ATOM")
    element = _record_value(record, "type_symbol").upper()
    name = _record_value(record, "label_atom_id", "auth_atom_id", default=element)
    residue = _record_value(record, "label_comp_id", "auth_comp_id", default="UNL")
    chain = _record_value(record, "auth_asym_id", "label_asym_id", default="A")[:1] or "A"
    sequence = _record_value(record, "auth_seq_id", "label_seq_id", default="1")

    try:
        residue_number = int(sequence)
    except ValueError:
        residue_number = 1

    # PDB puts one-letter elements in column 14, leaving column 13 blank; a
    # name written flush-left there would be read back as a different element.
    if len(name) < 4 and len(element) == 1:
        name = f" {name}"

    # PDB is fixed-column: the residue name field is three characters and the
    # atom name four. mmCIF has no such limit, and predictors do exceed it --
    # Boltz names its ligand "LIG1". An over-long value does not merely look
    # wrong, it shifts every following field one place right, so coordinates
    # are silently truncated and the structure becomes unreadable.
    residue = residue[:3]
    name = name[:4]

    return (
        f"{group:<6}{serial:>5} {name:<4}{'':1}{residue:>3} "
        f"{chain:1}{residue_number:>4}{'':1}   "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{0.0:>6.2f}          {element:>2}"
    )


def _is_ligand_record(record: dict[str, str]) -> bool:
    """Reports whether an atom belongs to a docked ligand rather than solvent."""

    if record.get("group_PDB", "").upper() != "HETATM":
        return False
    residue = _record_value(record, "label_comp_id", "auth_comp_id").upper()
    return residue not in _NON_LIGAND_RESIDUES


def _build_ligand(pdb_block: str, ligand_smiles: str) -> tuple[Any, bool]:
    """Builds a ligand molecule with 3D coordinates and correct bond orders.

    Proximity bonding recovers connectivity from the coordinates, but not bond
    orders -- an aromatic ring comes back as single bonds. Those are restored
    from the submitted SMILES, which is the only reliable record of what the
    molecule was meant to be.

    Returns:
        The ligand and whether bond orders were successfully assigned. A
        ligand is still returned when assignment fails, because a pose with
        approximate bonding is more useful to a geometry check than none.
    """

    pose = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False, proximityBonding=True)
    if pose is None:
        return None, False

    try:
        pose = Chem.RemoveHs(pose, sanitize=False)
    except Exception:  # noqa: BLE001 - RDKit raises bare exceptions here
        pass

    template = Chem.MolFromSmiles(ligand_smiles)
    if template is None:
        return pose, False

    try:
        return AllChem.AssignBondOrdersFromTemplate(template, pose), True
    except Exception:  # noqa: BLE001
        logger.debug("Bond order assignment failed for ligand %s", ligand_smiles)
        return pose, False


def split_complex(mmcif_text: str, ligand_smiles: str) -> ComplexParts | None:
    """Splits a predicted complex into a protein PDB block and a ligand molecule.

    Args:
        mmcif_text: mmCIF content as returned by the co-folding predictor.
        ligand_smiles: SMILES submitted for the ligand, used to restore bond
            orders that coordinates alone cannot supply.

    Returns:
        The split complex, or `None` when the file yields no atoms, no
        protein, or no ligand -- any of which makes a pose check meaningless.
    """

    records = parse_atom_site_records(mmcif_text)
    if not records:
        logger.debug("No _atom_site records found in predicted structure")
        return None

    protein_lines: list[str] = []
    ligand_lines: list[str] = []
    for record in records:
        target = ligand_lines if _is_ligand_record(record) else protein_lines
        if record.get("group_PDB", "").upper() == "HETATM" and target is protein_lines:
            continue  # solvent or ion: belongs to neither half
        line = _pdb_atom_line(record, serial=len(protein_lines) + len(ligand_lines) + 1)
        if line is not None:
            target.append(line)

    if not protein_lines or not ligand_lines:
        logger.debug(
            "Predicted structure split yielded %s protein and %s ligand atom(s)",
            len(protein_lines),
            len(ligand_lines),
        )
        return None

    ligand, bond_orders_assigned = _build_ligand("\n".join(ligand_lines) + "\nEND\n", ligand_smiles)
    if ligand is None:
        return None

    return ComplexParts(
        protein_pdb_block="\n".join(protein_lines) + "\nEND\n",
        ligand=ligand,
        protein_atom_count=len(protein_lines),
        ligand_atom_count=len(ligand_lines),
        bond_orders_assigned=bond_orders_assigned,
    )
