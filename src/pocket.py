"""Protein pocket discovery and residue extraction helpers."""

import glob
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from src.core.constants import (
    DEFAULT_DEEPCHEM_POCKET_PAD,
    DEFAULT_P2RANK_OUTPUT_DIRNAME,
    DEFAULT_POCKET_BOX_MARGIN_ANGSTROM,
    DEFAULT_POCKET_CONTACT_DISTANCE,
    DEFAULT_POCKET_MAX_BOX_ANGSTROM,
    DEFAULT_POCKET_MIN_BOX_ANGSTROM,
)
from src.core.exceptions import PocketDetectionError
from src.core.logging import get_logger
from src.core.messages import (
    empty_pocket_residue_selection,
    no_pocket_residues_matched,
    no_residues_near_pocket_center,
    no_residues_near_pocket_ligand,
    p2rank_executable_not_found,
    pocket_ligand_not_found,
    pocket_structure_unreadable,
    solvent_pocket_ligand,
    unparsable_pocket_residue,
)

logger = get_logger(__name__)


def setup_p2rank(search_root: str | Path | None = None) -> str:
    """Locates the `prank` executable and ensures execute permissions.

    Returns:
        Absolute path to the `prank` executable.

    Raises:
        PocketDetectionError: If no executable is found in project paths.
    """

    base_dir = Path(search_root) if search_root is not None else Path.cwd()
    script_path = base_dir / "prank" / "prank"

    if not script_path.exists():
        existing = glob.glob(str(base_dir / "p2rank*" / "prank"), recursive=True)
        if existing:
            script_path = Path(existing[0])
        else:
            raise PocketDetectionError(p2rank_executable_not_found(base_dir))

    current_mode = os.stat(script_path).st_mode
    os.chmod(script_path, current_mode | stat.S_IEXEC)
    return str(script_path)


def get_p2rank_pocket(pdb_path: str | Path, output_dir: str | Path = ".") -> str:
    """Runs P2Rank and returns top predicted pocket residue identifiers.

    Args:
        pdb_path: Path to input PDB file.

    Returns:
        P2Rank residue identifiers for the best prediction, if available.
    """

    p2rank_executable = setup_p2rank()
    p2rank_output_dir = Path(output_dir) / DEFAULT_P2RANK_OUTPUT_DIRNAME
    p2rank_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running P2Rank analysis on %s...", pdb_path)
    cmd = [
        p2rank_executable,
        "predict",
        "-f",
        os.fspath(pdb_path),
        "-o",
        os.fspath(p2rank_output_dir),
        "-visualizations",
        "0",
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    pdb_filename = Path(pdb_path).name
    pred_file = p2rank_output_dir / f"{pdb_filename}_predictions.csv"
    if not pred_file.exists():
        return "Unknown Pocket"

    pocket_df = pd.read_csv(pred_file)
    pocket_df.columns = pocket_df.columns.str.strip()
    if pocket_df.empty:
        return "Unknown Pocket"
    return str(pocket_df.iloc[0]["residue_ids"])


def _pocket_dimensions(pocket: Any) -> tuple[float, float, float]:
    """Returns a pocket's axis-aligned (x, y, z) extents in Angstroms."""

    return (
        pocket.x_range[1] - pocket.x_range[0],
        pocket.y_range[1] - pocket.y_range[0],
        pocket.z_range[1] - pocket.z_range[0],
    )


def _pocket_volume(pocket: Any) -> float:
    """Estimates pocket volume from axis-aligned ranges."""

    size_x, size_y, size_z = _pocket_dimensions(pocket)
    return size_x * size_y * size_z


def _meets_minimum_dimensions(pocket: Any, min_dimension: float) -> bool:
    """Checks that every axis of a pocket is at least `min_dimension` wide."""

    return all(size >= min_dimension for size in _pocket_dimensions(pocket))


def _select_pocket(
    pockets: list[Any],
    pocket_data: list[dict[str, float]],
    pocket_index: int,
    min_dimension: float = DEFAULT_POCKET_MIN_BOX_ANGSTROM,
) -> Any:
    """Selects a pocket, preferring the smallest volume among pockets that meet
    the minimum per-axis dimension -- this minimizes the docking search space
    while still guaranteeing a ligand-sized cavity. Falls back to the largest
    pocket overall when none qualify, since a degenerate near-zero-volume
    artifact would be a worse default than an oversized one.
    """

    logger.info("Detected %s candidate pockets", len(pocket_data))
    for row in pocket_data:
        logger.debug(
            "Pocket %s center=(%.2f, %.2f, %.2f) volume≈%.2f A^3",
            int(row["pocket_id"]),
            row["center_x"],
            row["center_y"],
            row["center_z"],
            row["volume_approx"],
        )

    if 0 <= pocket_index < len(pockets):
        logger.info("Selected configured pocket index %s.", pocket_index)
        return pockets[pocket_index]

    qualifying = [pocket for pocket in pockets if _meets_minimum_dimensions(pocket, min_dimension)]
    if qualifying:
        logger.info(
            "%s/%s pocket(s) meet the %.1fA minimum per-axis dimension; "
            "selecting the smallest by volume.",
            len(qualifying),
            len(pockets),
            min_dimension,
        )
        return min(qualifying, key=_pocket_volume)

    logger.warning(
        "No detected pocket meets the %.1fA minimum per-axis dimension; "
        "falling back to the largest pocket overall.",
        min_dimension,
    )
    return max(pockets, key=_pocket_volume)


def _expand_box_dimension(size: float, margin: float, max_dimension: float) -> float:
    """Isotropically expands a docking box dimension by `margin`, capped at the maximum."""

    return min(size + margin, max_dimension)


def get_binding_pockets_and_residues(
    pdb_path: str | Path,
    output_dir: str | Path = "results",
    backend: str = "deepchem",
    pocket_index: int = -1,
) -> tuple[str, str, dict[str, float] | None]:
    """Finds pockets and residues near the chosen pocket, using two
    complementary criteria: (i) a heavy atom within the expanded docking
    box, or (ii) any atom within `DEFAULT_POCKET_CONTACT_DISTANCE` of the
    pocket center.

    Args:
        pdb_path: Path to target protein PDB file.
        output_dir: Directory where detected pocket summary is saved.
        backend: Pocket detection backend. ``"deepchem"`` (the default, and
            the only backend the pipeline itself uses) runs DeepChem's
            ConvexHullPocketFinder and returns full geometry. ``"p2rank"``
            is an optional alternative that shells out to an external
            ``prank`` binary; it returns only a residue list, with no
            center or box dimensions, and is not used by
            `MoleculeGenerator.run`.
        pocket_index: Explicit 0-based pocket index to force. Any value
            outside `[0, len(pockets))` -- including the default `-1` --
            triggers automatic selection instead.

    Returns:
        Tuple containing pocket center string, nearby residue list string, and
        an optional dict with box dimensions ``{size_x, size_y, size_z}`` in
        Angstroms (only populated for the deepchem backend).
    """

    if backend == "p2rank":
        residues = get_p2rank_pocket(pdb_path, output_dir=output_dir)
        return "P2Rank", residues, None

    # Lazy import avoids importing DeepChem at module import time.
    import deepchem as dc

    finder = dc.dock.ConvexHullPocketFinder(pad=DEFAULT_DEEPCHEM_POCKET_PAD)
    pockets = finder.find_pockets(str(pdb_path))
    if not pockets:
        return "No pockets found", "Unknown", None

    pocket_data: list[dict[str, float]] = []
    for idx, p in enumerate(pockets, start=1):
        center = p.center()
        pocket_data.append(
            {
                "pocket_id": float(idx),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "center_z": float(center[2]),
                "size_x": float(p.x_range[1] - p.x_range[0]),
                "size_y": float(p.y_range[1] - p.y_range[0]),
                "size_z": float(p.z_range[1] - p.z_range[0]),
                "volume_approx": float(_pocket_volume(p)),
            }
        )

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(pocket_data).to_csv(os.path.join(output_dir, "all_pockets.csv"), index=False)

    best_pocket = _select_pocket(pockets, pocket_data, pocket_index=pocket_index)
    center = np.asarray(best_pocket.center(), dtype=float)

    raw_size_x, raw_size_y, raw_size_z = _pocket_dimensions(best_pocket)
    size_x = _expand_box_dimension(
        raw_size_x, DEFAULT_POCKET_BOX_MARGIN_ANGSTROM, DEFAULT_POCKET_MAX_BOX_ANGSTROM
    )
    size_y = _expand_box_dimension(
        raw_size_y, DEFAULT_POCKET_BOX_MARGIN_ANGSTROM, DEFAULT_POCKET_MAX_BOX_ANGSTROM
    )
    size_z = _expand_box_dimension(
        raw_size_z, DEFAULT_POCKET_BOX_MARGIN_ANGSTROM, DEFAULT_POCKET_MAX_BOX_ANGSTROM
    )
    if (size_x, size_y, size_z) != (raw_size_x, raw_size_y, raw_size_z):
        logger.info(
            "Adjusted docking box from (%.2f, %.2f, %.2f)A to (%.2f, %.2f, %.2f)A "
            "(margin=+%.1fA, max=%.1fA per axis)",
            raw_size_x,
            raw_size_y,
            raw_size_z,
            size_x,
            size_y,
            size_z,
            DEFAULT_POCKET_BOX_MARGIN_ANGSTROM,
            DEFAULT_POCKET_MAX_BOX_ANGSTROM,
        )

    pose_info: dict[str, float] = {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
        "raw_size_x": raw_size_x,
        "raw_size_y": raw_size_y,
        "raw_size_z": raw_size_z,
    }

    mol = Chem.MolFromPDBFile(os.fspath(pdb_path))
    if mol is None:
        pocket_desc = f"Center: {center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}"
        return pocket_desc, "Unknown", pose_info

    conformer = mol.GetConformer()
    nearby_residues: set[str] = set()
    half_extents = np.array([size_x, size_y, size_z], dtype=float) / 2.0

    for atom in mol.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        coords = np.array([position.x, position.y, position.z])
        offset = coords - center

        within_radius = np.linalg.norm(offset) <= DEFAULT_POCKET_CONTACT_DISTANCE
        # Heavy-atom-in-expanded-box criterion, complementary to the radius
        # criterion above: a residue qualifies if either one holds.
        within_expanded_box = atom.GetAtomicNum() != 1 and bool(
            np.all(np.abs(offset) <= half_extents)
        )
        if not (within_radius or within_expanded_box):
            continue

        info = atom.GetPDBResidueInfo()
        if info:
            residue_name = info.GetResidueName().strip()
            residue_number = info.GetResidueNumber()
            chain_id = info.GetChainId().strip()
            nearby_residues.add(f"{residue_name}{residue_number}_{chain_id}")

    residue_list = ", ".join(sorted(nearby_residues))
    pocket_desc = f"Center: {center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}"
    return pocket_desc, residue_list, pose_info


# --- User-defined pockets ---------------------------------------------------
#
# Geometry-based detection answers "where could a ligand sit?", which is the
# right question only when nothing is known about the target. Whenever a
# binding site is already established -- from a co-crystallised ligand, from
# mutagenesis, or from the literature -- that knowledge beats a hull search,
# and forcing a run through automatic detection just to reach a site the user
# could name is both slower and less reliable.
#
# Every mode below returns the same triple as `get_binding_pockets_and_residues`
# so the pipeline treats a user-defined pocket exactly like a detected one,
# including the chain filtering and sequence-position translation that follow.

_RESIDUE_TOKEN_PATTERNS = (
    # "ALA203_A" -- the form detection itself emits, so a residue list can be
    # copied out of one run's manifest and pinned for the next.
    re.compile(r"^(?P<name>[A-Za-z]{3})(?P<number>\d+)_(?P<chain>[A-Za-z0-9])$"),
    # "A:203" -- chain-qualified, the form a user is most likely to type.
    re.compile(r"^(?P<chain>[A-Za-z0-9]):(?P<number>\d+)$"),
    # "203" -- bare number, resolved against the target chain.
    re.compile(r"^(?P<number>\d+)$"),
)


def parse_pocket_residue_tokens(
    text: str, *, default_chain: str | None = None
) -> list[tuple[str | None, int]]:
    """Parses a user residue selection into (chain, residue number) pairs.

    Accepts comma- or whitespace-separated tokens in any of three forms:
    ``ALA203_A`` (as detection reports them), ``A:203``, or a bare ``203``
    resolved against `default_chain`.

    Raises:
        PocketDetectionError: If a token matches none of the forms. A typo
            must not silently drop a residue the user asked to constrain.
    """

    pairs: list[tuple[str | None, int]] = []
    for token in (part.strip() for part in re.split(r"[,\s]+", text) if part.strip()):
        for pattern in _RESIDUE_TOKEN_PATTERNS:
            match = pattern.match(token)
            if match is None:
                continue
            groups = match.groupdict()
            chain = groups.get("chain") or default_chain
            pairs.append((chain.upper() if chain else None, int(groups["number"])))
            break
        else:
            raise PocketDetectionError(unparsable_pocket_residue(token))
    if not pairs:
        raise PocketDetectionError(empty_pocket_residue_selection())
    return pairs


def _load_pdb_mol(pdb_path: str | Path) -> Any:
    """Loads a PDB for geometry queries, tolerating unsanitisable structures.

    Experimental structures routinely fail RDKit sanitisation over valence
    issues in ligands or modified residues. Those failures say nothing about
    whether the coordinates are usable, which is all this module needs.
    """

    mol = Chem.MolFromPDBFile(os.fspath(pdb_path), removeHs=False)
    if mol is None:
        mol = Chem.MolFromPDBFile(os.fspath(pdb_path), sanitize=False, removeHs=False)
    if mol is None:
        raise PocketDetectionError(pocket_structure_unreadable(pdb_path))
    return mol


def _residue_label(info: Any) -> str:
    """Formats one atom's residue as the ``ALA203_A`` label used downstream."""

    return f"{info.GetResidueName().strip()}{info.GetResidueNumber()}_{info.GetChainId().strip()}"


def _box_from_points(points: np.ndarray) -> dict[str, float]:
    """Builds a docking box enclosing `points`, with the standard margin."""

    center = points.mean(axis=0)
    raw = points.max(axis=0) - points.min(axis=0)
    sizes = [
        _expand_box_dimension(
            float(extent), DEFAULT_POCKET_BOX_MARGIN_ANGSTROM, DEFAULT_POCKET_MAX_BOX_ANGSTROM
        )
        for extent in raw
    ]
    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "size_x": sizes[0],
        "size_y": sizes[1],
        "size_z": sizes[2],
        "raw_size_x": float(raw[0]),
        "raw_size_y": float(raw[1]),
        "raw_size_z": float(raw[2]),
    }


def _residues_within(
    mol: Any, center: np.ndarray, distance: float
) -> tuple[set[str], list[np.ndarray]]:
    """Returns residues with any atom within `distance` of `center`."""

    conformer = mol.GetConformer()
    residues: set[str] = set()
    coords: list[np.ndarray] = []
    for atom in mol.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        point = np.array([position.x, position.y, position.z])
        if np.linalg.norm(point - center) > distance:
            continue
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        residues.add(_residue_label(info))
        coords.append(point)
    return residues, coords


def _ligand_atom_coords(mol: Any, selector: str) -> list[np.ndarray]:
    """Returns coordinates of the heteroatom group named by `selector`.

    `selector` is a residue name such as ``ATP``, optionally narrowed to one
    copy as ``ATP:A`` or ``ATP:A:401`` when the structure holds several.
    Water is never matched, so a solvent-named selector finds nothing rather
    than silently centring the pocket on a water molecule.
    """

    parts = [part.strip() for part in selector.split(":") if part.strip()]
    want_name = parts[0].upper()
    want_chain = parts[1].upper() if len(parts) > 1 else None
    want_number = int(parts[2]) if len(parts) > 2 else None

    if want_name in {"HOH", "WAT", "DOD"}:
        raise PocketDetectionError(solvent_pocket_ligand(selector))

    conformer = mol.GetConformer()
    coords: list[np.ndarray] = []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None or not info.GetIsHeteroAtom():
            continue
        if info.GetResidueName().strip().upper() != want_name:
            continue
        if want_chain is not None and info.GetChainId().strip().upper() != want_chain:
            continue
        if want_number is not None and info.GetResidueNumber() != want_number:
            continue
        position = conformer.GetAtomPosition(atom.GetIdx())
        coords.append(np.array([position.x, position.y, position.z]))
    return coords


def _residue_names_by_position(mol: Any) -> dict[tuple[str, int], str]:
    """Maps (chain, residue number) onto the residue's three-letter name."""

    names: dict[tuple[str, int], str] = {}
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        key = (info.GetChainId().strip().upper(), info.GetResidueNumber())
        names.setdefault(key, info.GetResidueName().strip())
    return names


def _residue_atom_coords(mol: Any, wanted: set[tuple[str, int]]) -> list[np.ndarray]:
    """Returns coordinates of every atom belonging to the wanted residues."""

    conformer = mol.GetConformer()
    coords: list[np.ndarray] = []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        if (info.GetChainId().strip().upper(), info.GetResidueNumber()) in wanted:
            position = conformer.GetAtomPosition(atom.GetIdx())
            coords.append(np.array([position.x, position.y, position.z]))
    return coords


def get_pocket_from_residues(
    pdb_path: str | Path,
    selection: str,
    *,
    default_chain: str | None = None,
) -> tuple[str, str, dict[str, float] | None]:
    """Builds a pocket from an explicit residue list.

    Residues the structure does not contain are dropped with a warning
    rather than passed on: a residue number that is not in the file cannot
    be translated to a sequence position, and sending it would constrain
    the prediction to a site that does not exist.

    Raises:
        PocketDetectionError: If no requested residue is found in the file.
    """

    pairs = parse_pocket_residue_tokens(selection, default_chain=default_chain)
    mol = _load_pdb_mol(pdb_path)
    names = _residue_names_by_position(mol)

    labels: list[str] = []
    found: set[tuple[str, int]] = set()
    missing: list[str] = []
    for chain, number in pairs:
        candidates = [(chain, number)] if chain else [key for key in names if key[1] == number]
        matched = [key for key in candidates if key in names]
        if not matched:
            missing.append(f"{chain or '?'}:{number}")
            continue
        for key in matched:
            found.add(key)
            labels.append(f"{names[key]}{key[1]}_{key[0]}")

    if missing:
        logger.warning(
            "%s requested pocket residue(s) are not present in %s and were dropped: %s",
            len(missing),
            pdb_path,
            ", ".join(missing),
        )
    if not labels:
        raise PocketDetectionError(no_pocket_residues_matched(pdb_path))

    coords = _residue_atom_coords(mol, found)
    box = _box_from_points(np.array(coords)) if coords else None
    center_text = (
        f"Center: {box['center_x']:.2f}, {box['center_y']:.2f}, {box['center_z']:.2f}"
        if box
        else "User-defined residues"
    )
    logger.info(
        "Using user-defined pocket from %s residue(s): %s",
        len(labels),
        ", ".join(sorted(labels)),
    )
    return center_text, ", ".join(sorted(set(labels))), box


def get_pocket_from_ligand(
    pdb_path: str | Path,
    selector: str,
    *,
    contact_distance: float = DEFAULT_POCKET_CONTACT_DISTANCE,
) -> tuple[str, str, dict[str, float] | None]:
    """Builds a pocket from the residues surrounding a bound ligand.

    This is the case where prior knowledge is strongest: a co-crystallised
    ligand marks the site experimentally, so the pocket is read off the
    structure instead of being guessed from its shape.

    Raises:
        PocketDetectionError: If no heteroatom group matches `selector`.
    """

    mol = _load_pdb_mol(pdb_path)
    ligand_coords = _ligand_atom_coords(mol, selector)
    if not ligand_coords:
        raise PocketDetectionError(pocket_ligand_not_found(selector, pdb_path))

    points = np.array(ligand_coords)
    center = points.mean(axis=0)
    residues, _ = _residues_within(mol, center, contact_distance)
    # The ligand's own atoms are not pocket residues.
    ligand_name = selector.split(":")[0].strip().upper()
    residues = {label for label in residues if not label.upper().startswith(ligand_name)}
    if not residues:
        raise PocketDetectionError(no_residues_near_pocket_ligand(selector, contact_distance))

    box = _box_from_points(points)
    logger.info(
        "Using pocket around ligand '%s' at (%.2f, %.2f, %.2f) with %s contact residue(s)",
        selector,
        center[0],
        center[1],
        center[2],
        len(residues),
    )
    return (
        f"Ligand {selector} center: {center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}",
        ", ".join(sorted(residues)),
        box,
    )


def get_pocket_from_center(
    pdb_path: str | Path,
    center: tuple[float, float, float],
    *,
    contact_distance: float = DEFAULT_POCKET_CONTACT_DISTANCE,
) -> tuple[str, str, dict[str, float] | None]:
    """Builds a pocket from a point in the structure's coordinate frame.

    Raises:
        PocketDetectionError: If no residue lies within `contact_distance`,
            which usually means the coordinates belong to a different frame
            than the supplied structure.
    """

    mol = _load_pdb_mol(pdb_path)
    origin = np.asarray(center, dtype=float)
    residues, coords = _residues_within(mol, origin, contact_distance)
    if not residues:
        raise PocketDetectionError(no_residues_near_pocket_center(center, contact_distance))

    box = _box_from_points(np.array(coords))
    logger.info(
        "Using pocket at (%.2f, %.2f, %.2f) with %s residue(s) within %.1fA",
        origin[0],
        origin[1],
        origin[2],
        len(residues),
        contact_distance,
    )
    return (
        f"Center: {origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f}",
        ", ".join(sorted(residues)),
        box,
    )
