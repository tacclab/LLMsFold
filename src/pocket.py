"""Protein pocket discovery and residue extraction helpers."""

import glob
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from src.core.constants import (
    DEFAULT_DEEPCHEM_POCKET_PAD,
    DEFAULT_P2RANK_OUTPUT_DIRNAME,
    DEFAULT_POCKET_CONTACT_DISTANCE,
)
from src.core.exceptions import PocketDetectionError
from src.core.logging import get_logger
from src.core.messages import p2rank_executable_not_found

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


def _pocket_volume(pocket: Any) -> float:
    """Estimates pocket volume from axis-aligned ranges."""

    return (
        (pocket.x_range[1] - pocket.x_range[0])
        * (pocket.y_range[1] - pocket.y_range[0])
        * (pocket.z_range[1] - pocket.z_range[0])
    )


def _select_pocket(
    pockets: list[Any], pocket_data: list[dict[str, float]], pocket_index: int
) -> Any:
    """Selects a pocket index, defaulting to largest volume when index is invalid."""

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

    logger.info("Pocket index %s out of range, defaulting to largest volume.", pocket_index)
    return max(pockets, key=_pocket_volume)


def _pick_pocket_id_from_cli(pocket_data: list[dict[str, float]]) -> int | None:
    """Prompts interactive users to choose a pocket id, defaulting to largest pocket."""

    if not sys.stdin.isatty() or not pocket_data:
        return None

    suggested = max(pocket_data, key=lambda row: row["volume_approx"])
    suggested_id = int(suggested["pocket_id"])
    response = input(
        f"Suggested pocket id: {suggested_id} (largest volume). "
        "Press Enter to accept or type another pocket id: "
    ).strip()
    if not response:
        return suggested_id

    try:
        selected_id = int(response)
    except ValueError:
        logger.warning("Invalid pocket id '%s'; using suggested pocket id %s.", response, suggested_id)
        return suggested_id

    if any(int(row["pocket_id"]) == selected_id for row in pocket_data):
        return selected_id

    logger.warning("Pocket id %s not found; using suggested pocket id %s.", selected_id, suggested_id)
    return suggested_id


def get_binding_pockets_and_residues(
    pdb_path: str | Path,
    output_dir: str | Path = "results",
    backend: str = "p2rank",
    pocket_index: int = 0,
) -> tuple[str, str, dict[str, float] | None]:
    """Finds pockets and residues within 8A around chosen pocket center.

    Args:
        pdb_path: Path to target protein PDB file.
        output_dir: Directory where detected pocket summary is saved.

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

    selected_pocket_id = _pick_pocket_id_from_cli(pocket_data)
    if selected_pocket_id is not None:
        best_pocket = pockets[selected_pocket_id - 1]
    else:
        best_pocket = _select_pocket(pockets, pocket_data, pocket_index=pocket_index)
    center = np.asarray(best_pocket.center(), dtype=float)

    pose_info: dict[str, float] = {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "size_x": float(best_pocket.x_range[1] - best_pocket.x_range[0]),
        "size_y": float(best_pocket.y_range[1] - best_pocket.y_range[0]),
        "size_z": float(best_pocket.z_range[1] - best_pocket.z_range[0]),
    }

    mol = Chem.MolFromPDBFile(os.fspath(pdb_path))
    if mol is None:
        pocket_desc = f"Center: {center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}"
        return pocket_desc, "Unknown", pose_info

    conformer = mol.GetConformer()
    nearby_residues: set[str] = set()

    for atom in mol.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        distance = np.linalg.norm(np.array([position.x, position.y, position.z]) - center)
        if distance > DEFAULT_POCKET_CONTACT_DISTANCE:
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
