"""Protein pocket discovery and residue extraction helpers."""

import glob
import os
import stat
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from src.core.logging import get_logger

logger = get_logger(__name__)


def setup_p2rank() -> str:
    """Locates the `prank` executable and ensures execute permissions.

    Returns:
        Absolute path to the `prank` executable.

    Raises:
        FileNotFoundError: If no executable is found in project paths.
    """

    base_dir = os.getcwd()
    script_path = os.path.join(base_dir, "prank", "prank")

    if not os.path.exists(script_path):
        existing = glob.glob(os.path.join(base_dir, "p2rank*", "prank"), recursive=True)
        if existing:
            script_path = existing[0]
        else:
            raise FileNotFoundError("P2Rank executable `prank` not found in project root.")

    current_mode = os.stat(script_path).st_mode
    os.chmod(script_path, current_mode | stat.S_IEXEC)
    return script_path


def get_p2rank_pocket(pdb_path: str) -> str:
    """Runs P2Rank and returns top predicted pocket residue identifiers.

    Args:
        pdb_path: Path to input PDB file.

    Returns:
        P2Rank residue identifiers for the best prediction, if available.
    """

    p2rank_executable = setup_p2rank()
    output_dir = os.path.abspath("p2rank_output")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Running P2Rank analysis on %s...", pdb_path)
    cmd = [p2rank_executable, "predict", "-f", pdb_path, "-o", output_dir, "-visualizations", "0"]
    subprocess.run(cmd, check=True, capture_output=True)

    pdb_filename = os.path.basename(pdb_path)
    pred_file = os.path.join(output_dir, f"{pdb_filename}_predictions.csv")
    if not os.path.exists(pred_file):
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


def _select_pocket_interactively(pockets: list[Any], pocket_data: list[dict[str, float]]) -> Any:
    """Prompts user for pocket selection while preserving previous behavior."""

    logger.info("Detected pockets:")
    for row in pocket_data:
        logger.info(
            f"[{int(row['pocket_id'])}] Center: ({row['center_x']:.2f}, {row['center_y']:.2f}, "
            f"{row['center_z']:.2f}), Volume ≈ {row['volume_approx']:.2f} Å³"
        )

    while True:
        try:
            choice = int(input(f"\nSelect pocket ID to use (1-{len(pockets)}), or 0 to use largest volume: "))
            if choice == 0:
                logger.info("Selected largest volume pocket automatically.")
                return max(pockets, key=_pocket_volume)
            if 1 <= choice <= len(pockets):
                logger.info("Selected pocket %s.", choice)
                return pockets[choice - 1]
            logger.warning("Invalid choice. Try again.")
        except ValueError:
            logger.warning("Please enter a number.")


def get_binding_pockets_and_residues(pdb_path: str, output_dir: str = "results") -> tuple[str, str]:
    """Finds pockets and residues within 8A around chosen pocket center.

    Args:
        pdb_path: Path to target protein PDB file.
        output_dir: Directory where detected pocket summary is saved.

    Returns:
        Tuple containing pocket center string and nearby residue list string.
    """

    # Lazy import avoids importing DeepChem at module import time.
    import deepchem as dc

    finder = dc.dock.ConvexHullPocketFinder(pad=5.0)
    pockets = finder.find_pockets(pdb_path)
    if not pockets:
        return "No pockets found", "Unknown"

    pocket_data: list[dict[str, float]] = []
    for idx, pocket in enumerate(pockets, start=1):
        center = pocket.center()
        pocket_data.append(
            {
                "pocket_id": float(idx),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "center_z": float(center[2]),
                "volume_approx": float(_pocket_volume(pocket)),
            }
        )

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(pocket_data).to_csv(os.path.join(output_dir, "all_pockets.csv"), index=False)

    best_pocket = _select_pocket_interactively(pockets, pocket_data)
    center = np.asarray(best_pocket.center(), dtype=float)

    mol = Chem.MolFromPDBFile(pdb_path)
    if mol is None:
        return f"Center: {center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}", "Unknown"

    conformer = mol.GetConformer()
    nearby_residues: set[str] = set()

    for atom in mol.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        distance = np.linalg.norm(np.array([position.x, position.y, position.z]) - center)
        if distance > 8.0:
            continue

        info = atom.GetPDBResidueInfo()
        if info:
            residue_name = info.GetResidueName().strip()
            residue_number = info.GetResidueNumber()
            chain_id = info.GetChainId().strip()
            nearby_residues.add(f"{residue_name}{residue_number}_{chain_id}")

    residue_list = ", ".join(sorted(nearby_residues))
    pocket_desc = f"Center: {center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}"
    return pocket_desc, residue_list
