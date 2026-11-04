"""Shared warning and error message builders."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx


def missing_required_setting(setting_name: str) -> str:
    """Returns a standardized missing-setting error message."""

    return f"Missing required configuration: {setting_name}"


def missing_environment_configuration() -> str:
    """Returns the top-level CLI configuration error message."""

    return "Missing required environment variables in `.env` or shell"


def invalid_runtime_options() -> str:
    """Returns the top-level runtime options error message."""

    return "Invalid runtime options"


def pdb_file_not_found(path: str | Path) -> str:
    """Returns a consistent missing-PDB message."""

    return f"PDB file not found at {path}"


def pdb_file_unreadable(path: str | Path) -> str:
    """Returns a sequence-extraction error for unreadable PDBs."""

    return f"Could not open or read PDB file: {path}"


def pdb_sequence_missing(path: str | Path) -> str:
    """Returns a sequence-extraction error for PDBs without residues."""

    return f"No SEQRES or ATOM residue records found in {path}"


def p2rank_executable_not_found(search_root: str | Path) -> str:
    """Returns a pocket-detection error for missing P2Rank binaries."""

    return f"P2Rank executable `prank` not found under {search_root}"


def pocket_detection_unavailable(reason: Exception) -> str:
    """Returns a fallback warning when pocket detection cannot run."""

    return f"Pocket detection unavailable ({reason}). Continuing without pocket constraints."


def invalid_smiles(smiles: str) -> str:
    """Returns a warning for invalid generated SMILES."""

    return f"Skipping invalid or un-kekulizable SMILES: {smiles}"


def no_molecules_generated() -> str:
    """Returns the warning used when no candidates survive the workflow."""

    return (
        "No molecules survived scoring and filtering. Check the iteration summaries and thresholds."
    )


def no_smiles_parsed() -> str:
    """Returns a concise warning for unparseable LLM output."""

    return "Could not parse any SMILES from LLM output"


def boltz_task_timeout(task_id: str, seconds: float) -> str:
    """Returns a timeout error for long-running Boltz jobs."""

    return f"Boltz task {task_id} timed out after {seconds:.1f}s"


def invalid_boltz_response(smiles: str) -> str:
    """Returns a warning for malformed Boltz responses."""

    return f"Discarding invalid Boltz response for {smiles}"


def boltz_structure_missing(smiles: str) -> str:
    """Returns a warning when Boltz returns scores but no structure payload."""

    return (
        f"Boltz returned affinity scores for {smiles} but no 'structure' or 'pdb' field "
        "in the response — structure cannot be persisted for this molecule."
    )


def persist_structure_unavailable(smiles: str, candidate_id: str) -> str:
    """Returns a warning when a high-affinity molecule has no stored structure."""

    return (
        f"No Boltz structure payload available for {candidate_id} ({smiles}); "
        "skipping structure save. Boltz may not have returned structure data for this molecule."
    )


def scaffold_analysis_failed(reason: Exception) -> str:
    """Returns a warning when scaffold diversity analysis could not run."""

    return (
        f"Scaffold diversity analysis unavailable ({reason}). The report is written "
        "without scaffold columns; run-level diversity statistics are omitted."
    )


def reference_scoring_failed(reason: Exception) -> str:
    """Returns a warning when the reference baseline could not be produced."""

    return (
        f"Reference inhibitor baseline could not be scored ({reason}). "
        "Candidate generation continues; no baseline report will be written."
    )


def no_reference_scores_returned() -> str:
    """Returns a warning when Boltz returned nothing for every reference."""

    return (
        "Boltz returned no usable scores for any reference inhibitor; skipping the baseline report."
    )


def pubchem_query_error(smiles: str, exc: Exception) -> str:
    """Returns a warning for degraded PubChem verification."""

    return f"PubChem verification degraded for {smiles}: {exc}"


def holdout_evaluator_unavailable(name: str, reason: str | None) -> str:
    """Returns a warning when an independent evaluator cannot run."""

    return (
        f"Holdout evaluator '{name}' unavailable: {reason or 'no reason reported'}. "
        "Its columns will be empty; results for this run rest on the in-loop "
        "predictor alone."
    )


def holdout_evaluator_failed(name: str, reason: Exception) -> str:
    """Returns a warning when an available evaluator raised during evaluation."""

    return (
        f"Holdout evaluator '{name}' failed during evaluation ({reason}). "
        "The report is written without its columns."
    )


def unparsable_pocket_residue(token: str) -> str:
    """Returns an error for a residue token in none of the accepted forms."""

    return (
        f"Could not read pocket residue '{token}'. Use a chain-qualified number "
        "('A:203'), a bare number ('203'), or the detected form ('ALA203_A')."
    )


def empty_pocket_residue_selection() -> str:
    """Returns an error when a residue selection contains no tokens."""

    return "Pocket residue selection is empty; give at least one residue."


def pocket_structure_unreadable(path: str | Path) -> str:
    """Returns an error when a PDB cannot be parsed for pocket geometry."""

    return f"Could not read structure for pocket selection: {path}"


def no_pocket_residues_matched(path: str | Path) -> str:
    """Returns an error when no requested residue exists in the structure."""

    return (
        f"None of the requested pocket residues are present in {path}. "
        "Check the chain id and the residue numbering used by this file."
    )


def pocket_ligand_not_found(selector: str, path: str | Path) -> str:
    """Returns an error when no heteroatom group matches the ligand selector."""

    return (
        f"No bound ligand matching '{selector}' was found in {path}. "
        "Give the residue name as written in the file, optionally narrowed "
        "as 'NAME:chain' or 'NAME:chain:number'."
    )


def solvent_pocket_ligand(selector: str) -> str:
    """Returns an error when the ligand selector names solvent."""

    return (
        f"'{selector}' names solvent, which cannot define a binding pocket. "
        "Give the residue name of the bound ligand instead."
    )


def no_residues_near_pocket_ligand(selector: str, distance: float) -> str:
    """Returns an error when a ligand has no protein residues around it."""

    return (
        f"No protein residue lies within {distance:.1f}A of ligand '{selector}'. "
        "The ligand may be unbound in this structure."
    )


def no_residues_near_pocket_center(center: tuple[float, float, float], distance: float) -> str:
    """Returns an error when a user-given center has no residues around it."""

    return (
        f"No residue lies within {distance:.1f}A of ({center[0]:.2f}, {center[1]:.2f}, "
        f"{center[2]:.2f}). Check that the coordinates match this structure's frame."
    )


def user_pocket_unavailable(mode: str, reason: Exception) -> str:
    """Returns the error raised when a user-specified pocket cannot be used."""

    return (
        f"The pocket you specified ({mode}) could not be resolved: {reason} "
        "The run was stopped rather than continuing against a different site."
    )


def boltz_request_rejected(smiles: str, response: "httpx.Response") -> str:
    """Returns an error naming the status and reason Boltz refused a request.

    The body is included because the status alone rarely distinguishes an
    expired key from an exhausted quota or a malformed payload, and all
    three otherwise surface identically as a missing prediction.
    """

    detail = response.text.strip()
    if len(detail) > 300:
        detail = detail[:300] + "..."
    return (
        f"Boltz rejected the request for {smiles} with HTTP "
        f"{response.status_code}: {detail or '(empty body)'}"
    )
