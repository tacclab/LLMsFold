"""Shared warning and error message builders."""

from pathlib import Path


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


def pubchem_query_error(smiles: str, exc: Exception) -> str:
    """Returns a warning for degraded PubChem verification."""

    return f"PubChem verification degraded for {smiles}: {exc}"
