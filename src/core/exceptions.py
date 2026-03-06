"""Custom exception hierarchy for user-facing runtime failures."""


class LLMsFoldError(Exception):
    """Base exception for domain-specific pipeline failures."""


class ConfigurationError(LLMsFoldError):
    """Raised when required runtime configuration is missing or invalid."""


class SequenceExtractionError(LLMsFoldError):
    """Raised when a protein sequence cannot be derived from a PDB file."""


class PocketDetectionError(LLMsFoldError):
    """Raised when binding pocket discovery cannot be initialized or completed."""


class PubChemError(LLMsFoldError):
    """Base exception for PubChem verification failures."""


class PubChemTransportError(PubChemError):
    """Raised when a PubChem request fails before a response is received."""

    def __init__(self, stage: str, exc: Exception) -> None:
        super().__init__(f"{stage} failed: {exc}")


class PubChemHTTPStatusError(PubChemError):
    """Raised when PubChem returns an unexpected HTTP status."""

    def __init__(self, stage: str, status_code: int, detail: str | None = None) -> None:
        message = f"{stage} returned HTTP {status_code}"
        if detail:
            message = f"{message} ({detail})"
        super().__init__(message)


class PubChemPayloadError(PubChemError):
    """Raised when a PubChem response payload is missing expected fields."""

    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(f"{stage} returned an invalid payload: {detail}")


class PubChemPollingTimeoutError(PubChemError):
    """Raised when PubChem substructure polling does not finish in time."""

    def __init__(self, stage: str, attempts: int, last_status_code: int | None = None) -> None:
        message = f"{stage} did not complete after {attempts} polling attempts"
        if last_status_code is not None:
            message = f"{message} (last HTTP {last_status_code})"
        super().__init__(message)
