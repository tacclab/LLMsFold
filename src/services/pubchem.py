"""Async PubChem API utilities."""

import asyncio
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

import httpx

from src.clients import get_cached_http_client
from src.core.config import GeneratorSettings, get_generator_settings
from src.core.exceptions import (
    PubChemError,
    PubChemHTTPStatusError,
    PubChemPayloadError,
    PubChemPollingTimeoutError,
    PubChemTransportError,
)
from src.core.logging import get_logger
from src.core.messages import pubchem_query_error
from src.schemas import PatentCheckResult

logger = get_logger(__name__)

_PUG_REST_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
_MAX_SUBSTRUCTURE_CIDS = 10
_PENDING_LISTKEY_STATUS_CODES = {202, 400}


class PubChemService:
    """Queries PubChem identity and substructure patent signals."""

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        settings: GeneratorSettings | None = None,
    ) -> None:
        """Initializes service with reusable async HTTP client."""

        self._settings = settings or get_generator_settings()
        self._client = http_client or get_cached_http_client()

    async def check_patents(self, smiles: str) -> PatentCheckResult:
        """Fetches PubChem identifiers and patent-derived novelty metrics.

        Args:
            smiles: Candidate molecule as SMILES.

        Returns:
            A validated `PatentCheckResult` structure.
        """

        result = PatentCheckResult()

        try:
            result.pubchem_cid = await self._resolve_cid(smiles)
        except PubChemError as exc:
            logger.warning(pubchem_query_error(smiles, exc))

        if result.pubchem_cid:
            try:
                result.identity_patents = await self._fetch_identity_patents(result.pubchem_cid)
            except PubChemError as exc:
                logger.warning(pubchem_query_error(smiles, exc))

        try:
            result.substructure_patents = await self._fetch_substructure_patents(smiles)
        except PubChemError as exc:
            logger.warning(pubchem_query_error(smiles, exc))

        return result

    async def _resolve_cid(self, smiles: str) -> int | None:
        """Returns the first positive PubChem CID for the input SMILES."""

        response = await self._get(
            f"{_PUG_REST_BASE_URL}/smiles/{quote(smiles, safe='')}/cids/JSON",
            stage="PubChem CID lookup",
        )
        if response.status_code == 200:
            payload = self._parse_json(response, stage="PubChem CID lookup")
            cids = self._extract_cids(payload, stage="PubChem CID lookup")
            return cids[0] if cids else None
        if response.status_code == 404:
            return None
        raise PubChemHTTPStatusError(
            "PubChem CID lookup",
            response.status_code,
            self._response_detail(response),
        )

    async def _fetch_identity_patents(self, cid: int) -> int:
        """Returns the number of identity-linked patent ids for a CID."""

        response = await self._get(
            f"{_PUG_REST_BASE_URL}/cid/{cid}/xrefs/PatentID/JSON",
            stage="PubChem identity patent lookup",
        )
        if response.status_code == 200:
            payload = self._parse_json(response, stage="PubChem identity patent lookup")
            information_list = payload.get("InformationList", {})
            if not isinstance(information_list, Mapping):
                raise PubChemPayloadError(
                    "PubChem identity patent lookup",
                    "missing InformationList object",
                )
            info_list = information_list.get("Information", [])
            if not isinstance(info_list, list):
                raise PubChemPayloadError(
                    "PubChem identity patent lookup",
                    "missing InformationList.Information list",
                )
            if not info_list:
                return 0
            first_record = info_list[0]
            if not isinstance(first_record, Mapping):
                raise PubChemPayloadError(
                    "PubChem identity patent lookup",
                    "InformationList.Information[0] is not an object",
                )
            patent_ids = first_record.get("PatentID", [])
            if patent_ids is None:
                return 0
            if not isinstance(patent_ids, list):
                raise PubChemPayloadError(
                    "PubChem identity patent lookup",
                    "PatentID is not a list",
                )
            return len(patent_ids)
        if response.status_code == 404:
            return 0
        raise PubChemHTTPStatusError(
            "PubChem identity patent lookup",
            response.status_code,
            self._response_detail(response),
        )

    async def _fetch_substructure_patents(self, smiles: str) -> int:
        """Returns a capped count of PubChem CIDs from substructure search."""

        response = await self._get(
            f"{_PUG_REST_BASE_URL}/substructure/smiles/{quote(smiles, safe='')}/JSON",
            stage="PubChem substructure search",
        )
        if response.status_code == 200:
            payload = self._parse_json(response, stage="PubChem substructure search")
            return self._count_cids(payload, stage="PubChem substructure search")
        if response.status_code == 202:
            payload = self._parse_json(response, stage="PubChem substructure search")
            waiting = payload.get("Waiting", {})
            if not isinstance(waiting, Mapping):
                raise PubChemPayloadError(
                    "PubChem substructure search",
                    "missing Waiting object",
                )
            list_key = waiting.get("ListKey")
            if not isinstance(list_key, str) or not list_key.strip():
                raise PubChemPayloadError(
                    "PubChem substructure search",
                    "202 response missing Waiting.ListKey",
                )
            return await self._poll_substructure_results(list_key)
        if response.status_code == 404:
            return 0
        raise PubChemHTTPStatusError(
            "PubChem substructure search",
            response.status_code,
            self._response_detail(response),
        )

    async def _poll_substructure_results(self, list_key: str) -> int:
        """Polls the PubChem listkey endpoint until results are ready or exhausted."""

        last_status_code: int | None = None
        for _ in range(self._settings.pubchem_listkey_attempts):
            await asyncio.sleep(self._settings.pubchem_poll_interval_seconds)
            response = await self._get(
                f"{_PUG_REST_BASE_URL}/listkey/{list_key}/cids/JSON",
                stage="PubChem substructure polling",
            )
            last_status_code = response.status_code
            if response.status_code == 200:
                payload = self._parse_json(response, stage="PubChem substructure polling")
                return self._count_cids(payload, stage="PubChem substructure polling")
            if response.status_code in _PENDING_LISTKEY_STATUS_CODES:
                continue
            raise PubChemHTTPStatusError(
                "PubChem substructure polling",
                response.status_code,
                self._response_detail(response),
            )

        raise PubChemPollingTimeoutError(
            "PubChem substructure polling",
            attempts=self._settings.pubchem_listkey_attempts,
            last_status_code=last_status_code,
        )

    async def _get(self, url: str, stage: str) -> httpx.Response:
        """Executes an HTTP request and normalizes transport failures."""

        try:
            return await self._client.get(
                url,
                timeout=self._settings.pubchem_timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise PubChemTransportError(stage, exc) from exc

    def _parse_json(self, response: httpx.Response, stage: str) -> dict[str, Any]:
        """Parses a JSON payload or raises a structured PubChem payload error."""

        try:
            payload = response.json()
        except ValueError as exc:
            raise PubChemPayloadError(stage, "response body is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise PubChemPayloadError(stage, "response body is not a JSON object")
        return payload

    def _extract_cids(self, payload: dict[str, Any], stage: str) -> list[int]:
        """Extracts positive integer CIDs from a PubChem payload."""

        identifier_list = payload.get("IdentifierList", {})
        if not isinstance(identifier_list, Mapping):
            raise PubChemPayloadError(stage, "missing IdentifierList object")
        raw_cids = identifier_list.get("CID", [])
        if raw_cids is None:
            return []
        if not isinstance(raw_cids, list):
            raise PubChemPayloadError(stage, "IdentifierList.CID is not a list")

        cids: list[int] = []
        for raw_cid in raw_cids:
            try:
                cid = int(raw_cid)
            except (TypeError, ValueError) as exc:
                raise PubChemPayloadError(stage, f"invalid CID value: {raw_cid!r}") from exc
            if cid > 0:
                cids.append(cid)
        return cids

    def _count_cids(self, payload: dict[str, Any], stage: str) -> int:
        """Returns the capped CID count used by the report."""

        return len(self._extract_cids(payload, stage=stage)[:_MAX_SUBSTRUCTURE_CIDS])

    def _response_detail(self, response: httpx.Response) -> str | None:
        """Extracts a compact detail string from a PubChem error response."""

        try:
            payload = response.json()
        except ValueError:
            return self._clean_detail(getattr(response, "text", None))

        if isinstance(payload, Mapping):
            fault = payload.get("Fault")
            if isinstance(fault, Mapping):
                details = fault.get("Details")
                if isinstance(details, list) and details:
                    return self._clean_detail(str(details[0]))
                message = fault.get("Message")
                if isinstance(message, str):
                    return self._clean_detail(message)

        return self._clean_detail(getattr(response, "text", None))

    def _clean_detail(self, detail: str | None) -> str | None:
        """Normalizes response details for concise warning messages."""

        if not detail:
            return None
        cleaned = " ".join(detail.strip().split())
        return cleaned[:160] if cleaned else None
