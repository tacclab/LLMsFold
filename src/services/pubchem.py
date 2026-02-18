"""Async PubChem API utilities."""

import asyncio

import httpx

from src.clients import get_cached_http_client
from src.core.logging import get_logger
from src.schemas import PatentCheckResult

logger = get_logger(__name__)


class PubChemService:
    """Queries PubChem identity and substructure patent signals."""

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        """Initializes service with reusable async HTTP client."""

        self._client = http_client or get_cached_http_client()

    async def check_patents(self, smiles: str) -> PatentCheckResult:
        """Fetches PubChem identifiers and patent-derived novelty metrics.

        Args:
            smiles: Candidate molecule as SMILES.

        Returns:
            A validated `PatentCheckResult` structure.
        """

        try:
            cid = None
            identity_patents = 0

            cid_url = (
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
                f"{smiles}/cids/JSON"
            )
            cid_response = await self._client.get(cid_url, timeout=10)
            if cid_response.status_code == 200:
                cid_payload = cid_response.json()
                cid_list = cid_payload.get("IdentifierList", {}).get("CID", [])
                if cid_list:
                    cid = int(cid_list[0])
                    xref_url = (
                        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                        f"{cid}/xrefs/PatentID/JSON"
                    )
                    xref_response = await self._client.get(xref_url, timeout=10)
                    if xref_response.status_code == 200:
                        xref_payload = xref_response.json()
                        info = xref_payload.get("InformationList", {}).get("Information", [{}])[0]
                        identity_patents = len(info.get("PatentID", []))

            substructure_patents = 0
            sub_url = (
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/substructure/smiles/"
                f"{smiles}/JSON"
            )
            sub_response = await self._client.get(sub_url, timeout=10)
            if sub_response.status_code == 202:
                list_key = sub_response.json().get("Waiting", {}).get("ListKey")
                if list_key:
                    for _ in range(3):
                        await asyncio.sleep(2)
                        list_url = (
                            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/listkey/"
                            f"{list_key}/cids/JSON"
                        )
                        list_response = await self._client.get(list_url, timeout=10)
                        if list_response.status_code == 200:
                            sub_cids = list_response.json().get("IdentifierList", {}).get("CID", [])
                            substructure_patents = len(sub_cids[:10])
                            break

            return PatentCheckResult(
                pubchem_cid=cid,
                identity_patents=identity_patents,
                substructure_patents=substructure_patents,
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            logger.warning("PubChem query error: %s", exc)
            return PatentCheckResult()


async def check_pubchem_patents(smiles: str) -> tuple[int | None, int, int]:
    """Backward-compatible functional wrapper around `PubChemService`."""

    result = await PubChemService().check_patents(smiles)
    return result.pubchem_cid, result.identity_patents, result.substructure_patents
