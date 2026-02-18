"""Tests for PubChem patent lookup service."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx

from src.schemas import PatentCheckResult
from src.services.pubchem import PubChemService, check_pubchem_patents


class FakeResponse:
    """Small helper to emulate httpx JSON responses."""

    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def test_check_patents_full_success_flow(monkeypatch) -> None:
    """Service resolves CID, identity patents, and substructure patents."""

    client = MagicMock()
    client.get = AsyncMock(
        side_effect=[
            FakeResponse(200, {"IdentifierList": {"CID": [123]}}),
            FakeResponse(200, {"InformationList": {"Information": [{"PatentID": ["P1", "P2"]}]}}),
            FakeResponse(202, {"Waiting": {"ListKey": "abc"}}),
            FakeResponse(200, {"IdentifierList": {"CID": list(range(20))}}),
        ]
    )

    monkeypatch.setattr("src.services.pubchem.asyncio.sleep", AsyncMock())

    service = PubChemService(http_client=client)
    result = asyncio.run(service.check_patents("CCO"))

    assert result.pubchem_cid == 123
    assert result.identity_patents == 2
    assert result.substructure_patents == 10


def test_check_patents_handles_http_error() -> None:
    """HTTP errors are swallowed and mapped to default zeroed result."""

    client = MagicMock()
    client.get = AsyncMock(side_effect=httpx.HTTPError("network down"))

    service = PubChemService(http_client=client)
    result = asyncio.run(service.check_patents("CCO"))

    assert result == PatentCheckResult()


def test_check_pubchem_patents_wrapper(monkeypatch) -> None:
    """Legacy wrapper returns tuple shape expected by older call sites."""

    async def _fake_check(_self, smiles: str) -> PatentCheckResult:
        assert smiles == "CCO"
        return PatentCheckResult(pubchem_cid=77, identity_patents=1, substructure_patents=3)

    monkeypatch.setattr(PubChemService, "check_patents", _fake_check)

    cid, identity_count, sub_count = asyncio.run(check_pubchem_patents("CCO"))

    assert cid == 77
    assert identity_count == 1
    assert sub_count == 3
