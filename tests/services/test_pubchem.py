"""Tests for PubChem patent lookup service."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import httpx

from src.schemas import PatentCheckResult
from src.services.pubchem import PubChemService


class FakeResponse:
    """Small helper to emulate httpx JSON responses."""

    def __init__(self, status_code: int, payload: dict, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        return self._payload


def test_check_patents_full_success_flow(monkeypatch) -> None:
    """Service resolves CID, identity patents, and substructure patents."""

    client = MagicMock()
    client.get = AsyncMock(
        side_effect=[
            FakeResponse(200, {"IdentifierList": {"CID": [123]}}),
            FakeResponse(200, {"InformationList": {"Information": [{"PatentID": ["P1", "P2"]}]}}),
            FakeResponse(404, {}),  # no drug/medication info
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
    assert result.has_drug_info is False


def test_check_patents_handles_http_error() -> None:
    """HTTP errors are swallowed and mapped to default zeroed result."""

    client = MagicMock()
    client.get = AsyncMock(side_effect=httpx.HTTPError("network down"))

    service = PubChemService(http_client=client)
    result = asyncio.run(service.check_patents("CCO"))

    assert result == PatentCheckResult()


def test_check_patents_treats_cid_zero_as_unknown_and_skips_xrefs() -> None:
    """CID 0 should not be interpreted as a known PubChem compound."""

    client = MagicMock()
    client.get = AsyncMock(
        side_effect=[
            FakeResponse(200, {"IdentifierList": {"CID": [0]}}),
            FakeResponse(200, {"IdentifierList": {"CID": [11, 12, 13]}}),
        ]
    )

    service = PubChemService(http_client=client)
    result = asyncio.run(service.check_patents("CCO"))

    assert result.pubchem_cid is None
    assert result.identity_patents == 0
    assert result.substructure_patents == 3
    assert client.get.await_count == 2


def test_check_patents_handles_immediate_substructure_results() -> None:
    """A completed substructure search can return 200 without listkey polling."""

    client = MagicMock()
    client.get = AsyncMock(
        side_effect=[
            FakeResponse(404, {}),
            FakeResponse(200, {"IdentifierList": {"CID": list(range(1, 16))}}),
        ]
    )

    service = PubChemService(http_client=client)
    result = asyncio.run(service.check_patents("CCO"))

    assert result.pubchem_cid is None
    assert result.identity_patents == 0
    assert result.substructure_patents == 10


def test_check_patents_url_encodes_smiles_special_characters() -> None:
    """SMILES containing '#' (triple bonds) must not leak into the URL fragment."""

    client = MagicMock()
    client.get = AsyncMock(
        side_effect=[
            FakeResponse(200, {"IdentifierList": {"CID": [123]}}),
            FakeResponse(200, {"InformationList": {"Information": []}}),
            FakeResponse(404, {}),  # no drug/medication info
            FakeResponse(200, {"IdentifierList": {"CID": []}}),
        ]
    )

    service = PubChemService(http_client=client)
    smiles = "C[C@H]1CN(C[C@H](C1)c2ccccc2)c3ccc(cc3)C#N"
    asyncio.run(service.check_patents(smiles))

    cid_lookup_url = client.get.await_args_list[0].args[0]
    substructure_url = client.get.await_args_list[3].args[0]

    assert "#" not in cid_lookup_url
    assert cid_lookup_url.endswith("/cids/JSON")
    assert "#" not in substructure_url
    assert substructure_url.endswith("/JSON")


def test_check_patents_detects_drug_medication_info() -> None:
    """A PUG View 200 for the drug-info heading sets `has_drug_info`."""

    client = MagicMock()
    client.get = AsyncMock(
        side_effect=[
            FakeResponse(200, {"IdentifierList": {"CID": [2244]}}),
            FakeResponse(200, {"InformationList": {"Information": []}}),
            FakeResponse(200, {"Record": {"RecordTitle": "Aspirin"}}),  # has drug info
            FakeResponse(200, {"IdentifierList": {"CID": []}}),
        ]
    )

    service = PubChemService(http_client=client)
    result = asyncio.run(service.check_patents("CC(=O)OC1=CC=CC=C1C(=O)O"))

    assert result.has_drug_info is True


def test_check_patents_preserves_identity_when_substructure_polling_times_out(
    monkeypatch,
    caplog,
) -> None:
    """Substructure polling failures should not erase an already resolved identity result."""

    client = MagicMock()
    client.get = AsyncMock(
        side_effect=[
            FakeResponse(200, {"IdentifierList": {"CID": [123]}}),
            FakeResponse(200, {"InformationList": {"Information": [{"PatentID": ["P1", "P2"]}]}}),
            FakeResponse(404, {}),  # no drug/medication info
            FakeResponse(202, {"Waiting": {"ListKey": "abc"}}),
            FakeResponse(400, {}, text="still processing"),
            FakeResponse(400, {}, text="still processing"),
            FakeResponse(400, {}, text="still processing"),
        ]
    )

    monkeypatch.setattr("src.services.pubchem.asyncio.sleep", AsyncMock())

    service = PubChemService(http_client=client)
    with caplog.at_level(logging.WARNING):
        result = asyncio.run(service.check_patents("CCO"))

    assert result.pubchem_cid == 123
    assert result.identity_patents == 2
    assert result.substructure_patents == 0
    assert "did not complete after 3 polling attempts" in caplog.text
