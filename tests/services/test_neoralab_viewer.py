"""Tests for NeoraLab viewer integration helpers."""

from __future__ import annotations

import asyncio

import httpx

from src.services.neoralab_viewer import NeoraLabViewerService


def test_authenticate_uses_client_credentials(monkeypatch) -> None:
    """Client credentials authentication should return the API access token."""

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/oauth/token"
        return httpx.Response(200, json={"access_token": "token-123", "expires_in": 3600})

    transport = httpx.MockTransport(handler)

    original_client = httpx.AsyncClient

    class MockAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)
    service = NeoraLabViewerService("https://neoralab.app", viewer_url="https://neoralab.app/app/viewer")

    try:
        token = asyncio.run(
            service.authenticate(client_id="client-id", client_secret="client-secret")
        )
    finally:
        monkeypatch.setattr(httpx, "AsyncClient", original_client)

    assert token == "token-123"


def test_upload_viewer_payload_returns_repository_item_id(monkeypatch) -> None:
    """Viewer uploads should create a repository item and return its id."""

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/repository/"
        assert request.headers["Authorization"] == "Bearer token-123"
        return httpx.Response(201, json={"id": "item-456"})

    transport = httpx.MockTransport(handler)

    original_client = httpx.AsyncClient

    class MockAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)
    service = NeoraLabViewerService("https://neoralab.app", viewer_url="https://neoralab.app/app/viewer")

    try:
        item_id = asyncio.run(
            service.upload_viewer_payload(
                access_token="token-123",
                candidate_id="candidate-0001",
                metadata={"smiles": "CCO"},
                structure_content="data_candidate\n_atom_site",
            )
        )
    finally:
        monkeypatch.setattr(httpx, "AsyncClient", original_client)

    assert item_id == "item-456"