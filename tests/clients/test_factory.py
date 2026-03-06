"""Tests for cached client factories."""

import asyncio
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.clients import factory


@pytest.fixture(autouse=True)
def clear_factory_caches() -> Iterator[None]:
    """Ensures each test starts with clean lru caches."""

    factory.get_cached_http_client.cache_clear()
    factory.get_cached_groq_client.cache_clear()
    yield
    factory.get_cached_http_client.cache_clear()
    factory.get_cached_groq_client.cache_clear()


def test_get_cached_http_client_reuses_single_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP client factory uses one cached client instance."""

    fake_client = MagicMock(name="async_client")
    async_client_cls = MagicMock(return_value=fake_client)
    monkeypatch.setattr(factory.httpx, "AsyncClient", async_client_cls)

    first = factory.get_cached_http_client()
    second = factory.get_cached_http_client()

    assert first is second
    assert async_client_cls.call_count == 1


@pytest.mark.parametrize("api_key", ["key-a", "key-b"])
def test_get_cached_groq_client_is_keyed(monkeypatch: pytest.MonkeyPatch, api_key: str) -> None:
    """Groq clients are cached by API key."""

    groq_ctor = MagicMock(side_effect=lambda api_key: {"client_for": api_key})
    monkeypatch.setattr(factory, "Groq", groq_ctor)

    first = factory.get_cached_groq_client(api_key)
    second = factory.get_cached_groq_client(api_key)

    assert first == {"client_for": api_key}
    assert second is first


def test_get_cached_groq_client_creates_distinct_entries_for_distinct_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different API keys should produce different cached objects."""

    groq_ctor = MagicMock(side_effect=lambda api_key: {"client_for": api_key})
    monkeypatch.setattr(factory, "Groq", groq_ctor)

    key_a_client = factory.get_cached_groq_client("k1")
    key_b_client = factory.get_cached_groq_client("k2")

    assert key_a_client != key_b_client
    assert groq_ctor.call_count == 2


def test_close_cached_clients_closes_open_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown routine closes open client and clears caches."""

    fake_client = MagicMock(name="async_client")
    fake_client.is_closed = False
    fake_client.aclose = AsyncMock()

    async_client_cls = MagicMock(return_value=fake_client)
    monkeypatch.setattr(factory.httpx, "AsyncClient", async_client_cls)

    factory.get_cached_http_client()
    asyncio.run(factory.close_cached_clients())

    fake_client.aclose.assert_awaited_once()
    assert factory.get_cached_http_client.cache_info().currsize == 0
    assert factory.get_cached_groq_client.cache_info().currsize == 0


def test_close_cached_clients_skips_already_closed_http_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Already closed clients should not be closed again."""

    fake_client = MagicMock(name="async_client")
    fake_client.is_closed = True
    fake_client.aclose = AsyncMock()

    async_client_cls = MagicMock(return_value=fake_client)
    monkeypatch.setattr(factory.httpx, "AsyncClient", async_client_cls)

    factory.get_cached_http_client()
    asyncio.run(factory.close_cached_clients())

    fake_client.aclose.assert_not_awaited()
