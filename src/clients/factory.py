"""Cached client factories shared across modules."""

from functools import lru_cache

import httpx
from groq import Groq


@lru_cache(maxsize=1)
def get_cached_http_client() -> httpx.AsyncClient:
    """Builds and caches a reusable `httpx.AsyncClient` instance."""

    timeout = httpx.Timeout(timeout=400.0, connect=30.0)
    return httpx.AsyncClient(timeout=timeout)


@lru_cache(maxsize=2)
def get_cached_groq_client(api_key: str) -> Groq:
    """Builds and caches a Groq client keyed by API key."""

    return Groq(api_key=api_key)


async def close_cached_clients() -> None:
    """Closes any cached network clients for clean async shutdown."""

    if get_cached_http_client.cache_info().currsize > 0:
        client = get_cached_http_client()
        if not client.is_closed:
            await client.aclose()
    get_cached_http_client.cache_clear()
    get_cached_groq_client.cache_clear()
