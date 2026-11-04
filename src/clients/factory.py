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
def get_cached_groq_client(api_key: str, max_retries: int = 6) -> Groq:
    """Builds and caches a Groq client keyed by API key.

    `max_retries` is raised above the SDK default because rate-limit
    windows are measured in minutes: a long run meters output tokens per
    minute, so a 429 has to be waited out rather than abandoned. The SDK
    honours the provider's `retry-after` header on each attempt.
    """

    return Groq(api_key=api_key, max_retries=max_retries)


async def close_cached_clients() -> None:
    """Closes any cached network clients for clean async shutdown."""

    if get_cached_http_client.cache_info().currsize > 0:
        client = get_cached_http_client()
        if not client.is_closed:
            await client.aclose()
    get_cached_http_client.cache_clear()
    get_cached_groq_client.cache_clear()
