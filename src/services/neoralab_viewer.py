"""NeoraLab viewer integration helpers."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx


class NeoraLabViewerService:
    """Authenticate with NeoraLab, upload viewer data, and open deep links."""

    def __init__(
        self,
        api_base_url: str,
        *,
        viewer_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._api_base_url = api_base_url.rstrip("/")
        self._viewer_url = (viewer_url or api_base_url.rstrip("/") + "/app/viewer").rstrip("/")
        self._timeout_seconds = timeout_seconds

    def _build_api_url(self, path: str) -> str:
        return f"{self._api_base_url}{path}"

    def _build_connect_error(self, action: str, exc: httpx.ConnectError) -> RuntimeError:
        message = str(exc)
        if "WRONG_VERSION_NUMBER" in message and self._api_base_url.startswith("https://localhost"):
            return RuntimeError(
                "NeoraLab connection failed during "
                f"{action}: HTTPS was used against a local HTTP server. Set "
                "NEORALAB_API_BASE_URL=http://localhost:8000 for the backend API."
            )
        return RuntimeError(f"NeoraLab connection failed during {action}: {message}")

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            for key in ("error_description", "detail", "error", "message"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value

        return response.text.strip() or response.reason_phrase or "Request failed."

    async def authenticate(self, *, client_id: str, client_secret: str) -> str:
        """Exchange client credentials for a NeoraLab access token."""

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds, follow_redirects=True
            ) as client:
                response = await client.post(
                    self._build_api_url("/oauth/token"),
                    data={
                        "grant_type": "client_credentials",
                        "client_id": client_id,
                        "client_secret": client_secret,
                    },
                )
        except httpx.ConnectError as exc:
            raise self._build_connect_error("authentication", exc) from exc

        if response.status_code >= 400:
            raise RuntimeError(
                f"NeoraLab authentication failed: {self._extract_error_message(response)}"
            )

        raw = response.content
        if not raw or not raw.strip():
            raise RuntimeError(
                "NeoraLab authentication failed: server returned an empty response "
                f"(HTTP {response.status_code})."
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"NeoraLab authentication failed: could not parse server response as JSON "
                f"(HTTP {response.status_code}): {response.text!r}"
            ) from exc

        access_token = payload.get("access_token") if isinstance(payload, dict) else None
        if not isinstance(access_token, str) or not access_token.strip():
            raise RuntimeError("NeoraLab authentication failed: missing access token.")
        return access_token

    async def upload_viewer_payload(
        self,
        *,
        access_token: str,
        candidate_id: str,
        metadata: dict[str, Any],
        structure_content: str,
        source: str = "llmsfold",
    ) -> str:
        """Persist viewer-ready data to the NeoraLab repository."""

        payload = {
            "item_type": "viewer",
            "name": f"LLMsFold {candidate_id}",
            "summary_metadata": {
                "candidate_id": candidate_id,
                "smiles": metadata.get("smiles"),
                "source": source,
            },
            "data": {
                "candidate_id": candidate_id,
                "source": source,
                "metadata": metadata,
                "structures": [
                    {
                        "filename": "structure.cif",
                        "format": "cif",
                        "content": structure_content,
                    }
                ],
            },
        }

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds, follow_redirects=True
            ) as client:
                response = await client.post(
                    self._build_api_url("/api/v1/repository/"),
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=payload,
                )
        except httpx.ConnectError as exc:
            raise self._build_connect_error("repository upload", exc) from exc

        if response.status_code >= 400:
            raise RuntimeError(
                f"NeoraLab repository upload failed: {self._extract_error_message(response)}"
            )

        raw = response.content
        if not raw or not raw.strip():
            raise RuntimeError(
                "NeoraLab repository upload failed: server returned an empty response "
                f"(HTTP {response.status_code})."
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"NeoraLab repository upload failed: could not parse server response as JSON "
                f"(HTTP {response.status_code}): {response.text!r}"
            ) from exc

        item_id = body.get("id") if isinstance(body, dict) else None
        if not isinstance(item_id, str) or not item_id.strip():
            raise RuntimeError("NeoraLab repository upload failed: missing item id.")
        return item_id

    def build_viewer_url(self, item_id: str) -> str:
        """Build the authenticated viewer deep link for a repository item."""

        parsed = urlparse(self._viewer_url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query["item"] = item_id
        return urlunparse(parsed._replace(query=urlencode(query)))