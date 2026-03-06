"""Tests for the NVIDIA Boltz async client."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.nvidia_client import BoltzClient
from src.schemas import BoltzPrediction, PocketContact


class FakeResponse:
    """Simple stand-in for httpx.Response used in async client tests."""

    def __init__(self, status_code: int, payload: dict, headers: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self) -> dict:
        return self._payload


@pytest.mark.parametrize(
    ("status_code", "expected_is_none"),
    [
        (200, False),
        (500, True),
    ],
)
def test_make_nvcf_call_direct_status_paths(status_code: int, expected_is_none: bool) -> None:
    """200 responses are parsed; non-200/202 failures return `None`."""

    payload = {
        "ptm_scores": [0.9],
        "iptm_scores": [0.8],
        "confidence_scores": [0.7],
        "complex_plddt_scores": [0.6],
        "affinities": {"L1": {"affinity_probability_binary": [0.95], "affinity_pic50": [7.0]}},
    }

    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=FakeResponse(status_code, payload))

    client = BoltzClient(api_key="token", http_client=http_client)
    result = asyncio.run(client.make_nvcf_call("CCO", "MKT"))

    assert (result is None) is expected_is_none


def test_make_nvcf_call_polls_for_202_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accepted jobs poll by task id and parse eventual payload."""

    accepted = FakeResponse(202, {}, headers={"nvcf-reqid": "task-123"})
    payload = {
        "ptm_scores": [1.0],
        "iptm_scores": [0.5],
        "confidence_scores": [0.4],
        "complex_plddt_scores": [0.3],
        "affinities": {"L1": {"affinity_probability_binary": [0.9], "affinity_pic50": [6.0]}},
    }

    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=accepted)

    client = BoltzClient(api_key="token", http_client=http_client)
    poll_mock = AsyncMock(return_value=payload)
    monkeypatch.setattr(client, "_poll_task", poll_mock)

    result = asyncio.run(
        client.make_nvcf_call(
            "CCO",
            "MKT",
            pocket_residues=[
                PocketContact(chain_id="A", residue_index=10),
                PocketContact(chain_id="A", residue_index=20),
            ],
        )
    )

    assert isinstance(result, BoltzPrediction)
    poll_mock.assert_awaited_once()

    call_kwargs = http_client.post.await_args.kwargs
    constraints = call_kwargs["json"]["constraints"]
    assert constraints[0]["contacts"] == [
        {"id": "A", "residue_index": 10},
        {"id": "A", "residue_index": 20},
    ]


def test_make_nvcf_call_returns_none_without_task_id() -> None:
    """202 without `nvcf-reqid` cannot be polled and returns `None`."""

    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=FakeResponse(202, {}, headers={}))

    client = BoltzClient(api_key="token", http_client=http_client)
    assert asyncio.run(client.make_nvcf_call("CCO", "MKT")) is None


def test_make_nvcf_call_retries_on_429_then_succeeds() -> None:
    """HTTP 429 responses are retried before failing the Boltz request."""

    payload = {
        "ptm_scores": [0.9],
        "iptm_scores": [0.8],
        "confidence_scores": [0.7],
        "complex_plddt_scores": [0.6],
        "affinities": {"L1": {"affinity_probability_binary": [0.95], "affinity_pic50": [7.0]}},
    }

    http_client = MagicMock()
    http_client.post = AsyncMock(
        side_effect=[
            FakeResponse(429, {}, headers={"Retry-After": "0"}),
            FakeResponse(200, payload),
        ]
    )

    client = BoltzClient(api_key="token", http_client=http_client)
    client._settings = client._settings.model_copy(
        update={
            "boltz_retry_attempts": 2,
            "boltz_retry_min_wait_seconds": 0.0,
            "boltz_retry_max_wait_seconds": 0.0,
        }
    )

    result = asyncio.run(client.make_nvcf_call("CCO", "MKT"))

    assert isinstance(result, BoltzPrediction)
    assert http_client.post.await_count == 2


def test_make_nvcf_call_returns_none_after_429_retry_exhaustion() -> None:
    """Repeated HTTP 429 responses return `None` after the retry budget is exhausted."""

    http_client = MagicMock()
    http_client.post = AsyncMock(
        side_effect=[
            FakeResponse(429, {}, headers={"Retry-After": "0"}),
            FakeResponse(429, {}, headers={"Retry-After": "0"}),
        ]
    )

    client = BoltzClient(api_key="token", http_client=http_client)
    client._settings = client._settings.model_copy(
        update={
            "boltz_retry_attempts": 2,
            "boltz_retry_min_wait_seconds": 0.0,
            "boltz_retry_max_wait_seconds": 0.0,
        }
    )

    assert asyncio.run(client.make_nvcf_call("CCO", "MKT")) is None
    assert http_client.post.await_count == 2


def test_poll_task_returns_json_when_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Polling loop returns payload on first 200 response."""

    http_client = MagicMock()
    http_client.get = AsyncMock(return_value=FakeResponse(200, {"ok": True}))

    client = BoltzClient(api_key="token", http_client=http_client)
    monkeypatch.setattr("src.nvidia_client.asyncio.sleep", AsyncMock())
    result = asyncio.run(client._poll_task("task", {"Authorization": "Bearer token"}))

    assert result == {"ok": True}


def test_poll_task_returns_none_on_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Polling exits with `None` on HTTP error response."""

    http_client = MagicMock()
    http_client.get = AsyncMock(return_value=FakeResponse(404, {}))

    client = BoltzClient(api_key="token", http_client=http_client)
    monkeypatch.setattr("src.nvidia_client.asyncio.sleep", AsyncMock())
    result = asyncio.run(client._poll_task("task", {"Authorization": "Bearer token"}))

    assert result is None


def test_compute_properties_builds_dataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Property computation collects one row per valid molecule with successful prediction."""

    prediction = BoltzPrediction.model_validate(
        {
            "ptm_scores": [0.9],
            "iptm_scores": [0.8],
            "confidence_scores": [0.7],
            "complex_plddt_scores": [0.6],
            "affinities": {"L1": {"affinity_probability_binary": [0.5], "affinity_pic50": [6.0]}},
        }
    )

    http_client = MagicMock()
    client = BoltzClient(api_key="token", http_client=http_client)
    monkeypatch.setattr(client, "make_nvcf_call", AsyncMock(return_value=prediction))

    fake_molecule = object()
    monkeypatch.setattr(
        "src.nvidia_client.Chem.MolFromSmiles",
        lambda smiles: fake_molecule if smiles == "CCO" else None,
    )
    monkeypatch.setattr("src.nvidia_client.Descriptors.MolWt", lambda _mol: 46.07)
    monkeypatch.setattr("src.nvidia_client.Descriptors.MolLogP", lambda _mol: 0.2)
    monkeypatch.setattr("src.nvidia_client.Descriptors.TPSA", lambda _mol: 20.23)
    monkeypatch.setattr("src.nvidia_client.Descriptors.NumHAcceptors", lambda _mol: 1)
    monkeypatch.setattr("src.nvidia_client.Descriptors.NumHDonors", lambda _mol: 1)
    monkeypatch.setattr("src.nvidia_client.Descriptors.NumRotatableBonds", lambda _mol: 0)
    monkeypatch.setattr("src.nvidia_client.QED.qed", lambda _mol: 0.55)
    monkeypatch.setattr("src.nvidia_client.sa_score_mol", lambda _mol: 2.34)

    dataframe = asyncio.run(
        client.compute_properties(
            ["CCO", "invalid"],
            "MKT",
            pocket_residues=[PocketContact(chain_id="A", residue_index=1)],
        )
    )

    assert list(dataframe["SMILES"]) == ["CCO"]
    assert dataframe.iloc[0]["Affinity_Prob"] == pytest.approx(0.5)
    assert dataframe.iloc[0]["pIC50"] == pytest.approx(6.0)
    assert dataframe.iloc[0]["IC50_uM"] == pytest.approx(1.0)


def test_compute_properties_updates_progress_for_each_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boltz scoring progress should advance once per input candidate."""

    prediction = BoltzPrediction.model_validate(
        {
            "ptm_scores": [0.9],
            "iptm_scores": [0.8],
            "confidence_scores": [0.7],
            "complex_plddt_scores": [0.6],
            "affinities": {"L1": {"affinity_probability_binary": [0.5], "affinity_pic50": [6.0]}},
        }
    )

    class DummyProgressBar:
        def __init__(self) -> None:
            self.updates = 0
            self.postfix_values: list[str] = []

        def __enter__(self) -> "DummyProgressBar":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def set_postfix_str(self, value: str) -> None:
            self.postfix_values.append(value)

        def update(self, amount: int = 1) -> None:
            self.updates += amount

    progress_bar = DummyProgressBar()
    monkeypatch.setattr("src.nvidia_client.make_progress_bar", lambda **_kwargs: progress_bar)

    http_client = MagicMock()
    client = BoltzClient(api_key="token", http_client=http_client)
    monkeypatch.setattr(client, "make_nvcf_call", AsyncMock(return_value=prediction))

    fake_molecule = object()
    monkeypatch.setattr(
        "src.nvidia_client.Chem.MolFromSmiles",
        lambda smiles: fake_molecule if smiles == "CCO" else None,
    )
    monkeypatch.setattr("src.nvidia_client.Descriptors.MolWt", lambda _mol: 46.07)
    monkeypatch.setattr("src.nvidia_client.Descriptors.MolLogP", lambda _mol: 0.2)
    monkeypatch.setattr("src.nvidia_client.Descriptors.TPSA", lambda _mol: 20.23)
    monkeypatch.setattr("src.nvidia_client.Descriptors.NumHAcceptors", lambda _mol: 1)
    monkeypatch.setattr("src.nvidia_client.Descriptors.NumHDonors", lambda _mol: 1)
    monkeypatch.setattr("src.nvidia_client.Descriptors.NumRotatableBonds", lambda _mol: 0)
    monkeypatch.setattr("src.nvidia_client.QED.qed", lambda _mol: 0.55)
    monkeypatch.setattr("src.nvidia_client.sa_score_mol", lambda _mol: 2.34)

    asyncio.run(client.compute_properties(["CCO", "invalid"], "MKT"))

    assert progress_bar.updates == 2
    assert progress_bar.postfix_values == ["CCO", "invalid"]
