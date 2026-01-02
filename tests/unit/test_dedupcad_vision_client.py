from __future__ import annotations

from typing import Any, List

import httpx
import pytest

from src.core.dedupcad_vision import (
    DedupCadVisionCircuitOpen,
    DedupCadVisionClient,
    DedupCadVisionResilienceConfig,
)
from src.core.vision.circuit_breaker import CircuitState


def _make_response(status_code: int, payload: Any | None = None, headers: dict[str, str] | None = None):
    request = httpx.Request("GET", "http://dedupcad-vision.local")
    return httpx.Response(status_code, json=payload, headers=headers, request=request)


async def _noop_sleep(*_args: Any, **_kwargs: Any) -> None:
    return None


class _ResponseStubClient:
    def __init__(self, responses: List[Any], calls: List[tuple[str, str]]):
        self._responses = responses
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def request(self, method: str, path: str, **_kwargs: Any):
        self._calls.append((method, path))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.asyncio
async def test_request_json_retries_on_http_500_then_succeeds(monkeypatch):
    responses = [
        _make_response(500, {"error": "boom"}),
        _make_response(200, {"ok": True}),
    ]
    calls: List[tuple[str, str]] = []

    def _client_factory(*_args: Any, **_kwargs: Any):
        return _ResponseStubClient(responses, calls)

    monkeypatch.setattr("src.core.dedupcad_vision.httpx.AsyncClient", _client_factory)
    monkeypatch.setattr("src.core.dedupcad_vision.asyncio.sleep", _noop_sleep)

    client = DedupCadVisionClient(
        resilience=DedupCadVisionResilienceConfig(retry_max_attempts=2)
    )
    payload = await client._request_json(
        "GET",
        "/health",
        endpoint="health",
        allow_retry=True,
    )

    assert payload["ok"] is True
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_request_json_retries_on_timeout_then_raises(monkeypatch):
    responses = [
        httpx.TimeoutException("timeout"),
        httpx.TimeoutException("timeout"),
    ]
    calls: List[tuple[str, str]] = []

    def _client_factory(*_args: Any, **_kwargs: Any):
        return _ResponseStubClient(responses, calls)

    monkeypatch.setattr("src.core.dedupcad_vision.httpx.AsyncClient", _client_factory)
    monkeypatch.setattr("src.core.dedupcad_vision.asyncio.sleep", _noop_sleep)

    client = DedupCadVisionClient(
        resilience=DedupCadVisionResilienceConfig(retry_max_attempts=2)
    )

    with pytest.raises(httpx.TimeoutException):
        await client._request_json(
            "GET",
            "/health",
            endpoint="health",
            allow_retry=True,
        )

    assert len(calls) == 2


@pytest.mark.asyncio
async def test_request_json_circuit_open_short_circuits(monkeypatch):
    calls: List[tuple[str, str]] = []

    def _client_factory(*_args: Any, **_kwargs: Any):
        return _ResponseStubClient([], calls)

    class _Breaker:
        state = CircuitState.OPEN

        def can_execute(self):
            return False

    client = DedupCadVisionClient()
    monkeypatch.setattr(client, "_get_circuit_breaker", lambda _endpoint: _Breaker())
    monkeypatch.setattr("src.core.dedupcad_vision.httpx.AsyncClient", _client_factory)

    with pytest.raises(DedupCadVisionCircuitOpen):
        await client._request_json(
            "GET",
            "/health",
            endpoint="health",
            allow_retry=True,
        )

    assert calls == []
