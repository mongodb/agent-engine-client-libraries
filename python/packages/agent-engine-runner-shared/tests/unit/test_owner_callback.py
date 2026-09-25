"""Tests for the shared owner-callback pre-attempt policy (``owner_delivered``).

This is the single place the fallback status matrix lives; call-site tests
(emit, custom events, secure wrapper, AER) verify wiring, not policy.
Mirrors TypeScript's tests/unit/owner_callback.test.ts.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from agent_engine_runner_shared.owner_callback import owner_delivered

_OWNER = "https://10-1-2-3.oe-headless.ns.svc.cluster.local:8443/stream/chunk"
_PAYLOAD = {"k": "v"}


class _TrackingBody(httpx.SyncByteStream):
    def __init__(self, chunk: bytes) -> None:
        self.chunk = chunk
        self.read = False

    def __iter__(self):
        self.read = True
        yield self.chunk


def _client_returning(status: int, body: str = "") -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(status, text=body))
    )


def test_returns_true_on_2xx() -> None:
    logs: list[str] = []
    with _client_returning(200) as client:
        assert owner_delivered(client, _OWNER, _PAYLOAD, logs.append) is True
    assert logs == []


def test_close_failure_does_not_duplicate_successful_owner_delivery() -> None:
    response = httpx.Response(200, request=httpx.Request("POST", _OWNER))
    client = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = response
    context.__exit__.side_effect = RuntimeError("close failed")
    client.stream.return_value = context

    assert owner_delivered(client, _OWNER, _PAYLOAD, lambda _msg: None) is True
    client.stream.assert_called_once()


@pytest.mark.parametrize("status", [307, 404, 500, 503])
def test_returns_false_on_owner_http_error_without_retrying(status: int) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(status, text="internal diagnostics")

    logs: list[str] = []
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert owner_delivered(client, _OWNER, _PAYLOAD, logs.append) is False
    assert calls["n"] == 1
    assert logs == [f"owner URL {_OWNER} unusable (HTTP {status}); falling back to service"]


def test_redirects_are_not_followed() -> None:
    # A 3xx counts as an owner failure; the redirect target must never be hit.
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(307, headers={"Location": "https://elsewhere.example/steal"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert owner_delivered(client, _OWNER, _PAYLOAD, lambda _msg: None) is False
    assert calls == [_OWNER]


def test_returns_false_on_transport_error_without_raising() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    logs: list[str] = []
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert owner_delivered(client, _OWNER, _PAYLOAD, logs.append) is False
    assert logs == [f"owner URL {_OWNER} unusable (connection refused); falling back to service"]


def test_never_logs_response_bodies() -> None:
    logs: list[str] = []
    with _client_returning(500, body="internal diagnostics") as client:
        owner_delivered(client, _OWNER, _PAYLOAD, logs.append)
    assert "internal diagnostics" not in logs[0]


def test_never_reads_response_bodies() -> None:
    body = _TrackingBody(b"internal diagnostics")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, stream=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert owner_delivered(client, _OWNER, _PAYLOAD, lambda _msg: None) is False
    assert body.read is False
