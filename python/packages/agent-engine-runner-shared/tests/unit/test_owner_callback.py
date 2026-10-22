"""Tests for the shared owner-callback pre-attempt policy (``owner_delivered``).

This is the single place the fallback status matrix lives; call-site tests
(emit, custom events, secure wrapper, AER) verify wiring, not policy.
Mirrors TypeScript's tests/unit/owner_callback.test.ts.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from agent_engine_runner_shared import owner_callback
from agent_engine_runner_shared.owner_callback import (
    MAX_DISCARD_BYTES,
    MAX_DISCARD_SECONDS,
    discard_budget_seconds,
    owner_delivered,
    post_without_reading_response_body,
)

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


def test_never_exposes_response_bodies(caplog: pytest.LogCaptureFixture) -> None:
    # Drained so the keep-alive connection can be pooled, but never surfaced.
    body = _TrackingBody(b"internal diagnostics")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, stream=body)

    logs: list[str] = []
    caplog.set_level("DEBUG")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert owner_delivered(client, _OWNER, _PAYLOAD, logs.append) is False
    assert body.read is True
    assert all("internal diagnostics" not in msg for msg in logs)
    assert "internal diagnostics" not in caplog.text


class _InfiniteBody(httpx.SyncByteStream):
    def __init__(self) -> None:
        self.bytes_yielded = 0

    def __iter__(self):
        while True:
            self.bytes_yielded += 8192
            yield b"x" * 8192


def test_drain_is_bounded() -> None:
    body = _InfiniteBody()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = post_without_reading_response_body(client, _OWNER, _PAYLOAD)
    assert response.status_code == 200
    assert body.bytes_yielded <= MAX_DISCARD_BYTES + 8192


class _AdvancingClock:
    """Monotonic clock advancing a fixed step per call; deterministic, no sleeping."""

    def __init__(self, step: float) -> None:
        self._now = 0.0
        self._step = step

    def monotonic(self) -> float:
        current = self._now
        self._now += self._step
        return current


class _ByteDripBody(httpx.SyncByteStream):
    """One byte per read — a slow drip once the clock advances per read."""

    def __init__(self) -> None:
        self.bytes_yielded = 0

    def __iter__(self):
        while True:
            self.bytes_yielded += 1
            yield b"x"


def test_drain_is_time_bounded(monkeypatch) -> None:
    # A peer that trickles bytes never trips the per-read timeout, so the byte
    # cap alone would let the drain run for a long time; the wall-clock budget
    # stops it early. Advance the clock one second per read instead of sleeping
    # on the real clock, so the test does not depend on CI scheduler timing —
    # mirroring the async twin in test_http_retry.py.
    monkeypatch.setattr(owner_callback, "time", _AdvancingClock(1.0))
    body = _ByteDripBody()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = post_without_reading_response_body(client, _OWNER, _PAYLOAD)

    assert response.status_code == 200
    assert body.bytes_yielded < MAX_DISCARD_BYTES


@pytest.mark.parametrize(
    ("timeout", "expected"),
    [
        (2.0, 2.0),
        (1, 1.0),
        (httpx.Timeout(1.5), 1.5),
        (httpx.Timeout(60.0, read=3.0), 3.0),
        (30.0, MAX_DISCARD_SECONDS),
        (httpx.USE_CLIENT_DEFAULT, MAX_DISCARD_SECONDS),
        (None, MAX_DISCARD_SECONDS),
        (httpx.Timeout(None), MAX_DISCARD_SECONDS),
    ],
)
def test_discard_budget_is_clamped_to_the_caller_timeout(timeout, expected) -> None:
    assert discard_budget_seconds(timeout) == expected


def test_drain_stops_earlier_with_a_short_caller_timeout(monkeypatch) -> None:
    # The wall-clock budget is clamped to the caller's per-request timeout, so a
    # 2s best-effort callback does not drain for the full MAX_DISCARD_SECONDS.
    monkeypatch.setattr(owner_callback, "time", _AdvancingClock(1.0))
    body = _ByteDripBody()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = post_without_reading_response_body(client, _OWNER, _PAYLOAD, timeout=2.0)

    assert response.status_code == 200
    assert body.bytes_yielded <= 3


@pytest.mark.parametrize("helper", ["owner_delivered", "post_without_reading_response_body"])
def test_timeout_reaches_client_stream(helper: str) -> None:
    client = MagicMock()
    client.stream.return_value.__enter__.return_value = httpx.Response(
        200, request=httpx.Request("POST", _OWNER)
    )
    if helper == "owner_delivered":
        owner_delivered(client, _OWNER, _PAYLOAD, lambda _msg: None, timeout=1.5)
    else:
        post_without_reading_response_body(client, _OWNER, _PAYLOAD, timeout=1.5)
    assert client.stream.call_args.kwargs["timeout"] == 1.5
    assert client.stream.call_args.kwargs["follow_redirects"] is False
