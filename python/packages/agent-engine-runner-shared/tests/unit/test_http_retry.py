"""Unit tests for post_json_with_retries owner-URL fallback and Retry-After.

The retry helper is the single transport policy behind the AER stream-chunk,
terminal-callback and function-mode tool-result POSTs. These tests pin the
owner-preferring behaviour: a single best-effort owner pre-attempt that never
consumes the trusted service URL's retry budget, falls back to the service URL
on ANY owner failure (transport error or any non-2xx response), is never
retried, and never raises; then the service URL keeps its full pre-owner retry
surface (4xx fail-fast, 5xx retried, bounded delta-seconds Retry-After).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import AsyncMock, MagicMock, Mock

import httpx
import pytest

import agent_engine_runner_shared.server.http_retry as http_retry
from agent_engine_runner_shared.server.http_retry import (
    parse_retry_after_seconds,
    post_json_with_retries,
)

SERVICE_URL = "https://oe.ns.svc.cluster.local:8443/stream/chunk"
OWNER_URL = "https://10-1-2-3.oe-headless.ns.svc.cluster.local:8443/stream/chunk"
PAYLOAD = {"execution_id": "exec-1", "chunk_type": "text", "content": "hello"}


class _Resp:
    def __init__(self, status_code: int = 200, headers: Optional[Dict[str, str]] = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=Mock(), response=self)


# Each scripted action is either a response to return or an exception to raise.
_Action = Union[_Resp, BaseException]


class _StreamContext:
    def __init__(self, client: "_ScriptedClient", url: str, json: Dict[str, Any]) -> None:
        self._client = client
        self._url = url
        self._json = json

    async def __aenter__(self) -> _Resp:
        return self._client._take_action(self._url, self._json)

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _ScriptedClient:
    """Async client returning per-URL scripted actions and recording every call."""

    def __init__(self, script: Dict[str, List[_Action]]) -> None:
        self._script = script
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def _take_action(self, url: str, json: Dict[str, Any]) -> _Resp:
        self.calls.append((url, json))
        action = self._script[url].pop(0)
        if isinstance(action, BaseException):
            raise action
        return action

    async def post(self, url: str, json: Dict[str, Any]) -> _Resp:
        return self._take_action(url, json)

    def stream(
        self,
        method: str,
        url: str,
        *,
        json: Dict[str, Any],
        follow_redirects: bool,
    ) -> _StreamContext:
        assert method == "POST"
        assert follow_redirects is False
        return _StreamContext(self, url, json)

    def urls(self) -> List[str]:
        return [url for url, _ in self.calls]


class _TrackingAsyncBody(httpx.AsyncByteStream):
    def __init__(self, chunk: bytes) -> None:
        self.chunk = chunk
        self.read = False

    async def __aiter__(self):
        self.read = True
        yield self.chunk


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> List[float]:
    """Replace asyncio.sleep with a recorder so backoff is observable and instant."""
    slept: List[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(http_retry.asyncio, "sleep", fake_sleep)
    return slept


async def test_owner_success_never_touches_service() -> None:
    client = _ScriptedClient({OWNER_URL: [_Resp(200)]})
    await post_json_with_retries(client, SERVICE_URL, PAYLOAD, owner_url=OWNER_URL)
    assert client.urls() == [OWNER_URL]


async def test_close_failure_does_not_duplicate_successful_owner_delivery() -> None:
    response = _Resp(200)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=response)
    context.__aexit__ = AsyncMock(side_effect=RuntimeError("close failed"))
    client = MagicMock()
    client.stream.return_value = context

    await post_json_with_retries(client, SERVICE_URL, PAYLOAD, owner_url=OWNER_URL)

    client.stream.assert_called_once()


async def test_close_failure_does_not_retry_successful_service_delivery() -> None:
    response = _Resp(200)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=response)
    context.__aexit__ = AsyncMock(side_effect=RuntimeError("close failed"))
    client = MagicMock()
    client.stream.return_value = context

    await post_json_with_retries(client, SERVICE_URL, PAYLOAD)

    client.stream.assert_called_once()


async def test_owner_transport_error_falls_back_to_service() -> None:
    client = _ScriptedClient(
        {OWNER_URL: [httpx.ConnectError("refused")], SERVICE_URL: [_Resp(200)]}
    )
    await post_json_with_retries(client, SERVICE_URL, PAYLOAD, owner_url=OWNER_URL)
    assert client.urls() == [OWNER_URL, SERVICE_URL]
    # Identical body delivered to the fallback target.
    assert client.calls[0][1] == client.calls[1][1] == PAYLOAD


async def test_owner_4xx_falls_back_to_service() -> None:
    # ANY non-2xx owner response marks the replica unusable: no raise, no retry,
    # fall through to the trusted service URL which succeeds.
    client = _ScriptedClient({OWNER_URL: [_Resp(404)], SERVICE_URL: [_Resp(200)]})
    await post_json_with_retries(client, SERVICE_URL, PAYLOAD, owner_url=OWNER_URL)
    assert client.urls() == [OWNER_URL, SERVICE_URL]
    assert client.calls[0][1] == client.calls[1][1] == PAYLOAD


async def test_owner_5xx_falls_back_to_service() -> None:
    # A 5xx owner response is likewise terminal for the owner (never retried on
    # the owner); the service URL takes over.
    client = _ScriptedClient({OWNER_URL: [_Resp(503)], SERVICE_URL: [_Resp(200)]})
    await post_json_with_retries(client, SERVICE_URL, PAYLOAD, owner_url=OWNER_URL)
    assert client.urls() == [OWNER_URL, SERVICE_URL]


async def test_owner_response_body_is_never_read() -> None:
    body = _TrackingAsyncBody(b"internal diagnostics")

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == OWNER_URL:
            return httpx.Response(500, stream=body)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await post_json_with_retries(client, SERVICE_URL, PAYLOAD, owner_url=OWNER_URL)

    assert body.read is False


async def test_service_response_body_is_never_read() -> None:
    body = _TrackingAsyncBody(b"internal diagnostics")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, stream=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await post_json_with_retries(client, SERVICE_URL, PAYLOAD, max_attempts=1)

    assert body.read is False


async def test_owner_is_never_retried() -> None:
    # The owner is a single pre-attempt: a scripted second owner action would be
    # a bug if it were ever consumed. Only one owner call is made.
    client = _ScriptedClient(
        {OWNER_URL: [_Resp(500)], SERVICE_URL: [httpx.ConnectError("x"), _Resp(200)]}
    )
    await post_json_with_retries(client, SERVICE_URL, PAYLOAD, owner_url=OWNER_URL)
    assert client.urls().count(OWNER_URL) == 1


async def test_owner_pre_attempt_does_not_consume_service_budget() -> None:
    # An unreachable owner must not steal an attempt from the trusted service:
    # the service still gets its full max_attempts (here 3) tries.
    client = _ScriptedClient(
        {
            OWNER_URL: [httpx.ConnectError("refused")],
            SERVICE_URL: [
                httpx.ConnectError("a"),
                httpx.TimeoutException("b"),
                _Resp(200),
            ],
        }
    )
    await post_json_with_retries(client, SERVICE_URL, PAYLOAD, owner_url=OWNER_URL, max_attempts=3)
    # 1 owner pre-attempt + a full 3-attempt service budget.
    assert client.urls() == [OWNER_URL, SERVICE_URL, SERVICE_URL, SERVICE_URL]


async def test_service_503_retry_after_honored(_no_sleep: List[float]) -> None:
    client = _ScriptedClient({SERVICE_URL: [_Resp(503, headers={"Retry-After": "2"}), _Resp(200)]})
    await post_json_with_retries(client, SERVICE_URL, PAYLOAD)
    assert _no_sleep == [2.0]


async def test_service_503_retry_after_capped_at_10s(_no_sleep: List[float]) -> None:
    client = _ScriptedClient(
        {SERVICE_URL: [_Resp(503, headers={"Retry-After": "999"}), _Resp(200)]}
    )
    await post_json_with_retries(client, SERVICE_URL, PAYLOAD)
    assert _no_sleep == [10.0]


async def test_service_503_http_date_retry_after_ignored(_no_sleep: List[float]) -> None:
    # Only the delta-seconds form is honored; a date falls back to exponential backoff.
    client = _ScriptedClient(
        {
            SERVICE_URL: [
                _Resp(503, headers={"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"}),
                _Resp(200),
            ]
        }
    )
    await post_json_with_retries(client, SERVICE_URL, PAYLOAD, base_delay=0.1)
    assert _no_sleep == [0.1]  # base_delay * 2**0


async def test_no_retry_on_4xx() -> None:
    client = _ScriptedClient({SERVICE_URL: [_Resp(400)]})
    with pytest.raises(httpx.HTTPStatusError):
        await post_json_with_retries(client, SERVICE_URL, PAYLOAD)
    assert client.urls() == [SERVICE_URL]


async def test_service_path_unchanged_without_owner() -> None:
    client = _ScriptedClient(
        {SERVICE_URL: [httpx.ConnectError("x"), httpx.TimeoutException("y"), _Resp(200)]}
    )
    await post_json_with_retries(client, SERVICE_URL, PAYLOAD)
    assert client.urls() == [SERVICE_URL, SERVICE_URL, SERVICE_URL]


@pytest.mark.parametrize(
    "error_type",
    [httpx.ReadError, httpx.WriteError, httpx.CloseError],
)
async def test_established_connection_failure_is_not_retried_by_default(
    error_type: type[httpx.NetworkError],
) -> None:
    request = httpx.Request("POST", SERVICE_URL)
    client = _ScriptedClient(
        {
            SERVICE_URL: [
                error_type("connection reset", request=request),
                _Resp(200),
            ]
        }
    )

    with pytest.raises(error_type):
        await post_json_with_retries(client, SERVICE_URL, PAYLOAD)

    assert client.urls() == [SERVICE_URL]


async def test_exhaustion_raises_last_exception() -> None:
    client = _ScriptedClient({SERVICE_URL: [httpx.ConnectError("a")] * 3})
    with pytest.raises(httpx.ConnectError):
        await post_json_with_retries(client, SERVICE_URL, PAYLOAD)


async def test_owner_unreachable_then_service_exhausts_raises() -> None:
    # The owner failure never masks a genuine service exhaustion.
    client = _ScriptedClient(
        {OWNER_URL: [httpx.ConnectError("owner")], SERVICE_URL: [httpx.ConnectError("svc")] * 3}
    )
    with pytest.raises(httpx.ConnectError):
        await post_json_with_retries(client, SERVICE_URL, PAYLOAD, owner_url=OWNER_URL)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2", 2.0),
        ("  10 ", 10.0),
        ("0", 0.0),
        (None, None),
        ("", None),
        ("-5", None),
        ("1.5", None),
        ("²", None),
        ("٣", None),
        ("Wed, 21 Oct 2099 07:28:00 GMT", None),
    ],
)
def test_parse_retry_after_seconds(value: Optional[str], expected: Optional[float]) -> None:
    assert parse_retry_after_seconds(value) == expected
