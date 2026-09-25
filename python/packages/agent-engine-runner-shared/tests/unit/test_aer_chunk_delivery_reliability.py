"""Tests for AER chunk-delivery reliability.

Two layers are exercised:

Part A — ``_send_stream_chunk`` retry semantics:
  - Transient httpx failures (connection error, 5xx, timeouts) are retried
    with bounded exponential backoff.
  - Terminal-chunk (``done`` / ``error``) exhaustion re-raises so the caller
    can observe.
  - Non-terminal-chunk (``text`` / ``subagent_start`` / ``subagent_end``)
    exhaustion logs at WARNING and returns; the run is not blocked.
  - Non-transient 4xx errors are not retried.
  - ``_chunk_seq`` is incremented once per call and popped on success or
    on terminal-failure re-raise.

Part B — caller-side wrap around the DONE chunk send in ``execute``:
  - Happy path: DONE chunk succeeds, then the COMPLETED callback is delivered.
    fires, ``execute`` returns ``status="completed"``.
  - DONE chunk send raises after retries: caller catches, logs ERROR, and
    STILL delivers the COMPLETED callback with the execution result.
    OE's existing ``exec.Status`` synthesis fallback uses the populated
    ``exec.Result`` to render a valid ``done`` SSE chunk for the user.
  - An exception originating before the DONE chunk path (e.g. agent
    execution itself raising) still routes through the outer ``except`` and
    reports status ``"ERROR"`` — the Part B wrap does not swallow legitimate
    agent errors.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from agent_engine_sdk import Message
from fastapi import HTTPException

from agent_engine_runner_shared.context import clear_execution_context, set_execution_context
from agent_engine_runner_shared.models import InterruptResult, StreamingResult
from agent_engine_runner_shared.server.aer import AERServer
from agent_engine_runner_shared.server.callback_delivery import CallbackDelivery, _CallbackKey
from agent_engine_runner_shared.server.chunk_types import (
    DONE,
    ERROR,
    SUBAGENT_END,
    TEXT,
    TIMEOUT_ERROR_CODE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server() -> AERServer:
    """Construct an AERServer with mocks, bypassing FastAPI / httpx wiring."""
    runtime = MagicMock()
    server = AERServer.__new__(AERServer)
    server.runtime = runtime  # type: ignore[attr-defined]
    server._client = None  # type: ignore[attr-defined]
    server._client_lock = asyncio.Lock()  # type: ignore[attr-defined]
    server._a2a_registered = False  # type: ignore[attr-defined]
    server._chunk_seq = {}  # type: ignore[attr-defined]
    server._owner_callback_url = {}  # type: ignore[attr-defined]
    server._callback_delivery = CallbackDelivery(  # type: ignore[attr-defined]
        lambda url: server._get_client(url)  # type: ignore[attr-defined]
    )
    server._durability_owner_id = "owner-test"  # type: ignore[attr-defined]
    server._send_callback_body = AsyncMock()  # type: ignore[attr-defined]
    return server


def _make_callback_server() -> AERServer:
    server = _make_server()
    del server._send_callback_body  # type: ignore[attr-defined]
    server._clients = {}  # type: ignore[attr-defined]
    server._client_cert_mtimes = {}  # type: ignore[attr-defined]
    return server


def _terminal_key(execution_id: str) -> _CallbackKey:
    return _CallbackKey(execution_id, "terminal")


def _suspension_key(execution_id: str, generation: int) -> _CallbackKey:
    return _CallbackKey(execution_id, "suspension", generation)


def _start_retained_callback(
    server: AERServer,
    oe_url: str,
    callback_body: dict[str, Any],
) -> None:
    callback_key = (
        _suspension_key(
            str(callback_body["execution_id"]),
            int(callback_body["suspend_generation"]),
        )
        if callback_body.get("status") == "SUSPENDED"
        else _terminal_key(str(callback_body["execution_id"]))
    )
    pending = server._callback_delivery._reserve(  # type: ignore[attr-defined]
        callback_key,
        oe_url,
        callback_body,
    )
    if pending is not None:
        server._callback_delivery._start_redelivery(callback_key)  # type: ignore[attr-defined]


def _record_callback(callbacks: list[dict[str, Any]]):
    async def _capture(_oe_url: str, body: dict[str, Any], **_kwargs: Any) -> None:
        callbacks.append(
            {
                key: value
                for key, value in body.items()
                if key != "execution_id" and value is not None
            }
        )

    return _capture


def _restore_real_callback_sender(server: AERServer) -> None:
    """Rebind the real callback transport helper on __new__-constructed servers.

    Most tests keep `_send_callback_body` mocked so they can assert payloads
    without touching the HTTP helper. The end-to-end owner-routing cases below
    intentionally exercise the real callback transport, so they must override
    that default mock.
    """

    server._send_callback_body = AERServer._send_callback_body.__get__(  # type: ignore[attr-defined]
        server,
        AERServer,
    )


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    """Build an httpx.HTTPStatusError shaped like response.raise_for_status() would raise."""
    request = httpx.Request("POST", "http://oe/stream/chunk")
    response = httpx.Response(status_code=status_code, request=request)
    return httpx.HTTPStatusError(f"{status_code} response", request=request, response=response)


class _ScriptedPost:
    """A scripted ``client.post`` whose attempts succeed / fail per a list.

    Each entry in ``script`` is either:
      - an Exception instance: raised on that attempt
      - an int status_code: a response whose ``raise_for_status`` raises on 4xx/5xx
      - a tuple ("ok",): success (no raise)
    """

    def __init__(self, script: list[Any]) -> None:
        self._script = script
        self.calls = 0
        self.bodies: list[dict[str, Any]] = []
        self.target_urls: list[str] = []

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        self.target_urls.append(str(args[0]))
        self.bodies.append(json.loads(json.dumps(kwargs["json"])))
        if self.calls > len(self._script):
            raise AssertionError(
                f"post called {self.calls} times but script only has {len(self._script)} entries"
            )
        entry = self._script[self.calls - 1]
        if isinstance(entry, Exception):
            raise entry
        # Success: return a response whose raise_for_status() is a no-op.
        response = MagicMock()
        response.raise_for_status = MagicMock(return_value=None)
        if isinstance(entry, int):
            response.status_code = entry
            if entry >= 400:
                response.raise_for_status = MagicMock(side_effect=_http_status_error(entry))
            else:
                response.status_code = 200
        return response


class _AsyncPostStream:
    def __init__(self, post: Any, url: str, payload: dict[str, Any]) -> None:
        self._post = post
        self._url = url
        self._payload = payload

    async def __aenter__(self) -> Any:
        return await self._post(self._url, json=self._payload)

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def _install_scripted_client(server: AERServer, script: list[Any]) -> _ScriptedPost:
    """Wire a scripted httpx client onto the server, returning the call recorder."""
    post = _ScriptedPost(script)
    client = MagicMock()
    client.post = post
    client.stream.side_effect = lambda method, url, *, json, follow_redirects: _AsyncPostStream(
        post, url, json
    )

    async def _fake_get_client(url: str) -> Any:
        return client

    server._get_client = _fake_get_client  # type: ignore[assignment]
    return post


# Speed up tests: the retry helper sleeps between attempts; collapse to zero.
@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fast_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("agent_engine_runner_shared.server.aer.asyncio.sleep", _fast_sleep)


# ---------------------------------------------------------------------------
# Terminal callback redelivery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "fields"),
    [
        ("COMPLETED", {"result": "done"}),
        ("ERROR", {"error": "agent failed"}),
        ("SUSPENDED", {"suspend_generation": 0, "suspend_reason": "approval"}),
    ],
)
async def test_transient_callback_failure_is_retained_until_ack(
    status: str,
    fields: dict[str, Any],
) -> None:
    server = _make_callback_server()
    post = _install_scripted_client(
        server,
        [
            httpx.ConnectError("unavailable-1"),
            httpx.ConnectError("unavailable-2"),
            httpx.ConnectError("unavailable-3"),
            ("ok",),
        ],
    )

    await server._report_callback(  # type: ignore[attr-defined]
        "http://oe",
        "exec-redelivery",
        status,
        **fields,
    )
    callback_key = (
        _suspension_key("exec-redelivery", 0)
        if status == "SUSPENDED"
        else _terminal_key("exec-redelivery")
    )
    retry_task = server._callback_delivery._tasks[callback_key]  # type: ignore[attr-defined]
    assert callback_key in server._callback_delivery._pending  # type: ignore[attr-defined]

    await retry_task

    assert post.calls == 4
    assert all(body == post.bodies[0] for body in post.bodies)
    assert post.bodies[0]["status"] == status
    assert server._callback_delivery._pending == {}  # type: ignore[attr-defined]
    assert server._callback_delivery._tasks == {}  # type: ignore[attr-defined]


async def test_generationless_suspended_callback_exhaustion_is_not_retained() -> None:
    server = _make_callback_server()
    post = _install_scripted_client(
        server,
        [httpx.ConnectError("unavailable")] * 3,
    )

    await server._send_callback_body(  # type: ignore[attr-defined]
        "http://oe",
        {
            "execution_id": "exec-suspended",
            "status": "SUSPENDED",
            "suspend_reason": "approval",
        },
    )

    assert post.calls == 3
    assert server._callback_delivery._pending == {}  # type: ignore[attr-defined]
    assert server._callback_delivery._tasks == {}  # type: ignore[attr-defined]


async def test_in_flight_suspension_does_not_suppress_resumed_terminal_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _make_callback_server()
    suspension_started = asyncio.Event()
    release_suspension = asyncio.Event()
    delivered: list[str] = []

    async def post_callback(
        _client: Any,
        _url: str,
        payload: dict[str, Any],
        **_kwargs: Any,
    ) -> None:
        status = str(payload["status"])
        if status == "SUSPENDED":
            suspension_started.set()
            await release_suspension.wait()
        delivered.append(status)

    monkeypatch.setattr(
        "agent_engine_runner_shared.server.callback_delivery.post_json_with_retries",
        post_callback,
    )
    suspended = asyncio.create_task(
        server._send_callback_body(  # type: ignore[attr-defined]
            "http://oe",
            {
                "execution_id": "exec-resumed",
                "status": "SUSPENDED",
                "suspend_generation": 0,
            },
        )
    )
    await suspension_started.wait()

    await server._send_callback_body(  # type: ignore[attr-defined]
        "http://oe",
        {"execution_id": "exec-resumed", "status": "COMPLETED"},
    )

    assert delivered == ["COMPLETED"]
    release_suspension.set()
    await suspended
    assert delivered == ["COMPLETED", "SUSPENDED"]


async def test_resumed_terminal_callback_stops_generationless_suspension_retries() -> None:
    server = _make_callback_server()
    suspension_started = asyncio.Event()
    release_suspension = asyncio.Event()
    terminal_started = asyncio.Event()
    release_terminal = asyncio.Event()
    delivered: list[str] = []

    async def post(_url: str, *, json: dict[str, Any]) -> Any:
        status = str(json["status"])
        delivered.append(status)
        if status == "SUSPENDED":
            suspension_started.set()
            await release_suspension.wait()
            raise httpx.ConnectError("suspension response lost")
        terminal_started.set()
        await release_terminal.wait()
        response = MagicMock()
        response.raise_for_status = MagicMock(return_value=None)
        return response

    client = MagicMock()
    client.stream.side_effect = lambda method, url, *, json, follow_redirects: _AsyncPostStream(
        post, url, json
    )

    async def get_client(_url: str) -> Any:
        return client

    server._get_client = get_client  # type: ignore[assignment]
    suspended = asyncio.create_task(
        server._send_callback_body(  # type: ignore[attr-defined]
            "http://oe",
            {"execution_id": "exec-stale-suspension", "status": "SUSPENDED"},
        )
    )
    await suspension_started.wait()

    terminal = asyncio.create_task(
        server._send_callback_body(  # type: ignore[attr-defined]
            "http://oe",
            {"execution_id": "exec-stale-suspension", "status": "COMPLETED"},
        )
    )
    await terminal_started.wait()
    release_suspension.set()
    await suspended

    assert delivered == ["SUSPENDED", "COMPLETED"]
    release_terminal.set()
    await terminal


async def test_suspension_generations_and_terminal_redeliver_independently() -> None:
    server = _make_callback_server()
    post = _install_scripted_client(
        server,
        [("ok",), ("ok",), ("ok",)],
    )
    _start_retained_callback(
        server,
        "http://oe",
        {
            "execution_id": "exec-resumed",
            "status": "SUSPENDED",
            "suspend_generation": 0,
        },
    )
    _start_retained_callback(
        server,
        "http://oe",
        {
            "execution_id": "exec-resumed",
            "status": "SUSPENDED",
            "suspend_generation": 1,
        },
    )
    _start_retained_callback(
        server,
        "http://oe",
        {"execution_id": "exec-resumed", "status": "COMPLETED"},
    )
    retry_tasks = list(server._callback_delivery._tasks.values())  # type: ignore[attr-defined]

    await asyncio.gather(*retry_tasks)

    assert {(body["status"], body.get("suspend_generation")) for body in post.bodies} == {
        ("SUSPENDED", 0),
        ("SUSPENDED", 1),
        ("COMPLETED", None),
    }
    assert server._callback_delivery._pending == {}  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "error_type",
    [httpx.ReadError, httpx.WriteError, httpx.CloseError],
)
async def test_established_connection_failure_is_retained_until_ack(
    error_type: type[httpx.NetworkError],
) -> None:
    server = _make_callback_server()
    request = httpx.Request("POST", "http://oe/executor/callback")
    post = _install_scripted_client(
        server,
        [
            error_type("connection reset", request=request),
            error_type("connection reset", request=request),
            error_type("connection reset", request=request),
            ("ok",),
        ],
    )

    await server._send_callback_body(  # type: ignore[attr-defined]
        "http://oe",
        {"execution_id": "exec-network-error", "status": "COMPLETED"},
    )
    retry_task = server._callback_delivery._tasks[  # type: ignore[attr-defined]
        _terminal_key("exec-network-error")
    ]
    await retry_task

    assert post.calls == 4
    assert server._callback_delivery._pending == {}  # type: ignore[attr-defined]
    assert server._callback_delivery._tasks == {}  # type: ignore[attr-defined]


async def test_5xx_callback_failure_is_redelivered() -> None:
    server = _make_callback_server()
    post = _install_scripted_client(server, [503, 503, 503, ("ok",)])

    await server._send_callback_body(  # type: ignore[attr-defined]
        "http://oe",
        {"execution_id": "exec-503", "status": "COMPLETED"},
    )
    retry_task = server._callback_delivery._tasks[_terminal_key("exec-503")]  # type: ignore[attr-defined]
    await retry_task

    assert post.calls == 4
    assert server._callback_delivery._pending == {}  # type: ignore[attr-defined]


async def test_owner_failure_retains_only_the_service_callback_target() -> None:
    server = _make_callback_server()
    server._owner_callback_url["exec-owner-redelivery"] = _OWNER  # type: ignore[attr-defined]
    post = _install_scripted_client(
        server,
        [
            httpx.ConnectError("owner unavailable"),
            httpx.ConnectError("service unavailable-1"),
            httpx.ConnectError("service unavailable-2"),
            httpx.ConnectError("service unavailable-3"),
            ("ok",),
        ],
    )

    await server._send_callback_body(  # type: ignore[attr-defined]
        _SERVICE,
        {"execution_id": "exec-owner-redelivery", "status": "COMPLETED"},
    )
    await server._callback_delivery._tasks[_terminal_key("exec-owner-redelivery")]  # type: ignore[attr-defined]

    assert post.target_urls == [
        f"{_OWNER}/executor/callback",
        f"{_SERVICE}/executor/callback",
        f"{_SERVICE}/executor/callback",
        f"{_SERVICE}/executor/callback",
        f"{_SERVICE}/executor/callback",
    ]


async def test_4xx_callback_failure_is_not_retained_or_retried() -> None:
    server = _make_callback_server()
    post = _install_scripted_client(server, [400])

    await server._send_callback_body(  # type: ignore[attr-defined]
        "http://oe",
        {"execution_id": "exec-400", "status": "ERROR"},
    )

    assert post.calls == 1
    assert server._callback_delivery._pending == {}  # type: ignore[attr-defined]
    assert server._callback_delivery._tasks == {}  # type: ignore[attr-defined]


async def test_callback_redelivery_is_independent_per_execution() -> None:
    server = _make_callback_server()
    delivered: list[str] = []

    async def post_callback(_oe_url: str, body: dict[str, Any]) -> None:
        delivered.append(str(body["execution_id"]))

    server._callback_delivery._post = post_callback  # type: ignore[assignment]
    _start_retained_callback(
        server,
        "http://oe",
        {"execution_id": "exec-a", "status": "COMPLETED"},
    )
    _start_retained_callback(
        server,
        "http://oe",
        {"execution_id": "exec-b", "status": "ERROR"},
    )
    retry_tasks = list(server._callback_delivery._tasks.values())  # type: ignore[attr-defined]

    await asyncio.gather(*retry_tasks)

    assert set(delivered) == {"exec-a", "exec-b"}
    assert server._callback_delivery._pending == {}  # type: ignore[attr-defined]


async def test_equal_duplicate_preserves_retained_callback_until_ack() -> None:
    server = _make_callback_server()
    post_started = asyncio.Event()
    release_post = asyncio.Event()

    async def post_callback(_oe_url: str, _body: dict[str, Any]) -> None:
        post_started.set()
        await release_post.wait()

    server._callback_delivery._post = post_callback  # type: ignore[assignment]
    callback = {"execution_id": "exec-duplicate", "status": "COMPLETED"}
    _start_retained_callback(server, "http://oe", callback)
    key = _terminal_key("exec-duplicate")
    retry_task = server._callback_delivery._tasks[key]  # type: ignore[attr-defined]
    await post_started.wait()
    retained = server._callback_delivery._pending[key]  # type: ignore[attr-defined]

    _start_retained_callback(
        server,
        "http://oe",
        {"execution_id": "exec-duplicate", "status": "COMPLETED"},
    )

    assert server._callback_delivery._pending[key] is retained  # type: ignore[attr-defined]
    assert server._callback_delivery._tasks[key] is retry_task  # type: ignore[attr-defined]
    release_post.set()
    await retry_task
    assert server._callback_delivery._pending == {}  # type: ignore[attr-defined]


async def test_conflicting_duplicate_is_rejected_before_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _make_callback_server()
    retry_waiting = asyncio.Event()

    async def wait_forever(_delay: float) -> None:
        retry_waiting.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("agent_engine_runner_shared.server.aer.asyncio.sleep", wait_forever)
    _start_retained_callback(
        server,
        "http://oe",
        {"execution_id": "exec-preflight-conflict", "status": "COMPLETED"},
    )
    await retry_waiting.wait()
    key = _terminal_key("exec-preflight-conflict")
    retained = server._callback_delivery._pending[key]  # type: ignore[attr-defined]
    retry_task = server._callback_delivery._tasks[key]  # type: ignore[attr-defined]
    post = _install_scripted_client(server, [("ok",)])

    await server._send_callback_body(  # type: ignore[attr-defined]
        "http://oe",
        {"execution_id": "exec-preflight-conflict", "status": "ERROR"},
    )

    assert post.calls == 0
    assert server._callback_delivery._pending[key] is retained  # type: ignore[attr-defined]
    assert retained.body["status"] == "COMPLETED"
    assert server._callback_delivery._tasks == {key: retry_task}  # type: ignore[attr-defined]
    await server.on_shutdown()


async def test_shutdown_cancels_callback_redelivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _make_callback_server()
    retry_started = asyncio.Event()

    async def wait_forever(_delay: float) -> None:
        retry_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("agent_engine_runner_shared.server.aer.asyncio.sleep", wait_forever)
    _start_retained_callback(
        server,
        "http://oe",
        {"execution_id": "exec-shutdown", "status": "COMPLETED"},
    )
    retry_task = server._callback_delivery._tasks[_terminal_key("exec-shutdown")]  # type: ignore[attr-defined]
    await retry_started.wait()

    await server.on_shutdown()

    assert retry_task.done()
    assert server._callback_delivery._pending == {}  # type: ignore[attr-defined]
    assert server._callback_delivery._tasks == {}  # type: ignore[attr-defined]


async def test_shutdown_cancels_in_flight_callback_without_retry() -> None:
    server = _make_callback_server()
    post_started = asyncio.Event()
    post_cancelled = asyncio.Event()
    post_calls = 0

    async def post_callback(_oe_url: str, _body: dict[str, Any]) -> None:
        nonlocal post_calls
        post_calls += 1
        post_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            post_cancelled.set()
            raise

    server._callback_delivery._post = post_callback  # type: ignore[assignment]
    _start_retained_callback(
        server,
        "http://oe",
        {"execution_id": "exec-in-flight-shutdown", "status": "COMPLETED"},
    )
    retry_task = server._callback_delivery._tasks[  # type: ignore[attr-defined]
        _terminal_key("exec-in-flight-shutdown")
    ]
    await post_started.wait()

    await server.on_shutdown()

    assert post_cancelled.is_set()
    assert post_calls == 1
    assert retry_task.done()
    assert server._callback_delivery._pending == {}  # type: ignore[attr-defined]
    assert server._callback_delivery._tasks == {}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Part A — _send_stream_chunk retry semantics
# ---------------------------------------------------------------------------


async def test_transient_connect_error_then_success_returns_normally() -> None:
    server = _make_server()
    post = _install_scripted_client(
        server,
        [httpx.ConnectError("transient"), ("ok",)],
    )

    await server._send_stream_chunk(  # type: ignore[attr-defined]
        "http://oe",
        "exec-1",
        chunk_type=TEXT,
        content="hello",
    )

    assert post.calls == 2
    # Counter incremented exactly once (one _send_stream_chunk call, two HTTP attempts).
    # Non-terminal chunks do not pop, so it lingers at 1.
    assert server._chunk_seq.get("exec-1") == 1  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "transient_exc",
    [
        httpx.ConnectTimeout("timed out connecting"),
        httpx.PoolTimeout("connection pool exhausted"),
        httpx.ReadTimeout("read timed out"),
        httpx.WriteTimeout("write timed out"),
    ],
    ids=["connect_timeout", "pool_timeout", "read_timeout", "write_timeout"],
)
async def test_transient_timeout_exceptions_are_retried(
    transient_exc: httpx.TimeoutException,
) -> None:
    """All four ``httpx.TimeoutException`` subclasses must be retried.

    ``ConnectTimeout`` and ``PoolTimeout`` are NOT subclasses of
    ``ConnectError``; they descend from ``TimeoutException``. Catching
    ``TimeoutException`` covers the full timeout family so OE pod
    warm-up does not abort on the first TCP-handshake timeout.
    """
    server = _make_server()
    post = _install_scripted_client(server, [transient_exc, ("ok",)])

    await server._send_stream_chunk(  # type: ignore[attr-defined]
        "http://oe",
        "exec-timeout",
        chunk_type=TEXT,
        content="hello",
    )

    assert post.calls == 2  # retried exactly once before succeeding
    assert server._chunk_seq.get("exec-timeout") == 1  # type: ignore[attr-defined]


async def test_transient_503_is_retried_and_succeeds_on_retry() -> None:
    server = _make_server()
    post = _install_scripted_client(server, [503, ("ok",)])

    await server._send_stream_chunk(  # type: ignore[attr-defined]
        "http://oe",
        "exec-2",
        chunk_type=TEXT,
        content="hi",
    )

    assert post.calls == 2
    assert server._chunk_seq.get("exec-2") == 1  # type: ignore[attr-defined]


async def test_terminal_chunk_exhaustion_raises_to_caller() -> None:
    server = _make_server()
    post = _install_scripted_client(
        server,
        [
            httpx.ConnectError("t1"),
            httpx.ConnectError("t2"),
            httpx.ConnectError("t3"),
        ],
    )

    with pytest.raises(httpx.ConnectError):
        await server._send_stream_chunk(  # type: ignore[attr-defined]
            "http://oe",
            "exec-3",
            chunk_type=DONE,
            content="final answer",
        )

    assert post.calls == 3
    # Terminal failure must pop _chunk_seq before re-raising.
    assert "exec-3" not in server._chunk_seq  # type: ignore[attr-defined]


async def test_non_terminal_chunk_exhaustion_logs_warning_and_returns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = _make_server()
    post = _install_scripted_client(
        server,
        [
            httpx.ConnectError("t1"),
            httpx.ConnectError("t2"),
            httpx.ConnectError("t3"),
        ],
    )

    caplog.set_level(logging.WARNING, logger="agent_engine_runner_shared.server.aer")
    # Should NOT raise — non-terminal degrades gracefully.
    await server._send_stream_chunk(  # type: ignore[attr-defined]
        "http://oe",
        "exec-4",
        chunk_type=SUBAGENT_END,
        content="",
        metadata={"subagent_name": "sec"},
    )

    assert post.calls == 3
    # WARNING-level log present (not DEBUG) so failures are visible in prod logs.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a WARNING-level log on non-terminal chunk exhaustion"
    assert any(SUBAGENT_END in r.getMessage() for r in warnings)
    # Non-terminal chunks never pop _chunk_seq.
    assert server._chunk_seq.get("exec-4") == 1  # type: ignore[attr-defined]


async def test_non_transient_400_is_not_retried(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Terminal 400 raises after a single attempt.
    server = _make_server()
    post = _install_scripted_client(server, [400])

    with pytest.raises(httpx.HTTPStatusError):
        await server._send_stream_chunk(  # type: ignore[attr-defined]
            "http://oe",
            "exec-5a",
            chunk_type=DONE,
            content="x",
        )
    assert post.calls == 1  # no retry on 4xx

    # Non-terminal 400 logs WARNING and returns, also without retry.
    server2 = _make_server()
    post2 = _install_scripted_client(server2, [400])
    caplog.set_level(logging.WARNING, logger="agent_engine_runner_shared.server.aer")
    await server2._send_stream_chunk(  # type: ignore[attr-defined]
        "http://oe",
        "exec-5b",
        chunk_type=TEXT,
        content="x",
    )
    assert post2.calls == 1  # no retry on 4xx


async def test_chunk_seq_increments_once_per_call_and_pops_on_success_and_terminal_fail() -> None:
    server = _make_server()
    # First call: non-terminal SUCCESS — seq goes 0 -> 1, does NOT pop.
    _install_scripted_client(server, [("ok",)])
    await server._send_stream_chunk(  # type: ignore[attr-defined]
        "http://oe",
        "exec-6",
        chunk_type=TEXT,
        content="a",
    )
    assert server._chunk_seq["exec-6"] == 1  # type: ignore[attr-defined]

    # Second call: terminal SUCCESS — seq goes 1 -> 2, then pops.
    _install_scripted_client(server, [("ok",)])
    await server._send_stream_chunk(  # type: ignore[attr-defined]
        "http://oe",
        "exec-6",
        chunk_type=DONE,
        content="done",
    )
    assert "exec-6" not in server._chunk_seq  # type: ignore[attr-defined]

    # Third call (new execution): terminal FAILURE — seq incremented but popped before re-raise.
    _install_scripted_client(
        server,
        [
            httpx.ConnectError("t1"),
            httpx.ConnectError("t2"),
            httpx.ConnectError("t3"),
        ],
    )
    with pytest.raises(httpx.ConnectError):
        await server._send_stream_chunk(  # type: ignore[attr-defined]
            "http://oe",
            "exec-7",
            chunk_type=DONE,
            content="x",
        )
    assert "exec-7" not in server._chunk_seq  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Part B — caller-side wrap in execute() around the DONE chunk send
# ---------------------------------------------------------------------------


def _execute_request(execution_id: str = "exec-b") -> Any:
    from agent_engine_runner_shared.models import ExecuteRequest

    return ExecuteRequest(
        execution_id=execution_id,
        message="hi",
        platform_api_url="http://oe",
        user_id="u-1",
        custom_headers={},
        resume=False,
    )


def _wire_execute_path(
    server: AERServer,
    *,
    execution_outcome: Optional[StreamingResult] = None,
    stream_raises: Optional[Exception] = None,
) -> None:
    """Stub out the agent runtime + memory writer + context plumbing so we can
    drive ``_handle_execute`` end-to-end without a real LangGraph build.

    Either ``execution_outcome`` (returned from the agent stream) OR
    ``stream_raises`` (an exception thrown by the agent stream) must be set.
    """
    server.runtime.get_agent = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
    server.runtime._memory_writer = None  # type: ignore[attr-defined]
    server.runtime.org_id = "org-1"  # type: ignore[attr-defined]

    async def _fake_stream(*_args: Any, **_kwargs: Any) -> Any:
        if stream_raises is not None:
            raise stream_raises
        return execution_outcome

    server._execute_via_agent_stream = _fake_stream  # type: ignore[assignment]


async def test_happy_path_done_chunk_succeeds_then_callback_completed_fires() -> None:
    server = _make_server()
    outcome = StreamingResult(content="42", messages=[])
    _wire_execute_path(server, execution_outcome=outcome)

    sent: list[dict[str, Any]] = []

    async def _capture_send(*_args: Any, **kwargs: Any) -> None:
        sent.append(kwargs)

    server._send_stream_chunk = _capture_send  # type: ignore[assignment]

    callbacks: list[dict[str, Any]] = []
    server._send_callback_body = _record_callback(callbacks)  # type: ignore[assignment]

    with patch.object(AERServer, "_get_client", new=AsyncMock(return_value=MagicMock())):
        response = await server._handle_execute(_execute_request("exec-b1"))

    # DONE chunk was attempted.
    assert any(call.get("chunk_type") == DONE for call in sent)
    # COMPLETED callback fired with the unmodified content.
    assert callbacks == [
        {"status": "COMPLETED", "result": "42"},
    ]
    assert response.status == "completed"
    assert response.result == "42"


async def test_execute_does_not_advertise_request_workspace_without_app_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _make_server()
    outcome = StreamingResult(content="42", messages=[])
    _wire_execute_path(server, execution_outcome=outcome)
    server._send_stream_chunk = AsyncMock()  # type: ignore[assignment]
    server._ensure_oe_registrations = AsyncMock()  # type: ignore[method-assign]
    monkeypatch.delenv("APP_ID", raising=False)
    request = _execute_request("exec-untrusted-workspace").model_copy(
        update={"workspace_id": "ws-from-request"}
    )

    with patch.object(AERServer, "_get_client", new=AsyncMock(return_value=MagicMock())):
        await server._handle_execute(request)

    server._ensure_oe_registrations.assert_not_awaited()  # type: ignore[attr-defined]


async def test_done_chunk_carries_final_message_artifact_metadata() -> None:
    server = _make_server()
    artifact = {"id": "chart-1", "kind": "chart", "title": "Loss frequency"}
    outcome = StreamingResult(
        content="Here is the chart.",
        messages=[
            Message(
                role="assistant",
                content="Here is the chart.",
                additional_kwargs={
                    "artifacts": [artifact],
                },
            ),
        ],
    )
    _wire_execute_path(server, execution_outcome=outcome)

    sent: list[dict[str, Any]] = []

    async def _capture_send(*_args: Any, **kwargs: Any) -> None:
        sent.append(kwargs)

    server._send_stream_chunk = _capture_send  # type: ignore[assignment]
    server._send_callback_body = AsyncMock()  # type: ignore[assignment]

    with patch.object(AERServer, "_get_client", new=AsyncMock(return_value=MagicMock())):
        await server._handle_execute(_execute_request("exec-artifact"))

    done_call = next(call for call in sent if call.get("chunk_type") == DONE)
    assert done_call["metadata"] == {
        "status": "completed",
        "artifacts": [artifact],
    }


async def test_done_chunk_failure_still_fires_completed_callback_with_original_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The critical reliability invariant: when DONE chunk delivery exhausts
    its retries, the caller-side wrap must still propagate the result to OE
    via the COMPLETED callback using the ORIGINAL
    ``execution_outcome.content`` — not the exception string.
    """
    server = _make_server()
    outcome = StreamingResult(content="the actual answer", messages=[])
    _wire_execute_path(server, execution_outcome=outcome)

    chunk_error = httpx.ConnectError("oe unreachable")

    async def _send_raises_on_done(*_args: Any, **kwargs: Any) -> None:
        if kwargs.get("chunk_type") == DONE:
            raise chunk_error

    server._send_stream_chunk = _send_raises_on_done  # type: ignore[assignment]

    callbacks: list[dict[str, Any]] = []

    server._send_callback_body = _record_callback(callbacks)  # type: ignore[assignment]

    caplog.set_level(logging.ERROR, logger="agent_engine_runner_shared.server.aer")

    with patch.object(AERServer, "_get_client", new=AsyncMock(return_value=MagicMock())):
        response = await server._handle_execute(_execute_request("exec-b2"))

    # The COMPLETED callback fired with the ORIGINAL execution_outcome.content,
    # not str(chunk_error) and not status="ERROR".
    assert callbacks == [
        {
            "status": "COMPLETED",
            "result": "the actual answer",
        },
    ]
    # The caller logged at ERROR — terminal chunk delivery failure is a real reliability event.
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors, "expected ERROR log when DONE chunk delivery permanently fails"
    # Execute returned the successful response shape.
    assert response.status == "completed"
    assert response.result == "the actual answer"


async def test_pre_done_exception_still_routes_through_outer_error_callback() -> None:
    """Negative case: when an exception originates BEFORE the DONE chunk path
    (e.g. inside ``_execute_via_agent_stream``), the outer ``except Exception``
    still delivers an ERROR callback with the original exception. Part B's
    wrap must not swallow legitimate agent errors.
    """
    server = _make_server()
    agent_error = RuntimeError("agent blew up before result")
    _wire_execute_path(server, stream_raises=agent_error)

    server._send_stream_chunk = AsyncMock()  # type: ignore[assignment]

    callbacks: list[dict[str, Any]] = []

    server._send_callback_body = _record_callback(callbacks)  # type: ignore[assignment]

    with patch.object(AERServer, "_get_client", new=AsyncMock(return_value=MagicMock())):
        with pytest.raises(HTTPException) as exc_info:
            await server._handle_execute(_execute_request("exec-b3"))

    assert exc_info.value.status_code == 500
    # Outer error path fired exactly one callback with status="ERROR".
    assert callbacks == [
        {
            "status": "ERROR",
            "error": "agent blew up before result",
        },
    ]


async def test_timeout_error_chunk_failure_still_reports_504_with_timeout_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Symmetric to the DONE chunk wrap: when execution times out AND the ERROR
    chunk delivery exhausts its retries, the timeout handler's caller-side
    wrap must still deliver the timeout ERROR callback
    and raise ``HTTPException(504)`` with the timeout message — not 500 with
    the chunk-delivery error masking the timeout root cause.
    """
    server = _make_server()
    _wire_execute_path(server, stream_raises=asyncio.TimeoutError())

    chunk_error = httpx.ConnectError("oe unreachable")

    async def _send_raises_on_error(*_args: Any, **kwargs: Any) -> None:
        if kwargs.get("chunk_type") == ERROR:
            raise chunk_error

    server._send_stream_chunk = _send_raises_on_error  # type: ignore[assignment]

    callbacks: list[dict[str, Any]] = []

    server._send_callback_body = _record_callback(callbacks)  # type: ignore[assignment]

    caplog.set_level(logging.ERROR, logger="agent_engine_runner_shared.server.aer")

    with patch.object(AERServer, "_get_client", new=AsyncMock(return_value=MagicMock())):
        with pytest.raises(HTTPException) as exc_info:
            await server._handle_execute(_execute_request("exec-b4"))

    # The HTTPException preserves 504 (gateway timeout) and the timeout root
    # cause — NOT 500 and NOT the chunk-delivery error string.
    assert exc_info.value.status_code == 504
    assert "Execution timed out" in str(exc_info.value.detail)

    # The callback carries status="ERROR" and the ORIGINAL timeout
    # message, not str(chunk_error), tagged so a consumer can tell a deadline
    # breach from a crash.
    assert callbacks == [
        {
            "status": "ERROR",
            "error": exc_info.value.detail,
            "metadata": {"error_code": TIMEOUT_ERROR_CODE},
        },
    ]

    # ERROR-level log captured the chunk-delivery failure for ops visibility.
    errors = [
        r for r in caplog.records if r.levelno == logging.ERROR and "ERROR chunk" in r.getMessage()
    ]
    assert errors, "expected ERROR log when ERROR chunk delivery permanently fails"


async def test_empty_interrupt_snapshot_fires_error_callback_and_propagates_500() -> None:
    """Report ERROR before raising so OE does not leave the execution running."""
    server = _make_server()
    interrupt = InterruptResult(suspend_payload={}, interrupts=[], resume_schema={"type": "object"})
    _wire_execute_path(server, execution_outcome=interrupt)

    server._send_stream_chunk = AsyncMock()  # type: ignore[assignment]

    callbacks: list[dict[str, Any]] = []

    server._send_callback_body = _record_callback(callbacks)  # type: ignore[assignment]

    with patch.object(AERServer, "_get_client", new=AsyncMock(return_value=MagicMock())):
        with pytest.raises(HTTPException) as exc_info:
            await server._handle_execute(_execute_request("exec-b6"))

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == ("Agent produced an empty interrupt snapshot; cannot suspend.")

    assert callbacks == [
        {
            "status": "ERROR",
            "error": "Agent produced an empty interrupt snapshot; cannot suspend.",
        },
    ]


async def test_empty_suspend_payload_fires_error_callback_and_propagates_500() -> None:
    """When the agent stream returns an InterruptResult with an empty
    suspend_payload, the handler must (a) deliver a terminal callback with
    ``status='ERROR'`` so OE marks the execution terminal, and (b) propagate
    ``HTTPException(500)`` with the original detail message — not the
    ``"500: ..."`` re-wrapped form produced by the outer ``except Exception``.
    """
    server = _make_server()
    interrupt = InterruptResult(suspend_payload={})
    _wire_execute_path(server, execution_outcome=interrupt)

    server._send_stream_chunk = AsyncMock()  # type: ignore[assignment]

    callbacks: list[dict[str, Any]] = []

    server._send_callback_body = _record_callback(callbacks)  # type: ignore[assignment]

    with patch.object(AERServer, "_get_client", new=AsyncMock(return_value=MagicMock())):
        with pytest.raises(HTTPException) as exc_info:
            await server._handle_execute(_execute_request("exec-b5"))

    # HTTPException(500) propagates untouched — detail is the suspend-payload
    # message, NOT prefixed with "500: " (which is what str(HTTPException)
    # would produce if the outer except re-wrapped it).
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == ("Agent produced an empty suspend payload; cannot suspend.")

    # The callback fired exactly once with status="ERROR" and the
    # suspend-payload error message — OE receives a terminal status update.
    assert callbacks == [
        {
            "status": "ERROR",
            "error": "Agent produced an empty suspend payload; cannot suspend.",
        },
    ]


# ---------------------------------------------------------------------------
# Part C — owner-callback fallback on the AER transports
# ---------------------------------------------------------------------------

_SERVICE = "http://oe.ns.svc:8000"
_OWNER = "http://10-1-2-3.oe-headless.ns.svc:8000"


class _URLRecordingPost:
    """A ``client.post`` that records target URLs; fails listed URLs with a
    transport error, returns a scripted non-2xx status for ``status_urls``, and
    succeeds otherwise."""

    def __init__(
        self,
        fail_urls: frozenset[str] = frozenset(),
        status_urls: dict[str, int] | None = None,
    ) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._fail_urls = fail_urls
        self._status_urls = status_urls or {}

    async def __call__(self, url: str, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((url, kwargs.get("json")))
        if url in self._fail_urls:
            raise httpx.ConnectError("owner replica refused")
        status = self._status_urls.get(url, 200)
        response = MagicMock()
        response.status_code = status
        response.is_success = 200 <= status < 300
        if status >= 400:
            response.raise_for_status = MagicMock(side_effect=_http_status_error(status))
        else:
            response.raise_for_status = MagicMock(return_value=None)
        return response

    def urls(self) -> list[str]:
        return [url for url, _ in self.calls]


class _AsyncPostStream:
    def __init__(self, post: _URLRecordingPost, url: str, payload: Any) -> None:
        self._post = post
        self._url = url
        self._payload = payload

    async def __aenter__(self) -> Any:
        return await self._post(self._url, json=self._payload)

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def _install_url_recording_client(
    server: AERServer,
    fail_urls: frozenset[str] = frozenset(),
    status_urls: dict[str, int] | None = None,
) -> _URLRecordingPost:
    post = _URLRecordingPost(fail_urls, status_urls)
    client = MagicMock()
    client.post = post
    client.stream.side_effect = lambda method, url, *, json, follow_redirects: _AsyncPostStream(
        post, url, json
    )

    async def _fake_get_client(url: str) -> Any:
        return client

    server._get_client = _fake_get_client  # type: ignore[assignment]
    return post


async def test_send_stream_chunk_routes_to_owner_url() -> None:
    """Regression: _post_chunk_with_retries must forward owner_url — a chunk
    with a registered owner goes to the owner replica, not the service."""
    server = _make_server()
    server._owner_callback_url["exec-own1"] = _OWNER  # type: ignore[attr-defined]
    post = _install_url_recording_client(server)

    await server._send_stream_chunk(  # type: ignore[attr-defined]
        _SERVICE,
        "exec-own1",
        chunk_type=TEXT,
        content="hi",
    )

    assert post.urls() == [f"{_OWNER}/stream/chunk"]


async def test_send_stream_chunk_falls_back_to_service_on_owner_transport_error() -> None:
    server = _make_server()
    server._owner_callback_url["exec-own2"] = _OWNER  # type: ignore[attr-defined]
    post = _install_url_recording_client(server, fail_urls=frozenset({f"{_OWNER}/stream/chunk"}))

    await server._send_stream_chunk(  # type: ignore[attr-defined]
        _SERVICE,
        "exec-own2",
        chunk_type=TEXT,
        content="hi",
    )

    assert post.urls() == [f"{_OWNER}/stream/chunk", f"{_SERVICE}/stream/chunk"]
    # Identical payload delivered across the fallback.
    assert post.calls[0][1] == post.calls[1][1]
    assert server._owner_callback_url == {}  # type: ignore[attr-defined]


async def test_send_stream_chunk_stops_retrying_owner_after_first_failure() -> None:
    server = _make_server()
    server._owner_callback_url["exec-own2b"] = _OWNER  # type: ignore[attr-defined]
    post = _install_url_recording_client(server, fail_urls=frozenset({f"{_OWNER}/stream/chunk"}))

    await server._send_stream_chunk(  # type: ignore[attr-defined]
        _SERVICE,
        "exec-own2b",
        chunk_type=TEXT,
        content="first",
    )
    await server._send_stream_chunk(  # type: ignore[attr-defined]
        _SERVICE,
        "exec-own2b",
        chunk_type=TEXT,
        content="second",
    )

    assert post.urls() == [
        f"{_OWNER}/stream/chunk",
        f"{_SERVICE}/stream/chunk",
        f"{_SERVICE}/stream/chunk",
    ]
    assert server._owner_callback_url == {}  # type: ignore[attr-defined]


async def test_send_stream_chunk_owner_failure_latches_across_transports() -> None:
    server = _make_server()
    _restore_real_callback_sender(server)
    execution_id = "exec-own-cross-transport"
    server._owner_callback_url[execution_id] = _OWNER  # type: ignore[attr-defined]
    post = _install_url_recording_client(
        server,
        fail_urls=frozenset({f"{_OWNER}/stream/chunk"}),
    )
    tokens = set_execution_context(
        execution_id=execution_id,
        wrapper=None,
        oe_url=_SERVICE,
        oe_owner_url=_OWNER,
    )
    try:
        await server._send_stream_chunk(  # type: ignore[attr-defined]
            _SERVICE,
            execution_id,
            chunk_type=TEXT,
            content="first",
        )
        await server._send_callback_body(  # type: ignore[attr-defined]
            _SERVICE,
            {"execution_id": execution_id, "status": "COMPLETED"},
        )
    finally:
        clear_execution_context(tokens)

    assert post.urls() == [
        f"{_OWNER}/stream/chunk",
        f"{_SERVICE}/stream/chunk",
        f"{_SERVICE}/executor/callback",
    ]


async def test_send_stream_chunk_falls_back_to_service_on_owner_http_error() -> None:
    """ANY non-2xx owner response falls back to the service URL; the owner is
    not retried."""
    server = _make_server()
    server._owner_callback_url["exec-own-http"] = _OWNER  # type: ignore[attr-defined]
    post = _install_url_recording_client(server, status_urls={f"{_OWNER}/stream/chunk": 500})

    await server._send_stream_chunk(  # type: ignore[attr-defined]
        _SERVICE,
        "exec-own-http",
        chunk_type=TEXT,
        content="hi",
    )

    assert post.urls() == [f"{_OWNER}/stream/chunk", f"{_SERVICE}/stream/chunk"]


def _execute_request_with_owner(execution_id: str, owner_url: str) -> Any:
    from agent_engine_runner_shared.models import ExecuteRequest

    return ExecuteRequest(
        execution_id=execution_id,
        message="hi",
        platform_api_url=_SERVICE,
        platform_api_owner_url=owner_url,
        user_id="u-1",
        custom_headers={},
        resume=False,
    )


def _execute_request_without_owner(execution_id: str) -> Any:
    from agent_engine_runner_shared.models import ExecuteRequest

    return ExecuteRequest(
        execution_id=execution_id,
        message="hi",
        platform_api_url=_SERVICE,
        user_id="u-1",
        custom_headers={},
        resume=False,
    )


async def test_handle_execute_valid_owner_routes_chunk_and_callback_to_owner() -> None:
    """End-to-end through _handle_execute with the real transports: a valid
    replica-shaped owner URL carries both the DONE chunk and the terminal
    executor callback; the per-execution registration is cleaned up after."""
    server = _make_server()
    _restore_real_callback_sender(server)
    outcome = StreamingResult(content="42", messages=[])
    _wire_execute_path(server, execution_outcome=outcome)
    post = _install_url_recording_client(server)

    response = await server._handle_execute(_execute_request_with_owner("exec-own3", _OWNER))

    assert response.status == "completed"
    assert f"{_OWNER}/stream/chunk" in post.urls()
    assert f"{_OWNER}/executor/callback" in post.urls()
    # No service-URL traffic on the two owner-preferred endpoints.
    assert f"{_SERVICE}/stream/chunk" not in post.urls()
    assert f"{_SERVICE}/executor/callback" not in post.urls()
    # The registration does not outlive the execution.
    assert server._owner_callback_url == {}  # type: ignore[attr-defined]


async def test_handle_execute_without_owner_uses_service_callbacks_only() -> None:
    """Even though capability advertisement always injects
    owner_callback_fallback=true, a request that omits platform_api_owner_url
    must stay on the stable service callback path."""
    server = _make_server()
    _restore_real_callback_sender(server)
    outcome = StreamingResult(content="42", messages=[])
    _wire_execute_path(server, execution_outcome=outcome)
    post = _install_url_recording_client(server)

    response = await server._handle_execute(_execute_request_without_owner("exec-own-miss"))

    assert response.status == "completed"
    assert post.urls() == [f"{_SERVICE}/stream/chunk", f"{_SERVICE}/executor/callback"]
    assert server._owner_callback_url == {}  # type: ignore[attr-defined]


async def test_handle_execute_forged_owner_url_never_contacted() -> None:
    """Security-negative through _handle_execute: a forged
    platform_api_owner_url is rejected by validation, the forged host is never
    contacted, and every callback uses the trusted service URL."""
    forged = "http://attacker.example:8000"
    server = _make_server()
    _restore_real_callback_sender(server)
    outcome = StreamingResult(content="42", messages=[])
    _wire_execute_path(server, execution_outcome=outcome)
    post = _install_url_recording_client(server)

    response = await server._handle_execute(_execute_request_with_owner("exec-own4", forged))

    assert response.status == "completed"
    assert all(url.startswith(_SERVICE) for url in post.urls())
    assert not any("attacker.example" in url for url in post.urls())
    assert f"{_SERVICE}/stream/chunk" in post.urls()
    assert f"{_SERVICE}/executor/callback" in post.urls()
    assert server._owner_callback_url == {}  # type: ignore[attr-defined]


async def test_handle_execute_owner_registration_cleaned_up_on_agent_error() -> None:
    """The finally guard pops the owner registration even when the agent
    stream raises (no per-execution leak)."""
    server = _make_server()
    _restore_real_callback_sender(server)
    _wire_execute_path(server, stream_raises=RuntimeError("agent blew up"))
    post = _install_url_recording_client(server)

    with pytest.raises(HTTPException):
        await server._handle_execute(_execute_request_with_owner("exec-own5", _OWNER))

    # The terminal ERROR callback still preferred the owner replica.
    assert f"{_OWNER}/executor/callback" in post.urls()
    assert server._owner_callback_url == {}  # type: ignore[attr-defined]
