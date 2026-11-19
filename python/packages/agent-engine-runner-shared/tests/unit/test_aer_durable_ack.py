"""Tests for ordered durable chunk acknowledgement (negotiated durable ack).

Two layers are exercised:

Part A — ``DurableChunkSender`` (``server/chunk_delivery.py``):
  - chunks are posted in production order, coalescing everything produced
    while a request is in flight into the next request (bounded batch);
  - a batch is retried unchanged until the engine acknowledges it, and an
    acknowledged chunk is never resent;
  - batches respect the engine's per-request bound and the waiting queue is
    bounded (producers block when it is full);
  - a failed sender raises from ``send``/``settle`` instead of dropping
    chunks, and wakes blocked producers;
  - the turn budget bounds retries; ``settle(timeout=...)`` bounds the
    error-path drain below the budget;
  - the owner pre-attempt follows the shared owner-callback policy (one
    best-effort attempt, service fallback, failure latched).

Part B — AER integration:
  - the capability advertisement carries ``durable_ack``;
  - with the negotiation on, ``_send_stream_chunk`` posts batches through the
    sender; the DONE frame and the settlement wait for the drain; an
    unacknowledged drain fails the turn (ERROR terminal + ERROR callback,
    never COMPLETED);
  - a suspension waits for the drain before the SUSPENDED settlement;
  - without the negotiation, the legacy one-chunk-per-POST path is untouched
    and no sender is registered.

Part C — journal mode (``stream_journal_enabled``):
  - the poster keeps its batch form and order, but producers never block and
    never fail the turn: a full queue drops the chunk instead;
  - an exhausted retry budget or an undeliverable chunk drops the batch and
    delivery continues, instead of failing the sender;
  - ``settle`` returns at once, so an unacknowledged turn still completes (and
    a suspension still settles) — ``durable_ack_incomplete`` is never produced;
  - a dispatch that carries both fields is a journal dispatch.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException

from agent_engine_runner_shared.models import ExecuteRequest, InterruptResult, StreamingResult
from agent_engine_runner_shared.progress import emit_step
from agent_engine_runner_shared.secure_wrapper import LLMInvocationError
from agent_engine_runner_shared.server import aer as aer_module
from agent_engine_runner_shared.server import stream_journal as sj
from agent_engine_runner_shared.server.aer import AERServer
from agent_engine_runner_shared.server.callback_delivery import CallbackDelivery
from agent_engine_runner_shared.server.chunk_delivery import (
    DURABLE_ACK_MAX_BATCH_CHUNKS,
    DurableAckError,
    DurableChunkSender,
)
from agent_engine_runner_shared.server.chunk_types import (
    DONE,
    DURABLE_ACK_INCOMPLETE_ERROR_CODE,
    ERROR,
    RUN_TIMEOUT_ERROR_CODE,
    TEXT,
)
from agent_engine_runner_shared.server.stream_journal import (
    REASON_COMPLETED,
    REASON_SUSPENDED,
    StreamJournalRegistry,
)

# ---------------------------------------------------------------------------
# Scripted transport
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, headers: Optional[dict[str, str]] = None) -> None:
        self.status_code = status_code
        self.is_success = 200 <= status_code < 300
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.is_success:
            return
        request = httpx.Request("POST", "http://oe/stream/chunk")
        response = httpx.Response(self.status_code, headers=self.headers, request=request)
        raise httpx.HTTPStatusError(
            f"{self.status_code} response", request=request, response=response
        )

    async def aiter_raw(self):  # pragma: no cover - the body is never inspected
        if False:
            yield b""


class _FakeStreamCtx:
    def __init__(self, client: "_FakeStreamClient", url: str, payload: Any) -> None:
        self._client = client
        self._url = url
        self._payload = payload

    async def __aenter__(self) -> _FakeResponse:
        entry = self._client.script.pop(0) if self._client.script else self._client.default
        if callable(entry):
            entry = await entry()
        if isinstance(entry, BaseException):
            raise entry
        if isinstance(entry, _FakeResponse):
            return entry
        return _FakeResponse(entry)

    async def __aexit__(self, *_exc: Any) -> bool:
        return False


class _FakeStreamClient:
    """Scripted ``client.stream`` surface; one script entry per POST attempt.

    Entries are a status int, an exception to raise, or a zero-arg coroutine
    function resolving to either (used to gate a response on an event). Once
    the script is exhausted, ``default`` (a status int) is returned.
    """

    def __init__(self, script: Optional[list[Any]] = None, *, default: int = 200) -> None:
        self.script: list[Any] = list(script or [])
        self.default = default
        self.calls: list[dict[str, Any]] = []
        self.timeout = None

    def stream(self, _method: str, url: str, *, json: Any = None, **_kwargs: Any) -> _FakeStreamCtx:
        self.calls.append({"url": str(url), "payload": json})
        return _FakeStreamCtx(self, str(url), json)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [call["payload"] for call in self.calls]

    @property
    def urls(self) -> list[str]:
        return [call["url"] for call in self.calls]


async def _wait_until(predicate: Any, *, timeout: float = 5.0) -> None:
    """Yield until ``predicate()`` is true, failing the test on timeout."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition was not reached in time")
        await asyncio.sleep(0.001)


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse retry backoff so budget-bounded tests finish instantly."""
    real_sleep = asyncio.sleep

    async def _fast_sleep(_seconds: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr("agent_engine_runner_shared.server.http_retry.asyncio.sleep", _fast_sleep)


def _sender(
    client: _FakeStreamClient,
    *,
    budget_s: float = 30.0,
    owner_url_getter: Optional[Any] = None,
    on_owner_failure: Optional[Any] = None,
    max_queued: int = 96,
    max_batch_bytes: Optional[int] = None,
) -> DurableChunkSender:
    async def _client_getter() -> Any:
        return client

    kwargs: dict[str, Any] = {}
    if max_batch_bytes is not None:
        kwargs["max_batch_bytes"] = max_batch_bytes
    return DurableChunkSender(
        execution_id="exec-durable",
        chunk_url="http://oe/stream/chunk",
        client_getter=_client_getter,
        deadline=asyncio.get_running_loop().time() + budget_s,
        owner_url_getter=owner_url_getter,
        on_owner_failure=on_owner_failure,
        max_queued_chunks=max_queued,
        **kwargs,
    )


def _chunk(seq: int, content: str = "") -> dict[str, Any]:
    return {
        "execution_id": "exec-durable",
        "chunk_type": TEXT,
        "content": content or f"c{seq}",
        "error": "",
        "metadata": {},
        "custom_event": None,
        "seq": seq,
    }


def _chunk_contents(body: dict[str, Any]) -> list[str]:
    return [chunk["content"] for chunk in body["chunks"]]


def _sent_chunks(body: dict[str, Any]) -> list[dict[str, Any]]:
    """The chunks a request carried, in either wire shape.

    Durable-ack dispatches post a batch envelope; journal mode posts one raw
    chunk per request, the legacy shape an unnegotiated engine parses.
    """
    return body.get("chunks", [body])


# ---------------------------------------------------------------------------
# Part A — DurableChunkSender
# ---------------------------------------------------------------------------


async def test_sender_posts_in_order_and_coalesces_while_in_flight() -> None:
    gate = asyncio.Event()

    async def _gated() -> int:
        await gate.wait()
        return 200

    client = _FakeStreamClient([_gated])
    sender = _sender(client)
    try:
        await sender.send(_chunk(1))
        await _wait_until(lambda: len(client.calls) == 1)  # first POST in flight
        # Produced while the first POST is in flight: coalesced, not posted.
        await sender.send(_chunk(2))
        await sender.send(_chunk(3))
        assert len(client.calls) == 1
        gate.set()
        await sender.settle()
    finally:
        await sender.aclose()

    assert len(client.calls) == 2
    assert _chunk_contents(client.bodies[0]) == ["c1"]
    assert _chunk_contents(client.bodies[1]) == ["c2", "c3"]
    # Batch shape: top-level execution_id; a per-chunk id is present and must
    # match it (the engine accepts either).
    for body in client.bodies:
        assert body["execution_id"] == "exec-durable"
        assert all(chunk["execution_id"] == body["execution_id"] for chunk in body["chunks"])


async def test_sender_retries_same_batch_until_acknowledged() -> None:
    client = _FakeStreamClient([503, httpx.ConnectError("down"), 503, 200])
    sender = _sender(client)
    try:
        await sender.send(_chunk(1))
        await sender.send(_chunk(2))
        await sender.settle()
    finally:
        await sender.aclose()

    assert len(client.calls) == 4
    first = client.bodies[0]
    assert all(body == first for body in client.bodies)
    assert _chunk_contents(first) == ["c1", "c2"]


async def test_sender_never_resends_acknowledged_chunks() -> None:
    gate = asyncio.Event()

    async def _gated() -> int:
        await gate.wait()
        return 200

    # First batch acks; the second batch fails once, then succeeds.
    client = _FakeStreamClient([_gated, 503, 200])
    sender = _sender(client)
    try:
        await sender.send(_chunk(1))
        await _wait_until(lambda: len(client.calls) == 1)
        await sender.send(_chunk(2))
        gate.set()
        await sender.settle()
    finally:
        await sender.aclose()

    assert _chunk_contents(client.bodies[0]) == ["c1"]
    assert _chunk_contents(client.bodies[1]) == ["c2"]
    assert _chunk_contents(client.bodies[2]) == ["c2"]
    # c1 was acknowledged by the first POST and is never seen again.
    assert all("c1" not in _chunk_contents(body) for body in client.bodies[1:])


async def test_sender_batches_respect_engine_bound() -> None:
    client = _FakeStreamClient()
    sender = _sender(client)
    try:
        for seq in range(1, 41):
            await sender.send(_chunk(seq))
        await sender.settle()
    finally:
        await sender.aclose()

    # 40 chunks queued before the poster first runs: one full batch and one
    # remainder, in production order.
    assert [len(body["chunks"]) for body in client.bodies] == [
        DURABLE_ACK_MAX_BATCH_CHUNKS,
        8,
    ]
    # Production order is preserved across the batches.
    flattened = [content for body in client.bodies for content in _chunk_contents(body)]
    assert flattened == [f"c{seq}" for seq in range(1, 41)]


async def test_sender_bounds_queue_and_blocks_producers() -> None:
    gate = asyncio.Event()

    async def _gated() -> int:
        await gate.wait()
        return 200

    client = _FakeStreamClient([_gated])
    sender = _sender(client, max_queued=2)
    try:
        await sender.send(_chunk(1))
        await _wait_until(lambda: len(client.calls) == 1)  # poster holds chunk 1
        await sender.send(_chunk(2))
        await sender.send(_chunk(3))
        # The queue is full (2 waiting): a third enqueue must block.
        blocked = asyncio.ensure_future(sender.send(_chunk(4)))
        await asyncio.sleep(0.01)
        assert not blocked.done()
        gate.set()
        await asyncio.wait_for(blocked, timeout=5)
        await sender.settle()
    finally:
        await sender.aclose()

    flattened = [content for body in client.bodies for content in _chunk_contents(body)]
    assert flattened == ["c1", "c2", "c3", "c4"]


async def test_sender_failure_wakes_blocked_producer() -> None:
    gate = asyncio.Event()

    async def _gated() -> int:
        await gate.wait()
        return 503

    client = _FakeStreamClient([_gated], default=503)
    sender = _sender(client, budget_s=0.1, max_queued=1)
    try:
        await sender.send(_chunk(1))
        await _wait_until(lambda: len(client.calls) == 1)
        await sender.send(_chunk(2))
        blocked = asyncio.ensure_future(sender.send(_chunk(3)))
        await asyncio.sleep(0.01)
        assert not blocked.done()
        gate.set()
        with pytest.raises(DurableAckError):
            await asyncio.wait_for(blocked, timeout=5)
    finally:
        await sender.aclose()


async def test_sender_settle_raises_when_budget_exhausted() -> None:
    client = _FakeStreamClient(default=503)
    sender = _sender(client, budget_s=0.15)
    try:
        await sender.send(_chunk(1))
        # The poster may fail (timed out) or this wait may hit the deadline
        # first (incomplete); both mean unacknowledged chunks.
        with pytest.raises(DurableAckError, match="acknowledgement (timed out|incomplete)"):
            await sender.settle()
        # The failure is sticky: later sends raise instead of queueing.
        with pytest.raises(DurableAckError):
            await sender.send(_chunk(2))
    finally:
        await sender.aclose()


async def test_sender_never_retries_client_error() -> None:
    client = _FakeStreamClient([400])
    sender = _sender(client, budget_s=30.0)
    try:
        await sender.send(_chunk(1))
        with pytest.raises(DurableAckError, match="failed"):
            await sender.settle()
    finally:
        await sender.aclose()
    assert len(client.calls) == 1


async def test_sender_settle_timeout_bounds_error_drain() -> None:
    client = _FakeStreamClient(default=503)
    sender = _sender(client, budget_s=30.0)
    try:
        await sender.send(_chunk(1))
        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(DurableAckError):
            await sender.settle(timeout=0.15)
        assert loop.time() - started < 5.0
    finally:
        await sender.aclose()


async def test_sender_caps_honored_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A cooperative 503 Retry-After hint is honored but capped (10s), exactly
    # like the bounded helper and the TypeScript twin: a huge hint must not
    # consume the turn budget without another attempt.
    requested_sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def _record_sleep(seconds: float) -> None:
        requested_sleeps.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr("agent_engine_runner_shared.server.http_retry.asyncio.sleep", _record_sleep)
    client = _FakeStreamClient([_FakeResponse(503, headers={"Retry-After": "600"}), 200])
    sender = _sender(client)
    try:
        await sender.send(_chunk(1))
        await sender.settle()
    finally:
        await sender.aclose()

    assert len(client.calls) == 2
    assert requested_sleeps, "the honored Retry-After must pause before retrying"
    assert max(requested_sleeps) <= 10.0, requested_sleeps


async def test_sender_owner_attempt_then_service_fallback() -> None:
    # The owner pre-attempt goes through the shared helper's POST path against
    # the same client: first call fails, the service call succeeds.
    client = _FakeStreamClient([503, 200])
    owner_failures: list[bool] = []
    owner_url = "http://owner-0/stream/chunk"

    sender = _sender(
        client,
        owner_url_getter=lambda: owner_url,
        on_owner_failure=lambda: owner_failures.append(True),
    )
    try:
        await sender.send(_chunk(1))
        await sender.settle()
    finally:
        await sender.aclose()

    assert owner_failures == [True]
    assert client.urls[0] == owner_url
    assert client.urls[1] == "http://oe/stream/chunk"


async def test_sender_batches_respect_engine_size_bound() -> None:
    # A small cap so the size bound is exercised without megabyte payloads:
    # each ~100-char chunk serializes to ~217 bytes with its envelope.
    client = _FakeStreamClient()
    sender = _sender(client, max_batch_bytes=700)
    try:
        for seq in range(1, 5):
            await sender.send(_chunk(seq, content="x" * 100))
        await sender.settle()
    finally:
        await sender.aclose()

    # Envelope + 3 chunks fits 700 bytes; the fourth starts a new batch.
    sizes = [len(body["chunks"]) for body in client.bodies]
    assert sizes == [3, 1], sizes
    for body in client.bodies:
        assert len(json.dumps(body, separators=(",", ":"), ensure_ascii=False)) <= 700


async def test_sender_fails_explicitly_on_oversized_chunk() -> None:
    client = _FakeStreamClient()
    sender = _sender(client, max_batch_bytes=420)
    try:
        await sender.send(_chunk(1, content="x" * 500))
        with pytest.raises(DurableAckError, match="exceeds the engine's request size limit"):
            await sender.settle()
    finally:
        await sender.aclose()
    # The oversized chunk is never POSTed.
    assert client.calls == []


async def test_sender_retries_established_connection_failures() -> None:
    # A read failure on a live connection is transient for idempotent delivery
    # and must not end the turn.
    client = _FakeStreamClient([httpx.ReadError("connection reset"), 200])
    sender = _sender(client)
    try:
        await sender.send(_chunk(1))
        await sender.settle()
    finally:
        await sender.aclose()
    assert len(client.calls) == 2


async def test_sender_stops_after_terminal_failure() -> None:
    # A fatal (non-retryable) response ends delivery for the turn: later
    # queued output must never be posted, and a producer blocked by the queue
    # cap is released with the failure.
    gate = asyncio.Event()

    async def _gated_failure() -> int:
        await gate.wait()
        return 400

    client = _FakeStreamClient([_gated_failure], default=200)
    sender = _sender(client, max_queued=2)
    try:
        await sender.send(_chunk(1))
        await _wait_until(lambda: len(client.calls) == 1)  # batch 1 in flight
        await sender.send(_chunk(2))
        await sender.send(_chunk(3))
        blocked = asyncio.ensure_future(sender.send(_chunk(4)))
        await asyncio.sleep(0.01)
        assert not blocked.done()
        gate.set()
        with pytest.raises(DurableAckError):
            await blocked
        with pytest.raises(DurableAckError):
            await sender.settle()
    finally:
        await sender.aclose()
    # Exactly one POST: the failed batch, and nothing after it.
    assert len(client.calls) == 1


async def test_sender_batches_respect_utf8_byte_bound_with_multibyte_content() -> None:
    # Escaped and multibyte content changes the wire size: the accounting must
    # count UTF-8 bytes (not code points), and the cap must force the split
    # well below the 32-chunk count.
    client = _FakeStreamClient()
    tricky = 'a"b\\c\n\u00e9\U0001f600' * 20
    sender = _sender(client, max_batch_bytes=800)
    try:
        for seq in range(1, 6):
            await sender.send(_chunk(seq, content=tricky))
        await sender.settle()
    finally:
        await sender.aclose()

    def wire(body: dict[str, Any]) -> bytes:
        return json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def chars(body: dict[str, Any]) -> int:
        return len(json.dumps(body, separators=(",", ":"), ensure_ascii=False))

    # A character-based accountant would pair two chunks (their combined
    # character size fits 800) even though their UTF-8 wire size does not:
    # establish the discrimination explicitly, then assert the runner never
    # pairs them.
    two_chunks = {
        "execution_id": "exec-durable",
        "chunks": [_chunk(1, content=tricky), _chunk(2, content=tricky)],
    }
    assert chars(two_chunks) <= 800 < len(wire(two_chunks))
    assert len(client.bodies) > 1, "the byte cap must force a split"
    assert all(len(body["chunks"]) == 1 for body in client.bodies)
    for body in client.bodies:
        assert len(wire(body)) <= 800, len(wire(body))
    # Production order is preserved across the split batches.
    assert [c["content"] for body in client.bodies for c in body["chunks"]] == [tricky] * 5


async def test_sender_aclose_stops_poster() -> None:
    client = _FakeStreamClient([200])
    sender = _sender(client)
    await sender.aclose()
    with pytest.raises(DurableAckError, match="closed"):
        await sender.send(_chunk(1))


# ---------------------------------------------------------------------------
# Part B — AER integration
# ---------------------------------------------------------------------------


def _make_server() -> AERServer:
    runtime = MagicMock()
    runtime._memory_writer = None
    # The per-execution graph build is awaited before the agent is materialized;
    # a bare MagicMock is not awaitable.
    runtime.build_graph = AsyncMock(return_value=None)
    runtime.org_id = "org-1"
    server = AERServer.__new__(AERServer)
    server.runtime = runtime
    server._clients = {}
    server._client_lock = asyncio.Lock()
    server._a2a_registered = False
    server._capabilities_advertised = False
    server._chunk_seq = {}
    server._durable_ack_senders = {}
    server._owner_callback_url = {}
    server._durability_owner_id = "owner-test"
    server._callback_delivery = CallbackDelivery(lambda url: server._get_client(url))
    server._send_callback_body = AsyncMock()
    return server


def _execute_request(
    execution_id: str = "exec-durable",
    *,
    durable_ack_enabled: bool = False,
    stream_journal_enabled: bool = False,
    stream_generation: int = 0,
) -> ExecuteRequest:
    return ExecuteRequest(
        execution_id=execution_id,
        message="hi",
        platform_api_url="http://oe",
        user_id="u-1",
        custom_headers={},
        resume=False,
        durable_ack_enabled=durable_ack_enabled,
        stream_journal_enabled=stream_journal_enabled,
        stream_generation=stream_generation,
    )


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


def _wire_execute_path(
    server: AERServer,
    *,
    execution_outcome: Any = None,
    stream: Any = None,
) -> None:
    server.runtime.get_agent = MagicMock(return_value=MagicMock())
    server._ensure_oe_registrations = AsyncMock()

    if stream is None:

        async def _fake_stream(*_args: Any, **_kwargs: Any) -> Any:
            return execution_outcome

        stream = _fake_stream
    server._execute_via_agent_stream = stream


def _install_client(server: AERServer, client: _FakeStreamClient) -> None:
    async def _fake_get_client(_url: str) -> Any:
        return client

    server._get_client = _fake_get_client


async def test_capability_advertise_includes_durable_ack() -> None:
    server = _make_server()
    server.runtime.agent_config = MagicMock()
    server.runtime.agent_config.language = "python"
    server.runtime.agent_config.framework = "langgraph"
    server.runtime.agent_config.features.explicit.return_value = {"durable_workflow": True}
    server.runtime.project_id = "proj-1"
    captured: dict[str, Any] = {}

    async def _post(_url: str, **kwargs: Any) -> Any:
        captured.update(kwargs.get("json") or {})
        response = MagicMock()
        response.status_code = 200
        return response

    client = MagicMock()
    client.post = _post

    async def _fake_get_client(_url: str) -> Any:
        return client

    server._get_client = _fake_get_client
    await server._advertise_capabilities("http://oe", "ws-1")

    assert captured["features"]["durable_ack"] is True
    assert captured["features"]["owner_callback_fallback"] is True
    assert captured["features"]["durable_workflow"] is True


async def test_negotiated_chunks_post_as_batches() -> None:
    server = _make_server()
    client = _FakeStreamClient()
    _install_client(server, client)
    request = _execute_request("exec-batch", durable_ack_enabled=True)
    server._start_durable_chunk_sender(request, MagicMock(failed=False))
    sender = server._durable_ack_senders["exec-batch"]
    try:
        await server._send_stream_chunk("http://oe", "exec-batch", chunk_type=TEXT, content="hello")
        await sender.settle()
    finally:
        await sender.aclose()

    assert len(client.calls) == 1
    body = client.bodies[0]
    assert body["execution_id"] == "exec-batch"
    assert [chunk["content"] for chunk in body["chunks"]] == ["hello"]
    assert body["chunks"][0]["seq"] == 1


async def test_execute_waits_for_ack_before_done_and_completed() -> None:
    server = _make_server()
    client = _FakeStreamClient()
    _install_client(server, client)
    callbacks: list[dict[str, Any]] = []
    server._send_callback_body = _record_callback(callbacks)

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        # Real chunk routing: three text chunks produced by the turn.
        for token in ("a", "b", "c"):
            await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content=token)
        return StreamingResult(content="abc", messages=[])

    _wire_execute_path(server, stream=_stream)

    request = _execute_request("exec-ordered", durable_ack_enabled=True)
    response = await server._handle_execute(request)

    assert response.status == "completed"
    assert callbacks == [{"status": "COMPLETED", "result": "abc"}]
    # Every chunk POST precedes the DONE POST, and the DONE frame is not batched.
    chunk_posts = [body for body in client.bodies if "chunks" in body]
    done_posts = [body for body in client.bodies if body.get("chunk_type") == DONE]
    assert len(done_posts) == 1
    flattened = [c["content"] for body in chunk_posts for c in body["chunks"]]
    assert flattened == ["a", "b", "c"]
    assert client.calls[-1]["payload"].get("chunk_type") == DONE
    # Sender torn down with the execution.
    assert server._durable_ack_senders == {}


async def test_execute_fails_turn_when_chunks_not_acknowledged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aer_module, "DURABLE_ACK_TURN_BUDGET_S", 0.2)
    server = _make_server()
    client = _FakeStreamClient(default=503)
    _install_client(server, client)
    callbacks: list[dict[str, Any]] = []
    server._send_callback_body = _record_callback(callbacks)

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="lost")
        return StreamingResult(content="lost", messages=[])

    _wire_execute_path(server, stream=_stream)

    request = _execute_request("exec-unacked", durable_ack_enabled=True)
    with pytest.raises(HTTPException) as excinfo:
        await server._handle_execute(request)

    assert excinfo.value.status_code == 500
    assert callbacks == [
        {
            "status": "ERROR",
            "error": excinfo.value.detail,
            "metadata": {"error_code": DURABLE_ACK_INCOMPLETE_ERROR_CODE},
        }
    ]
    # Never presented as complete: no DONE chunk, no COMPLETED callback.
    assert not any(body.get("chunk_type") == DONE for body in client.bodies)
    assert all(callback["status"] != "COMPLETED" for callback in callbacks)


async def test_suspension_waits_for_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _make_server()
    client = _FakeStreamClient()
    _install_client(server, client)
    callbacks: list[dict[str, Any]] = []
    server._send_callback_body = _record_callback(callbacks)

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="pre")
        return InterruptResult(suspend_payload={"suspend_reason": "approval"})

    _wire_execute_path(server, stream=_stream)
    request = _execute_request("exec-suspend", durable_ack_enabled=True)
    response = await server._handle_execute(request)

    assert response.status == "suspended"
    assert callbacks[0]["status"] == "SUSPENDED"
    # The text chunk landed before the settlement was staged.
    assert [c["content"] for body in client.bodies if "chunks" in body for c in body["chunks"]] == [
        "pre"
    ]


async def test_suspension_fails_turn_when_not_acknowledged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aer_module, "DURABLE_ACK_TURN_BUDGET_S", 0.2)
    server = _make_server()
    client = _FakeStreamClient(default=503)
    _install_client(server, client)
    callbacks: list[dict[str, Any]] = []
    server._send_callback_body = _record_callback(callbacks)

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="x")
        return InterruptResult(suspend_payload={"suspend_reason": "approval"})

    _wire_execute_path(server, stream=_stream)
    request = _execute_request("exec-suspend-lost", durable_ack_enabled=True)
    with pytest.raises(HTTPException) as excinfo:
        await server._handle_execute(request)

    assert excinfo.value.status_code == 500
    assert callbacks == [
        {
            "status": "ERROR",
            "error": excinfo.value.detail,
            "metadata": {"error_code": DURABLE_ACK_INCOMPLETE_ERROR_CODE},
        }
    ]


async def _stop_execution(server: AERServer, execution_id: str, task: Any) -> None:
    """Apply a real DrainRegistry Stop to an in-flight execution."""
    from agent_engine_runner_shared.server.drain import DrainRequest

    registry = server.drain_registry
    registry.begin_work(execution_id, task)
    registry.apply(
        DrainRequest(
            request_id="drain-1",
            execution_id=execution_id,
            reason="execution_cancelled",
            deadline_at_ms=int(time.time() * 1000) + 5_000,
        )
    )


async def test_execution_timeout_with_full_queue_settles_as_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A stalled chunk POST with a full waiting queue must not strand the
    # handler past the execution deadline: the timeout releases production and
    # settles with the timeout discriminator, bounded.
    monkeypatch.setattr(aer_module, "EXECUTION_TIMEOUT", 0.3)
    monkeypatch.setattr(aer_module, "DURABLE_ACK_ERROR_DRAIN_TIMEOUT_S", 0.2)
    stall = asyncio.Event()

    async def _stalled() -> int:
        await stall.wait()
        return 200

    server = _make_server()
    client = _FakeStreamClient([_stalled], default=200)
    _install_client(server, client)
    callbacks: list[dict[str, Any]] = []
    server._send_callback_body = _record_callback(callbacks)

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="c1")
        # Wait for the poster to hold its in-flight batch, then fill the cap
        # exactly: the next send blocks on it.
        await _wait_until(lambda: len(client.calls) == 1)
        for seq in range(2, 98):
            await server._send_stream_chunk(
                oe_url, execution_id, chunk_type=TEXT, content=f"c{seq}"
            )
        # Blocks on the queue cap; the execution timeout must release it.
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="blocked")
        raise AssertionError("production must not resume after the timeout")

    _wire_execute_path(server, stream=_stream)
    request = _execute_request("exec-timeout-full", durable_ack_enabled=True)
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(HTTPException) as excinfo:
        await server._handle_execute(request)
    assert loop.time() - started < 3.0
    assert excinfo.value.status_code == 504
    assert callbacks == [
        {
            "status": "ERROR",
            "error": excinfo.value.detail,
            "metadata": {"error_code": RUN_TIMEOUT_ERROR_CODE},
        }
    ]
    # The timeout ERROR terminal was attempted, and the poster is cleaned up.
    assert any(body.get("chunk_type") == ERROR for body in client.bodies)
    assert server._durable_ack_senders == {}


async def test_stop_during_blocked_production_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aer_module, "DURABLE_ACK_ERROR_DRAIN_TIMEOUT_S", 0.2)
    stall = asyncio.Event()

    async def _stalled() -> int:
        await stall.wait()
        return 200

    server = _make_server()
    client = _FakeStreamClient([_stalled], default=200)
    _install_client(server, client)
    callbacks: list[dict[str, Any]] = []
    server._send_callback_body = _record_callback(callbacks)
    blocked = asyncio.Event()

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="c1")
        await _wait_until(lambda: len(client.calls) == 1)
        for seq in range(2, 98):
            await server._send_stream_chunk(
                oe_url, execution_id, chunk_type=TEXT, content=f"c{seq}"
            )
        blocked.set()
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="blocked")
        raise AssertionError("production must not resume after the Stop")

    _wire_execute_path(server, stream=_stream)
    request = _execute_request("exec-stop-production", durable_ack_enabled=True)
    task = asyncio.ensure_future(server._handle_execute(request))
    await asyncio.wait_for(blocked.wait(), timeout=3)
    await _stop_execution(server, "exec-stop-production", task)
    response = await asyncio.wait_for(task, timeout=3)

    assert response.status == "cancelled"
    # A Stop is not a durable-ack failure: no coded ERROR settlement.
    assert all(callback["status"] != "ERROR" for callback in callbacks)
    assert server._durable_ack_senders == {}


async def test_stop_during_completion_settlement_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aer_module, "DURABLE_ACK_ERROR_DRAIN_TIMEOUT_S", 0.2)
    stall = asyncio.Event()

    async def _stalled() -> int:
        await stall.wait()
        return 200

    server = _make_server()
    client = _FakeStreamClient([_stalled], default=200)
    _install_client(server, client)
    callbacks: list[dict[str, Any]] = []
    server._send_callback_body = _record_callback(callbacks)

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="kept")
        return StreamingResult(content="kept", messages=[])

    _wire_execute_path(server, stream=_stream)
    settle_entered = asyncio.Event()
    original_settle = DurableChunkSender.settle

    async def _tracked_settle(self: DurableChunkSender, timeout: Any = None) -> None:
        settle_entered.set()
        await original_settle(self, timeout=timeout)

    monkeypatch.setattr(DurableChunkSender, "settle", _tracked_settle)
    request = _execute_request("exec-stop-completion", durable_ack_enabled=True)
    task = asyncio.ensure_future(server._handle_execute(request))
    # Stop only after the completion settlement is demonstrably entered.
    await asyncio.wait_for(settle_entered.wait(), timeout=3)
    sender = server._durable_ack_senders["exec-stop-completion"]
    await _stop_execution(server, "exec-stop-completion", task)
    response = await asyncio.wait_for(task, timeout=3)

    assert response.status == "cancelled"
    assert all(callback["status"] != "COMPLETED" for callback in callbacks)
    assert all(
        (callback.get("metadata") or {}).get("error_code") != DURABLE_ACK_INCOMPLETE_ERROR_CODE
        for callback in callbacks
    )
    # The poster was shut down with the turn.
    assert sender._poster.done()
    assert server._durable_ack_senders == {}


async def test_stop_during_suspension_settlement_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aer_module, "DURABLE_ACK_ERROR_DRAIN_TIMEOUT_S", 0.2)
    stall = asyncio.Event()

    async def _stalled() -> int:
        await stall.wait()
        return 200

    server = _make_server()
    client = _FakeStreamClient([_stalled], default=200)
    _install_client(server, client)
    callbacks: list[dict[str, Any]] = []
    server._send_callback_body = _record_callback(callbacks)

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="pre")
        return InterruptResult(suspend_payload={"suspend_reason": "approval"})

    _wire_execute_path(server, stream=_stream)
    settle_entered = asyncio.Event()
    original_settle = DurableChunkSender.settle

    async def _tracked_settle(self: DurableChunkSender, timeout: Any = None) -> None:
        settle_entered.set()
        await original_settle(self, timeout=timeout)

    monkeypatch.setattr(DurableChunkSender, "settle", _tracked_settle)
    request = _execute_request("exec-stop-suspension", durable_ack_enabled=True)
    task = asyncio.ensure_future(server._handle_execute(request))
    # Stop only after the suspension settlement is demonstrably entered.
    await asyncio.wait_for(settle_entered.wait(), timeout=3)
    sender = server._durable_ack_senders["exec-stop-suspension"]
    await _stop_execution(server, "exec-stop-suspension", task)
    response = await asyncio.wait_for(task, timeout=3)

    assert response.status == "cancelled"
    assert all(callback["status"] != "SUSPENDED" for callback in callbacks)
    assert all(
        (callback.get("metadata") or {}).get("error_code") != DURABLE_ACK_INCOMPLETE_ERROR_CODE
        for callback in callbacks
    )
    assert sender._poster.done()
    assert server._durable_ack_senders == {}


async def _empty_suspension_outcome(kind: str) -> Any:
    if kind == "interrupt_snapshot":
        return InterruptResult(suspend_payload={}, interrupts=[], resume_schema={"type": "object"})
    return InterruptResult(suspend_payload={})


@pytest.mark.parametrize(
    ("kind", "expected_error"),
    [
        ("interrupt_snapshot", "Agent produced an empty interrupt snapshot; cannot suspend."),
        ("suspend_payload", "Agent produced an empty suspend payload; cannot suspend."),
    ],
)
async def test_empty_suspension_drains_gated_output_within_the_window(
    kind: str,
    expected_error: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Both empty-suspension ERROR settlements must drain accepted output
    # best-effort: released inside the window, the acknowledgement precedes the
    # ERROR callback.
    monkeypatch.setattr(aer_module, "DURABLE_ACK_ERROR_DRAIN_TIMEOUT_S", 2.0)
    gate = asyncio.Event()
    events: list[str] = []

    async def _gated_ack() -> int:
        await gate.wait()
        events.append("acked")
        return 200

    server = _make_server()
    client = _FakeStreamClient([_gated_ack])
    _install_client(server, client)
    callbacks: list[dict[str, Any]] = []
    capture = _record_callback(callbacks)

    async def _record_cb(_oe_url: str, body: dict[str, Any], **kwargs: Any) -> None:
        events.append("callback")
        await capture(_oe_url, body, **kwargs)

    server._send_callback_body = _record_cb

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="kept")
        asyncio.get_running_loop().call_later(0.05, gate.set)
        return await _empty_suspension_outcome(kind)

    _wire_execute_path(server, stream=_stream)
    request = _execute_request(f"exec-empty-suspend-{kind}", durable_ack_enabled=True)
    with pytest.raises(HTTPException) as excinfo:
        await server._handle_execute(request)

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == expected_error
    assert "acked" in events and "callback" in events
    assert events.index("acked") < events.index("callback")
    assert callbacks == [{"status": "ERROR", "error": expected_error}]
    assert server._durable_ack_senders == {}


@pytest.mark.parametrize("kind", ["interrupt_snapshot", "suspend_payload"])
async def test_empty_suspension_withheld_ack_fails_bounded(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aer_module, "DURABLE_ACK_ERROR_DRAIN_TIMEOUT_S", 0.2)
    server = _make_server()
    client = _FakeStreamClient(default=503)
    _install_client(server, client)
    callbacks: list[dict[str, Any]] = []
    server._send_callback_body = _record_callback(callbacks)

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="kept")
        return await _empty_suspension_outcome(kind)

    _wire_execute_path(server, stream=_stream)
    request = _execute_request(f"exec-empty-withheld-{kind}", durable_ack_enabled=True)
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(HTTPException) as excinfo:
        await server._handle_execute(request)

    assert loop.time() - started < 3.0
    assert excinfo.value.status_code == 500
    # The original validation error is preserved (no durable-ack code).
    assert callbacks == [{"status": "ERROR", "error": excinfo.value.detail}]
    assert server._durable_ack_senders == {}


async def test_not_negotiated_path_keeps_single_chunk_posts() -> None:
    server = _make_server()
    client = _FakeStreamClient()
    _install_client(server, client)
    callbacks: list[dict[str, Any]] = []
    server._send_callback_body = _record_callback(callbacks)

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="x")
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="y")
        return StreamingResult(content="xy", messages=[])

    _wire_execute_path(server, stream=_stream)
    request = _execute_request("exec-legacy")  # durable_ack_enabled defaults to False
    response = await server._handle_execute(request)

    assert response.status == "completed"
    # Today's shape: one chunk per POST, no batch envelope anywhere.
    assert all("chunks" not in body for body in client.bodies)
    assert [body["content"] for body in client.bodies if body["chunk_type"] == TEXT] == ["x", "y"]
    assert callbacks == [{"status": "COMPLETED", "result": "xy"}]
    assert server._durable_ack_senders == {}


async def test_mid_turn_sender_failure_fails_the_turn() -> None:
    # A gated permanent delivery failure: production observes the recorded
    # failure (not a sleep) and must not return normally.
    gate = asyncio.Event()

    async def _gated_failure() -> int:
        await gate.wait()
        return 400

    server = _make_server()
    client = _FakeStreamClient([_gated_failure], default=200)
    _install_client(server, client)
    callbacks: list[dict[str, Any]] = []
    server._send_callback_body = _record_callback(callbacks)

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="a")
        gate.set()
        await _wait_until(lambda: server._durable_ack_senders[execution_id].failed is not None)
        # Production does not return normally: the failed sender raises rather
        # than silently dropping the chunk.
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="b")
        raise AssertionError("production must not continue past a failed sender")

    _wire_execute_path(server, stream=_stream)
    request = _execute_request("exec-mid", durable_ack_enabled=True)
    with pytest.raises(HTTPException) as excinfo:
        await server._handle_execute(request)

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail.startswith(
        "Stream chunk delivery could not be durably acknowledged"
    )
    # The coded ERROR callback and the attempted coded ERROR terminal.
    assert callbacks == [
        {
            "status": "ERROR",
            "error": excinfo.value.detail,
            "metadata": {"error_code": DURABLE_ACK_INCOMPLETE_ERROR_CODE},
        }
    ]
    error_terminals = [body for body in client.bodies if body.get("chunk_type") == ERROR]
    assert len(error_terminals) == 1
    assert error_terminals[0]["metadata"] == {"error_code": DURABLE_ACK_INCOMPLETE_ERROR_CODE}
    # Never presented as complete: no DONE chunk, no COMPLETED callback.
    assert not any(body.get("chunk_type") == DONE for body in client.bodies)
    assert all(callback["status"] != "COMPLETED" for callback in callbacks)


async def test_uncoded_llm_error_drains_gated_output_within_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The acknowledgement is gated and released inside the error-drain window:
    # the accepted output must be acknowledged before the settlement callback,
    # and the original uncoded error must be preserved.
    monkeypatch.setattr(aer_module, "DURABLE_ACK_ERROR_DRAIN_TIMEOUT_S", 2.0)
    gate = asyncio.Event()
    events: list[str] = []

    async def _gated_ack() -> int:
        await gate.wait()
        events.append("acked")
        return 200

    server = _make_server()
    client = _FakeStreamClient([_gated_ack])
    _install_client(server, client)
    callbacks: list[dict[str, Any]] = []
    capture = _record_callback(callbacks)

    async def _record_cb(_oe_url: str, body: dict[str, Any], **kwargs: Any) -> None:
        events.append("callback")
        await capture(_oe_url, body, **kwargs)

    server._send_callback_body = _record_cb

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="kept")
        # Release the acknowledgement inside the error-drain window.
        asyncio.get_running_loop().call_later(0.05, gate.set)
        raise LLMInvocationError("truncated relay response")

    _wire_execute_path(server, stream=_stream)
    request = _execute_request("exec-uncoded", durable_ack_enabled=True)
    with pytest.raises(HTTPException) as excinfo:
        await server._handle_execute(request)

    assert excinfo.value.status_code == 500
    assert "acked" in events and "callback" in events
    assert events.index("acked") < events.index("callback")
    assert callbacks == [
        {"status": "ERROR", "error": "LLM invocation error: truncated relay response"}
    ]


async def test_uncoded_llm_error_withheld_ack_preserves_the_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The acknowledgement never lands: the bounded drain gives up and the
    # original uncoded error and 500 are preserved.
    monkeypatch.setattr(aer_module, "DURABLE_ACK_ERROR_DRAIN_TIMEOUT_S", 0.2)
    server = _make_server()
    client = _FakeStreamClient(default=503)
    _install_client(server, client)
    callbacks: list[dict[str, Any]] = []
    server._send_callback_body = _record_callback(callbacks)

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="kept")
        raise LLMInvocationError("truncated relay response")

    _wire_execute_path(server, stream=_stream)
    request = _execute_request("exec-uncoded-withheld", durable_ack_enabled=True)
    with pytest.raises(HTTPException) as excinfo:
        await server._handle_execute(request)

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "LLM invocation error: truncated relay response"
    assert callbacks == [
        {"status": "ERROR", "error": "LLM invocation error: truncated relay response"}
    ]


async def test_ordinary_execution_error_drains_gated_output_within_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The acknowledgement is gated and released inside the error-drain window:
    # the accepted output must be acknowledged before the settlement callback,
    # and the original error must be preserved.
    monkeypatch.setattr(aer_module, "DURABLE_ACK_ERROR_DRAIN_TIMEOUT_S", 2.0)
    gate = asyncio.Event()
    events: list[str] = []

    async def _gated_ack() -> int:
        await gate.wait()
        events.append("acked")
        return 200

    server = _make_server()
    client = _FakeStreamClient([_gated_ack])
    _install_client(server, client)
    callbacks: list[dict[str, Any]] = []
    capture = _record_callback(callbacks)

    async def _record_cb(_oe_url: str, body: dict[str, Any], **kwargs: Any) -> None:
        events.append("callback")
        await capture(_oe_url, body, **kwargs)

    server._send_callback_body = _record_cb

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="kept")
        # Release the acknowledgement inside the error-drain window.
        asyncio.get_running_loop().call_later(0.05, gate.set)
        raise RuntimeError("agent blew up")

    _wire_execute_path(server, stream=_stream)
    request = _execute_request("exec-generic-gated", durable_ack_enabled=True)
    with pytest.raises(HTTPException) as excinfo:
        await server._handle_execute(request)

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "agent blew up"
    assert "acked" in events and "callback" in events
    assert events.index("acked") < events.index("callback")
    # The generic error keeps today's callback shape (no durable-ack code).
    assert callbacks == [{"status": "ERROR", "error": "agent blew up"}]


async def test_batch_payload_matches_engine_shape() -> None:
    """The engine validates a per-chunk execution_id against the top-level one
    (and stamps it when absent); the runner sends today's chunk shape."""
    server = _make_server()
    client = _FakeStreamClient()
    _install_client(server, client)
    request = _execute_request("exec-shape", durable_ack_enabled=True)
    server._start_durable_chunk_sender(request, MagicMock(failed=False))
    sender = server._durable_ack_senders["exec-shape"]
    try:
        await server._send_stream_chunk(
            "http://oe", "exec-shape", chunk_type=TEXT, content="hello", metadata={"source": "main"}
        )
        await sender.settle()
    finally:
        await sender.aclose()

    body = json.loads(json.dumps(client.bodies[0]))
    assert set(body) == {"execution_id", "chunks"}
    chunk = body["chunks"][0]
    assert set(chunk) == {
        "execution_id",
        "chunk_type",
        "content",
        "error",
        "metadata",
        "custom_event",
        "seq",
    }
    assert chunk["execution_id"] == "exec-shape"
    assert chunk["metadata"] == {"source": "main"}
    assert chunk["chunk_type"] == TEXT


# ---------------------------------------------------------------------------
# Part C — journal mode (stream_journal_enabled)
# ---------------------------------------------------------------------------


def _journal_sender(
    client: _FakeStreamClient,
    *,
    budget_s: float = 30.0,
    max_queued: int = 96,
    max_batch_bytes: Optional[int] = None,
) -> DurableChunkSender:
    async def _client_getter() -> Any:
        return client

    kwargs: dict[str, Any] = {}
    if max_batch_bytes is not None:
        kwargs["max_batch_bytes"] = max_batch_bytes
    return DurableChunkSender(
        execution_id="exec-journal",
        chunk_url="http://oe/stream/chunk",
        client_getter=_client_getter,
        deadline=asyncio.get_running_loop().time() + budget_s,
        max_queued_chunks=max_queued,
        journal_mode=True,
        **kwargs,
    )


async def test_journal_sender_posts_one_raw_chunk_per_request() -> None:
    # A journal dispatch never negotiates durable acknowledgement, and the engine
    # parses a batch envelope only under that negotiation. Live delivery keeps
    # the legacy single-chunk shape; only the failure policy changes.
    gate = asyncio.Event()

    async def _gated() -> int:
        await gate.wait()
        return 200

    client = _FakeStreamClient([_gated])
    sender = _journal_sender(client)
    try:
        await sender.send(_chunk(1))
        await _wait_until(lambda: len(client.calls) == 1)
        # Produced while the first POST is in flight: queued in order, not
        # coalesced into an envelope the engine would not parse.
        await sender.send(_chunk(2))
        await sender.send(_chunk(3))
        gate.set()
        await _wait_until(lambda: len(client.calls) == 3)
    finally:
        await sender.aclose()

    assert [c["content"] for body in client.bodies for c in _sent_chunks(body)] == [
        "c1",
        "c2",
        "c3",
    ]
    for body in client.bodies:
        assert "chunks" not in body, "journal mode must keep the legacy raw shape"
        assert body["execution_id"] == "exec-durable"
        assert body["chunk_type"] == TEXT
    assert sender.dropped == 0
    assert sender.failed is None


async def test_journal_sender_drops_instead_of_blocking_when_queue_is_full() -> None:
    stall = asyncio.Event()

    async def _stalled() -> int:
        await stall.wait()
        return 200

    client = _FakeStreamClient([_stalled])
    sender = _journal_sender(client, max_queued=2)
    try:
        # Batch 1 is in flight; the next two fill the queue.
        await sender.send(_chunk(1))
        await _wait_until(lambda: len(client.calls) == 1)
        await sender.send(_chunk(2))
        await sender.send(_chunk(3))
        # Every later send returns at once instead of blocking on the cap.
        for seq in range(4, 9):
            await asyncio.wait_for(sender.send(_chunk(seq)), timeout=1.0)
        assert sender.dropped == 5
        assert sender.failed is None
        stall.set()
    finally:
        await sender.aclose()


async def test_journal_sender_settle_does_not_wait_for_acknowledgement() -> None:
    # The engine never acknowledges; journal mode settlement still returns at
    # once and never raises, because the journal holds the turn's chunks.
    client = _FakeStreamClient(default=503)
    sender = _journal_sender(client, budget_s=0.2)
    try:
        await sender.send(_chunk(1))
        await asyncio.wait_for(sender.settle(), timeout=1.0)
        assert sender.failed is None
        # A later send still queues rather than raising on the failure.
        await asyncio.wait_for(sender.send(_chunk(2)), timeout=1.0)
    finally:
        await sender.aclose()


async def test_journal_sender_drops_a_failed_batch_and_keeps_delivering() -> None:
    # A non-retryable response drops the batch instead of failing the sender,
    # so delivery continues with what follows it.
    client = _FakeStreamClient([400, 200])
    sender = _journal_sender(client)
    try:
        await sender.send(_chunk(1))
        await _wait_until(lambda: len(client.calls) == 1)
        await sender.send(_chunk(2))
        await _wait_until(lambda: len(client.calls) == 2)
    finally:
        await sender.aclose()

    assert [c["content"] for body in client.bodies for c in _sent_chunks(body)] == ["c1", "c2"]
    assert sender.dropped == 1
    assert sender.failed is None


async def test_journal_sender_drops_an_oversized_chunk_and_keeps_delivering() -> None:
    # The durable path fails the sender on a chunk that can never fit the
    # request cap; journal mode drops it and delivers the rest.
    client = _FakeStreamClient()
    sender = _journal_sender(client, max_batch_bytes=420)
    try:
        await sender.send(_chunk(1, content="x" * 500))
        await sender.send(_chunk(2, content="ok"))
        await _wait_until(lambda: len(client.calls) == 1)
    finally:
        await sender.aclose()

    assert [c["content"] for body in client.bodies for c in _sent_chunks(body)] == ["ok"]
    assert sender.dropped == 1
    assert sender.failed is None


async def test_journal_dispatch_registers_a_journal_mode_sender() -> None:
    server = _make_server()
    request = _execute_request("exec-mode", stream_journal_enabled=True)
    server._start_durable_chunk_sender(request, MagicMock(failed=False))
    sender = server._durable_ack_senders["exec-mode"]
    try:
        assert sender.journal_mode is True
    finally:
        await sender.aclose()

    # A durable-ack dispatch still registers a durable-mode sender.
    durable_server = _make_server()
    durable_request = _execute_request("exec-durable-mode", durable_ack_enabled=True)
    durable_server._start_durable_chunk_sender(durable_request, MagicMock(failed=False))
    durable_sender = durable_server._durable_ack_senders["exec-durable-mode"]
    try:
        assert durable_sender.journal_mode is False
    finally:
        await durable_sender.aclose()


async def test_journal_dispatch_does_not_gate_settlement() -> None:
    # The decisive journal-mode behavior: a turn whose chunks are never
    # acknowledged still completes, and never produces durable_ack_incomplete.
    server = _make_server()
    client = _FakeStreamClient(default=400)
    _install_client(server, client)
    callbacks: list[dict[str, Any]] = []
    server._send_callback_body = _record_callback(callbacks)

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        for token in ("a", "b", "c"):
            await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content=token)
        return StreamingResult(content="abc", messages=[])

    _wire_execute_path(server, stream=_stream)

    request = _execute_request("exec-journal", stream_journal_enabled=True)
    response = await server._handle_execute(request)

    assert response.status == "completed"
    assert callbacks == [{"status": "COMPLETED", "result": "abc"}]
    # Sender torn down with the execution, and never presented as incomplete.
    assert server._durable_ack_senders == {}


async def test_journal_dispatch_wins_over_durable_ack() -> None:
    # A dispatch that carries both fields is a journal dispatch: settlement is
    # not gated even though the engine also negotiated durable acknowledgement.
    server = _make_server()
    client = _FakeStreamClient(default=400)
    _install_client(server, client)
    callbacks: list[dict[str, Any]] = []
    server._send_callback_body = _record_callback(callbacks)

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="x")
        return StreamingResult(content="x", messages=[])

    _wire_execute_path(server, stream=_stream)

    request = _execute_request("exec-both", durable_ack_enabled=True, stream_journal_enabled=True)
    response = await server._handle_execute(request)

    assert response.status == "completed"
    assert callbacks == [{"status": "COMPLETED", "result": "x"}]


async def test_journal_suspension_does_not_gate_settlement() -> None:
    server = _make_server()
    client = _FakeStreamClient(default=400)
    _install_client(server, client)
    callbacks: list[dict[str, Any]] = []
    server._send_callback_body = _record_callback(callbacks)

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="pre")
        return InterruptResult(suspend_payload={"suspend_reason": "approval"})

    _wire_execute_path(server, stream=_stream)
    request = _execute_request("exec-journal-suspend", stream_journal_enabled=True)
    response = await server._handle_execute(request)

    assert response.status == "suspended"
    assert callbacks[0]["status"] == "SUSPENDED"


class _JumpClock:
    """A ``time`` shim whose ``monotonic()`` jumps forward once armed.

    Lets a test place the turn's completion past its own execution deadline
    without sleeping, so the settlement deadline check is exercised exactly.
    """

    def __init__(self) -> None:
        self.jumped = False

    def monotonic(self) -> float:
        return time.monotonic() + (3600.0 if self.jumped else 0.0)

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


async def test_journal_completion_past_the_deadline_is_not_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # In journal mode settlement does not wait for the drain, so a turn that
    # finishes at (or past) its original deadline is still completed, not
    # failed as incomplete.
    clock = _JumpClock()
    monkeypatch.setattr(aer_module, "time", clock)
    server = _make_server()
    client = _FakeStreamClient(default=200)
    _install_client(server, client)
    callbacks: list[dict[str, Any]] = []
    server._send_callback_body = _record_callback(callbacks)

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="late")
        # The turn ran past its deadline before it settled.
        clock.jumped = True
        return StreamingResult(content="late", messages=[])

    _wire_execute_path(server, stream=_stream)
    request = _execute_request("exec-late", stream_journal_enabled=True)
    response = await server._handle_execute(request)

    assert response.status == "completed"
    assert callbacks == [{"status": "COMPLETED", "result": "late"}]


# ---------------------------------------------------------------------------
# Part D — the runner journals a journal dispatch's chunks
# ---------------------------------------------------------------------------


async def test_journal_dispatch_records_every_chunk_with_its_generation(
    tmp_path: Path,
) -> None:
    server = _make_server()
    server._stream_journals = StreamJournalRegistry(root=str(tmp_path))
    client = _FakeStreamClient(default=200)
    _install_client(server, client)
    server._send_callback_body = AsyncMock()

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        for token in ("a", "b"):
            await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content=token)
        return StreamingResult(content="ab", messages=[])

    _wire_execute_path(server, stream=_stream)
    request = _execute_request(
        "exec-journal-record", stream_journal_enabled=True, stream_generation=7
    )
    response = await server._handle_execute(request)

    assert response.status == "completed"
    # The writer settled inside the execution's finally, so the file is final.
    path = tmp_path / sj.journal_file_name("exec-journal-record", 7)
    records, end, torn = sj._read_records(path)
    assert torn is False
    # Both TEXT chunks and the terminal DONE frame: the journal is a complete
    # copy of what the engine was sent.
    assert [record["seq"] for record in records] == [1, 2, 3]
    assert [record["generation"] for record in records] == [7, 7, 7]
    assert [record["chunk"]["content"] for record in records] == ["a", "b", "ab"]
    # Whatever live delivery managed to hand the engine also carries the
    # generation, and journal mode keeps the legacy raw chunk shape the
    # unnegotiated engine parses (journal dispatch makes live delivery best
    # effort, so the last posts may not have drained — that is the point).
    for body in client.bodies:
        assert "chunks" not in body, "journal mode must keep the legacy raw shape"
        assert body.get("generation") == 7
    assert end is not None
    assert end["reason"] == REASON_COMPLETED
    # The journal is no longer registered as in flight.
    assert server.stream_journal_writers == {}


async def test_journal_dispatch_closes_its_journal_as_suspended(tmp_path: Path) -> None:
    server = _make_server()
    server._stream_journals = StreamJournalRegistry(root=str(tmp_path))
    client = _FakeStreamClient(default=200)
    _install_client(server, client)
    server._send_callback_body = AsyncMock()

    async def _suspend(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="x")
        return InterruptResult(suspend_payload={"suspend_reason": "approval"})

    _wire_execute_path(server, stream=_suspend)
    request = _execute_request(
        "exec-journal-suspend-reason", stream_journal_enabled=True, stream_generation=2
    )
    response = await server._handle_execute(request)

    assert response.status == "suspended"
    path = tmp_path / sj.journal_file_name("exec-journal-suspend-reason", 2)
    _records, end, _torn = sj._read_records(path)
    assert end is not None
    assert end["reason"] == REASON_SUSPENDED


async def test_journal_flag_with_zero_generation_leaves_the_chunk_path_untouched(
    tmp_path: Path,
) -> None:
    # The generation names the journal body; without one there is nothing to
    # key it by, so the dispatch must behave exactly like a non-journal one.
    server = _make_server()
    server._stream_journals = StreamJournalRegistry(root=str(tmp_path))
    client = _FakeStreamClient(default=200)
    _install_client(server, client)
    server._send_callback_body = AsyncMock()

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="a")
        return StreamingResult(content="a", messages=[])

    _wire_execute_path(server, stream=_stream)
    request = _execute_request("exec-no-generation", stream_journal_enabled=True)
    await server._handle_execute(request)

    assert server.stream_journal_writers == {}
    assert list(tmp_path.iterdir()) == []
    for body in client.bodies:
        sent = body.get("chunks") if isinstance(body.get("chunks"), list) else [body]
        assert all("generation" not in chunk for chunk in sent)


async def test_a_local_progress_emit_is_journaled_in_order(tmp_path: Path) -> None:
    # A local tool calls the public progress API. Its step must be sequenced and
    # recorded on the turn's journal, or a replay would miss a step the client
    # saw and still call the turn complete.
    server = _make_server()
    server._stream_journals = StreamJournalRegistry(root=str(tmp_path))
    client = _FakeStreamClient(default=200)
    _install_client(server, client)
    server._send_callback_body = AsyncMock()

    async def _stream(_agent: Any, _ctx: Any, _input: Any, oe_url: str, execution_id: str) -> Any:
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="before")
        # The documented pattern for an async tool: the emit arrives on a worker
        # thread, so admission has to be marshalled onto the AER loop.
        await asyncio.to_thread(emit_step, "halfway")
        # The healthy engine sees the live step while the tool is still running.
        await _wait_until(
            lambda: any(
                chunk.get("chunk_type") == "step"
                for body in client.bodies
                for chunk in _sent_chunks(body)
            )
        )
        await server._send_stream_chunk(oe_url, execution_id, chunk_type=TEXT, content="after")
        return StreamingResult(content="beforeafter", messages=[])

    _wire_execute_path(server, stream=_stream)
    request = _execute_request(
        "exec-journal-progress", stream_journal_enabled=True, stream_generation=11
    )
    response = await server._handle_execute(request)

    assert response.status == "completed"
    records, end, torn = sj._read_records(
        tmp_path / sj.journal_file_name("exec-journal-progress", 11)
    )
    assert torn is False
    assert [record["seq"] for record in records] == [1, 2, 3, 4]
    assert [(record["chunk"]["chunk_type"], record["chunk"]["content"]) for record in records] == [
        (TEXT, "before"),
        ("step", "halfway"),
        (TEXT, "after"),
        (DONE, "beforeafter"),
    ]
    assert all(record["generation"] == 11 for record in records)
    assert end is not None
    assert end["reason"] == REASON_COMPLETED
