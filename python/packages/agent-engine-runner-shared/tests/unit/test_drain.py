"""Execution-scoped drain receiver (POST /drain).

Mirrors TypeScript's ``tests/unit/server_drain.test.ts`` — the two runner
runtimes are parallel implementations, so the twins must stay behaviourally
identical. The cross-language request/response vectors live in
``client-libraries/test-fixtures/drain/contract.json`` and are pinned by
``test_drain_contract.py``.

These tests pin the receiver contract:

- a drain blocks new work for exactly one execution and cancels its tracked
  asyncio work — including a turn blocked on an outbound interaction, which
  is the AER case the OE admission gate alone cannot unblock;
- the outcome vocabulary distinguishes acceptance from quiescence
  (``accepted`` vs ``completed``), is honest about uncancellable work
  (``timed_out``), and never reports ``completed`` for an execution the
  runtime never served (``delivery_failed``/``execution_not_found``);
- retries are idempotent by request_id, coalesce per execution, and never
  extend the absolute deadline;
- records survive for a bounded TTL so late retries still see the outcome.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Optional
from unittest.mock import Mock

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from agent_engine_runner_shared.models import ToolPodExecuteRequest
from agent_engine_runner_shared.server.aer import AERServer
from agent_engine_runner_shared.server.drain import (
    DEFAULT_MAX_DEADLINE_MS,
    DrainRegistry,
    DrainRequest,
)
from agent_engine_runner_shared.server.tool import ToolServer


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)


def _req(
    execution_id: str = "exec-1",
    request_id: str = "drain-aaaa",
    deadline_offset_ms: int = 5000,
    reason: str = "execution_cancelled",
    workspace_id: Optional[str] = None,
) -> DrainRequest:
    return DrainRequest(
        request_id=request_id,
        execution_id=execution_id,
        reason=reason,
        deadline_at_ms=_now_ms() + deadline_offset_ms,
        workspace_id=workspace_id,
    )


def _body(resp) -> dict:
    return json.loads(bytes(resp.body))


async def _settled(registry: DrainRegistry, request: DrainRequest) -> dict:
    """Re-send the drain request until a final outcome is recorded — the same
    polling the caller performs, instead of a timing-sensitive sleep."""
    for _ in range(200):
        body = _body(registry.apply(request))
        if body["outcome"] != "accepted":
            return body
        await asyncio.sleep(0.01)
    raise AssertionError("drain did not finalize")


async def _wait_for_active(registry: DrainRegistry, execution_id: str) -> None:
    """Poll until work for the execution is registered — deterministic,
    unlike a fixed sleep that a slow dispatch can outlast."""
    for _ in range(500):
        entry = registry._entries.get(execution_id)
        if entry is not None and entry.active_count > 0:
            return
        await asyncio.sleep(0.005)
    raise AssertionError("work was never registered")


# ---------------------------------------------------------------------------
# Registry state machine
# ---------------------------------------------------------------------------


async def test_unknown_execution_is_delivery_failed() -> None:
    registry = DrainRegistry()
    resp = registry.apply(_req())
    assert resp.status_code == 200
    assert _body(resp) == {
        "outcome": "delivery_failed",
        "reason_code": "execution_not_found",
    }


async def test_restart_loses_all_drain_state() -> None:
    """A restarted runtime must not synthesize completed: its work is gone,
    but loss of the registry is not evidence the drain succeeded — the
    caller's durable record reconciles."""
    first = DrainRegistry()
    first.begin_work("exec-1")
    assert first.apply(_req()).status_code == 202

    restarted = DrainRegistry()
    resp = restarted.apply(_req())
    assert _body(resp)["outcome"] == "delivery_failed"


async def test_known_ended_execution_completes() -> None:
    registry = DrainRegistry()
    registry.end_work(registry.begin_work("exec-1"))

    resp = registry.apply(_req())
    assert resp.status_code == 200
    assert _body(resp) == {"outcome": "completed"}


async def test_active_execution_accepts_then_completes() -> None:
    registry = DrainRegistry()
    handle = registry.begin_work("exec-1")

    resp = registry.apply(_req())
    assert resp.status_code == 202
    assert _body(resp) == {"outcome": "accepted"}

    registry.end_work(handle)

    assert await _settled(registry, _req()) == {"outcome": "completed"}


async def test_drain_cancels_tracked_task_blocked_on_an_await() -> None:
    """The AER case: a turn parked on an outbound interaction (an OE callback
    that will never answer, an LLM stream) is an await on this task, so the
    drain reaches it — the OE admission gate alone cannot."""
    registry = DrainRegistry()
    reached_cancel: list[str] = []

    async def blocked_on_interaction() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            reached_cancel.append("cancelled")
            raise

    task = asyncio.create_task(blocked_on_interaction())
    handle = registry.begin_work("exec-1", task)
    await asyncio.sleep(0)

    assert registry.apply(_req()).status_code == 202
    with pytest.raises(asyncio.CancelledError):
        await task
    assert reached_cancel == ["cancelled"]

    registry.end_work(handle)
    assert await _settled(registry, _req()) == {"outcome": "completed"}


async def test_uncancellable_work_times_out_and_stays_timed_out() -> None:
    """Sync tools run on a thread the runtime cannot signal: the drain waits
    for the deadline and reports timed_out, and the thread finishing late
    does not flip the recorded outcome."""
    registry = DrainRegistry()
    handle = registry.begin_work("exec-1")  # no task: nothing to cancel

    assert registry.apply(_req(deadline_offset_ms=50)).status_code == 202
    await asyncio.sleep(0.15)

    resp = registry.apply(_req())
    assert _body(resp) == {
        "outcome": "timed_out",
        "reason_code": "deadline_exceeded",
    }

    registry.end_work(handle)  # the thread finishes after the deadline
    await asyncio.sleep(0)
    assert _body(registry.apply(_req()))["outcome"] == "timed_out"


async def test_duplicate_request_id_never_reruns_side_effects() -> None:
    registry = DrainRegistry()
    task = Mock(spec=asyncio.Task)
    registry.begin_work("exec-1", task)

    assert registry.apply(_req()).status_code == 202
    assert registry.apply(_req()).status_code == 202
    assert task.cancel.call_count == 1


async def test_different_request_ids_coalesce_into_one_drain() -> None:
    registry = DrainRegistry()
    task = Mock(spec=asyncio.Task)
    handle = registry.begin_work("exec-1", task)

    assert registry.apply(_req(request_id="drain-aaaa")).status_code == 202
    assert registry.apply(_req(request_id="drain-bbbb")).status_code == 202
    assert task.cancel.call_count == 1, "a second request_id must not re-notify"

    registry.end_work(handle)
    for request_id in ("drain-aaaa", "drain-bbbb"):
        assert await _settled(registry, _req(request_id=request_id)) == {"outcome": "completed"}


async def test_retry_never_extends_the_absolute_deadline() -> None:
    registry = DrainRegistry()
    registry.begin_work("exec-1")

    assert registry.apply(_req(deadline_offset_ms=50)).status_code == 202
    # A retry (new request_id) carrying a far-later deadline must coalesce
    # without moving the original deadline.
    assert (
        registry.apply(_req(request_id="drain-bbbb", deadline_offset_ms=60_000)).status_code == 202
    )

    await asyncio.sleep(0.15)
    assert _body(registry.apply(_req()))["outcome"] == "timed_out"


async def test_sibling_execution_is_unaffected() -> None:
    registry = DrainRegistry()
    sibling_task = asyncio.create_task(asyncio.sleep(3600))
    drained_task = asyncio.create_task(asyncio.sleep(3600))
    registry.begin_work("exec-A", sibling_task)
    registry.begin_work("exec-B", drained_task)
    await asyncio.sleep(0)

    assert registry.apply(_req(execution_id="exec-B")).status_code == 202

    assert not sibling_task.cancelled(), "draining exec-B must not touch exec-A"
    with pytest.raises(asyncio.CancelledError):
        await drained_task
    registry.end_work(registry.begin_work("exec-A"))  # exec-A still admits work
    with pytest.raises(HTTPException) as excinfo:
        registry.begin_work("exec-B")
    assert excinfo.value.status_code == 409

    sibling_task.cancel()
    await asyncio.gather(sibling_task, return_exceptions=True)


async def test_unknown_execution_drain_latches_admission_and_stays_stable() -> None:
    """The drain-before-dispatch race: a dispatch that slips past the OE gate
    and lands after the drain must not start, and a retry of the same request
    must see the same (not-found) answer for the retention window."""
    registry = DrainRegistry()
    assert _body(registry.apply(_req()))["outcome"] == "delivery_failed"

    with pytest.raises(HTTPException) as excinfo:
        registry.begin_work("exec-1")
    assert excinfo.value.status_code == 409

    assert _body(registry.apply(_req()))["outcome"] == "delivery_failed"


async def test_tombstone_drain_latches_admission() -> None:
    """A drain finalizing on an ended execution still latches: a resume
    reusing the execution id must not run while retries report completed."""
    registry = DrainRegistry()
    registry.end_work(registry.begin_work("exec-1"))
    assert _body(registry.apply(_req()))["outcome"] == "completed"

    with pytest.raises(HTTPException) as excinfo:
        registry.begin_work("exec-1")
    assert excinfo.value.status_code == 409


async def test_tombstone_drain_record_lives_a_full_ttl_from_finalization() -> None:
    """A drain landing near tombstone expiry restarts retention: the record
    must survive the full TTL from finalization, not the tombstone's original
    eviction time."""
    registry = DrainRegistry(record_ttl_s=0.4)
    handle = registry.begin_work("exec-1")
    registry.end_work(handle)  # tombstone eviction scheduled at t+0.4
    await asyncio.sleep(0.3)

    assert _body(registry.apply(_req()))["outcome"] == "completed"
    await asyncio.sleep(0.2)  # t=0.5: past the original eviction, within the restarted TTL
    assert _body(registry.apply(_req()))["outcome"] == "completed"
    await asyncio.sleep(0.3)  # t=0.8: past the restarted TTL
    assert _body(registry.apply(_req()))["outcome"] == "delivery_failed"


async def test_attach_task_cancels_when_the_drain_landed_first() -> None:
    """The SSE window: work is registered when the route returns, but the
    consuming task exists only once iteration starts. A drain in between must
    reach it at attach time."""
    registry = DrainRegistry()
    # route-time registration, task not yet known
    handle = registry.begin_work("exec-1", expect_attach=True)
    assert registry.apply(_req()).status_code == 202

    cancelled = asyncio.Event()

    async def consuming_stream() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(consuming_stream())
    await asyncio.sleep(0)
    registry.attach_task(handle, task)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


async def test_end_work_drops_tracked_handles() -> None:
    """A tombstone must not retain cancelled tasks (and their listeners) for
    its TTL."""
    registry = DrainRegistry()
    task = asyncio.create_task(asyncio.sleep(3600))
    handle = registry.begin_work("exec-1", task)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    registry.end_work(handle)
    entry = registry._entries["exec-1"]
    assert not entry.tasks


async def test_unattached_stream_registration_releases_after_grace() -> None:
    """A client that disconnects before the stream body ever iterates leaves
    no task to attach; the attach grace releases the registration so a later
    drain completes instead of waiting out its deadline on phantom work."""
    registry = DrainRegistry(attach_grace_s=0.05)
    registry.begin_work("exec-1", expect_attach=True)

    await asyncio.sleep(0.15)
    assert _body(registry.apply(_req()))["outcome"] == "completed"


async def test_attach_task_rejects_a_released_handle() -> None:
    """The other half of the grace contract: a consumer that shows up only
    after its registration was released must not start — a drain may already
    have reported completed, and untracked provider work must never coexist
    with that."""
    registry = DrainRegistry(attach_grace_s=0.05)
    handle = registry.begin_work("exec-1", expect_attach=True)
    await asyncio.sleep(0.15)

    task = asyncio.create_task(asyncio.sleep(3600))
    assert registry.attach_task(handle, task) is False
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    # The released work stayed released: a drain completes immediately
    # instead of waiting on work it cannot see.
    assert _body(registry.apply(_req()))["outcome"] == "completed"


async def test_end_work_is_idempotent() -> None:
    """Overlapping cleanup paths (stream generator, response background,
    shutdown) must not double-decrement the active count."""
    registry = DrainRegistry()
    handle = registry.begin_work("exec-1")
    registry.end_work(handle)
    registry.end_work(handle)

    # The count was not driven negative: new work registers, drains,
    # completes.
    second = registry.begin_work("exec-1")
    assert registry.apply(_req()).status_code == 202
    registry.end_work(second)
    assert await _settled(registry, _req()) == {"outcome": "completed"}


async def test_record_outlives_completion_for_the_ttl_then_expires() -> None:
    registry = DrainRegistry(record_ttl_s=0.05)
    registry.end_work(registry.begin_work("exec-1"))

    assert _body(registry.apply(_req()))["outcome"] == "completed"
    await asyncio.sleep(0.1)  # past the TTL: the eviction timer has run

    resp = registry.apply(_req())
    assert _body(resp)["outcome"] == "delivery_failed", (
        "after the record TTL a retry is treated as unknown and must not "
        "re-run side effects; the durable caller record owns the history"
    )


# ---------------------------------------------------------------------------
# Wiring through the real servers
# ---------------------------------------------------------------------------


def _tool_server(tool) -> ToolServer:
    mock_runtime = Mock()
    mock_runtime._tools = {"tool": tool}
    mock_runtime._tool_definitions = {"tool": {}}
    return ToolServer(mock_runtime)


def _tool_request() -> ToolPodExecuteRequest:
    return ToolPodExecuteRequest(
        execution_id="exec-1",
        tool_name="tool",
        arguments={},
        session_id="session-1",
    )


async def test_async_tool_call_is_cancelled_by_drain() -> None:
    """Flips the boundary pinned by test_tool_cancellation_boundary.py: a
    cooperating async tool now does get asked."""
    cleaned: list[str] = []

    async def cooperative_tool() -> str:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cleaned.append("cleaned up with the provider")
            raise
        return "unreachable"

    server = _tool_server(cooperative_tool)
    task = asyncio.create_task(server._handle_execute(_tool_request()))
    await _wait_for_active(server.drain_registry, "exec-1")

    assert server.drain_registry.apply(_req()).status_code == 202
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned == ["cleaned up with the provider"]
    assert await _settled(server.drain_registry, _req()) == {"outcome": "completed"}


async def test_sync_tool_call_times_out_honestly() -> None:
    """A sync tool on a thread cannot be signalled: the drain reports
    timed_out, the body still runs to completion, and the late finish does
    not change the recorded outcome."""
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()

    def blocking_tool() -> str:
        loop.call_soon_threadsafe(started.set)
        release.wait(timeout=5)
        return "external work completed"

    server = _tool_server(blocking_tool)
    task = asyncio.create_task(server._handle_execute(_tool_request()))
    await started.wait()

    assert server.drain_registry.apply(_req(deadline_offset_ms=50)).status_code == 202
    await asyncio.sleep(0.15)
    assert _body(server.drain_registry.apply(_req()))["outcome"] == "timed_out"

    release.set()
    result = await task
    assert result.status == "success", "the thread body still runs to completion"
    assert _body(server.drain_registry.apply(_req()))["outcome"] == "timed_out"


class _HangingAgent:
    """An agent whose turn parks on an await and never produces an event —
    the shape of a turn blocked inside an OE interaction."""

    def execute(self, ctx, agent_input):  # noqa: ANN001, ANN202 - fake SDK surface
        async def _stream():
            await asyncio.sleep(3600)
            yield  # pragma: no cover - unreachable by construction

        return _stream()


async def test_aer_turn_blocked_mid_execution_is_stopped_by_drain() -> None:
    runtime = Mock()
    runtime.get_agent.return_value = _HangingAgent()
    server = AERServer(runtime)
    app = server.create_app()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://aer.test"
    ) as client:
        execute_task = asyncio.create_task(
            client.post(
                "/execute",
                json={"execution_id": "exec-1", "platform_api_url": "http://oe.invalid"},
            )
        )
        # Deterministic: wait until the turn has registered its work rather
        # than racing a fixed sleep against a slow dispatch.
        await _wait_for_active(server.drain_registry, "exec-1")

        resp = await client.post(
            "/drain",
            json={
                "request_id": "drain-aaaa",
                "execution_id": "exec-1",
                "reason": "execution_cancelled",
                "deadline_at_ms": _now_ms() + 2000,
            },
        )
        assert resp.status_code == 202

        # The drained turn answers the held /execute cleanly: the OE's cancel
        # path owns the terminal Cancelled settlement, and a failed ack would
        # race it into an ERROR outcome for a run the operator cancelled.
        execute_resp = await asyncio.wait_for(execute_task, timeout=5)
        assert execute_resp.status_code == 200
        assert execute_resp.json()["status"] == "cancelled"

        drain_body = {
            "request_id": "drain-aaaa",
            "execution_id": "exec-1",
            "reason": "execution_cancelled",
            "deadline_at_ms": _now_ms() + 2000,
        }
        final = None
        for _ in range(200):
            final = await client.post("/drain", json=drain_body)
            if final.json()["outcome"] != "accepted":
                break
            await asyncio.sleep(0.01)
        assert final is not None
        assert final.status_code == 200
        assert final.json()["outcome"] == "completed"

        # The drained execution no longer admits work.
        replay = await client.post(
            "/execute",
            json={"execution_id": "exec-1", "platform_api_url": "http://oe.invalid"},
        )
        assert replay.status_code == 409


# ---------------------------------------------------------------------------
# Route-level validation, auth, and scope
# ---------------------------------------------------------------------------


def _drain_app(monkeypatch: pytest.MonkeyPatch, env: Optional[dict] = None):
    merged = {"RUNNER_AUTH_TOKEN": "s3cret", "APP_ID": "ws-1", **(env or {})}
    for key, value in merged.items():
        monkeypatch.setenv(key, value)
    server = _tool_server(lambda: "ok")
    return TestClient(server.create_app())


def _post_drain(client: TestClient, **overrides):
    body = {
        "request_id": "drain-aaaa",
        "execution_id": "exec-1",
        "reason": "execution_cancelled",
        "deadline_at_ms": _now_ms() + 5000,
        "workspace_id": "ws-1",
        **overrides,
    }
    # None means "omit the key entirely", not JSON null — the omission is
    # what the scoped/unscoped contract pins.
    if body["workspace_id"] is None:
        del body["workspace_id"]
    return client.post("/drain", json=body, headers={"Authorization": "Bearer s3cret"})


def test_drain_requires_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _drain_app(monkeypatch)
    assert client.post("/drain", json={}).status_code == 401
    assert (
        client.post("/drain", json={}, headers={"Authorization": "Bearer wrong"}).status_code == 401
    )


def test_drain_unknown_execution_via_http(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = _post_drain(_drain_app(monkeypatch))
    assert resp.status_code == 200
    assert resp.json() == {
        "outcome": "delivery_failed",
        "reason_code": "execution_not_found",
    }


def test_drain_validation_failures_are_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _drain_app(monkeypatch)
    assert _post_drain(client, reason="bogus").status_code == 422
    assert _post_drain(client, request_id="").status_code == 422
    assert _post_drain(client, request_id="bad id!").status_code == 422
    assert _post_drain(client, deadline_at_ms=0).status_code == 422
    assert _post_drain(client, execution_id="").status_code == 422


def test_drain_rejects_deadline_beyond_max(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _drain_app(monkeypatch)
    resp = _post_drain(client, deadline_at_ms=_now_ms() + DEFAULT_MAX_DEADLINE_MS + 60_000)
    assert resp.status_code == 400


def test_drain_rejects_workspace_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _drain_app(monkeypatch)
    assert _post_drain(client, workspace_id="someone-else").status_code == 403
    assert _post_drain(client, workspace_id="ws-1").status_code != 403


def test_drain_requires_workspace_id_on_a_scoped_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bearer auth proves the caller is the OE; it does not prove the named
    drain belongs to this runtime. With APP_ID set, the scope claim is
    mandatory — and JSON null is a missing claim, not a wildcard."""
    client = _drain_app(monkeypatch)
    assert _post_drain(client, workspace_id=None).status_code == 400  # key omitted
    resp = client.post(
        "/drain",
        json={
            "request_id": "drain-aaaa",
            "execution_id": "exec-1",
            "reason": "execution_cancelled",
            "deadline_at_ms": _now_ms() + 5000,
            "workspace_id": None,  # explicit null
        },
        headers={"Authorization": "Bearer s3cret"},
    )
    assert resp.status_code == 400


def test_drain_skips_workspace_check_when_unscoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local development runs a single unscoped runtime (no APP_ID): there is
    nothing to check against, so a request that omits workspace_id entirely
    is accepted."""
    client = _drain_app(monkeypatch, env={"APP_ID": ""})
    resp = _post_drain(client, workspace_id=None)
    assert resp.status_code == 200
    assert resp.json()["outcome"] == "delivery_failed"


def test_drain_rejects_invalid_tunables() -> None:
    """A zero/negative/non-finite TTL, deadline ceiling, or attach grace
    silently collapses a drain guarantee; malformed values fail fast instead.
    float() parses "nan"/"inf" without error, so they are tested explicitly."""
    with pytest.raises(ValueError):
        DrainRegistry(record_ttl_s=0)
    with pytest.raises(ValueError):
        DrainRegistry(max_deadline_ms=-1)
    with pytest.raises(ValueError):
        DrainRegistry(attach_grace_s=0)
    with pytest.raises(ValueError):
        DrainRegistry(record_ttl_s=float("nan"))
    with pytest.raises(ValueError):
        DrainRegistry(attach_grace_s=float("inf"))
    with pytest.raises(ValueError):
        DrainRegistry(env={"RUNNER_DRAIN_RECORD_TTL_S": "-5"})
    with pytest.raises(ValueError):
        DrainRegistry(env={"RUNNER_DRAIN_MAX_DEADLINE_MS": "bogus"})
    with pytest.raises(ValueError):
        DrainRegistry(env={"RUNNER_DRAIN_RECORD_TTL_S": "nan"})
    with pytest.raises(ValueError):
        DrainRegistry(env={"RUNNER_DRAIN_ATTACH_GRACE_S": "inf"})


def test_stream_dispatch_landing_after_drain_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The drain-before-dispatch race on the SSE route: the drain records and
    latches the (here unknown) execution, so a stream dispatched just ahead of
    it is refused at admission instead of starting."""
    monkeypatch.setenv("RUNNER_AUTH_TOKEN", "s3cret")
    server = _tool_server(lambda: "ok")
    client = TestClient(server.create_app())
    auth = {"Authorization": "Bearer s3cret"}

    resp = client.post(
        "/drain",
        json={
            "request_id": "drain-aaaa",
            "execution_id": "exec-1",
            "reason": "execution_cancelled",
            "deadline_at_ms": _now_ms() + 5000,
        },
        headers=auth,
    )
    assert resp.json()["outcome"] == "delivery_failed"

    stream = client.post(
        "/invoke_llm/stream",
        json={
            "execution_id": "exec-1",
            "arguments": {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        },
        headers=auth,
    )
    assert stream.status_code == 409
