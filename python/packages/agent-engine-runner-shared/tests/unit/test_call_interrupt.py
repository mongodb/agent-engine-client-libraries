"""Per-call interrupt receiver (POST /interrupt/call).

Mirrors TypeScript's ``tests/unit/server_call_interrupt.test.ts`` — the two
runner runtimes are parallel implementations, so the twins must stay
behaviourally identical. The cross-language request/response vectors live in
``client-libraries/test-fixtures/interrupt-call/contract.json`` and are pinned
by ``test_call_interrupt_contract.py``.

These tests pin the receiver contract:

- an abort signals exactly the named (execution, step) call — sibling calls
  of the same execution stay untouched;
- the execution is never admission-latched: unlike a drain, later calls of
  the same execution proceed (the anti-drain property);
- repeats are idempotent, a settled call answers ``already_settled``, and
  tracked work with no signal channel answers ``not_cancellable`` rather than
  claiming abandonment;
- a call aborted between registration and task attach is cancelled at attach
  time;
- workspace scoping matches the drain receiver's.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Optional
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from agent_engine_runner_shared.models import ToolPodExecuteRequest
from agent_engine_runner_shared.server.drain import CallInterruptOutcome, DrainRegistry
from agent_engine_runner_shared.server.tool import ToolServer


async def _run_forever() -> None:
    await asyncio.Event().wait()


def _body(resp) -> dict:
    return json.loads(bytes(resp.body))


# ---------------------------------------------------------------------------
# Registry: step-scoped abort
# ---------------------------------------------------------------------------


async def test_abort_signals_only_the_named_step() -> None:
    registry = DrainRegistry()
    t3 = asyncio.create_task(_run_forever())
    t4 = asyncio.create_task(_run_forever())
    registry.begin_work("exec-1", t3, step_number=3)
    registry.begin_work("exec-1", t4, step_number=4)

    outcome = registry.abort_call("exec-1", 3)

    assert outcome == CallInterruptOutcome.INTERRUPTED
    await asyncio.sleep(0)
    assert t3.cancelled()
    assert not t4.cancelled()
    t4.cancel()


async def test_abort_never_latches_admission() -> None:
    """The anti-drain property: an aborted execution accepts its next call."""
    registry = DrainRegistry()
    task = asyncio.create_task(_run_forever())
    registry.begin_work("exec-1", task, step_number=3)

    registry.abort_call("exec-1", 3)

    follow_up = asyncio.create_task(_run_forever())
    handle = registry.begin_work("exec-1", follow_up, step_number=4)  # must not raise
    assert handle is not None
    assert not follow_up.cancelled()
    follow_up.cancel()


async def test_abort_unknown_execution_is_not_found() -> None:
    registry = DrainRegistry()
    assert registry.abort_call("exec-x", 3) == CallInterruptOutcome.NOT_FOUND


async def test_abort_unknown_step_is_not_found() -> None:
    registry = DrainRegistry()
    task = asyncio.create_task(_run_forever())
    registry.begin_work("exec-1", task, step_number=3)

    assert registry.abort_call("exec-1", 9) == CallInterruptOutcome.NOT_FOUND
    assert not task.cancelled()
    task.cancel()


async def test_abort_settled_call_is_already_settled() -> None:
    registry = DrainRegistry()
    task = asyncio.create_task(_run_forever())
    handle = registry.begin_work("exec-1", task, step_number=3)
    task.cancel()
    registry.end_work(handle)

    assert registry.abort_call("exec-1", 3) == CallInterruptOutcome.ALREADY_SETTLED


async def test_abort_repeat_is_idempotent() -> None:
    registry = DrainRegistry()
    task = asyncio.create_task(_run_forever())
    registry.begin_work("exec-1", task, step_number=3)

    first = registry.abort_call("exec-1", 3)
    second = registry.abort_call("exec-1", 3)

    assert first == CallInterruptOutcome.INTERRUPTED
    assert second == CallInterruptOutcome.INTERRUPTED
    await asyncio.sleep(0)
    assert task.cancelled()


async def test_abort_tracked_work_without_a_signal_channel_is_honest() -> None:
    """A sync tool off-loaded to a thread registers without a task: the abort
    must not claim an abandonment the runtime cannot perform."""
    registry = DrainRegistry()
    registry.begin_work("exec-1", step_number=3)  # no task: tracked only

    assert registry.abort_call("exec-1", 3) == CallInterruptOutcome.NOT_CANCELLABLE


async def test_abort_repeat_of_track_only_work_stays_not_cancellable() -> None:
    """The missing-signal-channel answer is stable across repeats — a repeat
    must never flip to interrupted (matches the TypeScript receiver)."""
    registry = DrainRegistry()
    registry.begin_work("exec-1", step_number=3)

    assert registry.abort_call("exec-1", 3) == CallInterruptOutcome.NOT_CANCELLABLE
    assert registry.abort_call("exec-1", 3) == CallInterruptOutcome.NOT_CANCELLABLE


async def test_abort_before_attach_cancels_at_attach() -> None:
    """A stream's consuming task exists only once the response body iterates;
    an abort landing in that window must fire when the task attaches."""
    registry = DrainRegistry()
    handle = registry.begin_work("exec-1", expect_attach=True, step_number=3)

    assert registry.abort_call("exec-1", 3) == CallInterruptOutcome.INTERRUPTED

    task = asyncio.create_task(_run_forever())
    assert registry.attach_task(handle, task) is True
    await asyncio.sleep(0)
    assert task.cancelled()


# ---------------------------------------------------------------------------
# Route: POST /interrupt/call
# ---------------------------------------------------------------------------


def _server(monkeypatch: pytest.MonkeyPatch) -> ToolServer:
    monkeypatch.setenv("RUNNER_AUTH_TOKEN", "s3cret")
    monkeypatch.setenv("APP_ID", "ws-1")
    mock_runtime = Mock()
    mock_runtime._tools = {}
    mock_runtime._tool_definitions = {}
    return ToolServer(mock_runtime)


def _post(client: TestClient, workspace_id: Optional[str] = "ws-1", step: int = 3):
    body = {"execution_id": "exec-1", "step_number": step}
    if workspace_id is not None:
        body["workspace_id"] = workspace_id
    return client.post("/interrupt/call", json=body, headers={"Authorization": "Bearer s3cret"})


def test_route_interrupts_the_live_call(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _server(monkeypatch)
    server.drain_registry.begin_work("exec-1", step_number=3)
    client = TestClient(server.create_app())

    resp = _post(client)

    assert resp.status_code == 200
    # Track-only registration (no task): found, but honest about the limit.
    assert resp.json() == {"outcome": "not_cancellable"}


def test_route_unknown_call_is_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _server(monkeypatch)
    client = TestClient(server.create_app())

    resp = _post(client)

    assert resp.status_code == 200
    assert resp.json() == {"outcome": "not_found"}


def test_route_missing_workspace_scope_is_400(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _server(monkeypatch)
    client = TestClient(server.create_app())

    resp = _post(client, workspace_id=None)

    assert resp.status_code == 400


def test_route_workspace_mismatch_is_403(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _server(monkeypatch)
    client = TestClient(server.create_app())

    resp = _post(client, workspace_id="ws-2")

    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Wiring through the real tool-execute path
# ---------------------------------------------------------------------------


def _tool_server(tool) -> ToolServer:
    mock_runtime = Mock()
    mock_runtime._tools = {"tool": tool}
    mock_runtime._tool_definitions = {"tool": {}}
    return ToolServer(mock_runtime)


def _tool_request(step: int) -> ToolPodExecuteRequest:
    return ToolPodExecuteRequest(
        execution_id="exec-1",
        tool_name="tool",
        arguments={},
        session_id="session-1",
        step_number=step,
    )


async def test_execute_registers_the_step_and_the_abort_cancels_the_tool_body() -> None:
    """A tool call dispatched with a step number is aborted by
    (execution, step): the tool body is cancelled, not waited out."""
    cleaned: list[str] = []

    async def cooperative_tool() -> str:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cleaned.append("cleaned up with the provider")
            raise
        return "unreachable"

    server = _tool_server(cooperative_tool)
    task = asyncio.create_task(server._handle_execute(_tool_request(step=3)))
    for _ in range(500):
        entry = server.drain_registry._entries.get("exec-1")
        if entry is not None and entry.active_count > 0:
            break
        await asyncio.sleep(0.005)
    else:
        raise AssertionError("work was never registered")

    assert server.drain_registry.abort_call("exec-1", 3) == CallInterruptOutcome.INTERRUPTED
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned == ["cleaned up with the provider"]


async def test_execute_sync_tool_reports_not_cancellable_and_runs_out() -> None:
    """A sync tool on a thread has no signal channel: the abort answers
    not_cancellable and the body still runs to completion."""
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()

    def blocking_tool() -> str:
        loop.call_soon_threadsafe(started.set)
        release.wait(timeout=5)
        return "external work completed"

    server = _tool_server(blocking_tool)
    task = asyncio.create_task(server._handle_execute(_tool_request(step=3)))
    await started.wait()

    assert server.drain_registry.abort_call("exec-1", 3) == CallInterruptOutcome.NOT_CANCELLABLE

    release.set()
    result = await task
    assert result.status == "success", "the thread body still runs to completion"
