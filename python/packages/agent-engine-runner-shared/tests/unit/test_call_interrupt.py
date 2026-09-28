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
import time as _time
from typing import Any, Dict, Optional
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import agent_engine_runner_shared.secure_wrapper as _secure_wrapper
from agent_engine_runner_shared.models import ToolPodExecuteRequest
from agent_engine_runner_shared.secure_wrapper import (
    _CALL_INTERRUPTED,
    SecureToolWrapper,
    _invoke_registered_tool,
)
from agent_engine_runner_shared.server.drain import (
    CallInterruptOutcome,
    DrainOutcome,
    DrainRegistry,
)
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


# ---------------------------------------------------------------------------
# AER local (callback-routed) calls: the in-graph tool boundary
# ---------------------------------------------------------------------------


class _AsyncToolStub:
    """Minimal StructuredTool shape for the async invocation path."""

    func = None

    def __init__(self, body):
        self._body = body

    def invoke(self, kwargs):  # pragma: no cover - the async path must be taken
        raise AssertionError("sync path must not run for an async tool")

    async def ainvoke(self, kwargs):
        return await self._body()


class _SyncToolStub:
    func = True

    def __init__(self, body):
        self._body = body

    def invoke(self, kwargs):
        return self._body()


def _make_wrapper() -> tuple:
    registry = DrainRegistry()
    wrapper = SecureToolWrapper(oe_url="http://oe", execution_id="exec-1", call_registry=registry)
    return wrapper, registry


def _drive_local_tool(wrapper, tool, box: Dict[str, Any], step: int = 2) -> threading.Thread:
    """Run the callback-routed boundary on a worker thread, the way the
    framework's executor runs it in an AER."""

    def run() -> None:
        try:
            box["result"] = wrapper._execute_local_tool(
                tool_name="tool",
                step=step,
                local_executor=lambda: _invoke_registered_tool(tool, {}),
                tool_call_id=None,
                metadata=None,
                is_framework_control_flow=None,
            )
        except BaseException as exc:  # noqa: BLE001 - the test asserts on it
            box["error"] = exc

    t = threading.Thread(target=run)
    t.start()
    return t


def _await_channel(registry: DrainRegistry, step: int):
    """Wait until the tool body has registered its abort channel."""
    for _ in range(1000):
        entry = registry._entries.get("exec-1")
        handle = entry.by_step.get(step) if entry is not None else None
        if handle is not None and handle.cancel_callback is not None:
            return handle
        _time.sleep(0.005)
    raise AssertionError("the tool body never registered an abort channel")


async def _await_channel_async(registry: DrainRegistry, step: int):
    """_await_channel for loop-bound tests: yields instead of blocking, since
    the worker thread's marshalled registry calls need this loop to run."""
    for _ in range(1000):
        entry = registry._entries.get("exec-1")
        handle = entry.by_step.get(step) if entry is not None else None
        if handle is not None and handle.cancel_callback is not None:
            return handle
        await asyncio.sleep(0.005)
    raise AssertionError("the tool body never registered an abort channel")


def test_local_call_abort_cancels_the_body_and_continues_the_graph(monkeypatch) -> None:
    """A per-call abort of a callback-routed async tool stops the body
    mid-flight, reports the step interrupted, and hands the graph the
    stopped-call sentinel instead of unwinding the run."""
    cleaned: list[str] = []
    reports: list[dict] = []
    monkeypatch.setattr(_secure_wrapper, "report_oe_result", lambda **kw: reports.append(kw))

    async def slow_body() -> str:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cleaned.append("cleaned")
            raise
        return "unreachable"

    wrapper, registry = _make_wrapper()
    box: Dict[str, Any] = {}
    t = _drive_local_tool(wrapper, _AsyncToolStub(slow_body), box)

    _await_channel(registry, 2)
    assert registry.abort_call("exec-1", 2) == CallInterruptOutcome.INTERRUPTED

    t.join(timeout=5)
    assert not t.is_alive()
    assert box.get("result") is _CALL_INTERRUPTED, box
    assert cleaned == ["cleaned"]
    assert len(reports) == 1 and reports[0]["status"] == "interrupted", reports


def test_local_call_sync_tool_is_not_cancellable_and_runs_out(monkeypatch) -> None:
    """A sync body on the worker thread has no signal channel: the abort is
    honest, and the body still runs to completion."""
    reports: list[dict] = []
    monkeypatch.setattr(_secure_wrapper, "report_oe_result", lambda **kw: reports.append(kw))
    started = threading.Event()
    release = threading.Event()

    def blocking_body() -> str:
        started.set()
        release.wait(timeout=10)
        return "external work completed"

    wrapper, registry = _make_wrapper()
    box: Dict[str, Any] = {}
    t = _drive_local_tool(wrapper, _SyncToolStub(blocking_body), box)

    assert started.wait(timeout=5)
    for _ in range(1000):
        entry = registry._entries.get("exec-1")
        if entry is not None and entry.by_step.get(2) is not None:
            break
        _time.sleep(0.005)
    assert registry.abort_call("exec-1", 2) == CallInterruptOutcome.NOT_CANCELLABLE

    release.set()
    t.join(timeout=5)
    assert box.get("result") == "external work completed", box
    assert len(reports) == 1 and reports[0]["status"] == "success", reports


def test_local_call_abort_before_channel_attach_fires_at_attach() -> None:
    """The attach race for a declared async body: an abort that lands before
    the channel attaches is sticky-interrupted and fires at attach."""
    registry = DrainRegistry()
    handle = registry.begin_work("exec-1", None, step_number=2, channel_expected=True)
    fired: list[bool] = []

    assert registry.abort_call("exec-1", 2) == CallInterruptOutcome.INTERRUPTED

    assert registry.attach_cancel_callback(handle, lambda: fired.append(True))
    assert fired == [True], "the pending abort must fire at attach"
    registry.end_work(handle)


def test_local_call_refused_abort_never_fires_at_attach() -> None:
    """A refused abort (no channel expected — a sync body) must stay refused:
    no sticky bit, no cancellation at a later attach, no masked outcome."""
    registry = DrainRegistry()
    handle = registry.begin_work("exec-1", None, step_number=2)
    fired: list[bool] = []

    assert registry.abort_call("exec-1", 2) == CallInterruptOutcome.NOT_CANCELLABLE
    assert not handle.aborted

    assert registry.attach_cancel_callback(handle, lambda: fired.append(True))
    assert fired == []
    registry.end_work(handle)


def test_local_call_abort_precedes_registration_skips_the_body(monkeypatch) -> None:
    """The OE-claim → in-graph-dispatch handoff: a Stop can reach the runtime
    before the call registers. The retained abort settles the late-registering
    call as interrupted without running its body."""
    ran: list[bool] = []
    reports: list[dict] = []
    monkeypatch.setattr(_secure_wrapper, "report_oe_result", lambda **kw: reports.append(kw))

    async def slow_body() -> str:
        ran.append(True)
        return "unreachable"

    wrapper, registry = _make_wrapper()
    assert registry.abort_call("exec-1", 2) == CallInterruptOutcome.NOT_FOUND

    box: Dict[str, Any] = {}
    t = _drive_local_tool(wrapper, _AsyncToolStub(slow_body), box)
    t.join(timeout=5)

    assert not t.is_alive()
    assert ran == [], "a call stopped before registration must not run"
    assert box.get("result") is _CALL_INTERRUPTED, box
    assert len(reports) == 1 and reports[0]["status"] == "interrupted", reports


def test_local_call_refused_abort_lets_a_later_error_flow(monkeypatch) -> None:
    """A sync body that fails after a refused abort reports the error to the
    graph — the unhonored abort must not mask it as an interrupt."""
    reports: list[dict] = []
    monkeypatch.setattr(_secure_wrapper, "report_oe_result", lambda **kw: reports.append(kw))
    started = threading.Event()
    release = threading.Event()

    def failing_body() -> str:
        started.set()
        release.wait(timeout=10)
        raise ValueError("provider exploded")

    wrapper, registry = _make_wrapper()
    box: Dict[str, Any] = {}
    t = _drive_local_tool(wrapper, _SyncToolStub(failing_body), box)
    assert started.wait(timeout=5)

    assert registry.abort_call("exec-1", 2) == CallInterruptOutcome.NOT_CANCELLABLE

    release.set()
    t.join(timeout=5)
    assert isinstance(box.get("error"), ValueError), box
    assert len(reports) == 1 and reports[0]["status"] == "error", reports


def test_local_call_late_abort_does_not_repaint_a_genuine_error(monkeypatch) -> None:
    """A Stop landing while the error settlement report is blocked finds the
    settlement already claimed — already_settled, not a stop the durable
    record would contradict — and must not repaint the already-classified
    genuine error as a stopped call."""
    report_started = threading.Event()
    release_report = threading.Event()
    reports: list[dict] = []

    def blocking_report(**kw) -> None:
        reports.append(kw)
        report_started.set()
        release_report.wait(timeout=10)

    monkeypatch.setattr(_secure_wrapper, "report_oe_result", blocking_report)

    async def failing_body() -> str:
        raise ValueError("provider exploded")

    wrapper, registry = _make_wrapper()
    box: Dict[str, Any] = {}
    t = _drive_local_tool(wrapper, _AsyncToolStub(failing_body), box)

    assert report_started.wait(timeout=5)
    assert registry.abort_call("exec-1", 2) == CallInterruptOutcome.ALREADY_SETTLED

    release_report.set()
    t.join(timeout=5)
    assert isinstance(box.get("error"), ValueError), box
    assert len(reports) == 1 and reports[0]["status"] == "error", reports


def test_local_call_abort_during_result_report_reads_already_settled(monkeypatch) -> None:
    """A Stop landing while the success report is still in flight finds the
    call's settlement already claimed: already_settled, no cancel fires, and
    the real success reaches OE and the graph undisturbed."""
    report_started = threading.Event()
    release_report = threading.Event()
    reports: list[dict] = []

    def blocking_report(**kw) -> None:
        reports.append(kw)
        report_started.set()
        release_report.wait(timeout=10)

    monkeypatch.setattr(_secure_wrapper, "report_oe_result", blocking_report)

    async def quick_body() -> str:
        return "done"

    wrapper, registry = _make_wrapper()
    box: Dict[str, Any] = {}
    t = _drive_local_tool(wrapper, _AsyncToolStub(quick_body), box)

    assert report_started.wait(timeout=5)
    assert registry.abort_call("exec-1", 2) == CallInterruptOutcome.ALREADY_SETTLED

    release_report.set()
    t.join(timeout=5)
    assert box.get("result") == "done", box
    assert len(reports) == 1 and reports[0]["status"] == "success", reports


def test_local_call_abort_first_cleanup_error_reports_interrupted(monkeypatch) -> None:
    """A Stop whose cancellation cleanup raises an ordinary exception (a
    provider translating CancelledError into its own error type) settles
    coherently: the runtime honored the stop, so OE records interrupted and
    the graph gets the stopped sentinel — the trace must not contradict the
    verdict the caller already got."""
    reports: list[dict] = []
    monkeypatch.setattr(_secure_wrapper, "report_oe_result", lambda **kw: reports.append(kw))

    async def cleanup_failing_body() -> str:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError as exc:
            raise ValueError("provider cleanup exploded") from exc
        return "unreachable"

    wrapper, registry = _make_wrapper()
    box: Dict[str, Any] = {}
    t = _drive_local_tool(wrapper, _AsyncToolStub(cleanup_failing_body), box)

    _await_channel(registry, 2)
    assert registry.abort_call("exec-1", 2) == CallInterruptOutcome.INTERRUPTED
    t.join(timeout=5)

    assert len(reports) == 1, reports
    assert reports[0]["status"] == "interrupted", reports
    assert reports[0]["error"] == "CancelledError: This call was stopped before completing."
    assert reports[0]["tool_api_error"] is None
    assert box.get("result") is _CALL_INTERRUPTED, box


def test_local_call_abort_cancellation_carries_the_stopped_message(monkeypatch) -> None:
    """The cancelled body sees the stopped-call message, so a trace's
    cancellation row reads like the TypeScript twin's AbortError."""
    messages: list[str] = []
    monkeypatch.setattr(_secure_wrapper, "report_oe_result", lambda **kw: None)

    async def slow_body() -> str:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError as exc:
            messages.append(str(exc))
            raise
        return "unreachable"

    wrapper, registry = _make_wrapper()
    box: Dict[str, Any] = {}
    t = _drive_local_tool(wrapper, _AsyncToolStub(slow_body), box)

    _await_channel(registry, 2)
    assert registry.abort_call("exec-1", 2) == CallInterruptOutcome.INTERRUPTED
    t.join(timeout=5)

    assert messages == ["This call was stopped before completing."], messages


def test_local_call_abort_after_settle_reads_already_settled() -> None:
    registry = DrainRegistry()
    handle = registry.begin_work("exec-1", None, step_number=2)
    fired: list[bool] = []
    assert registry.attach_cancel_callback(handle, lambda: fired.append(True))
    registry.end_work(handle)

    assert registry.abort_call("exec-1", 2) == CallInterruptOutcome.ALREADY_SETTLED
    assert fired == []


def test_claim_settlement_then_abort_reads_already_settled() -> None:
    """Once the result report begins, a late abort is observed, not honored:
    the cancel channel never fires."""
    registry = DrainRegistry()
    handle = registry.begin_work("exec-1", None, step_number=2)
    fired: list[bool] = []
    assert registry.attach_cancel_callback(handle, lambda: fired.append(True))

    assert registry.claim_settlement(handle)
    assert registry.abort_call("exec-1", 2) == CallInterruptOutcome.ALREADY_SETTLED
    assert fired == []


def test_claim_settlement_after_abort_yields_the_outcome_to_the_stop() -> None:
    """An abort that landed first owns the outcome: the settlement claim
    reports it lost, so the boundary settles interrupted rather than
    reporting the body's result."""
    registry = DrainRegistry()
    handle = registry.begin_work("exec-1", None, step_number=2)
    fired: list[bool] = []
    assert registry.attach_cancel_callback(handle, lambda: fired.append(True))

    assert registry.abort_call("exec-1", 2) == CallInterruptOutcome.INTERRUPTED
    assert not registry.claim_settlement(handle)
    assert fired == [True]


def test_local_call_begin_rejected_while_draining() -> None:
    """A whole-run drain latched between the OE claim and the local dispatch:
    the tool must not start."""
    from agent_engine_runner_shared.server.drain import DrainRequest, _now_ms

    ran: list[bool] = []
    wrapper, registry = _make_wrapper()
    registry.apply(
        DrainRequest(
            request_id="drain-1",
            execution_id="exec-1",
            reason="execution_cancelled",
            deadline_at_ms=_now_ms() + 60_000,
            workspace_id=None,
        )
    )

    box: Dict[str, Any] = {}
    t = _drive_local_tool(wrapper, _SyncToolStub(lambda: ran.append(True)), box)
    t.join(timeout=5)
    assert isinstance(box.get("error"), asyncio.CancelledError), box
    assert ran == []


async def test_drain_cancels_a_live_local_call_body_and_finalizes(monkeypatch) -> None:
    """A whole-run drain reaches an in-graph tool body too: the worker-thread
    call is cancelled rather than waited out, its boundary reports interrupted
    and unwinds (never the per-call sentinel), and the drain finalizer completes
    instead of sleeping to its deadline — the off-loop end_work must wake it."""
    from agent_engine_runner_shared.server.drain import DrainRequest, _now_ms

    cleaned: list[str] = []
    reports: list[dict] = []
    monkeypatch.setattr(_secure_wrapper, "report_oe_result", lambda **kw: reports.append(kw))

    registry = DrainRegistry()
    # The /execute-wide registration, mirroring the AER handler: a task the
    # drain cancels first.
    main_handle = registry.begin_work("exec-1", asyncio.create_task(_run_forever()))

    async def slow_body() -> str:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cleaned.append("cleaned")
            raise
        return "unreachable"

    wrapper = SecureToolWrapper(oe_url="http://oe", execution_id="exec-1", call_registry=registry)
    box: Dict[str, Any] = {}
    t = _drive_local_tool(wrapper, _AsyncToolStub(slow_body), box)
    await _await_channel_async(registry, 2)

    outcome = registry.apply(
        DrainRequest(
            request_id="drain-1",
            execution_id="exec-1",
            reason="execution_cancelled",
            deadline_at_ms=_now_ms() + 5_000,
            workspace_id=None,
        )
    )
    assert outcome.status_code == 202

    # join off the loop: the worker's marshalled end_work needs this loop free.
    await asyncio.to_thread(t.join, 5)
    assert not t.is_alive(), "the drain must cancel the in-graph body, not wait it out"
    assert cleaned == ["cleaned"]
    assert isinstance(box.get("error"), asyncio.CancelledError), box
    assert reports and reports[-1]["status"] == "interrupted"

    registry.end_work(main_handle)
    record = None
    for _ in range(500):
        entry = registry._entries.get("exec-1")
        if entry is not None and entry.drain is not None and entry.drain.outcome is not None:
            record = entry.drain
            break
        await asyncio.sleep(0.01)
    assert record is not None, "drain never finalized"
    assert record.outcome == DrainOutcome.COMPLETED, record
