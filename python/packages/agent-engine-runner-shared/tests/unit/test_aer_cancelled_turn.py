"""Cancelled-turn checkpoint hook: execute teardown commits the turn only for
execution-cancelled drains — never for rolling restarts, timeouts, or normal
settlement — and a failing commit never changes the settlement path."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    TenantScope,
    WorkflowIdentity,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.hooks import (
    clear_workflow_adapter,
    register_workflow_adapter,
)
from agent_engine_runner_shared.server.aer import AERServer

_DURABLE_EXECUTE_IDENTITY = {
    "org_id": "org-1",
    "project_id": "project-1",
    "workspace_id": "workspace-1",
    "user_id": "user-1",
    "session_id": "session-1",
}


def _registering_get_agent(runtime, agent) -> None:  # noqa: ANN001, ANN202
    """Fresh-worker shape: the workflow adapter appears only when the runtime
    materializes the agent — never before — like the production SDK runtimes."""
    runtime.app_name = "test-app"
    runtime.app_version = "1"
    runtime.memory_enabled = False

    def _get_agent(**kwargs) -> object:  # noqa: ANN002, ANN003
        register_workflow_adapter("langgraph", "1")
        return agent

    runtime.get_agent.side_effect = _get_agent


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _wait_for_active(registry, execution_id: str) -> None:  # noqa: ANN001, ANN202
    """Poll until work for the execution is registered — deterministic, unlike
    a fixed sleep that a slow dispatch can outlast."""
    for _ in range(500):
        entry = registry._entries.get(execution_id)
        if entry is not None and entry.active_count > 0:
            return
        await asyncio.sleep(0.005)
    raise AssertionError("work was never registered")


class _HangingAgentWithHook:
    """A turn parked on an await, exposing the duck-typed cancelled-turn hook."""

    def __init__(self) -> None:
        self.commit_cancelled_turn = AsyncMock(return_value=True)

    def execute(self, ctx, agent_input):  # noqa: ANN001, ANN202 - fake SDK surface
        async def _stream():
            await asyncio.sleep(3600)
            yield  # pragma: no cover - unreachable by construction

        return _stream()


class _HangingAgentNoHook:
    """Same parked turn, no hook — the shape of a non-LangGraph framework."""

    def execute(self, ctx, agent_input):  # noqa: ANN001, ANN202 - fake SDK surface
        async def _stream():
            await asyncio.sleep(3600)
            yield  # pragma: no cover - unreachable by construction

        return _stream()


def _drain_body(reason: str) -> dict:
    return {
        "request_id": f"drain-{reason}",
        "execution_id": "exec-1",
        "reason": reason,
        "deadline_at_ms": _now_ms() + 4000,
    }


async def _run_execute_and_drain(
    agent, reason: str, configure=None, execute_overrides=None
) -> None:  # noqa: ANN001, ANN202
    """Park a turn, drain it, and drive the drain to its final outcome.

    The final outcome is only recorded after end_work runs, which follows the
    execute finally — so once the drain settles, the teardown hook has
    definitely fired (or definitively not).
    """
    runtime = Mock()
    runtime.build_graph = AsyncMock(return_value=None)
    runtime.get_agent.return_value = agent
    server = AERServer(runtime)
    if configure is not None:
        configure(runtime, server)
    app = server.create_app()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://aer.test"
    ) as client:
        execute_task = asyncio.create_task(
            client.post(
                "/execute",
                json={
                    "execution_id": "exec-1",
                    "platform_api_url": "http://oe.invalid",
                    **(execute_overrides or {}),
                },
            )
        )
        await _wait_for_active(server.drain_registry, "exec-1")

        resp = await client.post("/drain", json=_drain_body(reason))
        assert resp.status_code == 202

        # The drained turn answers the held /execute cleanly: the OE's cancel
        # path owns the terminal Cancelled settlement, so no error surfaces.
        execute_resp = await asyncio.wait_for(execute_task, timeout=10)
        assert execute_resp.status_code == 200
        assert execute_resp.json()["status"] == "cancelled"

        final = None
        for _ in range(200):
            final = await client.post("/drain", json=_drain_body(reason))
            if final.json()["outcome"] != "accepted":
                break
            await asyncio.sleep(0.01)
        assert final is not None
        assert final.status_code == 200
        assert final.json()["outcome"] == "completed"


async def test_execution_cancelled_drain_commits_the_turn() -> None:
    agent = _HangingAgentWithHook()
    await _run_execute_and_drain(agent, "execution_cancelled")
    agent.commit_cancelled_turn.assert_awaited_once()


async def test_the_cancelled_callback_carries_the_checkpoint_coordinate() -> None:
    """The staged CANCELLED callback carries the rescued checkpoint's
    coordinate — OE's terminal guard persists it for the branch-source
    selector; the runner is the only party that knows it."""
    agent = _HangingAgentWithHook()
    agent.commit_cancelled_turn = AsyncMock(
        return_value={"thread_id": "sess-1:ws-1", "checkpoint_id": "ck-1"}
    )
    callback = AsyncMock()
    await _run_execute_and_drain(
        agent,
        "execution_cancelled",
        configure=lambda runtime, server: setattr(  # noqa: B010
            server, "_send_callback_body", callback
        ),
    )
    agent.commit_cancelled_turn.assert_awaited_once()
    callback.assert_awaited_once()
    body = callback.await_args.args[1]
    assert body["status"] == "CANCELLED"
    assert body["metadata"]["langgraph_checkpoint"] == {
        "thread_id": "sess-1:ws-1",
        "checkpoint_id": "ck-1",
    }


async def test_a_non_dict_hook_return_stages_no_cancelled_callback() -> None:
    agent = _HangingAgentWithHook()  # the mock resolves True, not a coordinate
    callback = AsyncMock()
    await _run_execute_and_drain(
        agent,
        "execution_cancelled",
        configure=lambda runtime, server: setattr(  # noqa: B010
            server, "_send_callback_body", callback
        ),
    )
    agent.commit_cancelled_turn.assert_awaited_once()
    callback.assert_not_awaited()


async def test_rolling_restart_drain_does_not_commit() -> None:
    agent = _HangingAgentWithHook()
    await _run_execute_and_drain(agent, "rolling_restart")
    agent.commit_cancelled_turn.assert_not_awaited()


async def test_agent_without_the_hook_drains_cleanly() -> None:
    await _run_execute_and_drain(_HangingAgentNoHook(), "execution_cancelled")


async def test_a_failing_commit_does_not_change_settlement() -> None:
    agent = _HangingAgentWithHook()
    agent.commit_cancelled_turn.side_effect = RuntimeError("mongo down")
    await _run_execute_and_drain(agent, "execution_cancelled")
    agent.commit_cancelled_turn.assert_awaited_once()


class _NoopHeartbeat:
    """AttemptHeartbeat stand-in — the lease lifecycle is orthogonal here."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, ANN202
        ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


def _durable_attempt() -> AttemptContext:
    return AttemptContext(
        attempt_id="attempt-1",
        fencing_token=7,
        owner_id="aer-1",
        replay_mode=False,
        workflow_identity=WorkflowIdentity(
            tenant_scope=TenantScope(
                org_id="org-1",
                project_id="project-1",
                workspace_id="workspace-1",
            ),
            session_id="session-1",
            execution_id="exec-1",
        ),
        heartbeat_interval_ms=1000,
    )


async def test_durable_attempt_never_commits_a_native_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The durable gate lives at the call site: by teardown the attempt
    context is closing, so a durable cancellation must skip the hook without
    relying on the adapter re-deriving durable-ness. The adapter registers
    inside get_agent (fresh worker): eligibility computed before
    materialization would wrongly read native for this run."""
    monkeypatch.setattr("agent_engine_runner_shared.server.aer.AttemptHeartbeat", _NoopHeartbeat)
    agent = _HangingAgentWithHook()

    def configure(runtime, server) -> None:  # noqa: ANN001, ANN202
        _registering_get_agent(runtime, agent)
        server._start_durable_attempt = AsyncMock(return_value=_durable_attempt())

    clear_workflow_adapter()
    try:
        await _run_execute_and_drain(
            agent,
            "execution_cancelled",
            configure=configure,
            execute_overrides=_DURABLE_EXECUTE_IDENTITY,
        )
    finally:
        clear_workflow_adapter()
    agent.commit_cancelled_turn.assert_not_awaited()


async def _drain_to_completion(client: httpx.AsyncClient, reason: str) -> None:
    final = None
    for _ in range(200):
        final = await client.post("/drain", json=_drain_body(reason))
        if final.json()["outcome"] != "accepted":
            break
        await asyncio.sleep(0.01)
    assert final is not None
    assert final.status_code == 200
    assert final.json()["outcome"] == "completed"


async def test_cancel_during_the_routing_probe_still_commits_the_turn() -> None:
    """A cancel landing while the admitted request awaits pre-stream
    preparation is still recorded: the probe resolves native, the latched
    drain unwinds the turn at attach, and teardown commits it."""
    agent = _HangingAgentWithHook()
    probe_entered = asyncio.Event()
    release_probe = asyncio.Event()

    async def slow_probe(request, start_request):  # noqa: ANN001, ANN202
        probe_entered.set()
        await release_probe.wait()
        return None  # native routing

    runtime = Mock()
    runtime.build_graph = AsyncMock(return_value=None)
    runtime.get_agent.return_value = agent
    server = AERServer(runtime)
    server._start_durable_attempt = slow_probe
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
        await _wait_for_active(server.drain_registry, "exec-1")
        await asyncio.wait_for(probe_entered.wait(), timeout=5)

        resp = await client.post("/drain", json=_drain_body("execution_cancelled"))
        assert resp.status_code == 202

        release_probe.set()
        execute_resp = await asyncio.wait_for(execute_task, timeout=10)
        assert execute_resp.status_code == 200
        assert execute_resp.json()["status"] == "cancelled"
        await _drain_to_completion(client, "execution_cancelled")

    agent.commit_cancelled_turn.assert_awaited_once()


async def test_unresolved_routing_never_writes_native_history() -> None:
    """A lease call that fails after the cancel landed leaves a durable-eligible
    run's mode unknown — teardown must not write native history for it. The
    adapter registers inside get_agent (fresh worker): eligibility computed
    before materialization would wrongly read native and write."""
    agent = _HangingAgentWithHook()
    probe_entered = asyncio.Event()
    release_probe = asyncio.Event()

    async def failing_probe(request, start_request):  # noqa: ANN001, ANN202
        probe_entered.set()
        await release_probe.wait()
        raise RuntimeError("oe unreachable")

    runtime = Mock()
    runtime.build_graph = AsyncMock(return_value=None)
    runtime.get_agent.return_value = agent
    _registering_get_agent(runtime, agent)
    server = AERServer(runtime)
    server._start_durable_attempt = failing_probe
    app = server.create_app()

    clear_workflow_adapter()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://aer.test"
        ) as client:
            execute_task = asyncio.create_task(
                client.post(
                    "/execute",
                    json={
                        "execution_id": "exec-1",
                        "platform_api_url": "http://oe.invalid",
                        **_DURABLE_EXECUTE_IDENTITY,
                    },
                )
            )
            await _wait_for_active(server.drain_registry, "exec-1")
            await asyncio.wait_for(probe_entered.wait(), timeout=5)

            resp = await client.post("/drain", json=_drain_body("execution_cancelled"))
            assert resp.status_code == 202

            release_probe.set()
            execute_resp = await asyncio.wait_for(execute_task, timeout=10)
            assert execute_resp.status_code == 200
            assert execute_resp.json()["status"] == "cancelled"
            await _drain_to_completion(client, "execution_cancelled")
    finally:
        clear_workflow_adapter()

    agent.commit_cancelled_turn.assert_not_awaited()
