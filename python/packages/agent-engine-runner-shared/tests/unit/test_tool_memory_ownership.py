import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import Mock, patch

import pytest

from agent_engine_runner_shared import TenantRuntime
from agent_engine_runner_shared.context import clear_execution_context, set_execution_context
from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_KIND_MEMORY,
    ACTIVITY_KIND_TOOL,
    ActivityContext,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import WorkflowIdentity
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.models import ToolExecuteResponse
from agent_engine_runner_shared.secure_wrapper import SecureToolWrapper, create_secure_tool_function
from agent_engine_runner_shared.workflow import attempt_context_scope
from agent_engine_runner_shared.workflow.activity import completed_outcome
from agent_engine_runner_shared.workflow.client import ActivityDispatch, ActivityReplay


def _attempt() -> AttemptContext:
    return AttemptContext(
        attempt_id="attempt-1",
        fencing_token=7,
        owner_id="aer-1",
        workflow_identity=WorkflowIdentity(
            session_id="session-456",
            execution_id="execution-1",
        ),
    )


def _activity_context(activity_id: str) -> ActivityContext:
    return ActivityContext(
        workflow_identity=_attempt().workflow_identity,
        activity_id=activity_id,
        attempt_id="attempt-1",
        fencing_token=7,
    )


class _FakeWorkflow:
    def __init__(self, started: list[object]) -> None:
        self.started = iter(started)
        self.commands: list[Any] = []
        self.outcomes: list[Any] = []

    def start_activity(self, command: Any) -> object:
        self.commands.append(command)
        return next(self.started)

    def report_outcome(self, outcome: Any) -> None:
        self.outcomes.append(outcome)


@contextmanager
def _durable_execution(wrapper: SecureToolWrapper) -> Iterator[None]:
    tokens = set_execution_context(
        execution_id="execution-1",
        wrapper=wrapper,
        oe_url="http://oe:8000",
        user_id="user-123",
        session_id="session-456",
    )
    try:
        with attempt_context_scope(_attempt()):
            yield
    finally:
        clear_execution_context(tokens)


def _wrapper(workflow: _FakeWorkflow) -> SecureToolWrapper:
    wrapper = SecureToolWrapper("http://oe:8000", "execution-1")
    wrapper._workflow = workflow
    return wrapper


def _runtime(monkeypatch: pytest.MonkeyPatch) -> TenantRuntime:
    monkeypatch.setenv("ORG_ID", "org1")
    monkeypatch.setenv("PROJECT_ID", "proj1")
    runtime = TenantRuntime(app_name="Test")
    runtime._memory_engine = Mock()
    runtime._memory_engine.build_context.return_value = SimpleNamespace(
        formatted_context="customer context"
    )
    return runtime


def test_live_registered_tool_owns_memory_read_but_replay_and_standalone_do_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(monkeypatch)
    tool_invocations = 0

    def registered_tool() -> str:
        nonlocal tool_invocations
        tool_invocations += 1
        return runtime.build_context(
            query="customer C-001",
            user_id="user-123",
            session_id="session-456",
        )

    wrapped = create_secure_tool_function(registered_tool, "recall_customer")
    live = _FakeWorkflow([ActivityDispatch(context=_activity_context("tool-live"))])
    with (
        patch(
            "agent_engine_runner_shared.secure_wrapper.request_oe_approval",
            return_value=ToolExecuteResponse(proceed=True, route_to="callback"),
        ),
        patch("agent_engine_runner_shared.secure_wrapper.report_oe_result"),
        _durable_execution(_wrapper(live)),
    ):
        assert wrapped() == "customer context"

    assert tool_invocations == 1
    assert [command.activity_kind for command in live.commands] == [ACTIVITY_KIND_TOOL]

    replay = _FakeWorkflow(
        [
            ActivityReplay(
                outcome=completed_outcome(_activity_context("tool-replay"), "customer context")
            )
        ]
    )
    with _durable_execution(_wrapper(replay)):
        assert wrapped() == "customer context"

    assert tool_invocations == 1
    assert [command.activity_kind for command in replay.commands] == [ACTIVITY_KIND_TOOL]
    assert replay.outcomes == []

    standalone = _FakeWorkflow([ActivityDispatch(context=_activity_context("memory-standalone"))])
    with _durable_execution(_wrapper(standalone)):
        assert (
            runtime.build_context(
                query="standalone question",
                user_id="user-123",
                session_id="session-456",
            )
            == "customer context"
        )

    assert [command.activity_kind for command in standalone.commands] == [ACTIVITY_KIND_MEMORY]
    assert runtime._memory_engine is not None
    assert runtime._memory_engine.build_context.call_count == 2


@pytest.mark.asyncio
async def test_detached_tool_work_does_not_retain_memory_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(monkeypatch)
    release_descendant = asyncio.Event()
    descendant: asyncio.Task[str] | None = None

    async def read_after_tool() -> str:
        await release_descendant.wait()
        return runtime.build_context(
            query="after Tool",
            user_id="user-123",
            session_id="session-456",
        )

    def registered_tool() -> str:
        nonlocal descendant
        descendant = asyncio.create_task(read_after_tool())
        return "done"

    wrapped = create_secure_tool_function(registered_tool, "start_background_read")
    workflow = _FakeWorkflow(
        [
            ActivityDispatch(context=_activity_context("tool-live")),
            ActivityDispatch(context=_activity_context("memory-live")),
        ]
    )
    with (
        patch(
            "agent_engine_runner_shared.secure_wrapper.request_oe_approval",
            return_value=ToolExecuteResponse(proceed=True, route_to="callback"),
        ),
        patch("agent_engine_runner_shared.secure_wrapper.report_oe_result"),
        _durable_execution(_wrapper(workflow)),
    ):
        assert wrapped() == "done"
        release_descendant.set()
        assert descendant is not None
        assert await descendant == "customer context"

    assert [command.activity_kind for command in workflow.commands] == [
        ACTIVITY_KIND_TOOL,
        ACTIVITY_KIND_MEMORY,
    ]
