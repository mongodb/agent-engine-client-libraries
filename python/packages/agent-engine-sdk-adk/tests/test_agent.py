"""Durable-boundary tests for ADKBaseAgent."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from agent_engine_sdk import AgentInput, RequestContext
from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import Session
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.workflow import START, FunctionNode, Workflow, node
from google.genai import types

from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import WorkflowIdentity
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.generated.workflow.v1.state_pb2 import StateSnapshot
from agent_engine_runner_shared.workflow import attempt_context_scope
from agent_engine_sdk_adk.agent import ADKBaseAgent, UnsupportedDurableADKError
from agent_engine_sdk_adk.execution_session import DurableSession
from agent_engine_sdk_adk.platform_session import _ADAPTER_STATE_KEY
from agent_engine_sdk_adk.runtime import App
from agent_engine_sdk_adk.workflow import complete_durable_execution


def _agent() -> ADKBaseAgent:
    return ADKBaseAgent(adk_agent=MagicMock(), app_name="app")


def test_get_agent_runs_builder_with_customer_origin():
    from agent_engine_runner_shared.context import get_current_log_origin

    app = App("origin-test-app")
    seen = {}

    @app.entrypoint
    def builder():
        seen["origin"] = get_current_log_origin()
        return MagicMock()

    app.get_agent()
    assert seen["origin"] == "customer"


def _context(**overrides: Any) -> RequestContext:
    return RequestContext(
        session_id=overrides.get("session_id", "session-1"),
        user_id=overrides.get("user_id", "user-1"),
        resume=overrides.get("resume", False),
    )


def _input(**overrides: Any) -> AgentInput:
    return AgentInput(payload={"message": "hello", **overrides})


def _attempt(**overrides: Any) -> AttemptContext:
    attempt = AttemptContext(
        attempt_id="attempt-1",
        fencing_token=7,
        workflow_identity=WorkflowIdentity(
            session_id=overrides.get("session_id", "session-1"),
            execution_id="execution-1",
        ),
    )
    previous_state = overrides.get("previous_state")
    if previous_state is not None:
        attempt.previous_state.CopyFrom(previous_state)
    return attempt


async def _collect(agent: ADKBaseAgent, ctx: RequestContext, input: AgentInput) -> None:
    async for _ in agent.stream(ctx, input):
        pass


class _StaticToolset(BaseToolset):
    def __init__(self, tools: list[BaseTool]) -> None:
        super().__init__()
        self._tools = tools

    async def get_tools(self, readonly_context: Any = None) -> list[BaseTool]:
        del readonly_context
        return self._tools


class _NeverCalledLlm(BaseLlm):
    calls: ClassVar[int] = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        type(self).calls += 1
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="unexpected")],
            )
        )


@pytest.mark.asyncio
async def test_missing_durable_attempt_fails_before_adk_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = MagicMock()
    monkeypatch.setattr("google.adk.runners.Runner", runner)

    with pytest.raises(UnsupportedDurableADKError, match="OE-issued"):
        await _collect(_agent(), _context(), _input())

    runner.assert_not_called()


@pytest.mark.asyncio
async def test_previous_state_without_adk_state_fails_before_adk_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = MagicMock()
    monkeypatch.setattr("google.adk.runners.Runner", runner)

    with (
        attempt_context_scope(_attempt(previous_state=StateSnapshot())),
        pytest.raises(ValueError, match="adapter state"),
    ):
        await _collect(_agent(), _context(), _input())

    runner.assert_not_called()


@pytest.mark.asyncio
async def test_nested_agent_tool_fails_before_adk_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = MagicMock()
    monkeypatch.setattr("google.adk.runners.Runner", runner)
    child = LlmAgent(name="specialist", model="gemini-2.0-flash")
    parent = LlmAgent(
        name="router",
        model="gemini-2.0-flash",
        tools=[AgentTool(agent=child)],
    )
    workflow = Workflow(name="outer", edges=[(START, parent)])

    with (
        attempt_context_scope(_attempt()),
        pytest.raises(UnsupportedDurableADKError, match="AgentTool"),
    ):
        await _collect(
            ADKBaseAgent(adk_agent=workflow, app_name="app"),
            _context(),
            _input(),
        )

    runner.assert_not_called()


@pytest.mark.asyncio
async def test_parallel_worker_agent_tool_fails_before_adk_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = MagicMock()
    monkeypatch.setattr("google.adk.runners.Runner", runner)
    child = LlmAgent(name="specialist", model="gemini-2.0-flash")
    worker = LlmAgent(
        name="router",
        model="gemini-2.0-flash",
        tools=[AgentTool(agent=child)],
        parallel_worker=True,
    )
    workflow = Workflow(name="outer", edges=[(START, worker)])

    with (
        attempt_context_scope(_attempt()),
        pytest.raises(UnsupportedDurableADKError, match="AgentTool"),
    ):
        await _collect(
            ADKBaseAgent(adk_agent=workflow, app_name="app"),
            _context(),
            _input(),
        )

    runner.assert_not_called()


@pytest.mark.asyncio
async def test_parallel_worker_long_running_tool_fails_before_adk_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = MagicMock()
    monkeypatch.setattr("google.adk.runners.Runner", runner)

    def request_review(claim_id: str) -> None:
        del claim_id

    worker_agent = LlmAgent(
        name="claim_worker",
        model="gemini-2.0-flash",
        tools=[LongRunningFunctionTool(request_review)],
    )
    worker = node(
        worker_agent,
        parallel_worker=True,
        max_parallel_workers=2,
    )
    workflow = Workflow(name="outer", edges=[(START, worker)])

    with (
        attempt_context_scope(_attempt()),
        pytest.raises(UnsupportedDurableADKError, match="parallel_worker"),
    ):
        await _collect(
            ADKBaseAgent(adk_agent=workflow, app_name="app"),
            _context(),
            _input(),
        )

    runner.assert_not_called()


@pytest.mark.asyncio
async def test_parallel_worker_nested_workflow_long_running_tool_fails_before_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = MagicMock()
    monkeypatch.setattr("google.adk.runners.Runner", runner)

    def request_review(claim_id: str) -> None:
        del claim_id

    worker_agent = LlmAgent(
        name="claim_worker",
        model="gemini-2.0-flash",
        tools=[LongRunningFunctionTool(request_review)],
    )
    worker_workflow = Workflow(
        name="claim_workflow",
        edges=[(START, worker_agent)],
    )
    worker = node(
        worker_workflow,
        parallel_worker=True,
        max_parallel_workers=2,
    )
    workflow = Workflow(name="outer", edges=[(START, worker)])

    with (
        attempt_context_scope(_attempt()),
        pytest.raises(UnsupportedDurableADKError, match="parallel_worker"),
    ):
        await _collect(
            ADKBaseAgent(adk_agent=workflow, app_name="app"),
            _context(),
            _input(),
        )

    runner.assert_not_called()


@pytest.mark.asyncio
async def test_parallel_worker_toolset_fails_before_adk_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = MagicMock()
    monkeypatch.setattr("google.adk.runners.Runner", runner)

    def request_review(claim_id: str) -> None:
        del claim_id

    worker_agent = LlmAgent(
        name="claim_worker",
        model="gemini-2.0-flash",
        tools=[_StaticToolset([LongRunningFunctionTool(request_review)])],
    )
    worker = node(
        worker_agent,
        parallel_worker=True,
        max_parallel_workers=2,
    )
    workflow = Workflow(name="outer", edges=[(START, worker)])

    with (
        attempt_context_scope(_attempt()),
        pytest.raises(UnsupportedDurableADKError, match="toolsets"),
    ):
        await _collect(
            ADKBaseAgent(adk_agent=workflow, app_name="app"),
            _context(),
            _input(),
        )

    runner.assert_not_called()


@pytest.mark.asyncio
async def test_node_tool_fails_before_adk_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_tool_module = pytest.importorskip("google.adk.tools._node_tool")
    runner = MagicMock()
    monkeypatch.setattr("google.adk.runners.Runner", runner)
    child = FunctionNode(name="child", func=lambda: "child")
    child.input_schema = {"type": "object", "properties": {}}
    parent = LlmAgent(
        name="router",
        model="gemini-2.0-flash",
        tools=[node_tool_module.NodeTool(child)],
    )

    with (
        attempt_context_scope(_attempt()),
        pytest.raises(UnsupportedDurableADKError, match="NodeTool"),
    ):
        await _collect(
            ADKBaseAgent(adk_agent=parent, app_name="app"),
            _context(),
            _input(),
        )

    runner.assert_not_called()


@pytest.mark.asyncio
async def test_toolset_agent_tool_fails_before_parent_model() -> None:
    _NeverCalledLlm.calls = 0
    child = LlmAgent(name="specialist", model="gemini-2.0-flash")
    parent = LlmAgent(
        name="router",
        model=_NeverCalledLlm(model="parent-model"),
        tools=[_StaticToolset([AgentTool(agent=child)])],
    )

    with (
        attempt_context_scope(_attempt()),
        pytest.raises(UnsupportedDurableADKError, match="AgentTool"),
    ):
        await _collect(
            ADKBaseAgent(adk_agent=parent, app_name="app"),
            _context(),
            _input(),
        )

    assert _NeverCalledLlm.calls == 0


@pytest.mark.asyncio
async def test_configured_agent_tool_fails_before_adk_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = MagicMock()
    monkeypatch.setattr("google.adk.runners.Runner", runner)
    child = LlmAgent(name="specialist", model="gemini-2.0-flash")
    parent = LlmAgent(
        name="router",
        model="gemini-2.0-flash",
        tools=[AgentTool(agent=child)],
    )

    with pytest.raises(UnsupportedDurableADKError, match="AgentTool"):
        DurableSession(
            parent,
            app_name="app",
            attempt=_attempt(),
            user_id="user-1",
            resume_activity_ids=frozenset(),
        )

    runner.assert_not_called()


def test_sub_agent_agent_tool_fails_before_adk_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = MagicMock()
    monkeypatch.setattr("google.adk.runners.Runner", runner)
    nested = LlmAgent(name="nested", model="gemini-2.0-flash")
    reviewer = LlmAgent(
        name="reviewer",
        model="gemini-2.0-flash",
        tools=[AgentTool(agent=nested)],
    )
    root = LlmAgent(
        name="router",
        model="gemini-2.0-flash",
        sub_agents=[reviewer],
    )

    with pytest.raises(UnsupportedDurableADKError, match="AgentTool"):
        DurableSession(
            root,
            app_name="app",
            attempt=_attempt(),
            user_id="user-1",
            resume_activity_ids=frozenset(),
        )

    runner.assert_not_called()


@pytest.mark.asyncio
async def test_ordinary_toolset_remains_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("google.adk.runners.Runner", MagicMock())

    def lookup() -> str:
        return "found"

    parent = LlmAgent(
        name="router",
        model="gemini-2.0-flash",
        tools=[_StaticToolset([FunctionTool(lookup)])],
    )
    session = DurableSession(
        parent,
        app_name="app",
        attempt=_attempt(),
        user_id="user-1",
        resume_activity_ids=frozenset(),
    )

    assert [tool.name for tool in await parent.canonical_tools()] == ["lookup"]
    assert session.runner is not None


@pytest.mark.asyncio
async def test_invoke_returns_terminal_stream_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Runner:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def run_async(self, **_kwargs: Any) -> AsyncIterator[Event]:
            yield Event(
                author="agent",
                content=types.Content(
                    role="model", parts=[types.Part.from_text(text="done")]
                ),
            )

    async def _complete(*_args: Any) -> None:
        pass

    agent = _agent()
    monkeypatch.setattr("google.adk.runners.Runner", _Runner)
    monkeypatch.setattr(
        "agent_engine_sdk_adk.agent.complete_durable_execution", _complete
    )

    with attempt_context_scope(_attempt()):
        result = await agent.invoke(_context(), _input())

    assert result.response == {
        "status": "completed",
        "response": "done",
        "resumed": False,
    }


@pytest.mark.asyncio
async def test_terminal_finalization_records_serial_step_before_execution_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []

    class _Client:
        def __init__(self, oe_url: str) -> None:
            assert oe_url == "http://oe"

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            pass

        async def finalize_step(self, command: Any) -> list[Any]:
            calls.append(("step", command))
            return []

        async def complete_execution(self, command: Any) -> None:
            calls.append(("complete", command))

    monkeypatch.setattr(
        "agent_engine_runner_shared.workflow.settlement.get_current_oe_url",
        lambda: "http://oe",
    )
    monkeypatch.setattr(
        "agent_engine_runner_shared.workflow.settlement.AsyncWorkflowClient", _Client
    )
    attempt = _attempt()

    with attempt_context_scope(attempt):
        await complete_durable_execution(
            attempt,
            Session(id="session-1", app_name="app", user_id="user-1"),
        )

    assert [kind for kind, _ in calls] == ["step", "complete"]
    step = calls[0][1]
    completion = calls[1][1]
    assert step.workflow_identity == attempt.workflow_identity
    assert step.attempt_id == "attempt-1"
    assert step.fencing_token == 7
    assert step.step_ordinal == 1
    assert list(step.observed_activity_positions) == []
    assert _ADAPTER_STATE_KEY in step.state.properties
    assert completion.state == step.state
