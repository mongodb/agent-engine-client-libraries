"""Integration coverage across LangGraph nesting and durable activity replay."""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, TypedDict, cast
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolRuntime
from agent_engine_sdk.models import LLMStreamChunk

from agent_engine_sdk_langgraph.durable_subgraphs import DurableSubgraphResolver
from agent_engine_sdk_langgraph.durable_deep_agent import DurableDeepAgentMiddleware
from agent_engine_sdk_langgraph.secure_llm import SecureWrappedLLM
from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_KIND_TOOL,
    ActivityCommand,
    ActivityContext,
    ActivityOutcome,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import WorkflowIdentity
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
from agent_engine_runner_shared.secure_wrapper import OperationalStepAllocator
from agent_engine_runner_shared.workflow import (
    ChildOperationBoundary,
    attempt_context_scope,
    operation_path_resolver_scope,
)
from agent_engine_runner_shared.workflow.activity import run_serial_activity
from agent_engine_runner_shared.workflow.client import ActivityDispatch, ActivityReplay
from agent_engine_runner_shared.workflow.context import advance_step_ordinal


class _State(TypedDict, total=False):
    case_id: str


Position = tuple[tuple[tuple[str, int], ...], int, int]


class _ReplayClient:
    def __init__(self) -> None:
        self.commands: list[ActivityCommand] = []
        self._outcomes: dict[Position, ActivityOutcome] = {}
        self._positions_by_activity: dict[str, Position] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _position(command: ActivityCommand) -> Position:
        path = tuple(
            (segment.name, segment.ordinal)
            for segment in command.position.operation_path.segments
        )
        return (
            path,
            command.position.activity_ordinal,
            command.position.step_ordinal,
        )

    def start_activity(
        self, command: ActivityCommand
    ) -> ActivityDispatch | ActivityReplay:
        with self._lock:
            self.commands.append(command)
            position = self._position(command)
            if position in self._outcomes:
                return ActivityReplay(self._outcomes[position])

            activity_id = f"activity-{len(self.commands)}"
            self._positions_by_activity[activity_id] = position
            return ActivityDispatch(
                ActivityContext(
                    workflow_identity=command.workflow_identity,
                    activity_id=activity_id,
                    attempt_id=command.attempt_id,
                    fencing_token=command.fencing_token,
                )
            )

    def report_outcome(self, outcome: ActivityOutcome) -> None:
        with self._lock:
            self._outcomes[self._positions_by_activity[outcome.activity_id]] = outcome


def _graph(client: _ReplayClient, side_effects: list[str]) -> CompiledStateGraph:
    def activity(name: str):
        def run(state: _State) -> dict[str, Any]:
            run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name=name,
                activity_ordinal=1,
                semantic_input={"case_id": state.get("case_id")},
                execute=lambda _: side_effects.append(name) or {"source": name},
            )
            return {}

        return run

    policy = StateGraph(_State)
    policy.add_node("read_policy", activity("read_policy"))
    policy.add_edge(START, "read_policy")
    policy.add_edge("read_policy", END)

    investigation = StateGraph(_State)
    investigation.add_node("collect_evidence", activity("collect_evidence"))
    investigation.add_node("policy_analysis", policy.compile())
    investigation.add_edge(START, "collect_evidence")
    investigation.add_edge("collect_evidence", "policy_analysis")
    investigation.add_edge("policy_analysis", END)

    root = StateGraph(_State)
    root.add_node("coordinate_case", activity("coordinate_case"))
    root.add_node("case_investigation", investigation.compile())
    root.add_edge(START, "coordinate_case")
    root.add_edge("coordinate_case", "case_investigation")
    root.add_edge("case_investigation", END)
    return root.compile()


def _attempt(attempt_id: str, fencing_token: int, *, replay: bool) -> AttemptContext:
    return AttemptContext(
        attempt_id=attempt_id,
        fencing_token=fencing_token,
        replay_mode=replay,
        workflow_identity=WorkflowIdentity(
            session_id="support-session",
            execution_id="support-execution",
        ),
    )


def _wrapped_llm(
    monkeypatch: pytest.MonkeyPatch,
    client: _ReplayClient,
    side_effects: list[str],
) -> tuple[SimpleNamespace, SecureWrappedLLM]:
    class _DurableTestProxy(SecureLLMProxy):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._workflow = client

        def _stream_llm(
            self, invoke_request: Any, resolved_step: int
        ) -> Iterator[LLMStreamChunk]:
            prompt = str(invoke_request.messages[-1].content)
            side_effects.append(prompt)
            yield LLMStreamChunk(content=f"resolved:{prompt}")

    monkeypatch.setattr(
        "agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy", _DurableTestProxy
    )
    monkeypatch.setattr(
        "agent_engine_sdk_langgraph.secure_llm.Metrics.record_latency", MagicMock()
    )
    wrapper = SimpleNamespace(
        oe_url="http://oe:8000",
        execution_id="support-execution",
        operational_steps=OperationalStepAllocator(),
        durable_memory=None,
    )
    wrapper.next_operational_step = lambda: wrapper.operational_steps.next()
    return wrapper, SecureWrappedLLM(
        llm=MagicMock(),
        get_wrapper=lambda: wrapper,
        llm_id="support-model",
        model_name="test-model",
    )


def _command_path(command: ActivityCommand) -> tuple[tuple[str, int], ...]:
    return tuple(
        (segment.name, segment.ordinal)
        for segment in command.position.operation_path.segments
    )


@pytest.mark.anyio
async def test_direct_child_and_grandchild_reconstruct_paths_on_replay() -> None:
    client = _ReplayClient()
    side_effects: list[str] = []
    graph = _graph(client, side_effects)
    resolver = DurableSubgraphResolver(graph)

    for attempt in (
        _attempt("attempt-1", 1, replay=False),
        _attempt("attempt-2", 2, replay=True),
    ):
        with attempt_context_scope(attempt):
            with operation_path_resolver_scope(resolver):
                await graph.ainvoke({"case_id": "CASE-1001"})

    expected_paths = [
        (("agent", 1),),
        (("agent", 1), ("case_investigation", 1)),
        (
            ("agent", 1),
            ("case_investigation", 1),
            ("policy_analysis", 1),
        ),
    ]
    actual_paths = [
        tuple(
            (segment.name, segment.ordinal)
            for segment in command.position.operation_path.segments
        )
        for command in client.commands
    ]

    assert actual_paths == expected_paths * 2
    assert side_effects == ["coordinate_case", "collect_evidence", "read_policy"]


def test_real_secure_llm_wrapper_replays_sequential_calls_in_one_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ReplayClient()
    side_effects: list[str] = []
    wrapper, llm = _wrapped_llm(monkeypatch, client, side_effects)
    boundaries = [ChildOperationBoundary("case_investigation", "task-a")]

    for attempt in (
        _attempt("attempt-1", 1, replay=False),
        _attempt("attempt-2", 2, replay=True),
    ):
        wrapper.operational_steps = OperationalStepAllocator()
        with attempt_context_scope(attempt):
            with operation_path_resolver_scope(lambda: boundaries):
                llm._generate([HumanMessage(content="collect evidence")])
                llm._generate([HumanMessage(content="reconcile evidence")])

    assert [command.position.activity_ordinal for command in client.commands] == [
        1,
        2,
        1,
        2,
    ]
    assert side_effects == ["collect evidence", "reconcile evidence"]


def test_deep_agent_replays_sequential_same_name_dispatches() -> None:
    client = _ReplayClient()
    middleware = DurableDeepAgentMiddleware()
    side_effects: list[str] = []

    def dispatch(tool_call_id: str, name: str) -> None:
        request = ToolCallRequest(
            tool_call={
                "name": "task",
                "id": tool_call_id,
                "args": {"subagent_type": "research", "description": name},
            },
            tool=None,
            state={},
            runtime=cast(ToolRuntime[None, dict[str, Any]], None),
        )

        def handler(_: ToolCallRequest) -> ToolMessage:
            run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name=name,
                activity_ordinal=1,
                semantic_input={"query": name},
                execute=lambda _: side_effects.append(name) or {"result": name},
            )
            return ToolMessage(content="done", tool_call_id=tool_call_id)

        middleware.wrap_tool_call(request, handler)

    for attempt in (
        _attempt("attempt-1", 1, replay=False),
        _attempt("attempt-2", 2, replay=True),
    ):
        with attempt_context_scope(attempt):
            dispatch("task-a", "first research")
            advance_step_ordinal(1)
            dispatch("task-b", "second research")

    assert [_command_path(command) for command in client.commands] == [
        (("agent", 1), ("research", 1)),
        (("agent", 1), ("research", 2)),
        (("agent", 1), ("research", 1)),
        (("agent", 1), ("research", 2)),
    ]
    assert side_effects == ["first research", "second research"]


def test_deep_agent_replays_same_step_same_name_dispatches() -> None:
    client = _ReplayClient()
    middleware = DurableDeepAgentMiddleware()
    side_effects: list[str] = []
    siblings = (
        ("task-a", "first research"),
        ("task-b", "second research"),
    )

    def dispatch(tool_call_id: str, name: str) -> None:
        request = ToolCallRequest(
            tool_call={
                "name": "task",
                "id": tool_call_id,
                "args": {"subagent_type": "research", "description": name},
            },
            tool=None,
            state={},
            runtime=cast(ToolRuntime[None, dict[str, Any]], None),
        )

        def handler(_: ToolCallRequest) -> ToolMessage:
            run_serial_activity(
                client=client,  # type: ignore[arg-type]
                kind=ACTIVITY_KIND_TOOL,
                name=name,
                activity_ordinal=1,
                semantic_input={"query": name},
                execute=lambda _: side_effects.append(name) or {"result": name},
            )
            return ToolMessage(content="done", tool_call_id=tool_call_id)

        middleware.wrap_tool_call(request, handler)

    parent_state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "id": tool_call_id,
                        "args": {
                            "subagent_type": "research",
                            "description": name,
                        },
                    }
                    for tool_call_id, name in siblings
                ],
            )
        ]
    }

    for attempt in (
        _attempt("attempt-1", 1, replay=False),
        _attempt("attempt-2", 2, replay=True),
    ):
        with attempt_context_scope(attempt):
            middleware.after_model(parent_state, None)  # type: ignore[arg-type]
            dispatch(siblings[1][0], siblings[1][1])
            dispatch(siblings[0][0], siblings[0][1])

    assert [_command_path(command) for command in client.commands] == [
        (("agent", 1), ("research", 2)),
        (("agent", 1), ("research", 1)),
        (("agent", 1), ("research", 2)),
        (("agent", 1), ("research", 1)),
    ]
    assert side_effects == ["second research", "first research"]


def test_deep_agent_replays_concurrent_same_step_same_name_dispatches() -> None:
    client = _ReplayClient()
    middleware = DurableDeepAgentMiddleware()
    side_effects: list[str] = []
    siblings = (
        ("task-a", "first research"),
        ("task-b", "second research"),
    )
    parent_state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "id": tool_call_id,
                        "args": {
                            "subagent_type": "research",
                            "description": name,
                        },
                    }
                    for tool_call_id, name in siblings
                ],
            )
        ]
    }

    def run_attempt(attempt: AttemptContext, first_id: str) -> None:
        other_id = "task-a" if first_id == "task-b" else "task-b"
        entered = {tool_call_id: threading.Event() for tool_call_id, _ in siblings}
        start_other = threading.Event()
        errors: list[BaseException] = []
        original_preallocate = middleware._preallocate_pending_task_batch

        def gated_preallocate(request: ToolCallRequest) -> None:
            tool_call_id = request.tool_call["id"]
            assert tool_call_id is not None
            entered[tool_call_id].set()
            if tool_call_id == first_id:
                start_other.set()
                assert entered[other_id].wait(timeout=1)
            else:
                assert entered[first_id].wait(timeout=1)
            original_preallocate(request)

        def worker(tool_call_id: str, name: str, ctx: contextvars.Context) -> None:
            request = ToolCallRequest(
                tool_call={
                    "name": "task",
                    "id": tool_call_id,
                    "args": {"subagent_type": "research", "description": name},
                },
                tool=None,
                state=parent_state,
                runtime=cast(ToolRuntime[None, dict[str, Any]], None),
            )

            def handler(_: ToolCallRequest) -> ToolMessage:
                run_serial_activity(
                    client=client,  # type: ignore[arg-type]
                    kind=ACTIVITY_KIND_TOOL,
                    name=name,
                    activity_ordinal=1,
                    semantic_input={"query": name},
                    execute=lambda _: side_effects.append(name) or {"result": name},
                )
                return ToolMessage(content="done", tool_call_id=tool_call_id)

            try:
                if tool_call_id != first_id:
                    assert start_other.wait(timeout=1)
                ctx.run(middleware.wrap_tool_call, request, handler)
            except BaseException as exc:
                errors.append(exc)

        with attempt_context_scope(attempt):
            middleware.after_model(parent_state, None)  # type: ignore[arg-type]
            middleware._preallocate_pending_task_batch = gated_preallocate
            copies = {
                tool_call_id: contextvars.copy_context() for tool_call_id, _ in siblings
            }
            threads = [
                threading.Thread(
                    target=worker, args=(tool_call_id, name, copies[tool_call_id])
                )
                for tool_call_id, name in siblings
            ]
            try:
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
            finally:
                middleware._preallocate_pending_task_batch = original_preallocate
        assert errors == []
        assert entered[first_id].is_set()
        assert entered[other_id].is_set()

    run_attempt(_attempt("attempt-1", 1, replay=False), "task-b")
    run_attempt(_attempt("attempt-2", 2, replay=True), "task-a")

    expected_paths = {
        "first research": (("agent", 1), ("research", 1)),
        "second research": (("agent", 1), ("research", 2)),
    }
    first_attempt = {
        command.activity_name: _command_path(command) for command in client.commands[:2]
    }
    replay_attempt = {
        command.activity_name: _command_path(command) for command in client.commands[2:]
    }
    assert first_attempt == expected_paths
    assert replay_attempt == expected_paths
    assert set(side_effects) == set(expected_paths)
    assert len(side_effects) == 2


def test_deep_agent_keeps_compiled_parent_after_subagent_context_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ReplayClient()
    middleware = DurableDeepAgentMiddleware()
    resolver = DurableSubgraphResolver(_graph(client, []))
    side_effects: list[str] = []
    namespace = ""

    monkeypatch.setattr(
        "agent_engine_sdk_langgraph.durable_subgraphs.get_config",
        lambda: {"configurable": {"checkpoint_ns": namespace}},
    )

    request = ToolCallRequest(
        tool_call={
            "name": "task",
            "id": "task-research",
            "args": {
                "subagent_type": "research",
                "description": "collect evidence",
            },
        },
        tool=None,
        state={},
        runtime=cast(ToolRuntime[None, dict[str, Any]], None),
    )

    def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal namespace
        namespace = "subagent_model:subagent-task"
        run_serial_activity(
            client=client,  # type: ignore[arg-type]
            kind=ACTIVITY_KIND_TOOL,
            name="collect_evidence",
            activity_ordinal=1,
            semantic_input={"query": "collect evidence"},
            execute=lambda _: (
                side_effects.append("collect_evidence") or {"result": "evidence"}
            ),
        )
        return ToolMessage(content="done", tool_call_id="task-research")

    for attempt in (
        _attempt("attempt-1", 1, replay=False),
        _attempt("attempt-2", 2, replay=True),
    ):
        namespace = "case_investigation:compiled-task|collect_evidence:node-task"
        with attempt_context_scope(attempt):
            with operation_path_resolver_scope(resolver):
                middleware.wrap_tool_call(request, handler)

    assert [_command_path(command) for command in client.commands] == [
        (("agent", 1), ("case_investigation", 1), ("research", 1)),
        (("agent", 1), ("case_investigation", 1), ("research", 1)),
    ]
    assert side_effects == ["collect_evidence"]


def test_real_secure_llm_wrapper_replays_children_after_start_order_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ReplayClient()
    side_effects: list[str] = []
    wrapper, llm = _wrapped_llm(monkeypatch, client, side_effects)
    boundaries: list[ChildOperationBoundary] = []

    def invoke_children(order: tuple[str, ...], attempt: AttemptContext) -> None:
        wrapper.operational_steps = OperationalStepAllocator()
        with attempt_context_scope(attempt):
            with operation_path_resolver_scope(lambda: boundaries):
                for child_name in order:
                    boundaries[:] = [
                        ChildOperationBoundary(child_name, f"task-{child_name}")
                    ]
                    llm._generate([HumanMessage(content=f"work for {child_name}")])

    invoke_children(
        ("billing_investigation", "access_investigation"),
        _attempt("attempt-1", 1, replay=False),
    )
    invoke_children(
        ("access_investigation", "billing_investigation"),
        _attempt("attempt-2", 2, replay=True),
    )

    positions = [
        (_command_path(command), command.position.activity_ordinal)
        for command in client.commands
    ]
    assert positions == [
        ((("agent", 1), ("billing_investigation", 1)), 1),
        ((("agent", 1), ("access_investigation", 1)), 1),
        ((("agent", 1), ("access_investigation", 1)), 1),
        ((("agent", 1), ("billing_investigation", 1)), 1),
    ]
    assert side_effects == [
        "work for billing_investigation",
        "work for access_investigation",
    ]
