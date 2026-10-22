"""Integration coverage across LangGraph nesting and durable activity replay."""

from __future__ import annotations

import asyncio
import contextvars
import json
import operator
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Annotated, Any, Required, TypedDict, cast
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Send
from agent_engine_sdk.models import LLMStreamChunk, ToolCallChunk

from agent_engine_sdk_langgraph.durable_subgraphs import (
    DurableSubgraphResolver,
    SendTaskRecorder,
)
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
from agent_engine_runner_shared.secure_wrapper import (
    OperationalStepAllocator,
    SecureToolWrapper,
)
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
    tool_call: Callable[[str], ToolCallChunk | None] = lambda _: None,
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
            call = tool_call(prompt)
            if call is None:
                yield LLMStreamChunk(content=f"resolved:{prompt}")
            else:
                yield LLMStreamChunk(tool_calls=[call])

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


class _FrontierClient:
    """Finalizes suspension frontiers, answering waits recorded in ``answers``."""

    def __init__(self) -> None:
        self.answers: dict[str, str] = {}
        self._activity_ids: dict[bytes, str] = {}

    async def __aenter__(self) -> _FrontierClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        pass

    async def complete_execution(self, command: Any) -> None:
        del command

    async def finalize_step(self, command: Any) -> list[Any]:
        from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
            ACTIVITY_OUTCOME_KIND_COMPLETED,
            ACTIVITY_OUTCOME_KIND_SUSPENDED,
            StepActivityEntry,
        )

        entries = []
        for suspension in command.suspensions:
            key = suspension.position.SerializeToString(deterministic=True)
            activity_id = self._activity_ids.setdefault(
                key, f"wait-{len(self._activity_ids) + 1}"
            )
            answer = self.answers.get(activity_id)
            outcome = ActivityOutcome(
                workflow_identity=command.workflow_identity,
                activity_id=activity_id,
                attempt_id=command.attempt_id,
                fencing_token=command.fencing_token,
                outcome_kind=(
                    ACTIVITY_OUTCOME_KIND_SUSPENDED
                    if answer is None
                    else ACTIVITY_OUTCOME_KIND_COMPLETED
                ),
            )
            if answer is None:
                outcome.suspension.CopyFrom(suspension.suspension)
            else:
                outcome.result.string_value = answer
            entries.append(
                StepActivityEntry(position=suspension.position, outcome=outcome)
            )
        return entries


@pytest.mark.anyio
@pytest.mark.parametrize("method", ["invoke", "stream"])
@pytest.mark.parametrize("call", ["invoke", "stream"])
@pytest.mark.parametrize(
    "layout",
    [
        "pause-model",
        "model-pause",
        "pause-model-pause",
        "model-pause-model",
    ],
)
async def test_model_calls_around_pauses_are_dispatched_once_across_attempts(
    monkeypatch: pytest.MonkeyPatch, method: str, call: str, layout: str
) -> None:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import MessagesState
    from langgraph.types import interrupt
    from agent_engine_sdk import AgentInput, RequestContext
    from agent_engine_sdk_langgraph import LangGraphBaseAgent, PlatformCheckpointer
    from agent_engine_sdk_langgraph import durable_session as durable_session_module
    from agent_engine_sdk_langgraph import (
        platform_checkpointer as checkpointer_module,
    )
    from agent_engine_runner_shared.context import current_oe_url

    activities = _ReplayClient()
    dispatched: list[str] = []
    wrapper, llm = _wrapped_llm(monkeypatch, activities, dispatched)
    frontier = _FrontierClient()
    monkeypatch.setattr(checkpointer_module, "AsyncWorkflowClient", lambda _: frontier)
    monkeypatch.setattr(
        durable_session_module, "AsyncWorkflowClient", lambda _: frontier
    )

    steps = layout.split("-")
    pauses = steps.count("pause")
    replies: list[tuple[int, str]] = []

    def model_reply(prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]
        if call == "invoke":
            return str(llm.invoke(messages).content)
        return "".join(str(chunk.content) for chunk in llm.stream(messages))

    def review(state: Any) -> Any:
        # Each step sees the previous step's output, so a model call's prompt
        # records which answers reached the node before it.
        seen = "start"
        for index, step in enumerate(steps):
            if step == "pause":
                seen = interrupt({"pause": index})
            else:
                seen = model_reply(f"call-{index} after {seen}")
                replies.append((index, seen))
        return {"messages": [AIMessage(content=str(seen))]}

    graph: StateGraph[Any] = StateGraph(MessagesState)
    graph.add_node("review", review)
    graph.add_edge(START, "review")
    graph.add_edge("review", END)
    agent = LangGraphBaseAgent(
        graph.compile(checkpointer=PlatformCheckpointer(native=InMemorySaver()))
    )
    ctx = RequestContext(user_id="user-1", session_id="support-session")
    agent_input = AgentInput(payload={"message": "hi"})

    async def run(attempt: AttemptContext, resume: dict[str, str] | None) -> Any:
        # Each attempt is a fresh process: operational steps restart.
        wrapper.operational_steps = OperationalStepAllocator()
        run_ctx = (
            ctx
            if resume is None
            else ctx.model_copy(update={"resume": True, "resume_data": resume})
        )
        with attempt_context_scope(attempt):
            if method == "invoke":
                return (await agent.invoke(run_ctx, agent_input)).response
            events = [event async for event in agent.stream(run_ctx, agent_input)]
            return events[-1].data

    url_token = current_oe_url.set("http://oe")
    try:
        result = await run(_attempt("attempt-1", 1, replay=False), None)
        for number in range(1, pauses + 1):
            (waiting,) = result["interrupts"]
            frontier.answers[waiting["id"]] = f"answer-{number}"
            result = await run(
                _attempt(f"attempt-{number + 1}", number + 1, replay=True),
                {waiting["id"]: f"answer-{number}"},
            )
    finally:
        current_oe_url.reset(url_token)

    # Replay the node's logic with the recorded answers to get the one prompt
    # each model call must have been dispatched with.
    expected_prompts: list[str] = []
    seen, answered = "start", 0
    expected_replies: dict[int, str] = {}
    for index, step in enumerate(steps):
        if step == "pause":
            answered += 1
            seen = f"answer-{answered}"
        elif step == "model":
            prompt = f"call-{index} after {seen}"
            expected_prompts.append(prompt)
            seen = f"resolved:{prompt}"
            expected_replies[index] = seen
    assert result["response"] == seen
    # Every execution of the node, live or replayed, received the one recorded
    # reply for each model call.
    assert {index for index, _ in replies} == set(expected_replies)
    assert all(reply == expected_replies[index] for index, reply in replies)
    first_call = min(expected_replies)
    if any(step != "model" for step in steps[first_call + 1 :]):
        # A pause after a model call re-executes the node, so that
        # call's reply was observed again from the recorded activity.
        assert len(replies) > len(expected_replies)
    # Every model call is a recorded activity: later attempts and re-entries
    # of the node after an answer replay it instead of dispatching again, and
    # separate calls in the node stay separate activities.
    assert dispatched == expected_prompts
    llm_positions = [
        (command.position.step_ordinal, command.position.activity_ordinal)
        for command in activities.commands
    ]
    assert len(set(llm_positions)) == len(expected_prompts)


class _ReviewFrontierClient(_FrontierClient):
    """A frontier whose answers are JSON objects, as a review's answer is."""

    def __init__(self) -> None:
        super().__init__()
        self.review_answers: dict[str, dict[str, Any]] = {}

    async def finalize_step(self, command: Any) -> list[Any]:
        from google.protobuf import json_format
        from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
            ACTIVITY_OUTCOME_KIND_COMPLETED,
        )

        entries = await super().finalize_step(command)
        for entry in entries:
            answer = self.review_answers.get(entry.outcome.activity_id)
            if answer is not None:
                entry.outcome.outcome_kind = ACTIVITY_OUTCOME_KIND_COMPLETED
                entry.outcome.ClearField("suspension")
                json_format.ParseDict(answer, entry.outcome.result)
        return entries


@pytest.mark.anyio
@pytest.mark.parametrize("method", ["invoke", "stream"])
@pytest.mark.parametrize("call", ["invoke", "stream"])
@pytest.mark.parametrize("decision", ["approve", "deny"])
async def test_reviewed_model_call_pauses_after_its_activity_and_resolves_from_oe(
    monkeypatch: pytest.MonkeyPatch, method: str, call: str, decision: str
) -> None:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import MessagesState
    from agent_engine_sdk import AgentInput, RequestContext
    from agent_engine_sdk_langgraph import LangGraphBaseAgent, PlatformCheckpointer
    from agent_engine_sdk_langgraph import durable_session as durable_session_module
    from agent_engine_sdk_langgraph import (
        platform_checkpointer as checkpointer_module,
    )
    from agent_engine_runner_shared.context import current_oe_url
    from agent_engine_runner_shared.models import (
        GuardrailReviewHalt,
        ToolExecuteResponse,
    )
    from agent_engine_runner_shared.secure_wrapper import (
        PolicyDeniedException,
        raise_for_oe_rejection,
    )

    from langgraph.types import interrupt

    # The runtime registers the framework's interrupt as the pause at startup.
    monkeypatch.setattr("agent_engine_runner_shared.hooks._suspend_handler", interrupt)

    from agent_engine_runner_shared.context import current_user_id, current_wrapper
    from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import TenantScope
    from agent_engine_runner_shared.workflow.memory import DurableMemoryState

    # Durable memory is on: every completed activity must be closed in memory,
    # and only conversation content may be written to it.
    memory_batches: list[list[tuple[str, str]]] = []
    memory_commands: list[Any] = []

    def ensure_memory_written(command: Any) -> None:
        memory_commands.append(command)
        turns = [json.loads(write.payload_json) for write in command.memory_writes]
        memory_batches.append([(turn["role"], turn["content"]) for turn in turns])

    def wait_acknowledgements(activity_id: str) -> list[tuple[str, int, int]]:
        return [
            (command.attempt_id, command.fencing_token, len(command.memory_writes))
            for command in memory_commands
            if command.activity_id == activity_id
        ]

    activities = _ReplayClient()
    activities.ensure_memory_written = ensure_memory_written  # type: ignore[attr-defined]
    requests: list[tuple[int, str | None, bool]] = []

    class _ReviewingOEProxy(SecureLLMProxy):
        """Stands in for OE: halts the call for review, then answers the review."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._workflow = activities

        def _request_oe_execution(
            self,
            invoke_request: Any,
            step: int | None = None,
            review_id: str | None = None,
        ) -> ToolExecuteResponse:
            step = self._allocate_step(step)
            requests.append((step, review_id, self._follows_review_protocol()))
            if review_id is None:
                return ToolExecuteResponse(
                    proceed=False,
                    status="require_review",
                    reason="output needs human review",
                    guardrail_review=GuardrailReviewHalt(review_id="review-1"),
                )
            if decision == "approve":
                return ToolExecuteResponse(
                    proceed=True,
                    status="success",
                    result={"content": "the held answer"},
                )
            raise_for_oe_rejection("guardrail review denied by reviewer")
            raise AssertionError("unreachable")

    monkeypatch.setattr(
        "agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy", _ReviewingOEProxy
    )
    monkeypatch.setattr(
        "agent_engine_sdk_langgraph.secure_llm.Metrics.record_latency", MagicMock()
    )
    monkeypatch.setattr(
        "agent_engine_sdk_langgraph.secure_llm.Metrics.record_error", MagicMock()
    )
    wrapper = SimpleNamespace(
        oe_url="http://oe:8000",
        execution_id="support-execution",
        operational_steps=OperationalStepAllocator(),
        durable_memory=None,
        workflow=SimpleNamespace(ensure_memory_written=ensure_memory_written),
    )
    wrapper.next_operational_step = lambda: wrapper.operational_steps.next()
    llm = SecureWrappedLLM(
        llm=MagicMock(),
        get_wrapper=lambda: wrapper,
        llm_id="support-model",
        model_name="test-model",
    )
    frontier = _ReviewFrontierClient()
    monkeypatch.setattr(checkpointer_module, "AsyncWorkflowClient", lambda _: frontier)
    monkeypatch.setattr(
        durable_session_module, "AsyncWorkflowClient", lambda _: frontier
    )

    def answer_claim(state: Any) -> Any:
        messages = [HumanMessage(content="summarize the claim")]
        if call == "invoke":
            content = str(llm.invoke(messages).content)
        else:
            content = "".join(str(chunk.content) for chunk in llm.stream(messages))
        return {"messages": [AIMessage(content=content)]}

    graph: StateGraph[Any] = StateGraph(MessagesState)
    graph.add_node("answer_claim", answer_claim)
    graph.add_edge(START, "answer_claim")
    graph.add_edge("answer_claim", END)
    agent = LangGraphBaseAgent(
        graph.compile(checkpointer=PlatformCheckpointer(native=InMemorySaver()))
    )
    ctx = RequestContext(user_id="user-1", session_id="support-session")
    agent_input = AgentInput(payload={"message": "hi"})

    def scoped(attempt: AttemptContext) -> AttemptContext:
        attempt.workflow_identity.tenant_scope.CopyFrom(
            TenantScope(
                org_id="org-1", project_id="project-1", workspace_id="workspace-1"
            )
        )
        return attempt

    async def run(attempt: AttemptContext, resume: dict[str, Any] | None) -> Any:
        # Each attempt is a fresh process: operational steps restart, and the
        # turn's user input is pending again until memory has it.
        wrapper.operational_steps = OperationalStepAllocator()
        wrapper.durable_memory = DurableMemoryState("summarize the claim")
        attempt = scoped(attempt)
        run_ctx = (
            ctx
            if resume is None
            else ctx.model_copy(update={"resume": True, "resume_data": resume})
        )
        with attempt_context_scope(attempt):
            if method == "invoke":
                return (await agent.invoke(run_ctx, agent_input)).response
            events = [event async for event in agent.stream(run_ctx, agent_input)]
            return events[-1].data

    url_token = current_oe_url.set("http://oe")
    wrapper_token = current_wrapper.set(wrapper)
    user_token = current_user_id.set("user-1")
    try:
        # Attempt 1: OE halts the call. The halt is the LLM activity's result
        # and the node pauses for the review it names.
        paused = await run(_attempt("attempt-1", 1, replay=False), None)
        (waiting,) = paused["interrupts"]
        assert waiting["value"] == {"guardrail_review": {"review_id": "review-1"}}
        assert requests == [(1, None, True)]
        # The halt is closed in memory with nothing written.
        assert memory_batches == [[]]

        # The reviewer answers, and a new attempt picks the turn up.
        answer = {"guardrail_review": {"review_id": "review-1", "decision": decision}}
        frontier.review_answers[waiting["id"]] = answer
        resumed = run(_attempt("attempt-2", 2, replay=True), {waiting["id"]: answer})
        if decision == "deny":
            with pytest.raises(PolicyDeniedException, match="denied by reviewer"):
                await resumed
        else:
            result = await resumed
            assert result["response"] == "the held answer"

        # The halted call replayed from its recorded result. Only the resolving
        # call reached OE, naming the review, at a step after the halted one.
        (halted, resolving) = requests
        assert resolving[1:] == ("review-1", True)
        assert resolving[0] > halted[0]
        assert (
            len({activities._position(command) for command in activities.commands}) == 2
        )

        # The answered wait is closed in memory under the attempt that
        # resumed it, with nothing written.
        assert wait_acknowledgements(waiting["id"]) == [("attempt-2", 2, 0)]

        # Neither the halt nor the reviewer's decision is conversation: the
        # only content memory ever receives is the user's input with the
        # released response.
        written = [batch for batch in memory_batches if batch]
        if decision == "approve":
            assert written == [
                [("user", "summarize the claim"), ("assistant", "the held answer")]
            ]
        else:
            assert written == []

        if decision == "approve":
            # A later attempt replays the halt, the answer, and the released
            # response without reaching OE.
            replayed = await run(
                _attempt("attempt-3", 3, replay=True), {waiting["id"]: answer}
            )
            assert replayed["response"] == "the held answer"
            assert len(requests) == 2
            assert wait_acknowledgements(waiting["id"])[-1] == ("attempt-3", 3, 0)
    finally:
        current_user_id.reset(user_token)
        current_wrapper.reset(wrapper_token)
        current_oe_url.reset(url_token)


class _SendWork(TypedDict, total=False):
    case_id: Required[str]
    done: Annotated[list[str], operator.add]


class _OrderGate:
    """Make the worker named first in each round call its model first."""

    def __init__(self) -> None:
        self.first = ""
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def reset(self, first: str) -> None:
        self.first = first
        self._events.clear()

    def _event(self, round_key: str) -> threading.Event:
        with self._lock:
            return self._events.setdefault(round_key, threading.Event())

    @contextmanager
    def turn(self, case_id: str) -> Iterator[None]:
        name, _, round_key = case_id.partition("#")
        event = self._event(round_key)
        if name != self.first:
            assert event.wait(timeout=5)
        yield
        if name == self.first:
            event.set()


class _RoundState(TypedDict, total=False):
    cases: list[str]
    rounds: int
    done: Annotated[list[str], operator.add]


def _round_worker(
    llm: SecureWrappedLLM,
    gate: _OrderGate,
    answers: list[tuple[str, str]],
    shape: str,
) -> Any:
    def ask(state: _SendWork) -> dict[str, Any]:
        case_id = state["case_id"]
        with gate.turn(case_id):
            result = llm._generate([HumanMessage(content=f"work for {case_id}")])
        answers.append((case_id, str(result.generations[0].message.content)))
        return {"done": [case_id]}

    if shape == "plain":
        return ask
    inner = StateGraph(_SendWork, output_schema=_RoundState)
    inner.add_node("ask", ask)
    inner.add_edge(START, "ask")
    inner.add_edge("ask", END)
    compiled = inner.compile()
    if shape == "compiled":
        return compiled

    # A plain worker that runs its own graph, outside the compiled topology.
    def run_inner(state: _SendWork) -> dict[str, Any]:
        return {"done": compiled.invoke(state)["done"]}

    return run_inner


def _send_rounds(cases: list[str], round_number: int) -> list[Send]:
    return [Send("work", {"case_id": f"{case}#{round_number}"}) for case in cases]


def _run_send_attempts(
    graph: CompiledStateGraph,
    gate: _OrderGate,
    wrapper: SimpleNamespace,
    graph_input: dict[str, Any],
    *,
    use_async: bool = False,
) -> None:
    resolver = DurableSubgraphResolver(graph)
    for first, attempt in (
        ("billing", _attempt("attempt-1", 1, replay=False)),
        ("access", _attempt("attempt-2", 2, replay=True)),
    ):
        gate.reset(first)
        wrapper.operational_steps = OperationalStepAllocator()
        with attempt_context_scope(attempt), resolver.scope(SendTaskRecorder()):
            if use_async:
                asyncio.run(graph.ainvoke(graph_input))
            else:
                graph.invoke(graph_input)


# A worker that runs its own graph inside a compiled parent is outside the
# compiled topology and is rejected, as it was before Send support.
@pytest.mark.parametrize("shape", ["plain", "compiled"])
def test_send_rounds_inside_a_compiled_coordinator_replay_per_worker(
    monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    client = _ReplayClient()
    side_effects: list[str] = []
    wrapper, llm = _wrapped_llm(monkeypatch, client, side_effects)
    gate = _OrderGate()
    answers: list[tuple[str, str]] = []

    def gather(state: _RoundState) -> dict[str, Any]:
        return {"rounds": state.get("rounds", 0) + 1}

    coordinator = StateGraph(_RoundState)
    coordinator.add_node("work", _round_worker(llm, gate, answers, shape))
    coordinator.add_node("gather", gather)
    coordinator.add_conditional_edges(
        START, lambda state: _send_rounds(state["cases"], 1), ["work"]
    )
    coordinator.add_edge("work", "gather")
    coordinator.add_conditional_edges(
        "gather",
        lambda state: (_send_rounds(state["cases"], 2) if state["rounds"] < 2 else END),
        ["work", END],
    )
    root = StateGraph(_RoundState)
    root.add_node("coordinate", coordinator.compile())
    root.add_edge(START, "coordinate")
    root.add_edge("coordinate", END)
    graph = root.compile()

    _run_send_attempts(graph, gate, wrapper, {"cases": ["billing", "access"]})

    cases = ["billing#1", "access#1", "billing#2", "access#2"]
    assert sorted(answers) == sorted(
        (case, f"resolved:work for {case}") for case in cases * 2
    )
    assert sorted(side_effects) == sorted(f"work for {case}" for case in cases)


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("shape", ["plain", "compiled", "invoked"])
def test_send_workers_replay_their_own_results_by_worker_shape(
    monkeypatch: pytest.MonkeyPatch, shape: str, use_async: bool
) -> None:
    client = _ReplayClient()
    side_effects: list[str] = []
    wrapper, llm = _wrapped_llm(monkeypatch, client, side_effects)
    gate = _OrderGate()
    answers: list[tuple[str, str]] = []

    root = StateGraph(_RoundState)
    root.add_node("work", _round_worker(llm, gate, answers, shape))
    root.add_conditional_edges(
        START, lambda state: _send_rounds(state["cases"], 1), ["work"]
    )
    root.add_edge("work", END)
    graph = root.compile()

    _run_send_attempts(
        graph, gate, wrapper, {"cases": ["billing", "access"]}, use_async=use_async
    )

    cases = ["billing#1", "access#1"]
    assert sorted(answers) == sorted(
        (case, f"resolved:work for {case}") for case in cases * 2
    )
    assert sorted(side_effects) == sorted(f"work for {case}" for case in cases)
    assert {_command_path(command)[:2] for command in client.commands} == {
        (("agent", 1), ("langgraph.task:__pregel_push:1:0", 1)),
        (("agent", 1), ("langgraph.task:__pregel_push:1:1", 1)),
    }


class _ToolState(TypedDict, total=False):
    cases: list[str]
    calls: Annotated[list[dict[str, Any]], operator.add]
    done: Annotated[list[str], operator.add]


class _ToolWork(TypedDict, total=False):
    case_id: Required[str]
    call: Required[dict[str, Any]]
    calls: Annotated[list[dict[str, Any]], operator.add]
    done: Annotated[list[str], operator.add]


@pytest.mark.parametrize("compiled_worker", [False, True])
def test_send_worker_tools_replay_at_their_model_positions_from_other_tasks(
    monkeypatch: pytest.MonkeyPatch, compiled_worker: bool
) -> None:
    """Each model's ToolCall runs in its own Send task, not the model's task."""
    client = _ReplayClient()
    side_effects: list[str] = []
    wrapper, llm = _wrapped_llm(
        monkeypatch,
        client,
        side_effects,
        tool_call=lambda prompt: ToolCallChunk(
            id=f"call-{prompt.split()[-1]}",
            name=f"lookup_{prompt.split()[-1]}",
            args="{}",
            index=0,
        ),
    )
    tools = SecureToolWrapper("http://oe:8000", "support-execution")
    tools._workflow = client
    gate = _OrderGate()
    results: list[tuple[str, Any]] = []

    def native_tool(*, tool_name: str, **_: Any) -> dict[str, str]:
        side_effects.append(tool_name)
        return {"answer": tool_name}

    monkeypatch.setattr(tools, "_execute_tool_native", native_tool)

    def plan(state: _ToolWork) -> dict[str, Any]:
        with gate.turn(state["case_id"]):
            result = llm._generate([HumanMessage(content=f"plan {state['case_id']}")])
        return {"calls": cast(AIMessage, result.generations[0].message).tool_calls}

    def run_tool(state: _ToolWork) -> dict[str, Any]:
        call = state["call"]
        with gate.turn(call["name"].removeprefix("lookup_")):
            result = tools.execute_tool(
                call["name"], call["args"], tool_call_id=call["id"], is_local=True
            )
        results.append((call["name"], result))
        return {"done": [call["name"]]}

    def dispatch(state: dict[str, Any]) -> list[Send]:
        return [Send("tool", {"call": call}) for call in reversed(state["calls"])]

    if compiled_worker:
        worker = StateGraph(_ToolWork, output_schema=_ToolState)
        worker.add_node("plan", plan)
        worker.add_node("tool", run_tool)
        worker.add_edge(START, "plan")
        worker.add_conditional_edges("plan", dispatch, ["tool"])
        worker.add_edge("tool", END)
        root = StateGraph(_ToolState)
        root.add_node("work", worker.compile())
        root.add_edge("work", END)
    else:
        root = StateGraph(_ToolState)
        root.add_node("work", plan)
        root.add_node("tool", run_tool)
        # One task sends every worker's Tool, so the Tool runs in a task other
        # than its model's, at a different Send index.
        root.add_node("collect", lambda state: {})
        root.add_edge("work", "collect")
        root.add_conditional_edges("collect", dispatch, ["tool"])
        root.add_edge("tool", END)
    root.add_conditional_edges(
        START,
        lambda state: [Send("work", {"case_id": case}) for case in state["cases"]],
        ["work"],
    )
    graph = root.compile()
    resolver = DurableSubgraphResolver(graph)

    for first, attempt in (
        ("billing", _attempt("attempt-1", 1, replay=False)),
        ("access", _attempt("attempt-2", 2, replay=True)),
    ):
        gate.reset(first)
        wrapper.operational_steps = OperationalStepAllocator()
        with attempt_context_scope(attempt), resolver.scope(SendTaskRecorder()):
            graph.invoke({"cases": ["billing", "access"]})

    assert sorted(results) == sorted(
        (name, {"answer": name}) for name in ["lookup_billing", "lookup_access"] * 2
    )
    assert sorted(side_effects) == [
        "lookup_access",
        "lookup_billing",
        "plan access",
        "plan billing",
    ]
    # Each Tool replays at the position its model reserved, in the model's path.
    model_paths = {
        f"lookup_{case}": _command_path(command)
        for case, command in zip(
            ["billing", "access"],
            sorted(
                (
                    c
                    for c in client.commands[:4]
                    if c.activity_kind != ACTIVITY_KIND_TOOL
                ),
                key=_command_path,
            ),
        )
    }
    for command in client.commands:
        if command.activity_kind == ACTIVITY_KIND_TOOL:
            assert _command_path(command) == model_paths[command.activity_name]
            assert command.position.activity_ordinal == 2


def test_tool_handed_to_a_sibling_subgraph_records_in_that_subgraph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without Send, a Tool records where it runs, as before Send support."""
    client = _ReplayClient()
    side_effects: list[str] = []
    wrapper, llm = _wrapped_llm(
        monkeypatch,
        client,
        side_effects,
        tool_call=lambda prompt: ToolCallChunk(
            id="call-billing", name="lookup_billing", args="{}", index=0
        ),
    )
    tools = SecureToolWrapper("http://oe:8000", "support-execution")
    tools._workflow = client

    def native_tool(*, tool_name: str, **_: Any) -> dict[str, str]:
        side_effects.append(tool_name)
        return {"answer": tool_name}

    monkeypatch.setattr(tools, "_execute_tool_native", native_tool)

    def plan(state: _ToolState) -> dict[str, Any]:
        result = llm._generate([HumanMessage(content="plan billing")])
        return {"calls": cast(AIMessage, result.generations[0].message).tool_calls}

    def run_tools(state: _ToolState) -> dict[str, Any]:
        calls = state.get("calls", [])
        for call in calls:
            tools.execute_tool(
                call["name"], call["args"], tool_call_id=call["id"], is_local=True
            )
        return {"done": [call["name"] for call in calls]}

    planner = StateGraph(_ToolState)
    planner.add_node("plan", plan)
    planner.add_edge(START, "plan")
    planner.add_edge("plan", END)
    executor = StateGraph(_ToolState)
    executor.add_node("run", run_tools)
    executor.add_edge(START, "run")
    executor.add_edge("run", END)
    root = StateGraph(_ToolState)
    root.add_node("planner", planner.compile())
    root.add_node("tools", executor.compile())
    root.add_edge(START, "planner")
    root.add_edge("planner", "tools")
    root.add_edge("tools", END)
    graph = root.compile()
    resolver = DurableSubgraphResolver(graph)

    wrapper.operational_steps = OperationalStepAllocator()
    with attempt_context_scope(_attempt("attempt-1", 1, replay=False)):
        with resolver.scope(SendTaskRecorder()):
            graph.invoke({"cases": ["billing"]})

    model, tool = client.commands
    assert tool.activity_kind == ACTIVITY_KIND_TOOL
    assert [name for name, _ in _command_path(model)] == ["agent", "planner"]
    assert [name for name, _ in _command_path(tool)] == ["agent", "tools"]
    assert tool.position.activity_ordinal == 1
