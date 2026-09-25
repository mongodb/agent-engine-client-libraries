"""Real ADK coverage for configured sub-agent transfer paths."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, ClassVar

import pytest
from agent_engine_sdk import AgentInput, RequestContext
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import START, Workflow
from google.genai import types

from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_OUTCOME_KIND_COMPLETED,
    ACTIVITY_OUTCOME_KIND_SUSPENDED,
    ActivityOutcome,
    StepActivityEntry,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import WorkflowIdentity
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.workflow import (
    attempt_context_scope,
    current_attempt_context,
    current_operation_path,
)
from agent_engine_runner_shared.workflow.activity import semantic_input_from_json
from agent_engine_sdk_adk.agent import ADKBaseAgent
from agent_engine_sdk_adk.errors import UnsupportedDurableADKError
from agent_engine_sdk_adk.execution_session import DurableSession
from agent_engine_sdk_adk.route import DurableRouteAdapter
from agent_engine_sdk_adk.stream import AdkTurn
from agent_engine_sdk_adk.suspend import function_response_frontier_content


def _attempt(
    *,
    replay_mode: bool = False,
    attempt_id: str = "attempt-1",
    fencing_token: int = 7,
) -> AttemptContext:
    return AttemptContext(
        attempt_id=attempt_id,
        fencing_token=fencing_token,
        replay_mode=replay_mode,
        workflow_identity=WorkflowIdentity(
            session_id="session-1",
            execution_id="execution-1",
        ),
    )


def _path() -> list[tuple[str, int]]:
    return [
        (segment.name, segment.ordinal) for segment in current_operation_path().segments
    ]


async def _collect(runner: Runner) -> list[Event]:
    return [
        event
        async for event in runner.run_async(
            user_id="user-1",
            session_id="session-1",
            new_message=types.Content(
                role="user", parts=[types.Part.from_text(text="go")]
            ),
        )
    ]


class _NestedTransferLlm(BaseLlm):
    observations: ClassVar[list[tuple[str, list[tuple[str, int]]]]] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        self.observations.append((self.model, _path()))
        targets = {
            "router-model": ("reviewer", "to-reviewer"),
            "reviewer-model": ("specialist", "to-specialist"),
        }
        if self.model in targets:
            target, call_id = targets[self.model]
            part = types.Part(
                function_call=types.FunctionCall(
                    id=call_id,
                    name="transfer_to_agent",
                    args={"agent_name": target},
                )
            )
        else:
            part = types.Part.from_text(text="specialist complete")
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


@pytest.mark.asyncio
async def test_nested_transfer_uses_canonical_agent_node_path() -> None:
    _NestedTransferLlm.observations = []
    service = InMemorySessionService()
    await service.create_session(
        app_name="app", user_id="user-1", session_id="session-1"
    )
    specialist = LlmAgent(
        name="specialist", model=_NestedTransferLlm(model="specialist-model")
    )
    reviewer = LlmAgent(
        name="reviewer",
        model=_NestedTransferLlm(model="reviewer-model"),
        sub_agents=[specialist],
    )
    router = LlmAgent(
        name="router",
        model=_NestedTransferLlm(model="router-model"),
        sub_agents=[reviewer],
    )
    adapter = DurableRouteAdapter(router)
    runner = Runner(
        agent=router,
        app_name="app",
        session_service=service,
    )

    with attempt_context_scope(_attempt()), adapter.operation_path_scope():
        events = await _collect(runner)

    assert _NestedTransferLlm.observations == [
        ("router-model", [("agent", 1)]),
        ("reviewer-model", [("agent", 1), ("reviewer", 1)]),
        (
            "specialist-model",
            [("agent", 1), ("reviewer", 1), ("specialist", 1)],
        ),
    ]
    specialist_event = next(event for event in events if event.author == "specialist")
    assert specialist_event.node_info.path == "router@1/reviewer@1/specialist@1"
    assert [
        boundary.name
        for boundary in adapter.operation_boundaries_for_event(specialist_event) or ()
    ] == ["reviewer", "specialist"]


@pytest.mark.parametrize("mode", [None, "single_turn", "task"])
def test_workflow_node_with_sub_agents_requires_chat_mode(mode: str | None) -> None:
    specialist = LlmAgent(
        name="specialist", model=_NestedTransferLlm(model="specialist-model")
    )
    reviewer = LlmAgent(
        name="reviewer",
        model=_NestedTransferLlm(model="reviewer-model"),
        sub_agents=[specialist],
    )
    router = LlmAgent(
        name="router",
        mode=mode,
        model=_NestedTransferLlm(model="router-model"),
        sub_agents=[reviewer],
    )
    try:
        workflow = Workflow(name="outer", edges=[(START, router)])
    except ValueError as exc:
        assert mode == "task"
        assert "mode='task'" in str(exc)
        return

    with pytest.raises(
        UnsupportedDurableADKError,
        match="mode='chat'",
    ):
        DurableSession(
            workflow,
            app_name="app",
            attempt=_attempt(),
            user_id="user-1",
            resume_activity_ids=frozenset(),
        )


@pytest.mark.asyncio
async def test_nested_workflow_transfer_composes_canonical_paths() -> None:
    _NestedTransferLlm.observations = []
    service = InMemorySessionService()
    await service.create_session(
        app_name="app", user_id="user-1", session_id="session-1"
    )
    specialist = LlmAgent(
        name="specialist", model=_NestedTransferLlm(model="specialist-model")
    )
    reviewer = LlmAgent(
        name="reviewer",
        model=_NestedTransferLlm(model="reviewer-model"),
        sub_agents=[specialist],
    )
    router = LlmAgent(
        name="router",
        mode="chat",
        model=_NestedTransferLlm(model="router-model"),
        sub_agents=[reviewer],
    )
    inner = Workflow(name="inner", edges=[(START, router)])
    workflow = Workflow(name="outer", edges=[(START, inner)])
    adapter = DurableRouteAdapter(workflow)
    runner = Runner(
        node=workflow,
        app_name="app",
        session_service=service,
    )

    with attempt_context_scope(_attempt()), adapter.operation_path_scope():
        await _collect(runner)

    assert _NestedTransferLlm.observations == [
        ("router-model", [("agent", 1), ("inner", 1), ("router", 1)]),
        (
            "reviewer-model",
            [
                ("agent", 1),
                ("inner", 1),
                ("router", 1),
                ("reviewer", 1),
            ],
        ),
        (
            "specialist-model",
            [
                ("agent", 1),
                ("inner", 1),
                ("router", 1),
                ("reviewer", 1),
                ("specialist", 1),
            ],
        ),
    ]


class _ParallelTransferLlm(BaseLlm):
    observations: ClassVar[dict[str, list[tuple[str, int]]]] = {}
    specialist_arrivals: ClassVar[set[str]] = set()
    specialists_started: ClassVar[asyncio.Event]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        if self.model.endswith("-specialist"):
            self.specialist_arrivals.add(self.model)
            if len(self.specialist_arrivals) == 2:
                self.specialists_started.set()
            await asyncio.wait_for(self.specialists_started.wait(), timeout=1)
        self.observations[self.model] = _path()
        if self.model.endswith("-router"):
            part = types.Part(
                function_call=types.FunctionCall(
                    id=f"{self.model}-transfer",
                    name="transfer_to_agent",
                    args={"agent_name": "specialist"},
                )
            )
        else:
            part = types.Part.from_text(text=f"{self.model} complete")
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


@pytest.mark.asyncio
async def test_parallel_workflow_transfers_keep_sibling_paths_isolated() -> None:
    _ParallelTransferLlm.observations = {}
    _ParallelTransferLlm.specialist_arrivals = set()
    _ParallelTransferLlm.specialists_started = asyncio.Event()
    service = InMemorySessionService()
    await service.create_session(
        app_name="app", user_id="user-1", session_id="session-1"
    )

    def branch(name: str) -> LlmAgent:
        specialist = LlmAgent(
            name="specialist",
            model=_ParallelTransferLlm(model=f"{name}-specialist"),
        )
        return LlmAgent(
            name=f"{name}_router",
            mode="chat",
            model=_ParallelTransferLlm(model=f"{name}-router"),
            sub_agents=[specialist],
        )

    workflow = Workflow(
        name="outer", edges=[(START, (branch("left"), branch("right")))]
    )
    adapter = DurableRouteAdapter(workflow)
    runner = Runner(node=workflow, app_name="app", session_service=service)

    with attempt_context_scope(_attempt()), adapter.operation_path_scope():
        await _collect(runner)

    assert _ParallelTransferLlm.observations == {
        "left-router": [("agent", 1), ("left_router", 1)],
        "left-specialist": [
            ("agent", 1),
            ("left_router", 1),
            ("specialist", 1),
        ],
        "right-router": [("agent", 1), ("right_router", 1)],
        "right-specialist": [
            ("agent", 1),
            ("right_router", 1),
            ("specialist", 1),
        ],
    }


class _RoundTripTransferLlm(BaseLlm):
    calls: ClassVar[dict[str, int]] = {}
    observations: ClassVar[list[tuple[str, int, list[tuple[str, int]]]]] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        number = self.calls.get(self.model, 0) + 1
        self.calls[self.model] = number
        self.observations.append((self.model, number, _path()))
        if self.model == "router-model":
            target = "specialist"
            call_id = f"to-specialist-{number}"
        elif number == 1:
            target = "router"
            call_id = "return-to-router"
        else:
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="round trip complete")],
                )
            )
            return
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id=call_id,
                            name="transfer_to_agent",
                            args={"agent_name": target},
                        )
                    )
                ],
            )
        )


@pytest.mark.asyncio
async def test_repeated_transfer_reserves_each_canonical_run() -> None:
    _RoundTripTransferLlm.calls = {}
    _RoundTripTransferLlm.observations = []
    service = InMemorySessionService()
    await service.create_session(
        app_name="app", user_id="user-1", session_id="session-1"
    )
    specialist = LlmAgent(
        name="specialist", model=_RoundTripTransferLlm(model="specialist-model")
    )
    router = LlmAgent(
        name="router",
        model=_RoundTripTransferLlm(model="router-model"),
        sub_agents=[specialist],
    )
    adapter = DurableRouteAdapter(router)
    runner = Runner(
        agent=router,
        app_name="app",
        session_service=service,
    )

    with attempt_context_scope(_attempt()), adapter.operation_path_scope():
        await _collect(runner)

    assert _RoundTripTransferLlm.observations == [
        ("router-model", 1, [("agent", 1)]),
        ("specialist-model", 1, [("agent", 1), ("specialist", 1)]),
        ("router-model", 2, [("agent", 1)]),
        ("specialist-model", 2, [("agent", 1), ("specialist", 2)]),
    ]


@pytest.mark.asyncio
async def test_repeated_transfer_appends_to_workflow_path() -> None:
    _RoundTripTransferLlm.calls = {}
    _RoundTripTransferLlm.observations = []
    service = InMemorySessionService()
    await service.create_session(
        app_name="app", user_id="user-1", session_id="session-1"
    )
    specialist = LlmAgent(
        name="specialist", model=_RoundTripTransferLlm(model="specialist-model")
    )
    router = LlmAgent(
        name="router",
        mode="chat",
        model=_RoundTripTransferLlm(model="router-model"),
        sub_agents=[specialist],
    )
    workflow = Workflow(name="outer", edges=[(START, router)])
    adapter = DurableRouteAdapter(workflow)
    runner = Runner(node=workflow, app_name="app", session_service=service)

    with attempt_context_scope(_attempt()), adapter.operation_path_scope():
        await _collect(runner)

    assert _RoundTripTransferLlm.observations == [
        ("router-model", 1, [("agent", 1), ("router", 1)]),
        (
            "specialist-model",
            1,
            [("agent", 1), ("router", 1), ("specialist", 1)],
        ),
        ("router-model", 2, [("agent", 1), ("router", 2)]),
        (
            "specialist-model",
            2,
            [("agent", 1), ("router", 2), ("specialist", 1)],
        ),
    ]


class _WaitingAgent(BaseAgent):
    completions: ClassVar[int] = 0
    observations: ClassVar[list[list[tuple[str, int]]]] = []
    disallow_transfer_to_parent: bool = False

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        type(self).observations.append(_path())
        answered = any(
            part.function_response is not None
            and part.function_response.id == "specialist-input"
            for event in ctx.session.events
            if event.content is not None
            for part in event.content.parts or ()
        )
        if answered:
            type(self).completions += 1
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="specialist complete")],
                ),
            )
            return
        yield Event(
            author=self.name,
            long_running_tool_ids=["specialist-input"],
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id="specialist-input",
                            name="adk_request_input",
                            args={"message": "Approve specialist work?"},
                        )
                    )
                ],
            ),
        )


class _TransferToWaitingAgentLlm(BaseLlm):
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id="to-specialist",
                            name="transfer_to_agent",
                            args={"agent_name": "specialist"},
                        )
                    )
                ],
            )
        )


class _ReplayRouterLlm(BaseLlm):
    calls: ClassVar[int] = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        type(self).calls += 1
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id=f"to-specialist-{type(self).calls}",
                            name="transfer_to_agent",
                            args={"agent_name": "specialist"},
                        )
                    )
                ],
            )
        )


class _ReplayWaitingAgent(BaseAgent):
    observations: ClassVar[list[tuple[str, list[tuple[str, int]]]]] = []
    disallow_transfer_to_parent: bool = False

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        attempt = current_attempt_context()
        assert attempt is not None
        path = _path()
        type(self).observations.append((attempt.attempt_id, path))

        first_visit = path[-1] == ("specialist", 1) and (
            len(path) == 2 or path[-2] == ("router", 1)
        )
        if first_visit:
            yield Event(
                author=self.name,
                actions={"transfer_to_agent": "router"},
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id="return-to-router",
                                name="transfer_to_agent",
                                response={"result": "returned"},
                            )
                        )
                    ],
                ),
            )
            return

        answered = any(
            part.function_response is not None
            and part.function_response.id == "specialist-input"
            for event in ctx.session.events
            if event.content is not None
            for part in event.content.parts or ()
        )
        if answered:
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="specialist complete")],
                ),
            )
            return

        yield Event(
            author=self.name,
            long_running_tool_ids=["specialist-input"],
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id="specialist-input",
                            name="adk_request_input",
                            args={"message": "Approve the second specialist visit?"},
                        )
                    )
                ],
            ),
        )


class _WorkflowClient:
    def __init__(self, outcome_kind: int) -> None:
        self.outcome_kind = outcome_kind
        self.commands: list[Any] = []
        self.completions: list[Any] = []

    async def __aenter__(self) -> _WorkflowClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def finalize_step(self, command: Any) -> list[StepActivityEntry]:
        self.commands.append(command)
        if not command.suspensions:
            return []
        outcome = ActivityOutcome(
            workflow_identity=WorkflowIdentity(
                session_id="session-1", execution_id="execution-1"
            ),
            activity_id="activity-1",
            attempt_id="attempt-1",
            fencing_token=7,
            outcome_kind=self.outcome_kind,
        )
        if self.outcome_kind == ACTIVITY_OUTCOME_KIND_COMPLETED:
            outcome.result.CopyFrom(semantic_input_from_json({"approved": True}))
        return [
            StepActivityEntry(
                position=command.suspensions[0].position,
                outcome=outcome,
            )
        ]

    async def complete_execution(self, command: Any) -> None:
        self.completions.append(command)


def _replay_application(*, workflow_root: bool) -> ADKBaseAgent:
    router = LlmAgent(
        name="router",
        mode="chat" if workflow_root else None,
        model=_ReplayRouterLlm(model="router-model"),
        sub_agents=[_ReplayWaitingAgent(name="specialist")],
    )
    root: BaseAgent | Workflow = router
    if workflow_root:
        inner = Workflow(name="inner", edges=[(START, router)])
        root = Workflow(name="outer", edges=[(START, inner)])
    return ADKBaseAgent(adk_agent=root, app_name="app")


def _request_context(*, resume: bool = False) -> RequestContext:
    return RequestContext(
        session_id="session-1",
        user_id="user-1",
        resume=resume,
        resume_data={"activity-1": {"approved": True}} if resume else None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow_root", [False, True])
async def test_second_sub_agent_wait_replays_from_turn_start_after_replacement(
    monkeypatch: pytest.MonkeyPatch,
    workflow_root: bool,
) -> None:
    fresh_client = _WorkflowClient(ACTIVITY_OUTCOME_KIND_SUSPENDED)
    replay_client = _WorkflowClient(ACTIVITY_OUTCOME_KIND_COMPLETED)
    active_client = [fresh_client]
    monkeypatch.setattr(
        "agent_engine_sdk_adk.workflow.get_current_oe_url", lambda: "http://oe"
    )
    monkeypatch.setattr(
        "agent_engine_sdk_adk.workflow.AsyncWorkflowClient",
        lambda _url: active_client[0],
    )
    monkeypatch.setattr(
        "agent_engine_runner_shared.workflow.settlement.get_current_oe_url",
        lambda: "http://oe",
    )
    monkeypatch.setattr(
        "agent_engine_runner_shared.workflow.settlement.AsyncWorkflowClient",
        lambda _url: active_client[0],
    )
    input = AgentInput(payload={"message": "go"})

    _ReplayRouterLlm.calls = 0
    _ReplayWaitingAgent.observations = []
    with attempt_context_scope(_attempt()):
        first_events = [
            event
            async for event in _replay_application(workflow_root=workflow_root).stream(
                _request_context(), input
            )
        ]

    assert [event.event for event in first_events] == ["suspend"]
    first_suspension = fresh_client.commands[0].suspensions[0]
    assert not fresh_client.commands[0].HasField("state")
    first_visit_path = (
        [("agent", 1), ("inner", 1), ("router", 1), ("specialist", 1)]
        if workflow_root
        else [("agent", 1), ("specialist", 1)]
    )
    second_visit_path = (
        [("agent", 1), ("inner", 1), ("router", 2), ("specialist", 1)]
        if workflow_root
        else [("agent", 1), ("specialist", 2)]
    )
    assert [
        (segment.name, segment.ordinal)
        for segment in first_suspension.position.operation_path.segments
    ] == second_visit_path

    _ReplayRouterLlm.calls = 0
    active_client[0] = replay_client
    replay_attempt = _attempt(
        replay_mode=True,
        attempt_id="attempt-2",
        fencing_token=8,
    )
    assert not replay_attempt.HasField("previous_state")
    with attempt_context_scope(replay_attempt):
        replay_events = [
            event
            async for event in _replay_application(workflow_root=workflow_root).stream(
                _request_context(resume=True), input
            )
        ]

    replay_suspension = replay_client.commands[0].suspensions[0]
    assert replay_suspension.position == first_suspension.position
    assert _ReplayWaitingAgent.observations[:4] == [
        ("attempt-1", first_visit_path),
        ("attempt-1", second_visit_path),
        ("attempt-2", first_visit_path),
        ("attempt-2", second_visit_path),
    ]
    assert [event.event for event in replay_events] == ["token", "result"]
    assert replay_events[0].data["content"] == "specialist complete"
    assert replay_events[1].data["response"] == "specialist complete"
    assert len(replay_client.completions) == 1


@pytest.mark.asyncio
async def test_sub_agent_wait_keeps_canonical_path_across_resume() -> None:
    _WaitingAgent.completions = 0
    _WaitingAgent.observations = []
    service = InMemorySessionService()
    await service.create_session(
        app_name="app", user_id="user-1", session_id="session-1"
    )
    router = LlmAgent(
        name="router",
        model=_TransferToWaitingAgentLlm(model="router-model"),
        sub_agents=[_WaitingAgent(name="specialist")],
    )
    adapter = DurableRouteAdapter(router)
    runner = Runner(
        agent=router,
        app_name="app",
        session_service=service,
    )

    with attempt_context_scope(_attempt()):
        first = AdkTurn(
            runner,
            user_id="user-1",
            session_id="session-1",
            new_message=types.Content(
                role="user", parts=[types.Part.from_text(text="go")]
            ),
            route_adapter=adapter,
        )
        assert [event async for event in first.stream(emit_tokens=True)] == []
        frontier = first.wait_frontier()
        second = AdkTurn(
            runner,
            user_id="user-1",
            session_id="session-1",
            new_message=function_response_frontier_content(
                frontier, [{"approved": True}]
            ),
            route_adapter=adapter,
        )
        events = [event async for event in second.stream(emit_tokens=True)]

    assert len(frontier) == 1
    assert frontier[0].function_call_id == "specialist-input"
    assert [
        boundary.name for boundary in frontier[0].provenance.operation_boundaries
    ] == ["specialist"]
    assert second.wait_frontier() == ()
    assert [event.data["content"] for event in events] == ["specialist complete"]
    assert _WaitingAgent.completions == 1
    assert _WaitingAgent.observations == [
        [("agent", 1), ("specialist", 1)],
        [("agent", 1), ("specialist", 1)],
    ]
