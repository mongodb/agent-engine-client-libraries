"""ADK human-input and confirmation stream tests."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_engine_sdk import AgentInput, RequestContext
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.events.event import NodeInfo
from google.adk.events.request_input import RequestInput
from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import Session
from google.adk.workflow import START, FunctionNode, Workflow
from google.genai import types

from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_OUTCOME_KIND_COMPLETED,
    ACTIVITY_OUTCOME_KIND_SUSPENDED,
    ActivityOutcome,
    StepActivityEntry,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import WorkflowIdentity
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.workflow.activity import semantic_input_from_json
from agent_engine_runner_shared.workflow.context import attempt_context_scope
from agent_engine_runner_shared.workflow.protojson import proto_struct_to_json
from agent_engine_sdk_adk.agent import ADKBaseAgent, UnsupportedDurableADKError


def _context(**overrides: Any) -> RequestContext:
    return RequestContext(
        session_id="session-1",
        user_id="user-1",
        resume=overrides.get("resume", False),
        resume_data=overrides.get("resume_data"),
    )


def _input() -> AgentInput:
    return AgentInput(payload={"message": "please review"})


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
            session_id="session-1", execution_id="execution-1"
        ),
    )


def _request_input_event() -> Event:
    return Event(
        invocation_id="invocation-1",
        author="agent",
        node_info=NodeInfo(path="agent@1"),
        long_running_tool_ids=["manager_approval"],
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="manager_approval",
                        name="adk_request_input",
                        args={
                            "message": "Please review this time off request.",
                            "payload": {"days": 5},
                            "response_schema": {
                                "type": "object",
                                "properties": {"approved": {"type": "boolean"}},
                            },
                        },
                    )
                )
            ],
        ),
    )


def _request_input_event_with_text(text: str) -> Event:
    event = _request_input_event()
    assert event.content is not None
    event.content.parts = [
        types.Part.from_text(text=text),
        *(event.content.parts or []),
    ]
    return event


def _parallel_request_input_event(function_call_id: str, branch: str) -> Event:
    return Event(
        invocation_id="invocation-1",
        author="agent",
        branch=branch,
        node_info=NodeInfo(path=f"agent@1/{branch.replace('.', '/')}"),
        long_running_tool_ids=[function_call_id],
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id=function_call_id,
                        name="adk_request_input",
                        args={"message": f"Answer {branch}"},
                    )
                )
            ],
        ),
    )


def _human_review_event() -> Event:
    return Event(
        invocation_id="invocation-1",
        author="agent",
        node_info=NodeInfo(path="agent@1"),
        long_running_tool_ids=["review-1"],
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="review-1",
                        name="human_review",
                        args={
                            "claim_id": "CLM-123",
                            "reason": "High claim amount requires human approval",
                        },
                    )
                )
            ],
        ),
    )


def _confirmation_event() -> Event:
    return Event(
        invocation_id="invocation-1",
        author="agent",
        node_info=NodeInfo(path="agent@1"),
        long_running_tool_ids=["call_book_flight"],
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="call_book_flight",
                        name="adk_request_confirmation",
                        args={
                            "originalFunctionCall": {
                                "id": "tool-1",
                                "name": "book_flight",
                                "args": {"origin": "SEA"},
                            },
                            "toolConfirmation": {"hint": "Book SEA → JFK?"},
                        },
                    )
                )
            ],
        ),
    )


def _text_event(text: str) -> Event:
    return Event(
        invocation_id="invocation-1",
        author="agent",
        content=types.Content(role="model", parts=[types.Part.from_text(text=text)]),
    )


class _WaitingBaseAgent(BaseAgent):
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        event = _request_input_event()
        event.invocation_id = ctx.invocation_id
        event.author = self.name
        event.node_info = NodeInfo()
        yield event


class _ContinuationLlm(BaseLlm):
    calls: ClassVar[int] = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        type(self).calls += 1
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="continued")],
            ),
            partial=False,
        )


def _serial_wait_workflow() -> Workflow:
    def request_review() -> RequestInput:
        return RequestInput(
            interrupt_id="serial-review",
            message="Review the serial leaf",
            payload={"leaf": "left"},
        )

    left = FunctionNode(name="left", func=request_review)
    inner = Workflow(name="inner", edges=[(START, left)])
    joined = LlmAgent(
        name="joined",
        model=_ContinuationLlm(model="continuation-model"),
    )
    return Workflow(
        name="outer",
        edges=[(START, inner), (inner, joined)],
    )


class _Runner:
    def __init__(
        self,
        *,
        turns: list[list[Event]],
        session_service: Any,
    ) -> None:
        self._turns = list(turns)
        self.messages: list[Any] = []
        self._session_service = session_service

    async def run_async(
        self,
        *,
        user_id: str,
        session_id: str,
        new_message: Any = None,
        invocation_id: str | None = None,
    ) -> AsyncIterator[Event]:
        del invocation_id
        self.messages.append(new_message)
        events = self._turns.pop(0) if self._turns else []
        session = await self._session_service.get_session(
            app_name="app", user_id=user_id, session_id=session_id
        )
        assert session is not None
        for event in events:
            await self._session_service.append_event(session, event)
            yield event


class _WorkflowClient:
    def __init__(
        self,
        entry_batches: list[list[StepActivityEntry]],
        *,
        reverse_return: bool = False,
    ) -> None:
        self._entry_batches = entry_batches
        self._reverse_return = reverse_return
        self.commands: list[Any] = []

    async def __aenter__(self) -> _WorkflowClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def finalize_step(self, command: Any) -> list[StepActivityEntry]:
        self.commands.append(command)
        if not command.suspensions:
            return []
        entries = self._entry_batches.pop(0)
        for index, entry in enumerate(entries):
            if not entry.HasField("position"):
                entry.position.CopyFrom(command.suspensions[index].position)
        return list(reversed(entries)) if self._reverse_return else entries

    async def complete_execution(self, command: Any) -> None:
        raise AssertionError("complete_execution must not run on a fresh wait")


def _suspended_entries() -> list[StepActivityEntry]:
    return [
        StepActivityEntry(
            outcome=ActivityOutcome(
                workflow_identity=WorkflowIdentity(
                    session_id="session-1", execution_id="execution-1"
                ),
                activity_id="activity-1",
                attempt_id="attempt-1",
                fencing_token=7,
                outcome_kind=ACTIVITY_OUTCOME_KIND_SUSPENDED,
            )
        )
    ]


def _completed_entries(result: Any) -> list[StepActivityEntry]:
    return [
        StepActivityEntry(
            outcome=ActivityOutcome(
                workflow_identity=WorkflowIdentity(
                    session_id="session-1", execution_id="execution-1"
                ),
                activity_id="activity-1",
                attempt_id="attempt-1",
                fencing_token=7,
                outcome_kind=ACTIVITY_OUTCOME_KIND_COMPLETED,
                result=semantic_input_from_json(result),
            )
        )
    ]


def _frontier_entries(
    outcomes: list[tuple[str, int, Any]],
) -> list[StepActivityEntry]:
    entries: list[StepActivityEntry] = []
    for activity_id, outcome_kind, result in outcomes:
        outcome = ActivityOutcome(
            workflow_identity=WorkflowIdentity(
                session_id="session-1", execution_id="execution-1"
            ),
            activity_id=activity_id,
            attempt_id="attempt-1",
            fencing_token=7,
            outcome_kind=outcome_kind,
        )
        if outcome_kind == ACTIVITY_OUTCOME_KIND_COMPLETED:
            outcome.result.CopyFrom(semantic_input_from_json(result))
        entries.append(StepActivityEntry(outcome=outcome))
    return entries


class _SuspendFixture:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        turns: list[list[Event]],
        entries: list[StepActivityEntry],
        additional_entries: list[list[StepActivityEntry]] | None = None,
        reverse_return: bool = False,
    ) -> None:
        self.complete = AsyncMock()
        self.client = _WorkflowClient(
            [entries, *(additional_entries or [])],
            reverse_return=reverse_return,
        )
        self.order: list[str] = []

        def _runner(**kwargs: Any) -> _Runner:
            self.session_service = kwargs["session_service"]
            self.runner = _Runner(
                turns=turns, session_service=kwargs["session_service"]
            )
            return self.runner

        async def _complete(_attempt: AttemptContext, session: Session) -> None:
            self.order.append("commit")
            await self.complete(_attempt, session)

        monkeypatch.setattr("google.adk.runners.Runner", _runner)
        monkeypatch.setattr(
            "agent_engine_sdk_adk.workflow.get_current_oe_url", lambda: "http://oe"
        )
        monkeypatch.setattr(
            "agent_engine_sdk_adk.workflow.AsyncWorkflowClient",
            lambda _url: self.client,
        )
        self.agent = ADKBaseAgent(adk_agent=MagicMock(), app_name="app")
        monkeypatch.setattr(
            "agent_engine_sdk_adk.agent.complete_durable_execution", _complete
        )

    async def collect(self, *, ctx: RequestContext | None = None) -> list[Any]:
        events = []
        async for event in self.agent.stream(ctx or _context(), _input()):
            if event.event == "result":
                self.order.append("result")
            if event.event == "suspend":
                self.order.append("suspend")
            events.append(event)
        return events


@pytest.mark.asyncio
async def test_request_input_suspends_without_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _SuspendFixture(
        monkeypatch,
        turns=[[_text_event("prefix"), _request_input_event()]],
        entries=_suspended_entries(),
    )

    with attempt_context_scope(_attempt()):
        events = await run.collect()

    assert [event.event for event in events] == ["token", "suspend"]
    suspend = events[-1]
    assert suspend.data["response"] == "prefix"
    assert suspend.data["interrupts"][0]["id"] == "activity-1"
    assert suspend.data["resume_schema"]["required"] == ["resume_map"]
    assert run.order == ["suspend"]
    run.complete.assert_not_awaited()
    assert len(run.client.commands) == 1
    command = run.client.commands[0]
    assert list(command.state.properties.keys()) == []
    assert len(command.suspensions) == 1
    assert command.suspensions[0].activity_name == "adk_request_input"
    suspension_context = proto_struct_to_json(command.suspensions[0].suspension.context)
    assert suspension_context == {"value": suspend.data["interrupts"][0]["value"]}
    semantic = str(command.suspensions[0].semantic_input)
    assert "manager_approval" not in semantic


@pytest.mark.asyncio
async def test_plain_base_agent_wait_uses_root_operation_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _WorkflowClient([_suspended_entries()])
    complete = AsyncMock()
    monkeypatch.setattr(
        "agent_engine_sdk_adk.workflow.get_current_oe_url", lambda: "http://oe"
    )
    monkeypatch.setattr(
        "agent_engine_sdk_adk.workflow.AsyncWorkflowClient", lambda _url: client
    )
    monkeypatch.setattr(
        "agent_engine_sdk_adk.agent.complete_durable_execution", complete
    )
    application = ADKBaseAgent(
        adk_agent=_WaitingBaseAgent(name="waiting_agent"),
        app_name="app",
    )

    with attempt_context_scope(_attempt()):
        events = [event async for event in application.stream(_context(), _input())]

    assert [event.event for event in events] == ["suspend"]
    suspension = client.commands[0].suspensions[0]
    assert [
        segment.name for segment in suspension.position.operation_path.segments
    ] == ["agent"]
    complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_serial_wait_keeps_leaf_path_across_application_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ContinuationLlm.calls = 0
    fresh_client = _WorkflowClient([_suspended_entries()])
    replay_client = _WorkflowClient([_completed_entries({"decision": "approved"})])
    active_client = [fresh_client]
    complete = AsyncMock()
    monkeypatch.setattr(
        "agent_engine_sdk_adk.workflow.get_current_oe_url", lambda: "http://oe"
    )
    monkeypatch.setattr(
        "agent_engine_sdk_adk.workflow.AsyncWorkflowClient",
        lambda _url: active_client[0],
    )
    monkeypatch.setattr(
        "agent_engine_sdk_adk.agent.complete_durable_execution", complete
    )

    first_application = ADKBaseAgent(
        adk_agent=_serial_wait_workflow(),
        app_name="app",
    )
    with attempt_context_scope(_attempt()):
        first_events = [
            event async for event in first_application.stream(_context(), _input())
        ]

    assert [event.event for event in first_events] == ["suspend"]
    assert _ContinuationLlm.calls == 0
    first_suspension = fresh_client.commands[0].suspensions[0]
    assert [
        segment.name for segment in first_suspension.position.operation_path.segments
    ] == ["agent", "inner", "left"]

    active_client[0] = replay_client
    replacement_application = ADKBaseAgent(
        adk_agent=_serial_wait_workflow(),
        app_name="app",
    )
    with attempt_context_scope(
        _attempt(
            replay_mode=True,
            attempt_id="attempt-2",
            fencing_token=8,
        )
    ):
        replacement_events = [
            event
            async for event in replacement_application.stream(
                _context(
                    resume=True,
                    resume_data={"activity-1": {"decision": "approved"}},
                ),
                _input(),
            )
        ]

    replay_suspension = replay_client.commands[0].suspensions[0]
    assert [
        segment.name for segment in replay_suspension.position.operation_path.segments
    ] == ["agent", "inner", "left"]
    assert replay_suspension.position == first_suspension.position
    assert _ContinuationLlm.calls == 1
    assert [event.event for event in replacement_events] == ["token", "result"]
    assert replacement_events[0].data["content"] == "continued"
    assert replacement_events[-1].data["response"] == "continued"
    complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_wait_without_node_path_fails_before_suspension_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wait = _request_input_event()
    wait.node_info = NodeInfo()
    run = _SuspendFixture(
        monkeypatch,
        turns=[[wait]],
        entries=_suspended_entries(),
    )

    with (
        attempt_context_scope(_attempt()),
        pytest.raises(UnsupportedDurableADKError, match="node path"),
    ):
        await run.collect()

    assert run.client.commands == []
    run.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_parallel_waits_finalize_as_one_stably_ordered_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _SuspendFixture(
        monkeypatch,
        turns=[
            [
                _parallel_request_input_event("native-right", "right@1"),
                _parallel_request_input_event("native-left", "left@1"),
            ]
        ],
        entries=_frontier_entries(
            [
                ("activity-left", ACTIVITY_OUTCOME_KIND_SUSPENDED, None),
                ("activity-right", ACTIVITY_OUTCOME_KIND_SUSPENDED, None),
            ]
        ),
        reverse_return=True,
    )

    with attempt_context_scope(_attempt()):
        events = await run.collect()

    assert [event.event for event in events] == ["suspend"]
    assert [item["id"] for item in events[0].data["interrupts"]] == [
        "activity-left",
        "activity-right",
    ]
    command = run.client.commands[0]
    assert len(command.suspensions) == 2
    assert [
        entry.position.operation_path.segments[-1].name for entry in command.suspensions
    ] == ["left", "right"]


@pytest.mark.asyncio
async def test_parallel_wait_hides_events_below_a_waiting_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_wait = _parallel_request_input_event("native-left", "left@1")
    speculative_left_output = _text_event("missing tool result")
    speculative_left_output.branch = "left@1.worker"
    visible_right_output = _text_event("right prefix")
    visible_right_output.branch = "right@1"
    run = _SuspendFixture(
        monkeypatch,
        turns=[
            [
                left_wait,
                speculative_left_output,
                _parallel_request_input_event("duplicate-left", "left@1"),
                visible_right_output,
                _parallel_request_input_event("native-right", "right@1"),
            ]
        ],
        entries=_frontier_entries(
            [
                ("activity-left", ACTIVITY_OUTCOME_KIND_SUSPENDED, None),
                ("activity-right", ACTIVITY_OUTCOME_KIND_SUSPENDED, None),
            ]
        ),
    )

    with attempt_context_scope(_attempt()):
        events = await run.collect()

    assert [event.event for event in events] == ["token", "suspend"]
    assert events[0].data["content"] == "right prefix"
    assert events[-1].data["response"] == "right prefix"


@pytest.mark.asyncio
async def test_parallel_replay_injects_branch_scoped_function_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _SuspendFixture(
        monkeypatch,
        turns=[
            [
                _parallel_request_input_event("native-right", "right@1"),
                _parallel_request_input_event("native-left", "left@1"),
            ],
            [_text_event("joined")],
        ],
        entries=_frontier_entries(
            [
                ("activity-left", ACTIVITY_OUTCOME_KIND_COMPLETED, {"answer": "L"}),
                ("activity-right", ACTIVITY_OUTCOME_KIND_COMPLETED, {"answer": "R"}),
            ]
        ),
        reverse_return=True,
    )

    with attempt_context_scope(_attempt(replay_mode=True)):
        events = await run.collect(
            ctx=_context(
                resume=True,
                resume_data={
                    "activity-left": {"answer": "L"},
                    "activity-right": {"answer": "R"},
                },
            )
        )

    assert [event.event for event in events] == ["token", "result"]
    assert run.runner.messages[1] is None
    response_events = [
        event
        for event in run.session_service.session.events
        if event.author == "user" and event.get_function_responses()
    ]
    assert [event.branch for event in response_events] == ["left@1", "right@1"]
    assert [
        (response.id, response.response)
        for event in response_events
        for response in event.get_function_responses()
    ] == [
        ("native-left", {"answer": "L"}),
        ("native-right", {"answer": "R"}),
    ]


@pytest.mark.asyncio
async def test_request_input_replay_injects_function_response_and_suppresses_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _SuspendFixture(
        monkeypatch,
        turns=[
            [_text_event("prefix"), _request_input_event()],
            [_text_event("approved")],
        ],
        entries=_completed_entries({"approved": True}),
    )
    durable_memory = MagicMock()
    workflow = object()
    monkeypatch.setattr(
        "agent_engine_sdk_adk.workflow.get_current_wrapper",
        lambda: SimpleNamespace(
            durable_memory=durable_memory,
            workflow=workflow,
        ),
    )
    monkeypatch.setattr(
        "agent_engine_sdk_adk.workflow.get_current_user_id", lambda: "user-1"
    )

    with attempt_context_scope(_attempt(replay_mode=True)):
        events = await run.collect(
            ctx=_context(
                resume=True,
                resume_data={"activity-1": {"approved": True}},
            )
        )

    assert [event.event for event in events] == ["token", "result"]
    assert events[0].data["content"] == "approved"
    result = events[-1]
    assert result.data["response"] == "approved"
    assert run.order == ["commit", "result"]
    run.complete.assert_awaited_once()
    resume_message = run.runner.messages[1]
    response = resume_message.parts[0].function_response
    assert response.id == "manager_approval"
    assert response.name == "adk_request_input"
    assert response.response == {"approved": True}
    original = resume_message.parts[0]
    assert original.function_response is not None
    assert run.runner.messages[0].parts[0].text == "please review"
    assert [message["content"] for message in result.data["messages"]] == ["approved"]
    durable_memory.synchronize_tool.assert_called_once()
    memory_call = durable_memory.synchronize_tool.call_args
    assert memory_call.args[0] is workflow
    assert memory_call.args[1].workflow_identity == _attempt().workflow_identity
    assert memory_call.args[1].activity_id == "activity-1"
    assert memory_call.args[1].attempt_id == "attempt-1"
    assert memory_call.args[1].fencing_token == 7
    assert memory_call.args[2] == {"approved": True}
    assert memory_call.kwargs == {
        "user_id": "user-1",
        "tool_call_id": None,
        "tool_name": "adk_request_input",
    }


@pytest.mark.asyncio
async def test_wrapped_request_input_function_response_uses_domain_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _SuspendFixture(
        monkeypatch,
        turns=[
            [_text_event("prefix"), _human_review_event()],
            [_text_event("resolved")],
        ],
        entries=_completed_entries({"decision": "approved", "reviewer_notes": "ok"}),
    )

    with attempt_context_scope(_attempt(replay_mode=True)):
        events = await run.collect(
            ctx=_context(
                resume=True,
                resume_data={
                    "activity-1": {
                        "decision": "approved",
                        "reviewer_notes": "ok",
                    }
                },
            )
        )

    assert [event.event for event in events] == ["token", "result"]
    response = run.runner.messages[1].parts[0].function_response
    assert response.id == "review-1"
    assert response.name == "human_review"
    assert response.response == {
        "decision": "approved",
        "reviewer_notes": "ok",
    }


@pytest.mark.asyncio
async def test_confirmation_approve_replays_without_reemitting_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _SuspendFixture(
        monkeypatch,
        turns=[
            [_text_event("asking"), _confirmation_event()],
            [_text_event("booked")],
        ],
        entries=_completed_entries({"confirmed": True, "payload": None}),
    )

    with attempt_context_scope(_attempt(replay_mode=True)):
        events = await run.collect(
            ctx=_context(
                resume=True,
                resume_data={"activity-1": {"confirmed": True, "payload": None}},
            )
        )

    assert [event.data["content"] for event in events if event.event == "token"] == [
        "booked"
    ]
    response = run.runner.messages[1].parts[0].function_response
    assert response.name == "adk_request_confirmation"
    assert response.response == {"confirmed": True, "payload": None}
    assert run.order == ["commit", "result"]


@pytest.mark.asyncio
async def test_confirmation_reject_keeps_native_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _SuspendFixture(
        monkeypatch,
        turns=[
            [_confirmation_event()],
            [_text_event("cancelled")],
        ],
        entries=_completed_entries({"confirmed": False}),
    )

    with attempt_context_scope(_attempt(replay_mode=True)):
        events = await run.collect()

    response = run.runner.messages[1].parts[0].function_response
    assert response.response == {"confirmed": False, "payload": None}
    assert events[-1].data["response"] == "cancelled"


@pytest.mark.asyncio
async def test_resolved_frontier_advances_to_a_second_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _SuspendFixture(
        monkeypatch,
        turns=[
            [_request_input_event()],
            [
                _text_event("Please answer the next question."),
                _parallel_request_input_event("native-next", "next@1"),
            ],
        ],
        entries=_completed_entries({"approved": True}),
        additional_entries=[
            _frontier_entries([("activity-2", ACTIVITY_OUTCOME_KIND_SUSPENDED, None)])
        ],
    )

    with attempt_context_scope(_attempt(replay_mode=True)):
        events = await run.collect(
            ctx=_context(
                resume=True,
                resume_data={"activity-1": {"approved": True}},
            )
        )

    assert [event.event for event in events] == ["token", "suspend"]
    assert events[0].data["content"] == "Please answer the next question."
    assert events[1].data["interrupts"][0]["id"] == "activity-2"
    suspension_commands = [
        command for command in run.client.commands if command.suspensions
    ]
    assert [command.step_ordinal for command in suspension_commands] == [1, 2]
    assert [bool(command.suspensions) for command in run.client.commands] == [
        True,
        False,
        True,
    ]
    run.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_replay_suppresses_every_completed_frontier_before_current_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _SuspendFixture(
        monkeypatch,
        turns=[
            [_request_input_event()],
            [
                _text_event("already emitted"),
                _parallel_request_input_event("native-second", "second@1"),
            ],
            [_text_event("new suffix")],
        ],
        entries=_completed_entries({"approved": True}),
        additional_entries=[
            _frontier_entries(
                [
                    (
                        "activity-2",
                        ACTIVITY_OUTCOME_KIND_COMPLETED,
                        {"approved": True},
                    )
                ]
            )
        ],
    )

    with attempt_context_scope(_attempt(replay_mode=True)):
        events = await run.collect(
            ctx=_context(
                resume=True,
                resume_data={"activity-2": {"approved": True}},
            )
        )

    assert [event.event for event in events] == ["token", "result"]
    assert events[0].data["content"] == "new suffix"
    suspension_commands = [
        command for command in run.client.commands if command.suspensions
    ]
    assert [command.step_ordinal for command in suspension_commands] == [1, 2]
    assert [bool(command.suspensions) for command in run.client.commands] == [
        True,
        False,
        True,
        False,
    ]


@pytest.mark.asyncio
async def test_replacement_replay_never_reemits_historical_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _SuspendFixture(
        monkeypatch,
        turns=[
            [_request_input_event()],
            [
                _text_event("already emitted between frontiers"),
                _parallel_request_input_event("native-second", "second@1"),
            ],
            [_text_event("already emitted after the second frontier")],
        ],
        entries=_completed_entries({"approved": True}),
        additional_entries=[
            _frontier_entries(
                [
                    (
                        "activity-2",
                        ACTIVITY_OUTCOME_KIND_COMPLETED,
                        {"approved": True},
                    )
                ]
            )
        ],
    )

    with attempt_context_scope(_attempt(replay_mode=True)):
        events = await run.collect()

    assert [event.event for event in events] == ["result"]
    assert events[0].data["response"] == "already emitted after the second frontier"


@pytest.mark.asyncio
async def test_malformed_confirmation_continue_fails_as_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _SuspendFixture(
        monkeypatch,
        turns=[[_confirmation_event()]],
        entries=_completed_entries("yes"),
    )

    with (
        attempt_context_scope(_attempt(replay_mode=True)),
        pytest.raises(UnsupportedDurableADKError, match="confirmed"),
    ):
        await run.collect()

    run.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_invoke_returns_suspended_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _SuspendFixture(
        monkeypatch,
        turns=[[_text_event("I need a reviewer."), _request_input_event()]],
        entries=_suspended_entries(),
    )

    with attempt_context_scope(_attempt()):
        result = await run.agent.invoke(_context(), _input())

    assert result.response["status"] == "suspended"
    assert result.response["response"] == "I need a reviewer."
    assert result.response["interrupts"][0]["id"] == "activity-1"


@pytest.mark.asyncio
async def test_colocated_wait_text_is_kept_on_invoke_and_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_run = _SuspendFixture(
        monkeypatch,
        turns=[[_request_input_event_with_text("I need a reviewer.")]],
        entries=_suspended_entries(),
    )
    with attempt_context_scope(_attempt()):
        events = await stream_run.collect()

    invoke_run = _SuspendFixture(
        monkeypatch,
        turns=[[_request_input_event_with_text("I need a reviewer.")]],
        entries=_suspended_entries(),
    )
    with attempt_context_scope(_attempt()):
        result = await invoke_run.agent.invoke(_context(), _input())

    assert [event.event for event in events] == ["token", "suspend"]
    assert events[0].data["content"] == "I need a reviewer."
    assert events[-1].data["response"] == "I need a reviewer."
    assert result.response["response"] == "I need a reviewer."
