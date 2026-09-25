"""ADK serial streaming and durable terminal-fence tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_engine_sdk import AgentInput, Message, RequestContext
from google.adk.events import Event
from google.adk.sessions import Session
from google.genai import types

from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import WorkflowIdentity
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.memory import extract_turn_messages
from agent_engine_sdk_adk.agent import ADKBaseAgent, UnsupportedDurableADKError
from agent_engine_sdk_adk.platform_session import session_to_state_snapshot


def _context(*, user_id: str | None = "user-1") -> RequestContext:
    return RequestContext(session_id="session-1", user_id=user_id)


def _input() -> AgentInput:
    return AgentInput(payload={"message": "weather in Paris?"})


def _attempt(previous_session: Session | None = None) -> AttemptContext:
    attempt = AttemptContext(
        attempt_id="attempt-1",
        fencing_token=7,
        workflow_identity=WorkflowIdentity(
            session_id="session-1", execution_id="execution-1"
        ),
    )
    if previous_session is not None:
        attempt.previous_state.CopyFrom(session_to_state_snapshot(previous_session))
    return attempt


def _text_event(text: str, *, partial: bool = False) -> Event:
    return Event(
        invocation_id="invocation-1",
        author="agent",
        partial=partial,
        content=types.Content(role="model", parts=[types.Part.from_text(text=text)]),
    )


def _function_call_event(*, long_running: bool = False) -> Event:
    return Event(
        invocation_id="invocation-1",
        author="agent",
        long_running_tool_ids=["call-1"] if long_running else [],
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="call-1", name="weather", args={"city": "Paris"}
                    )
                )
            ],
        ),
    )


def _function_response_event() -> Event:
    return Event(
        invocation_id="invocation-1",
        author="agent",
        content=types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="call-1", name="weather", response={"temp": 22}
                    )
                )
            ],
        ),
    )


class _Runner:
    def __init__(
        self,
        *,
        events: list[Event],
        session_service: Any,
        failure: Exception | None = None,
    ) -> None:
        self._events = events
        self._session_service = session_service
        self._failure = failure

    async def run_async(
        self, *, user_id: str, session_id: str, new_message: Any
    ) -> AsyncIterator[Event]:
        if self._failure is not None:
            raise self._failure
        session = await self._session_service.get_session(
            app_name="app", user_id=user_id, session_id=session_id
        )
        assert session is not None
        for event in self._events:
            await self._session_service.append_event(session, event)
            yield event


class _Fixture:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        events: list[Event],
        *,
        failure: Exception | None = None,
        attempt: AttemptContext | None = None,
    ) -> None:
        self.complete = AsyncMock()
        self.order: list[str] = []

        def _runner(**kwargs: Any) -> _Runner:
            self.session_service = kwargs["session_service"]
            return _Runner(
                events=events,
                session_service=kwargs["session_service"],
                failure=failure,
            )

        async def _complete(_attempt_context: AttemptContext, session: Session) -> None:
            self.order.append("commit")
            await self.complete(_attempt_context, session)

        monkeypatch.setattr("google.adk.runners.Runner", _runner)
        monkeypatch.setattr(
            "agent_engine_sdk_adk.execution_session.current_attempt_context",
            lambda: attempt or _attempt(),
        )
        self.agent = ADKBaseAgent(adk_agent=MagicMock(), app_name="app")
        monkeypatch.setattr(
            "agent_engine_sdk_adk.agent.complete_durable_execution", _complete
        )

    async def collect(self, *, ctx: RequestContext | None = None) -> list[Any]:
        result = []
        async for event in self.agent.stream(ctx or _context(), _input()):
            if event.event == "result":
                self.order.append("result")
            result.append(event)
        return result

    async def terminal(self) -> Any:
        events = await self.collect()
        terminals = [event for event in events if event.event == "result"]
        assert len(terminals) == 1
        return terminals[0]


@pytest.fixture
def fixture(monkeypatch: pytest.MonkeyPatch) -> Any:
    return lambda events, failure=None, attempt=None: _Fixture(
        monkeypatch, events, failure=failure, attempt=attempt
    )


@pytest.mark.asyncio
async def test_result_contains_full_turn_in_order(fixture: Any) -> None:
    terminal = await fixture(
        [_function_call_event(), _function_response_event(), _text_event("22C")]
    ).terminal()

    messages = [Message.model_validate(value) for value in terminal.data["messages"]]
    assert [message.role for message in messages] == [
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[0].tool_calls is not None
    assert messages[0].tool_calls[0].id == "call-1"
    assert messages[1].tool_call_id == "call-1"
    assert messages[2].content == "22C"
    assert terminal.data["response"] == "22C"
    assert terminal.data["message_count"] == 3


@pytest.mark.asyncio
async def test_partial_tokens_stream_but_only_final_event_is_persisted(
    fixture: Any,
) -> None:
    run = fixture(
        [
            _text_event("2", partial=True),
            _text_event("2C", partial=True),
            _text_event("22C"),
        ]
    )

    events = await run.collect()

    assert [event.data["content"] for event in events if event.event == "token"] == [
        "2",
        "2C",
        "22C",
    ]
    terminal = next(event for event in events if event.event == "result")
    assert terminal.data["response"] == "22C"
    assert [message["content"] for message in terminal.data["messages"]] == ["22C"]
    committed_session = run.complete.await_args.args[1]
    assert [event.content.parts[0].text for event in committed_session.events] == [
        "22C"
    ]


@pytest.mark.asyncio
async def test_invoke_returns_only_the_final_non_partial_response(fixture: Any) -> None:
    run = fixture(
        [
            _text_event("2", partial=True),
            _text_event("2C", partial=True),
            _text_event("22C"),
        ]
    )

    result = await run.agent.invoke(_context(), _input())

    assert result.response["response"] == "22C"


@pytest.mark.asyncio
async def test_commit_completes_before_result_is_emitted(fixture: Any) -> None:
    run = fixture([_text_event("done")])

    await run.collect()

    assert run.order == ["commit", "result"]
    run.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_user_id_fails_before_adk_runner(
    fixture: Any,
) -> None:
    previous = Session(
        id="session-1",
        app_name="app",
        user_id="user-1",
        state={"turn": 1, "customer": {"name": "Ada"}},
    )
    run = fixture([_text_event("welcome back")], attempt=_attempt(previous))

    with pytest.raises(UnsupportedDurableADKError, match="user_id"):
        await run.collect(ctx=_context(user_id=None))

    run.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_runner_failure_does_not_commit(fixture: Any) -> None:
    run = fixture([], failure=RuntimeError("provider failed"))

    with pytest.raises(RuntimeError, match="provider failed"):
        await run.collect()

    run.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_result_messages_produce_one_stm_turn(fixture: Any) -> None:
    terminal = await fixture(
        [_function_call_event(), _function_response_event(), _text_event("22C")]
    ).terminal()
    result_messages = [
        Message.model_validate(value) for value in terminal.data["messages"]
    ]

    stm = extract_turn_messages(
        message="weather in Paris?",
        result_messages=result_messages,
        user_id="user-1",
    )

    assert [message["role"] for message in stm] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
