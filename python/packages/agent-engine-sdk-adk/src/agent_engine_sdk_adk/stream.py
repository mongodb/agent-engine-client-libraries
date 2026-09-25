"""Stream ADK Runner events as Atlas Agent Engine tokens, suspend, and result events.

On continue, suppress messages from before the wait and emit only the
messages that follow resume.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import aclosing
from typing import Any, Literal

from agent_engine_sdk import AgentInput, Message, StreamEvent
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import Session
from google.adk.sessions.base_session_service import BaseSessionService
from google.genai import types
from pydantic import BaseModel, Field

from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_sdk_adk.errors import UnsupportedDurableADKError
from agent_engine_sdk_adk.execution_session import DurableSession
from agent_engine_sdk_adk.messages import content_to_platform_messages
from agent_engine_sdk_adk.route import DurableRouteAdapter
from agent_engine_sdk_adk.suspend import (
    AdkSuspend,
    SuspendEventData,
    SuspendTranslationError,
    extract_suspends,
    function_response_frontier_content,
    suspend_frontier_stream_event,
)
from agent_engine_sdk_adk.workflow import FreshWaitFrontier, resolve_wait_frontier

__all__ = [
    "ResultEventData",
    "stream_invocation",
    "terminal_from_stream_event",
]

CompleteExecution = Callable[[AttemptContext, Session], Coroutine[Any, Any, None]]


class ResultEventData(BaseModel):
    """Payload of Atlas Agent Engine ``StreamEvent(event='result')``."""

    response: str = Field(description="Concatenated non-partial model text.")
    messages: list[Message] = Field(description="Visible ADK messages in this turn.")
    message_count: int = Field(description="len(messages); kept for AER consumers.")
    resumed: Literal[False] = Field(
        default=False,
        description=(
            "LangGraph native HITL sets this when ctx.resume continues a "
            "checkpoint. Durable continue is a new attempt, so this stays false."
        ),
    )


class AdkTurn:
    """One ``Runner.run_async`` through completion or runner quiescence."""

    def __init__(
        self,
        runner: Runner,
        *,
        user_id: str,
        session_id: str,
        new_message: types.Content | None,
        invocation_id: str | None = None,
        route_adapter: DurableRouteAdapter,
    ) -> None:
        self._runner = runner
        self._user_id = user_id
        self._session_id = session_id
        self._new_message = new_message
        self._invocation_id = invocation_id
        self._route_adapter = route_adapter
        self.messages: list[Message] = []
        self.final_response = ""
        self._waits: list[AdkSuspend] = []

    async def stream(self, *, emit_tokens: bool) -> AsyncIterator[StreamEvent]:
        """Run ADK until completion or all parallel routes become quiescent.

        ``emit_tokens`` controls whether Atlas Agent Engine ``token`` events go to the
        caller. The turn still runs either way (needed to reconstruct the
        wait). False while replaying messages from before the suspend, because
        the caller already received them. True for a fresh wait and for
        messages after resume.
        """
        run_kwargs: dict[str, Any] = {
            "user_id": self._user_id,
            "session_id": self._session_id,
            "new_message": self._new_message,
        }
        if self._invocation_id is not None:
            run_kwargs["invocation_id"] = self._invocation_id
        with self._route_adapter.operation_path_scope():
            async with aclosing(self._runner.run_async(**run_kwargs)) as events:
                async for event in events:
                    after_wait = any(
                        _branch_at_or_below(event.branch or "", wait.provenance.branch)
                        for wait in self._waits
                    )
                    try:
                        operation_boundaries = (
                            self._route_adapter.operation_boundaries_for_event(event)
                        )
                        suspends = extract_suspends(
                            event,
                            operation_boundaries=operation_boundaries or (),
                        )
                    except (RuntimeError, SuspendTranslationError) as error:
                        raise UnsupportedDurableADKError(str(error)) from error
                    if suspends and operation_boundaries is None:
                        raise UnsupportedDurableADKError(
                            "Google ADK wait event is missing its node path"
                        )
                    if not after_wait:
                        self._waits.extend(suspends)
                    if after_wait:
                        continue
                    async for token in self._visible_tokens(event, emit_tokens):
                        yield token

    def wait_frontier(self) -> tuple[AdkSuspend, ...]:
        """Return collected waits in deterministic ADK route order."""
        return tuple(sorted(self._waits, key=lambda wait: wait.provenance.order_key))

    async def _visible_tokens(
        self, event: Event, emit_tokens: bool
    ) -> AsyncIterator[StreamEvent]:
        if not event.content or not event.content.parts or event.author == "user":
            return
        if not event.partial:
            self.messages.extend(content_to_platform_messages(event.content))
        for part in event.content.parts:
            text = getattr(part, "text", None)
            if not text:
                continue
            if not event.partial:
                self.final_response += str(text)
            token = StreamEvent(
                event="token",
                data={
                    "content": str(text),
                    "source": "",
                    "tool_call_id": "",
                },
            )
            if emit_tokens:
                yield token


def _branch_at_or_below(branch: str, ancestor: str) -> bool:
    """Return whether an ADK event belongs to a route already waiting."""
    return not ancestor or branch == ancestor or branch.startswith(f"{ancestor}.")


def result_event(final_response: str, messages: list[Message]) -> StreamEvent:
    data = ResultEventData(
        response=final_response,
        messages=messages,
        message_count=len(messages),
    )
    return StreamEvent(event="result", data=data.model_dump(mode="json"))


async def _append_frontier_response_events(
    session: Session,
    *,
    session_service: BaseSessionService,
    suspends: tuple[AdkSuspend, ...],
    results: tuple[Any, ...],
) -> str:
    """Append one branch-scoped ADK event per response in a wait frontier.

    ADK stamps one branch on a user event. A single event containing responses
    for sibling branches is therefore visible to only one branch, which leaves
    the other response orphaned when ADK builds its next model request.
    """
    if len(suspends) < 2 or len(suspends) != len(results):
        raise SuspendTranslationError(
            "split resume requires an aligned multi-wait frontier"
        )

    matching_events: list[Event] = []
    for suspend in suspends:
        matching_event = next(
            (
                event
                for event in reversed(session.events)
                if (event.branch or "") == suspend.provenance.branch
                and any(
                    call.id == suspend.function_call_id
                    for call in event.get_function_calls()
                )
            ),
            None,
        )
        if matching_event is None or not matching_event.invocation_id:
            raise SuspendTranslationError(
                "ADK wait response has no matching function call in session history"
            )
        matching_events.append(matching_event)

    invocation_ids = {event.invocation_id for event in matching_events}
    if len(invocation_ids) != 1:
        raise SuspendTranslationError(
            "ADK wait frontier responses must belong to one invocation"
        )

    for suspend, result, matching_event in zip(
        suspends, results, matching_events, strict=True
    ):
        await session_service.append_event(
            session,
            Event(
                invocation_id=matching_event.invocation_id,
                author="user",
                branch=matching_event.branch,
                isolation_scope=matching_event.isolation_scope,
                content=function_response_frontier_content([suspend], [result]),
            ),
        )
    return invocation_ids.pop()


async def stream_invocation(
    *,
    session: DurableSession,
    input: AgentInput,
    complete_execution: CompleteExecution,
) -> AsyncIterator[StreamEvent]:
    """Run one ADK invocation and yield Atlas Agent Engine stream events."""
    attempt = session.attempt
    emit_tokens = not attempt.replay_mode
    new_message = session.original_message(input)
    invocation_id: str | None = None

    while True:
        turn = AdkTurn(
            session.runner,
            user_id=session.user_id,
            session_id=session.session_id,
            new_message=new_message,
            invocation_id=invocation_id,
            route_adapter=session.route_adapter,
        )
        async for event in turn.stream(emit_tokens=emit_tokens):
            yield event
        wait_frontier = turn.wait_frontier()
        if not wait_frontier:
            await complete_execution(attempt, session.scratch.session)
            yield result_event(turn.final_response, turn.messages)
            return

        resolution = await resolve_wait_frontier(
            attempt,
            session.scratch.session,
            wait_frontier,
        )
        if isinstance(resolution, FreshWaitFrontier):
            yield suspend_frontier_stream_event(
                resolution.suspends,
                interrupt_ids=resolution.activity_ids,
                messages=turn.messages,
                response=turn.final_response,
            )
            return

        if (
            not emit_tokens
            and session.resume_activity_ids
            and frozenset(resolution.activity_ids) == session.resume_activity_ids
        ):
            emit_tokens = True
        try:
            if len(resolution.suspends) == 1:
                new_message = function_response_frontier_content(
                    resolution.suspends,
                    resolution.results,
                )
                invocation_id = None
            else:
                invocation_id = await _append_frontier_response_events(
                    session.scratch.session,
                    session_service=session.scratch,
                    suspends=resolution.suspends,
                    results=resolution.results,
                )
                new_message = None
        except SuspendTranslationError as error:
            raise UnsupportedDurableADKError(str(error)) from error


def terminal_from_stream_event(
    event: StreamEvent,
    *,
    result_text: str,
    suspend: SuspendEventData | None,
) -> tuple[str, SuspendEventData | None]:
    """Fold one stream event into invoke() terminal state."""
    if event.event == "suspend":
        data = SuspendEventData.model_validate(event.data)
        return data.response, data
    if event.event == "result":
        data = ResultEventData.model_validate(event.data)
        return data.response, suspend
    return result_text, suspend
