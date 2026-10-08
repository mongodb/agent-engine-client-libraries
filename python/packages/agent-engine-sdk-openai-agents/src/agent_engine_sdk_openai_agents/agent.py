"""Run one native OpenAI Agents turn with OE as its durable authority."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

from agent_engine_sdk import (
    AgentInput,
    AgentOutput,
    BaseAgent,
    ExecutionResult,
    RequestContext,
    StreamEvent,
)
from agents import Agent, RunConfig, Runner, RunState
from agents.result import RunResultStreaming
from openai.types.responses import ResponseTextDeltaEvent

from agent_engine_runner_shared.execution import AgentExecutionResult
from agent_engine_runner_shared.workflow import current_attempt_context
from agent_engine_runner_shared.workflow.settlement import (
    SettledExecutionError,
    settle_execution,
)
from agent_engine_sdk_openai_agents.approvals import (
    ApprovalDecision,
    PendingApproval,
    approval_call_id,
    record_approvals,
    suspend_event,
)
from agent_engine_sdk_openai_agents.errors import (
    DurableOpenAIAgentsStateError,
    UnsupportedDurableOpenAIAgentsError,
)
from agent_engine_sdk_openai_agents.graph import agent_graph
from agent_engine_sdk_openai_agents.items import to_platform_messages, validate_items
from agent_engine_sdk_openai_agents.state import (
    decode_active_agent,
    decode_state,
    encode_state,
)
from agent_engine_sdk_openai_agents.tools import (
    ToolCallFailed,
    TurnContext,
    json_text,
)

__all__ = ["OpenAIAgentsBaseAgent"]


class OpenAIAgentsBaseAgent(BaseAgent):
    """Continue the committed conversation with one native Runner turn."""

    def __init__(
        self,
        agent: Agent[Any],
        *,
        app_name: str,
        registered_llms: Mapping[str, object],
        tool_issuer: object,
    ) -> None:
        self._agent = agent
        self._app_name = app_name
        self._registered_llms = registered_llms
        self._tool_issuer = tool_issuer

    def execute(self, ctx: RequestContext, input: AgentInput) -> ExecutionResult:
        return AgentExecutionResult(
            invoke=lambda: self.invoke(ctx, input),
            stream=lambda: self.stream(ctx, input),
        )

    async def invoke(self, ctx: RequestContext, input: AgentInput) -> AgentOutput:
        response: str | None = None
        async for event in self.stream(ctx, input):
            if event.event == "suspend" and isinstance(event.data, dict):
                data = event.data
                return AgentOutput(
                    response={
                        "status": "suspended",
                        "response": "",
                        "resumed": ctx.resume,
                        "interrupts": data["interrupts"],
                        "resume_schema": data["resume_schema"],
                        "suspend_payload": data["suspend_payload"],
                    }
                )
            if event.event == "result" and isinstance(event.data, dict):
                response = str(event.data["response"])
        if response is None:
            raise UnsupportedDurableOpenAIAgentsError("turn ended without a result")
        return AgentOutput(
            response={
                "status": "completed",
                "response": response,
                "resumed": ctx.resume,
            }
        )

    async def stream(
        self, ctx: RequestContext, input: AgentInput
    ) -> AsyncIterator[StreamEvent]:
        # Validated per turn rather than at build: the AER builds the agent
        # before its execution error handling, where a rejection would reach
        # the caller as an opaque failure instead of this explanation.
        agents = agent_graph(
            self._agent,
            registered_llms=self._registered_llms,
            tool_issuer=self._tool_issuer,
        )
        attempt = current_attempt_context()
        if attempt is None:
            raise UnsupportedDurableOpenAIAgentsError(
                "OpenAI Agents runs require an OE-issued durable workflow attempt"
            )
        # A durable resume replays the original turn; the answers themselves
        # come back from OE at the wait they resolve.
        resume_ids = _resume_activity_ids(ctx)
        prior = decode_state(attempt, app_name=self._app_name)
        # After a handoff, the agent that ended the last turn starts this one.
        active = decode_active_agent(attempt, app_name=self._app_name)
        starting_agent = self._agent if active is None else agents.get(active)
        if starting_agent is None:
            raise DurableOpenAIAgentsStateError(
                f"the conversation's active agent {active!r} is not part of this app"
            )
        run_input: list[Any] | RunState[Any] = [
            *(item.raw for item in prior),
            {"role": "user", "content": _user_message(input)},
        ]
        # The caller already saw the replayed prefix; show output again only
        # after the run passes the wait this resume answers.
        show_tokens = not resume_ids
        turn = TurnContext()
        while True:
            result = Runner.run_streamed(
                starting_agent,
                run_input,
                context=turn,
                run_config=RunConfig(
                    tracing_disabled=True,
                    # The agent graph already refuses shared names. Should
                    # one get past it, the SDK's default would keep one tool
                    # or handoff and drop the other with only a warning.
                    tool_name_collision_policy="error",
                ),
            )
            committed = False
            try:
                async for event in _turn_events(result, turn):
                    if isinstance(event, StreamEvent):
                        # A nested agent's progress, already in caller form.
                        if show_tokens:
                            yield event
                    elif (
                        show_tokens
                        and event.type == "raw_response_event"
                        and isinstance(event.data, ResponseTextDeltaEvent)
                    ):
                        yield StreamEvent(
                            event="token",
                            data={
                                "content": event.data.delta,
                                "source": "",
                                "tool_call_id": "",
                            },
                        )
                # A cancelled run loop ends the event stream quietly, without
                # an error or a final output; committing then would record a
                # turn that never finished.
                run_loop = result.run_loop_task
                if run_loop is None or run_loop.cancelled():
                    raise UnsupportedDurableOpenAIAgentsError(
                        "the OpenAI Agents run was cancelled before it finished"
                    )
                # A handoff may have changed who is active; the next turn can
                # only start from an agent this app's graph validated.
                active_agent = result.last_agent
                if agents.get(active_agent.name) is not active_agent:
                    raise UnsupportedDurableOpenAIAgentsError(
                        "the turn ended on an agent outside the app's agent graph"
                    )
                interruptions = list(result.interruptions)
                if interruptions:
                    # The batch's pending calls have no output yet, so this is
                    # not a validated item list; only the call order is read.
                    raw = cast(list[dict[str, Any]], result.to_input_list())
                    call_order = [
                        str(item.get("call_id"))
                        for item in raw
                        if item.get("type") == "function_call"
                    ]
                    waits = await record_approvals(attempt, interruptions, call_order)
                    if isinstance(waits[0], PendingApproval):
                        yield suspend_event(
                            cast(tuple[PendingApproval, ...], waits),
                            resumed=ctx.resume,
                        )
                        return
                    # Every wait in the frontier is answered; apply each answer
                    # so the SDK resumes without asking again.
                    decisions = cast(tuple[ApprovalDecision, ...], waits)
                    if any(d.activity_id in resume_ids for d in decisions):
                        show_tokens = True
                    by_call = {approval_call_id(i): i for i in interruptions}
                    state = result.to_state()
                    for decision in decisions:
                        item = by_call[decision.call_id]
                        if decision.approved:
                            state.approve(item)
                        else:
                            state.reject(
                                item, rejection_message=decision.rejection_message
                            )
                    run_input = state
                    continue
                items = validate_items(result.to_input_list())
                response = json_text(result.final_output)
                try:
                    await settle_execution(
                        attempt,
                        encode_state(
                            items,
                            app_name=self._app_name,
                            session_id=attempt.workflow_identity.session_id,
                            active_agent=active_agent.name,
                        ),
                    )
                except SettledExecutionError as error:
                    raise UnsupportedDurableOpenAIAgentsError(str(error)) from error
                committed = True
                messages = to_platform_messages(items[len(prior) :])
                yield StreamEvent(
                    event="result",
                    data={
                        "response": response,
                        "messages": [
                            m.model_dump(mode="json", exclude_none=True)
                            for m in messages
                        ],
                        "message_count": len(messages),
                        "resumed": ctx.resume,
                    },
                )
                return
            finally:
                # An abandoned or failed turn must not keep running model or
                # tool work in the background after OE stops waiting for it.
                if not committed and not result.is_complete:
                    result.cancel()
                await turn.drain()


def _resume_activity_ids(ctx: RequestContext) -> frozenset[str]:
    if not ctx.resume:
        return frozenset()
    if not isinstance(ctx.resume_data, dict) or not ctx.resume_data:
        raise UnsupportedDurableOpenAIAgentsError(
            "an OpenAI Agents resume must answer its pending approval in resume_map"
        )
    return frozenset(ctx.resume_data)


async def _turn_events(
    result: RunResultStreaming, turn: TurnContext
) -> AsyncIterator[Any]:
    """The run's own events, with nested agents' progress as it arrives.

    The SDK's stream is silent while a tool call runs, which is when a nested
    agent produces output, so both are read from the turn's one queue.
    """
    finished = object()
    reading = True

    async def forward_run_events() -> None:
        try:
            async for event in result.stream_events():
                turn.events.put_nowait(event)
            turn.events.put_nowait(finished)
        except ToolCallFailed as failure:
            # The platform tool path's own exception, carried through the SDK.
            turn.events.put_nowait(_RunFailed(failure.error))
        except Exception as error:
            turn.events.put_nowait(_RunFailed(error))
        except BaseException as error:
            # The SDK's stream can end in a cancellation this task did not
            # ask for; ending without a marker would leave the reader waiting.
            if reading:
                if isinstance(error, asyncio.CancelledError):
                    error = UnsupportedDurableOpenAIAgentsError(
                        "the OpenAI Agents run was cancelled before it finished"
                    )
                turn.events.put_nowait(_RunFailed(error))
            raise

    forwarding = asyncio.create_task(forward_run_events())
    try:
        while True:
            event = await turn.events.get()
            if event is finished:
                return
            if isinstance(event, _RunFailed):
                raise event.error from None
            yield event
    finally:
        reading = False
        forwarding.cancel()
        await asyncio.gather(forwarding, return_exceptions=True)


@dataclass(frozen=True)
class _RunFailed:
    error: BaseException


def _user_message(input: AgentInput) -> str:
    payload = input.payload
    message = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(message, str):
        raise TypeError("OpenAI Agents input payload.message must be a string")
    return message
