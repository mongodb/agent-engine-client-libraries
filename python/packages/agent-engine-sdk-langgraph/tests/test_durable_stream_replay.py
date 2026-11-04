"""A retried durable turn's stream, across the LangGraph adapter.

A model call that completed before the runner was lost is recorded. The
replacement attempt must reuse that record and restate it to the client as one
event, then stream only the lost call again. The provider call count is the
discriminator: the first prompt is sent once, the lost prompt twice.
"""

from __future__ import annotations

from typing import Iterator

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from agent_engine_sdk import AgentInput, RequestContext
from agent_engine_sdk.models import LLMStreamChunk
from agent_engine_sdk_langgraph import LangGraphBaseAgent, PlatformCheckpointer
from agent_engine_sdk_langgraph import durable_session as ds
from agent_engine_sdk_langgraph import platform_checkpointer as pc
from agent_engine_runner_shared.context import current_oe_url
from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_OUTCOME_KIND_FAILED,
)
from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
from agent_engine_runner_shared.secure_wrapper import OperationalStepAllocator
from agent_engine_runner_shared.workflow import attempt_context_scope
from test_durable_subgraphs_integration import (
    _FrontierClient,
    _ReplayClient,
    _attempt,
    _wrapped_llm,
)


# The runner dies while the second model call is streaming. The first call has
# already completed and reported its outcome.
class _RunnerLost(Exception):
    pass


@pytest.mark.anyio
async def test_retried_turn_restates_recorded_model_output_once(monkeypatch):
    client = _ReplayClient()
    wrapper, llm = _wrapped_llm(monkeypatch, client, [])
    calls: list[str] = []
    lost = {"second": True}

    class _LostDuringSecond(SecureLLMProxy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._workflow = client

        def _stream_llm(
            self, invoke_request, resolved_step
        ) -> Iterator[LLMStreamChunk]:
            prompt = str(invoke_request.messages[-1].content)
            calls.append(prompt)
            yield LLMStreamChunk(content=f"{prompt}-part1 ")
            if prompt == "second" and lost["second"]:
                raise _RunnerLost("runner killed")
            yield LLMStreamChunk(content=f"{prompt}-part2")

    monkeypatch.setattr(
        "agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy", _LostDuringSecond
    )
    frontier = _FrontierClient()
    monkeypatch.setattr(pc, "AsyncWorkflowClient", lambda _: frontier)
    monkeypatch.setattr(ds, "AsyncWorkflowClient", lambda _: frontier)

    async def first(state):
        return {"messages": [await llm.ainvoke([HumanMessage(content="first")])]}

    async def second(state):
        return {"messages": [await llm.ainvoke([HumanMessage(content="second")])]}

    graph = StateGraph(MessagesState)
    graph.add_node("first", first)
    graph.add_node("second", second)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    agent = LangGraphBaseAgent(
        graph.compile(checkpointer=PlatformCheckpointer(native=InMemorySaver()))
    )
    context = RequestContext(user_id="u", session_id="support-session")

    async def run(attempt) -> list[tuple[str, str | None]]:
        wrapper.operational_steps = OperationalStepAllocator()
        events: list[tuple[str, str | None]] = []
        with attempt_context_scope(attempt):
            try:
                async for event in agent.stream(
                    context, AgentInput(payload={"message": "hi"})
                ):
                    content = (
                        event.data.get("content")
                        if isinstance(event.data, dict)
                        else None
                    )
                    events.append((str(event.event), content))
            except _RunnerLost:
                # The runner is gone; the client receives the failed turn, not
                # another event.
                events.append(("EXC", "runner killed"))
        return events

    oe_url = current_oe_url.set("http://oe")
    try:
        first_attempt = await run(_attempt("attempt-1", 1, replay=False))
    finally:
        current_oe_url.reset(oe_url)

    assert ("token", "first-part1 ") in first_attempt
    assert ("token", "second-part1 ") in first_attempt

    # The runner died before the kernel recorded an outcome for the second
    # call, so only the completed first call is replayable.
    client._outcomes = {
        position: outcome
        for position, outcome in client._outcomes.items()
        if outcome.outcome_kind != ACTIVITY_OUTCOME_KIND_FAILED
    }
    lost["second"] = False

    oe_url = current_oe_url.set("http://oe")
    try:
        retried = await run(_attempt("attempt-2", 2, replay=True))
    finally:
        current_oe_url.reset(oe_url)

    tokens = [content for event, content in retried if event == "token" and content]
    assert tokens == [
        "first-part1 first-part2",
        "second-part1 ",
        "second-part2",
    ], (
        "the retried stream restates the recorded call once, then streams only the lost call"
    )
    assert calls == ["first", "second", "second"], (
        "the completed call must not reach the provider again; the lost call must"
    )
    assert ("result", None) in retried


@pytest.fixture
def anyio_backend():
    return "asyncio"
