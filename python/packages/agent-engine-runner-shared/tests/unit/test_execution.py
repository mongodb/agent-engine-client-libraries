from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from agent_engine_sdk import AgentOutput, StreamEvent

from agent_engine_runner_shared.execution import AgentExecutionResult


@pytest.mark.asyncio
async def test_execution_result_supports_await_and_stream() -> None:
    async def invoke() -> AgentOutput:
        return AgentOutput(response="done")

    async def stream() -> AsyncIterator[StreamEvent]:
        yield StreamEvent(event="token", data={"content": "hello"})

    result = AgentExecutionResult(invoke, stream)

    assert await result == AgentOutput(response="done")
    assert [event async for event in result] == [
        StreamEvent(event="token", data={"content": "hello"})
    ]


@pytest.mark.asyncio
async def test_closing_the_execution_stream_closes_the_adapter_stream() -> None:
    # The AER closes the iterator it got from execute(); the adapter's cleanup
    # must run then, not whenever its generator is garbage collected.
    closed: list[bool] = []

    async def invoke() -> AgentOutput:
        return AgentOutput(response="unused")

    async def stream() -> AsyncIterator[StreamEvent]:
        try:
            yield StreamEvent(event="token", data={"content": "a"})
            yield StreamEvent(event="token", data={"content": "b"})
        finally:
            closed.append(True)

    events = AgentExecutionResult(invoke, stream).__aiter__()
    await anext(events)
    await events.aclose()

    assert closed == [True]
