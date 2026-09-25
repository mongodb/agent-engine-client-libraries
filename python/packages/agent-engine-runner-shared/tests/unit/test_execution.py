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
