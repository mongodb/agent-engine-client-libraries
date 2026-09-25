"""Concrete execution result shared by framework adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Coroutine, Generator
from typing import Any

from agent_engine_sdk import AgentOutput, StreamEvent


class AgentExecutionResult:
    """Expose one agent execution as either an awaitable or an async stream."""

    def __init__(
        self,
        invoke: Callable[[], Coroutine[Any, Any, AgentOutput]],
        stream: Callable[[], AsyncIterator[StreamEvent]],
    ) -> None:
        self._invoke = invoke
        self._stream = stream

    def __await__(self) -> Generator[Any, None, AgentOutput]:
        return self._invoke().__await__()

    async def __aiter__(self) -> AsyncIterator[StreamEvent]:
        async for event in self._stream():
            yield event
