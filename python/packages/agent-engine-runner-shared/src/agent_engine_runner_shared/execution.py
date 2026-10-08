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
        # When the caller closes this iterator, close the adapter's stream now
        # rather than at garbage collection, so its cleanup (cancelling
        # framework work, releasing resources) runs while the request is live.
        stream = self._stream()
        try:
            async for event in stream:
                yield event
        finally:
            # An async generator has aclose(); a plain async iterator need not.
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()
