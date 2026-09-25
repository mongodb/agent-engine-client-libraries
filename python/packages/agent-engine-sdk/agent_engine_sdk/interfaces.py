"""Framework-neutral protocols for Atlas Agent Engine."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Generator
from typing import Any, Protocol, runtime_checkable

from agent_engine_sdk.models import (
    AgentInput,
    AgentOutput,
    LLMResponse,
    LLMStreamChunk,
    Message,
    RequestContext,
    StreamEvent,
)


@runtime_checkable
class ExecutionResult(AsyncIterable[StreamEvent], Protocol):
    """Result of an agent execution.

    Await for the final result, or async-iterate for streaming events.

    Usage::

        # Non-streaming
        output = await agent.execute(ctx, input)

        # Streaming
        async for event in agent.execute(ctx, input):
            handle(event)
    """

    def __await__(self) -> Generator[Any, None, AgentOutput]: ...

    def __aiter__(self) -> AsyncIterator[StreamEvent]: ...


@runtime_checkable
class BaseAgent(Protocol):
    """The contract between agent code and runtimes.

    Agents expose a single ``execute()`` method that returns a ``ExecutionResult``.
    Callers choose the execution mode:

    - ``await agent.execute(ctx, input)`` for a final result (JSON-style)
    - ``async for event in agent.execute(ctx, input)`` for streaming (SSE-style)
    """

    def execute(self, ctx: RequestContext, input: AgentInput) -> ExecutionResult: ...


@runtime_checkable
class BaseLLM(Protocol):
    """Framework-neutral LLM protocol.

    Used at runtime by the ToolPod.
    """

    def invoke(self, messages: list[Message], **kwargs: object) -> LLMResponse: ...

    async def ainvoke(
        self, messages: list[Message], **kwargs: object
    ) -> LLMResponse: ...

    def astream(
        self, messages: list[Message], **kwargs: object
    ) -> AsyncIterator[LLMStreamChunk]: ...


@runtime_checkable
class BaseExecutionCallback(Protocol):
    """Framework-neutral callback for observability during agent execution.

    Replaces LangChain's ``BaseCallbackHandler``. The AER injects an
    implementation that forwards node events to the OE for logging.
    Framework adapters (e.g. sdk-langgraph) bridge from the framework's
    native callback system to this protocol.

    Prefer subclassing ``NullExecutionCallback`` over implementing this
    Protocol directly — it provides no-op defaults for all methods including
    optional extensions (e.g. ``on_node_suspend``) so new methods never break
    existing implementations.
    """

    def on_node_start(
        self,
        node_name: str,
        inputs: dict[str, Any],
        *,
        run_id: str,
        parent_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def on_node_end(
        self,
        node_name: str,
        outputs: dict[str, Any],
        *,
        run_id: str,
        parent_run_id: str | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def on_node_error(
        self,
        node_name: str,
        error: str,
        *,
        run_id: str,
        parent_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...


class NullExecutionCallback:
    """Concrete base class with no-op defaults for all BaseExecutionCallback methods.

    Subclass this instead of implementing BaseExecutionCallback directly so that
    new methods added to the protocol are automatically satisfied — you only
    override the events you care about.

    Example:

    ```python
    class MyCallback(NullExecutionCallback):
        def on_node_start(self, node_name, inputs, *, run_id, **kwargs):
            print(f"starting {node_name}")
    ```
    """

    def on_node_start(
        self,
        node_name: str,
        inputs: dict[str, Any],
        *,
        run_id: str,
        parent_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pass

    def on_node_end(
        self,
        node_name: str,
        outputs: dict[str, Any],
        *,
        run_id: str,
        parent_run_id: str | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pass

    def on_node_error(
        self,
        node_name: str,
        error: str,
        *,
        run_id: str,
        parent_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pass

    def on_node_suspend(
        self,
        node_name: str,
        *,
        run_id: str,
        parent_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pass
