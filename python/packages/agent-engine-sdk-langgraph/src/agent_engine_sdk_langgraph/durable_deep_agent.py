"""Durable operation-path boundaries for Deep Agent task delegation.

Deep Agent exposes subagent dispatch through the public ``task`` tool and
middleware hooks, rather than as a compiled child graph:
https://docs.langchain.com/oss/python/deepagents/subagents
https://docs.langchain.com/oss/python/langchain/middleware/custom

Same-step ``task`` siblings stamp ToolCall-id ordinals after ``FinalizeStep``.
Stamping in ``after_model`` would key those ordinals to the model step, which
FinalizeStep then leaves behind.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ToolCallRequest,
)
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from agent_engine_runner_shared.workflow import (
    ChildOperationBoundary,
    child_operation_boundary_scope,
    current_attempt_context,
    preallocate_child_operation_ordinals,
)
from agent_engine_runner_shared.workflow.context import (
    current_pending_child_operation_batch,
    set_pending_child_operation_batch,
)

from .deep_agent_task import parse_task_call
from .platform_checkpointer import UnsupportedDurableGraphError

ToolCallResult = ToolMessage | Command[Any]

_ACTIVE_TASK_CALL_ID: ContextVar[str | None] = ContextVar(
    "durable_deep_agent_task_call_id", default=None
)


@contextmanager
def _deep_agent_task_scope(tool_call_id: str) -> Iterator[None]:
    token = _ACTIVE_TASK_CALL_ID.set(tool_call_id)
    try:
        yield
    finally:
        _ACTIVE_TASK_CALL_ID.reset(token)


def _stamp_deep_agent_message_ids(messages: list[Any]) -> None:
    """Replace Deep Agent's generated message ids before LangGraph persists them."""
    task_call_id = _ACTIVE_TASK_CALL_ID.get()
    if task_call_id is None:
        return
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            message.id = f"durable-deep-agent-input:{task_call_id}:{index}"
        elif isinstance(message, ToolMessage):
            message.id = (
                f"durable-deep-agent-tool-result:{task_call_id}:{message.tool_call_id}"
            )


def _stamp_task_result_message_id(result: ToolCallResult, tool_call_id: str) -> None:
    if isinstance(result, ToolMessage):
        result.id = f"durable-deep-agent-task-result:{tool_call_id}"
        return
    update = result.update
    if not isinstance(update, dict):
        return
    messages = update.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if isinstance(message, ToolMessage) and message.tool_call_id == tool_call_id:
            message.id = f"durable-deep-agent-task-result:{tool_call_id}"


def _messages_from_state(state: Any) -> list[Any]:
    if isinstance(state, dict):
        messages = state.get("messages") or []
    elif isinstance(state, list):
        messages = state
    else:
        messages = getattr(state, "messages", None) or []
    return list(messages) if isinstance(messages, list) else []


class DurableDeepAgentMiddleware(AgentMiddleware[AgentState, Any]):
    """Attribute local Deep Agent task work to a deterministic child path."""

    def __init__(
        self,
        *,
        unsupported_subagent_names: Collection[str] = (),
        retry_policy_subagent_names: Collection[str] = (),
    ) -> None:
        self.unsupported_subagent_names = frozenset(unsupported_subagent_names)
        self.retry_policy_subagent_names = frozenset(retry_policy_subagent_names)

    def _boundary_from_task(self, tool_call: Any) -> ChildOperationBoundary | None:
        task_call = parse_task_call(tool_call)
        if task_call is None:
            return None
        if not task_call.tool_call_id:
            raise ValueError(
                "durable Deep Agent task requires a non-empty tool call ID"
            )
        if not task_call.subagent_name:
            raise ValueError(
                "durable Deep Agent task requires a non-empty subagent type"
            )
        if task_call.subagent_name in self.unsupported_subagent_names:
            raise UnsupportedDurableGraphError(
                f"compiled Deep Agent subagent {task_call.subagent_name!r} must use "
                "checkpointer=None; independent child checkpointers are not supported "
                "during durable execution"
            )
        if task_call.subagent_name in self.retry_policy_subagent_names:
            raise UnsupportedDurableGraphError(
                f"compiled Deep Agent subagent {task_call.subagent_name!r} sets a "
                "retry policy; node retry policies are not supported during "
                "durable execution"
            )
        return ChildOperationBoundary(
            name=task_call.subagent_name,
            occurrence_key=task_call.tool_call_id,
        )

    def _boundary(self, request: ToolCallRequest) -> ChildOperationBoundary | None:
        if current_attempt_context() is None:
            return None
        return self._boundary_from_task(request.tool_call)

    def _task_boundaries_from_state(
        self, state: AgentState
    ) -> tuple[ChildOperationBoundary, ...]:
        messages = _messages_from_state(state)
        if not messages:
            return ()
        tool_calls = getattr(messages[-1], "tool_calls", None) or []
        boundaries: list[ChildOperationBoundary] = []
        for tool_call in tool_calls:
            boundary = self._boundary_from_task(tool_call)
            if boundary is not None:
                boundaries.append(boundary)
        return tuple(boundaries)

    def _pending_boundaries(
        self, request: ToolCallRequest
    ) -> tuple[ChildOperationBoundary, ...]:
        from_state = self._task_boundaries_from_state(request.state)
        if from_state:
            return from_state
        return current_pending_child_operation_batch()

    def _preallocate_pending_task_batch(self, request: ToolCallRequest) -> None:
        # Phase 2: stamp on the ToolNode superstep, after FinalizeStep advanced
        # the root step counter. Skip if this wrap is not in the stashed batch
        # (sequential later-step task calls must first-seen-allocate).
        if current_attempt_context() is None:
            return
        current = self._boundary_from_task(request.tool_call)
        if current is None:
            return
        pending = self._pending_boundaries(request)
        if not pending or current not in pending:
            return
        preallocate_child_operation_ordinals(pending)
        set_pending_child_operation_batch(())

    def after_model(
        self, state: AgentState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        # Phase 1: validate and stash. Stamping here would key ordinals to the
        # model step, which FinalizeStep then leaves behind. Mutate the
        # attempt-scoped holder so Pregel worker copies see the batch.
        if current_attempt_context() is None:
            return None
        set_pending_child_operation_batch(self._task_boundaries_from_state(state))
        return None

    async def aafter_model(
        self, state: AgentState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolCallResult],
    ) -> ToolCallResult:
        self._preallocate_pending_task_batch(request)
        boundary = self._boundary(request)
        if boundary is None:
            return handler(request)
        with (
            child_operation_boundary_scope(boundary),
            _deep_agent_task_scope(boundary.occurrence_key),
        ):
            result = handler(request)
            _stamp_task_result_message_id(result, boundary.occurrence_key)
            return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolCallResult]],
    ) -> ToolCallResult:
        self._preallocate_pending_task_batch(request)
        boundary = self._boundary(request)
        if boundary is None:
            return await handler(request)
        with (
            child_operation_boundary_scope(boundary),
            _deep_agent_task_scope(boundary.occurrence_key),
        ):
            result = await handler(request)
            _stamp_task_result_message_id(result, boundary.occurrence_key)
            return result
