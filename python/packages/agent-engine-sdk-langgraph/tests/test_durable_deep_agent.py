"""Durable operation-path scoping for Deep Agent ``task`` dispatch."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, cast

import pytest
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from langgraph.prebuilt.tool_node import ToolRuntime

from agent_engine_sdk_langgraph.durable_deep_agent import (
    DurableDeepAgentMiddleware,
    _stamp_deep_agent_message_ids,
)
from agent_engine_sdk_langgraph.platform_checkpointer import (
    UnsupportedDurableGraphError,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.workflow import (
    UnsupportedChildOperationFanOutError,
    attempt_context_scope,
    current_operation_path,
)
from agent_engine_runner_shared.workflow.context import (
    current_pending_child_operation_batch,
    reset_attempt_context,
    set_attempt_context,
)

_NO_RUNTIME = cast(ToolRuntime[None, dict[str, Any]], None)


def _tool_call_id(request: ToolCallRequest) -> str:
    tool_call_id = request.tool_call["id"]
    assert tool_call_id is not None
    return tool_call_id


def _after_model(middleware: DurableDeepAgentMiddleware, state: dict[str, Any]) -> None:
    middleware.after_model(state, None)  # type: ignore[arg-type]


def _request(
    *,
    name: str = "task",
    tool_call_id: str = "task-a",
    subagent_type: str = "research",
    state: dict[str, Any] | None = None,
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={
            "name": name,
            "id": tool_call_id,
            "args": {"subagent_type": subagent_type, "description": "research"},
        },
        tool=None,
        state=state if state is not None else {},
        runtime=_NO_RUNTIME,
    )


def _segments() -> list[tuple[str, int]]:
    return [
        (segment.name, segment.ordinal) for segment in current_operation_path().segments
    ]


def _result(request: ToolCallRequest) -> ToolMessage:
    return ToolMessage(content="done", tool_call_id=request.tool_call["id"])


@pytest.mark.parametrize(
    ("durable", "tool_name"),
    [(False, "task"), (True, "search_corpus")],
    ids=["native-task", "durable-non-task"],
)
def test_irrelevant_calls_remain_at_the_parent_path(
    durable: bool, tool_name: str
) -> None:
    middleware = DurableDeepAgentMiddleware()
    seen: list[list[tuple[str, int]]] = []

    def handler(request: ToolCallRequest) -> ToolMessage:
        seen.append(_segments())
        return _result(request)

    if durable:
        with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
            middleware.wrap_tool_call(_request(name=tool_name), handler)
    else:
        middleware.wrap_tool_call(_request(name=tool_name), handler)

    assert seen == [[("agent", 1)]]


def test_sync_task_scopes_descendants_and_restores_parent() -> None:
    middleware = DurableDeepAgentMiddleware()
    seen: list[list[tuple[str, int]]] = []

    def handler(request: ToolCallRequest) -> ToolMessage:
        seen.append(_segments())
        return _result(request)

    with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
        middleware.wrap_tool_call(_request(), handler)
        assert _segments() == [("agent", 1)]

    assert seen == [[("agent", 1), ("research", 1)]]


def test_durable_task_stamps_child_state_and_parent_result_messages() -> None:
    middleware = DurableDeepAgentMiddleware()
    child_messages = [
        HumanMessage(content="research", id="random-input"),
        AIMessage(content="", id="provider-run"),
        ToolMessage(
            content="updated",
            tool_call_id="write-todos-a",
            id="random-tool-result",
        ),
    ]

    def handler(request: ToolCallRequest) -> Command[Any]:
        _stamp_deep_agent_message_ids(child_messages)
        return Command(
            update={
                "messages": [
                    ToolMessage(content="done", tool_call_id=_tool_call_id(request))
                ]
            }
        )

    with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
        result = middleware.wrap_tool_call(_request(), handler)

    assert [message.id for message in child_messages] == [
        "durable-deep-agent-input:task-a:0",
        "provider-run",
        "durable-deep-agent-tool-result:task-a:write-todos-a",
    ]
    assert isinstance(result, Command)
    assert isinstance(result.update, dict)
    result_message = result.update["messages"][0]
    assert result_message.id == "durable-deep-agent-task-result:task-a"


def test_native_task_does_not_rewrite_child_messages() -> None:
    middleware = DurableDeepAgentMiddleware()
    message = HumanMessage(content="research", id="native-input")

    def handler(request: ToolCallRequest) -> ToolMessage:
        _stamp_deep_agent_message_ids([message])
        return _result(request)

    result = middleware.wrap_tool_call(_request(), handler)

    assert message.id == "native-input"
    assert isinstance(result, ToolMessage)
    assert result.id is None


def test_native_task_allows_compiled_subagent_with_checkpointer() -> None:
    middleware = DurableDeepAgentMiddleware(unsupported_subagent_names={"research"})
    seen: list[str] = []

    def handler(request: ToolCallRequest) -> ToolMessage:
        tool_call_id = request.tool_call["id"]
        assert tool_call_id is not None
        seen.append(tool_call_id)
        return _result(request)

    middleware.wrap_tool_call(_request(), handler)

    assert seen == ["task-a"]


def test_durable_task_rejects_compiled_subagent_with_checkpointer() -> None:
    middleware = DurableDeepAgentMiddleware(unsupported_subagent_names={"research"})
    handler_called = False

    def handler(request: ToolCallRequest) -> ToolMessage:
        nonlocal handler_called
        handler_called = True
        return _result(request)

    with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
        with pytest.raises(UnsupportedDurableGraphError, match="checkpointer=None"):
            middleware.wrap_tool_call(_request(), handler)

    assert not handler_called


@pytest.mark.anyio
async def test_async_task_keeps_scope_across_await_and_restores_on_error() -> None:
    middleware = DurableDeepAgentMiddleware()
    seen: list[list[tuple[str, int]]] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        await asyncio.sleep(0)
        seen.append(_segments())
        raise RuntimeError("subagent failed")

    with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
        with pytest.raises(RuntimeError, match="subagent failed"):
            await middleware.awrap_tool_call(_request(), handler)
        assert _segments() == [("agent", 1)]

    assert seen == [[("agent", 1), ("research", 1)]]


@pytest.mark.parametrize(
    ("tool_call_id", "subagent_type", "missing"),
    [("", "research", "tool call ID"), ("task-a", "", "subagent type")],
)
def test_durable_task_requires_stable_identity(
    tool_call_id: str, subagent_type: str, missing: str
) -> None:
    middleware = DurableDeepAgentMiddleware()
    handler = cast(
        Callable[[ToolCallRequest], ToolMessage],
        lambda request: _result(request),
    )

    with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
        with pytest.raises(ValueError, match=missing):
            middleware.wrap_tool_call(
                _request(tool_call_id=tool_call_id, subagent_type=subagent_type),
                handler,
            )


def _parent_state(*tool_calls: dict[str, Any]) -> dict[str, Any]:
    return {"messages": [AIMessage(content="", tool_calls=list(tool_calls))]}


def _task_call(tool_call_id: str, subagent_type: str = "research") -> dict[str, Any]:
    return {
        "name": "task",
        "id": tool_call_id,
        "args": {"subagent_type": subagent_type, "description": "research"},
    }


def test_same_step_tasks_keep_message_order_under_inverted_start() -> None:
    middleware = DurableDeepAgentMiddleware()
    seen: dict[str, list[tuple[str, int]]] = {}

    def handler(request: ToolCallRequest) -> ToolMessage:
        seen[_tool_call_id(request)] = _segments()
        return _result(request)

    parent = _parent_state(_task_call("task-a"), _task_call("task-b"))
    with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
        _after_model(middleware, parent)
        middleware.wrap_tool_call(
            _request(tool_call_id="task-b", state=parent), handler
        )
        middleware.wrap_tool_call(
            _request(tool_call_id="task-a", state=parent), handler
        )

    assert seen["task-a"] == [("agent", 1), ("research", 1)]
    assert seen["task-b"] == [("agent", 1), ("research", 2)]


def test_same_step_tasks_preallocate_from_wrap_state_without_after_model() -> None:
    middleware = DurableDeepAgentMiddleware()
    seen: dict[str, list[tuple[str, int]]] = {}

    def handler(request: ToolCallRequest) -> ToolMessage:
        seen[_tool_call_id(request)] = _segments()
        return _result(request)

    parent = _parent_state(_task_call("task-a"), _task_call("task-b"))
    with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
        middleware.wrap_tool_call(
            _request(tool_call_id="task-b", state=parent), handler
        )
        middleware.wrap_tool_call(
            _request(tool_call_id="task-a", state=parent), handler
        )

    assert seen["task-a"] == [("agent", 1), ("research", 1)]
    assert seen["task-b"] == [("agent", 1), ("research", 2)]


def test_same_step_tasks_preallocate_from_after_model_without_wrap_state() -> None:
    middleware = DurableDeepAgentMiddleware()
    seen: dict[str, list[tuple[str, int]]] = {}

    def handler(request: ToolCallRequest) -> ToolMessage:
        seen[_tool_call_id(request)] = _segments()
        return _result(request)

    with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
        _after_model(
            middleware,
            _parent_state(_task_call("task-a"), _task_call("task-b")),
        )
        middleware.wrap_tool_call(_request(tool_call_id="task-b"), handler)
        middleware.wrap_tool_call(_request(tool_call_id="task-a"), handler)

    assert seen["task-a"] == [("agent", 1), ("research", 1)]
    assert seen["task-b"] == [("agent", 1), ("research", 2)]


def test_same_step_task_batch_is_cleared_after_successful_preallocate() -> None:
    middleware = DurableDeepAgentMiddleware()

    def handler(request: ToolCallRequest) -> ToolMessage:
        return _result(request)

    with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
        _after_model(
            middleware,
            _parent_state(_task_call("task-a"), _task_call("task-b")),
        )
        middleware.wrap_tool_call(_request(tool_call_id="task-b"), handler)
        assert current_pending_child_operation_batch() == ()


def test_aborted_attempt_discards_pending_task_batch() -> None:
    middleware = DurableDeepAgentMiddleware()
    binding = set_attempt_context(AttemptContext(attempt_id="attempt-1"))
    try:
        _after_model(
            middleware,
            _parent_state(_task_call("task-a"), _task_call("task-b")),
        )
        assert current_pending_child_operation_batch() != ()
    finally:
        reset_attempt_context(binding)

    assert current_pending_child_operation_batch() == ()


def test_duplicate_same_step_task_ids_never_enter_a_child_handler() -> None:
    middleware = DurableDeepAgentMiddleware()
    handler_called = False

    def handler(request: ToolCallRequest) -> ToolMessage:
        nonlocal handler_called
        handler_called = True
        return _result(request)

    parent = _parent_state(_task_call("task-a"), _task_call("task-a"))
    with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
        _after_model(middleware, parent)
        with pytest.raises(
            UnsupportedChildOperationFanOutError, match="duplicate occurrence"
        ):
            middleware.wrap_tool_call(
                _request(tool_call_id="task-a", state=parent), handler
            )

    assert not handler_called


def test_unreadable_parent_batch_never_enters_a_child_handler() -> None:
    middleware = DurableDeepAgentMiddleware()

    with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
        with pytest.raises(ValueError, match="tool call ID"):
            _after_model(
                middleware,
                _parent_state(_task_call("task-a"), _task_call("")),
            )
