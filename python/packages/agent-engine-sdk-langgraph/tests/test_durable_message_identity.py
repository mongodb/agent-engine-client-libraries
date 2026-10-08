"""Durable message identity is assigned before LangGraph reducers run."""

from __future__ import annotations

from typing import Any

import pytest
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, Overwrite

from agent_engine_sdk_langgraph.agent import LangGraphBaseAgent
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import WorkflowIdentity
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.workflow import attempt_context_scope


def _attempt(attempt_id: str, fencing_token: int) -> AttemptContext:
    return AttemptContext(
        attempt_id=attempt_id,
        fencing_token=fencing_token,
        workflow_identity=WorkflowIdentity(execution_id="execution-1"),
    )


def _run_twice(graph: Any, graph_input: Any) -> tuple[Any, Any]:
    with attempt_context_scope(_attempt("attempt-1", 1)):
        first = graph.invoke(graph_input())
    with attempt_context_scope(_attempt("attempt-2", 2)):
        replay = graph.invoke(graph_input())
    return first, replay


def test_missing_ai_message_ids_are_stable_and_explicit_ids_are_preserved() -> None:
    builder = StateGraph(MessagesState)
    builder.add_node(
        "respond",
        lambda state: {
            "messages": [
                AIMessage(content="first"),
                AIMessage(content="second"),
                AIMessage(content="provider", id="provider-id"),
            ]
        },
    )
    builder.add_edge(START, "respond")
    graph = builder.compile()
    LangGraphBaseAgent(graph)

    first, replay = _run_twice(graph, lambda: {"messages": []})
    first_messages = first["messages"]
    replay_messages = replay["messages"]

    assert [message.id for message in first_messages] == [
        message.id for message in replay_messages
    ]
    assert first_messages[0].id != first_messages[1].id
    assert first_messages[0].id.startswith("durable-message:execution-1:1:")
    assert first_messages[2].id == "provider-id"


def test_tuple_message_id_is_stable_across_replacement_attempts() -> None:
    builder = StateGraph(MessagesState)
    builder.add_node(
        "respond",
        lambda state: {"messages": [("human", "hello")]},
    )
    builder.add_edge(START, "respond")
    graph = builder.compile()
    LangGraphBaseAgent(graph)

    first, replay = _run_twice(graph, lambda: {"messages": []})
    first_message = first["messages"][0]
    replay_message = replay["messages"][0]

    assert first_message.content == "hello"
    assert first_message.id == replay_message.id
    assert first_message.id.startswith("durable-message:execution-1:1:")


@pytest.mark.parametrize("message_type", ["assistant", "user"])
def test_type_alias_message_id_is_stable_across_replacement_attempts(
    message_type: str,
) -> None:
    builder = StateGraph(MessagesState)
    builder.add_node(
        "respond",
        lambda state: {"messages": [{"type": message_type, "content": "hello"}]},
    )
    builder.add_edge(START, "respond")
    graph = builder.compile()
    LangGraphBaseAgent(graph)

    first, replay = _run_twice(graph, lambda: {"messages": []})

    assert first["messages"][0].id == replay["messages"][0].id
    assert first["messages"][0].id.startswith("durable-message:execution-1:1:")


def test_role_like_non_message_state_is_unchanged() -> None:
    class State(MessagesState):
        profile: list[str]
        descriptor: dict[str, str]

    builder = StateGraph(State)
    builder.add_node(
        "respond",
        lambda state: {
            "messages": [AIMessage(content="done")],
            "profile": ["user", "admin"],
            "descriptor": {"type": "assistant", "content": "ordinary state"},
        },
    )
    builder.add_edge(START, "respond")
    graph = builder.compile()
    LangGraphBaseAgent(graph)

    first, _ = _run_twice(
        graph,
        lambda: {"messages": [], "profile": [], "descriptor": {}},
    )

    assert first["profile"] == ["user", "admin"]
    assert first["descriptor"] == {
        "type": "assistant",
        "content": "ordinary state",
    }


def test_mapping_command_message_id_is_stable_across_replacement_attempts() -> None:
    builder = StateGraph(MessagesState)
    builder.add_node(
        "respond",
        lambda state: Command(update={"messages": [AIMessage(content="replacement")]}),
    )
    builder.add_edge(START, "respond")
    graph = builder.compile()
    LangGraphBaseAgent(graph)

    first, replay = _run_twice(graph, lambda: {"messages": []})

    assert len(first["messages"]) == 1
    assert first["messages"][0].content == "replacement"
    assert first["messages"][0].id == replay["messages"][0].id
    assert first["messages"][0].id.startswith("durable-message:execution-1:1:")


def test_stock_tool_node_message_id_is_stable_across_replacement_attempts() -> None:
    @tool
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        return f"found:{order_id}"

    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode([lookup_order]))
    builder.add_edge(START, "tools")
    graph = builder.compile()
    LangGraphBaseAgent(graph)

    def graph_input() -> dict[str, list[AIMessage]]:
        return {
            "messages": [
                AIMessage(
                    content="",
                    id="assistant-plan",
                    tool_calls=[
                        {
                            "name": "lookup_order",
                            "args": {"order_id": "123"},
                            "id": "tool-call-1",
                        }
                    ],
                )
            ]
        }

    first, replay = _run_twice(graph, graph_input)
    first_tool = first["messages"][-1]
    replay_tool = replay["messages"][-1]

    assert isinstance(first_tool, ToolMessage)
    assert isinstance(replay_tool, ToolMessage)
    assert first_tool.id == replay_tool.id
    assert first_tool.id is not None
    assert first_tool.id.startswith("durable-message:execution-1:1:")
    assert first_tool.tool_call_id == "tool-call-1"


@pytest.mark.parametrize(
    "shape",
    ["native_remove_message", "langgraph_overwrite", "serialized_overwrite_wire"],
)
def test_deep_agent_patched_tool_message_id_is_stable_across_attempts(
    shape: str,
) -> None:
    """Deep Agents replaces history before a later reducer sees its patch.

    Covers every node-output shape the durable identity wrapper must stamp:
    the middleware's native update (a RemoveMessage + patched list), an
    explicit LangGraph ``Overwrite``, and the JSON-serialized
    ``__overwrite__`` wire form. The two explicit replacement forms bypass the
    ``add_messages`` reducer that ``RemoveMessage`` only speaks to, so the
    reducer-only marker is dropped before wrapping.
    """
    patch_tool_calls = PatchToolCallsMiddleware()

    def patch_dangling_tool_call(state: MessagesState) -> dict[str, Any] | None:
        update = patch_tool_calls.before_agent(state, None)  # type: ignore[arg-type]
        if update is None or shape == "native_remove_message":
            return update
        messages = [m for m in update["messages"] if not isinstance(m, RemoveMessage)]
        if shape == "langgraph_overwrite":
            return {"messages": Overwrite(messages)}
        return {"messages": {"__overwrite__": messages}}

    builder = StateGraph(MessagesState)
    builder.add_node("patch_tool_calls", patch_dangling_tool_call)
    builder.add_node(
        "respond",
        lambda state: {"messages": [AIMessage(content="continued")]},
    )
    builder.add_edge(START, "patch_tool_calls")
    builder.add_edge("patch_tool_calls", "respond")
    graph = builder.compile()
    LangGraphBaseAgent(graph)

    def graph_input() -> dict[str, list[AIMessage]]:
        return {
            "messages": [
                AIMessage(
                    content="",
                    id="assistant-plan",
                    tool_calls=[
                        {
                            "name": "lookup_order",
                            "args": {"order_id": "123"},
                            "id": "tool-call-1",
                        }
                    ],
                )
            ]
        }

    first, replay = _run_twice(graph, graph_input)
    first_tool = first["messages"][1]
    replay_tool = replay["messages"][1]

    assert isinstance(first_tool, ToolMessage)
    assert isinstance(replay_tool, ToolMessage)
    assert first_tool.id == replay_tool.id
    assert first_tool.id is not None
    assert first_tool.id.startswith("durable-message:execution-1:1:")
    assert first_tool.tool_call_id == "tool-call-1"
