"""Durable identity translation for directly composed LangGraph subgraphs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_engine_sdk_langgraph.durable_subgraphs import (
    DurableSubgraphResolver,
    reject_node_retry_policies,
)
from agent_engine_sdk_langgraph.platform_checkpointer import (
    UnsupportedDurableGraphError,
)
from agent_engine_runner_shared.workflow import ChildOperationBoundary


class _State(TypedDict, total=False):
    value: str


def _compiled_node(
    name: str, runnable: Callable | CompiledStateGraph
) -> CompiledStateGraph:
    builder = StateGraph(_State)
    builder.add_node(name, runnable)
    builder.add_edge(START, name)
    builder.add_edge(name, END)
    return builder.compile()


def _support_graph() -> CompiledStateGraph:
    policy = _compiled_node("review_policy", lambda state: state)
    investigation = _compiled_node("policy_analysis", policy)
    return _compiled_node("case_investigation", investigation)


def test_resolves_child_and_grandchild_namespace() -> None:
    resolver = DurableSubgraphResolver(_support_graph())

    assert resolver.has_compiled_children is True
    assert resolver.resolve_namespace("") == ()
    assert resolver.resolve_namespace(
        "case_investigation:task-a|policy_analysis:task-b|review_policy:task-c"
    ) == (
        ChildOperationBoundary("case_investigation", "task-a"),
        ChildOperationBoundary("policy_analysis", "task-b"),
    )
    assert resolver.resolve_namespace("case_investigation:task-a") == (
        ChildOperationBoundary("case_investigation", "task-a"),
    )


def test_reports_when_graph_has_no_compiled_children() -> None:
    resolver = DurableSubgraphResolver(_compiled_node("agent", lambda state: state))

    assert resolver.has_compiled_children is False


@pytest.mark.parametrize(
    "namespace",
    [
        "case_investigation",
        "case_investigation:",
        ":task-a",
        "unknown:task-a",
        "case_investigation:task-a|unknown:task-b",
        "case_investigation:task-a|policy_analysis:task-b|review_policy:task-c|unknown:task-d",
    ],
)
def test_rejects_unresolvable_nested_namespace(namespace: str) -> None:
    resolver = DurableSubgraphResolver(_support_graph())

    with pytest.raises(UnsupportedDurableGraphError, match="checkpoint namespace"):
        resolver.resolve_namespace(namespace)


def test_rejects_child_owned_checkpointer_only_when_validated() -> None:
    child_builder = StateGraph(_State)
    child_builder.add_node("review_policy", lambda state: state)
    child_builder.add_edge(START, "review_policy")
    child_builder.add_edge("review_policy", END)
    child = child_builder.compile(checkpointer=InMemorySaver())

    resolver = DurableSubgraphResolver(_compiled_node("case_investigation", child))

    with pytest.raises(UnsupportedDurableGraphError, match="checkpointer=None"):
        resolver.validate()


def test_rejects_a_node_retry_policy_at_any_depth() -> None:
    from langgraph.types import RetryPolicy

    def graph_with_retry() -> StateGraph:
        builder = StateGraph(_State)
        builder.add_node(
            "review_policy", lambda state: state, retry_policy=RetryPolicy()
        )
        builder.add_edge(START, "review_policy")
        builder.add_edge("review_policy", END)
        return builder

    with pytest.raises(
        UnsupportedDurableGraphError, match="'review_policy' sets a retry policy"
    ):
        reject_node_retry_policies(graph_with_retry().compile())

    with pytest.raises(
        UnsupportedDurableGraphError,
        match="'case_investigation/review_policy' sets a retry policy",
    ):
        reject_node_retry_policies(
            _compiled_node("case_investigation", graph_with_retry().compile())
        )

    # A compiled graph's own policy is the default for all its nodes.
    defaulted = StateGraph(_State)
    defaulted.add_node("review_policy", lambda state: state)
    defaulted.add_edge(START, "review_policy")
    defaulted.add_edge("review_policy", END)
    compiled = defaulted.compile()
    compiled.retry_policy = (RetryPolicy(),)
    with pytest.raises(
        UnsupportedDurableGraphError, match="'root' sets a default retry policy"
    ):
        reject_node_retry_policies(compiled)

    # So is a policy every node inherits from the builder's node defaults.
    inherited = StateGraph(_State)
    inherited.set_node_defaults(retry_policy=RetryPolicy())
    inherited.add_node("review_policy", lambda state: state)
    inherited.add_edge(START, "review_policy")
    inherited.add_edge("review_policy", END)
    with pytest.raises(
        UnsupportedDurableGraphError, match="'review_policy' sets a retry policy"
    ):
        reject_node_retry_policies(inherited.compile())

    plain = StateGraph(_State)
    plain.add_node("review_policy", lambda state: state)
    plain.add_edge(START, "review_policy")
    plain.add_edge("review_policy", END)
    reject_node_retry_policies(plain.compile())


def test_rejects_per_thread_child_checkpointer_with_actionable_error() -> None:
    child_builder = StateGraph(_State)
    child_builder.add_node("review_policy", lambda state: state)
    child_builder.add_edge(START, "review_policy")
    child_builder.add_edge("review_policy", END)
    child = child_builder.compile(checkpointer=True)

    resolver = DurableSubgraphResolver(_compiled_node("case_investigation", child))

    with pytest.raises(UnsupportedDurableGraphError, match="checkpointer=None"):
        resolver.validate()


def test_reads_namespace_from_current_langgraph_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent_engine_sdk_langgraph.durable_subgraphs.get_config",
        lambda: {
            "configurable": {
                "checkpoint_ns": "case_investigation:task-a|policy_analysis:task-b"
            }
        },
    )

    assert [
        (boundary.name, boundary.occurrence_key)
        for boundary in DurableSubgraphResolver(_support_graph())()
    ] == [
        ("case_investigation", "task-a"),
        ("policy_analysis", "task-b"),
    ]
