"""Discover compiled subgraphs and translate their checkpoint namespaces.

LangGraph exposes the active nested graph path through documented
``checkpoint_ns`` runtime config:
https://docs.langchain.com/oss/python/langgraph/persistence#checkpoint-namespace
"""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.config import get_config
from langgraph.graph.state import CompiledStateGraph

from agent_engine_sdk_langgraph.durable_errors import UnsupportedDurableGraphError
from agent_engine_runner_shared.workflow import ChildOperationBoundary

__all__ = ["DurableSubgraphResolver"]


@dataclass(frozen=True)
class _Topology:
    """One compiled graph's node names and directly compiled child graphs."""

    node_names: frozenset[str]
    children: dict[str, "_Topology"]


class DurableSubgraphResolver:
    """Translate LangGraph nesting into framework-neutral durable boundaries.

    Construction discovers the compiled-child topology once from the materialized
    graph. Later, activity admission calls this object to read the active
    LangGraph checkpoint namespace and resolve it against that saved topology.
    """

    def __init__(self, graph: CompiledStateGraph) -> None:
        errors: list[str] = []
        self._topology = _discover(graph, (), set(), errors)
        self._validation_errors = tuple(errors)

    def validate(self) -> None:
        """Reject graph shapes that cannot participate in a durable attempt."""
        if self._validation_errors:
            raise UnsupportedDurableGraphError(self._validation_errors[0])

    @property
    def has_compiled_children(self) -> bool:
        """Whether the root graph directly composes any compiled child graph."""
        return bool(self._topology.children)

    def __call__(self) -> tuple[ChildOperationBoundary, ...]:
        """Resolve the LangGraph node that is admitting a durable activity.

        The input is implicit: LangGraph's current ``RunnableConfig``, made
        available by ``get_config()`` while a graph node is running. Runner-shared
        invokes this callback from ``current_operation_path()`` as it constructs
        an activity command.

        The output contains one framework-neutral boundary for every compiled
        child graph in the active namespace. The root graph returns an empty
        tuple; agent-engine-runner-shared adds the fixed ``agent/1`` root segment.
        """
        config = get_config()
        configurable = config.get("configurable", {})
        namespace = configurable.get("checkpoint_ns", "")
        if not isinstance(namespace, str):
            raise UnsupportedDurableGraphError(
                "LangGraph checkpoint namespace must be a string"
            )
        return self.resolve_namespace(namespace)

    def resolve_namespace(
        self,
        namespace: str,
    ) -> tuple[ChildOperationBoundary, ...]:
        """Translate one checkpoint namespace into compiled-child boundaries.

        Args:
            namespace: A LangGraph path such as
                ``case_investigation:task-a|policy_analysis:task-b|review:task-c``.
                Each token contains a structural node name and an attempt-local
                runtime task identifier.

        Returns:
            The compiled graph portions of that path. For the example above,
            ``case_investigation/task-a`` and ``policy_analysis/task-b`` become
            two ``ChildOperationBoundary`` values. The final ordinary node
            (``review``) is validated against the topology but omitted because it
            is not a graph boundary. An empty namespace returns an empty tuple.
        Raises:
            UnsupportedDurableGraphError: If the namespace cannot be explained
                completely by the topology discovered at construction time.
        """
        if not namespace:
            return ()

        topology = self._topology
        boundaries: list[ChildOperationBoundary] = []
        segments = namespace.split("|")
        for index, segment in enumerate(segments):
            name, separator, occurrence_key = segment.partition(":")
            if (
                not separator
                or not name
                or not occurrence_key
                or name not in topology.node_names
            ):
                raise UnsupportedDurableGraphError(
                    f"unsupported LangGraph checkpoint namespace {namespace!r}"
                )
            child = topology.children.get(name)
            if child is None:
                if index != len(segments) - 1:
                    raise UnsupportedDurableGraphError(
                        f"unsupported LangGraph checkpoint namespace {namespace!r}"
                    )
                break
            boundaries.append(ChildOperationBoundary(name, occurrence_key))
            topology = child
        return tuple(boundaries)


def _discover(
    graph: CompiledStateGraph,
    path: tuple[str, ...],
    ancestors: set[int],
    errors: list[str],
) -> _Topology:
    """Build the topology used later to validate and translate namespaces.

    ``graph`` is the current compiled graph. ``path`` names its structural
    location, ``ancestors`` contains the Python object identities already on the
    recursion stack, and ``errors`` collects shapes that native LangGraph may run
    but durable execution cannot support. The returned tree retains every node
    name for validation and recurses only through compiled child graphs.
    """
    # id() is Python's in-process object identity. It is used only to notice the
    # same graph object recursively appearing in its own ancestry; it is never a
    # durable identifier and is not persisted or sent to OE.
    graph_object_id = id(graph)
    if graph_object_id in ancestors:
        errors.append(
            f"compiled subgraph topology at {'/'.join(path)} is cyclic and unsupported"
        )
        return _Topology(node_names=frozenset(), children={})

    descendants = ancestors | {graph_object_id}
    children: dict[str, _Topology] = {}
    for name, node in graph.builder.nodes.items():
        runnable = node.runnable
        if not isinstance(runnable, CompiledStateGraph):
            continue

        child_path = (*path, name)
        if runnable.checkpointer is not None:
            errors.append(
                f"compiled subgraph {'/'.join(child_path)!r} must use the default "
                "checkpointer=None; checkpointer=True, checkpointer=False, and "
                "independent child checkpointers are not supported by durable_workflow"
            )
        children[name] = _discover(runnable, child_path, descendants, errors)
    return _Topology(node_names=frozenset(graph.builder.nodes), children=children)
