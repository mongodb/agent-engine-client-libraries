"""Discover compiled subgraphs and translate their checkpoint namespaces.

LangGraph exposes the active nested graph path through documented
``checkpoint_ns`` runtime config:
https://docs.langchain.com/oss/python/langgraph/persistence#checkpoint-namespace

``Send`` workers run concurrently in one superstep, so each one gets its own
operation-path segment named by its superstep and its index in that superstep's
send list. LangGraph orders the send list the same way on every attempt; the
task id and namespace change per attempt.

Importing this module registers a process-wide LangChain configure hook that
adds the active ``SendTaskRecorder`` to every callback manager. Outside a durable
invocation no recorder is active and the hook adds nothing.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from dataclasses import dataclass

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook
from langgraph.config import get_config
from langgraph.graph.state import CompiledStateGraph

from agent_engine_sdk_langgraph.durable_errors import UnsupportedDurableGraphError
from agent_engine_runner_shared.workflow import (
    ChildOperationBoundary,
    operation_path_resolver_scope,
)

__all__ = ["DurableSubgraphResolver", "SendTaskRecorder"]

# LangGraph's task-path tag for tasks created by ``Send``.
_PUSH = "__pregel_push"


def _send_task_boundary(metadata: dict[str, Any]) -> ChildOperationBoundary | None:
    """Name a top-level Send task by its superstep and send index.

    The superstep keeps successive Send rounds of one graph run apart. Other
    tasks get no segment.
    """
    task_path = metadata.get("langgraph_path")
    step = metadata.get("langgraph_step")
    if (
        not isinstance(task_path, (tuple, list))
        or len(task_path) < 2
        or task_path[0] != _PUSH
        or not _is_index(task_path[1])
        or not _is_index(step)
    ):
        return None
    name = f"langgraph.task:{_PUSH}:{step}:{task_path[1]}"
    return ChildOperationBoundary(name, name, spawned_task=True)


def _is_index(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class SendTaskRecorder(BaseCallbackHandler):
    """Remember which checkpoint namespace segments belong to Send tasks.

    Inside a compiled child, LangGraph metadata describes only the child's own
    task, and the namespace names the parent Send task by an attempt-local id.
    A task's start callback carries both its namespace and its task path, so
    recording it lets the resolver name the enclosing Send worker. One recorder
    lives for one durable invocation.
    """

    # Record before the task's own work starts, including under async managers.
    run_inline = True

    def __init__(self) -> None:
        super().__init__()
        # Concurrent task starts write distinct keys and readers look up single
        # keys, so the dict needs no lock.
        self._boundaries: dict[str, ChildOperationBoundary] = {}

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del serialized, inputs, kwargs
        if not metadata:
            return
        boundary = _send_task_boundary(metadata)
        namespace = metadata.get("langgraph_checkpoint_ns")
        if boundary is None or not isinstance(namespace, str) or not namespace:
            return
        self._boundaries[namespace.rpartition("|")[2]] = boundary

    def boundary_for(self, namespace_segment: str) -> ChildOperationBoundary | None:
        """Return the Send task boundary for one ``node:task_id`` segment."""
        return self._boundaries.get(namespace_segment)


_SEND_TASKS: ContextVar[SendTaskRecorder | None] = ContextVar(
    "agent_engine_langgraph_send_tasks", default=None
)
# Adds the active recorder to every LangChain callback manager, including the
# ones LangGraph creates for each task, without changing the invocation config.
register_configure_hook(_SEND_TASKS, inheritable=True)


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

    def __init__(
        self, graph: CompiledStateGraph, *, scope_send_tasks: bool = True
    ) -> None:
        errors: list[str] = []
        self._topology = _discover(graph, (), set(), errors)
        self._validation_errors = tuple(errors)
        self._scope_send_tasks = scope_send_tasks

    def validate(self) -> None:
        """Reject graph shapes that cannot participate in a durable attempt."""
        if self._validation_errors:
            raise UnsupportedDurableGraphError(self._validation_errors[0])

    @property
    def has_compiled_children(self) -> bool:
        """Whether the root graph directly composes any compiled child graph."""
        return bool(self._topology.children)

    @contextmanager
    def scope(self, send_tasks: SendTaskRecorder) -> Iterator[None]:
        """Bind this resolver and the invocation's Send task recorder."""
        token = _SEND_TASKS.set(send_tasks if self._scope_send_tasks else None)
        try:
            with operation_path_resolver_scope(self):
                yield
        finally:
            _SEND_TASKS.reset(token)

    def __call__(self) -> tuple[ChildOperationBoundary, ...]:
        """Resolve the LangGraph node that is admitting a durable activity.

        The input is implicit: LangGraph's current ``RunnableConfig``, made
        available by ``get_config()`` while a graph node is running. Runner-shared
        invokes this callback from ``current_operation_path()`` as it constructs
        an activity command.

        The output contains one framework-neutral boundary for every compiled
        child graph in the active namespace, each preceded by a Send task
        boundary when a Send started that child, and a final Send task boundary
        when the running node is itself a Send task. In a graph without
        compiled children, a Send task that runs a nested graph itself also
        contributes its boundary. The root graph returns an empty tuple;
        agent-engine-runner-shared adds the fixed ``agent/1`` root segment.
        """
        config = get_config()
        configurable = config.get("configurable", {})
        namespace = configurable.get("checkpoint_ns", "")
        if not isinstance(namespace, str):
            raise UnsupportedDurableGraphError(
                "LangGraph checkpoint namespace must be a string"
            )
        send_tasks = _SEND_TASKS.get()
        if self.has_compiled_children:
            boundaries = self.resolve_namespace(namespace, send_tasks)
        elif send_tasks is not None:
            # Without compiled children only Send tasks add segments. A Send
            # worker that runs a graph itself appears as an outer segment.
            outer = namespace.split("|")[:-1] if namespace else []
            boundaries = tuple(
                boundary
                for boundary in map(send_tasks.boundary_for, outer)
                if boundary is not None
            )
        else:
            boundaries = ()
        if not self._scope_send_tasks:
            return boundaries
        send_task = _send_task_boundary(config.get("metadata") or {})
        return boundaries if send_task is None else (*boundaries, send_task)

    def resolve_namespace(
        self,
        namespace: str,
        send_tasks: SendTaskRecorder | None = None,
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
            A child segment that ``send_tasks`` recorded as a Send task is
            preceded by that task's boundary.
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
            send_task = send_tasks.boundary_for(segment) if send_tasks else None
            if send_task is not None:
                boundaries.append(send_task)
            boundaries.append(ChildOperationBoundary(name, occurrence_key))
            topology = child
        return tuple(boundaries)


def reject_node_retry_policies(graph: Any) -> None:
    """Reject a durable graph that sets a retry policy, on a node or as a default.

    A retry re-runs the node from the top. A durable node's side effects are
    recorded activities, which would be replayed or repeated depending on why
    the node ran again, and its own logic is deterministic, so there is
    nothing for a node retry to recover.
    """
    seen: set[int] = set()

    def visit(current: Any, current_path: tuple[str, ...]) -> None:
        nodes = getattr(getattr(current, "builder", None), "nodes", None)
        if id(current) in seen or not isinstance(nodes, dict):
            return
        seen.add(id(current))
        # A compiled graph's own policy is the default for all its nodes.
        if _sets_retry_policy(getattr(current, "retry_policy", None)):
            raise UnsupportedDurableGraphError(
                f"graph {'/'.join(current_path) or 'root'!r} sets a default retry "
                "policy; node retry policies are not supported by durable_workflow"
            )
        # The compiled node carries the effective policy, which may come from
        # the builder's node defaults instead of the node's own spec.
        compiled_nodes = getattr(current, "nodes", None)
        if not isinstance(compiled_nodes, dict):
            compiled_nodes = {}
        for name, node in nodes.items():
            node_path = (*current_path, name)
            if _sets_retry_policy(
                getattr(node, "retry_policy", None)
            ) or _sets_retry_policy(
                getattr(compiled_nodes.get(name), "retry_policy", None)
            ):
                raise UnsupportedDurableGraphError(
                    f"node {'/'.join(node_path)!r} sets a retry policy; node "
                    "retry policies are not supported by durable_workflow"
                )
            visit(getattr(node, "runnable", None), node_path)

    visit(graph, ())


def _sets_retry_policy(policy: Any) -> bool:
    # LangGraph stores "none" as None on a node and as an empty sequence on a
    # compiled graph.
    return policy is not None and policy != () and policy != []


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
