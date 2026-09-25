"""Map ADK's canonical Workflow node path to durable OE operation paths.

ADK assigns every Workflow node execution a path such as
``outer@1/inner@1/review@1`` before it runs the node. The path is the identity
ADK itself records on emitted events and uses to resume Workflow execution.

ADK does not currently carry that path into model or tool callback contexts.
The adapter therefore wraps the public ``BaseNode.run`` boundary on the
configured graph. The wrapper binds ``Context.node_path`` in a request-local
``ContextVar`` for the node's complete execution. Model and tool work inherits
that scope, including work launched concurrently by parallel Workflow nodes.

Configured ``sub_agents`` use the same public node path. When ADK enters a
selected child, the agent route peer reserves that child under the parent
encoded in the path before binding the child's complete path. ``AgentTool`` is
not supported because its private Runner does not preserve the caller path.

This requires a statically configured Workflow or agent tree. Application code
cannot add a new durable node at runtime with ``Context.run_node()`` because
that node was not present when the adapter instrumented the topology. Such
calls fail before the dynamic child can admit model, tool, or suspension
activity.

The root ADK node maps to OE's existing ``agent`` root. Each remaining ADK
segment becomes one OE child boundary. For example,
``outer@1/inner@1/review@1`` maps to ``agent -> inner -> review`` while retaining
the complete cumulative ADK prefixes as occurrence keys.

An emitted wait outlives the live node scope. Its event already contains the
same canonical path in ``event.node_info.path``; the adapter records boundaries
from that value and restores them only while allocating the OE suspension.
No configured-topology inference or synthetic runtime identity is required.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator, Sequence
from contextlib import aclosing, contextmanager
from contextvars import ContextVar
from functools import wraps
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from agent_engine_runner_shared.workflow import (
    ChildOperationBoundary,
    operation_path_resolver_scope,
    preallocate_child_operation_ordinals,
)
from agent_engine_sdk_adk.agent_route import AgentRouteAdapter
from agent_engine_sdk_adk.errors import UnsupportedDurableADKError

if TYPE_CHECKING:
    from google.adk.agents.context import Context
    from google.adk.events import Event
    from google.adk.workflow import BaseNode

__all__ = [
    "DurableRouteAdapter",
    "adk_operation_boundaries_from_node_path",
    "adk_operation_path_scope",
    "adk_suspension_operation_path_scope",
    "has_stable_operation_identity",
]


_CURRENT_WORKFLOW_PATH: ContextVar[tuple[ChildOperationBoundary, ...] | None] = (
    ContextVar("adk_durable_workflow_path", default=None)
)
_CURRENT_CONFIGURED_NODE_IDS: ContextVar[frozenset[int] | None] = ContextVar(
    "adk_durable_configured_node_ids", default=None
)
_CURRENT_ROUTE_ADAPTER: ContextVar[Any | None] = ContextVar(
    "adk_durable_route_adapter", default=None
)
_CURRENT_PARALLEL_WORKER_RUN_IDS: ContextVar[
    tuple[int, dict[int, list[str]]] | None
] = ContextVar("adk_durable_parallel_worker_run_ids", default=None)
_BASE_NODE_RUN_WRAPPER_ATTRIBUTE = "_agent_engine_durable_run_wrapper_installed"
_PARALLEL_WORKER_RUN_WRAPPER_ATTRIBUTE = "_durable_parallel_worker_wrapper_installed"
_RUN_NODE_GUARD_ATTRIBUTE = "_agent_engine_durable_run_node_guard_installed"
_WORKFLOW_REPLAY_WRAPPER_ATTRIBUTE = "_agent_engine_durable_replay_wrapper_installed"
_CONFIGURED_NODE_PROVENANCE_ATTRIBUTE = "_agent_engine_durable_node_provenance"


def adk_operation_boundaries_from_node_path(
    node_path: str,
) -> tuple[ChildOperationBoundary, ...]:
    """Translate ADK's root-to-node path into cumulative OE child boundaries.

    The first segment is omitted because the configured ADK root corresponds
    to OE's existing ``agent`` root. Given ``outer@1/inner@1/review@1``, this
    returns ``inner`` with occurrence ``outer@1/inner@1`` followed by
    ``review`` with occurrence ``outer@1/inner@1/review@1``.
    """
    raw_segments = node_path.split("/")
    if not node_path or any(not segment for segment in raw_segments):
        raise RuntimeError("Google ADK returned an invalid workflow node path")

    segments: list[tuple[str, str]] = []
    for segment in raw_segments:
        name, separator, run_id = segment.rpartition("@")
        if not separator or not name or not run_id:
            raise RuntimeError("Google ADK returned an invalid workflow node path")
        segments.append((name, run_id))

    boundaries: list[ChildOperationBoundary] = []
    for index, (name, _run_id) in enumerate(segments[1:], start=1):
        occurrence_key = "/".join(
            f"{prefix_name}@{prefix_run_id}"
            for prefix_name, prefix_run_id in segments[: index + 1]
        )
        boundaries.append(
            ChildOperationBoundary(name=name, occurrence_key=occurrence_key)
        )
    return tuple(boundaries)


def _current_boundaries() -> tuple[ChildOperationBoundary, ...]:
    return _CURRENT_WORKFLOW_PATH.get() or ()


def has_stable_operation_identity() -> bool:
    """Whether the active node is below the root operation path."""
    return bool(_current_boundaries())


@contextmanager
def adk_operation_path_scope() -> Generator[None, None, None]:
    """Install the ADK node-path resolver for one Runner invocation."""
    with operation_path_resolver_scope(_current_boundaries):
        yield


@contextmanager
def adk_suspension_operation_path_scope(
    boundaries: Sequence[ChildOperationBoundary],
) -> Generator[None, None, None]:
    """Bind a wait's recorded ADK path during OE suspension allocation."""
    token = _CURRENT_WORKFLOW_PATH.set(tuple(boundaries))
    try:
        with operation_path_resolver_scope(_current_boundaries):
            yield
    finally:
        _CURRENT_WORKFLOW_PATH.reset(token)


def configured_adk_nodes(root: BaseNode) -> tuple[BaseNode, ...]:
    """Return every node ADK materialized from the configured topology.

    In addition to Workflow graph nodes, ADK stores configured
    ``parallel_worker`` templates behind ``_inner_node`` and ``_node``. Those
    private containers are the only way ADK 2.x exposes the nodes it will later
    dispatch through ``Context.run_node()``. Following them here distinguishes
    configured workers from application-created dynamic nodes.
    """
    from google.adk.workflow import BaseNode as ADKNode

    configured: list[BaseNode] = []
    seen: set[int] = set()

    def visit(node: BaseNode) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        configured.append(node)

        graph = getattr(node, "graph", None)
        if graph is not None:
            for child in graph.nodes:
                if isinstance(child, ADKNode):
                    visit(child)

        for attribute in ("_inner_node", "_node"):
            child = getattr(node, attribute, None)
            if isinstance(child, ADKNode):
                visit(child)

    visit(root)
    return tuple(configured)


def _set_configured_node_provenance(
    node: BaseNode, provenance: object | None = None
) -> object:
    resolved = (
        provenance
        or getattr(node, _CONFIGURED_NODE_PROVENANCE_ATTRIBUTE, None)
        or object()
    )
    object.__setattr__(node, _CONFIGURED_NODE_PROVENANCE_ATTRIBUTE, resolved)
    return resolved


def _configured_run_node_edges(
    root: BaseNode,
) -> frozenset[tuple[tuple[str, ...], str, type[BaseNode], object]]:
    """Describe configured private templates ADK clones through run_node().

    These edges admit ADK-created copies of nodes already present in the static
    Workflow graph, not application-created nodes introduced during execution.
    """
    from google.adk.workflow import BaseNode as ADKNode

    edges: set[tuple[tuple[str, ...], str, type[BaseNode], object]] = set()
    seen: set[tuple[int, tuple[str, ...]]] = set()

    def visit(node: BaseNode, path: tuple[str, ...]) -> None:
        identity = (id(node), path)
        if identity in seen:
            return
        seen.add(identity)

        graph = getattr(node, "graph", None)
        if graph is not None:
            for child in graph.nodes:
                if isinstance(child, ADKNode):
                    visit(child, (*path, child.name))

        inner = getattr(node, "_inner_node", None)
        if isinstance(inner, ADKNode):
            target = getattr(inner, "_node", None)
            if isinstance(target, ADKNode):
                provenance = _set_configured_node_provenance(
                    node,
                    getattr(target, _CONFIGURED_NODE_PROVENANCE_ATTRIBUTE, None),
                )
                _set_configured_node_provenance(target, provenance)
            # Node delegates to this wrapper with the same Context.
            visit(inner, path)

        child = getattr(node, "_node", None)
        if isinstance(child, ADKNode):
            # ParallelWorker dispatches this template through Context.run_node.
            provenance = _set_configured_node_provenance(child)
            edges.add((path, child.name, type(child), provenance))
            visit(child, (*path, child.name))

    visit(root, (root.name,))
    return frozenset(edges)


def _node_path_names(node_path: str) -> tuple[str, ...]:
    names: list[str] = []
    for segment in node_path.split("/"):
        name, separator, run_id = segment.rpartition("@")
        if not separator or not name or not run_id:
            return ()
        names.append(name)
    return tuple(names)


def _install_context_run_node_guard() -> None:
    """Reject dynamic children on every ADK Context used by a durable turn."""
    from google.adk.agents.context import Context as ADKContext

    if getattr(ADKContext, _RUN_NODE_GUARD_ATTRIBUTE, False):
        return

    original_run_node = ADKContext.run_node

    @wraps(original_run_node)
    async def run_configured_node(
        context: ADKContext, target_node: Any, *args: Any, **kwargs: Any
    ) -> Any:
        parallel_worker_plan = _CURRENT_PARALLEL_WORKER_RUN_IDS.get()
        if (
            parallel_worker_plan is not None
            and id(target_node) == parallel_worker_plan[0]
            and "run_id" not in kwargs
        ):
            node_input = kwargs.get("node_input", args[0] if args else None)
            planned_run_ids = parallel_worker_plan[1].get(id(node_input))
            if planned_run_ids:
                kwargs["run_id"] = planned_run_ids.pop(0)

        configured_node_ids = _CURRENT_CONFIGURED_NODE_IDS.get()
        adapter = _CURRENT_ROUTE_ADAPTER.get()
        if (
            configured_node_ids is not None
            and id(target_node) not in configured_node_ids
            and (
                adapter is None
                or not adapter._allows_configured_run_node(
                    getattr(context, "_node_path", ""), target_node
                )
            )
        ):
            raise UnsupportedDurableADKError(
                "Google ADK durable workflows require a statically configured "
                "graph; runtime Context.run_node() targets are not supported"
            )
        return await original_run_node(context, target_node, *args, **kwargs)

    setattr(ADKContext, "run_node", run_configured_node)
    setattr(ADKContext, _RUN_NODE_GUARD_ATTRIBUTE, True)


def _install_workflow_replay_wrapper() -> None:
    """Keep replayed configured transfers inside ADK's transfer loop."""
    from google.adk.workflow.utils._replay_interceptor import InterceptionResult

    adk_workflow = import_module("google.adk.workflow._workflow")
    if not getattr(adk_workflow, _WORKFLOW_REPLAY_WRAPPER_ATTRIBUTE, False):
        original_workflow_interception = adk_workflow.check_interception

        @wraps(original_workflow_interception)
        def check_configured_transfer(*, node: BaseNode, recovered: Any) -> Any:
            adapter = _CURRENT_ROUTE_ADAPTER.get()
            if (
                adapter is not None
                and recovered is not None
                and recovered.transfer_to_agent is not None
                and adapter._agent_routes.is_configured_transfer_agent(node)
            ):
                # ADK's static Workflow replay otherwise returns the cached
                # transfer to the Workflow loop, which does not follow it.
                # Context replays the same node before following its target.
                return InterceptionResult(should_run=True)
            return original_workflow_interception(node=node, recovered=recovered)

        setattr(adk_workflow, "check_interception", check_configured_transfer)
        setattr(adk_workflow, _WORKFLOW_REPLAY_WRAPPER_ATTRIBUTE, True)

    dynamic_scheduler = import_module("google.adk.workflow._dynamic_node_scheduler")
    if getattr(dynamic_scheduler, _WORKFLOW_REPLAY_WRAPPER_ATTRIBUTE, False):
        return

    original_dynamic_interception = dynamic_scheduler.check_interception

    @wraps(original_dynamic_interception)
    def resume_configured_transfer_target(
        *, node: BaseNode, recovered: Any = None, current_run: Any = None
    ) -> Any:
        adapter = _CURRENT_ROUTE_ADAPTER.get()
        if (
            adapter is not None
            and recovered is not None
            and recovered.transfer_to_agent is None
            and recovered.interrupt_ids
            and not (recovered.interrupt_ids - recovered.resolved_ids)
            and adapter._agent_routes.is_configured_transfer_agent(node)
        ):
            # A transferred agent defaults rerun_on_resume to false, which
            # would expose the raw approval as output instead of finishing.
            return InterceptionResult(
                should_run=True,
                resume_inputs=recovered.resolved_responses,
            )
        return original_dynamic_interception(
            node=node,
            recovered=recovered,
            current_run=current_run,
        )

    setattr(
        dynamic_scheduler,
        "check_interception",
        resume_configured_transfer_target,
    )
    setattr(dynamic_scheduler, _WORKFLOW_REPLAY_WRAPPER_ATTRIBUTE, True)


def _install_base_node_run_wrapper() -> None:
    """Bind paths on ADK's public node boundary, including cloned agents."""
    from google.adk.workflow import BaseNode as ADKNode

    if getattr(ADKNode, _BASE_NODE_RUN_WRAPPER_ATTRIBUTE, False):
        return

    original_run = ADKNode.run

    @wraps(original_run)
    async def run_with_durable_path(
        node: BaseNode, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Event, None]:
        adapter = _CURRENT_ROUTE_ADAPTER.get()
        if adapter is None:
            async with aclosing(
                original_run(node, ctx=ctx, node_input=node_input)
            ) as events:
                async for event in events:
                    yield event
            return

        boundaries = adk_operation_boundaries_from_node_path(ctx.node_path)
        if adapter._agent_routes.is_transfer_agent_path(ctx.node_path):
            adapter._agent_routes.reserve_child(boundaries)
        token = _CURRENT_WORKFLOW_PATH.set(boundaries)
        try:
            async with aclosing(
                original_run(node, ctx=ctx, node_input=node_input)
            ) as events:
                async for event in events:
                    yield event
        finally:
            _CURRENT_WORKFLOW_PATH.reset(token)

    setattr(ADKNode, "run", run_with_durable_path)
    setattr(ADKNode, _BASE_NODE_RUN_WRAPPER_ATTRIBUTE, True)


def _install_parallel_worker_run_wrapper() -> None:
    """Reserve repeated configured worker paths before ADK starts tasks."""
    parallel_worker_module = import_module("google.adk.workflow._parallel_worker")
    parallel_worker = parallel_worker_module._ParallelWorker
    if getattr(parallel_worker, _PARALLEL_WORKER_RUN_WRAPPER_ATTRIBUTE, False):
        return

    original_run_impl = parallel_worker._run_impl

    @wraps(original_run_impl)
    async def run_with_preallocated_children(
        worker: Any, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
        adapter = _CURRENT_ROUTE_ADAPTER.get()
        item_count = (
            len(cast(list[Any], node_input)) if isinstance(node_input, list) else 1
        )
        if adapter is not None and item_count:
            child = worker._node
            child_run_counters = cast(
                dict[str, int], getattr(ctx, "_child_run_counters")
            )
            start = child_run_counters.get(child.name, 0)
            child_boundaries: list[ChildOperationBoundary] = []
            run_ids_by_input: dict[int, list[str]] = {}
            worker_inputs = (
                cast(list[Any], node_input)
                if isinstance(node_input, list)
                else [node_input]
            )
            for offset, worker_input in enumerate(worker_inputs, start=1):
                run_id = f"durable-{start + offset}"
                run_ids_by_input.setdefault(id(worker_input), []).append(run_id)
                child_path = f"{ctx.node_path}/{child.name}@{run_id}"
                child_boundaries.append(
                    adk_operation_boundaries_from_node_path(child_path)[-1]
                )
            child_run_counters[child.name] = start + item_count
            preallocate_child_operation_ordinals(child_boundaries)
            token = _CURRENT_PARALLEL_WORKER_RUN_IDS.set((id(child), run_ids_by_input))
        else:
            token = None

        try:
            async with aclosing(
                original_run_impl(worker, ctx=ctx, node_input=node_input)
            ) as outputs:
                async for output in outputs:
                    yield output
        finally:
            if token is not None:
                _CURRENT_PARALLEL_WORKER_RUN_IDS.reset(token)

    setattr(parallel_worker, "_run_impl", run_with_preallocated_children)
    setattr(parallel_worker, _PARALLEL_WORKER_RUN_WRAPPER_ATTRIBUTE, True)


class DurableRouteAdapter:
    """Install canonical path scopes and translate emitted wait events."""

    def __init__(self, root: BaseNode) -> None:
        from google.adk.agents import BaseAgent

        self._allows_pathless_root_events = isinstance(root, BaseAgent)
        self._agent_routes = AgentRouteAdapter(root, _CURRENT_WORKFLOW_PATH)
        configured_nodes = (
            *configured_adk_nodes(root),
            *self._agent_routes.sub_agents,
        )
        self._configured_node_ids = frozenset(id(node) for node in configured_nodes)
        self._configured_run_node_edges = _configured_run_node_edges(root)

        _install_context_run_node_guard()
        _install_workflow_replay_wrapper()
        _install_base_node_run_wrapper()
        _install_parallel_worker_run_wrapper()

    def _allows_configured_run_node(self, parent_path: str, node: Any) -> bool:
        """Allow only an ADK clone of a configured private node template."""
        return (
            _node_path_names(parent_path),
            getattr(node, "name", None),
            type(node),
            getattr(node, _CONFIGURED_NODE_PROVENANCE_ATTRIBUTE, None),
        ) in self._configured_run_node_edges

    @contextmanager
    def operation_path_scope(self) -> Generator[None, None, None]:
        """Install routing plus this adapter's configured-node allowlist."""
        node_token = _CURRENT_CONFIGURED_NODE_IDS.set(self._configured_node_ids)
        adapter_token = _CURRENT_ROUTE_ADAPTER.set(self)
        try:
            with adk_operation_path_scope():
                yield
        finally:
            _CURRENT_ROUTE_ADAPTER.reset(adapter_token)
            _CURRENT_CONFIGURED_NODE_IDS.reset(node_token)

    def operation_boundaries_for_event(
        self, event: Event
    ) -> tuple[ChildOperationBoundary, ...] | None:
        """Translate an event path, including a known plain-agent root."""
        node_info = event.node_info
        if not node_info.path:
            # ADK currently runs non-LlmAgent BaseAgent roots outside its node
            # runtime, so their events may omit node_info.path. Configured
            # children run through the node runtime and are stamped, so a
            # pathless event under an agent root represents the root itself.
            return () if self._allows_pathless_root_events else None
        return adk_operation_boundaries_from_node_path(node_info.path)
