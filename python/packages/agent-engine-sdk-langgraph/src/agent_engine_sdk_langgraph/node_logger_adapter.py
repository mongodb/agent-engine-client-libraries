"""LangGraph callback adapter for BaseExecutionCallback.

Bridges LangChain's BaseCallbackHandler interface to the framework-neutral
BaseExecutionCallback protocol. Extracts node names from LangGraph-specific
metadata and converts UUID run_ids to strings.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.errors import GraphInterrupt
from opentelemetry.trace import Span, SpanContext
from agent_engine_sdk.interfaces import BaseExecutionCallback

logger = logging.getLogger(__name__)


class LangGraphCallbackAdapter(BaseCallbackHandler):
    """Wraps a BaseExecutionCallback to satisfy LangChain's callback interface.

    Used by the LangGraph agent to forward node execution events to the
    platform's framework-neutral NodeExecutionLogger.
    """

    def __init__(self, callback: BaseExecutionCallback):
        super().__init__()
        self._callback = callback
        # run_ids skipped at start (nested chains inside a node): their
        # end/error callbacks are dropped so a ghost pair never reaches the
        # callback. Entries are deleted on end/error to bound the set's size.
        self._skipped_runs: set[str] = set()
        # LangChain Python passes no metadata to on_chain_end/on_chain_error,
        # so the node name is recovered from the start event by run_id — the
        # TS twin keeps the same map for the same reason. Entries are popped on
        # the terminal event to bound the map's size.
        self._run_node_names: dict[str, str] = {}

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if _is_nested_run(tags, metadata):
            self._skipped_runs.add(str(run_id))
            return
        node_name = _extract_node_name(serialized, tags, metadata)
        if not node_name or node_name.startswith("RunnableSequence"):
            return

        self._run_node_names[str(run_id)] = node_name
        self._callback.on_node_start(
            node_name=node_name,
            inputs=inputs,
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            metadata=metadata,
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if str(run_id) in self._skipped_runs:
            self._skipped_runs.discard(str(run_id))
            return
        node_name = self._run_node_names.pop(str(run_id), None) or _extract_node_name(
            {}, tags, metadata
        )
        if not node_name or node_name.startswith("RunnableSequence"):
            return

        self._callback.on_node_end(
            node_name=node_name,
            outputs=outputs,
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            metadata=metadata,
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if str(run_id) in self._skipped_runs:
            self._skipped_runs.discard(str(run_id))
            return
        node_name = self._run_node_names.pop(str(run_id), None) or _extract_node_name(
            {}, tags, metadata
        )
        if not node_name or node_name.startswith("RunnableSequence"):
            return

        if isinstance(error, asyncio.CancelledError):
            # Cooperative cancellation (per-call stop, run drain, pod
            # teardown), not a node failure: plain cancels carry an empty
            # message, and langgraph's background executor cancels leftover
            # tasks with a bare object() sentinel whose str() is an opaque
            # address — neither belongs on an error row.
            interrupt_fn = getattr(self._callback, "on_node_interrupted", None)
            if interrupt_fn is not None:
                interrupt_fn(
                    node_name=node_name,
                    run_id=str(run_id),
                    parent_run_id=str(parent_run_id) if parent_run_id else None,
                    metadata=metadata,
                )
            else:
                # Backward compat: callbacks that predate on_node_interrupted
                # fall back to on_node_end so the node timing is closed out.
                # The recorded status will be "success" rather than
                # "interrupted" — acceptable until the callback is upgraded.
                logger.warning(
                    "%s does not implement on_node_interrupted; CancelledError on node %r "
                    "recorded as success. Subclass NullExecutionCallback to fix this.",
                    type(self._callback).__name__,
                    node_name,
                )
                self._callback.on_node_end(
                    node_name=node_name,
                    outputs={},
                    run_id=str(run_id),
                    parent_run_id=str(parent_run_id) if parent_run_id else None,
                    metadata=metadata,
                )
            return

        if isinstance(error, GraphInterrupt):
            suspend_fn = getattr(self._callback, "on_node_suspend", None)
            if suspend_fn is not None:
                suspend_fn(
                    node_name=node_name,
                    run_id=str(run_id),
                    parent_run_id=str(parent_run_id) if parent_run_id else None,
                    metadata=metadata,
                )
            else:
                # Backward compat: callbacks that predate on_node_suspend fall back to
                # on_node_end so the node timing is closed out. The recorded status will
                # be "success" rather than "suspend" — acceptable until the callback is
                # upgraded to implement on_node_suspend or NullExecutionCallback.
                logger.warning(
                    "%s does not implement on_node_suspend; GraphInterrupt on node %r "
                    "recorded as success. Subclass NullExecutionCallback to fix this.",
                    type(self._callback).__name__,
                    node_name,
                )
                self._callback.on_node_end(
                    node_name=node_name,
                    outputs={},
                    run_id=str(run_id),
                    parent_run_id=str(parent_run_id) if parent_run_id else None,
                    metadata=metadata,
                )
            return

        self._callback.on_node_error(
            node_name=node_name,
            error=str(error),
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            metadata=metadata,
        )


def _extract_node_name(
    serialized: dict[str, Any],
    tags: list[str] | None,
    metadata: dict[str, Any] | None,
) -> str | None:
    """Extract the node name from LangGraph callback parameters."""
    if metadata and "langgraph_node" in metadata:
        return metadata["langgraph_node"]
    if tags:
        for tag in tags:
            if not tag.startswith("seq:step:"):
                return tag
    if serialized and "name" in serialized:
        return serialized["name"]
    return None


def _is_nested_run(tags: list[str] | None, metadata: dict[str, Any] | None) -> bool:
    """True for a nested chain running inside a node, not the node itself.

    Pregel stamps every node task run with a ``graph:step:N`` tag. A nested
    chain (the model-call sequence, a lambda, a router) inherits the node's
    ``langgraph_node`` metadata but never the tag — so metadata alone cannot
    tell a node execution from its own innards. Port of the TS twin's
    ``isRouterEvaluation``.
    """
    if not metadata or "langgraph_node" not in metadata:
        return False
    return not any(tag.startswith("graph:step:") for tag in (tags or []))


@dataclass
class _GraphTask:
    node: str
    step: int
    span_context: SpanContext | None = None


class GraphBranchLinkHandler(BaseCallbackHandler):
    """Links each LangGraph task that follows parallel branches to those branches.

    A span has one parent, so a node that runs after a fan-out (a join, or any
    follow-on step) cannot show every branch it waited for. Pregel starts step
    N only after all of step N-1 has finished, so when step N-1 ran more than
    one task, each step-N task gets an OTel span link to those tasks. An
    explicit ``join:a+b:c`` trigger narrows the links to the latest run of each
    named node. Serial steps get no links. Links carry only span IDs, the node
    name and the step.

    The node spans belong to the OpenInference instrumentor; this handler only
    adds links to them while they are open. It runs before the instrumentor's
    own handler, so the span may not exist yet at chain start — in that case
    the link is added at chain end, which still precedes the span ending.
    """

    # Inline in async runs too, so this handler keeps its place ahead of the
    # instrumentor's and chain end still lands while the span is open.
    run_inline = True

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        # graph run id -> step -> tasks that ran in that step
        self._steps: dict[UUID, dict[int, list[_GraphTask]]] = {}
        # task run id -> (graph run id, task, triggers) until the task's span is linked
        self._pending: dict[UUID, tuple[UUID, _GraphTask, tuple[str, ...]]] = {}

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if (
            parent_run_id is None
            or not metadata
            or "langgraph_node" not in metadata
            or not isinstance(metadata.get("langgraph_step"), int)
            or _is_nested_run(tags, metadata)
        ):
            return
        task = _GraphTask(
            node=str(metadata["langgraph_node"]), step=metadata["langgraph_step"]
        )
        triggers = tuple(str(t) for t in metadata.get("langgraph_triggers") or ())
        with self._lock:
            self._steps.setdefault(parent_run_id, {}).setdefault(task.step, []).append(
                task
            )
            self._pending[run_id] = (parent_run_id, task, triggers)
        self._link(run_id)

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id)

    def on_chain_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._finish(run_id)

    def _finish(self, run_id: UUID) -> None:
        self._link(run_id)
        with self._lock:
            self._pending.pop(run_id, None)
            # The graph run itself ending releases its step history.
            self._steps.pop(run_id, None)

    def _link(self, run_id: UUID) -> None:
        with self._lock:
            entry = self._pending.get(run_id)
        if entry is None:
            return
        graph_run_id, task, triggers = entry
        span = _instrumented_span(run_id)
        if span is None or not span.is_recording():
            return
        with self._lock:
            if self._pending.pop(run_id, None) is None:
                return
            task.span_context = span.get_span_context()
            predecessors = _predecessors(
                self._steps.get(graph_run_id, {}), task.step, triggers
            )
        for pred in predecessors:
            if pred.span_context is not None:
                span.add_link(
                    pred.span_context,
                    {"langgraph.node": pred.node, "langgraph.step": pred.step},
                )


def _predecessors(
    steps: dict[int, list[_GraphTask]], step: int, triggers: tuple[str, ...]
) -> list[_GraphTask]:
    """Tasks a step-``step`` task follows, or none when it follows a serial step."""
    for trigger in triggers:
        # join:<a>+<b>:<target> — the edge waits for every named node.
        if trigger.startswith("join:"):
            names = trigger.split(":")[1].split("+")
            earlier = [t for s in sorted(steps) if s < step for t in steps[s]]
            latest = {t.node: t for t in earlier if t.node in names}
            return [latest[n] for n in names if n in latest]
    previous = steps.get(step - 1, [])
    return list(previous) if len(previous) > 1 else []


def _instrumented_span(run_id: UUID) -> Span | None:
    from openinference.instrumentation.langchain import LangChainInstrumentor

    return LangChainInstrumentor().get_span(run_id)
