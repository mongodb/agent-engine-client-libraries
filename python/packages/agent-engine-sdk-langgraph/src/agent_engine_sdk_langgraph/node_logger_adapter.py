"""LangGraph callback adapter for BaseExecutionCallback.

Bridges LangChain's BaseCallbackHandler interface to the framework-neutral
BaseExecutionCallback protocol. Extracts node names from LangGraph-specific
metadata and converts UUID run_ids to strings.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.errors import GraphInterrupt
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
