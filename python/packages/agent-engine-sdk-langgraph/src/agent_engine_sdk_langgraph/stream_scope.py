"""Per-``__anext__`` ContextVar enter/exit pairing for LangGraph streaming.

A ``ContextVar.Token`` from ``set()`` may only be ``reset()`` in the same
Context that created it. Wrapping the whole async generator in
custom-event / identity scopes enters on the first ``__anext__``
and exits on ``aclose`` / exhaustion; if those two run in different Contexts
(``asyncio.wait_for`` child Task, cancel/``aclose`` from another task), reset
raises ``ValueError: Token was created in a different Context`` and the
custom-event ContextVar leaks onto the next request on that worker.

AER already uses ``asyncio.timeout`` (same-task) for the post-terminal drain
instead of ``wait_for``. The adapter still scopes each ``anext`` so enter and
reset always happen in whichever task drove that one step, and rebinds the
compiled-path resolver across iterations.

Do not "simplify" this into a generator-lifetime ``async with``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from agent_engine_sdk import StreamEvent

from agent_engine_sdk_langgraph.execution_session import ExecutionSession
from agent_engine_runner_shared.custom_events import (
    FeatureOffCustomEventTransport,
    custom_event_transport,
)
from agent_engine_runner_shared.workflow import operation_path_resolver_scope

_logger = logging.getLogger(__name__)

TERMINAL_AGENT_UPDATE_DRAIN_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class GraphStreamItem:
    """One ``astream(subgraphs=True, stream_mode=[...])`` yield.

    LangGraph's namespace is a tuple of ``"<node>:<task_id>"`` segments
    (``()`` for the root). A non-tuple namespace is treated as root so
    callers never branch on a malformed item.
    """

    namespace: tuple[str, ...]
    stream_mode: str
    payload: Any


class PostTerminalDrain:
    """Bounded wait after the first terminal root-agent ``updates`` payload.

    LangGraph 1.2.x can leave the iterator open after that update. The
    next ``anext`` is timed so a hung iterator cannot stall the stream;
    later payloads are not treated as a second terminal.
    """

    def __init__(self) -> None:
        self._awaiting = False
        self._seen = False

    def next_uses_timeout(self, *, enabled: bool) -> bool:
        return enabled and self._awaiting

    def received_item(self) -> None:
        self._awaiting = False

    def consider(self, is_terminal: bool) -> None:
        if self._seen or not is_terminal:
            return
        self._awaiting = True
        self._seen = True


def _graph_stream_item(raw: Any) -> GraphStreamItem:
    namespace, stream_mode, payload = raw
    if not isinstance(namespace, tuple):
        namespace = ()
    return GraphStreamItem(namespace, stream_mode, payload)


async def aclose_stream_iter(stream_iter: Any) -> None:
    """Close an async iterator when it supports generator-style ``aclose``."""
    aclose = getattr(stream_iter, "aclose", None)
    if aclose is None:
        return
    try:
        result = aclose()
        if inspect.isawaitable(result):
            await result
    except Exception:
        _logger.warning("Failed to close LangGraph stream iterator", exc_info=True)


async def anext_graph_item(
    stream_iter: Any, *, drain_timeout: bool
) -> GraphStreamItem | None:
    """Return the next astream item, or None at end-of-stream / drain timeout."""
    next_item = stream_iter.__anext__()
    try:
        if drain_timeout:
            try:
                async with asyncio.timeout(TERMINAL_AGENT_UPDATE_DRAIN_TIMEOUT_S):
                    return _graph_stream_item(await next_item)
            except TimeoutError:
                _logger.debug(
                    "Timed out waiting for LangGraph event after "
                    "terminal agent update; finishing stream"
                )
                return None
        return _graph_stream_item(await next_item)
    except StopAsyncIteration:
        return None


async def scoped_anext(
    stream: AsyncIterator[StreamEvent],
    session: ExecutionSession,
) -> StreamEvent:
    """Drive one ``__anext__`` under custom-event and identity ContextVars.

    ``emit_custom_event`` looks up a ContextVar. None is a silent no-op, so
    every stream step installs a delivery object: the live LangGraph writer
    when the workspace opted in, otherwise a rejector. This is not a
    platform-event transport; token/result/suspend frames do not go through it.

    The session's compiled-path resolver is rebound for each durable step.
    """
    # output_parser imports aclose_stream_iter from this module.
    from agent_engine_sdk_langgraph.output_parser import LangGraphCustomEventTransport

    custom_event_delivery = (
        LangGraphCustomEventTransport()
        if session.use_custom_parser
        else FeatureOffCustomEventTransport()
    )
    resolver = session.subgraph_resolver()
    with custom_event_transport(custom_event_delivery):
        if resolver is None:
            return await anext(stream)
        # AER can resume this async generator from a different asyncio task
        # while draining terminal events. ContextVar tokens belong to the task
        # context that created them, so a scope around the generator's full
        # lifetime could be entered in one task and reset in another. Scoping
        # each __anext__ keeps both operations in the task that executes that
        # iteration while still covering all graph work.
        with operation_path_resolver_scope(resolver):
            return await anext(stream)
