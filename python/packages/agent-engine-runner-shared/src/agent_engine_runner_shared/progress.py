"""emit / emit_step — send a mid-execution chunk from within a tool function.

Tool functions running inside a Tool Pod call these to stream events to the
client while the tool is still executing. Both execution_id and oe_url are
injected into Python contextvars by the Tool Pod server before the tool runs.

The call is best-effort: network errors are logged and swallowed so a failed
chunk emit never aborts the tool.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import httpx

from agent_engine_runner_shared.context import (
    current_execution_id,
    current_oe_url,
    get_current_oe_owner_url,
    report_oe_owner_url_failure,
)
from agent_engine_runner_shared.owner_callback import (
    owner_delivered,
    post_without_reading_response_body,
)
from agent_engine_runner_shared.server.chunk_types import (
    DONE,
    ERROR,
    STEP,
    SUBAGENT_END,
    SUBAGENT_START,
    TEXT,
)
from agent_engine_runner_shared.tls_client import get_shared_client
from agent_engine_runner_shared.utils import PROGRESS_TIMEOUT_SECS

logger = logging.getLogger(__name__)


def _post_chunk_owner_first(
    client: httpx.Client,
    *,
    service_url: str,
    owner_url: Optional[str],
    payload: dict,
    on_owner_failure: Optional[Callable[[], None]] = None,
    timeout: Any = httpx.USE_CLIENT_DEFAULT,
) -> None:
    """POST ``payload`` to ``owner_url`` first, falling back to ``service_url``.

    Best-effort single-shot delivery shared by :func:`emit` and the Tool Pod
    custom-event transport. Both are fire-and-forget, so this raises on a
    service failure and the caller swallows. Owner preference per
    :func:`agent_engine_runner_shared.owner_callback.owner_delivered`: one best-effort
    attempt against the replica-specific owner URL; any owner failure falls
    back to the trusted service URL and, when ``on_owner_failure`` is
    provided, runs it exactly once before the fallback (mirrors
    ``post_json_with_retries``) so callers can stop offering the owner on
    later emits. ``timeout`` overrides the shared client's default per request.
    """
    if owner_url is not None:
        if owner_delivered(
            client,
            owner_url,
            payload,
            lambda msg: logger.debug("emit: %s", msg),
            timeout=timeout,
        ):
            return
        if on_owner_failure is not None:
            on_owner_failure()
    response = post_without_reading_response_body(client, service_url, payload, timeout=timeout)
    response.raise_for_status()


# Chunk types reserved for infrastructure use. Tool code emitting these would
# either close the SSE stream prematurely (DONE/ERROR are terminal) or spoof
# infrastructure-only chunk types (TEXT/SUBAGENT_*).
_RESERVED_EVENTS = frozenset({DONE, ERROR, TEXT, SUBAGENT_START, SUBAGENT_END})


def emit(event: str, data: str) -> None:
    """Emit a chunk of the given event type to the client stream.

    General-purpose API. For the common textual-step case, prefer
    :func:`emit_step`.

    Args:
        event: Chunk type identifier (e.g. ``"step"``). Reserved infrastructure
            event types (``done``, ``error``, ``text``, ``subagent_start``,
            ``subagent_end``) raise ``ValueError`` to prevent tool code from
            prematurely closing or spoofing the stream. ``"custom_event"`` is
            not this API — use :func:`agent_engine_runner_shared.emit_custom_event` and
            ``features.use_custom_parser`` in ``agent.yaml``. Gateway drops the
            ``"custom_event"`` chunk_type unless that feature is on.
        data: Payload string for the event.

    Raises:
        ValueError: if ``event`` is a reserved infrastructure event type.

    Transport failures are logged at DEBUG and swallowed — best-effort delivery
    must never abort the tool function. No-op outside an execution context.
    Synchronous; async tool functions must wrap in ``asyncio.to_thread``.
    """
    if event in _RESERVED_EVENTS:
        raise ValueError(
            f"emit: event {event!r} is reserved for infrastructure use; "
            f"choose a non-terminal event type such as 'step'"
        )

    execution_id = current_execution_id.get()
    oe_url = current_oe_url.get()
    # Latch-aware: None once a previous owner pre-attempt in this execution
    # failed, so repeated emits stop re-paying the owner timeout.
    owner_url = get_current_oe_owner_url()

    if not execution_id or not oe_url:
        return

    try:
        client = get_shared_client(oe_url)
        _post_chunk_owner_first(
            client,
            service_url=f"{oe_url.rstrip('/')}/stream/chunk",
            owner_url=f"{owner_url.rstrip('/')}/stream/chunk" if owner_url else None,
            payload={
                "execution_id": execution_id,
                "chunk_type": event,
                "content": data,
                "metadata": {},
            },
            on_owner_failure=report_oe_owner_url_failure,
            timeout=PROGRESS_TIMEOUT_SECS,
        )
    except Exception as exc:
        # Best-effort: a failed chunk must never abort the tool function.
        # Catches transport errors, httpx.InvalidURL, AND ValueError from
        # fail-closed TLS (HTTPS URL with missing certs). The last case can
        # happen during deployment ordering issues (flag enabled before certs
        # mounted). Swallow all exceptions - emit() is fire-and-forget logging.
        # DEBUG so operators can diagnose OE connectivity without noise.
        logger.debug("emit: failed to send chunk: %s", exc)


def emit_step(message: str) -> None:
    """Emit a textual progress step from within a running tool function.

    Convenience wrapper around ``emit(event="step", data=message)``.

    Example::

        from agent_engine_runner_shared import emit_step

        @app.tool(is_local=False)
        def crawl_website(url: str) -> str:
            emit_step("Fetching page...")
            html = fetch(url)
            emit_step(f"Parsing links from {url}...")
            return parse(html)
    """
    emit(event=STEP, data=message)
