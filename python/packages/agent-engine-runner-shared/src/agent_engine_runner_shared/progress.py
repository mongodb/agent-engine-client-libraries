"""emit / emit_step — send a mid-execution chunk from within a tool function.

Tool functions running inside a Tool Pod call these to stream events to the
client while the tool is still executing. Both execution_id and oe_url are
injected into Python contextvars by the Tool Pod server before the tool runs.

The call is best-effort: network errors are logged and swallowed so a failed
chunk emit never aborts the tool.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from typing import Callable, Optional

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
from agent_engine_runner_shared.tls_client import create_httpx_client_with_tls

logger = logging.getLogger(__name__)

# Short timeout: chunk emits are best-effort fire-and-forget. A slow or
# unreachable OE must not delay the tool function itself.
_PROGRESS_TIMEOUT_SECS = 5.0

# Module-level client so the underlying TCP/TLS connection to the OE is pooled
# and reused across emit() calls. httpx.Client is thread-safe for request
# operations; atexit closes it deterministically on interpreter shutdown.
# Lazy-initialized with TLS configuration based on the OE_URL.
# CRITICAL: _client_lock protects concurrent lazy-init and rotation from threads
# spawned by asyncio.to_thread (docstring says emit() runs on concurrent threads).
_client: Optional[httpx.Client] = None
_client_oe_url: Optional[str] = None
_client_cert_mtimes: Optional[tuple[float, float, float]] = None
_client_lock = threading.Lock()


def _get_cert_mtimes() -> Optional[tuple[float, float, float]]:
    """Get modification times of TLS cert files for rotation detection."""
    try:
        cert_path = os.environ.get("TLS_CERT_PATH", "")
        key_path = os.environ.get("TLS_KEY_PATH", "")
        ca_path = os.environ.get("TLS_CA_CERT_PATH", "")

        # Only track mtimes if we're using file-based certs (container mode)
        # VM mode uses env vars with PEM content, which can't detect rotation
        if not (cert_path and key_path and ca_path):
            return None

        return (
            os.path.getmtime(cert_path),
            os.path.getmtime(key_path),
            os.path.getmtime(ca_path),
        )
    except (OSError, ValueError):
        # Cert files don't exist or aren't readable - not using TLS
        return None


def _get_client(oe_url: str) -> httpx.Client:
    """Get or create the module-level HTTP client, configured for the OE URL.

    Automatically invalidates the cached client if TLS certificate files have
    been rotated (based on mtime). This ensures the client picks up renewed
    certificates without requiring a pod restart.

    Thread-safe: protects lazy initialization and cert rotation from concurrent
    threads spawned by asyncio.to_thread when emit() is called from async tools.
    """
    global _client, _client_oe_url, _client_cert_mtimes

    # Check mtimes outside lock (read-only, no mutation)
    current_mtimes = _get_cert_mtimes()

    with _client_lock:
        # If the OE URL changed (unlikely but possible in testing), recreate the client
        if _client is not None and _client_oe_url != oe_url:
            _client.close()
            _client = None

        # Check if certs have been rotated (container mode only)
        if _client is not None and current_mtimes is not None:
            if _client_cert_mtimes != current_mtimes:
                logger.info(
                    "TLS certificates rotated, invalidating cached HTTP client",
                    extra={"old_mtimes": _client_cert_mtimes, "new_mtimes": current_mtimes},
                )
                _client.close()
                _client = None

        if _client is None:
            _client = create_httpx_client_with_tls(oe_url, _PROGRESS_TIMEOUT_SECS)
            _client_oe_url = oe_url
            _client_cert_mtimes = current_mtimes
            atexit.register(_client.close)

        return _client


def _post_chunk_owner_first(
    client: httpx.Client,
    *,
    service_url: str,
    owner_url: Optional[str],
    payload: dict,
    on_owner_failure: Optional[Callable[[], None]] = None,
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
    later emits.
    """
    if owner_url is not None:
        if owner_delivered(
            client,
            owner_url,
            payload,
            lambda msg: logger.debug("emit: %s", msg),
        ):
            return
        if on_owner_failure is not None:
            on_owner_failure()
    response = post_without_reading_response_body(client, service_url, payload)
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
        client = _get_client(oe_url)
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
