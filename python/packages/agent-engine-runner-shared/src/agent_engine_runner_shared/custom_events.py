"""Runtime-transparent custom-event emission across Atlas Agent Engine runtimes.

Authors call :func:`emit_custom_event` without knowing which runtime is
hosting the code (AER agent process vs Tool Pod / function mode). The active
runtime installs exactly one :class:`CustomEventTransport` before invoking
tenant code; the helper dispatches to that transport.

Atlas Agent Engine transports an opaque author-final JSON object as ``custom_event``.
It does not interpret keys — customers who need a shaped vocabulary (e.g.
Holly ``{event, data}``) build that object in their own helper.

Framework adapters (e.g. LangGraph in ``agent-engine-sdk-langgraph``) own how AER
delivers events. This package only owns the author API, the install/dispatch
hook, and the Tool Pod OE HTTP transport.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator, Optional, Protocol, runtime_checkable

from pydantic import ConfigDict, JsonValue, TypeAdapter, ValidationError

from agent_engine_runner_shared.context import (
    current_execution_id,
    current_oe_url,
    get_current_oe_owner_url,
    report_oe_owner_url_failure,
)
from agent_engine_runner_shared.progress import _get_client, _post_chunk_owner_first
from agent_engine_runner_shared.server.chunk_types import CUSTOM_EVENT

logger = logging.getLogger(__name__)
_JSON_OBJECT_ADAPTER = TypeAdapter(dict[str, JsonValue], config=ConfigDict(allow_inf_nan=False))


@runtime_checkable
class CustomEventTransport(Protocol):
    """Request-scoped delivery path for author custom events.

    The hosting runtime installs one transport per invocation. Tool Pods use
    :class:`OEHTTPCustomEventTransport`; AER framework adapters install their
    own implementation.
    """

    def emit_sync(self, payload: Mapping[str, JsonValue]) -> None:
        """Deliver one author-final JSON object from synchronous code."""

    async def emit(self, payload: Mapping[str, JsonValue]) -> None:
        """Deliver one author-final JSON object from asynchronous code."""


_current_custom_event_transport: ContextVar[Optional[CustomEventTransport]] = ContextVar(
    "current_custom_event_transport", default=None
)


def get_custom_event_transport() -> Optional[CustomEventTransport]:
    """Return the transport installed for this request, or None."""
    return _current_custom_event_transport.get()


def set_custom_event_transport(
    transport: Optional[CustomEventTransport],
) -> Token[Optional[CustomEventTransport]]:
    """Install ``transport`` for the current async task.

    Returns a :class:`~contextvars.Token` that :func:`clear_custom_event_transport`
    uses to restore the previous value (nested installs / concurrent tasks).
    """
    return _current_custom_event_transport.set(transport)


def clear_custom_event_transport(
    token: Token[Optional[CustomEventTransport]],
) -> None:
    """Restore the previous transport via ``ContextVar.reset(token)``."""
    _current_custom_event_transport.reset(token)


@contextmanager
def custom_event_transport(
    transport: CustomEventTransport,
) -> Iterator[CustomEventTransport]:
    """Install ``transport`` for the block, then restore the previous value."""
    token = set_custom_event_transport(transport)
    try:
        yield transport
    finally:
        clear_custom_event_transport(token)


def _normalize_payload(payload: object) -> dict[str, JsonValue]:
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"emit_custom_event expects a JSON-object mapping (got {type(payload).__name__})"
        )
    try:
        return _JSON_OBJECT_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise TypeError(
            "emit_custom_event expects a JSON-serializable object with string keys"
        ) from exc


def emit_custom_event_sync(payload: Mapping[str, Any]) -> None:
    """Emit an author-final JSON object from synchronous tenant code.

    Use :func:`emit_custom_event` from asynchronous code. This synchronous
    sibling is intended for ordinary synchronous Tool Pod handlers and
    synchronous LangGraph nodes.
    """
    normalized = _normalize_payload(payload)
    transport = _current_custom_event_transport.get()
    if transport is None:
        return
    transport.emit_sync(normalized)


async def emit_custom_event(payload: Mapping[str, Any]) -> None:
    """Emit an author-final JSON object through the installed runtime transport.

    Atlas Agent Engine does not interpret keys. Customers who need a shaped vocabulary
    (for example Holly ``{"event": ..., "data": ...}``) build that object
    themselves before calling this helper.

    Use :func:`emit_custom_event_sync` from synchronous tenant code. No-op when
    no transport is installed (outside an execution context).
    """
    normalized = _normalize_payload(payload)
    transport = _current_custom_event_transport.get()
    if transport is None:
        return
    await transport.emit(normalized)


class OEHTTPCustomEventTransport:
    """Tool Pod / function-mode transport: POST final ``custom_event`` chunks to OE.

    Delivery is awaited with a bounded timeout and is non-fatal: delivery
    failures are logged and swallowed so a progress event cannot fail the tool.
    """

    def _post(self, payload: Mapping[str, JsonValue]) -> None:
        execution_id = current_execution_id.get()
        oe_url = current_oe_url.get()
        # Latch-aware: None once a previous owner pre-attempt in this
        # execution failed (shared latch with progress.emit).
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
                    "chunk_type": CUSTOM_EVENT,
                    "custom_event": dict(payload),
                },
                on_owner_failure=report_oe_owner_url_failure,
            )
        except Exception as exc:  # noqa: BLE001 - best-effort; never fail the tool
            logger.debug("emit_custom_event: failed to send chunk: %s", exc)

    def emit_sync(self, payload: Mapping[str, JsonValue]) -> None:
        self._post(payload)

    async def emit(self, payload: Mapping[str, JsonValue]) -> None:
        # Reuse the TLS-aware pooled sync client from progress.emit; offload
        # the blocking POST so async tool handlers stay non-blocking.
        await asyncio.to_thread(self._post, payload)


class FeatureOffCustomEventTransport:
    """Fail closed when ``use_custom_parser`` is disabled.

    Runtime-neutral rejecting transport shared by AER streaming (when Atlas Agent Engine
    does not subscribe to LangGraph ``custom``) and Tool Pod / function mode
    (when OE POSTs would otherwise succeed while Gateway strips client
    delivery). Authors must not observe a successful emit when the workspace
    cannot deliver custom events to clients.
    """

    _MESSAGE = (
        "emit_custom_event requires features.use_custom_parser and a registered "
        "@app.output_parser; Atlas Agent Engine does not deliver custom events without "
        "that opt-in"
    )

    def emit_sync(self, payload: Mapping[str, JsonValue]) -> None:
        raise RuntimeError(self._MESSAGE)

    async def emit(self, payload: Mapping[str, JsonValue]) -> None:
        raise RuntimeError(self._MESSAGE)


def install_tool_custom_event_transport(
    *, enabled: bool = True
) -> Token[Optional[CustomEventTransport]]:
    """Install Tool Pod / function-mode custom-event transport; return reset token.

    When ``enabled`` is false (workspace does not have ``use_custom_parser``),
    install a rejecting transport so emit fails closed instead of POSTing an
    event that Gateway will strip for clients.
    """
    transport: CustomEventTransport = (
        OEHTTPCustomEventTransport() if enabled else FeatureOffCustomEventTransport()
    )
    return set_custom_event_transport(transport)
