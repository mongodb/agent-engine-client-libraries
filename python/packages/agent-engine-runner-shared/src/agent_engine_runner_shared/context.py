"""
Per-execution context management using contextvars.

This module provides thread-safe, coroutine-safe context for execution state.
Each concurrent execution gets its own isolated context, preventing cross-execution
contamination of execution_id, wrapper, and OE URL.

Best Practice: All context variables are available for log correlation.

Usage:
    from agent_engine_runner_shared.context import (
        current_execution_id,
        current_wrapper,
        current_oe_url,
        set_execution_context,
        clear_execution_context,
    )

    # Set context at start of execution
    tokens = set_execution_context(execution_id, wrapper, oe_url)

    try:
        # During execution, access via .get()
        exec_id = current_execution_id.get()
        wrapper = current_wrapper.get()
    finally:
        # Always clear context when done
        clear_execution_context(tokens)
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from agent_engine_runner_shared.error_reporting import _redact_text

if TYPE_CHECKING:
    from agent_engine_runner_shared.models import ToolAuthorization

# Per-execution context variables
current_execution_id: ContextVar[Optional[str]] = ContextVar("current_execution_id", default=None)
current_wrapper: ContextVar[Optional[Any]] = ContextVar("current_wrapper", default=None)
current_oe_url: ContextVar[Optional[str]] = ContextVar("current_oe_url", default=None)
# Validated replica-specific OE owner URL for callback fallback. Set
# only when the request supplied a valid owner URL; None otherwise. Consumed by
# the Tool Pod / in-process progress and custom-event emitters.
current_oe_owner_url: ContextVar[Optional[str]] = ContextVar("current_oe_owner_url", default=None)


@dataclass
class OwnerUrlFailureState:
    """One-way owner-failure latch: once an owner pre-attempt fails, later
    owner-preferring posts in this execution skip the owner URL entirely
    instead of re-paying the pre-attempt timeout on every emit. A mutable
    holder for the same reason as ``_SessionFinishState``: emit() may run on a
    worker thread via ``asyncio.to_thread``, whose copied context shares this
    object reference while a ``ContextVar.set`` there would be invisible to
    the parent. No lock: the only write is the one-way ``failed = True``,
    atomic under the GIL. Mirrors the AER stream path's per-execution
    owner-URL discard."""

    failed: bool = False


current_owner_url_failure: ContextVar[Optional["OwnerUrlFailureState"]] = ContextVar(
    "current_owner_url_failure", default=None
)
# Platform trace ID for log correlation.
current_trace_id: ContextVar[Optional[str]] = ContextVar("current_trace_id", default=None)
# Caller-supplied request identity; independent of platform trace correlation.
current_request_id: ContextVar[Optional[str]] = ContextVar("current_request_id", default=None)
# User ID for the current execution
current_user_id: ContextVar[Optional[str]] = ContextVar("current_user_id", default=None)
# Session/thread ID for conversation continuity
current_session_id: ContextVar[Optional[str]] = ContextVar("current_session_id", default=None)
# Workspace ID for workspace-scoped attribution
current_workspace_id: ContextVar[Optional[str]] = ContextVar("current_workspace_id", default=None)
# Delegated credential injected by OE for tool execution.
current_authorization: ContextVar[Optional["ToolAuthorization"]] = ContextVar(
    "current_authorization", default=None
)
# Custom headers forwarded from the caller (X-Mdb-Agent-Engine-Custom-* headers, prefix-stripped and lowercased)
current_custom_headers: ContextVar[Optional[Dict[str, str]]] = ContextVar(
    "current_custom_headers", default=None
)
current_execution_metadata: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "current_execution_metadata", default=None
)
# Opaque caller-provided payload for the current execution (the invocation
# request body beyond `message`). Agent code reads it via get_current_payload().
current_payload: ContextVar[Optional[Dict[str, Any]]] = ContextVar("current_payload", default=None)


@dataclass
class _SessionFinishState:
    """Mutable holder so a request from a copied context (LangGraph node,
    worker thread) is visible to the parent frame reading it later.

    Shared across real OS threads (LangGraph worker threads), not just
    copied contexts - request_session_finish()'s read-modify-write of
    `requested` and `closed` must be adjudicated together under one lock.
    Reading either field unlocked while a writer is mid-transition on the
    other reintroduces the exact race the lock exists to close, so every
    access (read or write, of either field) goes through `lock`.
    """

    requested: bool = False
    closed: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def close(self) -> None:
        """Close the latch: nothing outside this run can finish anymore.
        Called from the AER's `finally` once the execute frame ends -
        keeps the lock private to the holder rather than making callers
        take it themselves."""
        with self.lock:
            self.closed = True


# Holder for a session-finish request raised anywhere during the current
# execution. A ContextVar.set() inside a copied context is invisible to the
# parent, but mutating the holder object obtained from the parent frame is not.
current_session_finish: ContextVar[Optional["_SessionFinishState"]] = ContextVar(
    "current_session_finish", default=None
)


@dataclass
class _SuspendRequestState:
    """Out-of-band HITL suspend signal for the current tool call.

    Only ``SuspendPayload.to_json`` — the tool author's own code — writes
    ``payload``, so untrusted tool-result content, which cannot reach this
    frame, can never forge a suspend. A mutable holder for the same reason as
    ``_SessionFinishState``: a sync tool off-loaded to a worker thread mutates
    the shared object, which a copied-context ``ContextVar.set`` would not."""

    payload: Optional[Dict[str, Any]] = None


# Execution-level holder used by Tool Pod calls. In-process AER calls replace it
# with an isolated child holder for each approved tool invocation.
current_suspend_request: ContextVar[Optional["_SuspendRequestState"]] = ContextVar(
    "current_suspend_request", default=None
)


class SessionFinishStatus(str, Enum):
    """Outcome of a request_session_finish() call."""

    REQUESTED = "requested"  # this call recorded the request
    ALREADY_REQUESTED = "already_requested"  # a previous call in this run did
    UNAVAILABLE = "unavailable"  # no execution context (local dev, tool pod)


ContextTokens = Tuple[
    Token[Optional[str]],
    Token[Optional[Any]],
    Token[Optional[str]],
    Token[Optional[str]],
    Token[Optional[str]],
    Token[Optional[str]],
    Token[Optional[str]],
    Token[Optional[str]],
    Token[Optional["ToolAuthorization"]],
    Token[Optional[Dict[str, str]]],
    Token[Optional[Dict[str, Any]]],
    Token[Optional[Dict[str, Any]]],
    Token[Optional["_SessionFinishState"]],
    Token[Optional["_SuspendRequestState"]],
    Token[Optional["OwnerUrlFailureState"]],
    Token[Optional[str]],
]


def set_execution_context(
    execution_id: str,
    wrapper: Any,
    oe_url: str,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    custom_headers: Optional[Dict[str, str]] = None,
    authorization: Optional["ToolAuthorization"] = None,
    payload: Optional[Dict[str, Any]] = None,
    oe_owner_url: Optional[str] = None,
    owner_url_failure: Optional[OwnerUrlFailureState] = None,
    *,
    trace_id: Optional[str] = None,
) -> ContextTokens:
    """
    Set execution context for the current async task.

    Args:
        execution_id: The execution ID for this request
        wrapper: The SecureToolWrapper instance
        oe_url: The OE callback URL
        trace_id: Optional originating platform trace ID for log correlation
        request_id: Optional request ID, auto-generated as ``req-<12 hex>`` when omitted
        user_id: Optional user ID for the current execution
        session_id: Optional session/thread ID for conversation continuity
        workspace_id: Optional workspace ID for guardrails/cost attribution
        custom_headers: Optional caller-provided custom headers (X-Mdb-Agent-Engine-Custom-* prefix-stripped and lowercased)
        authorization: Optional delegated credential context for tool execution
        payload: Optional opaque caller-provided invocation payload
        oe_owner_url: Optional validated replica-specific OE owner URL for callback fallback

    Returns:
        Tuple of tokens for resetting context later
    """
    existing_execution_id = current_execution_id.get()
    existing_owner_failure = current_owner_url_failure.get()
    exec_token = current_execution_id.set(execution_id)
    wrap_token = current_wrapper.set(wrapper)
    url_token = current_oe_url.set(oe_url)
    owner_url_token = current_oe_owner_url.set(oe_owner_url)
    trace_token = current_trace_id.set(trace_id)
    req_token = current_request_id.set(request_id or f"req-{uuid.uuid4().hex[:12]}")
    user_token = current_user_id.set(user_id)
    session_token = current_session_id.set(session_id)
    workspace_token = current_workspace_id.set(workspace_id)
    authorization_token = current_authorization.set(authorization)
    custom_headers_token = current_custom_headers.set(custom_headers)
    metadata_token = current_execution_metadata.set({})
    payload_token = current_payload.set(payload)
    session_finish_token = current_session_finish.set(_SessionFinishState())
    suspend_request_token = current_suspend_request.set(_SuspendRequestState())
    shared_owner_failure = owner_url_failure
    if shared_owner_failure is None and existing_execution_id == execution_id:
        shared_owner_failure = existing_owner_failure
    owner_failure_token = current_owner_url_failure.set(
        shared_owner_failure or OwnerUrlFailureState()
    )
    return (
        exec_token,
        wrap_token,
        url_token,
        req_token,
        user_token,
        session_token,
        workspace_token,
        owner_url_token,
        authorization_token,
        custom_headers_token,
        metadata_token,
        payload_token,
        session_finish_token,
        suspend_request_token,
        owner_failure_token,
        trace_token,
    )


def clear_execution_context(tokens: ContextTokens) -> None:
    """
    Clear execution context using the tokens from set_execution_context.

    Args:
        tokens: The tuple of tokens returned by set_execution_context
    """
    (
        exec_token,
        wrap_token,
        url_token,
        req_token,
        user_token,
        session_token,
        workspace_token,
        owner_url_token,
        authorization_token,
        custom_headers_token,
        metadata_token,
        payload_token,
        session_finish_token,
        suspend_request_token,
        owner_failure_token,
        trace_token,
    ) = tokens
    current_execution_id.reset(exec_token)
    current_wrapper.reset(wrap_token)
    current_oe_url.reset(url_token)
    current_oe_owner_url.reset(owner_url_token)
    current_trace_id.reset(trace_token)
    current_request_id.reset(req_token)
    current_user_id.reset(user_token)
    current_session_id.reset(session_token)
    current_workspace_id.reset(workspace_token)
    current_authorization.reset(authorization_token)
    current_custom_headers.reset(custom_headers_token)
    current_execution_metadata.reset(metadata_token)
    current_payload.reset(payload_token)
    current_session_finish.reset(session_finish_token)
    current_suspend_request.reset(suspend_request_token)
    current_owner_url_failure.reset(owner_failure_token)


def get_current_wrapper() -> Optional[Any]:
    """Get the current execution's SecureToolWrapper, or None if not in execution context."""
    return current_wrapper.get()


def get_current_execution_id() -> Optional[str]:
    """Get the current execution ID, or None if not in execution context."""
    return current_execution_id.get()


def get_current_request_id() -> Optional[str]:
    """Get the current request ID, or None if not in execution context."""
    return current_request_id.get()


def get_current_trace_id() -> Optional[str]:
    """Get the current platform trace ID for log correlation, or None."""
    return current_trace_id.get()


def get_current_oe_url() -> Optional[str]:
    """Get the current OE URL, or None if not in execution context."""
    return current_oe_url.get()


def get_current_oe_owner_url() -> Optional[str]:
    """Get the validated replica-specific OE owner URL.

    None if absent or if a previous owner pre-attempt in this execution
    already failed (see :func:`report_oe_owner_url_failure`).
    """
    failure = current_owner_url_failure.get()
    if failure is not None and failure.failed:
        return None
    return current_oe_owner_url.get()


def report_oe_owner_url_failure() -> None:
    """Mark the current execution's owner URL unusable.

    One-way: after this, :func:`get_current_oe_owner_url` returns None for the
    rest of the execution so repeated emits stop re-paying the pre-attempt
    timeout against a dead owner. No-op outside an execution context.
    """
    failure = current_owner_url_failure.get()
    if failure is not None:
        failure.failed = True


def get_current_user_id() -> Optional[str]:
    """Get the current user ID, or None if not in execution context."""
    return current_user_id.get()


def get_current_session_id() -> Optional[str]:
    """Get the current session/thread ID, or None if not in execution context."""
    return current_session_id.get()


def get_current_workspace_id() -> Optional[str]:
    """Get the current workspace ID, or None if not in execution context."""
    return current_workspace_id.get()


def get_current_authorization() -> Optional["ToolAuthorization"]:
    """Get delegated authorization for the current execution, if available."""
    return current_authorization.get()


def get_current_custom_headers() -> Dict[str, str]:
    """Get caller-provided custom headers, or empty dict if not set.

    Platform-internal headers (``a2a-`` prefix) are stripped - agent code
    should never see A2A tokens or routing metadata.  Internal platform
    code that needs the full set (e.g. the A2A client) should call
    :func:`get_all_custom_headers` instead.
    """
    headers = current_custom_headers.get() or {}
    return {k: v for k, v in headers.items() if not k.startswith("a2a-")}


def get_all_custom_headers() -> Dict[str, str]:
    """Get all custom headers including platform-internal ``a2a-`` entries."""
    return current_custom_headers.get() or {}


def get_current_execution_metadata() -> Dict[str, Any]:
    """Get metadata accumulated for the current execution step."""
    return current_execution_metadata.get() or {}


def get_current_payload() -> Dict[str, Any]:
    """Get the caller-provided invocation payload, or empty dict if not set.

    This is the opaque request body the caller sent alongside ``message``
    (e.g. screen state, structured context). Agent code can read it from
    anywhere in the execution, including tool functions.
    """
    return current_payload.get() or {}


def request_session_finish() -> SessionFinishStatus:
    """Record that the agent considers this session finished.

    Mutates a holder shared with child contexts and worker threads on purpose:
    LangGraph nodes run in copied contexts, where a ContextVar.set is invisible
    to the AER frame that reads this at the end of the turn.

    `current_wrapper` is None for Tool/Function contexts (see server/tool.py) —
    only the AER holds the finish latch, so those contexts must also report
    UNAVAILABLE rather than a misleading success.
    """
    state = current_session_finish.get()
    if state is None or current_wrapper.get() is None:
        return SessionFinishStatus.UNAVAILABLE
    # requested and closed are adjudicated together under one lock: two
    # threads racing this call (or one racing the AER's close()) must never
    # both see a pre-decision snapshot and independently reach an outcome
    # that disagrees with what the other decided.
    with state.lock:
        if state.closed:
            return SessionFinishStatus.UNAVAILABLE
        if state.requested:
            return SessionFinishStatus.ALREADY_REQUESTED
        state.requested = True
        return SessionFinishStatus.REQUESTED


def is_session_finish_requested() -> bool:
    """Return whether request_session_finish() has been called during this execution."""
    state = current_session_finish.get()
    if state is None:
        return False
    with state.lock:
        return state.requested


def close_session_finish_latch() -> None:
    """Close the current execution's session-finish latch.

    Called from the AER's `finally` once the execute frame ends (covers the
    success, error, policy-denied, and suspend paths alike - a finish
    requested from post-turn work on a failed run is equally unactionable).
    After this, request_session_finish() reports UNAVAILABLE instead of
    promising a release that will never happen, because the caller (e.g. an
    asyncio.create_task scheduled during the turn but running after it)
    already got its answer from a latch that no longer has anyone reading
    it.
    """
    state = current_session_finish.get()
    if state is not None:
        state.close()


def record_suspend_request(payload: Dict[str, Any]) -> None:
    """Record an author-intended HITL suspend for the current tool call.

    Called only by ``SuspendPayload.to_json``, so the signal's provenance is
    the tool author's code, not tool-result data. A no-op outside
    an execution context (the holder is unset).
    """
    state = current_suspend_request.get()
    if state is not None:
        state.payload = payload


def get_requested_suspend() -> Optional[Dict[str, Any]]:
    """Return the suspend payload this tool call requested via
    ``SuspendPayload.to_json``, or None. The Tool Pod reads this after the tool
    returns to decide whether to report ``status="suspend"``."""
    state = current_suspend_request.get()
    return state.payload if state is not None else None


@contextmanager
def local_suspend_request_context() -> Iterator[None]:
    """Give one in-process tool call an isolated suspend marker."""
    token = current_suspend_request.set(_SuspendRequestState())
    try:
        yield
    finally:
        current_suspend_request.reset(token)


def record_current_memory_metadata(
    *,
    action: str,
    memory_type: str,
    content: Optional[str] = None,
    relevance_score: Optional[float] = None,
    query: Optional[str] = None,
) -> None:
    """Attach explicit memory metadata to the current execution step, if one exists."""
    metadata = current_execution_metadata.get()
    if metadata is None:
        return

    memory: Dict[str, Any] = {
        "action": action,
        "type": memory_type,
    }
    if content:
        # Content and query are user/LLM-derived and this metadata is returned
        # to the UI — scrub credential-shaped fragments before persisting.
        memory["content"] = _redact_text(content)
    if relevance_score is not None:
        memory["relevance_score"] = relevance_score
    if query:
        memory["query"] = _redact_text(query)

    memory_events = metadata.setdefault("memory_events", [])
    if isinstance(memory_events, list):
        memory_events.append(memory)
    metadata["memory"] = memory


# =========================================================================
# Customer-origin logging (causal attribution, not security provenance)
# =========================================================================

_CUSTOMER_ORIGIN = "customer"
_current_customer_origin: ContextVar[bool] = ContextVar("current_customer_origin", default=False)


def get_current_log_origin() -> Optional[str]:
    """Return ``\"customer\"`` inside a customer-code boundary, else None."""
    return _CUSTOMER_ORIGIN if _current_customer_origin.get() else None


@contextmanager
def customer_origin_scope() -> Iterator[None]:
    """Mark the dynamic extent of customer agent/tool code for log attribution.

    Framework-internal. Nested scopes are a no-op. Missing origin means
    unclassified, not a proven platform-authored record.
    """
    if _current_customer_origin.get():
        yield
        return
    token = _current_customer_origin.set(True)
    try:
        yield
    finally:
        _current_customer_origin.reset(token)
