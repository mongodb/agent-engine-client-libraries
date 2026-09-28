"""
SecureToolWrapper - Routes all tool/LLM calls through OE for logging and policy enforcement.

This module provides:
- SecureToolWrapper: Wraps tool calls to route through OE
- SecureWrappedLLM: Wraps LLM calls to route through OE

HITL is handled via LangGraph's native interrupt() / Command(resume=...) API.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, Iterator, Literal, Optional

import httpx
from fastapi import HTTPException

from agent_engine_runner_shared.context import (
    get_current_execution_id,
    get_current_oe_owner_url,
    get_current_user_id,
    get_requested_suspend,
    local_suspend_request_context,
    report_oe_owner_url_failure,
)
from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import ACTIVITY_KIND_TOOL
from agent_engine_runner_shared.models import (
    GuardrailMeta,
    SuspendPayload,
    ToolExecuteRequest,
    ToolExecuteResponse,
    ToolResultRequest,
    coerce_token_usage,
)
from agent_engine_runner_shared.owner_callback import (
    owner_delivered,
    post_without_reading_response_body,
)
from agent_engine_runner_shared.server.http_retry import (
    RETRY_AFTER_MAX_WAIT_S,
    parse_retry_after_seconds,
)
from agent_engine_runner_shared.tls_client import create_httpx_client_with_tls
from agent_engine_runner_shared.tool_api_error import (
    ToolAPIError,
    classify_tool_api_error,
    request_credential_values,
)
from agent_engine_runner_shared.tool_memory_ownership import tool_memory_read_ownership
from agent_engine_runner_shared.utils import (
    OE_RETRYABLE_MAX_ATTEMPTS,
    TOOL_READ_TIMEOUT,
    get_request_timeout,
    log_cached_result,
    log_policy_blocked,
    log_tool_request,
    log_tool_result,
)
from agent_engine_runner_shared.workflow import (
    DurableActivityDeniedError,
    DurableActivityInterrupted,
    DurableActivitySuspended,
    ReplayedActivityFailedError,
    WorkflowClient,
    WorkflowClientError,
    current_attempt_context,
    run_serial_activity,
)
from agent_engine_runner_shared.workflow.activity import semantic_input_from_json
from agent_engine_runner_shared.workflow.context import (
    activity_requires_reconstruction,
    allocate_activity_ordinal,
    tool_activity_key,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from agent_engine_runner_shared.server.drain import DrainRegistry
    from agent_engine_runner_shared.workflow.memory import DurableMemoryState

logger = logging.getLogger(__name__)

# Marker key for an OE-triggered call interruption, kept out of the plain
# JSON a real tool result could return. Namespaced to avoid collision with a
# tool's own artifact keys.
#
# Frozen wire contract: persisted into checkpoints and replay logs, and
# duplicated in agent-engine-runner-shared/src/secure_wrapper.ts. Changing either value
# breaks interrupt detection on already-checkpointed sessions and/or
# cross-language parity — keep the two in lockstep.
CALL_INTERRUPTED_ARTIFACT_KEY = "__agent_engine_oe_call_interrupted__"
INTERRUPTED_CALL_CONTENT = "This call was stopped before completing."


class _CallInterrupted:
    """Sentinel distinguishing an OE-triggered interrupt from any real tool
    return value. Identity-checked (``is``), never shape-checked — a tool
    cannot construct or return this by accident."""

    __slots__ = ()


_CALL_INTERRUPTED = _CallInterrupted()

# Durable-route wire form of the sentinel: activity outcomes must stay
# JSON-serializable for replay, so an interrupt is recorded under the reserved
# key. The reserved key is stripped from genuine tool results before they are
# recorded, so this exact shape can only ever mean an OE interrupt.
_INTERRUPTED_WIRE_MARKER = {CALL_INTERRUPTED_ARTIFACT_KEY: True}


def _current_trace_context() -> tuple[Optional[str], Optional[str]]:
    """Best-effort (trace_id, span_id) of the active OTel span.

    Lazily imported: this module is imported unconditionally by core tool
    routing, while OTel itself lives behind the optional ``tracing`` extra.
    A top-level import here would make tool execution depend on an extra
    that has nothing to do with it.
    """
    try:
        from agent_engine_runner_shared.tracing.setup import get_current_trace_context
    except ImportError:
        return None, None
    return get_current_trace_context()


@contextmanager
def _optional_span(name: str) -> Iterator["Span | None"]:
    """Start ``name`` as the active span, or yield ``None`` when the
    optional ``tracing`` extra isn't installed.

    Lazily imported for the same reason as ``_current_trace_context``: this
    module is imported unconditionally by core tool routing, while OTel lives
    behind the optional ``tracing`` extra. A real span auto-records an
    exception and sets ERROR status if one propagates out of the ``with``
    block — callers don't need their own try/except for that.
    """
    try:
        from opentelemetry import trace
    except ImportError:
        yield None
        return
    tracer = trace.get_tracer("runner-sdk")
    with tracer.start_as_current_span(name) as span:
        yield span


def extract_usage(response: Any, fallback_model: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract token usage from an LLM response's response_metadata.

    Handles provider-variant key names:
    - OpenAI: response_metadata.usage.{prompt_tokens, completion_tokens, total_tokens}
    - Anthropic: response_metadata.usage.{input_tokens, output_tokens}
    - Some providers: response_metadata.token_usage.{...}
    - Anthropic Usage objects and non-dict Mappings

    Returns dict with keys: prompt_tokens, completion_tokens, total_tokens, model.
    All values may be None if unavailable. Malformed blobs do not raise.
    """
    result: Dict[str, Any] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "model": fallback_model,
    }

    rm_raw = getattr(response, "response_metadata", None)
    rm = rm_raw if isinstance(rm_raw, Mapping) else {}

    # Coerce each key independently so an empty or malformed `usage`
    # still falls through to `token_usage`.
    usage_tu = coerce_token_usage(rm.get("usage"))
    if usage_tu is None:
        usage_tu = coerce_token_usage(rm.get("token_usage"))
    if usage_tu is None:
        usage_tu = coerce_token_usage(getattr(response, "usage_metadata", None))
    if usage_tu is None:
        usage_tu = coerce_token_usage(getattr(response, "usage", None))

    if usage_tu is not None:
        prompt = (
            usage_tu.prompt_tokens if usage_tu.prompt_tokens is not None else usage_tu.input_tokens
        )
        completion = (
            usage_tu.completion_tokens
            if usage_tu.completion_tokens is not None
            else usage_tu.output_tokens
        )
        total = usage_tu.total_tokens
        if prompt is not None:
            result["prompt_tokens"] = int(prompt)
        if completion is not None:
            result["completion_tokens"] = int(completion)
        if total is not None:
            result["total_tokens"] = int(total)
        elif result["prompt_tokens"] is not None and result["completion_tokens"] is not None:
            result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
        if usage_tu.model:
            result["model"] = usage_tu.model

    # Model name: prefer response_metadata, fall back to wrapper attribute
    model_name = rm.get("model_name") or rm.get("model") or result["model"] or fallback_model
    result["model"] = model_name

    return result


def _extract_pod_usage(
    usage: Optional[Dict[str, Any]], fallback_model: Optional[str] = None
) -> Dict[str, Any]:
    """Extract token usage from a tool-pod usage dict into report_oe_result kwargs.

    Uses ``is not None`` checks instead of ``or`` so that a valid 0 token count
    is not treated as missing.  Computes total_tokens when the provider omits it.
    """
    if not usage:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "model": fallback_model,
        }

    prompt = usage.get("input_tokens")
    if prompt is None:
        prompt = usage.get("prompt_tokens")

    completion = usage.get("output_tokens")
    if completion is None:
        completion = usage.get("completion_tokens")

    total = usage.get("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion

    model = usage.get("model")
    if not model:
        model = fallback_model

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "model": model,
    }


# =============================================================================
# Exceptions
# =============================================================================


class PolicyDeniedException(Exception):
    """Raised when OE policy engine denies execution."""

    def __init__(
        self,
        reason: str,
        guardrail_meta: Optional[GuardrailMeta] = None,
    ):
        self.reason = reason
        self.guardrail_meta = guardrail_meta
        super().__init__(f"Policy denied: {reason}")


class OERetryAfterError(Exception):
    """OE asked the runner to retry /tool/execute after a Retry-After delay."""

    def __init__(self, wait_s: float):
        self.wait_s = wait_s
        super().__init__(f"OE retry after {wait_s:g}s")


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""

    def __init__(self, error: str):
        self.error = error
        super().__init__(f"Tool execution error: {error}")


_TERMINAL_EXECUTION_REASONS = {
    "execution already error": (
        "This execution already ended in error; later tool calls are rejected."
    ),
    "execution already completed": (
        "This execution already completed; later tool calls are rejected."
    ),
    "execution already cancelled": ("This execution was cancelled; later tool calls are rejected."),
}


class TerminalExecutionError(ToolExecutionError):
    """Raised when OE rejects a tool call because the execution already ended."""


def raise_for_oe_rejection(
    reason: str | None,
    *,
    guardrail_meta: Optional[GuardrailMeta] = None,
) -> None:
    """Map a proceed=false OE response to the matching exception. Never returns."""
    text = reason or "Policy denied"
    mapped = _TERMINAL_EXECUTION_REASONS.get(text)
    if mapped is not None:
        raise TerminalExecutionError(mapped)
    raise PolicyDeniedException(text, guardrail_meta=guardrail_meta)


class ToolCallTimeoutError(ToolExecutionError):
    """Raised when a tool call outlives its deadline.

    Distinct from PolicyDeniedException on purpose. A timeout means the call was
    permitted and ran — it just ran too long — so reporting it as a denial sends
    the developer to debug governance instead of their tool.
    """

    def __init__(self, tool_name: str, timeout_seconds: float, elapsed_seconds: float):
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
        message = (
            f"Tool '{tool_name}' timed out after {elapsed_seconds:.1f}s "
            f"(deadline {timeout_seconds:.0f}s). It can be given longer via "
            f"RUNNER_TOOL_READ_TIMEOUT, or interrupted while running."
        )
        self.error = message
        Exception.__init__(self, message)


class OperationalStepAllocator:
    """Process-local operational step_number mint for one AER execution.

    Shared by SecureToolWrapper and SecureLLMProxy so parallel tools and LLM
    calls cannot collide under a single AER. Not a multi-OE authority.
    """

    __slots__ = ("_lock", "_n")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._n = 0

    def next(self) -> int:
        with self._lock:
            self._n += 1
            return self._n

    def observe_at_least(self, n: int) -> None:
        if n <= 0:
            return
        with self._lock:
            if n > self._n:
                self._n = n

    def current(self) -> int:
        with self._lock:
            return self._n


class _DurableToolActivityFailedError(Exception):
    """Internal, replayable form of a registered tool failure."""


_DURABLE_TOOL_FAILURE_PREFIX = "__agentic_durable_tool_failure_v1__:"
_DURABLE_TOOL_FAILURE_FALLBACK = "durable activity failed"


def _tool_error_message(error: Exception) -> str:
    message = error.error if isinstance(error, ToolExecutionError) else str(error)
    return message or _DURABLE_TOOL_FAILURE_FALLBACK


def _encode_durable_tool_failure(error: Exception) -> str:
    payload: dict[str, Any] = {
        "kind": "execution",
        "message": _tool_error_message(error),
    }
    if isinstance(error, ToolCallTimeoutError):
        payload.update(
            {
                "kind": "timeout",
                "tool_name": error.tool_name,
                "timeout_seconds": error.timeout_seconds,
                "elapsed_seconds": error.elapsed_seconds,
            }
        )
    elif isinstance(error, ExternalAPICallError):
        payload.update(
            {
                "kind": "external_api",
                "tool_api_error": error.tool_api_error.model_dump(mode="json"),
            }
        )
    return _DURABLE_TOOL_FAILURE_PREFIX + json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_durable_tool_failure(message: str) -> ToolExecutionError:
    if not message.startswith(_DURABLE_TOOL_FAILURE_PREFIX):
        return ToolExecutionError(message or _DURABLE_TOOL_FAILURE_FALLBACK)
    try:
        payload = json.loads(message.removeprefix(_DURABLE_TOOL_FAILURE_PREFIX))
    except (TypeError, ValueError):
        return ToolExecutionError(message)
    if not isinstance(payload, dict):
        return ToolExecutionError(message)
    failure_message = payload.get("message")
    if not isinstance(failure_message, str) or not failure_message:
        failure_message = _DURABLE_TOOL_FAILURE_FALLBACK
    kind = payload.get("kind")
    if kind == "external_api":
        try:
            tool_api_error = ToolAPIError.model_validate(payload.get("tool_api_error"))
        except (TypeError, ValueError):
            return ToolExecutionError(failure_message)
        return ExternalAPICallError(failure_message, tool_api_error)
    if kind != "timeout":
        return ToolExecutionError(failure_message)
    tool_name = payload.get("tool_name")
    timeout_seconds = payload.get("timeout_seconds")
    elapsed_seconds = payload.get("elapsed_seconds")
    if (
        not isinstance(tool_name, str)
        or not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or isinstance(elapsed_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or not math.isfinite(elapsed_seconds)
    ):
        return ToolExecutionError(failure_message)
    return ToolCallTimeoutError(tool_name, float(timeout_seconds), float(elapsed_seconds))


class ExternalAPICallError(ToolExecutionError):
    """Raised when a tool's external API call fails with a classified error."""

    def __init__(self, message: str, tool_api_error: ToolAPIError) -> None:
        super().__init__(message)
        self.tool_api_error = tool_api_error


class LLMInvocationError(Exception):
    """Raised when an LLM invocation fails after OE approval.

    ``source`` is optional invoke-owner attribution. Only ``"llm"`` means the
    provider failed; omit it for relay, truncation, or guardrail plumbing that
    uses this same exception type.

    ``error_code`` is the machine-readable classification the tool pod stamped
    on the failure (e.g. a provider credential rejection), when it did. It
    travels to the OE/UI on the ERROR chunk metadata instead of the generic
    invocation code so consumers can classify without string-matching prose.
    """

    def __init__(
        self,
        error: str,
        *,
        source: str | None = None,
        error_code: str | None = None,
    ) -> None:
        self.error = error
        self.source = source
        self.error_code = error_code
        super().__init__(f"LLM invocation error: {error}")


# =============================================================================
# OE Communication Helpers
# =============================================================================


def request_oe_approval(
    oe_url: str,
    execution_id: str,
    tool_name: str,
    arguments: Dict[str, Any],
    step: int,
    *,
    kind: Optional[str] = None,
    is_local: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    redact_fields: Optional[list[str]] = None,
    provider_type: Optional[str] = None,
    scopes: Optional[list[str]] = None,
    tool_call_id: Optional[str] = None,
    custom_headers: Optional[Dict[str, str]] = None,
    timeout: float | httpx.Timeout | None = None,
) -> ToolExecuteResponse:
    """
    Request approval from OE before executing a tool/LLM call.

    Returns:
        ToolExecuteResponse with proceed flag and optional cached_result

    Raises:
        PolicyDeniedException: If OE is unreachable
    """
    try:
        return _request_oe_approval(
            oe_url,
            execution_id,
            tool_name,
            arguments,
            step,
            kind=kind,
            is_local=is_local,
            metadata=metadata,
            redact_fields=redact_fields,
            provider_type=provider_type,
            scopes=scopes,
            tool_call_id=tool_call_id,
            custom_headers=custom_headers,
            timeout=timeout,
        )
    except OERetryAfterError as e:
        raise PolicyDeniedException("OE unreachable; blocking for safety") from e


# Captured at import so retryable can unwrap this wrapper even after tests
# patch the module-global name `request_oe_approval`.
_PUBLIC_REQUEST_OE_APPROVAL = request_oe_approval


def _request_oe_approval(
    oe_url: str,
    execution_id: str,
    tool_name: str,
    arguments: Dict[str, Any],
    step: int,
    *,
    kind: Optional[str] = None,
    is_local: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    redact_fields: Optional[list[str]] = None,
    provider_type: Optional[str] = None,
    scopes: Optional[list[str]] = None,
    tool_call_id: Optional[str] = None,
    custom_headers: Optional[Dict[str, str]] = None,
    timeout: float | httpx.Timeout | None = None,
) -> ToolExecuteResponse:
    """
    Request approval from OE before executing a tool/LLM call.

    Returns:
        ToolExecuteResponse with proceed flag and optional cached_result

    Raises:
        PolicyDeniedException: If OE is unreachable
    """
    trace_id, span_id = _current_trace_context()
    request = ToolExecuteRequest(
        execution_id=execution_id,
        tool_name=tool_name,
        arguments=arguments,
        step_number=step,
        tool_call_id=tool_call_id,
        kind=kind,
        is_local=is_local,
        redact_fields=redact_fields or [],
        provider_type=provider_type,
        scopes=scopes or [],
        metadata=metadata or {},
        custom_headers=custom_headers,
        trace_id=trace_id,
        span_id=span_id,
    )

    # A short connect budget is right — an unreachable OE should fail fast — but the
    # read has to outlast the tool itself, since the OE keeps the response open until
    # the tool returns. Callers with a different profile (the LLM proxy) pass their own.
    client_timeout = (
        timeout
        if timeout is not None
        else httpx.Timeout(get_request_timeout(), read=TOOL_READ_TIMEOUT)
    )
    started = time.monotonic()
    with create_httpx_client_with_tls(oe_url, client_timeout) as client:
        try:
            resp = client.post(f"{oe_url}/tool/execute", json=request.model_dump())
            resp.raise_for_status()
            return ToolExecuteResponse(**resp.json())
        except httpx.ReadTimeout as e:
            # Only a *read* timeout means the tool overran: the request reached the
            # OE and we gave up waiting for its result. A connect/write/pool
            # timeout never got the call to the OE, so it stays an unreachable-OE
            # failure below rather than being blamed on the tool.
            elapsed = time.monotonic() - started
            logger.error(f"Step {step}: Tool '{tool_name}' timed out after {elapsed:.1f}s")
            read_deadline = (
                client_timeout.read
                if isinstance(client_timeout, httpx.Timeout) and client_timeout.read is not None
                else elapsed
            )
            raise ToolCallTimeoutError(tool_name, read_deadline, elapsed) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                retry_after_s = parse_retry_after_seconds(e.response.headers.get("Retry-After"))
                if retry_after_s is not None:
                    raise OERetryAfterError(retry_after_s) from e
            logger.error(f"Step {step}: Failed to request OE approval: {e}", exc_info=True)
            raise PolicyDeniedException("OE unreachable; blocking for safety") from e
        except httpx.HTTPError as e:
            logger.error(f"Step {step}: Failed to request OE approval: {e}", exc_info=True)
            raise PolicyDeniedException("OE unreachable; blocking for safety") from e


def request_oe_approval_retryable(
    request: Callable[..., ToolExecuteResponse],
    **kwargs: Any,
) -> ToolExecuteResponse:
    """Repeat the same /tool/execute while OE says the failure is retryable.

    Reservation-loss returns status=error with retryable=true after releasing
    the step stamp. Retrying the same step here keeps durable activity from
    recording FAILED on a call that can still succeed.

    Wrapped in a span: without one, a retried call is invisible —
    the surrounding tool-node span just looks slower, with no record of how
    many attempts happened. Started before ``request`` (typically
    ``request_oe_approval``) runs, so the trace_id/span_id it reads via
    ``_current_trace_context()`` reflect this span.
    """
    if request is _PUBLIC_REQUEST_OE_APPROVAL:
        request = _request_oe_approval
    with _optional_span("secure_wrapper.request_oe_approval") as span:
        attempt_count = 0
        try:
            response: Optional[ToolExecuteResponse] = None
            for attempt in range(OE_RETRYABLE_MAX_ATTEMPTS):
                try:
                    response = request(**kwargs)
                except OERetryAfterError as exc:
                    attempt_count = attempt + 1
                    if attempt >= OE_RETRYABLE_MAX_ATTEMPTS - 1:
                        raise PolicyDeniedException("OE unreachable; blocking for safety") from exc
                    time.sleep(min(exc.wait_s, RETRY_AFTER_MAX_WAIT_S))
                    continue
                attempt_count = attempt + 1
                if response.status != "error" or not response.retryable:
                    break
            if response is None:
                raise PolicyDeniedException("OE unreachable; blocking for safety")
            return response
        finally:
            if span is not None:
                span.set_attribute("attempt_count", attempt_count)


def report_oe_result(
    oe_url: str,
    execution_id: str,
    tool_name: str,
    step: int,
    status: str,
    result: Any,
    error: Optional[str],
    duration_ms: float,
    pod_name: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    model: Optional[str] = None,
    kind: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tool_call_id: Optional[str] = None,
    owner_url: Optional[str] = None,
    on_owner_failure: Optional[Callable[[], None]] = None,
    tool_api_error: Optional[ToolAPIError] = None,
) -> None:
    """Report an execution result and require OE to acknowledge settlement.

    ``owner_url`` — when given, the replica-specific OE owner base URL (already
    validated against ``oe_url``). A single owner pre-attempt runs before the
    service loop and does not consume the service retry budget; the owner is
    best-effort, so any failure — a transport error OR any non-2xx response —
    marks the replica unusable and falls through to the trusted ``oe_url`` loop.
    The owner is never retried and an owner response never raises.

    Wrapped in a span: without one, a slow-to-ack OE or a retried
    settlement is invisible — the surrounding tool-node span just looks
    slower, with no record of how many attempts happened. Started before
    ``_current_trace_context()`` is read below, so the trace_id/span_id put
    on the wire reflect this span.
    """
    with _optional_span("secure_wrapper.report_oe_result") as span:
        trace_id, span_id = _current_trace_context()
        request = ToolResultRequest(
            execution_id=execution_id,
            step_number=step,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            status=status,
            result=result,
            error=error,
            duration_ms=duration_ms,
            pod_name=pod_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=model,
            kind=kind,
            metadata=metadata or {},
            trace_id=trace_id,
            span_id=span_id,
            tool_api_error=tool_api_error,
        )

        payload = request.model_dump()
        service_url = f"{oe_url}/tool/result"
        owner_result_url = f"{owner_url.rstrip('/')}/tool/result" if owner_url else None
        last_error: Optional[Exception] = None
        attempt_count = 0
        try:
            if owner_result_url is not None:
                attempt_count += 1
                owner_delivered_successfully = False
                try:
                    with create_httpx_client_with_tls(oe_url, get_request_timeout()) as client:
                        owner_delivered_successfully = owner_delivered(
                            client,
                            owner_result_url,
                            payload,
                            lambda msg: logger.warning("/tool/result %s URL %s", msg, service_url),
                        )
                except Exception as exc:  # noqa: BLE001 - owner is best-effort
                    if owner_delivered_successfully:
                        return
                    failure = str(exc) or type(exc).__name__
                    logger.warning(
                        "/tool/result owner URL %s unusable (%s); falling back to service URL %s",
                        owner_result_url,
                        failure,
                        service_url,
                    )
                if owner_delivered_successfully:
                    return
                if on_owner_failure is not None:
                    on_owner_failure()

            for attempt in range(3):
                attempt_count += 1
                service_delivered_successfully = False
                retry_after_s: Optional[float] = None
                try:
                    with create_httpx_client_with_tls(oe_url, get_request_timeout()) as client:
                        response = post_without_reading_response_body(client, service_url, payload)
                        response.raise_for_status()
                        service_delivered_successfully = True
                except httpx.HTTPStatusError as exc:
                    if 400 <= exc.response.status_code < 500:
                        raise ToolExecutionError(
                            f"OE rejected result settlement with HTTP {exc.response.status_code}"
                        ) from exc
                    last_error = exc
                    if exc.response.status_code == 503:
                        retry_after_s = parse_retry_after_seconds(
                            exc.response.headers.get("Retry-After")
                        )
                except httpx.TransportError as exc:
                    last_error = exc
                except httpx.HTTPError as exc:
                    last_error = exc
                except Exception as exc:
                    if service_delivered_successfully:
                        return
                    raise ToolExecutionError("Failed to prepare OE result settlement") from exc

                if service_delivered_successfully:
                    return
                if attempt < 2:
                    if retry_after_s is not None:
                        time.sleep(min(retry_after_s, RETRY_AFTER_MAX_WAIT_S))
                    else:
                        time.sleep(0.1 * (2**attempt))

            raise ToolExecutionError(
                "OE did not acknowledge tool result settlement"
            ) from last_error
        finally:
            if span is not None:
                span.set_attribute("attempt_count", attempt_count)


# Set by SecureToolWrapper._execute_local_tool around a callback-routed tool
# body: how the body registers its abort channel with the drain registry once
# its cancellable unit exists, so POST /interrupt/call can stop that one call.
# The whole local-execution chain runs on one framework worker thread, which
# keeps the handoff off every adapter's signature.
_pending_call_abort_channel: ContextVar[Optional[Callable[[Callable[[], None]], bool]]] = (
    ContextVar("pending_call_abort_channel", default=None)
)


def _run_coroutine_with_abort_channel(
    coro: "Coroutine[Any, Any, Any]",
    register_cancel: Callable[[Callable[[], None]], bool],
) -> Any:
    """Run a tool coroutine on this worker thread's private loop while the
    drain registry holds a thread-safe cancel channel for it. Same cleanup as
    asyncio.run; the difference is the registered task, which a per-call abort
    cancels so the call stops while the rest of the execution continues."""
    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(coro)

        def cancel() -> None:
            try:
                # The message surfaces in the trace's cancellation row, so it
                # reads like the TS twin's AbortError instead of a bare class
                # name.
                loop.call_soon_threadsafe(task.cancel, INTERRUPTED_CALL_CONTENT)
            except RuntimeError:
                # The loop already closed: the body ended and a late abort has
                # nothing left to stop.
                pass

        if not register_cancel(cancel):
            # The call's registration ended before the body started: run
            # untracked rather than drop tenant work.
            logger.warning("call abort channel attach failed; running the tool untracked")
        return loop.run_until_complete(task)
    finally:
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()


def _resolve_registered_tool_result(
    value: Any,
    register_cancel: Optional[Callable[[Callable[[], None]], bool]] = None,
) -> Any:
    if not inspect.isawaitable(value):
        return value
    if not inspect.iscoroutine(value):
        raise RuntimeError("Async local tools must return a coroutine")
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        if register_cancel is not None:
            return _run_coroutine_with_abort_channel(value, register_cancel)
        return asyncio.run(value)
    value.close()
    raise RuntimeError(
        "Async local tools require a framework worker thread; refusing to block a running event loop"
    )


def _invoke_registered_tool(original_tool: Any, kwargs: Dict[str, Any]) -> Any:
    register_cancel = _pending_call_abort_channel.get()
    async_invoke = getattr(original_tool, "ainvoke", None)
    if hasattr(original_tool, "invoke"):
        if getattr(original_tool, "func", None) is None and callable(async_invoke):
            return _resolve_registered_tool_result(async_invoke(kwargs), register_cancel)
        return _resolve_registered_tool_result(original_tool.invoke(kwargs), register_cancel)
    return _resolve_registered_tool_result(original_tool(**kwargs), register_cancel)


# =============================================================================
# SecureToolWrapper
# =============================================================================


class SecureToolWrapper:
    """
    Wrap tool calls so OE owns execution, replay, logging, and routing.

    Flow for each call:
    1. Send the intercepted tool call to OE (POST /tool/execute)
    2. OE returns a final outcome or routes the call back in process
    3. The wrapper logs the outcome and converts suspend payloads back into framework interrupts

    This enables:
    - Audit logging of all operations
    - Policy enforcement (block if not allowed)
    - Deterministic replay scenarios (resume after SUSPEND)
    """

    def __init__(
        self,
        oe_url: str,
        execution_id: str,
        custom_headers: Optional[Dict[str, str]] = None,
        durable_memory: Optional[DurableMemoryState] = None,
        oe_owner_url: Optional[str] = None,
        call_registry: Optional["DrainRegistry"] = None,
    ):
        """
        Initialize the wrapper.

        Args:
            oe_url: URL of the Orchestration Engine
            execution_id: Unique execution identifier
            custom_headers: Caller-provided custom headers forwarded from the invoke request
            durable_memory: Durable activity Memory state, when enabled
            oe_owner_url: Validated replica-specific OE owner URL for tool-result
                callback fallback, or None when the request supplied no usable one
            call_registry: Per-execution drain registry, enabling the per-call
                abort (POST /interrupt/call) for tools this wrapper runs locally;
                None leaves local calls unaddressable, as before
        """
        self.oe_url = oe_url.rstrip("/")
        self.oe_owner_url = oe_owner_url
        self._oe_owner_url_failed = False
        self.execution_id = execution_id
        self.durable_memory = durable_memory
        self._call_registry = call_registry
        # Shared with SecureLLMProxy for this execution (single-AER assumption).
        self.operational_steps = OperationalStepAllocator()
        self.custom_headers: Dict[str, str] = custom_headers or {}
        self._workflow: Any | None = None

    def _current_oe_owner_url(self) -> Optional[str]:
        if self._oe_owner_url_failed:
            return None
        if get_current_execution_id() == self.execution_id:
            return get_current_oe_owner_url()
        return self.oe_owner_url

    def _report_oe_owner_failure(self) -> None:
        self._oe_owner_url_failed = True
        if get_current_execution_id() == self.execution_id:
            report_oe_owner_url_failure()

    @property
    def step_counter(self) -> int:
        """Current operational-step watermark for compatibility readers."""
        return self.operational_steps.current()

    def next_operational_step(self) -> int:
        """Allocate the next operational step_number for this execution."""
        return self.operational_steps.next()

    def observe_operational_step(self, n: int) -> None:
        """Raise the allocator watermark (e.g. from OE latest_step_number)."""
        self.operational_steps.observe_at_least(n)

    @property
    def workflow(self) -> Any:
        """Lazily created workflow client for durable serial tool activities."""
        if self._workflow is None:
            self._workflow = WorkflowClient(self.oe_url, timeout=get_request_timeout())
        return self._workflow

    async def close(self) -> None:
        """Clean up resources owned by this execution's wrapper."""
        if self._workflow is not None:
            self._workflow.close()
            self._workflow = None

    def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        provider_type: Optional[str] = None,
        scopes: Optional[list[str]] = None,
        tool_call_id: Optional[str] = None,
        raw_on_interrupt: bool = False,
        redact_fields: Optional[list[str]] = None,
        is_local: bool = True,
        local_executor: Optional[Callable[[], Any]] = None,
        is_framework_control_flow: Optional[Callable[[BaseException], bool]] = None,
        call_channel_expected: bool = False,
    ) -> Any:
        """
        Execute a tool call through OE.

        Args:
            tool_name: Name of the tool
            arguments: Arguments to pass to the tool
            metadata: Optional execution metadata to send to OE
            tool_call_id: Stable LLM tool-call id used to join this call to its
                result and session message across the execution-log surface
            raw_on_interrupt: When True, return the private ``_CALL_INTERRUPTED``
                sentinel on interrupt instead of the plain ``{"interrupted": True}``
                dict. Only ``create_secure_tool_function`` should pass this; it
                coerces the sentinel immediately. On the durable workflow route
                the sentinel is recorded as a reserved wire marker (replay needs
                JSON) and restored on the caller's side of the replay boundary.
            is_local: Whether OE should return the approved call to this AER.
            local_executor: Callback that invokes the registered tool on the
                current framework stack. Required for local tools.

        Returns:
            Result from execution or cached result

        Raises:
            PolicyDeniedException: If OE blocks the call
            ToolExecutionError: If execution fails
            langgraph.types.interrupt: If result contains __suspend__ signal,
                or if OE returns an elicitation requiring user authorization
                (triggers suspend with authorization_url in context)
        """
        step = self.next_operational_step()
        # The tool's redact_fields policy must reach the debug argument dump —
        # it ships to the centralized log sink even at debug level.
        log_tool_request(tool_name, arguments, step, prefix="TOOL", fields_to_redact=redact_fields)

        if current_attempt_context() is None:
            return self._execute_tool_native(
                tool_name=tool_name,
                arguments=arguments,
                step=step,
                metadata=metadata,
                is_local=is_local,
                local_executor=local_executor,
                redact_fields=redact_fields,
                provider_type=provider_type,
                scopes=scopes,
                tool_call_id=tool_call_id,
                raw_on_interrupt=raw_on_interrupt,
                is_framework_control_flow=is_framework_control_flow,
                call_channel_expected=call_channel_expected,
            )

        # Platform-owned workflow route: wrap the call in ActivityCommand
        # identity so OE can replay a recorded outcome (event-log replay)
        # instead of re-running the tool. OE owns hashing and replay; the
        # native path above is unchanged.
        # Durable identity describes the application's tool invocation, not how
        # the platform routes or authorizes it. Runtime metadata, ToolCall ids,
        # providers, scopes, and local placement may change across a
        # replacement without changing the business operation being replayed.
        # Ordinals come from the request-scoped allocator. Prefer ToolCall id
        # so parallel siblings keep stable identity. Without one, allocate the
        # next free ordinal and run exclusive so overlap cannot redefine
        # identity by worker-arrival order.
        semantic_input: Dict[str, Any] = {"arguments": arguments}

        keyed = bool(tool_call_id)
        activity_ordinal = allocate_activity_ordinal(
            tool_activity_key(tool_call_id) if keyed else None
        )

        def _execute(activity_context: Any) -> Any:
            try:
                if (
                    is_local
                    and local_executor is not None
                    and activity_requires_reconstruction(activity_context.activity_id)
                ):
                    # ResolveActivities already authorized this exact activity
                    # position. Re-enter its captured callback directly so the
                    # framework can recreate interrupt(); the tool-execute cache
                    # still contains the earlier interrupted invocation.
                    result = local_executor()
                else:
                    result = self._execute_tool_native(
                        tool_name=tool_name,
                        arguments=arguments,
                        step=step,
                        metadata=metadata,
                        is_local=is_local,
                        local_executor=local_executor,
                        redact_fields=redact_fields,
                        provider_type=provider_type,
                        scopes=scopes,
                        tool_call_id=tool_call_id,
                        # The sentinel can't be replay-serialized; it is translated
                        # to the reserved wire marker below.
                        raw_on_interrupt=True,
                        is_framework_control_flow=is_framework_control_flow,
                        call_channel_expected=call_channel_expected,
                    )
            except PolicyDeniedException as denial:
                # Record the denial as a DENIED outcome (not FAILED) so a
                # replay reproduces policy-denial semantics.
                raise DurableActivityDeniedError(denial.reason) from denial
            except DurableActivitySuspended:
                raise
            except BaseException as error:
                if is_framework_control_flow is not None and is_framework_control_flow(error):
                    raise DurableActivityInterrupted(error) from error
                if not isinstance(error, Exception):
                    raise
                raise _DurableToolActivityFailedError(
                    _encode_durable_tool_failure(error)
                ) from error
            if result is _CALL_INTERRUPTED:
                return dict(_INTERRUPTED_WIRE_MARKER)
            if isinstance(result, dict) and CALL_INTERRUPTED_ARTIFACT_KEY in result:
                # A genuine tool result may not carry the reserved key, or it
                # could replay as a forged interrupt marker.
                result = {k: v for k, v in result.items() if k != CALL_INTERRUPTED_ARTIFACT_KEY}
            try:
                # Validate before leaving the registered-tool boundary so a
                # non-JSON result has the same public failure live and on replay.
                semantic_input_from_json(result)
            except Exception as error:
                raise _DurableToolActivityFailedError(
                    _encode_durable_tool_failure(error)
                ) from error
            return result

        try:
            result = run_serial_activity(
                client=self.workflow,
                kind=ACTIVITY_KIND_TOOL,
                name=tool_name,
                activity_ordinal=activity_ordinal,
                semantic_input=semantic_input,
                execute=_execute,
                on_activity_resolved=(
                    partial(
                        self.durable_memory.synchronize_tool,
                        user_id=get_current_user_id(),
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                    )
                    if self.durable_memory is not None
                    else None
                ),
                exclusive=not keyed,
            )
        except DurableActivityDeniedError as error:
            # Re-raise the original live denial (with guardrail identity) when
            # this attempt produced it; a replayed denial carries message only.
            if isinstance(error.__cause__, PolicyDeniedException):
                raise error.__cause__ from None
            raise PolicyDeniedException(str(error)) from error
        except (ReplayedActivityFailedError, _DurableToolActivityFailedError) as error:
            raise _decode_durable_tool_failure(str(error)) from error
        except WorkflowClientError:
            raise

        # The activity records an interrupt as the reserved wire marker (replay
        # requires JSON); restore the sentinel here, on the caller's side of the
        # replay boundary, so both routes coerce identically. Only the marker is
        # restored — a genuine tool result can't collide with it because
        # _execute strips the reserved key before the outcome is recorded.
        if result == _INTERRUPTED_WIRE_MARKER:
            return _CALL_INTERRUPTED if raw_on_interrupt else {"interrupted": True}
        return result

    def _execute_tool_native(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        step: int,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        is_local: bool = True,
        local_executor: Optional[Callable[[], Any]] = None,
        redact_fields: Optional[list[str]] = None,
        provider_type: Optional[str] = None,
        scopes: Optional[list[str]] = None,
        tool_call_id: Optional[str] = None,
        raw_on_interrupt: bool = False,
        is_framework_control_flow: Optional[Callable[[BaseException], bool]] = None,
        call_channel_expected: bool = False,
    ) -> Any:
        effective_is_local = is_local and not provider_type and not scopes
        effective_local_executor = local_executor if effective_is_local else None
        # 1. Request approval from OE
        response = request_oe_approval_retryable(
            request_oe_approval,
            oe_url=self.oe_url,
            execution_id=self.execution_id,
            tool_name=tool_name,
            arguments=arguments,
            step=step,
            is_local=effective_is_local,
            redact_fields=redact_fields,
            provider_type=provider_type,
            scopes=scopes,
            tool_call_id=tool_call_id,
            metadata=metadata,
            custom_headers=self.custom_headers,
        )

        if response.elicitation:
            from agent_engine_runner_shared.hooks import get_suspend_handler

            interrupt = get_suspend_handler()
            if interrupt is None:
                raise RuntimeError(
                    "No suspend handler registered. "
                    "Ensure the framework SDK calls register_suspend_handler() before run()."
                )

            interrupt(
                {
                    "suspend_reason": "authorization_required",
                    "suspend_context": {
                        "authorization_url": response.elicitation.authorization_url,
                        "elicitation_id": response.elicitation.elicitation_id,
                        "message": response.elicitation.message,
                        "created": response.elicitation.created,
                    },
                }
            )
            return  # interrupt() raises GraphInterrupt; defensive guard

        if not response.proceed:
            reason = response.reason or "Policy denied"
            if reason not in _TERMINAL_EXECUTION_REASONS:
                log_policy_blocked(tool_name, step, reason, prefix="TOOL")
            raise_for_oe_rejection(reason, guardrail_meta=response.guardrail_meta)

        if response.latest_step_number is not None:
            self.observe_operational_step(response.latest_step_number)

        if response.from_cache:
            log_cached_result(tool_name, step, prefix="TOOL")

        if response.route_to == "callback":
            if not effective_is_local:
                raise ToolExecutionError(
                    "OE returned a local callback route for a tool declared as remote"
                )
            if effective_local_executor is None:
                # Registered-tool adapters always install this executor. Merely
                # running in AER is not enough for a low-level caller: the wrapper
                # still needs the original framework tool object to invoke.
                raise ToolExecutionError(
                    "Local callback route is missing its registered tool executor"
                )
            return self._execute_local_tool(
                tool_name=tool_name,
                step=step,
                local_executor=effective_local_executor,
                tool_call_id=tool_call_id,
                metadata=metadata,
                is_framework_control_flow=is_framework_control_flow,
                call_channel_expected=call_channel_expected,
            )
        if response.route_to is not None:
            raise ToolExecutionError(f"Unexpected OE tool route: {response.route_to}")

        status = response.status or "error"
        result = response.result
        error = response.error
        duration_ms = response.duration_ms or 0.0

        log_tool_result(
            tool_name,
            step,
            status,
            result=result,
            error=error,
            duration_ms=duration_ms,
            prefix="TOOL",
        )

        if status == "interrupted":
            return _CALL_INTERRUPTED if raw_on_interrupt else {"interrupted": True}
        if status == "error":
            if response.tool_api_error is not None:
                raise ExternalAPICallError(error or "Unknown error", response.tool_api_error)
            raise ToolExecutionError(error or "Unknown error")
        # Suspend is honored only from the OE-confirmed status channel, whose
        # provenance is the tool author's SuspendPayload.to_json call — never
        # from sniffing result content, which a relayed untrusted payload
        # controls. A successful result is returned verbatim.
        if status == "suspend":
            return self._handle_suspend(result)
        if status != "success":
            raise ToolExecutionError(f"Unexpected OE tool status: {status}")

        return result

    def _execute_local_tool(
        self,
        *,
        tool_name: str,
        step: int,
        local_executor: Callable[[], Any],
        tool_call_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        is_framework_control_flow: Optional[Callable[[BaseException], bool]],
        call_channel_expected: bool = False,
    ) -> Any:
        """Execute an OE-approved local call on the current framework stack."""
        import socket

        # Register the call with the drain registry so POST /interrupt/call can
        # address it. Async bodies attach a thread-safe abort channel once their
        # task exists (see _run_coroutine_with_abort_channel); a sync body never
        # does, so its abort honestly answers not_cancellable and its outcome
        # still flows. This method runs on a framework worker thread, so every
        # registry mutation marshals onto the registry's owner loop.
        registry = self._call_registry
        handle = None
        channel_token = None
        if registry is not None:
            try:
                handle = registry.owner_loop_call(
                    partial(
                        registry.begin_work,
                        self.execution_id,
                        None,
                        step_number=step,
                        channel_expected=call_channel_expected,
                    )
                )
            except HTTPException:
                # A whole-run drain latched between the OE claim and this
                # dispatch: the run is terminal, so the tool must not start.
                raise asyncio.CancelledError()

            def _register_abort_channel(callback: Callable[[], None]) -> bool:
                # Narrowed: this closure is defined only when a registration exists.
                return registry.owner_loop_call(
                    partial(registry.attach_cancel_callback, handle, callback)
                )

            channel_token = _pending_call_abort_channel.set(_register_abort_channel)
        started = time.monotonic()
        try:
            try:
                if handle is not None and handle.aborted:
                    # The abort preceded this registration (the Stop landed in
                    # the OE-claim → in-graph-dispatch handoff): skip the body
                    # and settle interrupted.
                    raise asyncio.CancelledError(INTERRUPTED_CALL_CONTENT)
                with local_suspend_request_context():
                    from agent_engine_runner_shared.context import customer_origin_scope

                    with customer_origin_scope():
                        result = local_executor()
                        suspend_marker = get_requested_suspend()
                if (
                    registry is not None
                    and handle is not None
                    and not registry.owner_loop_call(partial(registry.claim_settlement, handle))
                ):
                    # A Stop claimed the outcome as the body returned: settle
                    # interrupted so the durable record matches the verdict the
                    # caller already got.
                    raise asyncio.CancelledError(INTERRUPTED_CALL_CONTENT)
            except BaseException as error:
                # Claim the settlement before the blocking report below: a Stop
                # landing mid-report reads already_settled instead of claiming
                # a stop the durable record will contradict. The claim's answer
                # is the stop decision — an abort that won first owns the
                # outcome, so an already-classified genuine error is never
                # repainted as a stopped call.
                stopped_by_call_interrupt = (
                    handle is not None
                    and registry is not None
                    and not registry.owner_loop_call(partial(registry.claim_settlement, handle))
                )
                is_control_flow = (
                    isinstance(error, asyncio.CancelledError)
                    or is_framework_control_flow is not None
                    and is_framework_control_flow(error)
                )
                if not is_control_flow and not isinstance(error, Exception):
                    raise
                duration_ms = (time.monotonic() - started) * 1000
                # The settlement winner, not the exception type alone, decides
                # the reported outcome: when the Stop claimed the call first, an
                # ordinary exception from cancellation cleanup (a provider
                # translating CancelledError into its own error type) is the
                # stop's noise, not the call's outcome.
                settle_interrupted = is_control_flow or stopped_by_call_interrupt
                status = "interrupted" if settle_interrupted else "error"
                classified = (
                    None
                    if settle_interrupted
                    else classify_tool_api_error(
                        error, None, credentials=request_credential_values()
                    )
                )
                error_text = classified[1] if classified else f"{type(error).__name__}: {error}"
                tool_api_error = classified[0] if classified else None
                log_tool_result(
                    tool_name,
                    step,
                    status,
                    result=None,
                    error=error_text,
                    duration_ms=duration_ms,
                    prefix="TOOL",
                )
                if stopped_by_call_interrupt and not is_control_flow:
                    # Report the stopped reason the caller's verdict already
                    # named; the cleanup exception stays in the log above as
                    # the diagnostic.
                    error_text = f"CancelledError: {INTERRUPTED_CALL_CONTENT}"
                try:
                    report_oe_result(
                        oe_url=self.oe_url,
                        execution_id=self.execution_id,
                        tool_name=tool_name,
                        step=step,
                        status=status,
                        result=None,
                        error=error_text,
                        duration_ms=duration_ms,
                        pod_name=socket.gethostname(),
                        metadata=metadata,
                        tool_call_id=tool_call_id,
                        owner_url=self._current_oe_owner_url(),
                        on_owner_failure=self._report_oe_owner_failure,
                        tool_api_error=tool_api_error,
                    )
                except Exception as settlement_error:
                    raise settlement_error from error
                if stopped_by_call_interrupt:
                    # This call was deliberately stopped: the graph continues
                    # with the stopped-call result instead of unwinding.
                    return _CALL_INTERRUPTED
                raise

            duration_ms = (time.monotonic() - started) * 1000
            pod_name = socket.gethostname()
            status = "suspend" if suspend_marker is not None else "success"
            reported_result = json.dumps(suspend_marker) if suspend_marker is not None else result

            log_tool_result(
                tool_name,
                step,
                status,
                result=reported_result,
                error=None,
                duration_ms=duration_ms,
                prefix="TOOL",
            )
            report_oe_result(
                oe_url=self.oe_url,
                execution_id=self.execution_id,
                tool_name=tool_name,
                step=step,
                status=status,
                result=reported_result,
                error=None,
                duration_ms=duration_ms,
                pod_name=pod_name,
                metadata=metadata,
                tool_call_id=tool_call_id,
                owner_url=self._current_oe_owner_url(),
                on_owner_failure=self._report_oe_owner_failure,
            )
            if suspend_marker is not None:
                # SuspendPayload.to_json records this out of band, so relayed tool
                # output cannot forge a wait.
                return self._handle_suspend(reported_result)
            return result
        finally:
            if channel_token is not None:
                _pending_call_abort_channel.reset(channel_token)
            if handle is not None and registry is not None:
                registry.owner_loop_call(partial(registry.end_work, handle))

    def _handle_suspend(self, result: Any) -> Any:
        """Fire the HITL interrupt for an OE-confirmed suspend.

        Validates the payload against ``SuspendPayload``. Durable activities
        commit the wait to OE; native executions call the framework suspend
        handler and use its resumed value as the tool result.
        """
        parsed = result
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
            except (json.JSONDecodeError, ValueError):
                parsed = None

        if isinstance(parsed, dict):
            payload = SuspendPayload.model_validate(parsed)
            from agent_engine_runner_shared.workflow import current_attempt_context

            if current_attempt_context() is not None:
                from agent_engine_runner_shared.hooks import get_durable_activity_suspend_handler

                durable_interrupt = get_durable_activity_suspend_handler()
                if durable_interrupt is None:
                    raise RuntimeError(
                        "No durable activity suspend handler registered. "
                        "The active framework does not support durable waits."
                    )
                # This wait is the tool activity's result, so durable replay can
                # safely return the recorded human decision at the same position.
                durable_interrupt(payload.model_dump())
                raise RuntimeError("durable activity suspend handler returned")

            from agent_engine_runner_shared.hooks import get_suspend_handler

            interrupt = get_suspend_handler()
            if interrupt is None:
                raise RuntimeError(
                    "No suspend handler registered. "
                    "Ensure the framework SDK calls register_suspend_handler() before run()."
                )
            human_decision = interrupt(payload.model_dump())
            if not isinstance(human_decision, dict):
                raise TypeError(
                    f"Expected dict from HITL interrupt, got {type(human_decision).__name__}"
                )
            return json.dumps(human_decision)
        raise ToolExecutionError("OE reported suspend but the result is not a suspend payload")


# NOTE: SecureWrappedLLM has been moved to agent-engine-sdk-langgraph.secure_llm
# to make agent-engine-runner-shared framework-agnostic.


# =============================================================================
# Context-Aware Tool Wrapper
# =============================================================================


def _is_two_element_sequence(value: Any) -> bool:
    """True for a two-element list/tuple (a serialized content+artifact pair)."""
    return isinstance(value, (list, tuple)) and len(value) == 2


def _coerce_content_and_artifact(
    result: Any,
    response_format: Literal["content", "content_and_artifact"],
    tool_declared_format: Literal["content", "content_and_artifact"] = "content",
) -> Any:
    """Shape a tool result into the ``(content, artifact)`` tuple ToolNode requires.

    ``response_format`` is the wire format the StructuredTool is actually built
    with. When it isn't ``"content_and_artifact"`` (e.g. the ADK SDK), results
    pass through unchanged.

    ``tool_declared_format`` is the tool author's own original format and only
    governs a normal (non-interrupted) result: only a tool that itself declared
    ``"content_and_artifact"`` gets its two-element result shape-matched into a
    tuple. Otherwise the result is always wrapped whole as ``(result, None)`` —
    a tool that never opted in must not have a genuine two-element result
    silently misread as content+artifact.

    An interrupted call is handled independently of ``tool_declared_format``:
    every tool gets the same marker once wired for ``content_and_artifact``.
    """
    if result is _CALL_INTERRUPTED:
        if response_format == "content_and_artifact":
            return (INTERRUPTED_CALL_CONTENT, {CALL_INTERRUPTED_ARTIFACT_KEY: True})
        return {"interrupted": True}
    if response_format != "content_and_artifact":
        return result
    if tool_declared_format == "content_and_artifact" and _is_two_element_sequence(result):
        content, artifact = result
        # Only the OE interrupted-status branch above may set this key — a
        # tool's own artifact must never be able to forge it.
        if isinstance(artifact, dict) and CALL_INTERRUPTED_ARTIFACT_KEY in artifact:
            artifact = {
                key: value
                for key, value in artifact.items()
                if key != CALL_INTERRUPTED_ARTIFACT_KEY
            }
        return (content, artifact)
    return (result, None)


def create_secure_tool_function(
    original_tool: Any,
    tool_name: str,
    *,
    allow_direct: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    is_local: bool = True,
    provider_type: Optional[str] = None,
    scopes: Optional[list[str]] = None,
    response_format: Literal["content", "content_and_artifact"] = "content",
    tool_declared_format: Optional[Literal["content", "content_and_artifact"]] = None,
    redact_fields: Optional[list[str]] = None,
    is_framework_control_flow: Optional[Callable[[BaseException], bool]] = None,
) -> Callable:
    """
    Create a wrapped tool function that routes through SecureToolWrapper.

    The wrapper is looked up from context at call time, allowing the tool
    to be built before execution context exists.

    Args:
        original_tool: The original LangChain tool
        tool_name: Name of the tool
        allow_direct: Whether to allow direct execution when wrapper is missing
        metadata: Optional execution metadata to send to OE
        is_local: Whether to execute the approved tool on the current AER stack
        response_format: The wire format the caller's StructuredTool is built
            with. See ``_coerce_content_and_artifact``.
        tool_declared_format: The tool author's own original format, used only
            to decide whether a normal result gets shape-matched into a tuple.
            Defaults to ``response_format``.

    Returns:
        Wrapped function that routes through OE via SecureToolWrapper
    """
    import functools

    from agent_engine_runner_shared.context import get_current_wrapper

    effective_tool_format = (
        tool_declared_format if tool_declared_format is not None else response_format
    )

    wrapped_source = (
        getattr(original_tool, "func", None)
        or getattr(original_tool, "coroutine", None)
        or original_tool
    )

    # Declare at registration time whether the body can attach an abort
    # channel: only a coroutine body gets one (its task lives on the worker
    # thread's private loop). Mirrors the value-based decision
    # _invoke_registered_tool makes at invoke time. A misdeclaration degrades
    # honestly either way — the registry never masks a refused abort.
    _tool_func = getattr(original_tool, "func", None)
    _tool_coroutine = getattr(original_tool, "coroutine", None)
    call_channel_expected = (
        (_tool_func is None and callable(getattr(original_tool, "ainvoke", None)))
        or inspect.iscoroutinefunction(_tool_func)
        or inspect.iscoroutinefunction(_tool_coroutine)
        or (
            _tool_func is None
            and _tool_coroutine is None
            and inspect.iscoroutinefunction(original_tool)
        )
    )

    @functools.wraps(wrapped_source)
    def wrapped_func(**kwargs):
        # The framework SDK (LangChain InjectedToolCallId / ADK function_call_id)
        # passes the stable tool-call id as a reserved keyword. Pop it so it is
        # not forwarded to the tool as an argument; None when unavailable.
        tool_call_id = kwargs.pop("tool_call_id", None)
        wrapper = get_current_wrapper()
        if wrapper is None:
            if not allow_direct:
                raise RuntimeError(
                    f"SecureToolWrapper missing for tool {tool_name}; "
                    "execution not allowed (RUNNER_ALLOW_DIRECT_TOOL_EXECUTION=false)"
                )
            logger.warning(f"No wrapper for tool {tool_name}, executing directly (UNSAFE)")
            direct_result = _invoke_registered_tool(original_tool, kwargs)
            return _coerce_content_and_artifact(
                direct_result, response_format, effective_tool_format
            )

        local_executor: Optional[Callable[[], Any]] = None
        if is_local:
            # Capture the registered tool object and this invocation's arguments.
            # OE approves first; invoking this closure afterward keeps execution
            # on the framework's original stack with its live native context.
            def invoke_registered_tool() -> Any:
                with tool_memory_read_ownership():
                    return _invoke_registered_tool(original_tool, kwargs)

            local_executor = invoke_registered_tool

        try:
            # Delegate to SecureToolWrapper which handles:
            # 1. Sends the intercepted tool call to OE
            # 2. Receives the final success/suspend/error outcome
            # 3. Converts suspend payloads back into the framework interrupt
            #
            # raw_on_interrupt=True: this is the one call site allowed to see
            # the private _CALL_INTERRUPTED sentinel — it's coerced into a safe
            # value immediately below, never returned from wrapped_func itself.
            result = wrapper.execute_tool(
                tool_name=tool_name,
                arguments=kwargs,
                provider_type=provider_type,
                scopes=scopes,
                tool_call_id=tool_call_id,
                metadata=metadata,
                is_local=is_local,
                local_executor=local_executor,
                is_framework_control_flow=is_framework_control_flow,
                raw_on_interrupt=True,
                redact_fields=redact_fields,
                call_channel_expected=call_channel_expected,
            )
        except ToolExecutionError as exc:
            logger.info(
                "Propagating tool execution error to graph for %s: %s",
                tool_name,
                exc.error,
            )
            raise

        return _coerce_content_and_artifact(result, response_format, effective_tool_format)

    return wrapped_func
