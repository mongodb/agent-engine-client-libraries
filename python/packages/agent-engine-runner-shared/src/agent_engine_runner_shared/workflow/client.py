"""Thin ProtoJSON HTTP clients for the OE workflow attempt and activity endpoints.

Vocabulary: an *attempt* is one runtime's fenced lease on a whole execution —
it is started once per invocation and renewed by heartbeats. An *activity* is
one replayable operation performed inside an attempt (an LLM, tool, or Memory call);
OE records its outcome so a later attempt can replay it instead of re-running
the side effect.

These clients own transport, response qualification, and stable error
translation. Workflow transitions, hashing, replay, and lease decisions are OE
authority and are never reproduced here.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import httpx

from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ActivityCommand,
    ActivityContext,
    ActivityHeartbeatRequest,
    ActivityMemoryCommand,
    ActivityOutcome,
    StepActivityEntry,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    WORKFLOW_ERROR_CODE_CONFLICT,
    WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
    WORKFLOW_ERROR_CODE_NOT_FOUND,
    WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN,
    WORKFLOW_ERROR_CODE_STALE_FENCE,
    WORKFLOW_ERROR_CODE_UNAUTHORIZED,
    WorkflowError,
    WorkflowIdentity,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import (
    AttemptContext,
    AttemptHeartbeatRequest,
    AttemptStartRequest,
    AttemptStartResponse,
)
from agent_engine_runner_shared.generated.workflow.v1.state_pb2 import (
    CompleteExecutionCommand,
    FinalizeStepCommand,
    FinalizeStepResponse,
)
from agent_engine_runner_shared.tls_client import (
    create_async_httpx_client_with_tls,
    create_httpx_client_with_tls,
)
from agent_engine_runner_shared.workflow.protojson import (
    WorkflowWireError,
    encode_protojson,
    parse_read,
)

__all__ = [
    "ACTIVITY_HEARTBEAT_PATH",
    "ACTIVITY_MEMORY_PATH",
    "ACTIVITY_OUTCOME_PATH",
    "ACTIVITY_START_PATH",
    "ATTEMPT_HEARTBEAT_PATH",
    "ATTEMPT_START_PATH",
    "EXECUTION_COMPLETE_PATH",
    "STEP_FINALIZE_PATH",
    "ActivityDispatch",
    "ActivityReplay",
    "AsyncWorkflowClient",
    "StartActivityResult",
    "WorkflowClient",
    "WorkflowClientError",
]

# The OE host bridge must register exactly these routes; a renamed server
# route would be indistinguishable from an old OE and silently pin sessions
# native, so integration tests should consume these constants.
ATTEMPT_START_PATH = "/executor/attempt/start"
ATTEMPT_HEARTBEAT_PATH = "/executor/attempt/heartbeat"
ACTIVITY_START_PATH = "/executor/activity/start"
ACTIVITY_OUTCOME_PATH = "/executor/activity/outcome"
ACTIVITY_MEMORY_PATH = "/executor/activity/memory"
ACTIVITY_HEARTBEAT_PATH = "/executor/activity/heartbeat"
STEP_FINALIZE_PATH = "/executor/step/finalize"
EXECUTION_COMPLETE_PATH = "/executor/complete"

_DEFAULT_TIMEOUT_SECONDS = 10.0

_STATUS_TO_CODE = {
    400: WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
    401: WORKFLOW_ERROR_CODE_UNAUTHORIZED,
    403: WORKFLOW_ERROR_CODE_UNAUTHORIZED,
    404: WORKFLOW_ERROR_CODE_NOT_FOUND,
    409: WORKFLOW_ERROR_CODE_CONFLICT,
    412: WORKFLOW_ERROR_CODE_STALE_FENCE,
}


@contextmanager
def _propagate_without_http_span(headers: dict[str, str]) -> Iterator[None]:
    """Propagate the active trace across an internal workflow transport call."""
    try:
        from opentelemetry.instrumentation.utils import suppress_http_instrumentation
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )
    except ImportError:
        # Tracing is an optional agent-engine-runner-shared dependency. Without it there is
        # no httpx instrumentation to suppress or active context to propagate.
        yield
        return

    TraceContextTextMapPropagator().inject(headers)
    with suppress_http_instrumentation():
        yield


class WorkflowClientError(Exception):
    """A workflow request failed with a stable `WorkflowErrorCode`."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _code_for_status(status_code: int) -> int:
    return _STATUS_TO_CODE.get(status_code, WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN)


def _validate_started_attempt(
    request: AttemptStartRequest, context: AttemptContext
) -> AttemptContext:
    """Reject an OE response that did not qualify the requested attempt."""
    if not context.attempt_id:
        reason = "attempt_id is missing"
    elif context.fencing_token <= 0:
        reason = "fencing_token is not positive"
    elif context.heartbeat_interval_ms <= 0:
        reason = "heartbeat_interval_ms is not positive"
    elif context.workflow_identity != request.workflow_identity:
        reason = "workflow_identity does not match the request"
    elif context.declaration != request.declaration:
        reason = "workflow declaration does not match the request"
    elif context.owner_id != request.owner_id:
        reason = "owner_id does not match the request"
    else:
        return context
    raise WorkflowClientError(
        WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
        f"attempt start response is not qualified: {reason}",
    )


def _interpret_start_response(
    status_code: int,
    body: bytes,
    request: AttemptStartRequest,
) -> AttemptContext | None:
    """Translate one attempt-start exchange into an accepted context or a stable error.

    A response carrying a parseable `WorkflowError` wins over the HTTP status. A
    404 without one comes from an OE that predates the workflow contract, so the
    invocation stays native.
    """
    try:
        response = parse_read(body, AttemptStartResponse())
    except WorkflowWireError as error:
        if status_code == 404:
            return None
        if 200 <= status_code < 300:
            raise WorkflowClientError(
                WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
                "attempt start response is not valid ProtoJSON",
            ) from error
        raise WorkflowClientError(
            _code_for_status(status_code),
            f"attempt start failed with HTTP {status_code}",
        ) from error

    if response.HasField("error"):
        raise WorkflowClientError(response.error.code, response.error.message)
    if not response.HasField("attempt_context"):
        # A bare coded WorkflowError body (unwrapped) parses into an empty
        # AttemptStartResponse under tolerant reads; it must still win over
        # the status — especially the 404-means-old-OE fallback below, which
        # would otherwise silently pin a session native on a real rejection.
        try:
            bare = parse_read(body, WorkflowError())
        except WorkflowWireError:
            bare = WorkflowError()
        if bare.code:
            raise WorkflowClientError(bare.code, bare.message or "attempt start failed")
    if not 200 <= status_code < 300:
        if status_code == 404:
            return None
        raise WorkflowClientError(
            _code_for_status(status_code),
            f"attempt start failed with HTTP {status_code}",
        )
    if not response.HasField("attempt_context"):
        raise WorkflowClientError(
            WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
            "attempt start response carries neither attempt_context nor error",
        )
    return _validate_started_attempt(request, response.attempt_context)


@dataclass(frozen=True)
class ActivityDispatch:
    """OE authorized new work; run it under this activity context."""

    context: ActivityContext


@dataclass(frozen=True)
class ActivityReplay:
    """OE already holds a terminal outcome for this position; do not redispatch."""

    outcome: ActivityOutcome


StartActivityResult = ActivityDispatch | ActivityReplay


def _decode_json_object(body: bytes) -> object:
    text = body.decode() if body else "{}"
    return json.loads(text or "{}")


def _raise_error_member(json_body: object, status_code: int, operation: str) -> None:
    """Raise the response's `error` member as a stable coded failure, if present."""
    if not isinstance(json_body, dict) or json_body.get("error") is None:
        return
    try:
        error = parse_read(json.dumps(json_body["error"]), WorkflowError())
    except WorkflowWireError:
        error = WorkflowError()
    if error.code:
        raise WorkflowClientError(error.code, error.message or f"{operation} failed")
    raise WorkflowClientError(
        _code_for_status(status_code),
        error.message or f"{operation} failed with HTTP {status_code}",
    )


def _raise_on_error_response(status_code: int, body: bytes, operation: str) -> None:
    """Translate an error response into a stable coded failure.

    Accepts an ``{"error": WorkflowError}`` member or a bare ``WorkflowError``
    body — and honors either even on a 2xx status, so a coded rejection (e.g.
    a stale fence) can never masquerade as a successful renewal or commit.
    Anything else maps from the HTTP status. Used by every endpoint that only
    signals success or failure (heartbeats, outcome reports).
    """
    try:
        json_body = _decode_json_object(body)
    except (ValueError, UnicodeDecodeError) as error:
        if 200 <= status_code < 300:
            return
        raise WorkflowClientError(
            _code_for_status(status_code),
            f"{operation} failed with HTTP {status_code}",
        ) from error
    _raise_error_member(json_body, status_code, operation)
    try:
        error = parse_read(json.dumps(json_body), WorkflowError())
    except WorkflowWireError:
        error = WorkflowError()
    if error.code:
        raise WorkflowClientError(
            error.code, error.message or f"{operation} failed with HTTP {status_code}"
        )
    if 200 <= status_code < 300:
        return
    raise WorkflowClientError(
        _code_for_status(status_code),
        error.message or f"{operation} failed with HTTP {status_code}",
    )


def _is_complete_workflow_identity(identity: WorkflowIdentity) -> bool:
    scope = identity.tenant_scope
    return bool(
        scope.org_id.strip()
        and scope.project_id.strip()
        and scope.workspace_id.strip()
        and identity.session_id.strip()
        and identity.execution_id.strip()
    )


def _require_dispatch_context(
    command: ActivityCommand, context: ActivityContext
) -> ActivityContext:
    """Reject an activity_context that cannot record an outcome under this command."""
    if (
        not context.activity_id
        or not context.attempt_id
        or context.fencing_token <= 0
        or not _is_complete_workflow_identity(context.workflow_identity)
    ):
        raise WorkflowClientError(
            WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
            "activity start response activity_context is missing identity fields",
        )
    if (
        context.attempt_id != command.attempt_id
        or context.fencing_token != command.fencing_token
        or context.workflow_identity != command.workflow_identity
    ):
        raise WorkflowClientError(
            WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
            "activity start response activity_context does not match the submitted command",
        )
    return context


def _interpret_activity_start_response(
    status_code: int, body: bytes, command: ActivityCommand
) -> StartActivityResult:
    """Translate a StartActivity exchange into a dispatch/replay decision.

    Provisional envelope until the OE contract publishes a generated
    StartActivity response message:
    `{activity_context} | {outcome} | {error}`.
    """
    try:
        json_body = _decode_json_object(body)
    except (ValueError, UnicodeDecodeError) as error:
        raise WorkflowClientError(
            WORKFLOW_ERROR_CODE_INVALID_ARGUMENT
            if 200 <= status_code < 300
            else _code_for_status(status_code),
            "activity start response is not valid JSON",
        ) from error

    _raise_error_member(json_body, status_code, "activity start")
    if not 200 <= status_code < 300:
        raise WorkflowClientError(
            _code_for_status(status_code),
            f"activity start failed with HTTP {status_code}",
        )
    if not isinstance(json_body, dict):
        raise WorkflowClientError(
            WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
            "activity start response must be a ProtoJSON object",
        )

    context: ActivityContext | None = None
    try:
        if json_body.get("outcome") is not None:
            return ActivityReplay(
                outcome=parse_read(json.dumps(json_body["outcome"]), ActivityOutcome())
            )
        if json_body.get("activity_context") is not None:
            context = parse_read(json.dumps(json_body["activity_context"]), ActivityContext())
    except WorkflowWireError as error:
        raise WorkflowClientError(WORKFLOW_ERROR_CODE_INVALID_ARGUMENT, str(error)) from error

    if context is not None:
        return ActivityDispatch(context=_require_dispatch_context(command, context))

    raise WorkflowClientError(
        WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
        "activity start response carries neither activity_context nor outcome",
    )


def _interpret_finalize_step_response(status_code: int, body: bytes) -> list[StepActivityEntry]:
    _raise_on_error_response(status_code, body, "step finalization")
    try:
        response = parse_read(body.decode() if body else "{}", FinalizeStepResponse())
    except (UnicodeDecodeError, WorkflowWireError) as error:
        raise WorkflowClientError(
            WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
            "step finalization response is not valid ProtoJSON",
        ) from error
    return list(response.entries)


def _transport_error(operation: str, error: httpx.TransportError) -> WorkflowClientError:
    # The request may have reached OE, so the outcome is unknown rather than failed.
    return WorkflowClientError(
        WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN,
        f"{operation} transport failure: {type(error).__name__}",
    )


class WorkflowClient:
    """Synchronous ProtoJSON client for the OE workflow attempt endpoints."""

    def __init__(
        self,
        oe_url: str,
        *,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = oe_url.rstrip("/")
        self._owns_client = client is None
        self._client = (
            client if client is not None else create_httpx_client_with_tls(self._base_url, timeout)
        )

    def start_attempt(self, request: AttemptStartRequest) -> AttemptContext | None:
        """Activate this runtime's fenced attempt for one whole execution.

        Returns ``None`` when OE has no workflow support (the invocation
        stays native). Individual operations inside the attempt are started
        separately via ``start_activity``.
        """
        response = self._post(ATTEMPT_START_PATH, encode_protojson(request), "attempt start")
        return _interpret_start_response(response.status_code, response.content, request)

    def heartbeat(self, request: AttemptHeartbeatRequest) -> None:
        """Renew the attempt lease under its current fence."""
        response = self._post(
            ATTEMPT_HEARTBEAT_PATH, encode_protojson(request), "attempt heartbeat"
        )
        _raise_on_error_response(response.status_code, response.content, "attempt heartbeat")

    def start_activity(self, command: ActivityCommand) -> StartActivityResult:
        """Start one replayable operation inside the current attempt.

        OE either authorizes new work (dispatch) or returns the outcome it
        recorded for this position (replay).
        """
        response = self._post(ACTIVITY_START_PATH, encode_protojson(command), "activity start")
        return _interpret_activity_start_response(response.status_code, response.content, command)

    def report_outcome(self, outcome: ActivityOutcome) -> None:
        """Persist one finished or suspended outcome under its OE-issued fence."""
        response = self._post(ACTIVITY_OUTCOME_PATH, encode_protojson(outcome), "activity outcome")
        _raise_on_error_response(response.status_code, response.content, "activity outcome")

    def ensure_memory_written(self, command: ActivityMemoryCommand) -> None:
        """Ensure one wrapper-produced activity Memory batch is acknowledged."""
        response = self._post(
            ACTIVITY_MEMORY_PATH,
            encode_protojson(command),
            "activity Memory synchronization",
        )
        _raise_on_error_response(
            response.status_code,
            response.content,
            "activity Memory synchronization",
        )

    def heartbeat_activity(self, request: ActivityHeartbeatRequest) -> None:
        """Renew one activity lease under its current fencing token."""
        response = self._post(
            ACTIVITY_HEARTBEAT_PATH, encode_protojson(request), "activity heartbeat"
        )
        _raise_on_error_response(response.status_code, response.content, "activity heartbeat")

    def finalize_step(self, command: FinalizeStepCommand) -> list[StepActivityEntry]:
        """Commit a root step or return its positioned suspension outcomes."""
        response = self._post(
            STEP_FINALIZE_PATH,
            encode_protojson(command),
            "step finalization",
        )
        return _interpret_finalize_step_response(response.status_code, response.content)

    def complete_execution(self, command: CompleteExecutionCommand) -> None:
        """Commit final state after a successful application invocation."""
        response = self._post(
            EXECUTION_COMPLETE_PATH,
            encode_protojson(command),
            "execution completion",
        )
        _raise_on_error_response(
            response.status_code,
            response.content,
            "execution completion",
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "WorkflowClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _post(self, path: str, payload: str, operation: str) -> httpx.Response:
        headers = {"Content-Type": "application/json"}
        try:
            with _propagate_without_http_span(headers):
                return self._client.post(
                    f"{self._base_url}{path}",
                    content=payload,
                    headers=headers,
                )
        except httpx.TransportError as error:
            raise _transport_error(operation, error) from error


class AsyncWorkflowClient:
    """Asynchronous ProtoJSON client with the same semantics as `WorkflowClient`."""

    def __init__(
        self,
        oe_url: str,
        *,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = oe_url.rstrip("/")
        self._timeout = timeout
        self._owns_client = client is None
        self._client = client
        self._client_lock = asyncio.Lock()
        self._closed = False

    async def start_attempt(self, request: AttemptStartRequest) -> AttemptContext | None:
        """Activate this runtime's fenced attempt for one whole execution.

        Returns ``None`` when OE has no workflow support (the invocation
        stays native). Individual operations inside the attempt are started
        separately via ``start_activity``.
        """
        response = await self._post(ATTEMPT_START_PATH, encode_protojson(request), "attempt start")
        return _interpret_start_response(response.status_code, response.content, request)

    async def heartbeat(self, request: AttemptHeartbeatRequest) -> None:
        """Renew the attempt lease under its current fence."""
        response = await self._post(
            ATTEMPT_HEARTBEAT_PATH, encode_protojson(request), "attempt heartbeat"
        )
        _raise_on_error_response(response.status_code, response.content, "attempt heartbeat")

    async def start_activity(self, command: ActivityCommand) -> StartActivityResult:
        """Start one replayable operation inside the current attempt.

        OE either authorizes new work (dispatch) or returns the outcome it
        recorded for this position (replay).
        """
        response = await self._post(
            ACTIVITY_START_PATH, encode_protojson(command), "activity start"
        )
        return _interpret_activity_start_response(response.status_code, response.content, command)

    async def report_outcome(self, outcome: ActivityOutcome) -> None:
        """Persist one finished or suspended outcome under its OE-issued fence."""
        response = await self._post(
            ACTIVITY_OUTCOME_PATH, encode_protojson(outcome), "activity outcome"
        )
        _raise_on_error_response(response.status_code, response.content, "activity outcome")

    async def ensure_memory_written(self, command: ActivityMemoryCommand) -> None:
        """Ensure one wrapper-produced activity Memory batch is acknowledged."""
        response = await self._post(
            ACTIVITY_MEMORY_PATH,
            encode_protojson(command),
            "activity Memory synchronization",
        )
        _raise_on_error_response(
            response.status_code,
            response.content,
            "activity Memory synchronization",
        )

    async def heartbeat_activity(self, request: ActivityHeartbeatRequest) -> None:
        """Renew one activity lease under its current fencing token."""
        response = await self._post(
            ACTIVITY_HEARTBEAT_PATH, encode_protojson(request), "activity heartbeat"
        )
        _raise_on_error_response(response.status_code, response.content, "activity heartbeat")

    async def finalize_step(self, command: FinalizeStepCommand) -> list[StepActivityEntry]:
        """Commit a root step or return its positioned suspension outcomes."""
        response = await self._post(
            STEP_FINALIZE_PATH,
            encode_protojson(command),
            "step finalization",
        )
        return _interpret_finalize_step_response(response.status_code, response.content)

    async def complete_execution(self, command: CompleteExecutionCommand) -> None:
        """Commit final state after a successful application invocation."""
        response = await self._post(
            EXECUTION_COMPLETE_PATH,
            encode_protojson(command),
            "execution completion",
        )
        _raise_on_error_response(
            response.status_code,
            response.content,
            "execution completion",
        )

    async def close(self) -> None:
        # Taken under the client lock so a close racing a first call cannot
        # leave a just-created owned client unclosed.
        async with self._client_lock:
            self._closed = True
            if self._owns_client and self._client is not None:
                await self._client.aclose()

    async def __aenter__(self) -> "AsyncWorkflowClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        # The TLS factory must be awaited, so the owned client is created on
        # first use; the lock keeps concurrent first calls from each creating
        # (and one leaking) a client.
        async with self._client_lock:
            if self._closed:
                raise RuntimeError("AsyncWorkflowClient is closed")
            if self._client is None:
                self._client = await create_async_httpx_client_with_tls(
                    self._base_url, self._timeout
                )
            return self._client

    async def _post(self, path: str, payload: str, operation: str) -> httpx.Response:
        client = await self._ensure_client()
        headers = {"Content-Type": "application/json"}
        try:
            with _propagate_without_http_span(headers):
                return await client.post(
                    f"{self._base_url}{path}",
                    content=payload,
                    headers=headers,
                )
        except httpx.TransportError as error:
            raise _transport_error(operation, error) from error
