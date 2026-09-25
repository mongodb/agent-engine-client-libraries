"""Activity helpers for OE-owned durable workflow identity.

SDKs carry semantic input and a deterministic position; OE owns hashing and
replay. No client-side input digest is computed or sent. Concurrent activities
at distinct positions are supported when adapters preallocate ordinals from
stable framework identity before dispatch.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from google.protobuf import struct_pb2

from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_OUTCOME_KIND_COMPLETED,
    ACTIVITY_OUTCOME_KIND_DENIED,
    ACTIVITY_OUTCOME_KIND_FAILED,
    ACTIVITY_OUTCOME_KIND_SUSPENDED,
    ActivityCommand,
    ActivityContext,
    ActivityKind,
    ActivityOutcome,
    ActivitySuspension,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    WORKFLOW_ERROR_CODE_CONFLICT,
    WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
    WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN,
    WORKFLOW_ERROR_CODE_UNAUTHORIZED,
    ActivityPosition,
    OperationPath,
    WorkflowError,
    WorkflowErrorCode,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.utils import get_request_timeout
from agent_engine_runner_shared.workflow.client import (
    ActivityReplay,
    StartActivityResult,
    WorkflowClient,
    WorkflowClientError,
)
from agent_engine_runner_shared.workflow.context import (
    activity_requires_reconstruction,
    current_attempt_context,
    current_operation_path,
    current_step_ordinal,
    record_interrupted_activity,
    record_observed_activity,
    record_reconstructed_activity_interrupt,
)
from agent_engine_runner_shared.workflow.protojson import (
    json_to_proto_struct,
    json_to_proto_value,
    proto_struct_to_json,
    proto_value_to_json,
)

__all__ = [
    "DurableActivityDeniedError",
    "DurableActivityInterrupted",
    "DurableActivitySuspended",
    "ReplayedActivityFailedError",
    "build_activity_command",
    "completed_outcome",
    "denied_outcome",
    "failed_outcome",
    "run_serial_activity",
    "run_streaming_activity",
    "semantic_input_from_json",
    "suspended_outcome",
    "unwrap_activity_outcome",
    "value_to_json",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")
ActivityResolvedHook = Callable[[WorkflowClient, ActivityContext, Any], None]


class DurableActivityDeniedError(Exception):
    """Durable policy denial recorded as ACTIVITY_OUTCOME_KIND_DENIED."""


class DurableActivityControlFlow(Exception):
    """Framework control flow that must cross tool-level error handling."""


class DurableActivitySuspended(DurableActivityControlFlow):
    """Control flow indicating that an activity is waiting for an external result."""

    def __init__(self, reason: str, context: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.context = context


class DurableActivityInterrupted(DurableActivityControlFlow):
    """In-process framework interrupt that leaves the activity STARTED."""

    def __init__(self, control_flow: BaseException) -> None:
        super().__init__(str(control_flow))
        self.control_flow = control_flow


class ReplayedActivityFailedError(WorkflowClientError):
    """A recorded FAILED outcome returned by activity replay."""


def semantic_input_from_json(value: Any) -> struct_pb2.Value:
    """Convert an arbitrary JSON-shaped value into `google.protobuf.Value`."""
    return json_to_proto_value(value)


def value_to_json(value: struct_pb2.Value) -> Any:
    """Decode a `google.protobuf.Value` back to plain JSON."""
    return proto_value_to_json(value)


def build_activity_command(
    *,
    attempt: AttemptContext,
    kind: ActivityKind,
    name: str,
    activity_ordinal: int,
    semantic_input: Any,
    step_ordinal: int | None = None,
    operation_path: OperationPath | None = None,
) -> ActivityCommand:
    """Build one ActivityCommand from an OE-issued attempt context.

    ``activity_ordinal`` is monotonic for the whole execution within the
    operation path. ``step_ordinal`` is the active root superstep
    (``LatestCommittedStepOrdinal + 1``); when omitted it is taken from the
    request-scoped step counter.
    """
    if not attempt.workflow_identity.execution_id:
        raise WorkflowClientError(
            WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
            "ActivityCommand requires AttemptContext.workflow_identity",
        )
    if activity_ordinal <= 0:
        raise WorkflowClientError(
            WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
            "activity_ordinal must be positive",
        )
    resolved_step = current_step_ordinal() if step_ordinal is None else step_ordinal
    if resolved_step <= 0:
        raise WorkflowClientError(
            WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
            "step_ordinal must be positive",
        )
    if operation_path is None:
        operation_path = current_operation_path()
    return ActivityCommand(
        workflow_identity=attempt.workflow_identity,
        attempt_id=attempt.attempt_id,
        fencing_token=attempt.fencing_token,
        position=ActivityPosition(
            operation_path=operation_path,
            activity_ordinal=activity_ordinal,
            step_ordinal=resolved_step,
        ),
        activity_kind=kind,
        activity_name=name,
        semantic_input=semantic_input_from_json(semantic_input),
    )


def completed_outcome(
    context: ActivityContext,
    result: Any,
) -> ActivityOutcome:
    """Build a COMPLETED outcome under the activity's OE-issued fence."""
    return ActivityOutcome(
        workflow_identity=context.workflow_identity,
        activity_id=context.activity_id,
        attempt_id=context.attempt_id,
        fencing_token=context.fencing_token,
        outcome_kind=ACTIVITY_OUTCOME_KIND_COMPLETED,
        result=semantic_input_from_json(result),
    )


def _context_from_outcome(outcome: ActivityOutcome) -> ActivityContext:
    return ActivityContext(
        workflow_identity=outcome.workflow_identity,
        activity_id=outcome.activity_id,
        attempt_id=outcome.attempt_id,
        fencing_token=outcome.fencing_token,
    )


def failed_outcome(
    context: ActivityContext,
    message: str,
    code: WorkflowErrorCode = WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN,
) -> ActivityOutcome:
    """Build a FAILED outcome under the activity's OE-issued fence."""
    return ActivityOutcome(
        workflow_identity=context.workflow_identity,
        activity_id=context.activity_id,
        attempt_id=context.attempt_id,
        fencing_token=context.fencing_token,
        outcome_kind=ACTIVITY_OUTCOME_KIND_FAILED,
        error=WorkflowError(code=code, message=message),
    )


def denied_outcome(context: ActivityContext, message: str) -> ActivityOutcome:
    """Build a DENIED outcome for a policy denial under the OE-issued fence."""
    return ActivityOutcome(
        workflow_identity=context.workflow_identity,
        activity_id=context.activity_id,
        attempt_id=context.attempt_id,
        fencing_token=context.fencing_token,
        outcome_kind=ACTIVITY_OUTCOME_KIND_DENIED,
        # OE WorkflowError.Validate rejects UNSPECIFIED; denied fixtures use UNAUTHORIZED.
        error=WorkflowError(
            code=WORKFLOW_ERROR_CODE_UNAUTHORIZED,
            message=message,
        ),
    )


def suspended_outcome(
    activity_context: ActivityContext,
    *,
    reason: str,
    context: dict[str, Any],
) -> ActivityOutcome:
    """Build a SUSPENDED outcome under the activity's OE-issued fence."""
    return ActivityOutcome(
        workflow_identity=activity_context.workflow_identity,
        activity_id=activity_context.activity_id,
        attempt_id=activity_context.attempt_id,
        fencing_token=activity_context.fencing_token,
        outcome_kind=ACTIVITY_OUTCOME_KIND_SUSPENDED,
        suspension=ActivitySuspension(
            reason=reason,
            context=json_to_proto_struct(context),
        ),
    )


def unwrap_activity_outcome(outcome: ActivityOutcome) -> Any:
    """Map a terminal ActivityOutcome to a return value or a stable SDK error.

    Callers branch on exception type, not message text.
    """
    if outcome.outcome_kind == ACTIVITY_OUTCOME_KIND_COMPLETED:
        return value_to_json(outcome.result)
    if outcome.outcome_kind == ACTIVITY_OUTCOME_KIND_DENIED:
        raise DurableActivityDeniedError(
            outcome.error.message or "durable activity denied by policy"
        )
    if outcome.outcome_kind == ACTIVITY_OUTCOME_KIND_FAILED:
        raise ReplayedActivityFailedError(
            outcome.error.code or WORKFLOW_ERROR_CODE_OUTCOME_UNKNOWN,
            outcome.error.message or "durable activity failed",
        )
    if outcome.outcome_kind == ACTIVITY_OUTCOME_KIND_SUSPENDED:
        raise DurableActivitySuspended(
            outcome.suspension.reason or "durable activity suspended",
            proto_struct_to_json(outcome.suspension.context),
        )
    raise WorkflowClientError(
        WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
        "activity outcome kind is unspecified",
    )


# Concurrent activities share one attempt when adapters assign stable positions.
# LLM calls use the platform-owned operational step; adapters may also
# preallocate tool positions from framework call IDs. Remaining unkeyed effects
# stay exclusive.
# The gate is attempt-local: multiple nonexclusive starts may overlap, but an
# exclusive start conflicts with any active activity and any start conflicts
# with an active exclusive owner.
_ATTEMPT_GATES: dict[str, "_AttemptActivityGate"] = {}
_ATTEMPT_GATES_LOCK = threading.Lock()


class _AttemptActivityGate:
    """Per-attempt concurrent-admission counters."""

    __slots__ = ("exclusive", "nonexclusive")

    def __init__(self) -> None:
        self.exclusive = False
        self.nonexclusive = 0


class _ActivityAdmission:
    """Admit one activity start under the attempt-local concurrency gate."""

    _CONFLICT_MESSAGE = (
        "concurrent activities require a stable framework identity; "
        "unkeyed or LLM activities must run without overlapping siblings"
    )

    def __init__(self, attempt_id: str, *, exclusive: bool) -> None:
        self._attempt_id = attempt_id
        self._exclusive = exclusive

    def __enter__(self) -> None:
        with _ATTEMPT_GATES_LOCK:
            gate = _ATTEMPT_GATES.setdefault(self._attempt_id, _AttemptActivityGate())
            if self._exclusive:
                if gate.exclusive or gate.nonexclusive > 0:
                    raise WorkflowClientError(
                        WORKFLOW_ERROR_CODE_CONFLICT,
                        self._CONFLICT_MESSAGE,
                    )
                gate.exclusive = True
                return
            if gate.exclusive:
                raise WorkflowClientError(
                    WORKFLOW_ERROR_CODE_CONFLICT,
                    self._CONFLICT_MESSAGE,
                )
            gate.nonexclusive += 1

    def __exit__(self, *exc_info: object) -> None:
        with _ATTEMPT_GATES_LOCK:
            gate = _ATTEMPT_GATES.get(self._attempt_id)
            if gate is None:
                return
            if self._exclusive:
                gate.exclusive = False
            else:
                gate.nonexclusive = max(0, gate.nonexclusive - 1)
            if not gate.exclusive and gate.nonexclusive == 0:
                _ATTEMPT_GATES.pop(self._attempt_id, None)


def _report_denied(client: WorkflowClient, context: ActivityContext, error: Exception) -> None:
    """Record a DENIED outcome so a replay reproduces the policy denial."""
    try:
        client.report_outcome(denied_outcome(context, str(error)))
    except WorkflowClientError as report_error:
        raise report_error from error
    except Exception:  # noqa: BLE001 - outcome report is best-effort here
        logger.exception("Failed to report denied activity outcome")


def _report_failure(client: WorkflowClient, context: ActivityContext, error: Exception) -> None:
    """Record a FAILED outcome for a dispatched activity whose worker raised.

    A coded report failure (e.g. stale fence) supersedes an ordinary worker
    failure. Unsupported durable control flow remains the explicit terminal
    error because this client does not attempt to recover or reinterpret it.
    """
    try:
        client.report_outcome(failed_outcome(context, str(error)))
    except WorkflowClientError as report_error:
        if isinstance(error, DurableActivityControlFlow):
            # Unsupported durable control flow is already the terminal failure.
            # Do not replace it with a second failure from best-effort reporting.
            logger.warning(
                "Failed to report durable activity failure; preserving the "
                "original unsupported-operation error",
                exc_info=True,
            )
            return
        raise report_error from error
    except Exception:  # noqa: BLE001 - outcome report is best-effort here
        logger.exception("Failed to report failed activity outcome")


def _require_attempt(attempt: AttemptContext | None, caller: str) -> AttemptContext:
    if attempt is None:
        attempt = current_attempt_context()
    if attempt is None:
        raise WorkflowClientError(
            WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
            f"{caller} requires an OE-issued AttemptContext",
        )
    return attempt


def run_serial_activity(
    *,
    client: WorkflowClient,
    kind: ActivityKind,
    name: str,
    activity_ordinal: int,
    semantic_input: Any,
    execute: Callable[[ActivityContext], T],
    on_activity_resolved: ActivityResolvedHook | None = None,
    attempt: AttemptContext | None = None,
    step_ordinal: int | None = None,
    exclusive: bool = False,
) -> T:
    """Start one activity: replay a recorded outcome or dispatch then report.

    Position identity (``activity_ordinal`` within the operation path and step)
    is what lets OE replay a recorded outcome instead of re-running the
    operation. Keyed siblings may run concurrently; pass ``exclusive=True`` for
    starts that lack a stable framework identity. Heartbeats are owned by the
    runtime lifecycle, not this helper.
    """
    attempt = _require_attempt(attempt, "run_serial_activity")
    command = build_activity_command(
        attempt=attempt,
        kind=kind,
        name=name,
        activity_ordinal=activity_ordinal,
        step_ordinal=step_ordinal,
        semantic_input=semantic_input,
    )
    exclusive_guard = _ActivityAdmission(attempt.attempt_id, exclusive=exclusive)
    with exclusive_guard:
        return _run_dispatched_activity(
            client=client,
            command=command,
            execute=execute,
            on_activity_resolved=on_activity_resolved,
        )


def _run_dispatched_activity(
    *,
    client: WorkflowClient,
    command: ActivityCommand,
    execute: Callable[[ActivityContext], T],
    on_activity_resolved: ActivityResolvedHook | None,
) -> T:
    started: StartActivityResult = client.start_activity(command)
    record_observed_activity(command.position)
    reconstructing = False
    if isinstance(started, ActivityReplay):
        # The current resume batch is the authority for reconstruction. Re-enter
        # only an activity whose stable id is present there; every other replay
        # returns its recorded result without running the activity body again.
        reconstructing = activity_requires_reconstruction(started.outcome.activity_id)
        if not reconstructing:
            # Ordinary replay returns the recorded outcome without re-running
            # the side effect. A resolved framework interrupt is different:
            # its body must run again so the framework can recreate interrupt().
            result = unwrap_activity_outcome(started.outcome)
            if on_activity_resolved is not None:
                on_activity_resolved(client, _context_from_outcome(started.outcome), result)
            return result
        context = _context_from_outcome(started.outcome)
    else:
        context = started.context

    try:
        result = execute(context)
    except DurableActivityDeniedError as error:
        # ResolveActivities already made a reconstructed interrupt terminal.
        # Preserve the callback's denial instead of replacing it with a second,
        # conflicting outcome report.
        if not reconstructing:
            _report_denied(client, context, error)
        raise
    except DurableActivityInterrupted as interrupted:
        if reconstructing and record_reconstructed_activity_interrupt(context.activity_id) > 1:
            raise RuntimeError(
                "a durable local tool cannot raise another native interrupt "
                "after its recorded answer is resumed"
            ) from interrupted.control_flow
        # FinalizeStep needs the exact command that started this activity. The
        # framework adapter joins it to the native interrupt after LangGraph
        # has written the checkpoint that exposes the new interrupt id.
        record_interrupted_activity(command, interrupted.control_flow)
        raise interrupted.control_flow
    except DurableActivitySuspended as suspension:
        if reconstructing:
            raise RuntimeError(
                "a durable local tool cannot suspend again after its recorded "
                "native interrupt answer is resumed"
            ) from suspension
        # OE must commit the wait before the adapter releases attempt-local
        # graph scratch and reports the execution as suspended. Terminal-fail
        # only on a confirmed conflict (for example another sibling is still
        # unfinished). Ambiguous delivery (OUTCOME_UNKNOWN) must not invent a
        # FAILED outcome that can contradict a response-lost suspension.
        try:
            client.report_outcome(
                suspended_outcome(
                    context,
                    reason=suspension.reason,
                    context=suspension.context,
                )
            )
        except WorkflowClientError as error:
            if error.code == WORKFLOW_ERROR_CODE_CONFLICT:
                _report_failure(client, context, error)
                raise suspension from error
            raise
        raise
    except Exception as error:
        if not reconstructing:
            _report_failure(client, context, error)
        raise
    if reconstructing:
        # ResolveActivities already made this position terminal with the
        # interrupt answer. The resumed graph state, not a second activity
        # outcome, records the code that ran after interrupt() returned. The
        # reconstructed tool result must still reach its post-outcome hook so
        # Memory can acknowledge the already-terminal activity before this
        # step commits.
        if on_activity_resolved is not None:
            on_activity_resolved(client, context, result)
        return result
    try:
        # Building the outcome can fail (e.g. a non-JSON result); the
        # dispatched activity must still reach a terminal state.
        outcome = completed_outcome(context, result)
    except Exception as error:
        _report_failure(client, context, error)
        raise
    client.report_outcome(outcome)
    if on_activity_resolved is not None:
        on_activity_resolved(client, context, result)
    return result


def run_streaming_activity(
    *,
    kind: ActivityKind,
    name: str,
    activity_ordinal: int,
    semantic_input: Any,
    execute: Callable[[], Iterator[T]],
    replay: Callable[[Any], Iterator[T]],
    fold: Callable[[list[T]], Any],
    on_activity_resolved: ActivityResolvedHook | None = None,
    client: WorkflowClient | None = None,
    oe_url: str | None = None,
    attempt: AttemptContext | None = None,
    step_ordinal: int | None = None,
    exclusive: bool = False,
) -> Iterator[T]:
    """Run one activity whose worker streams items to the caller.

    Same protocol as `run_serial_activity` for a generator-shaped worker:
    a recorded outcome is translated back into items via `replay` (event-log
    replay — the side effect is not re-run); a fresh dispatch yields
    `execute()`'s items through to the caller while collecting them, then
    records `fold(items)` as the terminal outcome under the OE-issued fence.
    An abandoned or failed stream reports a FAILED outcome so the dispatched
    activity never dangles until lease expiry.

    Pass an injected `client` to reuse a caller-owned connection; otherwise an
    owned client is created from `oe_url` for exactly this activity's lifetime.
    Pass ``exclusive=True`` when the start lacks a stable framework identity.
    """
    attempt = _require_attempt(attempt, "run_streaming_activity")
    command = build_activity_command(
        attempt=attempt,
        kind=kind,
        name=name,
        activity_ordinal=activity_ordinal,
        step_ordinal=step_ordinal,
        semantic_input=semantic_input,
    )

    owned: WorkflowClient | None = None
    if client is None:
        if not oe_url:
            raise WorkflowClientError(
                WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
                "run_streaming_activity requires a client or an oe_url",
            )
        owned = client = WorkflowClient(oe_url, timeout=get_request_timeout())
    exclusive_guard = _ActivityAdmission(attempt.attempt_id, exclusive=exclusive)
    try:
        with exclusive_guard:
            yield from _run_dispatched_streaming_activity(
                client=client,
                command=command,
                execute=execute,
                replay=replay,
                fold=fold,
                on_activity_resolved=on_activity_resolved,
            )
    finally:
        if owned is not None:
            owned.close()


def _run_dispatched_streaming_activity(
    *,
    client: WorkflowClient,
    command: ActivityCommand,
    execute: Callable[[], Iterator[T]],
    replay: Callable[[Any], Iterator[T]],
    fold: Callable[[list[T]], Any],
    on_activity_resolved: ActivityResolvedHook | None,
) -> Iterator[T]:
    started: StartActivityResult = client.start_activity(command)
    record_observed_activity(command.position)
    if isinstance(started, ActivityReplay):
        # OE already holds a recorded outcome for this position
        # (event-log replay): translate it back into items without
        # re-running the side effect.
        result = unwrap_activity_outcome(started.outcome)
        if on_activity_resolved is not None:
            on_activity_resolved(client, _context_from_outcome(started.outcome), result)
        yield from replay(result)
        return

    collected: list[T] = []
    try:
        for item in execute():
            collected.append(item)
            yield item
    except DurableActivityDeniedError as error:
        # A live policy denial is a DENIED outcome, not a failure, so
        # a replay reproduces the same denial semantics.
        _report_denied(client, started.context, error)
        raise
    except Exception as error:
        _report_failure(client, started.context, error)
        raise
    except BaseException as error:
        # GeneratorExit (abandoned stream) or cancellation: report a
        # terminal failure best-effort, then let the exit proceed.
        try:
            client.report_outcome(
                failed_outcome(started.context, f"stream abandoned: {type(error).__name__}")
            )
        except Exception:  # noqa: BLE001 - never mask generator shutdown
            logger.exception("Failed to report abandoned activity outcome")
        raise
    try:
        # Folding or building the outcome can fail (e.g. a non-JSON
        # payload); the dispatched activity must still reach a
        # terminal state.
        result = fold(collected)
        outcome = completed_outcome(started.context, result)
    except Exception as error:
        _report_failure(client, started.context, error)
        raise
    client.report_outcome(outcome)
    if on_activity_resolved is not None:
        on_activity_resolved(client, started.context, result)
