"""Request-scoped workflow attempt context.

The only legitimate source of an `AttemptContext` is the OE-issued attempt-start
response. Application code never constructs or mutates one; an absent context
means the invocation is native.

`step_ordinal` is the active root superstep for this attempt. Activities admit
at that ordinal; each committed `FinalizeStep` advances it. The counter is
request-local and starts at 1 for an ordinary attempt or immediately after the
immutable source cutoff for a branch attempt.

The counter lives in a mutable holder so asyncio task copies of the ContextVar
still share one ordinal (LangGraph schedules checkpointer puts on child tasks).

`activity_ordinal` is monotonic within one operation path. Sibling identities
are allocated from stable keys (for example ToolCall ids) before concurrent
dispatch so inverted worker order cannot redefine identity.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import NamedTuple

from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import ActivityCommand
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    ActivityPosition,
    OperationPath,
    OperationPathSegment,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext

__all__ = [
    "AttemptContextBinding",
    "ChildOperationBoundary",
    "InterruptedActivity",
    "OperationPathResolver",
    "UnsupportedChildOperationFanOutError",
    "observed_activity_positions",
    "advance_step_ordinal",
    "allocate_activity_ordinal",
    "activity_requires_reconstruction",
    "attempt_context_scope",
    "child_operation_boundary_scope",
    "current_attempt_context",
    "current_operation_path",
    "current_pending_child_operation_batch",
    "current_step_ordinal",
    "interrupted_activities",
    "operation_path_resolver_scope",
    "preallocate_activity_ordinals",
    "preallocate_child_operation_ordinals",
    "record_observed_activity",
    "record_reconstructed_activity_interrupt",
    "record_interrupted_activity",
    "reset_attempt_context",
    "set_attempt_context",
    "set_activity_reconstruction_ids",
    "set_pending_child_operation_batch",
    "tool_activity_key",
]


@dataclass(frozen=True)
class ChildOperationBoundary:
    """Framework-neutral identity for one nested operation."""

    name: str
    occurrence_key: str


OperationPathResolver = Callable[[], Sequence[ChildOperationBoundary]]


class UnsupportedChildOperationFanOutError(RuntimeError):
    """One structural child produced ambiguous occurrences in the same step."""


_OperationPathKey = tuple[tuple[str, int], ...]


class _StructuralChild(NamedTuple):
    """A named child below one resolved parent operation path."""

    parent: _OperationPathKey
    name: str


class _StepChild(NamedTuple):
    """A structural child as used within one root workflow step."""

    step_ordinal: int
    child: _StructuralChild


class _ChildOccurrence(NamedTuple):
    """One attempt-local LangGraph occurrence of a step's structural child."""

    step_child: _StepChild
    occurrence_key: str


@dataclass
class _StepCounter:
    value: int


def _activity_position_key(
    position: ActivityPosition,
) -> tuple[int, tuple[tuple[str, int], ...], int]:
    return (
        position.step_ordinal,
        tuple((segment.name, segment.ordinal) for segment in position.operation_path.segments),
        position.activity_ordinal,
    )


@dataclass
class _ObservedActivityPositions:
    """Collects activity positions observed by this attempt."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _positions: dict[tuple[int, tuple[tuple[str, int], ...], int], ActivityPosition] = field(
        default_factory=dict
    )

    def record(self, position: ActivityPosition) -> None:
        copy = ActivityPosition()
        copy.CopyFrom(position)
        with self._lock:
            self._positions[_activity_position_key(position)] = copy

    def for_step(self, step_ordinal: int) -> list[ActivityPosition]:
        with self._lock:
            return [
                self._positions[key] for key in sorted(self._positions) if key[0] == step_ordinal
            ]


@dataclass(frozen=True)
class InterruptedActivity:
    """One durable activity exited by framework-native control flow."""

    command: ActivityCommand
    control_flow: BaseException


@dataclass
class _InterruptedActivities:
    """Attempt-local interrupted activities shared with worker task copies."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _items: list[InterruptedActivity] = field(default_factory=list)

    def record(self, command: ActivityCommand, control_flow: BaseException) -> None:
        command_copy = ActivityCommand()
        command_copy.CopyFrom(command)
        with self._lock:
            self._items.append(InterruptedActivity(command_copy, control_flow))

    def for_step(self, step_ordinal: int) -> list[InterruptedActivity]:
        with self._lock:
            return [
                item for item in self._items if item.command.position.step_ordinal == step_ordinal
            ]


@dataclass
class _ActivityReconstruction:
    """Attempt-local state for framework control-flow reconstruction."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _activity_ids: frozenset[str] = frozenset()
    _interrupt_counts: dict[str, int] = field(default_factory=dict)

    def replace(self, activity_ids: Sequence[str]) -> None:
        with self._lock:
            self._activity_ids = frozenset(activity_ids)

    def contains(self, activity_id: str) -> bool:
        with self._lock:
            return activity_id in self._activity_ids

    def record_interrupt(self, activity_id: str) -> int:
        with self._lock:
            if activity_id not in self._activity_ids:
                raise RuntimeError(f'activity "{activity_id}" is not being reconstructed')
            count = self._interrupt_counts.get(activity_id, 0) + 1
            self._interrupt_counts[activity_id] = count
            return count


@dataclass
class _ActivityOrdinalAllocator:
    """Request-local activity ordinals scoped by operation path."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _next_by_path: dict[_OperationPathKey, int] = field(default_factory=dict)
    _by_path_and_key: dict[tuple[_OperationPathKey, str], int] = field(default_factory=dict)

    def preallocate(self, path: _OperationPathKey, keys: Sequence[str]) -> list[int]:
        """Assign ordinals to keys in list order. Idempotent per key."""
        ordinals: list[int] = []
        with self._lock:
            for key in keys:
                if not key:
                    raise ValueError("activity ordinal key must be non-empty")
                path_key = (path, key)
                if path_key not in self._by_path_and_key:
                    ordinal = self._next_by_path.get(path, 1)
                    self._next_by_path[path] = ordinal + 1
                    self._by_path_and_key[path_key] = ordinal
                ordinals.append(self._by_path_and_key[path_key])
        return ordinals

    def allocate(self, path: _OperationPathKey, key: str | None = None) -> int:
        """Return the ordinal for `key`, or the next free ordinal when omitted."""
        with self._lock:
            if key is None:
                ordinal = self._next_by_path.get(path, 1)
                self._next_by_path[path] = ordinal + 1
                return ordinal
            if not key:
                raise ValueError("activity ordinal key must be non-empty")
            path_key = (path, key)
            if path_key not in self._by_path_and_key:
                ordinal = self._next_by_path.get(path, 1)
                self._next_by_path[path] = ordinal + 1
                self._by_path_and_key[path_key] = ordinal
            return self._by_path_and_key[path_key]


@dataclass
class _ChildOperationPathAllocator:
    """Assign stable ordinals to nested operation occurrences."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _next_ordinal_by_child: dict[_StructuralChild, int] = field(default_factory=dict)
    _ordinal_by_occurrence: dict[_ChildOccurrence, int] = field(default_factory=dict)
    _keys_by_step_child: dict[_StepChild, set[str]] = field(default_factory=dict)

    def preallocate(
        self,
        parent: _OperationPathKey,
        step_ordinal: int,
        boundaries: Sequence[ChildOperationBoundary],
    ) -> list[int]:
        """Assign ordinals to child boundaries in list order. Idempotent per key.

        The complete batch is validated before any assignment so a duplicate
        sibling identity cannot leave a partial reservation.
        """
        if not boundaries:
            raise ValueError("child operation name and occurrence key must be non-empty")
        ordinals: list[int] = []
        with self._lock:
            seen: set[tuple[str, str]] = set()
            for boundary in boundaries:
                if not boundary.name or not boundary.occurrence_key:
                    raise ValueError("child operation name and occurrence key must be non-empty")
                identity = (boundary.name, boundary.occurrence_key)
                if identity in seen:
                    raise UnsupportedChildOperationFanOutError(
                        f"child {boundary.name!r} has duplicate occurrence "
                        f"{boundary.occurrence_key!r} in the same batch"
                    )
                seen.add(identity)
            for boundary in boundaries:
                ordinals.append(self._assign(parent, step_ordinal, boundary, preallocating=True))
        return ordinals

    def resolve(
        self,
        boundaries: Sequence[ChildOperationBoundary],
        step_ordinal: int,
    ) -> OperationPath:
        """Turn ordered child boundaries into one hierarchical operation path.

        Repeated activities from the same occurrence reuse its ordinal. Re-entry
        at a later root step gets the next ordinal for that structural child.
        A second unpreallocated occurrence key for the same child within one
        root step is rejected; adapters must preallocate the complete batch
        before concurrent dispatch.
        """
        # Every operation path starts at the one durable root invocation.
        segments = [OperationPathSegment(name="agent", ordinal=1)]
        with self._lock:
            for boundary in boundaries:
                parent = tuple((segment.name, segment.ordinal) for segment in segments)
                ordinal = self._assign(parent, step_ordinal, boundary, preallocating=False)
                segments.append(OperationPathSegment(name=boundary.name, ordinal=ordinal))
        return OperationPath(segments=segments)

    def _assign(
        self,
        parent: _OperationPathKey,
        step_ordinal: int,
        boundary: ChildOperationBoundary,
        *,
        preallocating: bool,
    ) -> int:
        if not boundary.name or not boundary.occurrence_key:
            raise ValueError("child operation name and occurrence key must be non-empty")

        # The segments accumulated so far identify this boundary's parent.
        # Including their ordinals keeps equal child names under different
        # parent occurrences independent.
        child = _StructuralChild(parent, boundary.name)
        step_child = _StepChild(step_ordinal, child)
        occurrence = _ChildOccurrence(step_child, boundary.occurrence_key)
        ordinal = self._ordinal_by_occurrence.get(occurrence)
        if ordinal is None:
            seen_keys = self._keys_by_step_child.get(step_child)
            if seen_keys and not preallocating:
                raise UnsupportedChildOperationFanOutError(
                    f"child {boundary.name!r} has multiple occurrences in root step "
                    f"{step_ordinal}; deterministic preallocation is required"
                )
            ordinal = self._next_ordinal_by_child.get(child, 1)
            self._next_ordinal_by_child[child] = ordinal + 1
            self._ordinal_by_occurrence[occurrence] = ordinal
        self._keys_by_step_child.setdefault(step_child, set()).add(boundary.occurrence_key)
        return ordinal


@dataclass
class _PendingChildOperationBatch:
    """Mutable attempt-scoped sibling batch shared with Pregel worker copies."""

    boundaries: tuple[ChildOperationBoundary, ...] = ()


class AttemptContextBinding(NamedTuple):
    """Tokens from `set_attempt_context`; pass to `reset_attempt_context`."""

    attempt_token: Token[AttemptContext | None]
    step_token: Token[_StepCounter | None]
    activity_token: Token[_ActivityOrdinalAllocator | None]
    child_path_token: Token[_ChildOperationPathAllocator | None]
    pending_child_batch_token: Token[_PendingChildOperationBatch | None]
    observed_positions_token: Token[_ObservedActivityPositions | None]
    interrupted_activities_token: Token[_InterruptedActivities | None]
    activity_reconstruction_token: Token[_ActivityReconstruction | None]


_ATTEMPT_CONTEXT: ContextVar[AttemptContext | None] = ContextVar(
    "workflow_attempt_context", default=None
)
_STEP_ORDINAL: ContextVar[_StepCounter | None] = ContextVar("workflow_step_ordinal", default=None)
_ACTIVITY_ORDINALS: ContextVar[_ActivityOrdinalAllocator | None] = ContextVar(
    "workflow_activity_ordinals", default=None
)
_CHILD_OPERATION_PATHS: ContextVar[_ChildOperationPathAllocator | None] = ContextVar(
    "workflow_child_operation_paths", default=None
)
_PENDING_CHILD_BATCH: ContextVar[_PendingChildOperationBatch | None] = ContextVar(
    "workflow_pending_child_operation_batch", default=None
)
_OPERATION_PATH_RESOLVER: ContextVar[OperationPathResolver | None] = ContextVar(
    "workflow_operation_path_resolver", default=None
)
_OBSERVED_ACTIVITY_POSITIONS: ContextVar[_ObservedActivityPositions | None] = ContextVar(
    "workflow_observed_activity_positions", default=None
)
_INTERRUPTED_ACTIVITIES: ContextVar[_InterruptedActivities | None] = ContextVar(
    "workflow_interrupted_activities", default=None
)
_ACTIVITY_RECONSTRUCTION: ContextVar[_ActivityReconstruction | None] = ContextVar(
    "workflow_activity_reconstruction", default=None
)


def tool_activity_key(tool_call_id: str) -> str:
    """Stable allocator key for a Tool activity identified by ToolCall id."""
    if not tool_call_id:
        raise ValueError("tool_call_id must be non-empty")
    return f"tool:{tool_call_id}"


def current_attempt_context() -> AttemptContext | None:
    """Return the OE-issued attempt context for this invocation, if any."""
    return _ATTEMPT_CONTEXT.get()


def current_operation_path() -> OperationPath:
    """Resolve the path at activity admission, defaulting to the root agent.

    Calling the scoped resolver here lets a framework adapter inspect its current
    runtime context before the resulting path is copied into the activity command.
    """
    resolver = _OPERATION_PATH_RESOLVER.get()
    if current_attempt_context() is None or resolver is None:
        return OperationPath(segments=[OperationPathSegment(name="agent", ordinal=1)])

    allocator = _CHILD_OPERATION_PATHS.get()
    if allocator is None:
        raise RuntimeError("child operation paths require an active durable attempt")
    boundaries = resolver()
    return allocator.resolve(boundaries, current_step_ordinal())


@contextmanager
def operation_path_resolver_scope(resolver: OperationPathResolver) -> Iterator[None]:
    """Bind a framework adapter's path resolver for one graph invocation."""
    token = _OPERATION_PATH_RESOLVER.set(resolver)
    try:
        yield
    finally:
        _OPERATION_PATH_RESOLVER.reset(token)


@contextmanager
def child_operation_boundary_scope(
    boundary: ChildOperationBoundary,
) -> Iterator[None]:
    """Append one child boundary to the active framework operation path.

    Framework adapters use this around dynamic child execution whose boundary
    is not visible in the parent graph topology. If a parent resolver is active,
    its boundaries are captured before the child changes framework runtime context
    and remain the prefix; otherwise the supplied boundary starts the nested path
    below the fixed root agent segment.
    """
    parent_resolver = _OPERATION_PATH_RESOLVER.get()
    parent_boundaries = tuple(parent_resolver()) if parent_resolver is not None else ()

    def composed_resolver() -> tuple[ChildOperationBoundary, ...]:
        return (*parent_boundaries, boundary)

    token = _OPERATION_PATH_RESOLVER.set(composed_resolver)
    try:
        yield
    finally:
        _OPERATION_PATH_RESOLVER.reset(token)


def current_step_ordinal() -> int:
    """Return the active superstep ordinal for durable activity admission."""
    counter = _STEP_ORDINAL.get()
    if counter is None or counter.value <= 0:
        raise RuntimeError(
            "step_ordinal requires an active durable attempt context; "
            "bind AttemptContext before admitting activities or committing steps"
        )
    return counter.value


def record_observed_activity(position: ActivityPosition) -> None:
    """Record one activity position encountered by this attempt."""
    observed = _OBSERVED_ACTIVITY_POSITIONS.get()
    if observed is not None:
        observed.record(position)


def observed_activity_positions(step_ordinal: int) -> list[ActivityPosition]:
    """Return activity positions this attempt observed in one step."""
    observed = _OBSERVED_ACTIVITY_POSITIONS.get()
    return [] if observed is None else observed.for_step(step_ordinal)


def record_interrupted_activity(command: ActivityCommand, control_flow: BaseException) -> None:
    """Remember an activity that must be reconstructed by its adapter."""
    interrupted = _INTERRUPTED_ACTIVITIES.get()
    if interrupted is not None:
        interrupted.record(command, control_flow)


def interrupted_activities(step_ordinal: int) -> list[InterruptedActivity]:
    """Return framework-interrupted activities observed in one workflow step."""
    interrupted = _INTERRUPTED_ACTIVITIES.get()
    return [] if interrupted is None else interrupted.for_step(step_ordinal)


def set_activity_reconstruction_ids(activity_ids: Sequence[str]) -> None:
    """Select resume-map activities eligible for provenance-checked reconstruction."""
    reconstruction = _ACTIVITY_RECONSTRUCTION.get()
    if reconstruction is not None:
        reconstruction.replace(activity_ids)


def activity_requires_reconstruction(activity_id: str) -> bool:
    """Return whether replay must enter this activity instead of short-circuiting."""
    reconstruction = _ACTIVITY_RECONSTRUCTION.get()
    return reconstruction is not None and reconstruction.contains(activity_id)


def record_reconstructed_activity_interrupt(activity_id: str) -> int:
    """Count framework interrupts raised while reconstructing one activity."""
    reconstruction = _ACTIVITY_RECONSTRUCTION.get()
    if reconstruction is None:
        raise RuntimeError(f'activity "{activity_id}" is not being reconstructed')
    return reconstruction.record_interrupt(activity_id)


def advance_step_ordinal(committed_step_ordinal: int) -> int:
    """After a committed FinalizeStep, target the next superstep for activities."""
    if committed_step_ordinal <= 0:
        raise ValueError("committed step_ordinal must be positive")
    counter = _STEP_ORDINAL.get()
    if counter is None:
        raise RuntimeError(
            "step_ordinal requires an active durable attempt context; "
            "bind AttemptContext before admitting activities or committing steps"
        )
    counter.value = committed_step_ordinal + 1
    return counter.value


def _require_allocator() -> _ActivityOrdinalAllocator:
    allocator = _ACTIVITY_ORDINALS.get()
    if allocator is None:
        raise RuntimeError(
            "activity_ordinal requires an active durable attempt context; "
            "bind AttemptContext before admitting activities or committing steps"
        )
    return allocator


def _current_operation_path_key() -> _OperationPathKey:
    return tuple((segment.name, segment.ordinal) for segment in current_operation_path().segments)


def _require_child_path_allocator() -> _ChildOperationPathAllocator:
    allocator = _CHILD_OPERATION_PATHS.get()
    if allocator is None:
        raise RuntimeError(
            "child operation paths require an active durable attempt context; "
            "bind AttemptContext before admitting activities or committing steps"
        )
    return allocator


def preallocate_child_operation_ordinals(
    boundaries: Sequence[ChildOperationBoundary],
) -> list[int]:
    """Assign child path ordinals to boundaries in the given order.

    Call this before concurrent same-name child dispatch so inverted worker
    scheduling cannot redefine nested operation paths. Idempotent per
    occurrence key. A later unpreallocated sibling in the same root step is
    still rejected.

    Ordinals are keyed by the current root step. Observing a sibling batch on
    the producer superstep is not enough: stamp on the step those children
    will admit work, typically after the root FinalizeStep that advances the
    counter.
    """
    parent = tuple((segment.name, segment.ordinal) for segment in current_operation_path().segments)
    return _require_child_path_allocator().preallocate(parent, current_step_ordinal(), boundaries)


def preallocate_activity_ordinals(keys: Sequence[str]) -> list[int]:
    """Assign monotonic ordinals to stable keys in the given order.

    Call this before concurrent dispatch (for example after an LLM returns a
    ToolCall batch) so inverted worker scheduling cannot redefine identity.
    """
    return _require_allocator().preallocate(_current_operation_path_key(), keys)


def allocate_activity_ordinal(key: str | None = None) -> int:
    """Return a deterministic activity ordinal for `key`, or the next free one."""
    return _require_allocator().allocate(_current_operation_path_key(), key)


def current_pending_child_operation_batch() -> tuple[ChildOperationBoundary, ...]:
    """Return the attempt-scoped sibling batch, or empty when none is bound."""
    holder = _PENDING_CHILD_BATCH.get()
    if holder is None:
        return ()
    return holder.boundaries


def set_pending_child_operation_batch(
    boundaries: Sequence[ChildOperationBoundary],
) -> None:
    """Replace the attempt-scoped sibling batch in the shared mutable holder.

    The holder is created when the attempt is bound so Pregel worker copies
    see mutations from `after_model` without a process-global map.
    """
    holder = _PENDING_CHILD_BATCH.get()
    if holder is None:
        return
    holder.boundaries = tuple(boundaries)


def _initial_step_ordinal(context: AttemptContext) -> int:
    if not context.HasField("branch_lineage"):
        return 1
    lineage = context.branch_lineage
    if (
        not lineage.HasField("source_workflow_identity")
        or not lineage.source_workflow_identity.session_id
        or not lineage.source_workflow_identity.execution_id
        or lineage.source_step_ordinal <= 0
        or not lineage.source_state_hash
    ):
        raise ValueError("branch_lineage must identify a positive immutable source cutoff")
    return lineage.source_step_ordinal + 1


def _bind_attempt_holders(context: AttemptContext) -> AttemptContextBinding:
    # Validate the starting step before any ContextVar is bound so a malformed
    # branch_lineage cannot leak an attempt into later requests.
    initial_step_ordinal = _initial_step_ordinal(context)
    return AttemptContextBinding(
        attempt_token=_ATTEMPT_CONTEXT.set(context),
        step_token=_STEP_ORDINAL.set(_StepCounter(initial_step_ordinal)),
        activity_token=_ACTIVITY_ORDINALS.set(_ActivityOrdinalAllocator()),
        child_path_token=_CHILD_OPERATION_PATHS.set(_ChildOperationPathAllocator()),
        pending_child_batch_token=_PENDING_CHILD_BATCH.set(_PendingChildOperationBatch()),
        observed_positions_token=_OBSERVED_ACTIVITY_POSITIONS.set(_ObservedActivityPositions()),
        interrupted_activities_token=_INTERRUPTED_ACTIVITIES.set(_InterruptedActivities()),
        activity_reconstruction_token=_ACTIVITY_RECONSTRUCTION.set(_ActivityReconstruction()),
    )


@contextmanager
def attempt_context_scope(context: AttemptContext) -> Iterator[None]:
    """Bind one OE-issued attempt context for the duration of an invocation."""
    binding = _bind_attempt_holders(context)
    try:
        yield
    finally:
        reset_attempt_context(binding)


def set_attempt_context(context: AttemptContext) -> AttemptContextBinding:
    """Token-based binding for server lifecycles that cannot use one `with` block.

    Mirrors `agent_engine_runner_shared.context.set_execution_context`; pair every call
    with `reset_attempt_context` in the request's `finally`. Also starts the
    request-local superstep counter at the attempt's first writable step and a
    fresh activity-ordinal allocator.
    """
    return _bind_attempt_holders(context)


def reset_attempt_context(binding: AttemptContextBinding) -> None:
    """Unbind the attempt context set by `set_attempt_context`."""
    _ATTEMPT_CONTEXT.reset(binding.attempt_token)
    _STEP_ORDINAL.reset(binding.step_token)
    _ACTIVITY_ORDINALS.reset(binding.activity_token)
    _CHILD_OPERATION_PATHS.reset(binding.child_path_token)
    _PENDING_CHILD_BATCH.reset(binding.pending_child_batch_token)
    _OBSERVED_ACTIVITY_POSITIONS.reset(binding.observed_positions_token)
    _INTERRUPTED_ACTIVITIES.reset(binding.interrupted_activities_token)
    _ACTIVITY_RECONSTRUCTION.reset(binding.activity_reconstruction_token)
