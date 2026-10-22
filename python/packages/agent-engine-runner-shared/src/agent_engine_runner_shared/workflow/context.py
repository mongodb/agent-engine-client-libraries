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
    "AmbiguousActivityKeyError",
    "UnsupportedChildOperationFanOutError",
    "observed_activity_positions",
    "advance_step_ordinal",
    "allocate_activity_ordinal",
    "allocate_activity_position",
    "in_spawned_task",
    "activity_requires_reconstruction",
    "attempt_context_scope",
    "child_operation_boundary_scope",
    "current_attempt_context",
    "current_operation_path",
    "operation_path_scope",
    "current_pending_child_operation_batch",
    "current_step_ordinal",
    "interrupted_activities",
    "interrupted_activity_answered",
    "next_scoped_call_index",
    "operation_path_resolver_scope",
    "preallocate_activity_ordinals",
    "preallocate_child_operation_ordinals",
    "record_observed_activity",
    "record_reconstructed_activity_interrupt",
    "record_interrupted_activity",
    "reset_attempt_context",
    "set_attempt_context",
    "set_activity_reconstruction_ids",
    "note_activity_reconstruction_id",
    "set_pending_child_operation_batch",
    "tool_activity_key",
    "note_oauth_authorization_wait",
    "forget_oauth_authorization_wait",
    "is_oauth_authorization_wait",
]


@dataclass(frozen=True)
class ChildOperationBoundary:
    """Framework-neutral identity for one nested operation.

    ``spawned_task`` marks a task the framework started at runtime on behalf of
    its enclosing graph, such as a LangGraph ``Send`` task. A keyed activity
    inside one may take the position its key was preallocated at elsewhere in
    that enclosing graph; see ``allocate_activity_position``.
    """

    name: str
    occurrence_key: str
    spawned_task: bool = False


OperationPathResolver = Callable[[], Sequence[ChildOperationBoundary]]


class UnsupportedChildOperationFanOutError(RuntimeError):
    """One structural child produced ambiguous occurrences in the same step."""


class AmbiguousActivityKeyError(RuntimeError):
    """One key was preallocated at two positions a spawned task could take."""


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
class _ControlWaits:
    """Request-local IDs of waits backed by a recorded platform control outcome."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _ids: set[str] = field(default_factory=set)

    def note(self, control_id: str) -> None:
        """Retain a platform control ID for this one attempt."""
        with self._lock:
            self._ids.add(control_id)

    def has(self, control_id: str) -> bool:
        """Check request-local provenance rather than trusting a pause's shape."""
        with self._lock:
            return control_id in self._ids

    def forget(self, control_id: str) -> None:
        """Release one answered control wait without affecting a sibling's identity."""
        with self._lock:
            self._ids.discard(control_id)


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
    _answered: set[bytes] = field(default_factory=set)

    def record(self, command: ActivityCommand, control_flow: BaseException) -> None:
        command_copy = ActivityCommand()
        command_copy.CopyFrom(command)
        with self._lock:
            self._items.append(InterruptedActivity(command_copy, control_flow))

    def mark_answered(self, positions: Sequence[ActivityPosition]) -> None:
        with self._lock:
            self._answered.update(
                position.SerializeToString(deterministic=True) for position in positions
            )

    def is_answered(self, position: ActivityPosition) -> bool:
        """Return whether this attempt has reconstructed and answered the exact pause."""
        with self._lock:
            return position.SerializeToString(deterministic=True) in self._answered

    def for_step(
        self, step_ordinal: int, *, include_answered: bool = False
    ) -> list[InterruptedActivity]:
        with self._lock:
            return [
                item
                for item in self._items
                if item.command.position.step_ordinal == step_ordinal
                and (
                    include_answered
                    or item.command.position.SerializeToString(deterministic=True)
                    not in self._answered
                )
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

    def note(self, activity_id: str) -> None:
        """Include an OE-validated captured-local callback in this attempt's reconstruction set."""
        with self._lock:
            self._activity_ids = self._activity_ids | {activity_id}

    def record_interrupt(self, activity_id: str) -> int:
        with self._lock:
            if activity_id not in self._activity_ids:
                raise RuntimeError(f'activity "{activity_id}" is not being reconstructed')
            count = self._interrupt_counts.get(activity_id, 0) + 1
            self._interrupt_counts[activity_id] = count
            return count


def _within(path: _OperationPathKey, scope: _OperationPathKey) -> bool:
    return path[: len(scope)] == scope


@dataclass
class _ActivityOrdinalAllocator:
    """Request-local activity ordinals scoped by operation path."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _next_by_path: dict[_OperationPathKey, int] = field(default_factory=dict)
    _by_path_and_key: dict[tuple[_OperationPathKey, str], int] = field(default_factory=dict)
    _preallocated_paths_by_key: dict[str, list[_OperationPathKey]] = field(default_factory=dict)
    # Key -> {spawning scope: owner} for keys a spawned task took, so a later
    # preallocation of the same key in that scope fails instead of having lost
    # a race it would win on another attempt. Kept per scope so a take in one
    # graph cannot hide a take in another.
    _taken_by_spawned_task: dict[str, dict[_OperationPathKey, _OperationPathKey]] = field(
        default_factory=dict
    )
    # Keyed by id(); the scope object is retained so its id cannot be reused
    # by another scope while this attempt lives.
    _calls_by_scope: dict[int, tuple[object, int]] = field(default_factory=dict)

    def next_in_scope(self, scope: object) -> int:
        """Count calls within one scope object, starting at 1."""
        with self._lock:
            _, count = self._calls_by_scope.get(id(scope), (scope, 0))
            self._calls_by_scope[id(scope)] = (scope, count + 1)
            return count + 1

    def preallocate(self, path: _OperationPathKey, keys: Sequence[str]) -> list[int]:
        """Assign ordinals to keys in list order. Idempotent per key."""
        ordinals: list[int] = []
        with self._lock:
            for key in keys:
                if not key:
                    raise ValueError("activity ordinal key must be non-empty")
                takes = self._taken_by_spawned_task.get(key, {})
                if any(path != owner and _within(path, scope) for scope, owner in takes.items()):
                    raise AmbiguousActivityKeyError(
                        f"activity key {key!r} was preallocated at two positions in one "
                        "graph; tool call ids must be unique within a turn"
                    )
                path_key = (path, key)
                if path_key not in self._by_path_and_key:
                    ordinal = self._next_by_path.get(path, 1)
                    self._next_by_path[path] = ordinal + 1
                    self._by_path_and_key[path_key] = ordinal
                    self._preallocated_paths_by_key.setdefault(key, []).append(path)
                ordinals.append(self._by_path_and_key[path_key])
        return ordinals

    def allocate_position(
        self,
        path: _OperationPathKey,
        key: str | None,
        spawning_scope: _OperationPathKey | None,
    ) -> tuple[_OperationPathKey, int]:
        """Return the path and ordinal for `key`, preferring where it was preallocated.

        Only an activity inside a spawned task (``spawning_scope`` is the path of
        the graph that spawned it) may move: a tool a model requested and that
        then runs in its own LangGraph Send task keeps the model's position. The
        key must have been preallocated exactly once within that scope, even
        when the tool runs in the task that preallocated it. Any other activity
        is allocated in ``path``, so paths that contain no spawned task record
        exactly as they did before spawned tasks existed.
        """
        with self._lock:
            if key and spawning_scope is not None:
                owners = [
                    candidate
                    for candidate in self._preallocated_paths_by_key.get(key, [])
                    if _within(candidate, spawning_scope)
                ]
                if len(owners) > 1:
                    raise AmbiguousActivityKeyError(
                        f"activity key {key!r} was preallocated at {len(owners)} positions "
                        "in one graph; tool call ids must be unique within a turn"
                    )
                if owners:
                    self._taken_by_spawned_task.setdefault(key, {})[spawning_scope] = owners[0]
                    return owners[0], self._by_path_and_key[(owners[0], key)]
        return path, self.allocate(path, key)

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
    _spawned_task_paths: set[_OperationPathKey] = field(default_factory=set)

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
                if boundary.spawned_task:
                    self._spawned_task_paths.add((*parent, (boundary.name, ordinal)))
        return OperationPath(segments=segments)

    def spawning_scope(self, path: _OperationPathKey) -> _OperationPathKey | None:
        """Return the path of the graph that spawned the innermost task in `path`."""
        with self._lock:
            for end in range(len(path), 0, -1):
                if path[:end] in self._spawned_task_paths:
                    return path[: end - 1]
        return None

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
    guardrail_review_waits_token: Token[_ControlWaits | None]
    oauth_authorization_waits_token: Token[_ControlWaits | None]


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
_BOUND_OPERATION_PATH: ContextVar[OperationPath | None] = ContextVar(
    "workflow_bound_operation_path", default=None
)
_OPERATION_PATH_RESOLVER: ContextVar[OperationPathResolver | None] = ContextVar(
    "workflow_operation_path_resolver", default=None
)
_OBSERVED_ACTIVITY_POSITIONS: ContextVar[_ObservedActivityPositions | None] = ContextVar(
    "workflow_observed_activity_positions", default=None
)
_GUARDRAIL_REVIEW_WAITS: ContextVar[_ControlWaits | None] = ContextVar(
    "workflow_guardrail_review_waits", default=None
)
_OAUTH_AUTHORIZATION_WAITS: ContextVar[_ControlWaits | None] = ContextVar(
    "workflow_oauth_authorization_waits", default=None
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
    bound = _BOUND_OPERATION_PATH.get()
    if bound is not None:
        path = OperationPath()
        path.CopyFrom(bound)
        return path
    resolver = _OPERATION_PATH_RESOLVER.get()
    if current_attempt_context() is None or resolver is None:
        return OperationPath(segments=[OperationPathSegment(name="agent", ordinal=1)])

    allocator = _CHILD_OPERATION_PATHS.get()
    if allocator is None:
        raise RuntimeError("child operation paths require an active durable attempt")
    boundaries = resolver()
    return allocator.resolve(boundaries, current_step_ordinal())


@contextmanager
def operation_path_scope(path: OperationPath) -> Iterator[None]:
    """Preserve an OAuth call's graph path while a framework task isolates its pause.

    Reads are copied; the task-local view is restored on every exit.
    """
    token = _BOUND_OPERATION_PATH.set(path)
    try:
        yield
    finally:
        _BOUND_OPERATION_PATH.reset(token)


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


def note_guardrail_review_wait(review_id: str) -> None:
    """Record that this attempt pauses for a review the platform halted a call for."""
    waits = _GUARDRAIL_REVIEW_WAITS.get()
    if waits is not None:
        waits.note(review_id)


def is_guardrail_review_wait(review_id: object) -> bool:
    """Whether this attempt paused for that review.

    A pause is a review wait because the platform recorded the halt behind it,
    not because of what its value looks like: an application may pause with a
    value of any shape.
    """
    waits = _GUARDRAIL_REVIEW_WAITS.get()
    return waits is not None and isinstance(review_id, str) and waits.has(review_id)


def note_oauth_authorization_wait(activity_id: str) -> None:
    """Register the recorded challenge activity, independent of its coalesced Broker flow."""
    waits = _OAUTH_AUTHORIZATION_WAITS.get()
    if waits is not None:
        waits.note(activity_id)


def is_oauth_authorization_wait(activity_id: object) -> bool:
    """Recognize the opaque activity identity registered at this attempt's consent boundary."""
    waits = _OAUTH_AUTHORIZATION_WAITS.get()
    return waits is not None and isinstance(activity_id, str) and waits.has(activity_id)


def forget_oauth_authorization_wait(activity_id: str) -> None:
    """Release consent provenance after its suspend handler returns an answer."""
    waits = _OAUTH_AUTHORIZATION_WAITS.get()
    if waits is not None:
        waits.forget(activity_id)


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


def interrupted_activities(
    step_ordinal: int, *, include_answered: bool = False
) -> list[InterruptedActivity]:
    """Return framework-interrupted activities observed in one workflow step.

    Answered activities are excluded by default. One LangGraph task can pause
    several times and every pause carries the same native interrupt id, so
    only the pauses still waiting may claim that id.
    """
    interrupted = _INTERRUPTED_ACTIVITIES.get()
    if interrupted is None:
        return []
    return interrupted.for_step(step_ordinal, include_answered=include_answered)


def mark_interrupted_activities_answered(positions: Sequence[ActivityPosition]) -> None:
    """Record that these interrupted activities received their answer in this attempt."""
    interrupted = _INTERRUPTED_ACTIVITIES.get()
    if interrupted is not None:
        interrupted.mark_answered(positions)


def interrupted_activity_answered(position: ActivityPosition) -> bool:
    """Return whether the adapter settled this pause in the current attempt."""
    interrupted = _INTERRUPTED_ACTIVITIES.get()
    return interrupted is not None and interrupted.is_answered(position)


def set_activity_reconstruction_ids(activity_ids: Sequence[str]) -> None:
    """Select resume-map activities eligible for provenance-checked reconstruction."""
    reconstruction = _ACTIVITY_RECONSTRUCTION.get()
    if reconstruction is not None:
        reconstruction.replace(activity_ids)


def activity_requires_reconstruction(activity_id: str) -> bool:
    """Return whether replay must enter this activity instead of short-circuiting."""
    reconstruction = _ACTIVITY_RECONSTRUCTION.get()
    return reconstruction is not None and reconstruction.contains(activity_id)


def note_activity_reconstruction_id(activity_id: str) -> None:
    """Select a captured-local callback after OE validates its owning activity."""
    reconstruction = _ACTIVITY_RECONSTRUCTION.get()
    if reconstruction is not None:
        reconstruction.note(activity_id)


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


def allocate_activity_position(key: str | None = None) -> tuple[OperationPath, int]:
    """Return the operation path and ordinal for a possibly preallocated `key`.

    Use this where the activity may run in a spawned task other than the one
    that preallocated its key; pass the returned path to the activity command.
    """
    current = _current_operation_path_key()
    path, ordinal = _require_allocator().allocate_position(current, key, _spawning_scope(current))
    return (
        OperationPath(
            segments=[OperationPathSegment(name=name, ordinal=index) for name, index in path]
        ),
        ordinal,
    )


def in_spawned_task() -> bool:
    """Whether the current operation path runs inside a spawned task."""
    return _spawning_scope(_current_operation_path_key()) is not None


def _spawning_scope(path: _OperationPathKey) -> _OperationPathKey | None:
    child_paths = _CHILD_OPERATION_PATHS.get()
    return child_paths.spawning_scope(path) if child_paths is not None else None


def next_scoped_call_index(scope: object) -> int:
    """Return the 1-based index of this call within `scope` for the current attempt.

    An adapter passes the framework object that lives for one run of a unit
    the framework may re-run, such as one run of a graph task. The index
    restarts for each run, so a key built from it names the same call on every
    run and its recorded activity replays.
    """
    return _require_allocator().next_in_scope(scope)


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
    """Bind fresh attempt-local holders and return tokens that restore the caller's context."""
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
        guardrail_review_waits_token=_GUARDRAIL_REVIEW_WAITS.set(_ControlWaits()),
        oauth_authorization_waits_token=_OAUTH_AUTHORIZATION_WAITS.set(_ControlWaits()),
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
    _GUARDRAIL_REVIEW_WAITS.reset(binding.guardrail_review_waits_token)
    _OAUTH_AUTHORIZATION_WAITS.reset(binding.oauth_authorization_waits_token)
