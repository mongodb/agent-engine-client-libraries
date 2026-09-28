"""
Execution-scoped drain receiver for Runner SDK servers.

Mirrors TypeScript's ``agent-engine-runner-shared/src/server/drain.ts``. Both runtimes
are parallel implementations of the same server architecture, so this file
and its TypeScript twin must be changed together.

When the OE durably cancels an execution it POSTs ``/drain`` to the Tool/AER
runtime owning that execution. The runtime blocks new work for the execution,
cancels tracked in-flight tasks where the asyncio model allows it, and records
an idempotent final outcome the caller retrieves by re-sending the same
request.

Contract (pinned by the shared fixture in
``client-libraries/test-fixtures/drain/contract.json``):

- ``POST /drain {request_id, execution_id, reason, deadline_at_ms, workspace_id}``
  — ``workspace_id`` is mandatory on workspace-scoped runtimes (``APP_ID``
  set) and must match it exactly.
- ``202 {"outcome": "accepted"}`` while draining; ``200`` with the final
  outcome (``completed`` / ``timed_out`` / ``delivery_failed``) afterwards.
- ``deadline_at_ms`` is absolute. The runtime never extends it.
- ``accepted`` is HTTP-level acceptance only; ``completed`` is the only
  quiescence signal.
- An execution this runtime never served is ``delivery_failed`` with
  ``reason_code=execution_not_found`` — exact OE targeting does not prove a
  false ``completed`` safe, since one execution can span AER and Tool
  runtimes independently.

State is process-local by design: a restarted runtime has lost its active
work and answers ``delivery_failed``; reconciliation across restarts belongs
to the caller's durable record, not to this registry.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional, Set

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "CallInterruptOutcome",
    "DrainOutcome",
    "DrainReason",
    "DrainReasonCode",
    "DrainRequest",
    "DrainResponse",
    "DrainRegistry",
    "register_drain_route",
]

# How long a finalized drain record (and never-drained tombstone) survives for
# idempotent retries. Must be >= the caller-side retry/reconciliation window.
DEFAULT_RECORD_TTL_S = 900.0
# Upper bound on the caller-chosen drain window; the runtime never extends a
# deadline, and rejects ones set unreasonably far out.
DEFAULT_MAX_DEADLINE_MS = 60_000
# Grace for work registered before its task exists (an SSE stream's consuming
# task starts only when the response body iterates). Without it, a client
# disconnecting before iteration starts would leak the registration forever —
# and a later drain would wait out its deadline against phantom work.
DEFAULT_ATTACH_GRACE_S = 30.0
# How long an abort that preceded its call's registration stays claimable by
# begin_work. The window covers the OE-claim → in-graph-dispatch handoff; the
# platform's dispatch claim makes a same-step collision inside it impossible.
PRE_ABORT_RETENTION_S = 60.0

_REQUEST_ID_PATTERN = r"^[A-Za-z0-9._:-]+$"


class DrainReason(str, Enum):
    EXECUTION_CANCELLED = "execution_cancelled"
    # Forward-compatible value for controlled replacement; not validated end
    # to end by this change.
    ROLLING_RESTART = "rolling_restart"


class DrainOutcome(str, Enum):
    ACCEPTED = "accepted"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    DELIVERY_FAILED = "delivery_failed"


class DrainReasonCode(str, Enum):
    EXECUTION_NOT_FOUND = "execution_not_found"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    EXECUTION_DRAINING = "execution_draining"


class CallInterruptOutcome(str, Enum):
    """Outcome of one per-call abort (POST /interrupt/call). Unlike a drain,
    an interrupt is surgical: the execution is never admission-latched and
    later calls of the same execution proceed."""

    INTERRUPTED = "interrupted"
    NOT_FOUND = "not_found"
    ALREADY_SETTLED = "already_settled"
    # The call is tracked but has no signal channel (a sync tool on a worker
    # thread): reported honestly rather than claiming an abandonment the
    # runtime cannot perform.
    NOT_CANCELLABLE = "not_cancellable"


class DrainRequest(BaseModel):
    request_id: str = Field(..., min_length=1, max_length=128, pattern=_REQUEST_ID_PATTERN)
    execution_id: str = Field(..., min_length=1, max_length=256)
    reason: DrainReason
    deadline_at_ms: int = Field(..., gt=0)
    workspace_id: Optional[str] = Field(default=None, max_length=256)


class DrainResponse(BaseModel):
    outcome: DrainOutcome
    reason_code: Optional[DrainReasonCode] = None


class _DrainRecord:
    """One drain per execution. Later request_ids coalesce into it; they never
    start a second finalizer or re-run notification."""

    def __init__(self, request_id: str, reason: DrainReason, deadline_at_ms: int) -> None:
        self.request_ids: Set[str] = {request_id}
        self.reason = reason
        self.deadline_at_ms = deadline_at_ms
        self.outcome: Optional[DrainOutcome] = None  # None = still draining
        self.reason_code: Optional[DrainReasonCode] = None
        self.finalized_at_ms: Optional[int] = None


class _ExecutionEntry:
    """Registry row for one execution_id.

    Lifecycle: created when work starts (``active_count`` > 0) -> tombstone
    when the count returns to zero -> evicted after the record TTL. The
    ``draining`` latch is one-way: a drained execution never accepts work
    again for the entry's lifetime.
    """

    def __init__(self) -> None:
        self.active_count = 0
        self.tasks: Set[asyncio.Task] = set()
        self.draining = False
        self.drain: Optional[_DrainRecord] = None
        self.idle = asyncio.Event()
        self.eviction: Optional[asyncio.TimerHandle] = None
        # Live or settled work by step number, for the per-call abort. Done
        # handles stay mapped so a late abort reads already_settled; the
        # entry's TTL eviction reclaims them.
        self.by_step: Dict[int, "WorkHandle"] = {}


class WorkHandle:
    """One registered unit of in-flight work.

    Ending is idempotent — a second ``end_work`` is a no-op — so overlapping
    cleanup paths (stream generator, response background, shutdown) cannot
    corrupt the active count. Created by ``DrainRegistry.begin_work`` only.
    """

    __slots__ = (
        "entry",
        "execution_id",
        "task",
        "attached",
        "attach_timer",
        "done",
        "settling",
        "step_number",
        "aborted",
        "cancel_callback",
        "channel_expected",
    )

    def __init__(self, entry: _ExecutionEntry, execution_id: str) -> None:
        self.entry = entry
        self.execution_id = execution_id
        self.task: Optional[asyncio.Task] = None
        self.attached = False
        self.attach_timer: Optional[asyncio.TimerHandle] = None
        self.done = False
        # Set when the call's result report begins: the body already produced
        # its outcome, so a late abort reads already_settled instead of
        # claiming a stop the durable record will contradict.
        self.settling = False
        self.step_number: Optional[int] = None
        # Set by abort_call only when the abort is honored: an abort that
        # lands before the signal channel attaches fires at attach time. A
        # refused abort (not_cancellable) never sets it, so the call's real
        # outcome still flows.
        self.aborted = False
        # Abort channel for work whose cancellable unit lives off the server's
        # event loop (a tool body on a worker thread's private loop). Invoked
        # by abort_call; must be thread-safe. None means no signal channel.
        self.cancel_callback: Optional[Callable[[], None]] = None
        # The caller declares a cancel_callback will attach shortly (an async
        # tool body whose loop starts on its worker thread). A pre-attach
        # abort is then sticky-interrupted instead of not_cancellable.
        self.channel_expected = False


class DrainRegistry:
    """Execution-scoped admission + active-work registry for one runtime process.

    All mutating methods are synchronous and never await between check and
    set, so check-and-set is atomic under the single event loop each server
    runs on.
    """

    def __init__(
        self,
        *,
        record_ttl_s: Optional[float] = None,
        max_deadline_ms: Optional[int] = None,
        attach_grace_s: Optional[float] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        source = os.environ if env is None else env
        self._record_ttl_s = _positive(
            record_ttl_s
            if record_ttl_s is not None
            else float(source.get("RUNNER_DRAIN_RECORD_TTL_S", "") or DEFAULT_RECORD_TTL_S),
            "RUNNER_DRAIN_RECORD_TTL_S",
        )
        self._max_deadline_ms = int(
            _positive(
                max_deadline_ms
                if max_deadline_ms is not None
                else int(source.get("RUNNER_DRAIN_MAX_DEADLINE_MS", "") or DEFAULT_MAX_DEADLINE_MS),
                "RUNNER_DRAIN_MAX_DEADLINE_MS",
            )
        )
        self._attach_grace_s = _positive(
            attach_grace_s
            if attach_grace_s is not None
            else float(source.get("RUNNER_DRAIN_ATTACH_GRACE_S", "") or DEFAULT_ATTACH_GRACE_S),
            "RUNNER_DRAIN_ATTACH_GRACE_S",
        )
        self._entries: Dict[str, _ExecutionEntry] = {}
        # Aborts that arrived before their call registered, by
        # (execution_id, step_number), with a monotonic deadline. The window
        # is the OE-claim → in-graph-dispatch handoff; the retention covers
        # graph scheduling delays, not call lifetimes.
        self._pre_aborted: Dict[tuple, float] = {}
        # The loop that owns entry state. Captured on the first on-loop
        # registration; worker-thread callers (a tool body's boundary) must be
        # marshalled onto it via owner_loop_call — the check-and-set invariants
        # and asyncio.Event/timer wakeups are not safe to drive off-loop.
        self._owner_loop: Optional[asyncio.AbstractEventLoop] = None

    def _pop_pre_aborted(self, execution_id: str, step_number: int) -> bool:
        deadline = self._pre_aborted.pop((execution_id, step_number), None)
        self._sweep_pre_aborted()
        return deadline is not None and deadline > _now_monotonic()

    def _sweep_pre_aborted(self) -> None:
        if not self._pre_aborted:
            return
        now = _now_monotonic()
        for key, deadline in list(self._pre_aborted.items()):
            if deadline <= now:
                del self._pre_aborted[key]

    @property
    def max_deadline_ms(self) -> int:
        return self._max_deadline_ms

    def check_admission(self, execution_id: str) -> None:
        """Reject work for a draining execution without registering it."""
        entry = self._entries.get(execution_id)
        if entry is not None and entry.draining:
            raise HTTPException(
                status_code=409,
                detail=DrainReasonCode.EXECUTION_DRAINING.value,
            )

    def _capture_owner_loop(self) -> None:
        if self._owner_loop is not None:
            return
        try:
            self._owner_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

    def owner_loop_call(self, fn: Callable[[], Any]) -> Any:
        """Run fn on the registry's owner loop, blocking the caller until done.

        The registry's check-and-set invariants and its asyncio.Event/timer
        wakeups are single-loop state; tool bodies run on framework worker
        threads, so their begin/attach/end calls marshal through here. On the
        owner loop itself — or before any loop is known (single-threaded test
        harnesses) — fn runs inline.
        """
        loop = self._owner_loop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if loop is None or running is loop:
            return fn()
        box: Dict[str, Any] = {}
        done = threading.Event()

        def run() -> None:
            try:
                box["result"] = fn()
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                box["error"] = exc
            finally:
                done.set()

        try:
            loop.call_soon_threadsafe(run)
        except RuntimeError:
            # Owner loop already closed (server shutdown): run inline rather
            # than drop the bookkeeping.
            return fn()
        done.wait()
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def abort_call(self, execution_id: str, step_number: int) -> CallInterruptOutcome:
        """Signal exactly one in-flight call, leaving the execution open.

        The surgical sibling of a drain: no admission latch, no record, no
        deadline — the named call's task is cancelled (or marked to cancel at
        attach), and later steps of the same execution proceed. Idempotent: a
        repeat abort of the same step re-reads the same outcome.
        """
        entry = self._entries.get(execution_id)
        handle = entry.by_step.get(step_number) if entry is not None else None
        if handle is None:
            # The abort can race the call's registration: OE claims the call,
            # then the runtime registers only when the in-graph body starts.
            # Retain it briefly so begin_work settles the late registration
            # as interrupted instead of running a call the caller stopped.
            self._sweep_pre_aborted()
            self._pre_aborted[(execution_id, step_number)] = (
                _now_monotonic() + PRE_ABORT_RETENTION_S
            )
            return CallInterruptOutcome.NOT_FOUND
        if handle.done or handle.settling:
            return CallInterruptOutcome.ALREADY_SETTLED
        if handle.cancel_callback is not None:
            handle.aborted = True
            handle.cancel_callback()
            return CallInterruptOutcome.INTERRUPTED
        if handle.task is not None:
            handle.aborted = True
            handle.task.cancel()
            return CallInterruptOutcome.INTERRUPTED
        if handle.channel_expected or handle.attach_timer is not None:
            # The signal channel attaches when the body's cancellable unit
            # exists (or a stream consumer attaches): the abort fires then.
            handle.aborted = True
            return CallInterruptOutcome.INTERRUPTED
        # Tracked work with no signal channel (a sync tool on a worker
        # thread): the runtime can only wait it out. The refusal leaves
        # handle.aborted alone so the body's real outcome still flows.
        return CallInterruptOutcome.NOT_CANCELLABLE

    def claim_settlement(self, handle: "Optional[WorkHandle]") -> bool:
        """Atomically close a call's interruptibility as its result report begins.

        Marshalled onto the owner loop like every registry mutation, so it
        serializes against abort_call: whichever claims first wins the call's
        outcome. Returns False when an honored abort got there first — the
        caller must report the stopped outcome, not the body's.
        """
        if handle is None:
            return True
        handle.settling = True
        return not handle.aborted

    def begin_work(
        self,
        execution_id: str,
        task: "Optional[asyncio.Task]" = None,
        *,
        expect_attach: bool = False,
        step_number: "Optional[int]" = None,
        channel_expected: bool = False,
    ) -> WorkHandle:
        """Register in-flight work for an execution, rejecting drained ones.

        ``task`` is the cancellable handle the drain will cancel; pass None
        for work the runtime can only wait out (e.g. a sync tool off-loaded
        to a thread). With ``expect_attach`` the task is attached later via
        ``attach_task`` (an SSE stream's consuming task exists only once the
        response body iterates); if nothing attaches within the attach grace
        the registration releases itself, so a client that vanishes before
        iteration cannot leak phantom work.

        ``step_number`` makes the work addressable by the per-call abort
        (``abort_call``); work registered without it is only ever drained
        execution-wide. ``channel_expected`` declares that a cancel channel
        attaches shortly (an async tool body starting its loop), so a
        pre-attach abort is sticky-interrupted rather than not_cancellable.
        """
        self._capture_owner_loop()
        entry = self._entries.get(execution_id)
        if entry is None:
            entry = _ExecutionEntry()
            self._entries[execution_id] = entry
        else:
            if entry.draining:
                raise HTTPException(
                    status_code=409,
                    detail=DrainReasonCode.EXECUTION_DRAINING.value,
                )
            self._cancel_eviction(entry)
            if entry.active_count == 0:
                # Reviving a tombstone (e.g. a HITL resume reusing the id).
                entry.idle = asyncio.Event()
        handle = WorkHandle(entry, execution_id)
        entry.active_count += 1
        if task is not None:
            handle.task = task
            handle.attached = True
            entry.tasks.add(task)
        elif expect_attach:
            handle.attach_timer = asyncio.get_running_loop().call_later(
                self._attach_grace_s, self._release_unattached, handle
            )
        handle.step_number = step_number
        handle.channel_expected = channel_expected
        if step_number is not None:
            if self._pop_pre_aborted(execution_id, step_number):
                # The abort preceded the registration: the caller already
                # stopped this call, so the boundary must not run the body.
                handle.aborted = True
            # A step maps to one in-flight call by the platform's dispatch
            # claim, so this can only replace a done handle.
            entry.by_step[step_number] = handle
        return handle

    def attach_task(self, handle: WorkHandle, task: "Optional[asyncio.Task]") -> bool:
        """Attach the consuming task to handle-registered work.

        A drain that landed between admission and attach cancels the task
        here, at attach time. Returns False when the handle was already
        released (the attach grace expired or the work ended): the caller
        must not start the work, since a drain may already have reported
        quiescence for this execution.
        """
        if handle.done:
            return False
        if handle.attached or task is None:
            return True
        handle.attached = True
        handle.task = task
        if handle.attach_timer is not None:
            handle.attach_timer.cancel()
            handle.attach_timer = None
        handle.entry.tasks.add(task)
        if handle.entry.draining or handle.aborted:
            task.cancel()
        return True

    def attach_cancel_callback(
        self, handle: WorkHandle, callback: "Optional[Callable[[], None]]"
    ) -> bool:
        """Attach the abort signal channel for handle-registered work.

        ``callback`` must be safe to invoke from the server's event-loop thread
        (e.g. it hands the cancellation to a worker thread's loop). Passing
        None declares the work has no signal channel — a sync tool body on a
        worker thread — closing the attach window so a per-call abort answers
        not_cancellable instead of waiting out the grace. An abort that landed
        before the attach fires the callback here. Returns False when the
        handle already ended: the caller must not start the work.
        """
        if handle.done:
            return False
        if handle.attached:
            return True
        handle.attached = True
        handle.cancel_callback = callback
        if handle.attach_timer is not None:
            handle.attach_timer.cancel()
            handle.attach_timer = None
        if (handle.aborted or handle.entry.draining) and callback is not None:
            callback()
        return True

    def end_work(self, handle: "Optional[WorkHandle]") -> None:
        if handle is None or handle.done:
            return
        handle.done = True
        if handle.attach_timer is not None:
            handle.attach_timer.cancel()
            handle.attach_timer = None
        entry = handle.entry
        if handle.task is not None:
            entry.tasks.discard(handle.task)
        entry.active_count -= 1
        if entry.active_count > 0:
            return
        # No work remains: drop every tracked task (drained or not) so a
        # tombstone never retains tasks for its TTL.
        entry.tasks.clear()
        entry.idle.set()
        # A pending drain's finalizer owns eviction from here; otherwise the
        # tombstone (and any finalized drain record) expires on the TTL.
        if entry.drain is None or entry.drain.outcome is not None:
            self._schedule_eviction(handle.execution_id, entry)

    def _release_unattached(self, handle: WorkHandle) -> None:
        if handle.done or handle.attached:
            return
        logger.warning(
            "Releasing stream registration for execution %s: no consumer attached "
            "within the attach grace (client likely disconnected before iteration)",
            handle.execution_id,
        )
        self.end_work(handle)

    def apply(self, request: DrainRequest) -> JSONResponse:
        """Apply one drain request; idempotent by request_id, coalescing per execution."""
        self._capture_owner_loop()
        entry = self._entries.get(request.execution_id)
        if entry is not None and entry.drain is not None:
            # Same request_id or a different one: one drain per execution.
            entry.drain.request_ids.add(request.request_id)
            return self._current(entry.drain)
        if entry is None:
            # Unknown execution: not applied here, but still record the outcome
            # and latch admission. A dispatch that slips past the OE gate and
            # lands after this drain must not start, and a retry of the same
            # request must see the same answer for the retention window.
            entry = _ExecutionEntry()
            entry.draining = True
            record = _DrainRecord(request.request_id, request.reason, request.deadline_at_ms)
            record.outcome = DrainOutcome.DELIVERY_FAILED
            record.reason_code = DrainReasonCode.EXECUTION_NOT_FOUND
            record.finalized_at_ms = _now_ms()
            entry.drain = record
            self._entries[request.execution_id] = entry
            self._schedule_eviction(request.execution_id, entry)
            return self._final(DrainOutcome.DELIVERY_FAILED, DrainReasonCode.EXECUTION_NOT_FOUND)
        if entry.active_count == 0:
            # Known, already-ended execution: nothing to drain here. Latch
            # admission anyway — a drained execution never accepts work again —
            # and restart retention so the record lives a full TTL from now.
            entry.draining = True
            record = _DrainRecord(request.request_id, request.reason, request.deadline_at_ms)
            record.outcome = DrainOutcome.COMPLETED
            record.finalized_at_ms = _now_ms()
            entry.drain = record
            self._schedule_eviction(request.execution_id, entry)
            return self._final(DrainOutcome.COMPLETED)
        entry.draining = True
        entry.drain = _DrainRecord(request.request_id, request.reason, request.deadline_at_ms)
        for task in entry.tasks:
            task.cancel()
        for step_handle in entry.by_step.values():
            # Work tracked with an off-loop cancel channel (an in-graph tool
            # body on a worker thread) is not in entry.tasks: fire its channel
            # too, or the drain waits out the body for no reason. Not marked
            # aborted — that bit belongs to the per-call abort alone, so the
            # boundary can still tell a drain apart from a stop-this-call.
            if not step_handle.done and step_handle.cancel_callback is not None:
                step_handle.cancel_callback()
        asyncio.create_task(self._finalize(request.execution_id, entry, entry.drain))
        logger.info(
            "Drain accepted for execution %s (reason=%s, active=%d)",
            request.execution_id,
            request.reason.value,
            entry.active_count,
        )
        return JSONResponse(
            status_code=202,
            content=DrainResponse(outcome=DrainOutcome.ACCEPTED).model_dump(exclude_none=True),
        )

    async def _finalize(
        self, execution_id: str, entry: _ExecutionEntry, record: _DrainRecord
    ) -> None:
        remaining_s = record.deadline_at_ms / 1000 - time.time()
        try:
            await asyncio.wait_for(entry.idle.wait(), timeout=max(0.0, remaining_s))
            record.outcome = DrainOutcome.COMPLETED
        except (asyncio.TimeoutError, TimeoutError):
            # Late completion must not overwrite the recorded timeout.
            record.outcome = DrainOutcome.TIMED_OUT
            record.reason_code = DrainReasonCode.DEADLINE_EXCEEDED
        record.finalized_at_ms = _now_ms()
        logger.info(
            "Drain %s for execution %s",
            record.outcome.value,
            execution_id,
        )
        if entry.active_count == 0:
            self._schedule_eviction(execution_id, entry)

    def _schedule_eviction(self, execution_id: str, entry: _ExecutionEntry) -> None:
        self._cancel_eviction(entry)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Off-loop callers (test harness setup) skip eviction; it is
            # memory hygiene, not correctness.
            return
        entry.eviction = loop.call_later(self._record_ttl_s, self._evict, execution_id, entry)

    def _cancel_eviction(self, entry: _ExecutionEntry) -> None:
        if entry.eviction is not None:
            entry.eviction.cancel()
            entry.eviction = None

    def _evict(self, execution_id: str, entry: _ExecutionEntry) -> None:
        # A retried drain after eviction finds no entry and gets
        # execution_not_found; it never re-runs side effects.
        if self._entries.get(execution_id) is entry and entry.active_count == 0:
            self._entries.pop(execution_id, None)

    @staticmethod
    def _current(record: _DrainRecord) -> JSONResponse:
        if record.outcome is None:
            return JSONResponse(
                status_code=202,
                content=DrainResponse(outcome=DrainOutcome.ACCEPTED).model_dump(exclude_none=True),
            )
        return DrainRegistry._final(record.outcome, record.reason_code)

    @staticmethod
    def _final(
        outcome: DrainOutcome, reason_code: Optional[DrainReasonCode] = None
    ) -> JSONResponse:
        return JSONResponse(
            status_code=200,
            content=DrainResponse(outcome=outcome, reason_code=reason_code).model_dump(
                exclude_none=True
            ),
        )


def _now_ms() -> int:
    return int(time.time() * 1000)


def _now_monotonic() -> float:
    return time.monotonic()


def _positive(value: float, name: str) -> float:
    """Reject non-positive or non-finite drain tunables rather than silently
    collapsing retention, the deadline ceiling, or the attach grace. float()
    parses "nan" and "inf" successfully, so malformed env values reach here
    as non-finite numbers, not errors."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value}")
    return value


def register_drain_route(
    app: FastAPI,
    registry: DrainRegistry,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> None:
    """Register ``POST /drain`` on a runner server.

    The bearer-auth middleware already gates the path; this adds the
    execution/workspace-scoped validation on top. Callers of this endpoint
    must not follow redirects — the bearer token must never be forwarded to
    another host.

    Scope check: the bearer token authenticates the caller but does not
    establish that the named drain belongs to this runtime, so when the
    platform scopes this process to a workspace (``APP_ID`` set — every
    managed runtime), the request must name that workspace exactly; a missing
    ``workspace_id`` is a 400, a mismatch a 403. When ``APP_ID`` is unset
    (local development against a single unscoped runtime) no workspace check
    is possible and none is enforced.
    """
    source = os.environ if env is None else env
    workspace_id = (source.get("APP_ID") or "").strip()

    @app.post(
        "/drain",
        response_model=DrainResponse,
        responses={
            202: {
                "model": DrainResponse,
                "description": "Drain accepted; the final outcome is pending.",
            },
            400: {
                "description": "Missing workspace_id on a scoped runtime, or deadline beyond the configured maximum."
            },
            403: {"description": "workspace_id does not match this runtime's workspace."},
        },
    )
    async def drain(request: DrainRequest) -> JSONResponse:
        if workspace_id:
            if not request.workspace_id:
                raise HTTPException(status_code=400, detail="workspace_id required")
            if request.workspace_id != workspace_id:
                raise HTTPException(status_code=403, detail="workspace mismatch")
        if request.deadline_at_ms > _now_ms() + registry.max_deadline_ms:
            raise HTTPException(status_code=400, detail="deadline exceeds maximum")
        return registry.apply(request)
