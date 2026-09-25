"""Per-call interrupt receiver for Runner SDK servers.

Mirrors TypeScript's ``agent-engine-runner-shared/src/server/callInterrupt.ts``. Both
runtimes are parallel implementations of the same server architecture, so
this file and its TypeScript twin must be changed together.

When the OE's per-call interrupt fires, it POSTs ``/interrupt/call`` with the
execution and the step to abandon. The runtime cancels exactly that call's
tracked work and nothing else — unlike ``POST /drain``, which is terminal for
the execution, an interrupt is surgical: no admission latch, no record, and
later steps of the same execution proceed.

Contract (pinned by the shared fixture in
``client-libraries/test-fixtures/interrupt-call/contract.json``):

- ``POST /interrupt/call {execution_id, step_number, workspace_id}`` —
  ``workspace_id`` is mandatory on workspace-scoped runtimes (``APP_ID``
  set) and must match it exactly.
- Always ``200`` with an outcome: ``interrupted`` (the call's task was
  signalled, or the abort is recorded to fire at attach), ``not_found`` (no
  such in-flight call), ``already_settled`` (the call ended first),
  ``not_cancellable`` (tracked work with no signal channel — e.g. a sync
  tool on a worker thread — reported honestly rather than claimed
  abandoned).
- Idempotent by (execution_id, step_number): a repeat re-reads the outcome.
"""

from __future__ import annotations

import logging
import os
from typing import Mapping, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_engine_runner_shared.server.drain import CallInterruptOutcome, DrainRegistry

logger = logging.getLogger(__name__)

__all__ = [
    "CallInterruptRequest",
    "CallInterruptResponse",
    "register_call_interrupt_route",
]


class CallInterruptRequest(BaseModel):
    execution_id: str = Field(..., min_length=1, max_length=256)
    step_number: int = Field(..., ge=0)
    workspace_id: Optional[str] = Field(default=None, max_length=256)


class CallInterruptResponse(BaseModel):
    outcome: CallInterruptOutcome


def register_call_interrupt_route(
    app: FastAPI,
    registry: DrainRegistry,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> None:
    """Register ``POST /interrupt/call`` on a runner server.

    The bearer-auth middleware already gates the path; this adds the
    execution/workspace-scoped validation on top, identical to the drain
    route's: when the platform scopes this process to a workspace (``APP_ID``
    set — every managed runtime), the request must name that workspace
    exactly; a missing ``workspace_id`` is a 400, a mismatch a 403. When
    ``APP_ID`` is unset (local development) no workspace check is possible
    and none is enforced.
    """
    source = os.environ if env is None else env
    workspace_id = (source.get("APP_ID") or "").strip()

    @app.post(
        "/interrupt/call",
        response_model=CallInterruptResponse,
        responses={
            400: {"description": "Missing workspace_id on a scoped runtime."},
            403: {"description": "workspace_id does not match this runtime's workspace."},
        },
    )
    async def interrupt_call(request: CallInterruptRequest) -> CallInterruptResponse:
        if workspace_id:
            if not request.workspace_id:
                raise HTTPException(status_code=400, detail="workspace_id required")
            if request.workspace_id != workspace_id:
                raise HTTPException(status_code=403, detail="workspace mismatch")
        outcome = registry.abort_call(request.execution_id, request.step_number)
        # The runtime-side evidence of a per-call abort, mirroring the drain
        # receiver's accepted/finalized markers.
        logger.info(
            "Call interrupt %s for execution %s (step %d)",
            outcome.value,
            request.execution_id,
            request.step_number,
        )
        return CallInterruptResponse(outcome=outcome)
