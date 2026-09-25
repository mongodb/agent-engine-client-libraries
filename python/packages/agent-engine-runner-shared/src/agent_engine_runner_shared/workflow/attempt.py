"""Build framework-neutral workflow attempt lifecycle commands.

Only OE may mint an ``AttemptContext``. These helpers assemble requests and
state transitions around that issued identity. Missing tenant identity keeps
attempt start native.
"""

from __future__ import annotations

import os
import socket
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import StepSuspensionEntry
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    TenantScope,
    WorkflowIdentity,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import (
    AttemptContext,
    AttemptStartRequest,
    WorkflowDeclaration,
)
from agent_engine_runner_shared.generated.workflow.v1.state_pb2 import (
    CompleteExecutionCommand,
    FinalizeStepCommand,
    StateSnapshot,
)
from agent_engine_runner_shared.workflow.context import (
    current_step_ordinal,
    observed_activity_positions,
)

if TYPE_CHECKING:
    from agent_engine_runner_shared.models import ExecuteRequest

__all__ = [
    "attempt_start_request_from_execute",
    "complete_execution_command",
    "finalize_current_step_command",
    "finalize_current_step_suspensions_command",
    "heartbeat_interval_seconds_from_attempt",
    "new_durability_owner_id",
    "workflow_declaration_for_app",
    "workflow_identity_from_execute_request",
]

# Default workflow version when agent.yaml does not yet declare one.
DEFAULT_WORKFLOW_VERSION = "1"


def finalize_current_step_command(
    attempt: AttemptContext,
    state: StateSnapshot,
) -> FinalizeStepCommand:
    """Build the settled finalization for the active framework step."""
    step_ordinal = current_step_ordinal()
    return FinalizeStepCommand(
        workflow_identity=attempt.workflow_identity,
        attempt_id=attempt.attempt_id,
        fencing_token=attempt.fencing_token,
        state=state,
        step_ordinal=step_ordinal,
        observed_activity_positions=observed_activity_positions(step_ordinal),
    )


def finalize_current_step_suspensions_command(
    attempt: AttemptContext,
    suspensions: Sequence[StepSuspensionEntry],
) -> FinalizeStepCommand:
    """Build a suspension finalization for the active framework step."""
    if not suspensions:
        raise ValueError("step suspension finalization requires at least one entry")
    step_ordinal = current_step_ordinal()
    if any(entry.position.step_ordinal != step_ordinal for entry in suspensions):
        raise ValueError("step suspension entry belongs to another workflow step")
    return FinalizeStepCommand(
        workflow_identity=attempt.workflow_identity,
        attempt_id=attempt.attempt_id,
        fencing_token=attempt.fencing_token,
        step_ordinal=step_ordinal,
        observed_activity_positions=observed_activity_positions(step_ordinal),
        suspensions=suspensions,
    )


def complete_execution_command(
    attempt: AttemptContext,
    state: StateSnapshot,
) -> CompleteExecutionCommand:
    """Build the fenced completion for one accepted final state."""
    return CompleteExecutionCommand(
        workflow_identity=attempt.workflow_identity,
        attempt_id=attempt.attempt_id,
        fencing_token=attempt.fencing_token,
        state=state,
    )


def new_durability_owner_id() -> str:
    """Stable-enough owner id for one AER process lifetime."""
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4()}"


def workflow_identity_from_execute_request(
    request: ExecuteRequest, session_id: str
) -> WorkflowIdentity | None:
    """Build WorkflowIdentity, or ``None`` when the request lacks the keys OE needs."""
    org_id = (request.org_id or "").strip()
    project_id = (request.project_id or "").strip()
    workspace_id = (request.workspace_id or "").strip()
    execution_id = (request.execution_id or "").strip()
    resolved_session = session_id.strip()
    if not all((org_id, project_id, workspace_id, execution_id, resolved_session)):
        return None
    return WorkflowIdentity(
        tenant_scope=TenantScope(
            org_id=org_id,
            project_id=project_id,
            workspace_id=workspace_id,
        ),
        session_id=resolved_session,
        execution_id=execution_id,
    )


def workflow_declaration_for_app(
    app_name: str,
    workflow_version: str | None = None,
    *,
    memory_enabled: bool = False,
) -> WorkflowDeclaration | None:
    """Declaration sent on StartAttempt so OE can pin recovery to this adapter.

    The declaration is OE's recovery-pinning contract: ``workflow_version``
    should be the developer-configured application version so incompatible
    deployments are distinguishable. ``None`` when no framework SDK registered
    a workflow adapter; an unregistered framework must never start an attempt.
    """
    from agent_engine_runner_shared.hooks import get_workflow_adapter

    adapter = get_workflow_adapter()
    if adapter is None:
        return None
    adapter_name, adapter_version = adapter
    return WorkflowDeclaration(
        workflow_name=app_name,
        workflow_version=workflow_version or DEFAULT_WORKFLOW_VERSION,
        adapter_name=adapter_name,
        adapter_version=adapter_version,
        memory_enabled=memory_enabled,
    )


def attempt_start_request_from_execute(
    request: ExecuteRequest,
    session_id: str,
    owner_id: str,
    app_name: str,
    workflow_version: str | None = None,
    *,
    memory_enabled: bool = False,
) -> AttemptStartRequest | None:
    """Assemble a ProtoJSON AttemptStartRequest, or ``None`` for native routing.

    ``None`` covers incomplete tenant identity and frameworks that never
    registered a workflow adapter.
    """
    workflow_identity = workflow_identity_from_execute_request(request, session_id)
    if workflow_identity is None:
        return None
    declaration = workflow_declaration_for_app(
        app_name,
        workflow_version,
        memory_enabled=memory_enabled,
    )
    if declaration is None:
        return None
    return AttemptStartRequest(
        workflow_identity=workflow_identity,
        owner_id=owner_id,
        declaration=declaration,
    )


def heartbeat_interval_seconds_from_attempt(heartbeat_interval_ms: int) -> float:
    """Convert OE's int64 heartbeat interval into a positive number of seconds."""
    if heartbeat_interval_ms <= 0:
        raise ValueError("AttemptContext.heartbeat_interval_ms must be a positive duration")
    return heartbeat_interval_ms / 1000.0
