"""Reconnect identity helpers for OE-owned durable executions.

The reconnect key is the OE-issued ``WorkflowIdentity.execution_id`` after the
first stream/execute assignment. Pre-execution-id invoke retries are not
deduplicated — callers must treat them as a new dispatch.
"""

from __future__ import annotations

from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
    WorkflowIdentity,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.workflow.client import WorkflowClientError

__all__ = [
    "execution_id_from_attempt",
    "is_pre_execution_id_retry",
    "require_reconnect_identity",
]


def require_reconnect_identity(identity: WorkflowIdentity) -> str:
    """Assert a WorkflowIdentity carries the execution_id used for reconnect.

    There is no separate reconnect token or Last-Event-ID protocol — the
    same execution_id resumes OE stream buffering and lease work.
    """
    if not identity.execution_id:
        raise WorkflowClientError(
            WORKFLOW_ERROR_CODE_INVALID_ARGUMENT,
            "reconnect requires WorkflowIdentity.execution_id",
        )
    return identity.execution_id


def execution_id_from_attempt(attempt: AttemptContext) -> str:
    """Return the reconnect key from an OE-issued attempt context."""
    return require_reconnect_identity(attempt.workflow_identity)


def is_pre_execution_id_retry(execution_id: str | None) -> bool:
    """True when an invoke/stream request has not yet been assigned an execution_id.

    Known gap: retries in this state are not deduplicated — OE may create a
    second execution. Prefer waiting for the first assigned id.
    """
    return execution_id is None or execution_id == ""
