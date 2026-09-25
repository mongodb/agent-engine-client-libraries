"""Thin runtime surface for the generated `workflow.v1` contract.

This package owns ProtoJSON transport helpers, the attempt/activity HTTP
clients, the request-scoped attempt context, and the attempt lifecycle
utilities. The generated messages under
`agent_engine_runner_shared.generated.workflow.v1` remain the only wire vocabulary.

Only the stable consumer surface is re-exported here. Endpoint path
constants, request-assembly helpers, and server-lifecycle utilities are
internals that platform code imports by their submodule path; they may
change between slices.
"""

from agent_engine_runner_shared.workflow.activity import (
    DurableActivityDeniedError,
    DurableActivityInterrupted,
    DurableActivitySuspended,
    ReplayedActivityFailedError,
    run_serial_activity,
)
from agent_engine_runner_shared.workflow.attempt import (
    complete_execution_command,
    finalize_current_step_command,
    finalize_current_step_suspensions_command,
)
from agent_engine_runner_shared.workflow.client import (
    AsyncWorkflowClient,
    WorkflowClient,
    WorkflowClientError,
)
from agent_engine_runner_shared.workflow.context import (
    ChildOperationBoundary,
    OperationPathResolver,
    UnsupportedChildOperationFanOutError,
    allocate_activity_ordinal,
    attempt_context_scope,
    child_operation_boundary_scope,
    current_attempt_context,
    current_operation_path,
    operation_path_resolver_scope,
    preallocate_activity_ordinals,
    preallocate_child_operation_ordinals,
    tool_activity_key,
)
from agent_engine_runner_shared.workflow.protojson import WorkflowWireError
from agent_engine_runner_shared.workflow.settlement import (
    SettledExecutionError,
    settle_execution,
)

__all__ = [
    "AsyncWorkflowClient",
    "ChildOperationBoundary",
    "DurableActivityDeniedError",
    "DurableActivityInterrupted",
    "DurableActivitySuspended",
    "OperationPathResolver",
    "ReplayedActivityFailedError",
    "UnsupportedChildOperationFanOutError",
    "WorkflowClient",
    "WorkflowClientError",
    "WorkflowWireError",
    "SettledExecutionError",
    "allocate_activity_ordinal",
    "attempt_context_scope",
    "child_operation_boundary_scope",
    "complete_execution_command",
    "current_attempt_context",
    "current_operation_path",
    "finalize_current_step_command",
    "finalize_current_step_suspensions_command",
    "operation_path_resolver_scope",
    "preallocate_activity_ordinals",
    "preallocate_child_operation_ordinals",
    "run_serial_activity",
    "settle_execution",
    "tool_activity_key",
]
