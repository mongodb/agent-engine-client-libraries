"""OE step finalization for durable ADK waits and completion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from google.adk.sessions import Session

from agent_engine_runner_shared.context import (
    get_current_oe_url,
    get_current_user_id,
    get_current_wrapper,
)
from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_KIND_TOOL,
    ACTIVITY_OUTCOME_KIND_SUSPENDED,
    ActivityContext,
    ActivitySuspension,
    StepSuspensionEntry,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    WORKFLOW_ERROR_CODE_CONFLICT,
    ActivityPosition,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.generated.workflow.v1.state_pb2 import (
    FinalizeStepCommand,
)
from agent_engine_runner_shared.workflow import (
    AsyncWorkflowClient,
    SettledExecutionError,
    finalize_current_step_command,
    settle_execution,
)
from agent_engine_runner_shared.workflow.activity import (
    build_activity_command,
    unwrap_activity_outcome,
)
from agent_engine_runner_shared.workflow.client import WorkflowClientError
from agent_engine_runner_shared.workflow.context import (
    advance_step_ordinal,
    allocate_activity_ordinal,
    current_step_ordinal,
    observed_activity_positions,
    record_observed_activity,
)
from agent_engine_runner_shared.workflow.protojson import json_to_proto_struct
from agent_engine_sdk_adk.errors import UnsupportedDurableADKError
from agent_engine_sdk_adk.platform_session import session_to_state_snapshot
from agent_engine_sdk_adk.route import adk_suspension_operation_path_scope
from agent_engine_sdk_adk.suspend import AdkSuspend

__all__ = [
    "CompletedWaitFrontier",
    "FreshWaitFrontier",
    "complete_durable_execution",
    "resolve_wait_frontier",
]


@dataclass(frozen=True)
class FreshWaitFrontier:
    """OE atomically recorded a new wait frontier."""

    suspends: tuple[AdkSuspend, ...]
    activity_ids: tuple[str, ...]


@dataclass(frozen=True)
class CompletedWaitFrontier:
    """OE replayed every answer in a previously completed wait frontier."""

    suspends: tuple[AdkSuspend, ...]
    activity_ids: tuple[str, ...]
    results: tuple[Any, ...]


def _require_oe_url() -> str:
    oe_url = get_current_oe_url()
    if not oe_url:
        raise UnsupportedDurableADKError(
            "Google ADK durable state requires the OE callback URL"
        )
    return oe_url


async def complete_durable_execution(attempt: AttemptContext, session: Session) -> None:
    """Finalize a settled step and complete the execution."""
    snapshot = session_to_state_snapshot(session)
    try:
        await settle_execution(attempt, snapshot)
    except SettledExecutionError as error:
        raise UnsupportedDurableADKError(str(error)) from error


def _suspension_entry(
    attempt: AttemptContext, suspend: AdkSuspend
) -> StepSuspensionEntry:
    with adk_suspension_operation_path_scope(suspend.provenance.operation_boundaries):
        command = build_activity_command(
            attempt=attempt,
            kind=ACTIVITY_KIND_TOOL,
            name=suspend.kind,
            activity_ordinal=allocate_activity_ordinal(),
            semantic_input=suspend.semantic_input,
        )
    record_observed_activity(command.position)
    return StepSuspensionEntry(
        position=command.position,
        activity_kind=command.activity_kind,
        activity_name=command.activity_name,
        semantic_input=command.semantic_input,
        suspension=ActivitySuspension(
            reason=suspend.kind,
            context=json_to_proto_struct({"value": suspend.value}),
        ),
    )


def _position_key(
    position: ActivityPosition,
) -> tuple[int, tuple[tuple[str, int], ...], int]:
    return (
        position.step_ordinal,
        tuple(
            (segment.name, segment.ordinal)
            for segment in position.operation_path.segments
        ),
        position.activity_ordinal,
    )


async def resolve_wait_frontier(
    attempt: AttemptContext,
    session: Session,
    suspends: tuple[AdkSuspend, ...],
) -> FreshWaitFrontier | CompletedWaitFrontier:
    """Finalize a frontier, committing and advancing it once all waits resolve."""
    wrapper = get_current_wrapper()
    entries = [_suspension_entry(attempt, suspend) for suspend in suspends]
    step_ordinal = current_step_ordinal()
    command = FinalizeStepCommand(
        workflow_identity=attempt.workflow_identity,
        attempt_id=attempt.attempt_id,
        fencing_token=attempt.fencing_token,
        step_ordinal=step_ordinal,
        observed_activity_positions=observed_activity_positions(step_ordinal),
        suspensions=entries,
    )
    try:
        async with AsyncWorkflowClient(_require_oe_url()) as client:
            returned_entries = await client.finalize_step(command)
            returned_by_position = {
                _position_key(entry.position): entry for entry in returned_entries
            }
            ordered_outcomes = [
                returned_by_position[_position_key(entry.position)].outcome
                for entry in entries
            ]
            if ordered_outcomes[0].outcome_kind == ACTIVITY_OUTCOME_KIND_SUSPENDED:
                return FreshWaitFrontier(
                    suspends=suspends,
                    activity_ids=tuple(
                        outcome.activity_id for outcome in ordered_outcomes
                    ),
                )

            if wrapper is not None and wrapper.durable_memory is not None:
                for suspend, outcome in zip(suspends, ordered_outcomes, strict=True):
                    await asyncio.to_thread(
                        wrapper.durable_memory.synchronize_tool,
                        wrapper.workflow,
                        ActivityContext(
                            workflow_identity=outcome.workflow_identity,
                            activity_id=outcome.activity_id,
                            attempt_id=outcome.attempt_id,
                            fencing_token=outcome.fencing_token,
                        ),
                        unwrap_activity_outcome(outcome),
                        user_id=get_current_user_id(),
                        tool_call_id=None,
                        tool_name=suspend.kind,
                    )
            snapshot = session_to_state_snapshot(session)
            await client.finalize_step(finalize_current_step_command(attempt, snapshot))
    except WorkflowClientError as error:
        if error.code == WORKFLOW_ERROR_CODE_CONFLICT:
            raise UnsupportedDurableADKError(
                "Google ADK wait frontier could not be recorded under the current fence"
            ) from error
        raise
    advance_step_ordinal(step_ordinal)
    return CompletedWaitFrontier(
        suspends=suspends,
        activity_ids=tuple(outcome.activity_id for outcome in ordered_outcomes),
        results=tuple(unwrap_activity_outcome(outcome) for outcome in ordered_outcomes),
    )
