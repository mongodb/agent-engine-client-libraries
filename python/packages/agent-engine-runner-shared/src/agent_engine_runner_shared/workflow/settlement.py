"""Terminal settlement shared by durable framework adapters."""

from __future__ import annotations

from agent_engine_runner_shared.context import get_current_oe_url
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.generated.workflow.v1.state_pb2 import StateSnapshot
from agent_engine_runner_shared.workflow.attempt import (
    complete_execution_command,
    finalize_current_step_command,
)
from agent_engine_runner_shared.workflow.client import AsyncWorkflowClient


class SettledExecutionError(RuntimeError):
    """The active durable turn could not be settled as a terminal result."""


async def settle_execution(attempt: AttemptContext, state: StateSnapshot) -> None:
    """Finalize the active step, then complete it with the same state."""
    oe_url = get_current_oe_url()
    if not oe_url:
        raise SettledExecutionError("durable state requires the OE callback URL")
    async with AsyncWorkflowClient(oe_url) as client:
        entries = await client.finalize_step(finalize_current_step_command(attempt, state))
        if entries:
            raise SettledExecutionError("settled step finalization returned suspension entries")
        await client.complete_execution(complete_execution_command(attempt, state))
