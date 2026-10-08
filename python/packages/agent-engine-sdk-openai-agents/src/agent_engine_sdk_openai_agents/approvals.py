"""Record OpenAI Agents approval interruptions as OE wait frontiers.

A tool with ``needs_approval`` stops the native run with a ``ToolApprovalItem``
for each call that needs it. The adapter records those waits with OE as one
suspension frontier: the first time, OE creates the waits and the turn
suspends; once the caller answers every wait through ``/invoke`` with
``resume_map``, replay reaches the same frontier and OE returns the recorded
answers, which the adapter applies through ``RunState``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from agent_engine_sdk import StreamEvent
from agents import ToolApprovalItem
from agents.tool import DEFAULT_APPROVAL_REJECTION_MESSAGE
from pydantic import JsonValue

from agent_engine_runner_shared.context import (
    get_current_oe_url,
    get_current_user_id,
    get_current_wrapper,
)
from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_KIND_TOOL,
    ACTIVITY_OUTCOME_KIND_SUSPENDED,
    ActivityContext,
    ActivityOutcome,
    ActivitySuspension,
    StepSuspensionEntry,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    WORKFLOW_ERROR_CODE_CONFLICT,
    ActivityPosition,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.workflow import AsyncWorkflowClient
from agent_engine_runner_shared.workflow.activity import (
    build_activity_command,
    unwrap_activity_outcome,
)
from agent_engine_runner_shared.workflow.attempt import (
    finalize_current_step_suspensions_command,
)
from agent_engine_runner_shared.workflow.client import WorkflowClientError
from agent_engine_runner_shared.workflow.context import (
    allocate_activity_ordinal,
    record_observed_activity,
)
from agent_engine_runner_shared.workflow.protojson import json_to_proto_struct
from agent_engine_sdk_openai_agents.errors import UnsupportedDurableOpenAIAgentsError

__all__ = [
    "APPROVAL_ANSWER_SCHEMA",
    "ApprovalDecision",
    "PendingApproval",
    "approval_call_id",
    "record_approvals",
    "suspend_event",
]

SUSPENSION_REASON = "tool_approval"

# One answer per pending approval: approve, or reject with an optional
# rejection_message, the SDK's own name for the text the model reads in place
# of the tool's result. An approval carries no message, since the SDK's
# approve has nowhere to put one.
APPROVAL_ANSWER_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "required": ["confirmed"],
    "properties": {
        "confirmed": {"type": "boolean"},
        # Not empty: omit it for the SDK's default rejection text.
        "rejection_message": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
    "if": {"properties": {"confirmed": {"const": True}}},
    "then": {"not": {"required": ["rejection_message"]}},
}


@dataclass(frozen=True)
class PendingApproval:
    """OE recorded a new wait; the turn suspends on it."""

    call_id: str
    activity_id: str
    value: dict[str, JsonValue]


@dataclass(frozen=True)
class ApprovalDecision:
    """OE returned the caller's recorded answer for this wait."""

    call_id: str
    activity_id: str
    approved: bool
    rejection_message: str | None


def _approval_value(item: ToolApprovalItem) -> dict[str, JsonValue]:
    raw = cast(Any, item.raw_item)
    call_id = getattr(raw, "call_id", None)
    name = item.tool_name or getattr(raw, "name", None)
    arguments = getattr(raw, "arguments", None)
    if not isinstance(call_id, str) or not call_id or not isinstance(name, str):
        raise UnsupportedDurableOpenAIAgentsError(
            "only function-tool approvals with a call id are supported"
        )
    try:
        args: JsonValue = (
            json.loads(arguments) if isinstance(arguments, str) and arguments else {}
        )
    except json.JSONDecodeError as error:
        raise UnsupportedDurableOpenAIAgentsError(
            f"tool {name!r} approval has arguments that are not JSON"
        ) from error
    return {
        "tool_call": {"name": name, "args": args, "call_id": call_id},
        "response_schema": APPROVAL_ANSWER_SCHEMA,
    }


def _decision(call_id: str, activity_id: str, answer: object) -> ApprovalDecision:
    # The answer is caller input that OE validated against the resume schema;
    # check it again at this sink rather than trusting the transport.
    if not isinstance(answer, dict):
        raise UnsupportedDurableOpenAIAgentsError("approval answer must be an object")
    answer = cast(dict[str, object], answer)
    confirmed = answer.get("confirmed")
    rejection_message = answer.get("rejection_message")
    if (
        set(answer) - {"confirmed", "rejection_message"}
        or not isinstance(confirmed, bool)
        or not (
            rejection_message is None
            or (isinstance(rejection_message, str) and rejection_message)
        )
        or (confirmed and rejection_message is not None)
    ):
        raise UnsupportedDurableOpenAIAgentsError(
            "approval answer must be {confirmed: true} or "
            "{confirmed: false, rejection_message?: non-empty string}"
        )
    return ApprovalDecision(
        call_id=call_id,
        activity_id=activity_id,
        approved=confirmed,
        rejection_message=rejection_message,
    )


async def record_approvals(
    attempt: AttemptContext,
    interruptions: list[ToolApprovalItem],
    call_order: Sequence[str],
) -> tuple[PendingApproval, ...] | tuple[ApprovalDecision, ...]:
    """Record the run's approval waits with OE, or return their recorded answers.

    Every pending approval joins one frontier, in the model's call order; OE
    resolves the whole frontier at once, so the waits are either all pending or
    all answered.
    """
    if not interruptions:
        raise UnsupportedDurableOpenAIAgentsError("there is no approval to record")
    position = {call_id: index for index, call_id in enumerate(call_order)}
    values = [_approval_value(item) for item in interruptions]
    call_ids = [_call_id(value) for value in values]
    if len(set(call_ids)) != len(call_ids) or not set(call_ids) <= set(position):
        raise UnsupportedDurableOpenAIAgentsError(
            "approvals must name distinct calls from the run's input"
        )
    # The SDK does not promise its interruption order on a resumed run, and a
    # wait's ordinal depends on when it is allocated.
    ordered = sorted(zip(call_ids, values, strict=True), key=lambda i: position[i[0]])
    commands = [
        build_activity_command(
            attempt=attempt,
            kind=ACTIVITY_KIND_TOOL,
            name=f"approve:{cast(dict[str, JsonValue], value['tool_call'])['name']}",
            # Keyed by the model's call id so every replay reaches the same wait.
            activity_ordinal=allocate_activity_ordinal(f"approval:{call_id}"),
            semantic_input=value["tool_call"],
        )
        for call_id, value in ordered
    ]
    entries: list[StepSuspensionEntry] = []
    for command, (_, value) in zip(commands, ordered, strict=True):
        record_observed_activity(command.position)
        entries.append(
            StepSuspensionEntry(
                position=command.position,
                activity_kind=command.activity_kind,
                activity_name=command.activity_name,
                semantic_input=command.semantic_input,
                suspension=ActivitySuspension(
                    reason=SUSPENSION_REASON,
                    context=json_to_proto_struct({"value": value}),
                ),
            )
        )
    oe_url = get_current_oe_url()
    if not oe_url:
        raise UnsupportedDurableOpenAIAgentsError(
            "OpenAI Agents approvals require the OE callback URL"
        )
    try:
        async with AsyncWorkflowClient(oe_url) as client:
            returned = await client.finalize_step(
                finalize_current_step_suspensions_command(attempt, entries)
            )
    except WorkflowClientError as error:
        if error.code == WORKFLOW_ERROR_CODE_CONFLICT:
            raise UnsupportedDurableOpenAIAgentsError(
                "the approval waits could not be recorded under the current fence"
            ) from error
        raise
    by_position = {_position_key(entry.position): entry.outcome for entry in returned}
    outcomes = [
        by_position.get(_position_key(command.position)) for command in commands
    ]
    if len(returned) != len(commands) or None in outcomes:
        raise UnsupportedDurableOpenAIAgentsError(
            "OE returned an unexpected approval frontier"
        )
    resolved = cast(list[ActivityOutcome], outcomes)
    suspended = [
        outcome.outcome_kind == ACTIVITY_OUTCOME_KIND_SUSPENDED for outcome in resolved
    ]
    if all(suspended):
        return tuple(
            PendingApproval(
                call_id=call_id, activity_id=outcome.activity_id, value=value
            )
            for (call_id, value), outcome in zip(ordered, resolved, strict=True)
        )
    if any(suspended):
        raise UnsupportedDurableOpenAIAgentsError(
            "OE returned a partly answered approval frontier"
        )
    # Every answer is checked before any is acknowledged, so a malformed one
    # fails the turn without a partly acknowledged frontier.
    answers = [unwrap_activity_outcome(outcome) for outcome in resolved]
    decisions = [
        _decision(call_id, outcome.activity_id, answer)
        for (call_id, _), outcome, answer in zip(
            ordered, resolved, answers, strict=True
        )
    ]
    for (_, value), command, outcome, answer, decision in zip(
        ordered, commands, resolved, answers, decisions, strict=True
    ):
        await _acknowledge(outcome, answer, command.activity_name, decision, value)
    return tuple(decisions)


async def _acknowledge(
    outcome: ActivityOutcome,
    answer: object,
    name: str,
    decision: ApprovalDecision,
    value: dict[str, JsonValue],
) -> None:
    wrapper = get_current_wrapper()
    if wrapper is None or wrapper.durable_memory is None:
        return
    # A wait never runs through the secure tool wrapper, so nothing else
    # acknowledges its resolved activity to durable Memory, and a step with an
    # unacknowledged activity cannot commit. An approval is not conversation
    # content: the tool then runs and writes its own result. A rejection is:
    # the model reads its text as the call's output, so Memory gets the same
    # tool message and the call is not left open.
    rejection = None if decision.approved else rejection_text(decision)
    tool_call = cast(dict[str, JsonValue], value["tool_call"])
    await asyncio.to_thread(
        wrapper.durable_memory.synchronize_tool,
        wrapper.workflow,
        ActivityContext(
            workflow_identity=outcome.workflow_identity,
            activity_id=outcome.activity_id,
            attempt_id=outcome.attempt_id,
            fencing_token=outcome.fencing_token,
        ),
        answer if rejection is None else rejection,
        user_id=get_current_user_id(),
        tool_call_id=None if rejection is None else decision.call_id,
        tool_name=name if rejection is None else str(tool_call["name"]),
    )


def rejection_text(decision: ApprovalDecision) -> str:
    """The text the SDK gives the model as a rejected call's output."""
    if decision.rejection_message is None:
        return DEFAULT_APPROVAL_REJECTION_MESSAGE
    return decision.rejection_message


def approval_call_id(item: ToolApprovalItem) -> str:
    """The model's call id for the call an interruption asks about."""
    return _call_id(_approval_value(item))


def _call_id(value: dict[str, JsonValue]) -> str:
    return cast(str, cast(dict[str, JsonValue], value["tool_call"])["call_id"])


def _position_key(position: ActivityPosition) -> bytes:
    return position.SerializeToString(deterministic=True)


def suspend_event(pending: Sequence[PendingApproval], *, resumed: bool) -> StreamEvent:
    """The caller-facing suspend event for a frontier of pending approvals."""
    interrupts: list[JsonValue] = [
        {"id": wait.activity_id, "value": wait.value} for wait in pending
    ]
    ids = [wait.activity_id for wait in pending]
    return StreamEvent(
        event="suspend",
        data={
            # Retained for callers that read one payload; interrupts has all.
            "suspend_payload": pending[0].value,
            "resumed": resumed,
            "messages": [],
            "metadata": {},
            "interrupts": interrupts,
            "resume_schema": {
                "type": "object",
                "required": ["resume_map"],
                "properties": {
                    "resume_map": {
                        "type": "object",
                        "required": cast(list[JsonValue], ids),
                        "properties": {
                            wait_id: APPROVAL_ANSWER_SCHEMA for wait_id in ids
                        },
                        "additionalProperties": False,
                    }
                },
                "additionalProperties": True,
            },
            "response": "",
        },
    )
