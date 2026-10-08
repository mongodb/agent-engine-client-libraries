"""OE state snapshots for the durable OpenAI Agents adapter.

``properties`` carries the native Runner input list, which is the only replay
authority. ``messages`` is an explicit-field projection for history readers
and is never read back.
"""

from __future__ import annotations

from typing import cast

from agent_engine_sdk import Message
from pydantic import JsonValue

from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.generated.workflow.v1.state_pb2 import (
    MESSAGE_ROLE_ASSISTANT,
    MESSAGE_ROLE_SYSTEM,
    MESSAGE_ROLE_TOOL,
    MESSAGE_ROLE_USER,
    StateSnapshot,
    WorkflowMessage,
)
from agent_engine_runner_shared.workflow.protojson import (
    json_to_proto_struct,
    json_to_proto_value,
    proto_struct_to_json,
)
from agent_engine_sdk_openai_agents.errors import DurableOpenAIAgentsStateError
from agent_engine_sdk_openai_agents.items import (
    NativeItem,
    to_platform_messages,
    validate_items,
)

__all__ = ["STATE_KEY", "decode_active_agent", "decode_state", "encode_state"]

STATE_KEY = "__agent_engine_openai_agents__"
_VERSION = 1
_ENVELOPE_KEYS = frozenset(
    {"version", "app_name", "session_id", "items", "active_agent"}
)
_ROLE_TO_WORKFLOW = {
    "user": MESSAGE_ROLE_USER,
    "assistant": MESSAGE_ROLE_ASSISTANT,
    "tool": MESSAGE_ROLE_TOOL,
    "system": MESSAGE_ROLE_SYSTEM,
}


def encode_state(
    items: list[NativeItem], *, app_name: str, session_id: str, active_agent: str
) -> StateSnapshot:
    """Encode validated items as the next committed snapshot.

    ``active_agent`` names the agent that ended the turn. After a handoff the
    item list does not reliably say who continues the conversation.
    """
    envelope: dict[str, JsonValue] = {
        "version": _VERSION,
        "app_name": app_name,
        "session_id": session_id,
        "items": [item.raw for item in items],
        "active_agent": active_agent,
    }
    return StateSnapshot(
        properties=json_to_proto_struct({STATE_KEY: envelope}),
        messages=[_workflow_message(m) for m in to_platform_messages(items)],
        message_encoding_version=1,
    )


def decode_state(attempt: AttemptContext, *, app_name: str) -> list[NativeItem]:
    """Return the committed items this attempt continues from."""
    envelope = _envelope(attempt, app_name)
    if envelope is None:
        return []
    items = envelope["items"]
    if not isinstance(items, list):
        raise DurableOpenAIAgentsStateError(
            "previous OpenAI Agents state items must be an array"
        )
    return validate_items(items)


def decode_active_agent(attempt: AttemptContext, *, app_name: str) -> str | None:
    """Return the agent that ended the last committed turn, if one is recorded."""
    envelope = _envelope(attempt, app_name)
    if envelope is None:
        return None
    active_agent = envelope["active_agent"]
    if not isinstance(active_agent, str) or not active_agent:
        raise DurableOpenAIAgentsStateError(
            "previous OpenAI Agents state active agent must be a name"
        )
    return active_agent


def _envelope(attempt: AttemptContext, app_name: str) -> dict[str, JsonValue] | None:
    if not attempt.HasField("previous_state"):
        return None
    envelope = proto_struct_to_json(attempt.previous_state.properties).get(STATE_KEY)
    if not isinstance(envelope, dict):
        raise DurableOpenAIAgentsStateError(
            "previous state is missing OpenAI Agents adapter state"
        )
    envelope = cast(dict[str, JsonValue], envelope)
    version = envelope.get("version")
    # JSON true decodes to Python True, which compares equal to 1.
    if (
        frozenset(envelope) != _ENVELOPE_KEYS
        or isinstance(version, bool)
        or version != _VERSION
    ):
        raise DurableOpenAIAgentsStateError(
            "previous OpenAI Agents state has an unsupported envelope"
        )
    # A branch's first attempt continues from its source session's snapshot.
    expected_session_id = (
        attempt.branch_lineage.source_workflow_identity.session_id
        if attempt.HasField("branch_lineage")
        else attempt.workflow_identity.session_id
    )
    if (envelope["app_name"], envelope["session_id"]) != (
        app_name,
        expected_session_id,
    ):
        raise DurableOpenAIAgentsStateError(
            "previous OpenAI Agents state belongs to a different app or session"
        )
    return envelope


def _workflow_message(message: Message) -> WorkflowMessage:
    result = WorkflowMessage(
        role=_ROLE_TO_WORKFLOW[message.role],
        content=json_to_proto_value(cast(JsonValue, message.content)),
    )
    for call in message.tool_calls or []:
        result.tool_calls.append(
            json_to_proto_struct(call.model_dump(mode="json", exclude_none=True))
        )
    if message.tool_call_id is not None:
        result.tool_call_id = message.tool_call_id
    if message.id is not None:
        result.id = message.id
    return result
