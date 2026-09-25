"""Translate between LangGraph channel values and OE-owned session state."""

from __future__ import annotations

from typing import Any

from agent_engine_runner_shared.generated.workflow.v1.state_pb2 import StateSnapshot
from agent_engine_runner_shared.workflow.protojson import (
    json_to_proto_struct,
    proto_struct_to_json,
)

from .messages import lc_to_workflow_message, workflow_to_lc_message

__all__ = ["channel_values_to_state_snapshot", "state_snapshot_to_channel_values"]


def _is_langgraph_control_channel(name: str) -> bool:
    """Identify checkpoint bookkeeping that is not application session state."""
    return (
        name.startswith("__") or name.startswith("branch:") or name.startswith("start:")
    )


def channel_values_to_state_snapshot(
    channel_values: dict[str, Any],
    *,
    include_legacy_source: bool = False,
) -> StateSnapshot:
    """Extract committed application values from a LangGraph checkpoint.

    LangGraph control channels remain attempt-local scratch. Conversation
    messages use their dedicated workflow representation so role, tool-call,
    and message metadata survive reconstruction; all other application
    channels remain in the snapshot's generic properties object.
    """
    properties = {
        name: channel_values[name]
        for name in sorted(channel_values)
        if name != "messages" and not _is_langgraph_control_channel(name)
    }
    messages = [
        lc_to_workflow_message(
            message,
            include_legacy_source=include_legacy_source,
        )
        for message in channel_values.get("messages", [])
    ]
    return StateSnapshot(
        properties=json_to_proto_struct(properties),
        messages=messages,
        message_encoding_version=1,
    )


def state_snapshot_to_channel_values(snapshot: StateSnapshot) -> dict[str, Any]:
    """Reconstruct the LangGraph channel values committed by an earlier turn.

    The protobuf snapshot uses JSON because each application defines its own
    graph state schema. The compiled graph remains the owner of those concrete
    channel types.
    """
    channel_values = proto_struct_to_json(snapshot.properties)
    for name in sorted(channel_values):
        if name == "messages" or _is_langgraph_control_channel(name):
            raise ValueError(
                f'previous workflow state contains reserved LangGraph channel "{name}"'
            )
    if snapshot.messages:
        channel_values["messages"] = [
            workflow_to_lc_message(message, path=f"messages[{index}]")
            for index, message in enumerate(snapshot.messages)
        ]
    return channel_values
