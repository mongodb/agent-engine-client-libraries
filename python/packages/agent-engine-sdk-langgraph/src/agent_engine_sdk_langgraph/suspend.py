"""LangGraph ``interrupt()`` HITL suspend payloads for the adapter.

Owns validation, ``resume_schema``, and the invoke/stream suspend shapes the
AER consumes. This is adapter suspend, not ``App.suspend``. Durable sessions
publish OE activity IDs while native sessions publish LangGraph interrupt IDs.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, cast

from langgraph.types import Interrupt
from agent_engine_sdk import AgentOutput, Message, StreamEvent
from pydantic import JsonValue


def canonical_interrupt_value(interrupt_id: Any, value: Any) -> str:
    """Validate an interrupt and canonically serialize its value for comparison."""
    if not isinstance(interrupt_id, str) or not interrupt_id:
        raise RuntimeError("LangGraph returned an empty or invalid interrupt id")
    try:
        return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        value_type = type(value).__name__
        raise ValueError(
            f'LangGraph interrupt "{interrupt_id}" has a non-JSON-serializable '
            f"value of type {value_type}"
        ) from exc


def _resume_schema(interrupt_ids: list[str]) -> dict[str, Any]:
    resume_map = {
        "type": "object",
        "required": interrupt_ids,
        "properties": {interrupt_id: {} for interrupt_id in interrupt_ids},
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "required": ["resume_map"],
        "properties": {"resume_map": resume_map},
        "additionalProperties": True,
    }


def _interrupt_snapshot(
    interrupts: Iterable[Interrupt | dict[str, Any]],
) -> list[dict[str, Any]]:
    snapshot = [
        (
            {"id": interrupt.id, "value": interrupt.value}
            if isinstance(interrupt, Interrupt)
            else interrupt
        )
        for interrupt in interrupts
    ]
    unique: list[dict[str, Any]] = []
    values_by_id: dict[str, str] = {}
    for item in snapshot:
        interrupt_id = item["id"]
        serialized_value = canonical_interrupt_value(interrupt_id, item["value"])
        # LangGraph can project a child interrupt onto its parent with the same
        # ID. Only an identical payload is safe to collapse for resume routing.
        previous_value = values_by_id.get(interrupt_id)
        if previous_value is None:
            values_by_id[interrupt_id] = serialized_value
            unique.append(item)
        elif previous_value != serialized_value:
            raise RuntimeError(
                f'LangGraph returned conflicting values for interrupt "{interrupt_id}"'
            )
    return unique


def invoke_suspend_output(
    *,
    response: str,
    thread_id: str,
    checkpoint_id: str | None,
    interrupts: Iterable[Interrupt | dict[str, Any]],
    message_count: int,
    resumed: bool,
) -> AgentOutput:
    pending = _interrupt_snapshot(interrupts)
    interrupt_ids = [str(interrupt["id"]) for interrupt in pending]
    return AgentOutput(
        response=cast(
            JsonValue,
            {
                "response": response,
                "execution_id": thread_id,
                "status": "suspended",
                "suspend_context": {
                    "checkpoint_id": checkpoint_id,
                    "interrupt_values": [interrupt["value"] for interrupt in pending],
                },
                "interrupts": pending,
                "resume_schema": _resume_schema(interrupt_ids),
                "message_count": message_count,
                "resumed": resumed,
            },
        )
    )


def suspend_event(
    payload: Any,
    *,
    resumed: bool,
    messages: list[Message],
    metadata: dict[str, Any],
    interrupts: list[dict[str, Any]] | None = None,
    resume_schema: dict[str, Any] | None = None,
) -> StreamEvent:
    """AER ``suspend`` stream frame. HITL adds ``interrupts`` / ``resume_schema``."""
    data: dict[str, JsonValue] = {
        # Retained so older runners can still consume a new adapter event.
        "suspend_payload": cast(JsonValue, payload),
        "resumed": resumed,
        "messages": cast(JsonValue, [message.model_dump() for message in messages]),
        "metadata": cast(JsonValue, metadata),
    }
    if interrupts is not None and resume_schema is not None:
        data["interrupts"] = cast(JsonValue, interrupts)
        data["resume_schema"] = cast(JsonValue, resume_schema)
    return StreamEvent(data=data, event="suspend")


def stream_hitl_suspend_event(
    captured_interrupts: Iterable[Interrupt | dict[str, Any]],
    *,
    checkpoint_id: str | None,
    resumed: bool,
    messages: list[Message],
) -> StreamEvent:
    interrupts = _interrupt_snapshot(captured_interrupts)
    interrupt_ids = [str(interrupt["id"]) for interrupt in interrupts]
    return suspend_event(
        interrupts[0]["value"],
        interrupts=interrupts,
        resume_schema=_resume_schema(interrupt_ids),
        resumed=resumed,
        messages=messages,
        metadata={"checkpoint_id": checkpoint_id},
    )
