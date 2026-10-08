"""Native OpenAI Agents input items carried as durable state.

The Runner input list is the adapter's reconstruction state. It is validated in
one pass whenever it crosses the durable boundary, and the typed result feeds
every projection so no caller re-parses raw items.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from agent_engine_sdk import LLMToolCall, Message
from agent_engine_sdk.models import Role
from pydantic import JsonValue, TypeAdapter, ValidationError

from agent_engine_runner_shared.workflow.protojson import json_to_proto_value
from agent_engine_sdk_openai_agents.errors import DurableOpenAIAgentsStateError

__all__ = [
    "FunctionCall",
    "FunctionOutput",
    "NativeItem",
    "TextMessage",
    "to_platform_messages",
    "validate_items",
]

_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_TEXT_PART_TYPES = frozenset({"input_text", "output_text"})
# The platform has no developer role; developer instructions are system text.
_ROLES: dict[str, Role] = {
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "developer": "system",
}


@dataclass(frozen=True)
class TextMessage:
    raw: dict[str, JsonValue]
    role: Role
    text: str
    item_id: str | None


@dataclass(frozen=True)
class FunctionCall:
    raw: dict[str, JsonValue]
    call_id: str
    name: str
    arguments: JsonValue


@dataclass(frozen=True)
class FunctionOutput:
    raw: dict[str, JsonValue]
    call_id: str
    output: str


NativeItem = TextMessage | FunctionCall | FunctionOutput


def validate_items(values: Iterable[object]) -> list[NativeItem]:
    """Validate a Runner input list and return its typed items.

    Function calls come in batches: the calls one model response made, then
    exactly one output for each before any other item. Tool identity comes
    from the call ids, so each call id appears once in the whole list.
    """
    items: list[NativeItem] = []
    seen_call_ids: set[str] = set()
    unanswered: dict[str, FunctionCall] = {}
    answering = False
    for index, value in enumerate(values):
        item = _parse_item(_json_object(value), index)
        if isinstance(item, FunctionCall):
            if answering:
                raise DurableOpenAIAgentsStateError(
                    f"item {index} starts a new call before every call in the "
                    "batch has its output"
                )
            if item.call_id in seen_call_ids:
                raise DurableOpenAIAgentsStateError(
                    f"item {index} repeats call_id {item.call_id!r}"
                )
            seen_call_ids.add(item.call_id)
            unanswered[item.call_id] = item
        elif isinstance(item, FunctionOutput):
            if unanswered.pop(item.call_id, None) is None:
                raise DurableOpenAIAgentsStateError(
                    f"item {index} is a function output without its call"
                )
            answering = bool(unanswered)
        elif unanswered:
            raise DurableOpenAIAgentsStateError(
                f"item {index} does not answer function call {next(iter(unanswered))!r}"
            )
        items.append(item)
    if unanswered:
        raise DurableOpenAIAgentsStateError(
            f"function call {next(iter(unanswered))!r} has no output"
        )
    return items


def to_platform_messages(items: Iterable[NativeItem]) -> list[Message]:
    """Project validated items onto framework-neutral platform messages."""
    messages: list[Message] = []
    for item in items:
        if isinstance(item, TextMessage):
            messages.append(Message(role=item.role, content=item.text, id=item.item_id))
        elif isinstance(item, FunctionCall):
            call = LLMToolCall(
                id=item.call_id, name=item.name, args=item.arguments, type="function"
            )
            previous = messages[-1] if messages else None
            # A batch of calls is one assistant turn; chat providers require
            # every call's output to follow the message that made the calls.
            if (
                previous is not None
                and previous.role == "assistant"
                and previous.tool_calls
                and not previous.content
            ):
                previous.tool_calls.append(call)
            else:
                messages.append(
                    Message(role="assistant", content="", tool_calls=[call])
                )
        else:
            messages.append(
                Message(role="tool", content=item.output, tool_call_id=item.call_id)
            )
    return messages


def _json_object(value: object) -> dict[str, JsonValue]:
    try:
        detached = _JSON.validate_python(value)
    except ValidationError as error:
        raise DurableOpenAIAgentsStateError("item is not JSON") from error
    if not isinstance(detached, dict):
        raise DurableOpenAIAgentsStateError("item must be a JSON object")
    _require_storable(detached, "item")
    return detached


def _require_storable(value: JsonValue, what: str) -> None:
    """Fail at validation for values OE state cannot hold exactly.

    Python JSON accepts NaN, Infinity, and unbounded integers; the OE snapshot
    does not, and would otherwise reject them only when the turn settles.
    """
    try:
        json_to_proto_value(value)
    except TypeError as error:
        raise DurableOpenAIAgentsStateError(
            f"{what} contains a value OE state cannot store"
        ) from error


def _parse_item(raw: dict[str, JsonValue], index: int) -> NativeItem:
    item_type = raw.get("type")
    if "role" in raw and item_type in (None, "message"):
        return _parse_message(raw, index)
    if item_type == "function_call":
        return FunctionCall(
            raw=raw,
            call_id=_required_str(raw, "call_id", index),
            name=_required_str(raw, "name", index),
            arguments=_json_arguments(raw, index),
        )
    if item_type == "function_call_output":
        output = raw.get("output")
        if not isinstance(output, str):
            raise DurableOpenAIAgentsStateError(
                f"item {index} function output must be text"
            )
        return FunctionOutput(
            raw=raw, call_id=_required_str(raw, "call_id", index), output=output
        )
    if item_type == "message":
        raise DurableOpenAIAgentsStateError(
            f"item {index} is a message but has no role"
        )
    raise DurableOpenAIAgentsStateError(
        f"item {index} has unsupported type {item_type!r}"
    )


def _parse_message(raw: dict[str, JsonValue], index: int) -> TextMessage:
    raw_role = raw.get("role")
    role = _ROLES.get(raw_role) if isinstance(raw_role, str) else None
    if role is None:
        raise DurableOpenAIAgentsStateError(
            f"item {index} has unsupported role {raw_role!r}"
        )
    item_id = raw.get("id")
    return TextMessage(
        raw=raw,
        role=role,
        text=_message_text(raw.get("content"), index),
        item_id=item_id if isinstance(item_id, str) else None,
    )


def _message_text(content: JsonValue, index: int) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise DurableOpenAIAgentsStateError(f"item {index} content must be text")
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in _TEXT_PART_TYPES:
            raise DurableOpenAIAgentsStateError(
                f"item {index} contains a non-text content part"
            )
        text = part.get("text")
        if not isinstance(text, str):
            raise DurableOpenAIAgentsStateError(
                f"item {index} contains a text part without text"
            )
        parts.append(text)
    return "".join(parts)


def _required_str(raw: dict[str, JsonValue], key: str, index: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise DurableOpenAIAgentsStateError(f"item {index} requires {key}")
    return value


def _json_arguments(raw: dict[str, JsonValue], index: int) -> JsonValue:
    arguments = raw.get("arguments")
    if not isinstance(arguments, str):
        raise DurableOpenAIAgentsStateError(
            f"item {index} function arguments must be JSON text"
        )
    try:
        parsed = cast(JsonValue, json.loads(arguments))
    except json.JSONDecodeError as error:
        raise DurableOpenAIAgentsStateError(
            f"item {index} function arguments are not valid JSON"
        ) from error
    _require_storable(parsed, f"item {index} function arguments")
    return parsed
