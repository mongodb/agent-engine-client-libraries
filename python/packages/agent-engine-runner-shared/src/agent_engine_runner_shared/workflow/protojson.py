"""Pinned ProtoJSON profile helpers for `workflow.v1` wire messages."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from google.protobuf import json_format
from google.protobuf.message import Message
from google.protobuf.struct_pb2 import Struct, Value

__all__ = [
    "WorkflowWireError",
    "encode_protojson",
    "json_to_proto_struct",
    "json_to_proto_value",
    "parse_command",
    "parse_read",
    "proto_struct_to_json",
    "proto_value_to_json",
]

MessageT = TypeVar("MessageT", bound=Message)
_MAX_EXACT_PROTOJSON_INTEGER = (1 << 53) - 1


class WorkflowWireError(ValueError):
    """A workflow wire payload violates the pinned ProtoJSON profile."""


def encode_protojson(message: Message) -> str:
    """Serialize one wire message with proto field names and defaults omitted."""
    return json.dumps(
        json_format.MessageToDict(message, preserving_proto_field_name=True),
        separators=(",", ":"),
    )


def json_to_proto_value(value: Any) -> Value:
    """Encode one JSON-shaped Python value as ``google.protobuf.Value``."""
    return _parse_json_value(value, Value())


def json_to_proto_struct(value: Mapping[str, Any]) -> Struct:
    """Encode one JSON object as ``google.protobuf.Struct``."""
    return _parse_json_value(value, Struct())


def proto_value_to_json(value: Value) -> Any:
    """Decode ``google.protobuf.Value`` to its JSON-shaped Python value."""
    return json_format.MessageToDict(value)


def proto_struct_to_json(value: Struct) -> dict[str, Any]:
    """Decode ``google.protobuf.Struct`` to a JSON-shaped Python object."""
    return cast(dict[str, Any], json_format.MessageToDict(value))


def parse_command(payload: str | bytes, message: MessageT) -> MessageT:
    """Parse a command payload strictly; unknown fields are rejected.

    Not re-exported from the package: the runtime currently only sends
    commands, so this strict leg of the profile exists for consumers that
    receive OE-issued commands.
    """
    return _parse(payload, message, ignore_unknown_fields=False)


def parse_read(payload: str | bytes, message: MessageT) -> MessageT:
    """Parse a read-response payload; unknown fields are tolerated."""
    return _parse(payload, message, ignore_unknown_fields=True)


def _parse(payload: str | bytes, message: MessageT, *, ignore_unknown_fields: bool) -> MessageT:
    try:
        text = payload.decode() if isinstance(payload, bytes) else payload
        return json_format.Parse(text, message, ignore_unknown_fields=ignore_unknown_fields)
    except (json_format.ParseError, UnicodeDecodeError) as error:
        raise WorkflowWireError(str(error)) from error


def _parse_json_value(value: Any, message: MessageT) -> MessageT:
    """Use one strict JSON profile for generic workflow values.

    The JSON round trip rejects Python-only values and non-finite floats before
    they can enter a cross-language workflow message.
    """
    try:
        encoded = json.dumps(value, allow_nan=False)
        _reject_inexact_integers(json.loads(encoded))
        return json_format.Parse(encoded, message)
    except (TypeError, ValueError, json_format.ParseError) as error:
        raise TypeError("durable workflow values must be JSON-safe") from error


def _reject_inexact_integers(value: Any) -> None:
    """Reject integers that ``google.protobuf.Value`` cannot preserve exactly."""
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, int):
        if abs(value) > _MAX_EXACT_PROTOJSON_INTEGER:
            raise ValueError("integer exceeds the exact ProtoJSON range")
        return
    if isinstance(value, list):
        for item in value:
            _reject_inexact_integers(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _reject_inexact_integers(item)
