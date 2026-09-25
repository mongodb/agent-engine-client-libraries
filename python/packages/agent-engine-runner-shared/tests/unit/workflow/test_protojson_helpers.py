"""Tests for the shared workflow ProtoJSON encode/parse helpers."""

from __future__ import annotations

import json

import pytest
from google.protobuf.struct_pb2 import Struct, Value

from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import (
    AttemptContext,
    AttemptStartRequest,
    WorkflowDeclaration,
)
from agent_engine_runner_shared.workflow import WorkflowWireError
from agent_engine_runner_shared.workflow.protojson import (
    encode_protojson,
    json_to_proto_struct,
    json_to_proto_value,
    parse_command,
    parse_read,
    proto_struct_to_json,
    proto_value_to_json,
)


def test_encode_emits_snake_case_and_omits_defaults() -> None:
    request = AttemptStartRequest(
        owner_id="aer-1",
        declaration=WorkflowDeclaration(workflow_name="insurance"),
    )

    assert json.loads(encode_protojson(request)) == {
        "owner_id": "aer-1",
        "declaration": {"workflow_name": "insurance"},
    }


def test_encode_uses_string_int64() -> None:
    context = AttemptContext(attempt_id="attempt-1", fencing_token=7)

    assert json.loads(encode_protojson(context)) == {
        "attempt_id": "attempt-1",
        "fencing_token": "7",
    }


def test_parse_command_accepts_snake_case_and_lower_camel() -> None:
    snake = parse_command('{"owner_id":"aer-1"}', AttemptStartRequest())
    camel = parse_command('{"ownerId":"aer-1"}', AttemptStartRequest())

    assert snake.owner_id == "aer-1"
    assert camel.owner_id == "aer-1"


def test_parse_command_rejects_unknown_fields() -> None:
    with pytest.raises(WorkflowWireError):
        parse_command('{"owner_id":"aer-1","future_field":true}', AttemptStartRequest())


def test_parse_command_rejects_malformed_json() -> None:
    with pytest.raises(WorkflowWireError):
        parse_command("not json", AttemptStartRequest())


def test_parse_read_ignores_unknown_fields() -> None:
    parsed = parse_read('{"attempt_id":"attempt-1","future_field":true}', AttemptContext())

    assert parsed.attempt_id == "attempt-1"


def test_parse_read_rejects_malformed_values() -> None:
    with pytest.raises(WorkflowWireError):
        parse_read('{"fencing_token":"not-a-number"}', AttemptContext())


def test_generic_json_values_round_trip_through_protobuf() -> None:
    value = {"nested": ["text", 3, True, None]}

    encoded_value = json_to_proto_value(value)
    encoded_struct = json_to_proto_struct(value)

    assert isinstance(encoded_value, Value)
    assert isinstance(encoded_struct, Struct)
    assert proto_value_to_json(encoded_value) == value
    assert proto_struct_to_json(encoded_struct) == value


def test_generic_json_values_reject_non_finite_numbers() -> None:
    with pytest.raises(TypeError, match="JSON-safe"):
        json_to_proto_value(float("nan"))


@pytest.mark.parametrize("value", [2**53, -(2**53), {"nested": [2**53]}])
def test_generic_json_values_reject_inexact_integers(value: object) -> None:
    with pytest.raises(TypeError, match="JSON-safe"):
        json_to_proto_value(value)


@pytest.mark.parametrize("value", [(2**53) - 1, -((2**53) - 1)])
def test_generic_json_values_accept_exact_integers(value: int) -> None:
    assert proto_value_to_json(json_to_proto_value(value)) == value
