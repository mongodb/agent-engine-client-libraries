"""Cross-language ProtoJSON profile tests for workflow.v1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from google.protobuf.json_format import MessageToDict, Parse, ParseError
from google.protobuf.message_factory import GetMessageClass

from agent_engine_runner_shared.generated.workflow.v1 import (
    activity_pb2,
    common_pb2,
    runtime_pb2,
    state_pb2,
)
from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import ActivityCommand
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext

_FIXTURE_DIR = Path(__file__).parents[8] / "proto" / "workflow" / "v1" / "testdata" / "protojson"
_WIRE_MESSAGES = _FIXTURE_DIR / "wire_messages.json"
_WIRE_ENUMS = _FIXTURE_DIR / "wire_enums.json"
_WIRE_MODULES = (
    activity_pb2,
    common_pb2,
    runtime_pb2,
    state_pb2,
)


def test_every_wire_message_has_populated_protojson_coverage() -> None:
    expected = json.loads(_WIRE_MESSAGES.read_text())
    assert isinstance(expected, dict)
    descriptors = sorted(
        (
            descriptor
            for module in _WIRE_MODULES
            for descriptor in module.DESCRIPTOR.message_types_by_name.values()
        ),
        key=lambda descriptor: descriptor.full_name,
    )

    assert [descriptor.full_name for descriptor in descriptors] == sorted(expected)
    for descriptor in descriptors:
        message_type = GetMessageClass(descriptor)
        fixtures = expected[descriptor.full_name]
        assert fixtures
        assert {name for fixture in fixtures for name in fixture} == {
            field.name for field in descriptor.fields
        }
        for fixture in fixtures:
            parsed = Parse(json.dumps(fixture), message_type())
            assert MessageToDict(parsed, preserving_proto_field_name=True) == fixture
        assert MessageToDict(message_type(), preserving_proto_field_name=True) == {}


def test_attempt_context_protojson_profile() -> None:
    expected = json.loads(_WIRE_MESSAGES.read_text())["mongodb.agentic.workflow.v1.AttemptContext"][
        0
    ]
    message = AttemptContext(
        attempt_id="attempt-1",
        fencing_token=7,
        owner_id="aer-1",
        replay_mode=True,
        workflow_identity=common_pb2.WorkflowIdentity(
            tenant_scope=common_pb2.TenantScope(
                org_id="org-1",
                project_id="project-1",
                workspace_id="workspace-1",
            ),
            session_id="session-1",
            execution_id="execution-1",
        ),
        declaration=runtime_pb2.WorkflowDeclaration(
            workflow_name="insurance-agent",
            workflow_version="1",
            adapter_name="langgraph",
            adapter_version="1",
            memory_enabled=True,
        ),
        heartbeat_interval_ms=5000,
        previous_state=state_pb2.StateSnapshot(properties={"claim_count": 1}),
        branch_lineage=runtime_pb2.BranchLineage(
            source_workflow_identity=common_pb2.WorkflowIdentity(
                tenant_scope=common_pb2.TenantScope(
                    org_id="org-1",
                    project_id="project-1",
                    workspace_id="workspace-1",
                ),
                session_id="source-session",
                execution_id="source-execution",
            ),
            source_step_ordinal=4,
            source_state_hash="sha256:source",
        ),
    )

    assert MessageToDict(message, preserving_proto_field_name=True) == expected

    parsed = Parse(
        json.dumps(
            {
                "attemptId": "attempt-1",
                "fencingToken": "7",
                "ownerId": "aer-1",
                "replayMode": True,
            }
        ),
        AttemptContext(),
    )
    assert parsed.attempt_id == "attempt-1"
    assert parsed.fencing_token == 7
    assert MessageToDict(AttemptContext(), preserving_proto_field_name=True) == {}


def test_every_wire_enum_has_protojson_coverage() -> None:
    expected = json.loads(_WIRE_ENUMS.read_text())
    actual = {
        descriptor.full_name: [value.name for value in descriptor.values]
        for module in _WIRE_MODULES
        for descriptor in module.DESCRIPTOR.enum_types_by_name.values()
    }
    assert actual == expected


def test_protojson_unknown_field_policy() -> None:
    with pytest.raises(ParseError):
        Parse(
            '{"activity_name":"tool","future_field":true}',
            ActivityCommand(),
            ignore_unknown_fields=False,
        )

    parsed = Parse(
        '{"attempt_id":"attempt-1","future_field":true}',
        AttemptContext(),
        ignore_unknown_fields=True,
    )
    assert parsed.attempt_id == "attempt-1"
