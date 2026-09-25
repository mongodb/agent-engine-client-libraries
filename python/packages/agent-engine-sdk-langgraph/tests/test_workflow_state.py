from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from agent_engine_runner_shared.generated.workflow.v1.state_pb2 import (
    MESSAGE_ROLE_ASSISTANT,
    MESSAGE_ROLE_USER,
    StateSnapshot,
    WorkflowMessage,
)
from agent_engine_runner_shared.workflow.protojson import (
    json_to_proto_struct,
    json_to_proto_value,
    proto_struct_to_json,
    proto_value_to_json,
)

from agent_engine_sdk_langgraph.messages import (
    lc_to_workflow_message,
    workflow_to_lc_message,
)
from agent_engine_sdk_langgraph.workflow_state import (
    channel_values_to_state_snapshot,
    state_snapshot_to_channel_values,
)


def test_workflow_message_contract_requires_explicit_field_review() -> None:
    assert set(StateSnapshot.DESCRIPTOR.fields_by_name) == {
        "properties",
        "messages",
        "message_encoding_version",
        "replay_properties",
    }
    assert set(WorkflowMessage.DESCRIPTOR.fields_by_name) == {
        "role",
        "content",
        "tool_calls",
        "tool_call_id",
        "name",
        "id",
        "source_message",
        "artifacts",
        "additional_kwargs",
        "response_metadata",
        "is_error",
        "tool_artifact",
        "platform_artifacts",
        "invalid_tool_calls",
    }


def test_workflow_message_round_trip_preserves_platform_message_fields() -> None:
    original = AIMessage(
        content=[
            {"type": "text", "text": "first"},
            {
                "type": "image",
                "url": "https://example.test/chart.png",
                "mime_type": "image/png",
            },
            {
                "type": "document",
                "url": "https://example.test/report.pdf",
                "filename": "report.pdf",
            },
        ],
        id="message-1",
        name="assistant",
        additional_kwargs={
            "artifacts": [
                {
                    "id": "chart-1",
                    "kind": "chart",
                    "title": "Claim totals",
                    "data": [{"label": "approved", "value": 7}],
                }
            ],
            "provider_extension": {
                "nullable": None,
                "count": 42,
                "score": 0.75,
            },
        },
        response_metadata={"model": "future-model", "finish_reason": "tool_use"},
    )

    encoded = lc_to_workflow_message(original)
    restored = workflow_to_lc_message(encoded)

    assert isinstance(restored, AIMessage)
    assert restored.model_dump() == original.model_dump()
    assert encoded.role == MESSAGE_ROLE_ASSISTANT
    assert proto_value_to_json(encoded.content) == original.content
    assert encoded.id == "message-1"
    assert encoded.name == "assistant"
    assert [
        proto_struct_to_json(artifact) for artifact in encoded.platform_artifacts
    ] == [
        {
            "id": "chart-1",
            "kind": "chart",
            "title": "Claim totals",
            "data": [{"label": "approved", "value": 7}],
        }
    ]
    assert proto_struct_to_json(encoded.additional_kwargs) == {
        "provider_extension": {"nullable": None, "count": 42, "score": 0.75}
    }
    assert proto_struct_to_json(encoded.response_metadata) == {
        "model": "future-model",
        "finish_reason": "tool_use",
    }
    assert not encoded.HasField("source_message")


def test_legacy_source_message_is_readable_but_not_written_for_new_state() -> None:
    original = AIMessage(
        content="legacy",
        id="message-legacy",
        additional_kwargs={"claim_evidence_stage": "verified"},
        response_metadata={"model": "legacy-model"},
    )

    legacy = lc_to_workflow_message(original, include_legacy_source=True)
    restored = workflow_to_lc_message(legacy)
    compact = lc_to_workflow_message(restored)

    assert legacy.HasField("source_message")
    assert restored.model_dump() == original.model_dump()
    assert not compact.HasField("source_message")
    assert proto_struct_to_json(compact.additional_kwargs) == {
        "claim_evidence_stage": "verified"
    }
    assert proto_struct_to_json(compact.response_metadata) == {"model": "legacy-model"}


def test_state_snapshot_round_trip_preserves_all_json_shapes_and_messages() -> None:
    messages = [
        HumanMessage(content="hello", id="message-1"),
        AIMessage(
            content=[
                {"type": "text", "text": "plain"},
                {
                    "type": "image",
                    "url": "https://example.test/image.png",
                },
            ],
            tool_calls=[
                {
                    "name": "search",
                    "args": {"limit": 10},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
            id="message-2",
        ),
        ToolMessage(
            content=[{"type": "json", "rows": [1, 2]}],
            tool_call_id="call-1",
            id="message-3",
            status="error",
            artifact={"source_rows": [1, 2]},
        ),
    ]
    channel_values = {
        "messages": messages,
        "count": 42,
        "ratio": 2.5,
        "enabled": True,
        "nullable": None,
        "nested": {"items": ["one", {"two": 2}]},
        "__pregel_tasks": ["not durable application state"],
    }

    snapshot = channel_values_to_state_snapshot(channel_values)
    restored = state_snapshot_to_channel_values(snapshot)

    assert snapshot.message_encoding_version == 1
    assert restored["count"] == 42
    assert type(restored["count"]) is float
    assert restored["ratio"] == 2.5
    assert type(restored["ratio"]) is float
    assert restored["enabled"] is True
    assert restored["nullable"] is None
    assert restored["nested"] == {"items": ["one", {"two": 2}]}
    assert "__pregel_tasks" not in restored
    assert [message.model_dump() for message in restored["messages"]] == [
        message.model_dump() for message in messages
    ]


def test_workflow_message_round_trip_preserves_single_text_block_list() -> None:
    original = AIMessage(content=[{"type": "text", "text": "hello"}])

    restored = workflow_to_lc_message(lc_to_workflow_message(original))

    assert restored.content == original.content


def test_workflow_message_round_trip_preserves_invalid_tool_calls() -> None:
    original = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "name": "lookup",
                "args": '{"query":',
                "id": "call-invalid",
                "error": "Malformed arguments",
                "type": "invalid_tool_call",
            }
        ],
    )

    encoded = lc_to_workflow_message(original)
    restored = workflow_to_lc_message(encoded)

    assert [
        proto_struct_to_json(tool_call) for tool_call in encoded.invalid_tool_calls
    ] == original.invalid_tool_calls
    assert isinstance(restored, AIMessage)
    assert restored.invalid_tool_calls == original.invalid_tool_calls


def test_workflow_message_round_trip_preserves_opaque_platform_artifact() -> None:
    original = HumanMessage(
        content="see attachment",
        additional_kwargs={
            "artifacts": [{"type": "document", "name": "report.pdf", "pages": [1, 2]}]
        },
    )

    encoded = lc_to_workflow_message(original)
    restored = workflow_to_lc_message(encoded)

    assert [
        proto_struct_to_json(artifact) for artifact in encoded.platform_artifacts
    ] == [{"type": "document", "name": "report.pdf", "pages": [1, 2]}]
    assert restored.additional_kwargs == original.additional_kwargs


@pytest.mark.parametrize("artifacts", ["not-an-array", [{"valid": True}, "bad"]])
def test_workflow_message_rejects_malformed_platform_artifacts(
    artifacts: object,
) -> None:
    message = HumanMessage(
        content="see attachment",
        additional_kwargs={"artifacts": artifacts},
    )

    with pytest.raises(TypeError, match=r"additional_kwargs\.artifacts"):
        lc_to_workflow_message(message)


@pytest.mark.parametrize(
    "name",
    ["messages", "__pregel_tasks", "branch:agent:tools", "start:agent"],
)
def test_state_snapshot_rejects_reserved_properties(name: str) -> None:
    snapshot = StateSnapshot(properties={name: "forged"})

    with pytest.raises(
        ValueError,
        match=f'previous workflow state contains reserved LangGraph channel "{name}"',
    ):
        state_snapshot_to_channel_values(snapshot)


def test_workflow_message_common_envelope_is_independently_usable() -> None:
    message = WorkflowMessage(
        role=MESSAGE_ROLE_USER,
        content=json_to_proto_value("hello"),
        id="message-1",
    )
    message.platform_artifacts.append(
        json_to_proto_struct(
            {"id": "source-1", "kind": "source", "url": "https://example.test"}
        )
    )

    restored = workflow_to_lc_message(message)

    assert isinstance(restored, HumanMessage)
    assert restored.content == "hello"
    assert restored.id == "message-1"
    assert restored.additional_kwargs == {
        "artifacts": [
            {"id": "source-1", "kind": "source", "url": "https://example.test"}
        ]
    }


def test_explicit_message_role_is_authoritative_for_reconstruction() -> None:
    message = lc_to_workflow_message(AIMessage(content="hello"))
    message.role = MESSAGE_ROLE_USER

    restored = workflow_to_lc_message(message)

    assert isinstance(restored, HumanMessage)
    assert restored.content == "hello"


def test_tool_error_round_trip_uses_explicit_field() -> None:
    original = ToolMessage(
        content="lookup failed",
        tool_call_id="call-1",
        id="tool-1",
        status="error",
        artifact=["machine", {"readable": True}],
    )

    encoded = lc_to_workflow_message(original)
    restored = workflow_to_lc_message(encoded)

    assert encoded.is_error is True
    assert proto_value_to_json(encoded.tool_artifact) == [
        "machine",
        {"readable": True},
    ]
    assert isinstance(restored, ToolMessage)
    assert restored.status == "error"
    assert restored.model_dump() == original.model_dump()


def test_tool_artifact_normalizes_explicit_null_to_absence() -> None:
    with_null = lc_to_workflow_message(
        ToolMessage(
            content="no machine-readable result",
            tool_call_id="call-1",
            artifact=None,
        )
    )
    without_artifact = lc_to_workflow_message(
        ToolMessage(content="no machine-readable result", tool_call_id="call-1")
    )

    assert with_null == without_artifact
    assert not with_null.HasField("tool_artifact")
    assert not without_artifact.HasField("tool_artifact")


def test_checkpoint_round_trip_preserves_canonical_durable_state() -> None:
    channel_values = {
        "messages": [
            HumanMessage(
                content=[{"type": "text", "text": "hello"}],
                id="human-1",
                name="customer",
                additional_kwargs={"nullable": None},
                response_metadata={"source": "test"},
            ),
            SystemMessage(content="follow policy", id="system-1"),
            AIMessage(
                content="",
                id="assistant-1",
                name="assistant",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {"policy": "POL-1"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
                invalid_tool_calls=[
                    {
                        "name": "search",
                        "args": '{"query":',
                        "id": "call-invalid",
                        "error": "Malformed arguments",
                        "type": "invalid_tool_call",
                    }
                ],
                additional_kwargs={"provider_extension": {"enabled": True}},
                response_metadata={"finish_reason": "tool_calls"},
            ),
            ToolMessage(
                content="not found",
                tool_call_id="call-1",
                id="tool-1",
                name="lookup",
                status="error",
            ),
            ToolMessage(
                content="no machine-readable result",
                tool_call_id="call-2",
                artifact=None,
            ),
            ToolMessage(
                content="chart generated",
                tool_call_id="call-3",
                artifact={"series": [1, None, 3]},
                additional_kwargs={"artifacts": [{"id": "chart-1", "kind": "chart"}]},
            ),
        ],
        "count": 3,
        "nullable": None,
        "nested": {"items": ["one", {"two": 2}]},
    }
    serializer = JsonPlusSerializer()
    checkpointed = serializer.loads_typed(serializer.dumps_typed(channel_values))

    assert channel_values_to_state_snapshot(
        checkpointed
    ) == channel_values_to_state_snapshot(channel_values)
