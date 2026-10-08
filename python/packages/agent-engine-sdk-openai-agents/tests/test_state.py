from __future__ import annotations

from typing import Any

import pytest
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    WorkflowIdentity,
)
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import (
    AttemptContext,
    BranchLineage,
)
from agent_engine_runner_shared.generated.workflow.v1.state_pb2 import (
    MESSAGE_ROLE_ASSISTANT,
    MESSAGE_ROLE_SYSTEM,
    MESSAGE_ROLE_TOOL,
    MESSAGE_ROLE_USER,
    StateSnapshot,
)
from agent_engine_runner_shared.workflow.protojson import (
    json_to_proto_struct,
    proto_struct_to_json,
    proto_value_to_json,
)
from agent_engine_sdk_openai_agents import DurableOpenAIAgentsStateError
from agent_engine_sdk_openai_agents.items import validate_items
from agent_engine_sdk_openai_agents.state import (
    STATE_KEY,
    decode_active_agent,
    decode_state,
    encode_state,
)

APP = "support"
SESSION = "session-1"


def _assistant(text: str) -> dict[str, Any]:
    return ResponseOutputMessage(
        id="msg-1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    ).model_dump(mode="json", exclude_none=True)


def _call(call_id: str) -> dict[str, Any]:
    return ResponseFunctionToolCall(
        id=f"fc-{call_id}",
        call_id=call_id,
        type="function_call",
        name="lookup_order",
        arguments='{"order_id": "A1"}',
        status="completed",
    ).model_dump(mode="json", exclude_none=True)


def _output(call_id: str, output: object = "shipped") -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _turn() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "Where is order A1?"},
        _call("call-1"),
        _output("call-1"),
        _assistant("It shipped."),
    ]


def _attempt(
    snapshot: StateSnapshot | None,
    *,
    session_id: str = SESSION,
    branch_source_session_id: str | None = None,
) -> AttemptContext:
    attempt = AttemptContext(workflow_identity=WorkflowIdentity(session_id=session_id))
    if snapshot is not None:
        attempt.previous_state.CopyFrom(snapshot)
    if branch_source_session_id is not None:
        attempt.branch_lineage.CopyFrom(
            BranchLineage(
                source_workflow_identity=WorkflowIdentity(
                    session_id=branch_source_session_id
                )
            )
        )
    return attempt


def _snapshot(
    items: list[dict[str, Any]],
    session_id: str = SESSION,
    active_agent: str = "support",
) -> StateSnapshot:
    return encode_state(
        validate_items(items),
        app_name=APP,
        session_id=session_id,
        active_agent=active_agent,
    )


def _with_envelope(*, without: str | None = None, **overrides: object) -> StateSnapshot:
    envelope = proto_struct_to_json(_snapshot(_turn()).properties)[STATE_KEY]
    assert isinstance(envelope, dict)
    changed = {**envelope, **overrides}
    changed.pop(without, None)
    return StateSnapshot(properties=json_to_proto_struct({STATE_KEY: changed}))


def test_first_turn_starts_from_empty_history() -> None:
    assert decode_state(_attempt(None), app_name=APP) == []


def test_later_turn_reconstructs_the_committed_runner_input() -> None:
    restored = decode_state(_attempt(_snapshot(_turn())), app_name=APP)

    assert [item.raw for item in restored] == _turn()


def test_committed_messages_are_an_explicit_field_projection() -> None:
    snapshot = _snapshot(_turn())

    assert snapshot.message_encoding_version == 1
    assert [message.role for message in snapshot.messages] == [
        MESSAGE_ROLE_USER,
        MESSAGE_ROLE_ASSISTANT,
        MESSAGE_ROLE_TOOL,
        MESSAGE_ROLE_ASSISTANT,
    ]
    call, output, answer = snapshot.messages[1:]
    assert proto_struct_to_json(call.tool_calls[0]) == {
        "id": "call-1",
        "name": "lookup_order",
        "args": {"order_id": "A1"},
        "type": "function",
    }
    assert output.tool_call_id == "call-1"
    assert proto_value_to_json(output.content) == "shipped"
    assert proto_value_to_json(answer.content) == "It shipped."
    assert answer.id == "msg-1"
    assert not any(message.HasField("source_message") for message in snapshot.messages)


def test_a_batch_of_calls_projects_as_one_assistant_turn() -> None:
    items = [
        {"role": "user", "content": "Where are A1 and A2?"},
        _call("call-1"),
        _call("call-2"),
        _output("call-2", "A2 shipped"),
        _output("call-1", "A1 shipped"),
        {"role": "assistant", "content": "Both shipped."},
    ]

    snapshot = _snapshot(items)

    # Chat providers need every output to follow the one message that made
    # the calls, so the batch cannot become two assistant messages.
    assert [message.role for message in snapshot.messages] == [
        MESSAGE_ROLE_USER,
        MESSAGE_ROLE_ASSISTANT,
        MESSAGE_ROLE_TOOL,
        MESSAGE_ROLE_TOOL,
        MESSAGE_ROLE_ASSISTANT,
    ]
    calls = snapshot.messages[1].tool_calls
    assert [proto_struct_to_json(call)["id"] for call in calls] == ["call-1", "call-2"]
    assert [message.tool_call_id for message in snapshot.messages[2:4]] == [
        "call-2",
        "call-1",
    ]
    assert [
        item.raw for item in decode_state(_attempt(snapshot), app_name=APP)
    ] == items


def test_branch_continues_from_its_source_session_snapshot() -> None:
    snapshot = _snapshot(_turn(), session_id="source-session")
    attempt = _attempt(
        snapshot, session_id="branch-session", branch_source_session_id="source-session"
    )

    assert len(decode_state(attempt, app_name=APP)) == 4


@pytest.mark.parametrize(
    ("attempt", "app_name"),
    [
        pytest.param(_attempt(_snapshot(_turn())), "other-app", id="other-app"),
        pytest.param(
            _attempt(_snapshot(_turn()), session_id="other"), APP, id="other-session"
        ),
        pytest.param(
            _attempt(
                _snapshot(_turn(), session_id="source"),
                session_id="branch",
                branch_source_session_id="another-source",
            ),
            APP,
            id="other-branch-source",
        ),
        pytest.param(_attempt(StateSnapshot()), APP, id="foreign-snapshot"),
        pytest.param(_attempt(_with_envelope(version=2)), APP, id="unknown-version"),
        pytest.param(
            _attempt(_with_envelope(without="active_agent")),
            APP,
            id="without-active-agent",
        ),
        pytest.param(_attempt(_with_envelope(version=True)), APP, id="boolean-version"),
        pytest.param(_attempt(_with_envelope(extra=True)), APP, id="unknown-key"),
        pytest.param(_attempt(_with_envelope(items={})), APP, id="items-not-array"),
    ],
)
def test_state_from_another_owner_or_shape_is_rejected(
    attempt: AttemptContext, app_name: str
) -> None:
    with pytest.raises(DurableOpenAIAgentsStateError):
        decode_state(attempt, app_name=app_name)


def test_the_active_agent_is_committed_with_the_conversation() -> None:
    attempt = _attempt(_snapshot(_turn(), active_agent="billing"))

    assert decode_active_agent(attempt, app_name=APP) == "billing"
    assert decode_active_agent(_attempt(None), app_name=APP) is None


@pytest.mark.parametrize("active_agent", ["", 7, None])
def test_a_stored_active_agent_must_be_a_name(active_agent: object) -> None:
    attempt = _attempt(_with_envelope(active_agent=active_agent))

    with pytest.raises(DurableOpenAIAgentsStateError, match="active agent"):
        decode_active_agent(attempt, app_name=APP)


def test_stored_items_are_revalidated_on_decode() -> None:
    attempt = _attempt(_with_envelope(items=[_call("call-1")]))

    with pytest.raises(DurableOpenAIAgentsStateError, match="has no output"):
        decode_state(attempt, app_name=APP)


def test_developer_instructions_project_as_system_text() -> None:
    (item,) = validate_items([{"role": "developer", "content": "Be brief."}])

    assert item.raw["role"] == "developer"
    snapshot = encode_state(
        [item], app_name=APP, session_id=SESSION, active_agent="support"
    )
    assert snapshot.messages[0].role == MESSAGE_ROLE_SYSTEM


@pytest.mark.parametrize(
    ("items", "match"),
    [
        ([_call("c1")], "has no output"),
        ([_call("c1"), {"role": "user", "content": "hi"}], "does not answer"),
        ([_call("c1"), _output("c2")], "without its call"),
        ([_output("c1")], "without its call"),
        (
            [_call("c1"), _output("c1"), _call("c1"), _output("c1")],
            "repeats call_id",
        ),
        # A batch's calls are all answered before anything else follows.
        ([_call("c1"), _call("c2"), _output("c1")], "has no output"),
        (
            [_call("c1"), _call("c2"), _output("c1"), _call("c3"), _output("c2")],
            "new call before",
        ),
        (
            [_call("c1"), _call("c2"), _output("c1"), {"role": "user", "content": "x"}],
            "does not answer",
        ),
        ([_call("c1"), _call("c2"), _output("c1"), _output("c1")], "without its call"),
        ([_call("c1"), _output("c1", [{"type": "input_text", "text": "x"}])], "text"),
        ([{**_call("c1"), "arguments": "{"}], "not valid JSON"),
        ([{**_call("c1"), "arguments": '{"x": NaN}'}], "cannot store"),
        ([{**_call("c1"), "arguments": '{"x": 9007199254740993}'}], "cannot store"),
        (
            [{"role": "user", "content": "x", "provider_data": {"x": float("inf")}}],
            "cannot store",
        ),
        ([{**_call("c1"), "call_id": ""}], "requires call_id"),
        ([{"role": "tool", "content": "x"}], "unsupported role"),
        ([{"role": ["user"], "content": "x"}], "unsupported role"),
        ([{"type": "message", "content": "x"}], "message but has no role"),
        (
            [{"role": "user", "content": [{"type": "input_image", "image_url": "u"}]}],
            "non-text",
        ),
        ([{"type": "web_search_call", "id": "w1"}], "unsupported type"),
        (["not an object"], "JSON object"),
        ([{"role": "user", "content": object()}], "not JSON"),
    ],
)
def test_items_outside_the_text_and_batch_contract_are_rejected(
    items: list[Any], match: str
) -> None:
    with pytest.raises(DurableOpenAIAgentsStateError, match=match):
        validate_items(items)
