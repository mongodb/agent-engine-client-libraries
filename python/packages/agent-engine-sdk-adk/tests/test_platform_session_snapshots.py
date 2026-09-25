"""Durable ADK session snapshot translation tests for platform_session."""

from __future__ import annotations

import pytest
from google.adk.events import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions import Session
from google.genai import types

from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import WorkflowIdentity
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import (
    AttemptContext,
    BranchLineage,
)
from agent_engine_runner_shared.generated.workflow.v1.state_pb2 import (
    MESSAGE_ROLE_ASSISTANT,
    MESSAGE_ROLE_TOOL,
    MESSAGE_ROLE_USER,
    StateSnapshot,
)
from agent_engine_runner_shared.workflow.protojson import proto_struct_to_json
from agent_engine_sdk_adk.platform_session import (
    _ADAPTER_STATE_KEY,
    DurableADKStateError,
    PlatformSessionService,
    session_from_state_snapshot,
    session_to_state_snapshot,
)


def _event(
    *,
    event_id: str,
    author: str,
    role: str,
    parts: list[types.Part],
    state_delta: dict[str, object] | None = None,
) -> Event:
    return Event(
        id=event_id,
        invocation_id="invocation-1",
        author=author,
        actions=EventActions(state_delta=state_delta or {}),
        content=types.Content(role=role, parts=parts),
    )


def _session() -> Session:
    return Session(
        id="session-1",
        app_name="app",
        user_id="user-1",
        state={"turns": 2, "temp:scratch": "discard"},
        events=[
            _event(
                event_id="user-1",
                author="user",
                role="user",
                parts=[types.Part.from_text(text="weather")],
            ),
            _event(
                event_id="assistant-1",
                author="agent",
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id="call-1", name="weather", args={"city": "Paris"}
                        )
                    )
                ],
            ),
            _event(
                event_id="tool-1",
                author="agent",
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id="call-1", name="weather", response={"temp": 22}
                        )
                    )
                ],
            ),
            _event(
                event_id="assistant-2",
                author="agent",
                role="model",
                parts=[types.Part.from_text(text="22C")],
            ),
        ],
        last_update_time=123.0,
    )


def test_session_snapshot_round_trip_preserves_adk_state_events_and_messages() -> None:
    original = _session()
    snapshot = session_to_state_snapshot(original)

    envelope = proto_struct_to_json(snapshot.properties)[_ADAPTER_STATE_KEY]
    assert proto_struct_to_json(snapshot.replay_properties) == {"turns": 2}
    assert envelope["app_name"] == "app"
    assert "user_id" not in envelope
    assert envelope["session_id"] == "session-1"
    assert envelope["state"] == {"turns": 2}
    assert len(envelope["events"]) == 4
    assert [message.role for message in snapshot.messages] == [
        MESSAGE_ROLE_USER,
        MESSAGE_ROLE_ASSISTANT,
        MESSAGE_ROLE_TOOL,
        MESSAGE_ROLE_ASSISTANT,
    ]
    assert snapshot.messages[1].tool_calls[0]["id"] == "call-1"
    assert snapshot.messages[2].tool_call_id == "call-1"
    assert snapshot.message_encoding_version == 1
    assert not snapshot.messages[0].HasField("source_message")

    restored = session_from_state_snapshot(
        snapshot,
        app_name="app",
        user_id="user-1",
        session_id="session-1",
    )

    assert restored.state == {"turns": 2}
    assert [event.model_dump(mode="json") for event in restored.events] == [
        event.model_dump(mode="json") for event in original.events
    ]


@pytest.mark.parametrize("state", [{}, {"temp:scratch": "discard"}])
def test_session_snapshot_preserves_explicit_empty_replay_state(
    state: dict[str, object],
) -> None:
    session = _session()
    session.state = state

    snapshot = session_to_state_snapshot(session)
    restored = StateSnapshot.FromString(snapshot.SerializeToString())

    assert restored.HasField("replay_properties")
    assert proto_struct_to_json(restored.replay_properties) == {}
    envelope = proto_struct_to_json(restored.properties)[_ADAPTER_STATE_KEY]
    assert envelope["state"] == {}
    assert len(envelope["events"]) == 4


def test_fresh_session_requires_no_previous_snapshot() -> None:
    session = session_from_state_snapshot(
        None,
        app_name="app",
        user_id="user-1",
        session_id="session-1",
    )

    assert session.state == {}
    assert session.events == []


def test_previous_state_without_adk_envelope_fails() -> None:
    with pytest.raises(DurableADKStateError, match="adapter state"):
        session_from_state_snapshot(
            StateSnapshot(),
            app_name="app",
            user_id="user-1",
            session_id="session-1",
        )


def test_previous_state_for_another_session_fails() -> None:
    snapshot = session_to_state_snapshot(_session())

    with pytest.raises(DurableADKStateError, match="session identity"):
        session_from_state_snapshot(
            snapshot,
            app_name="app",
            user_id="user-1",
            session_id="foreign",
        )


def test_branch_attempt_restores_source_snapshot_under_new_session_identity() -> None:
    snapshot = session_to_state_snapshot(_session())
    attempt = AttemptContext(
        workflow_identity=WorkflowIdentity(
            session_id="branch-session",
            execution_id="branch-execution",
        ),
        branch_lineage=BranchLineage(
            source_workflow_identity=WorkflowIdentity(
                session_id="session-1",
                execution_id="execution-2",
            ),
            source_step_ordinal=1,
            source_state_hash="sha256:source",
        ),
    )
    attempt.previous_state.CopyFrom(snapshot)

    service = PlatformSessionService.from_attempt(
        attempt,
        app_name="app",
        user_id="user-1",
    )

    assert service.session.id == "branch-session"
    assert service.session.state == {"turns": 2}
    assert len(service.session.events) == 4


def test_branch_attempt_rejects_snapshot_from_wrong_source_session() -> None:
    snapshot = session_to_state_snapshot(_session())

    with pytest.raises(DurableADKStateError, match="session identity"):
        session_from_state_snapshot(
            snapshot,
            app_name="app",
            user_id="user-1",
            session_id="branch-session",
            branch_source_session_id="other-source",
        )


def test_previous_state_uses_current_user() -> None:
    original = _session()
    snapshot = session_to_state_snapshot(original)

    restored = session_from_state_snapshot(
        snapshot,
        app_name="app",
        user_id="user-2",
        session_id="session-1",
    )

    assert restored.user_id == "user-2"
    assert restored.state == {"turns": 2}
    assert [event.model_dump(mode="json") for event in restored.events] == [
        event.model_dump(mode="json") for event in original.events
    ]


def test_non_json_session_state_fails_before_commit() -> None:
    session = _session()
    session.state["bad"] = object()

    with pytest.raises(TypeError, match="JSON-safe"):
        session_to_state_snapshot(session)
