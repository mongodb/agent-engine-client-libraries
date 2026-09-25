"""Attempt-local ADK session scratch, analog of LangGraph PlatformCheckpointer.

OE owns durable conversation state. ADK's Runner still needs a
``BaseSessionService``; ``PlatformSessionService`` is that in-memory scratch
for one OE attempt. ADK is durable-only, so there is no native session DB
path — reconstruction always comes from OE ``previous_state``.
"""

from __future__ import annotations

from typing import Any, cast

from google.adk.events import Event
from google.adk.sessions import Session
from google.adk.sessions.base_session_service import (
    BaseSessionService,
    GetSessionConfig,
    ListSessionsResponse,
)

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
from agent_engine_sdk_adk.messages import content_to_platform_messages

__all__ = [
    "DurableADKStateError",
    "PlatformSessionService",
    "session_from_state_snapshot",
    "session_to_state_snapshot",
]

# Envelope key for adapter state inside persisted session snapshots. The
# string is part of the stored snapshot format, so changing it invalidates
# restore of sessions persisted with an older key.
_ADAPTER_STATE_KEY = "__agent_engine_adk__"
# Restore fails closed on any other key so leftover shapes cannot load as a
# current session.
_ADAPTER_ENVELOPE_KEYS = frozenset(
    {"app_name", "session_id", "state", "events", "last_update_time"}
)
_ROLE_TO_WORKFLOW = {
    "user": MESSAGE_ROLE_USER,
    "assistant": MESSAGE_ROLE_ASSISTANT,
    "tool": MESSAGE_ROLE_TOOL,
    "system": MESSAGE_ROLE_SYSTEM,
}


class DurableADKStateError(ValueError):
    """An OE state snapshot cannot seed the current ADK attempt."""


class PlatformSessionService(BaseSessionService):
    """In-memory ADK SessionService for one OE-issued attempt.

    Runner calls this as ``session_service``. Durable reconstruction comes from
    OE ``previous_state``, not from this store.
    """

    def __init__(self, session: Session, *, execution_id: str) -> None:
        if not execution_id:
            raise ValueError("platform ADK session requires an execution id")
        self._session = session
        self._execution_id = execution_id
        self._initial_update_time = session.last_update_time
        self._event_ordinal = 0

    @property
    def session(self) -> Session:
        return self._session

    def _require_identity(self, app_name: str, user_id: str, session_id: str) -> None:
        if (
            app_name != self._session.app_name
            or user_id != self._session.user_id
            or session_id != self._session.id
        ):
            raise ValueError(
                "platform ADK session access does not match the OE-issued identity"
            )

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Session:
        resolved_session_id = session_id or self._session.id
        self._require_identity(app_name, user_id, resolved_session_id)
        raise ValueError("platform ADK session already exists")

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: GetSessionConfig | None = None,
    ) -> Session | None:
        self._require_identity(app_name, user_id, session_id)
        selected = self._session.model_copy(deep=True)
        if config is not None:
            if config.num_recent_events is not None:
                if config.num_recent_events == 0:
                    selected.events = []
                else:
                    selected.events = selected.events[-config.num_recent_events :]
            if config.after_timestamp is not None:
                selected.events = [
                    event
                    for event in selected.events
                    if event.timestamp >= config.after_timestamp
                ]
        return selected

    async def list_sessions(
        self, *, app_name: str, user_id: str | None = None
    ) -> ListSessionsResponse:
        if app_name != self._session.app_name or (
            user_id is not None and user_id != self._session.user_id
        ):
            raise ValueError(
                "platform ADK session access does not match the OE-issued identity"
            )
        selected = self._session.model_copy(deep=True)
        selected.events = []
        return ListSessionsResponse(sessions=[selected])

    async def delete_session(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> None:
        self._require_identity(app_name, user_id, session_id)
        raise ValueError("platform ADK session cannot be deleted")

    async def get_user_state(self, *, app_name: str, user_id: str) -> dict[str, Any]:
        if app_name != self._session.app_name or user_id != self._session.user_id:
            raise ValueError(
                "platform ADK session access does not match the OE-issued identity"
            )
        prefix = "user:"
        return {
            key.removeprefix(prefix): value
            for key, value in self._session.state.items()
            if key.startswith(prefix)
        }

    async def append_event(self, session: Session, event: Event) -> Event:
        self._require_identity(session.app_name, session.user_id, session.id)
        if event.partial:
            return await super().append_event(session, event)

        self._event_ordinal += 1
        event.invocation_id = self._execution_id
        event.id = f"{self._execution_id}:event:{self._event_ordinal}"
        event.timestamp = self._initial_update_time + self._event_ordinal
        canonical_event = event.model_copy(deep=True)
        appended = await super().append_event(session, event)
        if session is self._session:
            canonical_appended = appended
        else:
            canonical_appended = await super().append_event(
                self._session, canonical_event
            )
        self._session.last_update_time = canonical_appended.timestamp
        return appended

    @classmethod
    def from_attempt(
        cls,
        attempt: AttemptContext,
        *,
        app_name: str,
        user_id: str,
    ) -> PlatformSessionService:
        """Restore OE previous state into scratch for this attempt."""
        previous_state = (
            attempt.previous_state if attempt.HasField("previous_state") else None
        )
        branch_source_session_id = None
        if attempt.HasField("branch_lineage"):
            branch_source_session_id = (
                attempt.branch_lineage.source_workflow_identity.session_id
            )
        session = session_from_state_snapshot(
            previous_state,
            app_name=app_name,
            user_id=user_id,
            session_id=attempt.workflow_identity.session_id,
            branch_source_session_id=branch_source_session_id,
        )
        return cls(
            session,
            execution_id=attempt.workflow_identity.execution_id,
        )


def _workflow_messages_for_event(event: Event) -> list[WorkflowMessage]:
    if event.partial or event.content is None:
        return []
    platform_messages = content_to_platform_messages(event.content)
    result: list[WorkflowMessage] = []
    for index, message in enumerate(platform_messages):
        workflow_message = WorkflowMessage(
            role=_ROLE_TO_WORKFLOW[message.role],
            content=json_to_proto_value(message.content),
        )
        for tool_call in message.tool_calls or []:
            workflow_message.tool_calls.append(
                json_to_proto_struct(
                    tool_call.model_dump(mode="json", exclude_none=True)
                )
            )
        if message.tool_call_id is not None:
            workflow_message.tool_call_id = message.tool_call_id
        if message.name is not None:
            workflow_message.name = message.name
        if event.id:
            workflow_message.id = event.id if index == 0 else f"{event.id}-{index}"
        result.append(workflow_message)
    return result


def session_to_state_snapshot(session: Session) -> StateSnapshot:
    """Encode complete ADK reconstruction state and portable messages."""
    durable_state = {
        key: value
        for key, value in session.state.items()
        if not key.startswith("temp:")
    }
    envelope = {
        "app_name": session.app_name,
        "session_id": session.id,
        "state": durable_state,
        "events": [
            event.model_dump(mode="json")
            for event in session.events
            if not event.partial
        ],
        "last_update_time": session.last_update_time,
    }
    snapshot = StateSnapshot(
        properties=json_to_proto_struct({_ADAPTER_STATE_KEY: envelope}),
        replay_properties=json_to_proto_struct(durable_state),
        message_encoding_version=1,
    )
    for event in session.events:
        snapshot.messages.extend(_workflow_messages_for_event(event))
    return snapshot


def _require_dict(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DurableADKStateError(
            f"previous ADK adapter state {field} must be an object"
        )
    return cast(dict[str, Any], value)


def session_from_state_snapshot(
    snapshot: StateSnapshot | None,
    *,
    app_name: str,
    user_id: str,
    session_id: str,
    branch_source_session_id: str | None = None,
) -> Session:
    """Create one attempt-local ADK session from OE-issued previous state."""
    if snapshot is None:
        return Session(id=session_id, app_name=app_name, user_id=user_id)

    properties = proto_struct_to_json(snapshot.properties)
    envelope = properties.get(_ADAPTER_STATE_KEY)
    if not isinstance(envelope, dict):
        raise DurableADKStateError("previous state is missing ADK adapter state")
    envelope = cast(dict[str, Any], envelope)
    unknown_keys = set(envelope) - _ADAPTER_ENVELOPE_KEYS
    if unknown_keys:
        raise DurableADKStateError(
            "previous ADK adapter state has an unsupported envelope shape"
        )
    # user_id identifies the current caller, not the project-scoped conversation.
    expected_snapshot_session_id = branch_source_session_id or session_id
    expected_identity = (app_name, expected_snapshot_session_id)
    actual_identity = (
        envelope.get("app_name"),
        envelope.get("session_id"),
    )
    if actual_identity != expected_identity:
        raise DurableADKStateError(
            "previous ADK adapter state has a conflicting session identity"
        )

    state = _require_dict(envelope.get("state"), "state")
    events_payload = envelope.get("events")
    if not isinstance(events_payload, list):
        raise DurableADKStateError("previous ADK adapter state events must be an array")
    event_values = cast(list[Any], events_payload)
    try:
        events = [Event.model_validate(value) for value in event_values]
    except (TypeError, ValueError) as error:
        raise DurableADKStateError(
            "previous ADK adapter state contains an invalid event"
        ) from error
    last_update_time = envelope.get("last_update_time", 0.0)
    if not isinstance(last_update_time, (int, float)):
        raise DurableADKStateError(
            "previous ADK adapter state last_update_time must be numeric"
        )
    return Session(
        id=session_id,
        app_name=app_name,
        user_id=user_id,
        state=state,
        events=events,
        last_update_time=float(last_update_time),
    )
