"""Platform ADK session service tests."""

from __future__ import annotations

import pytest
from google.adk.events import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions import Session
from google.adk.sessions.base_session_service import GetSessionConfig
from google.genai import types

from agent_engine_sdk_adk.platform_session import (
    PlatformSessionService,
    session_to_state_snapshot,
)


def _session() -> Session:
    return Session(
        id="session-1",
        app_name="app",
        user_id="user-1",
        state={"count": 1},
        events=[],
    )


@pytest.mark.asyncio
async def test_append_event_updates_only_the_attempt_local_session() -> None:
    service = PlatformSessionService(_session(), execution_id="execution-1")
    session = await service.get_session(
        app_name="app", user_id="user-1", session_id="session-1"
    )
    assert session is not None
    event = Event(
        invocation_id="invocation-1",
        author="agent",
        actions=EventActions(state_delta={"count": 2, "temp:scratch": "value"}),
        content=types.Content(role="model", parts=[types.Part.from_text(text="done")]),
    )

    await service.append_event(session, event)

    assert service.session.state == {"count": 2, "temp:scratch": "value"}
    assert service.session.events == [event]
    assert event.invocation_id == "execution-1"
    assert event.id == "execution-1:event:1"


@pytest.mark.asyncio
async def test_get_session_applies_event_filters_without_mutating_source() -> None:
    session = _session()
    session.events = [
        Event(invocation_id="one", author="agent", timestamp=1.0),
        Event(invocation_id="two", author="agent", timestamp=2.0),
    ]
    service = PlatformSessionService(session, execution_id="execution-1")

    selected = await service.get_session(
        app_name="app",
        user_id="user-1",
        session_id="session-1",
        config=GetSessionConfig(num_recent_events=1),
    )

    assert selected is not None
    assert [event.invocation_id for event in selected.events] == ["two"]
    assert len(service.session.events) == 2


@pytest.mark.asyncio
async def test_append_to_filtered_session_preserves_canonical_history() -> None:
    session = _session()
    session.events = [
        Event(id="one", invocation_id="one", author="agent", timestamp=1.0),
        Event(id="two", invocation_id="two", author="agent", timestamp=2.0),
    ]
    session.last_update_time = 2.0
    service = PlatformSessionService(session, execution_id="execution-1")
    selected = await service.get_session(
        app_name="app",
        user_id="user-1",
        session_id="session-1",
        config=GetSessionConfig(num_recent_events=1),
    )
    assert selected is not None

    await service.append_event(
        selected,
        Event(
            invocation_id="random",
            author="agent",
            actions=EventActions(state_delta={"count": 2}),
            content=types.Content(
                role="model", parts=[types.Part.from_text(text="new")]
            ),
        ),
    )

    assert [event.id for event in selected.events] == [
        "two",
        "execution-1:event:1",
    ]
    assert [event.id for event in service.session.events] == [
        "one",
        "two",
        "execution-1:event:1",
    ]
    assert service.session.state["count"] == 2


@pytest.mark.asyncio
async def test_foreign_session_access_fails_explicitly() -> None:
    service = PlatformSessionService(_session(), execution_id="execution-1")

    with pytest.raises(ValueError, match="platform ADK session"):
        await service.get_session(
            app_name="app", user_id="user-1", session_id="foreign"
        )


@pytest.mark.asyncio
async def test_replayed_events_produce_identical_snapshots() -> None:
    previous = _session()
    previous.last_update_time = 10.0
    first = PlatformSessionService(
        previous.model_copy(deep=True), execution_id="execution-1"
    )
    retry = PlatformSessionService(
        previous.model_copy(deep=True), execution_id="execution-1"
    )

    async def append_events(
        service: PlatformSessionService,
        *,
        invocation_id: str,
        event_id: str,
        timestamp: float,
    ) -> None:
        session = await service.get_session(
            app_name="app", user_id="user-1", session_id="session-1"
        )
        assert session is not None
        await service.append_event(
            session,
            Event(
                id=f"{event_id}-partial",
                invocation_id=invocation_id,
                author="agent",
                partial=True,
                timestamp=timestamp,
            ),
        )
        await service.append_event(
            session,
            Event(
                id=event_id,
                invocation_id=invocation_id,
                author="agent",
                timestamp=timestamp + 1,
                content=types.Content(
                    role="model", parts=[types.Part.from_text(text="done")]
                ),
            ),
        )

    await append_events(
        first,
        invocation_id="random-invocation-a",
        event_id="random-event-a",
        timestamp=100.0,
    )
    await append_events(
        retry,
        invocation_id="random-invocation-b",
        event_id="random-event-b",
        timestamp=200.0,
    )

    first_event = first.session.events[0]
    assert first_event.invocation_id == "execution-1"
    assert first_event.id == "execution-1:event:1"
    assert first_event.timestamp == 11.0
    assert first.session.last_update_time == 11.0
    assert session_to_state_snapshot(first.session).SerializeToString(
        deterministic=True
    ) == session_to_state_snapshot(retry.session).SerializeToString(deterministic=True)
