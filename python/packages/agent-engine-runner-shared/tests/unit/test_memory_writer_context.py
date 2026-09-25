"""Tests for MemoryWriter execution context propagation into background threads."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent_engine_runner_shared.context import (
    clear_execution_context,
    get_current_execution_id,
    set_execution_context,
)
from agent_engine_runner_shared.memory import MemoryWriter


def _make_writer() -> MemoryWriter:
    return MemoryWriter(memory_engine=MagicMock(), max_workers=1)


def test_write_turn_async_propagates_execution_id():
    """Background thread sees the current_execution_id from the calling context."""
    observed: list[str | None] = []

    def capturing_write_turn(*args, **kwargs):
        observed.append(get_current_execution_id())

    mock_engine = MagicMock()
    mock_engine.write_turn = capturing_write_turn

    writer = MemoryWriter(memory_engine=mock_engine, max_workers=1)
    tokens = set_execution_context(
        "exec-bg-001", MagicMock(), "http://oe", user_id="u1", session_id="s1"
    )
    try:
        writer.write_turn_async(
            message="hello",
            result_messages=[MagicMock(role="user", content="hello", tool_calls=None)],
            session_id="s1",
            org_id="org1",
            user_id="u1",
            project_id="proj1",
        )
        writer.shutdown(wait=True)
    finally:
        clear_execution_context(tokens)

    assert len(observed) > 0, "write_turn should have been called"
    assert all(eid == "exec-bg-001" for eid in observed), (
        f"Expected exec-bg-001 in all calls, got {observed}"
    )


def test_write_turn_async_no_context_does_not_raise():
    """write_turn_async works normally when no execution context is set."""
    mock_engine = MagicMock()
    writer = MemoryWriter(memory_engine=mock_engine, max_workers=1)

    # No set_execution_context call — contextvar is None
    writer.write_turn_async(
        message="hello",
        result_messages=[MagicMock(role="user", content="hello", tool_calls=None)],
        session_id="s1",
        org_id="org1",
        user_id="u1",
        project_id="proj1",
    )
    writer.shutdown(wait=True)

    mock_engine.write_turn.assert_called()


def test_write_turn_sync_forwards_metadata_to_engine():
    """Developer-supplied metadata reaches each per-message write_turn call."""
    captured: list[dict] = []

    def capturing_write_turn(*args, **kwargs):
        captured.append(kwargs)

    mock_engine = MagicMock()
    mock_engine.write_turn = capturing_write_turn

    writer = MemoryWriter(memory_engine=mock_engine, max_workers=1)
    writer.write_turn_sync(
        message="hello",
        result_messages=[MagicMock(role="user", content="hello", tool_calls=None)],
        session_id="s1",
        org_id="org1",
        user_id="u1",
        project_id="proj1",
        metadata={"channel": "slack"},
    )

    assert captured, "write_turn should have been called"
    assert all(kw.get("metadata") == {"channel": "slack"} for kw in captured)


def test_write_turn_sync_metadata_defaults_none():
    captured: list[dict] = []

    def capturing_write_turn(*args, **kwargs):
        captured.append(kwargs)

    mock_engine = MagicMock()
    mock_engine.write_turn = capturing_write_turn

    writer = MemoryWriter(memory_engine=mock_engine, max_workers=1)
    writer.write_turn_sync(
        message="hello",
        result_messages=[MagicMock(role="user", content="hello", tool_calls=None)],
        session_id="s1",
        org_id="org1",
        user_id="u1",
        project_id="proj1",
    )

    assert captured
    assert all(kw.get("metadata") is None for kw in captured)


def test_write_turn_sync_propagates_execution_id():
    """Sync writes also see the current_execution_id (no special handling needed,
    but verifying the baseline so async can be compared against it)."""
    observed: list[str | None] = []

    def capturing_write_turn(*args, **kwargs):
        observed.append(get_current_execution_id())

    mock_engine = MagicMock()
    mock_engine.write_turn = capturing_write_turn

    writer = MemoryWriter(memory_engine=mock_engine, max_workers=1)
    tokens = set_execution_context(
        "exec-sync-001", MagicMock(), "http://oe", user_id="u1", session_id="s1"
    )
    try:
        writer.write_turn_sync(
            message="hello",
            result_messages=[MagicMock(role="user", content="hello", tool_calls=None)],
            session_id="s1",
            org_id="org1",
            user_id="u1",
            project_id="proj1",
        )
    finally:
        clear_execution_context(tokens)

    assert len(observed) > 0
    assert all(eid == "exec-sync-001" for eid in observed)


def test_write_turn_sync_stamps_idempotency_key_on_single_turn():
    """A caller-supplied key reaches the engine on a one-turn write."""
    captured: list[dict] = []

    def capturing_write_turn(*args, **kwargs):
        captured.append(kwargs)

    mock_engine = MagicMock()
    mock_engine.write_turn = capturing_write_turn

    writer = MemoryWriter(memory_engine=mock_engine, max_workers=1)
    writer.write_turn_sync(
        message="hello",
        result_messages=[MagicMock(role="user", content="hello", tool_calls=None)],
        session_id="s1",
        org_id="org1",
        user_id="u1",
        project_id="proj1",
        idempotency_key="turn-42",
    )

    assert len(captured) == 1
    assert captured[0]["idempotency_key"] == "turn-42"


def test_write_turn_sync_ignores_idempotency_key_on_multi_turn():
    """One key cannot represent a multi-message write; it must not stamp
    every turn, or key-based dedupe would collapse distinct history."""
    captured: list[dict] = []

    def capturing_write_turn(*args, **kwargs):
        captured.append(kwargs)

    mock_engine = MagicMock()
    mock_engine.write_turn = capturing_write_turn

    writer = MemoryWriter(memory_engine=mock_engine, max_workers=1)
    writer.write_turn_sync(
        message="hello",
        result_messages=[MagicMock(role="assistant", content="hi", tool_calls=None)],
        session_id="s1",
        org_id="org1",
        user_id="u1",
        project_id="proj1",
        idempotency_key="turn-42",
    )

    assert len(captured) == 2
    assert all("idempotency_key" not in kw for kw in captured)


def test_write_turn_async_forwards_idempotency_key():
    """The background write carries the caller's key through the thread pool."""
    captured: list[dict] = []

    def capturing_write_turn(*args, **kwargs):
        captured.append(kwargs)

    mock_engine = MagicMock()
    mock_engine.write_turn = capturing_write_turn

    writer = MemoryWriter(memory_engine=mock_engine, max_workers=1)
    writer.write_turn_async(
        message="hello",
        result_messages=[MagicMock(role="user", content="hello", tool_calls=None)],
        session_id="s1",
        org_id="org1",
        user_id="u1",
        project_id="proj1",
        idempotency_key="turn-42",
    )
    writer.shutdown(wait=True)

    assert len(captured) == 1
    assert captured[0]["idempotency_key"] == "turn-42"


def test_native_write_preserves_valid_turns_before_malformed_tail():
    """A malformed later native turn must not roll back earlier writes."""
    captured: list[dict] = []
    mock_engine = MagicMock()
    mock_engine.write_turn.side_effect = lambda **kwargs: captured.append(kwargs)
    writer = MemoryWriter(memory_engine=mock_engine, max_workers=1)

    writer.write_turn_sync(
        message="hello",
        result_messages=[
            MagicMock(role="assistant", content="hi", tool_calls=None),
            MagicMock(role="tool", content="", tool_call_id="", name="lookup"),
        ],
        session_id="s1",
        org_id="org1",
        user_id="u1",
        project_id="proj1",
    )

    assert [(turn["role"], turn["content"]) for turn in captured] == [
        ("user", "hello"),
        ("assistant", "hi"),
    ]
