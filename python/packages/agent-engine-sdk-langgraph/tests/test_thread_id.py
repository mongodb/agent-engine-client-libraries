"""Tests for workspace-scoped thread_id composition.

These two functions are the single source of truth shared by the write path
(``agent.py``) and the read path (``query.py``); a round-trip mismatch here
would silently break conversation continuity, so they are tested in isolation.
"""

from __future__ import annotations

import pytest

from agent_engine_sdk_langgraph.thread_id import (
    scoped_thread_id,
    session_id_from_thread_id,
    thread_ids_for_query,
    thread_ids_for_sessions_query,
)


class TestScopedThreadID:
    def test_appends_workspace_suffix(self) -> None:
        assert scoped_thread_id("sess-1", "ws-1") == "sess-1:ws-1"

    def test_same_inputs_are_deterministic(self) -> None:
        """Continuity depends on the same (session, workspace) always
        resolving to the same key."""
        assert scoped_thread_id("sess-1", "ws-1") == scoped_thread_id("sess-1", "ws-1")

    def test_different_workspaces_isolate(self) -> None:
        """The whole point of the ticket: same session_id, different agent →
        different checkpoint key."""
        assert scoped_thread_id("sess-1", "ws-a") != scoped_thread_id("sess-1", "ws-b")

    @pytest.mark.parametrize("workspace_id", ["", None])
    def test_no_workspace_falls_back_to_bare_session(self, workspace_id) -> None:
        assert scoped_thread_id("sess-1", workspace_id) == "sess-1"

    @pytest.mark.parametrize("session_id", ["", None])
    def test_missing_session_uses_default(self, session_id) -> None:
        assert scoped_thread_id(session_id, "ws-1") == "default:ws-1"


class TestSessionIDFromThreadID:
    def test_strips_workspace_suffix(self) -> None:
        assert session_id_from_thread_id("sess-1:ws-1", "ws-1") == "sess-1"

    def test_round_trips_with_scoped_thread_id(self) -> None:
        composite = scoped_thread_id("sess-1", "ws-1")
        assert session_id_from_thread_id(composite, "ws-1") == "sess-1"

    @pytest.mark.parametrize("workspace_id", ["", None])
    def test_no_workspace_returns_thread_id_unchanged(self, workspace_id) -> None:
        assert session_id_from_thread_id("sess-1", workspace_id) == "sess-1"

    def test_legacy_unscoped_thread_id_is_left_intact(self) -> None:
        """A checkpoint written before scoping existed has no ``:ws`` suffix;
        stripping must not mangle it."""
        assert session_id_from_thread_id("legacy-session", "ws-1") == "legacy-session"

    def test_only_trailing_workspace_suffix_is_stripped(self) -> None:
        """A workspace id appearing mid-string must not be stripped — only the
        trailing scope suffix is removed."""
        assert session_id_from_thread_id("ws-1-extra:ws-1", "ws-1") == "ws-1-extra"


class TestThreadIDsForQuery:
    def test_no_workspace_returns_bare_session_only(self) -> None:
        assert thread_ids_for_query("sess-1", "") == ["sess-1"]
        assert thread_ids_for_query("sess-1", None) == ["sess-1"]

    def test_with_workspace_returns_only_scoped_key(self) -> None:
        """Bare keys are shared across workspaces; once a scope is known,
        reads must never include them."""
        assert thread_ids_for_query("sess-1", "ws-1") == ["sess-1:ws-1"]

    def test_sessions_query_dedupes_across_sessions(self) -> None:
        assert thread_ids_for_sessions_query(["s1", "s2"], "ws-1") == [
            "s1:ws-1",
            "s2:ws-1",
        ]
