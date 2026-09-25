"""Tests for the AER's framework-agnostic query routes.

Exercises the two endpoints the AER registers on behalf of an
:class:`AERQueryPlugin`:

* ``GET /query/sessions?session_ids=...`` — workspace-already-scoped
  session summary enrichment.
* ``GET /query/sessions/{session_id}/messages`` — per-session decoded
  conversation history.

Both routes are pure delegation — they return 501 when no plugin is
registered and forward the call to the plugin otherwise. The plugin
implementation itself is exercised by the framework adapter's own
tests (see ``agent-engine-sdk-langgraph/tests/test_query.py``).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

from agent_engine_sdk.models import (
    SessionMessage,
    SessionMessagesResponse,
    SessionsSummaryResponse,
    SessionSummary,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_engine_runner_shared.server.aer import AERServer


class _StubPlugin:
    """In-memory AERQueryPlugin used by these route tests.

    The plugin records every call so tests can assert which sessions
    the route forwarded, and returns canned responses pre-loaded by the
    test setup.
    """

    def __init__(
        self,
        *,
        summary: SessionsSummaryResponse | None = None,
        messages: SessionMessagesResponse | None = None,
        summary_error: Exception | None = None,
        messages_error: Exception | None = None,
    ) -> None:
        self._summary = summary or SessionsSummaryResponse(sessions=[])
        self._messages = messages or SessionMessagesResponse(messages=[])
        self._summary_error = summary_error
        self._messages_error = messages_error
        self.summary_calls: list[list[str]] = []
        self.message_calls: list[str] = []

    async def get_summaries_for_sessions(self, session_ids: list[str]) -> SessionsSummaryResponse:
        self.summary_calls.append(list(session_ids))
        if self._summary_error is not None:
            raise self._summary_error
        return self._summary

    async def get_messages_for_session(self, session_id: str) -> SessionMessagesResponse:
        self.message_calls.append(session_id)
        if self._messages_error is not None:
            raise self._messages_error
        return self._messages


def _make_client(
    plugin: Any | None,
    *,
    raise_server_exceptions: bool = True,
) -> TestClient:
    """Build a FastAPI app with only the AER query routes wired up.

    Bypasses the full AERServer constructor (which would try to load an
    agent graph) by instantiating directly and registering routes on a
    bare FastAPI app.

    ``raise_server_exceptions=False`` lets tests observe HTTP 500
    responses for unhandled server-side exceptions (the production
    behavior) instead of having the TestClient re-raise.
    """
    runtime = MagicMock()
    runtime.get_query_plugin = MagicMock(return_value=plugin)
    runtime._tool_definitions = {}
    runtime.get_agent = MagicMock(return_value=_NullAgent())

    server = AERServer.__new__(AERServer)
    server.runtime = runtime  # type: ignore[attr-defined]
    server._client = None  # type: ignore[attr-defined]
    server._client_lock = asyncio.Lock()  # type: ignore[attr-defined]
    server._a2a_registered = False  # type: ignore[attr-defined]
    server._chunk_seq = {}  # type: ignore[attr-defined]
    server._owner_callback_url = {}  # type: ignore[attr-defined]

    app = FastAPI()
    server.register_routes(app)
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


class _NullAgent:
    """Minimal stand-in for the agent required by make_execution_router."""

    def execute(self, _ctx: Any, _input: Any) -> Any:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# /query/sessions
# ---------------------------------------------------------------------------


def test_session_summaries_returns_501_when_no_plugin_registered() -> None:
    client = _make_client(plugin=None)
    response = client.get("/query/sessions", params={"session_ids": ["s1"]})
    assert response.status_code == 501


def test_session_summaries_returns_empty_list_for_no_ids_without_plugin() -> None:
    # The endpoint short-circuits on empty session_ids before checking the
    # plugin so the OE can issue a no-op call without tripping 501.
    client = _make_client(plugin=None)
    response = client.get("/query/sessions")
    assert response.status_code == 200
    assert response.json() == {"sessions": []}


def test_session_summaries_does_not_invoke_plugin_for_empty_ids() -> None:
    """The empty-session_ids short-circuit must skip the plugin even when
    one is registered. Guards against a refactor that moves the early
    return below the plugin lookup."""
    plugin = _StubPlugin()
    client = _make_client(plugin=plugin)
    response = client.get("/query/sessions")
    assert response.status_code == 200
    assert response.json() == {"sessions": []}
    assert plugin.summary_calls == []


def test_session_summaries_forwards_session_ids_to_plugin() -> None:
    plugin = _StubPlugin(
        summary=SessionsSummaryResponse(
            sessions=[
                SessionSummary(
                    session_id="s1",
                    last_activity="2026-05-22T10:00:00+00:00",
                    created_at="2026-05-22T09:00:00+00:00",
                    message_count=3,
                    first_message_preview="hello",
                ),
            ]
        )
    )
    client = _make_client(plugin=plugin)
    response = client.get(
        "/query/sessions",
        params=[("session_ids", "s1"), ("session_ids", "s2")],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sessions"][0]["session_id"] == "s1"
    assert body["sessions"][0]["message_count"] == 3
    assert body["sessions"][0]["first_message_preview"] == "hello"
    # Both repeated query params must reach the plugin.
    assert plugin.summary_calls == [["s1", "s2"]]


# ---------------------------------------------------------------------------
# /query/sessions/{session_id}/messages
# ---------------------------------------------------------------------------


def test_session_messages_returns_501_when_no_plugin_registered() -> None:
    client = _make_client(plugin=None)
    response = client.get("/query/sessions/s1/messages")
    assert response.status_code == 501


def test_session_messages_forwards_path_param_to_plugin() -> None:
    plugin = _StubPlugin(
        messages=SessionMessagesResponse(
            messages=[
                SessionMessage(
                    id="m1",
                    role="user",
                    content="hi",
                    timestamp="2026-05-22T09:00:00+00:00",
                    session_id="s1",
                ),
                SessionMessage(
                    id="m2",
                    role="assistant",
                    content="hello",
                    timestamp="2026-05-22T09:00:05+00:00",
                    session_id="s1",
                ),
            ]
        )
    )
    client = _make_client(plugin=plugin)
    response = client.get("/query/sessions/s1/messages")
    assert response.status_code == 200
    body = response.json()
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][1]["role"] == "assistant"
    assert plugin.message_calls == ["s1"]


def test_session_messages_returns_empty_list_when_plugin_finds_nothing() -> None:
    plugin = _StubPlugin(messages=SessionMessagesResponse(messages=[]))
    client = _make_client(plugin=plugin)
    response = client.get("/query/sessions/unknown/messages")
    assert response.status_code == 200
    assert response.json() == {"messages": []}
    assert plugin.message_calls == ["unknown"]


# ---------------------------------------------------------------------------
# Plugin exception → 500 (contract: routes are pure delegation, no swallow)
# ---------------------------------------------------------------------------


def test_session_summaries_plugin_exception_surfaces_as_500() -> None:
    """Plugins are trusted to fail loudly; the route does not swallow.

    Uses ``raise_server_exceptions=False`` so the TestClient surfaces the
    real HTTP response (500) instead of re-raising the underlying
    exception — that's what a production caller would see.
    """
    plugin = _StubPlugin(summary_error=RuntimeError("checkpoint store down"))
    client = _make_client(plugin=plugin, raise_server_exceptions=False)
    response = client.get("/query/sessions", params={"session_ids": ["s1"]})
    assert response.status_code == 500
    # And the plugin was actually invoked (the route didn't reject upstream).
    assert plugin.summary_calls == [["s1"]]


def test_session_messages_plugin_exception_surfaces_as_500() -> None:
    plugin = _StubPlugin(messages_error=RuntimeError("serde drift"))
    client = _make_client(plugin=plugin, raise_server_exceptions=False)
    response = client.get("/query/sessions/s1/messages")
    assert response.status_code == 500
    assert plugin.message_calls == ["s1"]


# ---------------------------------------------------------------------------
# Length-cap defense (session_ids has max_length=200)
# ---------------------------------------------------------------------------


def test_session_summaries_rejects_over_length_session_ids_list() -> None:
    """The route caps session_ids at 200 so the AER can't be DoS-amplified
    via a single oversized $in scan against the shared checkpoint store."""
    plugin = _StubPlugin()
    client = _make_client(plugin=plugin)
    too_many_ids = [f"s{i}" for i in range(201)]
    response = client.get(
        "/query/sessions",
        params=[("session_ids", sid) for sid in too_many_ids],
    )
    # FastAPI surfaces query-parameter validation failures as 422.
    assert response.status_code == 422
    # The plugin is never reached when validation rejects the request.
    assert plugin.summary_calls == []


def test_session_summaries_accepts_exactly_max_session_ids() -> None:
    """At the cap (200), the request still succeeds — the cap is inclusive."""
    plugin = _StubPlugin()
    client = _make_client(plugin=plugin)
    max_allowed_ids = [f"s{i}" for i in range(200)]
    response = client.get(
        "/query/sessions",
        params=[("session_ids", sid) for sid in max_allowed_ids],
    )
    assert response.status_code == 200
    assert len(plugin.summary_calls[0]) == 200
