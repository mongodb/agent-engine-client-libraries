"""Framework-agnostic query plugin protocol for the AER.

The AER hosts two read endpoints that surface framework-specific persisted
state — per-session summary and conversation messages. The implementations
live in framework adapter packages (e.g. ``agent-engine-sdk-langgraph``); the AER
itself only hosts the routes and delegates to whichever plugin the runtime
has been configured with.

Trust model
-----------

These endpoints take no workspace identifier on the wire. The AER trusts its
caller (the platform's Orchestration Engine proxy) to pass only session_ids
that belong to the caller's workspace; the OE establishes that scope by
consulting the ``executions`` collection — which is tagged with
``workspace_id`` — before issuing the request. The framework plugin then
narrows reads by its own session key. For the LangGraph plugin that key is
the workspace-scoped composite ``f"{session_id}:{workspace_id}"``. The
workspace is resolved the same way on read and write — ``APP_ID`` when set,
otherwise the wire ``workspace_id`` from the most recent ``/execute`` — so a
plugin keyed to one workspace cannot read another's checkpoints even if a
stray session_id slipped past the proxy — defense in depth, not a substitute
for the upstream ownership check.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_engine_sdk.models import (
    SessionMessagesResponse,
    SessionsSummaryResponse,
)


@runtime_checkable
class AERQueryPlugin(Protocol):
    """Read-side plugin that surfaces framework-specific session state.

    Implementations live alongside framework adapters and read from the
    adapter's persistence (LangGraph checkpoint collections, ADK state
    store, etc.).
    """

    async def get_summaries_for_sessions(self, session_ids: list[str]) -> SessionsSummaryResponse:
        """Given a list of session_ids, return a SessionsSummaryResponse."""
        ...

    async def get_messages_for_session(self, session_id: str) -> SessionMessagesResponse:
        """Given a session_id, return a SessionMessagesResponse."""
        ...
