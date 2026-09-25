"""Tests for AER workspace_id resolution used to scope checkpoint thread_ids.

The write path and read-side query plugin both delegate to
``resolve_checkpoint_workspace_id`` so reads and writes cannot diverge.
``_resolve_workspace_id`` in ``agent_engine_runner_shared.server.aer`` is a thin wrapper.
"""

from __future__ import annotations

from agent_engine_runner_shared.models import ExecuteRequest
from agent_engine_runner_shared.server.aer import _resolve_workspace_id


def _request(workspace_id: str | None) -> ExecuteRequest:
    return ExecuteRequest(
        execution_id="exec-1",
        platform_api_url="http://oe:8000",
        workspace_id=workspace_id,
    )


class TestResolveWorkspaceID:
    def test_app_id_is_authoritative_over_wire_value(self, monkeypatch):
        # APP_ID (pod identity) wins even when the wire value diverges, so the
        # write-path scope matches the APP_ID-scoped read path.
        monkeypatch.setenv("APP_ID", "ws-pod")
        assert _resolve_workspace_id(_request("ws-other")) == "ws-pod"

    def test_app_id_used_when_wire_value_omitted(self, monkeypatch):
        # An omitted wire workspace_id must not produce an unscoped write while
        # reads stay APP_ID-scoped.
        monkeypatch.setenv("APP_ID", "ws-pod")
        assert _resolve_workspace_id(_request(None)) == "ws-pod"

    def test_falls_back_to_wire_value_when_app_id_unset(self, monkeypatch):
        monkeypatch.delenv("APP_ID", raising=False)
        assert _resolve_workspace_id(_request("ws-wire")) == "ws-wire"

    def test_falls_back_to_wire_value_when_app_id_empty(self, monkeypatch):
        monkeypatch.setenv("APP_ID", "")
        assert _resolve_workspace_id(_request("ws-wire")) == "ws-wire"

    def test_none_when_neither_set(self, monkeypatch):
        # Local dev: no APP_ID and no wire value — reads are unscoped too.
        monkeypatch.delenv("APP_ID", raising=False)
        assert _resolve_workspace_id(_request(None)) is None
