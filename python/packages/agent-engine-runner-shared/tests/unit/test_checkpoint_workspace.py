"""Tests for checkpoint workspace_id resolution shared by read and write paths."""

from __future__ import annotations

import pytest

from agent_engine_runner_shared.checkpoint_workspace import resolve_checkpoint_workspace_id


class TestResolveCheckpointWorkspaceID:
    def test_app_id_is_authoritative_over_wire_value(self, monkeypatch):
        monkeypatch.setenv("APP_ID", "ws-pod")
        assert resolve_checkpoint_workspace_id("ws-other") == "ws-pod"

    def test_app_id_used_when_wire_value_omitted(self, monkeypatch):
        monkeypatch.setenv("APP_ID", "ws-pod")
        assert resolve_checkpoint_workspace_id(None) == "ws-pod"

    def test_falls_back_to_wire_value_when_app_id_unset(self, monkeypatch):
        monkeypatch.delenv("APP_ID", raising=False)
        assert resolve_checkpoint_workspace_id("ws-wire") == "ws-wire"

    def test_falls_back_to_wire_value_when_app_id_empty(self, monkeypatch):
        monkeypatch.setenv("APP_ID", "")
        assert resolve_checkpoint_workspace_id("ws-wire") == "ws-wire"

    def test_none_when_neither_set(self, monkeypatch):
        monkeypatch.delenv("APP_ID", raising=False)
        monkeypatch.delenv("REQUIRE_PROJECT_SCOPED_DB", raising=False)
        assert resolve_checkpoint_workspace_id(None) is None

    def test_fails_closed_when_managed_scope_is_required(self, monkeypatch):
        monkeypatch.delenv("APP_ID", raising=False)
        monkeypatch.setenv("REQUIRE_PROJECT_SCOPED_DB", "true")
        with pytest.raises(RuntimeError, match="checkpoint workspace scope is required"):
            resolve_checkpoint_workspace_id("attacker-workspace")
