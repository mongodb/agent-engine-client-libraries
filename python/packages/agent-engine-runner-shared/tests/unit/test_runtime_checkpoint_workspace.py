"""Tests for TenantRuntime checkpoint workspace resolution."""

from __future__ import annotations

import pytest

from agent_engine_runner_shared.runtime import TenantRuntime


class TestCheckpointWorkspaceResolution:
    def test_get_checkpoint_workspace_id_uses_app_id(self, monkeypatch):
        monkeypatch.setenv("APP_ID", "ws-pod")
        runtime = TenantRuntime(app_name="test")
        runtime.note_checkpoint_wire_workspace_id("ws-wire")

        assert runtime.get_checkpoint_workspace_id() == "ws-pod"

    def test_get_checkpoint_workspace_id_falls_back_to_wire(self, monkeypatch):
        monkeypatch.delenv("APP_ID", raising=False)
        runtime = TenantRuntime(app_name="test")
        runtime.note_checkpoint_wire_workspace_id("ws-wire")

        assert runtime.get_checkpoint_workspace_id() == "ws-wire"

    def test_get_checkpoint_workspace_id_unscoped_when_neither_set(self, monkeypatch):
        """No resolvable scope means an explicitly unscoped runtime (local dev /
        tests, where REQUIRE_PROJECT_SCOPED_DB is unset), so reads must match
        the bare-keyed write path rather than fail closed."""
        monkeypatch.delenv("APP_ID", raising=False)
        monkeypatch.delenv("REQUIRE_PROJECT_SCOPED_DB", raising=False)
        runtime = TenantRuntime(app_name="test")

        assert runtime.get_checkpoint_workspace_id() == ""

    def test_get_checkpoint_workspace_id_fails_closed_when_scope_required(self, monkeypatch):
        monkeypatch.delenv("APP_ID", raising=False)
        monkeypatch.setenv("REQUIRE_PROJECT_SCOPED_DB", "true")
        runtime = TenantRuntime(app_name="test")

        with pytest.raises(RuntimeError, match="checkpoint workspace scope is required"):
            runtime.get_checkpoint_workspace_id()
