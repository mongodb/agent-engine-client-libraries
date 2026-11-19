"""Tests for ADK App.memory wiring over shared app-bound adapters."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from agent_engine_sdk_memory import Memory
from agent_engine_sdk_memory.models import CreateSemanticResult

from agent_engine_runner_shared import RuntimeMode
from agent_engine_runner_shared.memory_appbound import (
    AppBoundCrudClient,
    AppBoundRuntime,
)


def _make_app(mode: RuntimeMode = RuntimeMode.AER) -> Any:
    """Create an App with a mocked TenantRuntime."""
    with patch("agent_engine_sdk_adk.runtime.TenantRuntime") as MockRuntime:
        mock_rt = MagicMock()
        mock_rt.mode = mode
        # Real configs default to no MCP servers; the mock must match or App
        # construction would attempt remote tools/list discovery.
        mock_rt.agent_config.mcp.servers = {}
        MockRuntime.return_value = mock_rt
        from agent_engine_sdk_adk.runtime import App

        app = App(app_name="test-agent")
        app._runtime = mock_rt
        return app


class TestAppMemory:
    def test_memory_returns_unified_facade(self) -> None:
        app = _make_app()

        assert isinstance(app.memory, Memory)
        assert isinstance(app.memory._runtime, AppBoundRuntime)
        assert isinstance(app.memory._client, AppBoundCrudClient)

    def test_memory_is_singleton(self) -> None:
        app = _make_app()

        assert app.memory is app.memory

    def test_memory_adapters_wrap_app_runtime(self) -> None:
        app = _make_app()

        assert app.memory._runtime._runtime is app._runtime
        assert app.memory._client._runtime is app._runtime

    def test_save_semantic_returns_the_accepted_operation(self) -> None:
        app = _make_app()
        accepted = CreateSemanticResult(
            id=None,
            label="contact_pref",
            has_embedding=False,
            operation_id="op-1",
            operation_status="queued",
        )
        app._runtime.create_semantic.return_value = accepted

        result = app.memory.save_semantic(
            text="User prefers email", label="contact_pref", user_id="u1"
        )

        assert result == accepted
        sent = app._runtime.create_semantic.call_args.kwargs
        assert (sent["text"], sent["label"], sent["user_id"]) == (
            "User prefers email",
            "contact_pref",
            "u1",
        )

    def test_search_semantic_returns_chunks(self) -> None:
        app = _make_app()
        app._runtime.search_semantic.return_value = [
            {"id": "doc-1", "content": "prefers email", "similarity_score": 0.9}
        ]

        chunks = app.memory.search_semantic(query="contact preference", user_id="u1")

        assert len(chunks) == 1
        assert chunks[0].content == "prefers email"
        assert chunks[0].similarity_score == 0.9
