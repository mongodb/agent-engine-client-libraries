"""Tests for ADK App.memory wiring over shared app-bound adapters."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from agent_engine_sdk_memory import Memory

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

    def test_save_semantic_acknowledged_mapping(self) -> None:
        app = _make_app()
        app._runtime.save_semantic.return_value = True

        result = app.memory.save_semantic(
            text="User prefers email", label="contact_pref", user_id="u1"
        )

        assert result.acknowledged is True
        assert result.label == "contact_pref"

    def test_save_semantic_failure_not_acknowledged(self) -> None:
        app = _make_app()
        app._runtime.save_semantic.return_value = False

        result = app.memory.save_semantic(
            text="User prefers email", label="contact_pref", user_id="u1"
        )

        assert result.acknowledged is False

    def test_search_semantic_returns_chunks(self) -> None:
        app = _make_app()
        app._runtime.search_semantic.return_value = [
            {"id": "doc-1", "content": "prefers email", "similarity_score": 0.9}
        ]

        chunks = app.memory.search_semantic(query="contact preference", user_id="u1")

        assert len(chunks) == 1
        assert chunks[0].content == "prefers email"
        assert chunks[0].similarity_score == 0.9
