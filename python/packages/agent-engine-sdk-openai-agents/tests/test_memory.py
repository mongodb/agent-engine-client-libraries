"""App.memory wiring over the shared app-bound adapters."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from agent_engine_sdk_memory import Memory

from agent_engine_runner_shared.memory_appbound import (
    AppBoundCrudClient,
    AppBoundRuntime,
)
from agent_engine_sdk_openai_agents import App


@pytest.fixture
def app() -> App:
    app = App(app_name="support")
    app._runtime = MagicMock()
    return app


def test_memory_is_one_facade_over_the_app_runtime(app: App) -> None:
    memory = app.memory

    assert isinstance(memory, Memory)
    assert memory is app.memory
    assert isinstance(memory._runtime, AppBoundRuntime)
    assert isinstance(memory._client, AppBoundCrudClient)
    assert memory._runtime._runtime is app._runtime
    assert memory._client._runtime is app._runtime


@pytest.mark.parametrize("stored", [True, False])
def test_save_semantic_reports_whether_the_runtime_stored_it(
    app: App, stored: bool
) -> None:
    app._runtime.save_semantic.return_value = stored

    result = app.memory.save_semantic(
        text="Prefers email", label="contact_pref", user_id="u1"
    )

    assert result.acknowledged is stored
    assert result.label == "contact_pref"
    # The user id attributes the memory, so it must reach the runtime unchanged.
    sent = app._runtime.save_semantic.call_args.kwargs
    assert (sent["text"], sent["label"], sent["user_id"]) == (
        "Prefers email",
        "contact_pref",
        "u1",
    )


def test_search_semantic_returns_typed_chunks(app: App) -> None:
    app._runtime.search_semantic.return_value = [
        {"id": "doc-1", "content": "prefers email", "similarity_score": 0.9}
    ]

    chunks = app.memory.search_semantic(query="contact preference", user_id="u1")

    assert [(chunk.content, chunk.similarity_score) for chunk in chunks] == [
        ("prefers email", 0.9)
    ]
