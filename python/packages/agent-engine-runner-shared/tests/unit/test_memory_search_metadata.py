"""Search recalls record the effective query in memory-event metadata.

The query recorded is the string actually sent to the memory engine, which
may differ from the tool-call arguments (e.g. an agent-side fallback computed
inside the tool body). Lookup/list recalls and writes send no query, so the
key must stay absent for them.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_engine_runner_shared.context import (
    clear_execution_context,
    get_current_execution_metadata,
    set_execution_context,
)
from agent_engine_runner_shared.memory import (
    create_semantic_memory,
    get_semantic_memory,
    list_episodic_memories,
    search_episodic_memory,
    search_semantic_memory,
    search_taxonomic_memory,
)


@pytest.fixture
def execution_context():
    tokens = set_execution_context(
        "exec-1", MagicMock(), "http://oe", user_id="u1", session_id="s1"
    )
    try:
        yield
    finally:
        clear_execution_context(tokens)


def _current_memory_event() -> dict:
    return get_current_execution_metadata().get("memory", {})


def test_search_semantic_records_query(execution_context) -> None:
    engine = MagicMock()
    engine.fetch_semantic_memories.return_value = [
        SimpleNamespace(metadata={"label": "l"}, text="t", similarity_score=0.9)
    ]

    search_semantic_memory(engine, query="customer interactions", org_id="o", project_id="p")

    event = _current_memory_event()
    assert event["query"] == "customer interactions"
    events = get_current_execution_metadata()["memory_events"]
    assert events[0]["query"] == "customer interactions"


def test_search_episodic_records_query(execution_context) -> None:
    engine = MagicMock()
    engine.fetch_episodic_memories.return_value = [
        SimpleNamespace(
            title="t",
            content="c",
            summary_text="s",
            session_id="s1",
            participants=[],
            tags=[],
            similarity_score=0.5,
        )
    ]

    search_episodic_memory(engine, query="customer interactions for u1", org_id="o", project_id="p")

    assert _current_memory_event()["query"] == "customer interactions for u1"


def test_search_taxonomic_records_query(execution_context) -> None:
    engine = MagicMock()
    engine.fetch_taxonomic_memories.return_value = [
        SimpleNamespace(
            term="premium",
            definition="Periodic payment",
            domain="insurance",
            related_terms=[],
            similarity_score=0.1,
        )
    ]

    search_taxonomic_memory(engine, query="premium cost", org_id="o", project_id="p")

    assert _current_memory_event()["query"] == "premium cost"


def test_search_with_empty_query_omits_query_key(execution_context) -> None:
    engine = MagicMock()
    engine.fetch_semantic_memories.return_value = []

    search_semantic_memory(engine, query="", org_id="o", project_id="p")

    assert "query" not in _current_memory_event()


def test_write_omits_query(execution_context) -> None:
    engine = MagicMock()
    engine.create_semantic.return_value = SimpleNamespace(id="mem-1")

    create_semantic_memory(engine, text="fact", label="l", org_id="o", user_id="u1")

    event = _current_memory_event()
    assert event["action"] == "write"
    assert "query" not in event


def test_label_lookup_omits_query(execution_context) -> None:
    engine = MagicMock()
    engine.get_semantic.return_value = {"label": "name", "text": "Kshitiz"}

    get_semantic_memory(engine, org_id="o", project_id="p", label="name")

    event = _current_memory_event()
    assert event["action"] == "recall"
    assert "query" not in event


def test_list_recall_omits_query(execution_context) -> None:
    engine = MagicMock()
    engine.list_episodic.return_value = [{"id": "1", "title": "t", "summary_text": "s"}]

    list_episodic_memories(engine, org_id="o", project_id="p")

    event = _current_memory_event()
    assert event["action"] == "recall"
    assert "query" not in event


def test_query_with_embedded_credentials_is_redacted(execution_context) -> None:
    """The query and recalled content are persisted in execution-log metadata
    returned to the UI, so credential-shaped fragments must be scrubbed before
    either is stored."""
    engine = MagicMock()
    engine.fetch_semantic_memories.return_value = [
        SimpleNamespace(
            metadata={"label": "l"},
            text="see https://user:p%40ss@internal.example.com/db",
            similarity_score=0.9,
        )
    ]

    search_semantic_memory(
        engine,
        query=(
            "status of https://user:p%40ss@internal.example.com/db"
            " and mongodb+srv://admin:hunter2@cluster0.example.net/prod"
            " with Authorization: Bearer abc123.def456 and api_key=sk-live-999"
        ),
        org_id="o",
        project_id="p",
    )

    event = _current_memory_event()
    stored = event["query"]
    for fragment in ("p%40ss", "hunter2", "abc123.def456", "sk-live-999"):
        assert fragment not in stored
        assert fragment not in event["content"]
    assert "https://<redacted>:<redacted>@internal.example.com" in stored
    assert "mongodb+srv://<redacted>:<redacted>@cluster0.example.net" in stored
    assert "Bearer <redacted>" in stored
    assert "api_key=<redacted>" in stored
