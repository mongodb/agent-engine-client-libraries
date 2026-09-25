"""Tests for the per-source context path.

Covers ``agent_engine_runner_shared.memory.build_context_from_sources`` (the module helper)
and ``TenantRuntime.build_context_from_sources`` (identity guards, the
stm/session_id rule, and the durable-activity round-trip that must preserve the
full ``ContextResponse`` metadata across a checkpoint).
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from agent_engine_sdk_memory.models import (
    ContextMetadata,
    ContextResponse,
    MemoryChunk,
    MemorySource,
    RetrievalMode,
    SourceSpec,
)

from agent_engine_runner_shared import TenantRuntime
from agent_engine_runner_shared.context import clear_execution_context, set_execution_context
from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_KIND_MEMORY,
    ActivityContext,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import WorkflowIdentity
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.memory import (
    build_context_from_sources as memory_build_context_from_sources,
)
from agent_engine_runner_shared.workflow import attempt_context_scope
from agent_engine_runner_shared.workflow.activity import completed_outcome, value_to_json
from agent_engine_runner_shared.workflow.client import ActivityDispatch, ActivityReplay


def _attempt() -> AttemptContext:
    return AttemptContext(
        attempt_id="attempt-1",
        fencing_token=7,
        owner_id="aer-1",
        workflow_identity=WorkflowIdentity(
            session_id="session-456",
            execution_id="execution-1",
        ),
    )


def _activity_context() -> ActivityContext:
    return ActivityContext(
        workflow_identity=_attempt().workflow_identity,
        activity_id="activity-1",
        attempt_id="attempt-1",
        fencing_token=7,
    )


class _FakeWorkflow:
    def __init__(self, started: object) -> None:
        self.started = started
        self.commands: list[Any] = []
        self.outcomes: list[Any] = []

    def start_activity(self, command: Any) -> object:
        self.commands.append(command)
        if isinstance(self.started, Exception):
            raise self.started
        return self.started

    def report_outcome(self, outcome: Any) -> None:
        self.outcomes.append(outcome)


@contextmanager
def _durable_execution(workflow: _FakeWorkflow):
    tokens = set_execution_context(
        execution_id="execution-1",
        wrapper=SimpleNamespace(workflow=workflow),
        oe_url="http://oe:8000",
        user_id="user-123",
        session_id="session-456",
    )
    try:
        with attempt_context_scope(_attempt()):
            yield
    finally:
        clear_execution_context(tokens)


def _response_with_metadata() -> ContextResponse:
    """A ContextResponse carrying the per-source metadata the flat path drops."""
    return ContextResponse(
        formatted_context="merged context",
        metadata=ContextMetadata(
            token_count=12,
            ranking_strategy="client_rrf",
            source_outcomes=[
                {
                    "source": "semantic",
                    "requested_mode": "hybrid",
                    "effective_mode": "hybrid",
                    "count": 3,
                    "error": None,
                }
            ],
        ),
    )


_CHUNK_TS = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def _full_response() -> ContextResponse:
    """A maximal ContextResponse exercising every field the checkpoint carries.

    Covers the shapes the thin fixture omits: list-formatted context, the typed
    integer maps (``token_count``/``memory_counts``), float ``timing``, a
    ``None`` inside ``source_outcomes``, and a ``selected_memories`` chunk with a
    ``datetime`` and free-form nested metadata. This is what the durable replay
    test round-trips through the protobuf ``Value`` boundary to prove nothing is
    dropped or coerced wrong. Values are kept to types that survive that boundary
    losslessly (no integers above 2**53, which the shared activity layer would
    narrow to a double regardless of this method).
    """
    return ContextResponse(
        formatted_context=[{"role": "system", "text": "recent turn"}],
        metadata=ContextMetadata(
            token_count=42,
            memory_counts={"semantic": 2, "episodic": 1},
            timing={"retrieval": 0.123, "rerank": 0.045},
            ranking_strategy="voyage_rerank",
            source_outcomes=[
                {
                    "source": "semantic",
                    "requested_mode": "hybrid",
                    "effective_mode": "hybrid",
                    "count": 2,
                    "error": None,
                },
                {
                    "source": "episodic",
                    "requested_mode": "semantic",
                    "effective_mode": "text",
                    "count": 1,
                    "error": "degraded",
                },
            ],
        ),
        selected_memories=[
            MemoryChunk(
                id="mem-1",
                content="user prefers dark mode",
                source="semantic",
                timestamp=_CHUNK_TS,
                similarity_score=0.875,
                metadata={"channel": "web", "confidence": 0.5, "label": "theme"},
            )
        ],
    )


class TestMemoryBuildContextFromSourcesHelper:
    """The module-level helper delegates to engine.build_context2."""

    def test_returns_empty_response_when_memory_disabled(self):
        result = memory_build_context_from_sources(
            memory_engine=None,
            query="q",
            sources=[SourceSpec(source=MemorySource.SEMANTIC)],
            org_id="org1",
            user_id="u1",
            project_id="proj1",
        )

        assert isinstance(result, ContextResponse)
        assert result.formatted_context == ""
        assert result.metadata.ranking_strategy is None

    def test_delegates_and_preserves_full_response(self):
        engine = Mock()
        engine.build_context2.return_value = _response_with_metadata()

        result = memory_build_context_from_sources(
            memory_engine=engine,
            query="q",
            sources=[SourceSpec(source=MemorySource.SEMANTIC, mode=RetrievalMode.HYBRID)],
            org_id="org1",
            user_id="u1",
            project_id="proj1",
            session_id="s1",
            rerank=True,
            max_tokens=2048,
        )

        assert result.metadata.ranking_strategy == "client_rrf"
        assert result.metadata.source_outcomes is not None
        assert result.metadata.source_outcomes[0]["source"] == "semantic"
        kwargs = engine.build_context2.call_args[1]
        assert kwargs["org_id"] == "org1"
        assert kwargs["project_id"] == "proj1"
        assert kwargs["session_id"] == "s1"
        assert kwargs["rerank"] is True
        assert kwargs["max_tokens"] == 2048

    def test_omits_optional_kwargs_when_absent(self):
        engine = Mock()
        engine.build_context2.return_value = _response_with_metadata()

        memory_build_context_from_sources(
            memory_engine=engine,
            query="q",
            sources=[SourceSpec(source=MemorySource.SEMANTIC)],
            org_id="org1",
            user_id="u1",
            project_id="proj1",
        )

        kwargs = engine.build_context2.call_args[1]
        assert "session_id" not in kwargs
        assert "max_tokens" not in kwargs

    def test_returns_empty_response_when_engine_raises(self):
        engine = Mock()
        engine.build_context2.side_effect = Exception("boom")

        result = memory_build_context_from_sources(
            memory_engine=engine,
            query="q",
            sources=[SourceSpec(source=MemorySource.SEMANTIC)],
            org_id="org1",
            user_id="u1",
            project_id="proj1",
        )

        assert isinstance(result, ContextResponse)
        assert result.formatted_context == ""

    @pytest.mark.parametrize("max_tokens", [0, -1, 1.5])
    def test_rejects_invalid_max_tokens(self, max_tokens: object):
        engine = Mock()

        with pytest.raises(ValueError, match="^max_tokens must be a positive integer$"):
            memory_build_context_from_sources(
                memory_engine=engine,
                query="q",
                sources=[SourceSpec(source=MemorySource.SEMANTIC)],
                org_id="org1",
                user_id="u1",
                project_id="proj1",
                max_tokens=max_tokens,  # type: ignore[arg-type]
            )

        engine.build_context2.assert_not_called()


class TestRuntimeBuildContextFromSources:
    """TenantRuntime per-source context delegation and guards."""

    def test_returns_empty_response_when_memory_disabled(self, monkeypatch):
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = None

        result = runtime.build_context_from_sources(
            query="q",
            sources=[SourceSpec(source=MemorySource.SEMANTIC)],
            user_id="user_123",
        )

        assert isinstance(result, ContextResponse)
        assert result.formatted_context == ""

    def test_stm_source_without_session_id_returns_empty(self, monkeypatch):
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()

        result = runtime.build_context_from_sources(
            query="q",
            sources=[SourceSpec(source=MemorySource.STM)],
            user_id="user_123",
            session_id=None,
        )

        assert result.formatted_context == ""
        runtime._memory_engine.build_context2.assert_not_called()

    def test_non_stm_sources_do_not_require_session_id(self, monkeypatch):
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        runtime._memory_engine.build_context2.return_value = _response_with_metadata()

        result = runtime.build_context_from_sources(
            query="q",
            sources=[
                SourceSpec(source=MemorySource.SEMANTIC),
                SourceSpec(source=MemorySource.EPISODIC),
            ],
            user_id="user_123",
            session_id=None,
        )

        assert result.metadata.ranking_strategy == "client_rrf"
        runtime._memory_engine.build_context2.assert_called_once()
        kwargs = runtime._memory_engine.build_context2.call_args[1]
        assert kwargs["org_id"] == "org1"
        assert kwargs["project_id"] == "proj1"

    def test_durable_dispatch_round_trips_full_response(self, monkeypatch):
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = Mock()
        runtime._memory_engine.build_context2.return_value = _response_with_metadata()
        workflow = _FakeWorkflow(ActivityDispatch(context=_activity_context()))

        with _durable_execution(workflow):
            result = runtime.build_context_from_sources(
                query="coverage question",
                sources=[SourceSpec(source=MemorySource.SEMANTIC, mode=RetrievalMode.HYBRID)],
                user_id="user-123",
                session_id="session-456",
                rerank=True,
            )

        # The metadata survives the checkpoint round-trip.
        assert isinstance(result, ContextResponse)
        assert result.metadata.ranking_strategy == "client_rrf"
        assert result.metadata.source_outcomes is not None
        assert result.metadata.source_outcomes[0]["source"] == "semantic"

        (command,) = workflow.commands
        assert command.activity_kind == ACTIVITY_KIND_MEMORY
        assert command.activity_name == "memory.build_context_from_sources"
        semantic_input = value_to_json(command.semantic_input)
        assert semantic_input["query"] == "coverage question"
        assert semantic_input["rerank"] is True
        assert semantic_input["sources"] == [{"source": "semantic", "mode": "hybrid", "top_k": 20}]
        (outcome,) = workflow.outcomes
        # The checkpointed result is a JSON-shaped dict, not a bare string.
        assert value_to_json(outcome.result)["metadata"]["ranking_strategy"] == ("client_rrf")

    def test_durable_replay_reconstructs_response_without_engine(self, monkeypatch):
        # The replay path is the one that actually decodes through the protobuf
        # Value boundary (the live dispatch path returns the pre-serialization
        # Python object). Use the maximal fixture and check every field so a
        # silent drop or wrong coercion of any nested shape would fail here.
        monkeypatch.setenv("ORG_ID", "org1")
        monkeypatch.setenv("PROJECT_ID", "proj1")
        runtime = TenantRuntime(app_name="Test")
        runtime._memory_engine = None
        recorded = completed_outcome(
            _activity_context(),
            _full_response().model_dump(mode="json"),
        )
        workflow = _FakeWorkflow(ActivityReplay(outcome=recorded))

        with _durable_execution(workflow):
            result = runtime.build_context_from_sources(
                query="coverage question",
                sources=[SourceSpec(source=MemorySource.SEMANTIC)],
                user_id="user-123",
                session_id="session-456",
            )

        assert isinstance(result, ContextResponse)
        # List-formatted context survives as a list, not stringified.
        assert result.formatted_context == [{"role": "system", "text": "recent turn"}]
        # Typed integer maps round-trip as ints despite the double-only wire.
        assert result.metadata.token_count == 42
        assert result.metadata.memory_counts == {"semantic": 2, "episodic": 1}
        assert result.metadata.timing == {"retrieval": 0.123, "rerank": 0.045}
        assert result.metadata.ranking_strategy == "voyage_rerank"
        # A None inside a nested dict survives as None (not "" or missing).
        assert result.metadata.source_outcomes is not None
        assert result.metadata.source_outcomes[0]["error"] is None
        assert result.metadata.source_outcomes[1]["error"] == "degraded"
        # selected_memories and their datetime / nested metadata survive.
        assert result.selected_memories is not None
        assert len(result.selected_memories) == 1
        chunk = result.selected_memories[0]
        assert chunk.id == "mem-1"
        assert chunk.content == "user prefers dark mode"
        assert chunk.source == "semantic"
        assert chunk.timestamp == _CHUNK_TS
        assert chunk.similarity_score == 0.875
        assert chunk.metadata == {"channel": "web", "confidence": 0.5, "label": "theme"}
        # Replay does not re-report an outcome.
        assert workflow.outcomes == []
