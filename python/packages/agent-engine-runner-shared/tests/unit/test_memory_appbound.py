"""Tests for the app-bound adapter pair over a mock TenantRuntime.

No live stack: a ``MagicMock`` stands in for the runtime, and the two
contextvar getters are monkeypatched to drive ``request_context()``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest
from agent_engine_sdk_memory.errors import MemoryNotSupportedError
from agent_engine_sdk_memory.models import (
    ContextMetadata,
    ContextResponse,
    CreateEpisodicResult,
    CreateProceduralResult,
    CreateSemanticResult,
    CreateTaxonomicResult,
    CustomMemoryRetrieveResult,
    CustomMemorySaveResult,
    MemoryChunk,
    MemorySource,
    RetrievalMode,
    SourceSpec,
    WriteTurnResult,
)
from agent_engine_sdk_memory.protocol import (
    MemoryCrudClient,
    MemoryRequestContext,
    MemoryRuntime,
)

from agent_engine_runner_shared.memory_appbound import (
    AppBoundCrudClient,
    AppBoundRuntime,
)

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# AppBoundRuntime
# ---------------------------------------------------------------------------


def test_appbound_runtime_satisfies_protocol() -> None:
    rt = AppBoundRuntime(MagicMock())
    assert isinstance(rt, MemoryRuntime)


def test_request_context_reads_contextvars(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_engine_runner_shared.memory_appbound as mod

    monkeypatch.setattr(mod, "get_current_user_id", lambda: "u-1")
    monkeypatch.setattr(mod, "get_current_session_id", lambda: "s-1")

    rt = AppBoundRuntime(MagicMock())
    ctx = rt.request_context()
    assert ctx == MemoryRequestContext(user_id="u-1", session_id="s-1", agent_id=None)


def test_request_context_is_fresh_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_engine_runner_shared.memory_appbound as mod

    monkeypatch.setattr(mod, "get_current_user_id", lambda: "u-1")
    monkeypatch.setattr(mod, "get_current_session_id", lambda: "s-1")
    rt = AppBoundRuntime(MagicMock())
    first = rt.request_context()
    assert first is not None
    assert first.user_id == "u-1"

    monkeypatch.setattr(mod, "get_current_user_id", lambda: "u-2")
    monkeypatch.setattr(mod, "get_current_session_id", lambda: "s-2")
    second = rt.request_context()
    assert second is not None
    assert second.user_id == "u-2"
    assert second.session_id == "s-2"


def test_build_context_wraps_string() -> None:
    tenant = MagicMock()
    tenant.build_context.return_value = "formatted context here"
    rt = AppBoundRuntime(tenant)

    out = rt.build_context(
        query="q",
        user_id="u",
        session_id="s",
        metadata_filter={"k": "v"},
        enabled_sources={"semantic"},
        top_k=3,
    )
    assert isinstance(out, ContextResponse)
    assert out.formatted_context == "formatted context here"
    tenant.build_context.assert_called_once()
    _, kwargs = tenant.build_context.call_args
    assert kwargs["query"] == "q"
    assert kwargs["user_id"] == "u"
    assert kwargs["session_id"] == "s"
    assert kwargs["metadata_filter"] == {"k": "v"}
    assert kwargs["enabled_sources"] == {"semantic"}
    # top_k is accepted for interface parity but not forwarded.
    assert "top_k" not in kwargs


def test_build_context_forwards_explicit_max_tokens() -> None:
    tenant = MagicMock()
    tenant.build_context.return_value = "ctx"
    rt = AppBoundRuntime(tenant)

    out = rt.build_context(query="q", max_tokens=2048)
    assert isinstance(out, ContextResponse)
    assert out.formatted_context == "ctx"
    _, kwargs = tenant.build_context.call_args
    assert kwargs["max_tokens"] == 2048
    assert "top_k" not in kwargs


def test_build_context_omits_max_tokens_when_absent() -> None:
    tenant = MagicMock()
    tenant.build_context.return_value = "ctx"
    rt = AppBoundRuntime(tenant)

    out = rt.build_context(query="q", top_k=3)
    assert isinstance(out, ContextResponse)
    _, kwargs = tenant.build_context.call_args
    assert "max_tokens" not in kwargs
    assert "top_k" not in kwargs


def test_build_context_rejects_format_style_and_include_memories() -> None:
    # The app-bound path flattens to a jinja2 string, so response shaping cannot
    # be honored; it is rejected loudly rather than silently dropped.
    tenant = MagicMock()
    rt = AppBoundRuntime(tenant)

    with pytest.raises(MemoryNotSupportedError, match="format_style"):
        rt.build_context(query="q", format_style="openai")
    with pytest.raises(MemoryNotSupportedError, match="format_style"):
        rt.build_context(query="q", include_memories=True)
    tenant.build_context.assert_not_called()


def test_build_context_from_sources_passes_response_through() -> None:
    tenant = MagicMock()
    # Unlike build_context (which returns a bare string), the per-source path
    # returns a full ContextResponse whose per-source metadata must survive.
    tenant.build_context_from_sources.return_value = ContextResponse(
        formatted_context="merged context",
        metadata=ContextMetadata(
            ranking_strategy="client_rrf",
            source_outcomes=[
                {
                    "source": "semantic",
                    "requested_mode": "hybrid",
                    "effective_mode": "hybrid",
                    "count": 2,
                    "error": None,
                }
            ],
        ),
    )
    rt = AppBoundRuntime(tenant)

    out = rt.build_context_from_sources(
        query="q",
        sources=[SourceSpec(source=MemorySource.SEMANTIC, mode=RetrievalMode.HYBRID)],
        user_id="u",
        session_id="s",
        rerank=True,
        max_tokens=2048,
    )

    assert isinstance(out, ContextResponse)
    assert out.formatted_context == "merged context"
    # The response is returned verbatim, not re-wrapped, so metadata is intact.
    assert out.metadata.ranking_strategy == "client_rrf"
    assert out.metadata.source_outcomes is not None
    assert out.metadata.source_outcomes[0]["source"] == "semantic"

    tenant.build_context_from_sources.assert_called_once()
    _, kwargs = tenant.build_context_from_sources.call_args
    assert kwargs["query"] == "q"
    assert kwargs["user_id"] == "u"
    assert kwargs["session_id"] == "s"
    assert kwargs["rerank"] is True
    assert kwargs["max_tokens"] == 2048
    assert [s.source for s in kwargs["sources"]] == [MemorySource.SEMANTIC.value]
    # build_context is the flat path; it must not be used here.
    tenant.build_context.assert_not_called()


def test_build_context_from_sources_rejects_format_style_and_include_memories() -> None:
    # Not threaded through TenantRuntime yet; rejected loudly rather than
    # silently dropped.
    tenant = MagicMock()
    rt = AppBoundRuntime(tenant)

    with pytest.raises(MemoryNotSupportedError, match="format_style"):
        rt.build_context_from_sources(
            query="q",
            sources=[SourceSpec(source=MemorySource.SEMANTIC)],
            format_style="openai",
        )
    with pytest.raises(MemoryNotSupportedError, match="format_style"):
        rt.build_context_from_sources(
            query="q",
            sources=[SourceSpec(source=MemorySource.SEMANTIC)],
            include_memories=True,
        )
    tenant.build_context_from_sources.assert_not_called()


def test_build_context_empty_string_no_crash() -> None:
    tenant = MagicMock()
    tenant.build_context.return_value = ""
    rt = AppBoundRuntime(tenant)

    out = rt.build_context(query="q")
    assert isinstance(out, ContextResponse)
    assert out.formatted_context == ""


def test_search_semantic_maps_dicts_to_chunks() -> None:
    ts = datetime(2026, 6, 17)
    tenant = MagicMock()
    tenant.search_semantic.return_value = [
        {
            "id": "doc-1",
            "content": "hello",
            "timestamp": ts,
            "similarity_score": 0.9,
            "label": "greeting",
        }
    ]
    rt = AppBoundRuntime(tenant)

    chunks = rt.search_semantic(query="q", user_id="u")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, MemoryChunk)
    assert chunk.id == "doc-1"
    assert chunk.content == "hello"
    assert chunk.source == MemorySource.SEMANTIC.value
    assert chunk.similarity_score == 0.9
    assert chunk.metadata is not None
    # COMPLETE source dict is preserved under metadata (no popping).
    assert chunk.metadata["label"] == "greeting"
    assert chunk.metadata["content"] == "hello"
    assert chunk.metadata["id"] == "doc-1"


def test_search_semantic_missing_fields_still_valid() -> None:
    tenant = MagicMock()
    tenant.search_semantic.return_value = [{"label": "no-content-no-id"}]
    rt = AppBoundRuntime(tenant)

    chunks = rt.search_semantic(query="q")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.id == ""
    assert chunk.content == ""
    # Missing timestamps use epoch, not query-time now (recency safety).
    assert chunk.timestamp == _EPOCH
    assert chunk.metadata is not None
    assert chunk.metadata["label"] == "no-content-no-id"


def test_search_semantic_uses_mongo_id_and_text_fallback() -> None:
    tenant = MagicMock()
    tenant.search_semantic.return_value = [{"_id": 123, "text": "from text field"}]
    rt = AppBoundRuntime(tenant)

    chunk = rt.search_semantic(query="q")[0]
    assert chunk.id == "123"
    assert chunk.content == "from text field"


def test_search_semantic_parses_created_at_iso_string() -> None:
    tenant = MagicMock()
    tenant.search_semantic.return_value = [{"text": "x", "created_at": "2024-06-01T12:00:00Z"}]
    rt = AppBoundRuntime(tenant)

    chunk = rt.search_semantic(query="q")[0]
    assert chunk.timestamp == datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_search_episodes_source_is_episodic() -> None:
    tenant = MagicMock()
    tenant.search_episodes.return_value = [{"id": "e-1", "content": "ep"}]
    rt = AppBoundRuntime(tenant)

    chunk = rt.search_episodes(query="q", session_id="s")[0]
    assert chunk.source == MemorySource.EPISODIC.value
    _, kwargs = tenant.search_episodes.call_args
    assert kwargs["session_id"] == "s"


def test_search_episodes_summary_fallback_for_content() -> None:
    tenant = MagicMock()
    tenant.search_episodes.return_value = [{"title": "t", "summary": "short summary"}]
    rt = AppBoundRuntime(tenant)

    chunk = rt.search_episodes(query="q", session_id="s")[0]
    assert chunk.content == "short summary"


def test_search_episodes_empty_content_falls_through_to_summary() -> None:
    # TenantRuntime always emits a content key (often ""); blank must not
    # block the summary fallback.
    tenant = MagicMock()
    tenant.search_episodes.return_value = [{"title": "t", "content": "", "summary": "from summary"}]
    rt = AppBoundRuntime(tenant)

    chunk = rt.search_episodes(query="q", session_id="s")[0]
    assert chunk.content == "from summary"


def test_search_taxonomic_source_is_taxonomic() -> None:
    tenant = MagicMock()
    tenant.search_taxonomic.return_value = [
        {
            "id": "t-1",
            "definition": "Amount paid out of pocket",
            "term": "deductible",
            "domain": "insurance",
        }
    ]
    rt = AppBoundRuntime(tenant)

    chunk = rt.search_taxonomic(query="q", domain="insurance")[0]
    assert chunk.source == MemorySource.TAXONOMIC.value
    # Compose matches memory-server MemoryChunk.from_taxonomic.
    assert chunk.content == "insurance/deductible: Amount paid out of pocket"
    assert chunk.metadata is not None
    assert chunk.metadata["term"] == "deductible"
    assert chunk.metadata["domain"] == "insurance"
    assert chunk.metadata["definition"] == "Amount paid out of pocket"
    _, kwargs = tenant.search_taxonomic.call_args
    assert kwargs["domain"] == "insurance"


def test_search_taxonomic_content_without_domain() -> None:
    tenant = MagicMock()
    tenant.search_taxonomic.return_value = [{"term": "premium", "definition": "Periodic payment"}]
    rt = AppBoundRuntime(tenant)

    chunk = rt.search_taxonomic(query="q")[0]
    assert chunk.content == "premium: Periodic payment"


def test_search_taxonomic_definition_only_content() -> None:
    tenant = MagicMock()
    tenant.search_taxonomic.return_value = [{"definition": "only def"}]
    rt = AppBoundRuntime(tenant)

    chunk = rt.search_taxonomic(query="q")[0]
    assert chunk.content == "only def"


def test_discover_procedures_returns_list_unchanged() -> None:
    tenant = MagicMock()
    payload: list[dict[str, Any]] = [{"procedure": "p1"}, {"procedure": "p2"}]
    tenant.discover_procedures.return_value = payload
    rt = AppBoundRuntime(tenant)

    out = rt.discover_procedures(
        query="q",
        top_k=7,
        user_id="u-1",
        visibility="private",
        metadata_filter={"k": "v"},
    )
    assert out == payload
    args, kwargs = tenant.discover_procedures.call_args
    assert args == ()
    assert kwargs["query"] == "q"
    assert kwargs["user_id"] == "u-1"
    assert kwargs["top_k"] == 7
    assert kwargs["visibility"] == "private"
    assert kwargs["metadata_filter"] == {"k": "v"}


def test_record_turn_user_maps_to_message() -> None:
    tenant = MagicMock()
    # memory_writer truthy so preflight queues the write
    tenant.memory_writer = object()
    rt = AppBoundRuntime(tenant)

    out = rt.record_turn(role="user", content="hi", session_id="s-1", user_id="u-1")
    assert isinstance(out, WriteTurnResult)
    assert out.acknowledged is True
    assert out.session_id == "s-1"
    tenant.write_turn_async.assert_called_once_with(
        message="hi",
        result_messages=[],
        user_id="u-1",
        session_id="s-1",
        include_user_turn=True,
        metadata=None,
        idempotency_key=None,
    )


def test_record_turn_forwards_idempotency_key_user() -> None:
    tenant = MagicMock()
    tenant.memory_writer = object()
    rt = AppBoundRuntime(tenant)

    rt.record_turn(
        role="user",
        content="hi",
        session_id="s-1",
        user_id="u-1",
        idempotency_key="turn-42",
    )
    _, kwargs = tenant.write_turn_async.call_args
    assert kwargs["idempotency_key"] == "turn-42"


def test_record_turn_forwards_idempotency_key_assistant() -> None:
    tenant = MagicMock()
    tenant.memory_writer = object()
    rt = AppBoundRuntime(tenant)

    rt.record_turn(
        role="assistant",
        content="Noted",
        session_id="s-1",
        user_id="u-1",
        idempotency_key="turn-43",
    )
    _, kwargs = tenant.write_turn_async.call_args
    assert kwargs["idempotency_key"] == "turn-43"


def test_record_turn_assistant_maps_to_result_messages() -> None:
    tenant = MagicMock()
    tenant.memory_writer = object()
    rt = AppBoundRuntime(tenant)

    out = rt.record_turn(role="assistant", content="Noted", session_id="s-1", user_id="u-1")
    assert out.acknowledged is True
    tenant.write_turn_async.assert_called_once()
    _, kwargs = tenant.write_turn_async.call_args
    assert kwargs["message"] == ""
    assert kwargs["include_user_turn"] is False
    assert kwargs["session_id"] == "s-1"
    assert kwargs["user_id"] == "u-1"
    assert len(kwargs["result_messages"]) == 1
    msg = kwargs["result_messages"][0]
    assert msg.role == "assistant"
    assert msg.content == "Noted"
    assert msg.tool_calls is None


def test_record_turn_role_case_insensitive() -> None:
    tenant = MagicMock()
    tenant.memory_writer = object()
    rt = AppBoundRuntime(tenant)

    rt.record_turn(role="USER", content="hi", session_id="s-1", user_id="u-1")
    _, user_kwargs = tenant.write_turn_async.call_args
    assert user_kwargs["message"] == "hi"
    assert user_kwargs["include_user_turn"] is True

    tenant.reset_mock()
    tenant.memory_writer = object()
    rt.record_turn(role="Assistant", content="ok", session_id="s-1", user_id="u-1")
    _, asst_kwargs = tenant.write_turn_async.call_args
    assert asst_kwargs["include_user_turn"] is False
    assert asst_kwargs["result_messages"][0].role == "assistant"


def test_record_turn_not_acknowledged_when_memory_disabled() -> None:
    tenant = MagicMock()
    tenant.memory_writer = None
    rt = AppBoundRuntime(tenant)

    out = rt.record_turn(role="user", content="hi", session_id="s-1", user_id="u-1")
    assert out.acknowledged is False
    tenant.write_turn_async.assert_not_called()


def test_record_turn_not_acknowledged_when_identity_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_engine_runner_shared.memory_appbound as mod

    monkeypatch.setattr(mod, "get_current_user_id", lambda: None)
    monkeypatch.setattr(mod, "get_current_session_id", lambda: None)
    tenant = MagicMock()
    tenant.memory_writer = object()
    rt = AppBoundRuntime(tenant)

    out = rt.record_turn(role="user", content="hi")
    assert out.acknowledged is False
    tenant.write_turn_async.assert_not_called()


def test_record_turn_not_acknowledged_when_content_blank() -> None:
    tenant = MagicMock()
    tenant.memory_writer = object()
    rt = AppBoundRuntime(tenant)

    out = rt.record_turn(role="user", content="   ", session_id="s-1", user_id="u-1")
    assert out.acknowledged is False
    tenant.write_turn_async.assert_not_called()


@pytest.mark.parametrize("role", ["tool", "system", "human", "garbage"])
def test_record_turn_rejects_non_user_assistant_roles(role: str) -> None:
    tenant = MagicMock()
    tenant.memory_writer = object()
    rt = AppBoundRuntime(tenant)

    with pytest.raises(MemoryNotSupportedError, match="role="):
        rt.record_turn(role=role, content="x", session_id="s-1", user_id="u-1")
    tenant.write_turn_async.assert_not_called()


def test_record_turn_rejects_tool_metadata() -> None:
    # write_turn_async carries only (message, result_messages); per-turn tool
    # metadata cannot be represented, so it is rejected rather than dropped.
    tenant = MagicMock()
    tenant.memory_writer = object()
    rt = AppBoundRuntime(tenant)

    with pytest.raises(MemoryNotSupportedError):
        rt.record_turn(
            role="tool",
            content="result",
            tool_call_id="tc-1",
            tool_name="lookup",
            session_id="s-1",
            user_id="u-1",
        )
    tenant.write_turn_async.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_name": "gpt-x"},
        {"is_error": True},
        {"tool_calls": [{"id": "1", "name": "t", "arguments": {}}]},
    ],
)
def test_record_turn_rejects_metadata_on_allowed_roles(kwargs: dict) -> None:
    tenant = MagicMock()
    tenant.memory_writer = object()
    rt = AppBoundRuntime(tenant)

    with pytest.raises(MemoryNotSupportedError, match="tool-call and model"):
        rt.record_turn(
            role="user",
            content="hi",
            session_id="s-1",
            user_id="u-1",
            **kwargs,
        )
    tenant.write_turn_async.assert_not_called()


def test_record_turn_forwards_metadata_user() -> None:
    tenant = MagicMock()
    tenant.memory_writer = object()
    rt = AppBoundRuntime(tenant)

    rt.record_turn(
        role="user",
        content="hi",
        session_id="s-1",
        user_id="u-1",
        metadata={"channel": "slack"},
    )
    _, kwargs = tenant.write_turn_async.call_args
    assert kwargs["metadata"] == {"channel": "slack"}


def test_record_turn_forwards_metadata_assistant() -> None:
    tenant = MagicMock()
    tenant.memory_writer = object()
    rt = AppBoundRuntime(tenant)

    rt.record_turn(
        role="assistant",
        content="Noted",
        session_id="s-1",
        user_id="u-1",
        metadata={"channel": "slack"},
    )
    _, kwargs = tenant.write_turn_async.call_args
    assert kwargs["metadata"] == {"channel": "slack"}


def test_record_turn_metadata_defaults_none() -> None:
    tenant = MagicMock()
    tenant.memory_writer = object()
    rt = AppBoundRuntime(tenant)

    rt.record_turn(role="user", content="hi", session_id="s-1", user_id="u-1")
    _, kwargs = tenant.write_turn_async.call_args
    assert kwargs["metadata"] is None


# ---------------------------------------------------------------------------
# AppBoundCrudClient
# ---------------------------------------------------------------------------


def test_appbound_crud_satisfies_protocol() -> None:
    crud = AppBoundCrudClient(MagicMock())
    assert isinstance(crud, MemoryCrudClient)


def test_create_semantic_maps_bool_to_acknowledged() -> None:
    tenant = MagicMock()
    tenant.save_semantic.return_value = True
    crud = AppBoundCrudClient(tenant)

    meta = {"k": "v"}
    res = crud.create_semantic(
        label="L",
        text="T",
        user_id="u",
        agent_id="agent-1",
        metadata=meta,
        upsert=True,
    )
    assert isinstance(res, CreateSemanticResult)
    assert res.label == "L"
    assert res.has_embedding is False
    assert res.acknowledged is True
    assert res.id == ""

    # Protocol → TenantRuntime: metadata is forwarded as metadata; upsert must
    # be forwarded (tenant default is True, protocol default is False).
    _, kwargs = tenant.save_semantic.call_args
    assert kwargs["metadata"] is meta
    assert "contextual_metadata" not in kwargs
    assert kwargs["upsert"] is True
    assert kwargs["agent_id"] == "agent-1"


def test_create_semantic_false_is_not_acknowledged() -> None:
    tenant = MagicMock()
    tenant.save_semantic.return_value = False
    crud = AppBoundCrudClient(tenant)

    res = crud.create_semantic(label="L", text="T", user_id="u")
    assert res.acknowledged is False


def test_create_episodic_maps_doc_id() -> None:
    tenant = MagicMock()
    tenant.save_episode.return_value = "ep-99"
    crud = AppBoundCrudClient(tenant)

    res = crud.create_episodic(
        title="E",
        content="C",
        user_id="u",
        session_id="s",
        summary_text="short summary",
        agent_id="agent-1",
    )
    assert isinstance(res, CreateEpisodicResult)
    assert res.id == "ep-99"
    assert res.title == "E"
    assert res.acknowledged is True

    # Protocol → TenantRuntime: summary_text remaps to summary.
    _, kwargs = tenant.save_episode.call_args
    assert kwargs["summary"] == "short summary"
    assert "summary_text" not in kwargs
    assert kwargs["agent_id"] == "agent-1"


def test_create_episodic_none_doc_id_not_acknowledged() -> None:
    tenant = MagicMock()
    tenant.save_episode.return_value = None
    crud = AppBoundCrudClient(tenant)

    res = crud.create_episodic(title="E", content="C", user_id="u", session_id="s")
    assert res.id == ""
    assert res.acknowledged is False


def test_create_taxonomic_maps_doc_id() -> None:
    tenant = MagicMock()
    tenant.create_taxonomic.return_value = "tax-1"
    crud = AppBoundCrudClient(tenant)

    res = crud.create_taxonomic(domain="d", term="t", definition="def", user_id="u")
    assert isinstance(res, CreateTaxonomicResult)
    assert res.id == "tax-1"
    assert res.domain == "d"
    assert res.term == "t"
    assert res.acknowledged is True


def test_create_taxonomic_forwards_metadata() -> None:
    tenant = MagicMock()
    tenant.create_taxonomic.return_value = "tax-1"
    crud = AppBoundCrudClient(tenant)

    crud.create_taxonomic(
        domain="d",
        term="t",
        definition="def",
        user_id="u",
        metadata={"origin": "import"},
    )
    _, kwargs = tenant.create_taxonomic.call_args
    assert kwargs["metadata"] == {"origin": "import"}


def test_create_procedural_maps_dict() -> None:
    tenant = MagicMock()
    tenant.save_procedure.return_value = {
        "id": "proc-1",
        "procedure": "deploy",
        "has_embedding": True,
    }
    crud = AppBoundCrudClient(tenant)

    res = crud.create_procedural(
        procedure="deploy",
        description="d",
        content="c",
        user_id="u",
        update_existing=True,
    )
    assert isinstance(res, CreateProceduralResult)
    assert res.id == "proc-1"
    assert res.procedure == "deploy"
    assert res.has_embedding is True
    assert res.acknowledged is True
    _, kwargs = tenant.save_procedure.call_args
    assert kwargs["update_existing"] is True


def test_create_procedural_forwards_metadata() -> None:
    tenant = MagicMock()
    tenant.save_procedure.return_value = {"id": "proc-1", "procedure": "deploy"}
    crud = AppBoundCrudClient(tenant)

    crud.create_procedural(
        procedure="deploy",
        description="d",
        content="c",
        user_id="u",
        metadata={"origin": "import"},
    )
    _, kwargs = tenant.save_procedure.call_args
    assert kwargs["metadata"] == {"origin": "import"}


def test_create_procedural_none_not_acknowledged() -> None:
    tenant = MagicMock()
    tenant.save_procedure.return_value = None
    crud = AppBoundCrudClient(tenant)

    res = crud.create_procedural(procedure="deploy", description="d", content="c", user_id="u")
    assert res.id == ""
    assert res.procedure == "deploy"
    assert res.has_embedding is False
    assert res.acknowledged is False


def test_create_procedural_honors_backend_acknowledged_false() -> None:
    # TenantRuntime create_fallback can return a dict with acknowledged=False.
    tenant = MagicMock()
    tenant.save_procedure.return_value = {
        "id": "proc-2",
        "procedure": "deploy",
        "has_embedding": False,
        "acknowledged": False,
    }
    crud = AppBoundCrudClient(tenant)

    res = crud.create_procedural(procedure="deploy", description="d", content="c", user_id="u")
    assert res.id == "proc-2"
    assert res.acknowledged is False


def test_get_semantic_forwards_and_returns() -> None:
    tenant = MagicMock()
    tenant.get_semantic.return_value = {"label": "L"}
    crud = AppBoundCrudClient(tenant)

    assert crud.get_semantic(label="L", user_id="u") == {"label": "L"}


def test_get_taxonomic_without_term_is_unsupported() -> None:
    crud = AppBoundCrudClient(MagicMock())
    with pytest.raises(MemoryNotSupportedError):
        crud.get_taxonomic(domain="d", term=None)


def test_get_procedural_without_name_is_unsupported() -> None:
    crud = AppBoundCrudClient(MagicMock())
    with pytest.raises(MemoryNotSupportedError):
        crud.get_procedural(procedure=None)


def test_get_taxonomic_forwards_term() -> None:
    tenant = MagicMock()
    tenant.get_taxonomic_term.return_value = {"term": "t"}
    crud = AppBoundCrudClient(tenant)

    assert crud.get_taxonomic(domain="d", term="t") == {"term": "t"}
    _, kwargs = tenant.get_taxonomic_term.call_args
    assert kwargs["domain"] == "d"
    assert kwargs["term"] == "t"


def test_get_distinct_domains_returns_list() -> None:
    tenant = MagicMock()
    tenant.list_taxonomic_domains.return_value = ["a", "b"]
    crud = AppBoundCrudClient(tenant)

    assert crud.get_distinct_domains() == ["a", "b"]


def test_list_episodic_forwards() -> None:
    tenant = MagicMock()
    tenant.list_episodes.return_value = [{"id": "e-1"}]
    crud = AppBoundCrudClient(tenant)

    assert crud.list_episodic(user_id="u", session_id="s") == [{"id": "e-1"}]


def test_get_procedural_maps_kwarg_name() -> None:
    tenant = MagicMock()
    tenant.get_procedure.return_value = {"procedure": "deploy"}
    crud = AppBoundCrudClient(tenant)

    assert crud.get_procedural(procedure="deploy") == {"procedure": "deploy"}
    _, kwargs = tenant.get_procedure.call_args
    assert kwargs["procedure_name"] == "deploy"


# ---------------------------------------------------------------------------
# Ambient identity end-to-end through the unified facade
# ---------------------------------------------------------------------------


def test_facade_save_semantic_uses_ambient_user_id() -> None:
    """The facade resolves user_id from the ambient context with no call arg."""
    from agent_engine_sdk_memory import Memory

    from agent_engine_runner_shared.context import (
        clear_execution_context,
        set_execution_context,
    )

    tenant = MagicMock()
    tenant.save_semantic.return_value = True
    memory = Memory(
        runtime=AppBoundRuntime(tenant),
        client=AppBoundCrudClient(tenant),
    )

    tokens = set_execution_context(
        execution_id="exec-1",
        wrapper=MagicMock(),
        oe_url="http://oe",
        user_id="ambient-user",
    )
    try:
        result = memory.save_semantic(text="t", label="l")
    finally:
        clear_execution_context(tokens)

    assert isinstance(result, CreateSemanticResult)
    assert result.acknowledged is True
    _, kwargs = tenant.save_semantic.call_args
    assert kwargs["user_id"] == "ambient-user"


# ---------------------------------------------------------------------------
# Identity binding through the unified facade (app-bound)
# ---------------------------------------------------------------------------


def _set_ambient(
    monkeypatch: pytest.MonkeyPatch,
    *,
    user_id: str | None,
    session_id: str | None = None,
) -> None:
    """Point the contextvar getters the app-bound runtime reads."""
    import agent_engine_runner_shared.memory_appbound as mod

    monkeypatch.setattr(mod, "get_current_user_id", lambda: user_id)
    monkeypatch.setattr(mod, "get_current_session_id", lambda: session_id)


def _appbound_facade(tenant: MagicMock) -> Any:
    from agent_engine_sdk_memory import Memory

    return Memory(runtime=AppBoundRuntime(tenant), client=AppBoundCrudClient(tenant))


def test_appbound_runtime_ctx_fills_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no call arg and no bind, ambient identity reaches the tenant."""
    tenant = MagicMock()
    tenant.search_semantic.return_value = []
    _set_ambient(monkeypatch, user_id="ambient-u")
    memory = _appbound_facade(tenant)

    memory.search_semantic("q")

    _, kwargs = tenant.search_semantic.call_args
    assert kwargs["user_id"] == "ambient-u"


def test_appbound_tenancy_comes_from_tenant_not_public_arg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The facade forwards only user identity; org/project stay tenant-internal."""
    tenant = MagicMock()
    tenant.search_semantic.return_value = []
    tenant.save_semantic.return_value = True
    _set_ambient(monkeypatch, user_id="u")
    memory = _appbound_facade(tenant)

    memory.search_semantic("q")
    _, search_kwargs = tenant.search_semantic.call_args
    assert search_kwargs["user_id"] == "u"
    assert "org_id" not in search_kwargs
    assert "project_id" not in search_kwargs

    memory.save_semantic(text="t", label="l")
    _, save_kwargs = tenant.save_semantic.call_args
    assert save_kwargs["user_id"] == "u"
    assert "org_id" not in save_kwargs
    assert "project_id" not in save_kwargs


def test_appbound_no_cross_user_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sequential calls under different ambient users never carry each other's id."""
    tenant = MagicMock()
    tenant.search_semantic.return_value = []
    memory = _appbound_facade(tenant)

    _set_ambient(monkeypatch, user_id="user-a")
    memory.search_semantic("q")
    _, first = tenant.search_semantic.call_args
    assert first["user_id"] == "user-a"

    _set_ambient(monkeypatch, user_id="user-b")
    memory.search_semantic("q")
    _, second = tenant.search_semantic.call_args
    assert second["user_id"] == "user-b"


# ---------------------------------------------------------------------------
# Retry / idempotency: shared HTTP transport vs app-bound delegation
# ---------------------------------------------------------------------------


def test_appbound_record_turn_delegates_without_shared_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_engine_runner_shared.memory_appbound as mod

    monkeypatch.setattr(mod, "get_current_user_id", lambda: "u")
    monkeypatch.setattr(mod, "get_current_session_id", lambda: "s")
    tenant = MagicMock()
    tenant.memory_writer = object()
    memory = _appbound_facade(tenant)

    # App-bound delegates the write to the TenantRuntime, which owns retry and
    # durability. There is no shared HTTP transport loop here; acknowledged is
    # True only when the write is queued (writer present + resolvable identity +
    # non-empty content), not a blank success claim. Facade record_turn has no
    # user_id kwarg — identity comes from ambient request_context.
    result = memory.record_turn(role="user", content="hi", session_id="s")

    tenant.write_turn_async.assert_called_once()
    assert result.acknowledged is True


# ---------------------------------------------------------------------------
# Custom-type memory
# ---------------------------------------------------------------------------


def test_create_custom_delegates_to_tenant_runtime() -> None:
    tenant = MagicMock()
    tenant.save_custom.return_value = CustomMemorySaveResult(
        id="m1", type="tickets", tags={"queue": "billing"}, has_embedding=True
    )
    client = AppBoundCrudClient(tenant)
    result = client.create_custom(
        memory_type="tickets",
        content="c",
        tags={"queue": "billing"},
        contextual_metadata=None,
    )
    tenant.save_custom.assert_called_once_with(
        memory_type="tickets",
        content="c",
        tags={"queue": "billing"},
        contextual_metadata=None,
    )
    assert result.id == "m1"


def test_retrieve_custom_delegates_to_tenant_runtime() -> None:
    tenant = MagicMock()
    tenant.retrieve_custom.return_value = CustomMemoryRetrieveResult(results=[], count=0)
    client = AppBoundCrudClient(tenant)
    result = client.retrieve_custom(memory_type="tickets", query="q", tags=None, top_k=10)
    tenant.retrieve_custom.assert_called_once_with(
        memory_type="tickets",
        query="q",
        tags=None,
        top_k=10,
    )
    assert result.count == 0
