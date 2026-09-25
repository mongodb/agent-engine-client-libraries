"""Tests for the transport-free Memory facade."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any
from unittest.mock import Mock

import pytest
from agent_engine_sdk_memory._http_runtime import _OE_PROFILE
from agent_engine_sdk_memory.errors import MemoryIdentityError
from agent_engine_sdk_memory.memory import Memory
from agent_engine_sdk_memory.models import (
    ContextMetadata,
    ContextResponse,
    CreateEpisodicResult,
    CreateProceduralResult,
    CreateSemanticResult,
    CreateTaxonomicResult,
    CustomMemoryRetrieveResult,
    CustomMemorySaveResult,
    FormatStyle,
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


class FakeRuntime:
    """Records the kwargs every Protocol op received.

    Method signatures intentionally mirror ``MemoryRuntime`` exactly (no
    ``**kwargs`` catch-alls), so any drift between the facade's call shape and
    the port contract fails these tests with a TypeError instead of passing
    silently.
    """

    def __init__(self) -> None:
        self.calls: dict[str, dict[str, Any]] = {}

    def _record(self, name: str, args: dict[str, Any]) -> None:
        self.calls[name] = {k: v for k, v in args.items() if k != "self"}

    def record_turn(
        self,
        *,
        role: str,
        content: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        is_error: bool = False,
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> WriteTurnResult:
        self._record("record_turn", locals())
        return WriteTurnResult(id="1", session_id="s", turn_seq=0)

    def build_context(
        self,
        *,
        query: str,
        session_id: str | None = None,
        user_id: str | None = None,
        visibility: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        enabled_sources: set[str] | None = None,
        top_k: int = 50,
        max_tokens: int | None = None,
        format_style: FormatStyle | str | None = None,
        include_memories: bool = False,
    ) -> ContextResponse:
        self._record("build_context", locals())
        return ContextResponse(formatted_context="ctx", metadata=ContextMetadata())

    def build_context_from_sources(
        self,
        *,
        query: str,
        sources: list[SourceSpec],
        session_id: str | None = None,
        user_id: str | None = None,
        visibility: str | None = None,
        rerank: bool = False,
        max_tokens: int | None = None,
        format_style: FormatStyle | str | None = None,
        include_memories: bool = False,
    ) -> ContextResponse:
        self._record("build_context_from_sources", locals())
        return ContextResponse(formatted_context="ctx", metadata=ContextMetadata())

    def search_semantic(
        self,
        *,
        query: str,
        user_id: str | None = None,
        visibility: str | None = None,
        top_k: int = 50,
    ) -> list[MemoryChunk]:
        self._record("search_semantic", locals())
        return []

    def search_episodes(
        self,
        *,
        query: str,
        user_id: str | None = None,
        visibility: str | None = None,
        session_id: str | None = None,
        top_k: int = 50,
    ) -> list[MemoryChunk]:
        self._record("search_episodes", locals())
        return []

    def search_taxonomic(
        self,
        *,
        query: str,
        user_id: str | None = None,
        domain: str | None = None,
        visibility: str | None = None,
        top_k: int = 50,
    ) -> list[MemoryChunk]:
        self._record("search_taxonomic", locals())
        return []

    def discover_procedures(
        self,
        *,
        query: str,
        user_id: str | None = None,
        visibility: str | None = None,
        tags: list[str] | None = None,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self._record("discover_procedures", locals())
        return []


class FakeClient:
    """Records the kwargs every CRUD method received.

    Method signatures intentionally mirror ``MemoryCrudClient`` exactly (no
    ``**kwargs`` catch-alls), so any drift between the facade's call shape and
    the port contract fails these tests with a TypeError instead of passing
    silently.
    """

    def __init__(self) -> None:
        self.calls: dict[str, dict[str, Any]] = {}

    def _record(self, name: str, args: dict[str, Any]) -> None:
        self.calls[name] = {k: v for k, v in args.items() if k != "self"}

    def create_semantic(
        self,
        *,
        label: str,
        text: str,
        user_id: str,
        source: str = "agent",
        visibility: str = "private",
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        upsert: bool = False,
    ) -> CreateSemanticResult:
        self._record("create_semantic", locals())
        return CreateSemanticResult(id="semantic-id", label=label, has_embedding=False)

    def get_semantic(
        self,
        *,
        label: str,
        user_id: str | None = None,
        visibility: str | None = None,
    ) -> Any | None:
        self._record("get_semantic", locals())
        return {"id": "s"}

    def create_episodic(
        self,
        *,
        title: str,
        content: str,
        user_id: str,
        session_id: str,
        summary_text: str | None = None,
        participants: list[str] | None = None,
        tags: list[str] | None = None,
        visibility: str = "private",
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CreateEpisodicResult:
        self._record("create_episodic", locals())
        return CreateEpisodicResult(id="episode-id", title=title, has_embedding=False)

    def list_episodic(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        visibility: str | None = None,
        limit: int = 20,
    ) -> list[Any]:
        self._record("list_episodic", locals())
        return []

    def create_taxonomic(
        self,
        *,
        domain: str,
        term: str,
        definition: str,
        user_id: str,
        related_terms: list[str] | None = None,
        visibility: str = "org",
        metadata: dict[str, Any] | None = None,
    ) -> CreateTaxonomicResult:
        self._record("create_taxonomic", locals())
        return CreateTaxonomicResult(
            id="tax-id", domain=domain, term=term, has_embedding=False
        )

    def get_taxonomic(
        self,
        *,
        domain: str,
        term: str | None = None,
        user_id: str | None = None,
        visibility: str | None = None,
    ) -> Any | None:
        self._record("get_taxonomic", locals())
        return {"term": "t"}

    def get_distinct_domains(self, *, visibility: str | None = None) -> list[str]:
        self._record("get_distinct_domains", locals())
        return ["d"]

    def create_procedural(
        self,
        *,
        procedure: str,
        description: str,
        content: str,
        user_id: str,
        steps: list[dict[str, Any]] | None = None,
        resources: list[dict[str, Any]] | None = None,
        allowed_tools: list[str] | None = None,
        compatibility: str | None = None,
        license: str | None = None,
        trigger_conditions: list[str] | None = None,
        tags: list[str] | None = None,
        visibility: str = "private",
        agent_id: str | None = None,
        extraction_source: str | None = None,
        source_format: str | None = None,
        source_path: str | None = None,
        metadata: dict[str, Any] | None = None,
        update_existing: bool = False,
    ) -> CreateProceduralResult:
        self._record("create_procedural", locals())
        return CreateProceduralResult(id="p", procedure=procedure, has_embedding=False)

    def get_procedural(
        self,
        *,
        procedure: str | None = None,
        user_id: str | None = None,
        visibility: str | None = None,
        include_deleted: bool = False,
    ) -> Any | None:
        self._record("get_procedural", locals())
        return {"id": "p"}

    def create_custom(
        self,
        *,
        memory_type: str,
        content: str,
        tags: Mapping[str, Any] | None = None,
        contextual_metadata: dict[str, Any] | None = None,
    ) -> CustomMemorySaveResult:
        self._record("create_custom", locals())
        return CustomMemorySaveResult(
            id="1", type=memory_type, tags={}, has_embedding=True
        )

    def retrieve_custom(
        self,
        *,
        memory_type: str,
        query: str,
        tags: Mapping[str, Any] | None = None,
        top_k: int = 10,
    ) -> CustomMemoryRetrieveResult:
        self._record("retrieve_custom", locals())
        return CustomMemoryRetrieveResult(results=[], count=0)


def test_fake_runtime_conforms_to_runtime_port() -> None:
    runtime: MemoryRuntime = FakeRuntime()
    assert isinstance(runtime, MemoryRuntime)


def test_fake_client_conforms_to_crud_port() -> None:
    client: MemoryCrudClient = FakeClient()
    assert isinstance(client, MemoryCrudClient)


def _bound(
    runtime: FakeRuntime | None = None, client: FakeClient | None = None
) -> Memory:
    return Memory(runtime=runtime or FakeRuntime(), client=client or FakeClient()).bind(
        MemoryRequestContext(user_id="bound-user")
    )


# ---------------------------------------------------------------------------
# Constructor modes
# ---------------------------------------------------------------------------


def test_service_account_token_mode_constructs() -> None:
    assert isinstance(Memory(service_account_token="test-api-key"), Memory)


def test_base_url_constructs_oe_profile() -> None:
    memory = Memory(base_url="http://x")
    assert memory._runtime._profile is _OE_PROFILE
    assert memory._client is not None


def test_no_config_raises_value_error() -> None:
    with pytest.raises(ValueError, match="api_key"):
        Memory()


def test_runtime_injection_constructs() -> None:
    assert isinstance(Memory(runtime=FakeRuntime()), Memory)


# ---------------------------------------------------------------------------
# bind() is non-mutating
# ---------------------------------------------------------------------------


def test_bind_returns_new_handle_without_mutating_original() -> None:
    runtime = FakeRuntime()
    m = Memory(runtime=runtime)
    ctx = MemoryRequestContext(user_id="u")
    b = m.bind(ctx)

    assert b is not m
    assert m._bound_ctx is None
    assert b._bound_ctx is ctx


def test_bind_on_bound_handle_is_non_mutating() -> None:
    m = Memory(runtime=FakeRuntime())
    ctx = MemoryRequestContext(user_id="u")
    other = MemoryRequestContext(user_id="v")
    b = m.bind(ctx)
    b2 = b.bind(other)

    assert b2 is not b
    assert b._bound_ctx is ctx
    assert b2._bound_ctx is other


# ---------------------------------------------------------------------------
# record_turn()
# ---------------------------------------------------------------------------


def test_record_turn_delegates_and_autogenerates_idempotency_key() -> None:
    runtime = FakeRuntime()
    m = Memory(runtime=runtime).bind(MemoryRequestContext(user_id="u"))
    result = m.record_turn(role="user", content="hi", session_id="s")

    # the runtime's result is returned unchanged
    assert isinstance(result, WriteTurnResult)
    assert result.id == "1"

    call = runtime.calls["record_turn"]
    assert call["role"] == "user"
    assert call["content"] == "hi"
    assert call["session_id"] == "s"
    assert call["user_id"] == "u"
    key = call["idempotency_key"]
    assert key is not None
    # parses as a UUID
    uuid.UUID(key)


def test_record_turn_forwards_explicit_idempotency_key() -> None:
    runtime = FakeRuntime()
    m = Memory(runtime=runtime)
    m.record_turn(role="user", content="hi", idempotency_key="fixed")

    assert runtime.calls["record_turn"]["idempotency_key"] == "fixed"


def test_record_turn_forwards_metadata() -> None:
    runtime = FakeRuntime()
    m = Memory(runtime=runtime)
    m.record_turn(role="user", content="hi", metadata={"channel": "slack"})

    assert runtime.calls["record_turn"]["metadata"] == {"channel": "slack"}


def test_record_turn_metadata_defaults_none() -> None:
    runtime = FakeRuntime()
    m = Memory(runtime=runtime)
    m.record_turn(role="user", content="hi")

    assert runtime.calls["record_turn"]["metadata"] is None


def test_record_turn_allows_no_session_id() -> None:
    runtime = FakeRuntime()
    m = Memory(runtime=runtime)
    m.record_turn(role="assistant", content="ok")

    assert runtime.calls["record_turn"]["session_id"] is None


def test_record_turn_identity_flows_from_bind() -> None:
    runtime = FakeRuntime()
    m = Memory(runtime=runtime).bind(MemoryRequestContext(user_id="bound-user"))
    m.record_turn(role="user", content="hi")

    assert runtime.calls["record_turn"]["user_id"] == "bound-user"


# ---------------------------------------------------------------------------
# Search / context ops route to the runtime
# ---------------------------------------------------------------------------


def test_search_semantic_delegates_to_runtime() -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)
    m.search_semantic(query="q")

    call = runtime.calls["search_semantic"]
    assert call["query"] == "q"
    assert call["user_id"] == "bound-user"


def test_search_semantic_call_arg_overrides_bind() -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)
    m.search_semantic(query="q", user_id="explicit")

    assert runtime.calls["search_semantic"]["user_id"] == "explicit"


def test_search_semantic_blank_call_arg_does_not_shadow_bind() -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)
    m.search_semantic(query="q", user_id="")

    assert runtime.calls["search_semantic"]["user_id"] == "bound-user"


class FakeRuntimeWithContext(FakeRuntime):
    """A runtime exposing an ambient identity via ``request_context()``,
    standing in for the app-bound contextvars principal.
    """

    def request_context(self) -> MemoryRequestContext:
        return MemoryRequestContext(user_id="ambient-user")


def test_read_without_visibility_uses_ambient_user_id() -> None:
    runtime = FakeRuntimeWithContext()
    Memory(runtime=runtime).search_semantic(query="q")

    assert runtime.calls["search_semantic"]["user_id"] == "ambient-user"


def test_read_with_visibility_does_not_borrow_ambient_user_id() -> None:
    # An explicit visibility asks for a cross-scope read, so the ambient
    # principal must not silently narrow the result to that user.
    runtime = FakeRuntimeWithContext()
    Memory(runtime=runtime).search_semantic(query="q", visibility="org")

    assert runtime.calls["search_semantic"]["user_id"] is None
    assert runtime.calls["search_semantic"]["visibility"] == "org"


def test_private_read_still_uses_ambient_user_id() -> None:
    # "private" is the user's own scope, not a cross-user one, so the ambient
    # principal must still apply — otherwise a private read silently widens to
    # every user's private memories in the tenant.
    runtime = FakeRuntimeWithContext()
    Memory(runtime=runtime).search_semantic(query="q", visibility="private")

    assert runtime.calls["search_semantic"]["user_id"] == "ambient-user"
    assert runtime.calls["search_semantic"]["visibility"] == "private"


def test_read_with_visibility_still_honors_bound_user_id() -> None:
    # Suppression skips only the ambient fallback; an explicitly bound identity
    # is the caller's deliberate scope and still wins.
    runtime = FakeRuntimeWithContext()
    m = Memory(runtime=runtime).bind(MemoryRequestContext(user_id="bound-user"))
    m.search_semantic(query="q", visibility="org")

    assert runtime.calls["search_semantic"]["user_id"] == "bound-user"


def test_search_episodes_delegates_to_runtime() -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)
    m.search_episodes(query="q", session_id="s")

    call = runtime.calls["search_episodes"]
    assert call["query"] == "q"
    assert call["session_id"] == "s"
    assert call["user_id"] == "bound-user"


def test_search_episodes_does_not_inherit_bound_session() -> None:
    # Episodic memory is stored session-unscoped, so a bound session
    # must not be inherited as a filter. user_id still inherits from bind.
    runtime = FakeRuntime()
    m = Memory(runtime=runtime, client=FakeClient()).bind(
        MemoryRequestContext(user_id="bound-user", session_id="bound-session")
    )
    m.search_episodes(query="q")

    call = runtime.calls["search_episodes"]
    assert call["session_id"] is None
    assert call["user_id"] == "bound-user"


def test_search_episodes_explicit_session_overrides_bind_suppression() -> None:
    runtime = FakeRuntime()
    m = Memory(runtime=runtime, client=FakeClient()).bind(
        MemoryRequestContext(user_id="bound-user", session_id="bound-session")
    )
    m.search_episodes(query="q", session_id="explicit-session")

    assert runtime.calls["search_episodes"]["session_id"] == "explicit-session"


def test_list_episodes_does_not_inherit_bound_session() -> None:
    client = FakeClient()
    m = Memory(runtime=FakeRuntime(), client=client).bind(
        MemoryRequestContext(user_id="bound-user", session_id="bound-session")
    )
    m.list_episodes()

    call = client.calls["list_episodic"]
    assert call["session_id"] is None
    assert call["user_id"] == "bound-user"


def test_search_unified_episodic_leg_does_not_inherit_bound_session() -> None:
    runtime = FakeRuntime()
    m = Memory(runtime=runtime, client=FakeClient()).bind(
        MemoryRequestContext(user_id="bound-user", session_id="bound-session")
    )
    m.search(query="q", sources="episodic")

    call = runtime.calls["search_episodes"]
    assert call["session_id"] is None
    assert call["user_id"] == "bound-user"


def test_record_turn_still_inherits_bound_session() -> None:
    # Regression guard: the read-side session suppression must NOT leak into writes —
    # record_turn still inherits the bound session.
    runtime = FakeRuntime()
    m = Memory(runtime=runtime, client=FakeClient()).bind(
        MemoryRequestContext(user_id="bound-user", session_id="bound-session")
    )
    m.record_turn(role="user", content="hi")

    assert runtime.calls["record_turn"]["session_id"] == "bound-session"


def test_search_taxonomic_delegates_to_runtime() -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)
    m.search_taxonomic(query="q", domain="d")

    call = runtime.calls["search_taxonomic"]
    assert call["domain"] == "d"
    assert call["user_id"] == "bound-user"


def test_discover_procedures_delegates_to_runtime() -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)
    m.discover_procedures(query="q", tags=["x"])

    call = runtime.calls["discover_procedures"]
    assert call["query"] == "q"
    assert call["tags"] == ["x"]
    assert call["user_id"] == "bound-user"


def test_build_context_delegates_to_runtime_with_session() -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)
    m.build_context(query="q", session_id="s")

    call = runtime.calls["build_context"]
    assert call["query"] == "q"
    assert call["session_id"] == "s"
    assert call["user_id"] == "bound-user"


def test_build_context_accepts_deprecated_thread_id_alias() -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)
    m.build_context(query="q", thread_id="legacy")

    assert runtime.calls["build_context"]["session_id"] == "legacy"


def test_build_context_session_id_wins_over_thread_id() -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)
    m.build_context(query="q", session_id="s", thread_id="legacy")

    assert runtime.calls["build_context"]["session_id"] == "s"


def test_build_context_blank_session_id_does_not_suppress_thread_id() -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)
    m.build_context(query="q", session_id="", thread_id="legacy")

    assert runtime.calls["build_context"]["session_id"] == "legacy"


def test_build_context_forwards_explicit_max_tokens() -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)
    m.build_context(query="q", max_tokens=2048)

    call = runtime.calls["build_context"]
    assert call["max_tokens"] == 2048
    assert call["top_k"] == 50


def test_build_context_omits_max_tokens_keyword_when_absent() -> None:
    runtime = FakeRuntime()
    spy = Mock(wraps=runtime.build_context)
    runtime.build_context = spy  # type: ignore[method-assign]
    m = _bound(runtime)
    m.build_context(query="q", session_id="s")

    spy.assert_called_once()
    assert "max_tokens" not in spy.call_args.kwargs
    assert runtime.calls["build_context"]["top_k"] == 50


def test_build_context_forwards_format_style_and_include_memories() -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)
    m.build_context(query="q", format_style="jinja2", include_memories=True)

    call = runtime.calls["build_context"]
    assert call["format_style"] == "jinja2"
    assert call["include_memories"] is True


def test_build_context_omits_format_style_and_include_memories_when_unset() -> None:
    runtime = FakeRuntime()
    spy = Mock(wraps=runtime.build_context)
    runtime.build_context = spy  # type: ignore[method-assign]
    m = _bound(runtime)
    m.build_context(query="q", session_id="s")

    spy.assert_called_once()
    assert "format_style" not in spy.call_args.kwargs
    assert "include_memories" not in spy.call_args.kwargs


def test_build_context_accepts_format_style_enum_member() -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)
    m.build_context(query="q", format_style=FormatStyle.JINJA2)

    # Str-enum members equal their string value, so the recorded call matches.
    assert runtime.calls["build_context"]["format_style"] == "jinja2"


@pytest.mark.parametrize("format_style", ["jinj2", "", "markdown"])
def test_build_context_rejects_invalid_format_style(format_style: str) -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)

    with pytest.raises(ValueError, match="^format_style must be one of:"):
        m.build_context(query="q", format_style=format_style)

    assert "build_context" not in runtime.calls


def test_build_context_accepts_format_style_case_insensitively() -> None:
    # The server coerces case, so the client-side check must not reject it.
    runtime = FakeRuntime()
    m = _bound(runtime)
    m.build_context(query="q", format_style="OpenAI")

    assert runtime.calls["build_context"]["format_style"] == "OpenAI"


@pytest.mark.parametrize(
    "max_tokens",
    [0, -1, True, False, 1.5, float("nan"), float("inf"), float("-inf")],
)
def test_build_context_rejects_invalid_max_tokens(max_tokens: object) -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)

    with pytest.raises(ValueError, match="^max_tokens must be a positive integer$"):
        m.build_context(query="q", max_tokens=max_tokens)  # type: ignore[arg-type]

    assert "build_context" not in runtime.calls


def test_build_context_from_sources_delegates_with_resolved_identity() -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)
    specs = [
        SourceSpec(source="semantic", mode="hybrid", top_k=20),
        SourceSpec(source="episodic", metadata_filter={"topic": "x"}),
    ]
    m.build_context_from_sources(query="q", sources=specs, session_id="s", rerank=True)

    call = runtime.calls["build_context_from_sources"]
    assert call["query"] == "q"
    assert call["session_id"] == "s"
    assert call["user_id"] == "bound-user"
    assert call["rerank"] is True
    assert call["sources"] == specs


def test_build_context_from_sources_accepts_thread_id_alias() -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)
    m.build_context_from_sources(
        query="q", sources=[SourceSpec(source="semantic")], thread_id="legacy"
    )

    assert runtime.calls["build_context_from_sources"]["session_id"] == "legacy"


def test_build_context_from_sources_rejects_empty_sources() -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)

    with pytest.raises(ValueError, match="at least one SourceSpec"):
        m.build_context_from_sources(query="q", sources=[])

    assert "build_context_from_sources" not in runtime.calls


def test_build_context_from_sources_rejects_duplicate_source() -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)
    specs = [
        SourceSpec(source="semantic", mode="text"),
        SourceSpec(source="semantic", mode="hybrid"),
    ]

    with pytest.raises(ValueError, match="at most once"):
        m.build_context_from_sources(query="q", sources=specs)

    assert "build_context_from_sources" not in runtime.calls


def test_build_context_from_sources_rejects_invalid_max_tokens() -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)

    with pytest.raises(ValueError, match="^max_tokens must be a positive integer$"):
        m.build_context_from_sources(
            query="q", sources=[SourceSpec(source="semantic")], max_tokens=0
        )

    assert "build_context_from_sources" not in runtime.calls


def test_build_context_from_sources_rejects_stm_without_session() -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)  # binds user_id only, no session

    with pytest.raises(MemoryIdentityError, match="stm"):
        m.build_context_from_sources(
            query="q",
            sources=[SourceSpec(source="stm"), SourceSpec(source="semantic")],
        )

    assert "build_context_from_sources" not in runtime.calls


@pytest.mark.parametrize("session_id", ["", "   "])
def test_build_context_from_sources_rejects_stm_with_blank_session(
    session_id: str,
) -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)

    with pytest.raises(MemoryIdentityError, match="stm"):
        m.build_context_from_sources(
            query="q",
            sources=[SourceSpec(source="stm")],
            session_id=session_id,
        )

    assert "build_context_from_sources" not in runtime.calls


def test_build_context_from_sources_allows_stm_with_session() -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)
    m.build_context_from_sources(
        query="q", sources=[SourceSpec(source="stm")], session_id="s"
    )

    assert runtime.calls["build_context_from_sources"]["session_id"] == "s"


def test_build_context_from_sources_forwards_format_style_and_include_memories() -> (
    None
):
    runtime = FakeRuntime()
    m = _bound(runtime)
    m.build_context_from_sources(
        query="q",
        sources=[SourceSpec(source="semantic")],
        format_style="openai",
        include_memories=True,
    )

    call = runtime.calls["build_context_from_sources"]
    assert call["format_style"] == "openai"
    assert call["include_memories"] is True


def test_build_context_from_sources_omits_format_kwargs_when_unset() -> None:
    runtime = FakeRuntime()
    spy = Mock(wraps=runtime.build_context_from_sources)
    runtime.build_context_from_sources = spy  # type: ignore[method-assign]
    m = _bound(runtime)
    m.build_context_from_sources(query="q", sources=[SourceSpec(source="semantic")])

    spy.assert_called_once()
    assert "format_style" not in spy.call_args.kwargs
    assert "include_memories" not in spy.call_args.kwargs


@pytest.mark.parametrize("format_style", ["jinj2", "", "markdown"])
def test_build_context_from_sources_rejects_invalid_format_style(
    format_style: str,
) -> None:
    runtime = FakeRuntime()
    m = _bound(runtime)

    with pytest.raises(ValueError, match="^format_style must be one of:"):
        m.build_context_from_sources(
            query="q",
            sources=[SourceSpec(source="semantic")],
            format_style=format_style,
        )

    assert "build_context_from_sources" not in runtime.calls


def test_source_spec_coerces_string_enums() -> None:
    spec = SourceSpec(source="semantic", mode="hybrid")
    # use_enum_values stores the raw string values, ready for the wire.
    assert spec.source == MemorySource.SEMANTIC.value
    assert spec.mode == RetrievalMode.HYBRID.value


def test_source_spec_rejects_top_k_outside_server_bounds() -> None:
    from pydantic import ValidationError

    # top_k mirrors the server's per-source bounds (1..200), so an out-of-range
    # value fails locally as a ValidationError rather than as a wire 422.
    with pytest.raises(ValidationError):
        SourceSpec(source="semantic", top_k=201)
    with pytest.raises(ValidationError):
        SourceSpec(source="semantic", top_k=0)
    # The boundaries themselves are accepted.
    assert SourceSpec(source="semantic", top_k=200).top_k == 200
    assert SourceSpec(source="semantic", top_k=1).top_k == 1


# ---------------------------------------------------------------------------
# CRUD ops route to the client
# ---------------------------------------------------------------------------


def test_save_semantic_delegates_to_client_with_resolved_identity() -> None:
    client = FakeClient()
    m = _bound(client=client)
    m.save_semantic(text="t", label="lbl")

    call = client.calls["create_semantic"]
    assert call["text"] == "t"
    assert call["label"] == "lbl"
    assert call["user_id"] == "bound-user"
    assert call["metadata"] is None
    assert "org_id" not in call
    assert "project_id" not in call


def test_save_semantic_forwards_metadata_to_client() -> None:
    client = FakeClient()
    m = _bound(client=client)
    m.save_semantic(text="t", label="lbl", metadata={"channel": "web"})

    assert client.calls["create_semantic"]["metadata"] == {"channel": "web"}


def test_get_semantic_delegates_to_client() -> None:
    client = FakeClient()
    m = _bound(client=client)
    m.get_semantic(label="lbl")

    assert client.calls["get_semantic"]["label"] == "lbl"


def test_save_episode_delegates_to_client() -> None:
    client = FakeClient()
    m = _bound(client=client)
    m.save_episode(title="t", content="c", session_id="s")

    call = client.calls["create_episodic"]
    assert call["title"] == "t"
    assert call["user_id"] == "bound-user"
    assert call["session_id"] == "s"
    assert call["metadata"] is None


def test_save_episode_forwards_metadata_to_client() -> None:
    client = FakeClient()
    m = _bound(client=client)
    m.save_episode(title="t", content="c", session_id="s", metadata={"channel": "web"})

    assert client.calls["create_episodic"]["metadata"] == {"channel": "web"}


def test_list_episodes_delegates_to_client() -> None:
    client = FakeClient()
    m = _bound(client=client)
    m.list_episodes()

    assert client.calls["list_episodic"]["user_id"] == "bound-user"


def test_save_taxonomic_delegates_to_client() -> None:
    client = FakeClient()
    m = _bound(client=client)
    m.save_taxonomic(domain="d", term="t", definition="def")

    call = client.calls["create_taxonomic"]
    assert call["domain"] == "d"
    assert call["term"] == "t"
    assert call["metadata"] is None


def test_save_taxonomic_forwards_metadata() -> None:
    client = FakeClient()
    m = _bound(client=client)
    m.save_taxonomic(domain="d", term="t", definition="def", metadata={"k": "v"})

    assert client.calls["create_taxonomic"]["metadata"] == {"k": "v"}


def test_get_taxonomic_term_delegates_to_client() -> None:
    client = FakeClient()
    m = _bound(client=client)
    m.get_taxonomic_term(domain="d", term="t")

    call = client.calls["get_taxonomic"]
    assert call["domain"] == "d"
    assert call["term"] == "t"


def test_list_domains_delegates_to_client() -> None:
    client = FakeClient()
    m = _bound(client=client)
    m.list_domains()

    assert "get_distinct_domains" in client.calls


def test_save_procedure_delegates_to_client() -> None:
    client = FakeClient()
    m = _bound(client=client)
    m.save_procedure(procedure="p", description="d", content="c")

    call = client.calls["create_procedural"]
    assert call["procedure"] == "p"
    assert call["user_id"] == "bound-user"


def test_save_procedure_forwards_update_existing() -> None:
    client = FakeClient()
    m = _bound(client=client)
    m.save_procedure(procedure="p", description="d", content="c", update_existing=True)

    assert client.calls["create_procedural"]["update_existing"] is True


def test_save_procedure_forwards_metadata() -> None:
    client = FakeClient()
    m = _bound(client=client)
    m.save_procedure(procedure="p", description="d", content="c", metadata={"k": "v"})

    assert client.calls["create_procedural"]["metadata"] == {"k": "v"}


def test_get_procedure_delegates_to_client() -> None:
    client = FakeClient()
    m = _bound(client=client)
    m.get_procedure("proc-name")

    assert client.calls["get_procedural"]["procedure"] == "proc-name"


# ---------------------------------------------------------------------------
# Required identity is enforced before any delegation
# ---------------------------------------------------------------------------


def test_save_semantic_without_resolvable_user_id_raises() -> None:
    m = Memory(runtime=FakeRuntime(), client=FakeClient())
    with pytest.raises(MemoryIdentityError, match="user_id"):
        m.save_semantic(text="t", label="lbl")


def test_save_episode_without_resolvable_session_id_raises() -> None:
    m = _bound()
    with pytest.raises(MemoryIdentityError, match="session_id"):
        m.save_episode(title="t", content="c")


def test_save_taxonomic_without_resolvable_user_id_raises() -> None:
    m = Memory(runtime=FakeRuntime(), client=FakeClient())
    with pytest.raises(MemoryIdentityError, match="user_id"):
        m.save_taxonomic(domain="d", term="t", definition="def")


def test_save_procedure_without_resolvable_user_id_raises() -> None:
    m = Memory(runtime=FakeRuntime(), client=FakeClient())
    with pytest.raises(MemoryIdentityError, match="user_id"):
        m.save_procedure(procedure="p", description="d", content="c")


# ---------------------------------------------------------------------------
# CRUD without a configured client fails with a clear error
# ---------------------------------------------------------------------------


def test_crud_without_client_raises_clear_error() -> None:
    from agent_engine_sdk_memory.errors import MemoryNotSupportedError

    m = Memory(runtime=FakeRuntime()).bind(MemoryRequestContext(user_id="u"))
    with pytest.raises(MemoryNotSupportedError, match="CRUD client"):
        m.save_semantic(text="t", label="lbl")


def test_runtime_ops_work_without_client() -> None:
    runtime = FakeRuntime()
    m = Memory(runtime=runtime).bind(MemoryRequestContext(user_id="u"))
    m.search_semantic(query="q")

    assert runtime.calls["search_semantic"]["user_id"] == "u"


# ---------------------------------------------------------------------------
# Runtime-context identity resolution (duck-typed request_context())
# ---------------------------------------------------------------------------


class FakeAmbientRuntime(FakeRuntime):
    """A runtime exposing the optional ``request_context()`` accessor.

    Each call returns a fresh context and bumps ``request_context_calls`` so a
    test can assert per-call freshness rather than construction-time capture.
    """

    def __init__(self, ctx: MemoryRequestContext | None) -> None:
        super().__init__()
        self._ctx = ctx
        self.request_context_calls = 0

    def request_context(self) -> MemoryRequestContext | None:
        self.request_context_calls += 1
        return self._ctx


def test_save_semantic_resolves_user_id_from_runtime_context() -> None:
    runtime = FakeAmbientRuntime(MemoryRequestContext(user_id="u-amb"))
    client = FakeClient()
    m = Memory(runtime=runtime, client=client)
    m.save_semantic(text="t", label="lbl")

    assert client.calls["create_semantic"]["user_id"] == "u-amb"


def test_save_episode_resolves_user_and_session_from_runtime_context() -> None:
    runtime = FakeAmbientRuntime(
        MemoryRequestContext(user_id="u-amb", session_id="s-amb")
    )
    client = FakeClient()
    m = Memory(runtime=runtime, client=client)
    m.save_episode(title="t", content="c")

    call = client.calls["create_episodic"]
    assert call["user_id"] == "u-amb"
    assert call["session_id"] == "s-amb"


def test_call_arg_wins_over_runtime_context() -> None:
    runtime = FakeAmbientRuntime(MemoryRequestContext(user_id="u-amb"))
    client = FakeClient()
    m = Memory(runtime=runtime, client=client)
    m.save_semantic(text="t", label="lbl", user_id="call")

    assert client.calls["create_semantic"]["user_id"] == "call"


def test_bind_wins_over_runtime_context_but_loses_to_call_arg() -> None:
    runtime = FakeAmbientRuntime(MemoryRequestContext(user_id="u-amb"))
    client = FakeClient()
    m = Memory(runtime=runtime, client=client).bind(
        MemoryRequestContext(user_id="bound")
    )

    m.save_semantic(text="t", label="lbl")
    assert client.calls["create_semantic"]["user_id"] == "bound"

    m.save_semantic(text="t", label="lbl", user_id="call")
    assert client.calls["create_semantic"]["user_id"] == "call"


def test_runtime_without_request_context_resolves_to_none() -> None:
    # A plain FakeRuntime never defines request_context(); identity must come
    # only from call args / bind, so an unresolved required field still raises.
    m = Memory(runtime=FakeRuntime(), client=FakeClient())
    with pytest.raises(MemoryIdentityError, match="user_id"):
        m.save_semantic(text="t", label="lbl")


def test_request_context_read_fresh_on_every_resolve() -> None:
    runtime = FakeAmbientRuntime(MemoryRequestContext(user_id="u-amb"))
    m = Memory(runtime=runtime, client=FakeClient())

    m.save_semantic(text="t", label="lbl")
    m.save_semantic(text="t2", label="lbl2")

    assert runtime.request_context_calls == 2


def test_request_context_returning_none_is_treated_as_no_context() -> None:
    runtime = FakeAmbientRuntime(None)
    m = Memory(runtime=runtime, client=FakeClient())
    with pytest.raises(MemoryIdentityError, match="user_id"):
        m.save_semantic(text="t", label="lbl")
