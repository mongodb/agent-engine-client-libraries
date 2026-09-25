"""App-bound memory adapters over a ``agent_engine_runner_shared.TenantRuntime``.

These adapters let the unified ``agent_engine_sdk_memory.Memory`` facade run in
app-bound mode (inside an agent, on the platform stack) by wrapping a single
``TenantRuntime`` and reimplementing no memory logic. ``AppBoundRuntime``
satisfies the ``MemoryRuntime`` Protocol plus the duck-typed ``request_context()``
accessor the facade reads to resolve ambient identity. ``AppBoundCrudClient``
satisfies the ``MemoryCrudClient`` Protocol.

The work here is return-shape mapping: ``TenantRuntime`` returns loosely-typed
``str`` / ``bool`` / ``dict | None`` / ``list[dict]``, while the Protocol returns
the facade's typed result models. Each delegating method maps its output into
the typed model.

Identity scope: like the standalone Memory SDK, explicit ``user_id`` /
``session_id`` arguments win over the ambient request identity, so app-bound
agent code can address another identity within the same org/project. This is
intentional — the facade's documented semantics, not a leak — and is the
accepted platform posture (decided on PR #4946 review). Callers that must not
cross identities should omit the explicit arguments and let the request-scoped
contextvars supply them.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

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
    FormatStyle,
    MemoryChunk,
    MemorySource,
    SourceSpec,
    WriteTurnResult,
)
from agent_engine_sdk_memory.protocol import MemoryRequestContext

from agent_engine_runner_shared.context import get_current_session_id, get_current_user_id

if TYPE_CHECKING:
    from agent_engine_runner_shared.runtime import TenantRuntime

# Sentinel for missing timestamps: never use query-time now(), which would make
# every chunk look fresh and distort recency. Matches agent_engine_sdk_memory's
# _procedure_to_chunk fallback.
_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _nonempty_str(raw: Any) -> str | None:
    """Return a non-blank string, else None.

    TenantRuntime episodic dicts always emit a ``content`` key (often ``""``
    when only ``summary`` is useful). Treat blank strings as missing so later
    fallbacks (definition compose, summary) can still run.
    """
    if raw is None:
        return None
    text = str(raw)
    return text if text.strip() else None


def _content_from_result(d: dict[str, Any]) -> str:
    """Resolve display content from a loosely-typed TenantRuntime result dict.

    TenantRuntime search shapes differ by source: semantic uses ``content``/``text``,
    taxonomic uses ``term``/``definition``/``domain`` (no content field), episodic
    may only have ``summary``. Taxonomic compose matches memory-server
    ``MemoryChunk.from_taxonomic`` (``domain/term: definition``).
    """
    for key in ("content", "text"):
        text = _nonempty_str(d.get(key))
        if text is not None:
            return text

    definition = _nonempty_str(d.get("definition"))
    if definition is not None:
        term = d.get("term")
        domain = d.get("domain")
        if term is not None and domain is not None:
            return f"{domain}/{term}: {definition}"
        if term is not None:
            return f"{term}: {definition}"
        return str(definition)

    for key in ("summary", "summary_text"):
        text = _nonempty_str(d.get(key))
        if text is not None:
            return text

    return ""


def _timestamp_from_result(d: dict[str, Any]) -> datetime:
    """Resolve a timestamp, falling back to epoch when absent or unparseable."""
    for key in ("timestamp", "created_at", "updated_at"):
        raw = d.get(key)
        if raw is None:
            continue
        if isinstance(raw, datetime):
            return raw if raw.tzinfo is not None else raw.replace(tzinfo=timezone.utc)
        if isinstance(raw, str) and raw:
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return _EPOCH_UTC


def _chunk(d: dict[str, Any], source: MemorySource) -> MemoryChunk:
    """Map a loosely-typed TenantRuntime result dict into a ``MemoryChunk``.

    Defensive over the backend's untyped dicts: ``id``/``content``/``timestamp``
    are pulled with fallbacks (taxonomic compose, summary, epoch for missing
    times), and the COMPLETE source dict is preserved under ``metadata`` so call
    sites can read loose fields (``title``, ``summary``, ``tags``, ``term``,
    ``domain``, ``definition``, ``related_terms``) via ``chunk.metadata.get(...)``.
    """
    raw_id = d.get("id")
    if raw_id is None:
        raw_id = d.get("_id")
    chunk_id = "" if raw_id is None else str(raw_id)

    content = _content_from_result(d)
    timestamp = _timestamp_from_result(d)

    similarity_raw = d.get("similarity_score")
    similarity = float(similarity_raw) if isinstance(similarity_raw, (int, float)) else None

    embedding_raw = d.get("embedding")
    embedding = embedding_raw if isinstance(embedding_raw, list) else None

    return MemoryChunk(
        id=chunk_id,
        content=content,
        source=source,
        timestamp=timestamp,
        similarity_score=similarity,
        embedding=embedding,
        metadata=dict(d),
    )


class AppBoundRuntime:
    """``MemoryRuntime`` over a ``TenantRuntime``, mapping its outputs into the
    facade's typed models and exposing per-call ambient identity via
    ``request_context()``.
    """

    def __init__(self, runtime: TenantRuntime) -> None:
        self._runtime = runtime

    def request_context(self) -> MemoryRequestContext | None:
        """Return a fresh identity context from the agent_engine_runner_shared contextvars.

        Read on every call because ``app.memory`` is a long-lived singleton over
        per-request contextvars; blank/None fields are normalized to "unset" by
        the facade's identity resolver.
        """
        return MemoryRequestContext(
            user_id=get_current_user_id(),
            session_id=get_current_session_id(),
        )

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
        # write_turn_async carries (message, result_messages, include_user_turn,
        # metadata). It cannot represent per-turn tool-call/model
        # metadata — reject those loudly instead of silently dropping fields.
        # Plain user/assistant roles map onto the coarser shape; other roles
        # (tool, system, …) cannot be represented faithfully and are rejected.
        if (
            tool_calls is not None
            or tool_call_id is not None
            or tool_name is not None
            or model_name is not None
            or is_error
        ):
            raise MemoryNotSupportedError(
                "tool-call and model turn metadata are not supported for "
                "app-bound record_turn; record the turn without tool metadata"
            )

        role_key = (role or "").strip().lower()
        text = content or ""
        if role_key not in ("user", "assistant"):
            raise MemoryNotSupportedError(
                "app-bound record_turn supports only role='user' and "
                f"role='assistant' without tool/model metadata; got role={role!r}"
            )

        # Mirror TenantRuntime.write_turn_async / write_turn_to_memory skip
        # conditions so acknowledged is honest for fire-and-forget no-ops.
        # idempotency_key is forwarded and stamped on the single emitted turn,
        # so a caller retrying with a stable key dedupes server-side the way
        # the Memory facade documents. agent_id is accepted for interface
        # parity but not forwarded — TenantRuntime has no parameter for it.
        resolved_user_id = user_id or get_current_user_id()
        resolved_session_id = session_id or get_current_session_id()
        will_queue = (
            self._runtime.memory_writer is not None
            and bool(resolved_user_id)
            and bool(resolved_session_id)
            and bool(text.strip())
        )

        if will_queue:
            if role_key == "user":
                self._runtime.write_turn_async(
                    message=text,
                    result_messages=[],
                    user_id=resolved_user_id,
                    session_id=resolved_session_id,
                    include_user_turn=True,
                    metadata=metadata,
                    idempotency_key=idempotency_key,
                )
            else:
                # extract_turn_messages expects Message-like attrs: role, content,
                # tool_calls. include_user_turn=False so we do not invent a user turn.
                assistant_msg = SimpleNamespace(
                    role="assistant",
                    content=text,
                    tool_calls=None,
                )
                self._runtime.write_turn_async(
                    message="",
                    result_messages=[assistant_msg],
                    user_id=resolved_user_id,
                    session_id=resolved_session_id,
                    include_user_turn=False,
                    metadata=metadata,
                    idempotency_key=idempotency_key,
                )

        return WriteTurnResult(
            id="",
            session_id=resolved_session_id or "",
            turn_seq=0,
            acknowledged=will_queue,
        )

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
        # top_k remains runtime-owned in app-bound mode: accepted for interface
        # parity but NOT forwarded. max_tokens is supported and forwarded when
        # explicit; OE/Memory Server already expose that context-budget control.
        # format_style/include_memories cannot be honored here: TenantRuntime
        # flattens the response to a jinja2 string, so there is no channel for
        # an alternate format or the selected chunks. Reject loudly instead of
        # silently dropping them (same rule as record_turn tool metadata).
        if format_style is not None or include_memories:
            raise MemoryNotSupportedError(
                "format_style and include_memories are not supported for "
                "app-bound build_context; use an HTTP mode (api_key/project_id) "
                "for response shaping"
            )
        kwargs: dict[str, Any] = dict(
            query=query,
            user_id=user_id,
            session_id=session_id,
            visibility=visibility,
            metadata_filter=metadata_filter,
            enabled_sources=enabled_sources,
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        formatted = self._runtime.build_context(**kwargs)
        return ContextResponse(
            formatted_context=formatted,
            metadata=ContextMetadata(),
        )

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
        # format_style/include_memories are not threaded through TenantRuntime's
        # per-source path yet; reject loudly instead of silently dropping them.
        if format_style is not None or include_memories:
            raise MemoryNotSupportedError(
                "format_style and include_memories are not supported for "
                "app-bound build_context_from_sources; use an HTTP mode "
                "(api_key/project_id) for response shaping"
            )
        # Unlike build_context, the per-source path returns a full ContextResponse
        # so the per-source metadata (ranking_strategy, source_outcomes) reaches
        # the caller; pass it straight through rather than re-wrapping a string.
        return self._runtime.build_context_from_sources(
            query=query,
            sources=sources,
            user_id=user_id,
            session_id=session_id,
            visibility=visibility,
            rerank=rerank,
            max_tokens=max_tokens,
        )

    def search_semantic(
        self,
        *,
        query: str,
        user_id: str | None = None,
        visibility: str | None = None,
        top_k: int = 50,
    ) -> list[MemoryChunk]:
        results = self._runtime.search_semantic(
            query=query,
            user_id=user_id,
            visibility=visibility,
            top_k=top_k,
        )
        return [_chunk(d, MemorySource.SEMANTIC) for d in results]

    def search_episodes(
        self,
        *,
        query: str,
        user_id: str | None = None,
        visibility: str | None = None,
        session_id: str | None = None,
        top_k: int = 50,
    ) -> list[MemoryChunk]:
        results = self._runtime.search_episodes(
            query=query,
            user_id=user_id,
            visibility=visibility,
            session_id=session_id,
            top_k=top_k,
        )
        return [_chunk(d, MemorySource.EPISODIC) for d in results]

    def search_taxonomic(
        self,
        *,
        query: str,
        user_id: str | None = None,
        domain: str | None = None,
        visibility: str | None = None,
        top_k: int = 50,
    ) -> list[MemoryChunk]:
        results = self._runtime.search_taxonomic(
            query=query,
            user_id=user_id,
            domain=domain,
            visibility=visibility,
            top_k=top_k,
        )
        return [_chunk(d, MemorySource.TAXONOMIC) for d in results]

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
        return self._runtime.discover_procedures(
            query=query,
            user_id=user_id,
            visibility=visibility,
            tags=tags,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            metadata_filter=metadata_filter,
        )


class AppBoundCrudClient:
    """``MemoryCrudClient`` over a ``TenantRuntime``, mapping its CRUD returns
    into the typed ``Create*Result`` models.

    ``has_embedding`` is reported ``False`` for ops where TenantRuntime surfaces
    no embedding status (it returns only ``bool`` / doc-id), and ``id`` defaults
    to the empty string where TenantRuntime gives no document id — these are the
    honest mappings of the platform backend's coarser return shapes.
    """

    def __init__(self, runtime: TenantRuntime) -> None:
        self._runtime = runtime

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
        saved = self._runtime.save_semantic(
            text=text,
            label=label,
            user_id=user_id,
            source=source,
            visibility=visibility,
            metadata=metadata,
            upsert=upsert,
            agent_id=agent_id,
        )
        return CreateSemanticResult(
            id="",
            label=label,
            has_embedding=False,
            acknowledged=saved,
        )

    def get_semantic(
        self,
        *,
        label: str,
        user_id: str | None = None,
        visibility: str | None = None,
    ) -> Any | None:
        return self._runtime.get_semantic(
            label=label,
            user_id=user_id,
            visibility=visibility,
        )

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
        doc_id = self._runtime.save_episode(
            title=title,
            content=content,
            user_id=user_id,
            summary=summary_text,
            session_id=session_id,
            participants=participants,
            tags=tags,
            visibility=visibility,
            agent_id=agent_id,
            metadata=metadata,
        )
        return CreateEpisodicResult(
            id=doc_id or "",
            title=title,
            has_embedding=False,
            acknowledged=doc_id is not None,
        )

    def list_episodic(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        visibility: str | None = None,
        limit: int = 20,
    ) -> list[Any]:
        return self._runtime.list_episodes(
            user_id=user_id,
            session_id=session_id,
            visibility=visibility,
            limit=limit,
        )

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
        doc_id = self._runtime.create_taxonomic(
            domain=domain,
            term=term,
            definition=definition,
            related_terms=related_terms,
            visibility=visibility,
            user_id=user_id,
            metadata=metadata,
        )
        return CreateTaxonomicResult(
            id=doc_id or "",
            domain=domain,
            term=term,
            has_embedding=False,
            acknowledged=doc_id is not None,
        )

    def get_taxonomic(
        self,
        *,
        domain: str,
        term: str | None = None,
        user_id: str | None = None,
        visibility: str | None = None,
    ) -> Any | None:
        # Direct mode can list a whole domain; TenantRuntime requires a concrete
        # term, so a term-less lookup is a capability gap of this backend.
        if term is None:
            raise MemoryNotSupportedError(
                "the app-bound backend has no list-by-domain taxonomic getter; pass a concrete term"
            )
        return self._runtime.get_taxonomic_term(
            domain=domain,
            term=term,
            user_id=user_id,
            visibility=visibility,
        )

    def get_distinct_domains(
        self,
        *,
        visibility: str | None = None,
    ) -> list[str]:
        return self._runtime.list_taxonomic_domains(visibility=visibility)

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
        update_existing: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> CreateProceduralResult:
        saved = self._runtime.save_procedure(
            procedure=procedure,
            description=description,
            content=content,
            user_id=user_id,
            steps=steps,
            resources=resources,
            allowed_tools=allowed_tools,
            compatibility=compatibility,
            license=license,
            trigger_conditions=trigger_conditions,
            tags=tags,
            visibility=visibility,
            agent_id=agent_id,
            extraction_source=extraction_source,
            source_format=source_format,
            source_path=source_path,
            update_existing=update_existing,
            metadata=metadata,
        )
        if saved is None:
            proc_id = ""
            proc_name = procedure
            has_embedding = False
            # TenantRuntime returns None on hard failure (no engine / exception).
            acknowledged = False
        else:
            proc_id = str(saved.get("id") or "")
            proc_name = str(saved.get("procedure") or procedure)
            has_embedding = bool(saved.get("has_embedding", False))
            # Prefer the backend flag when present (create_fallback includes it);
            # otherwise a non-None dict means the write path returned a document.
            acknowledged = bool(saved.get("acknowledged", True))
        return CreateProceduralResult(
            id=proc_id,
            procedure=proc_name,
            has_embedding=has_embedding,
            acknowledged=acknowledged,
        )

    def get_procedural(
        self,
        *,
        procedure: str | None = None,
        user_id: str | None = None,
        visibility: str | None = None,
        include_deleted: bool = False,
    ) -> Any | None:
        # Direct mode can list all procedures; TenantRuntime requires a concrete
        # name, so a name-less lookup is a capability gap of this backend.
        if procedure is None:
            raise MemoryNotSupportedError(
                "the app-bound backend has no list-all procedural getter; "
                "pass a concrete procedure name"
            )
        return self._runtime.get_procedure(
            procedure_name=procedure,
            user_id=user_id,
            visibility=visibility,
            include_deleted=include_deleted,
        )

    def create_custom(
        self,
        *,
        memory_type: str,
        content: str,
        tags: Mapping[str, Any] | None = None,
        contextual_metadata: dict[str, Any] | None = None,
    ) -> CustomMemorySaveResult:
        return self._runtime.save_custom(
            memory_type=memory_type,
            content=content,
            tags=dict(tags) if tags is not None else None,
            contextual_metadata=contextual_metadata,
        )

    def retrieve_custom(
        self,
        *,
        memory_type: str,
        query: str,
        tags: Mapping[str, Any] | None = None,
        top_k: int = 10,
    ) -> CustomMemoryRetrieveResult:
        return self._runtime.retrieve_custom(
            memory_type=memory_type,
            query=query,
            tags=dict(tags) if tags is not None else None,
            top_k=top_k,
        )


__all__ = ["AppBoundCrudClient", "AppBoundRuntime"]
