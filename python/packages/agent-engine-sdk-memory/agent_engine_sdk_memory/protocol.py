"""Seam primitives: the immutable request context and the runtime Protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent_engine_sdk_memory.models import (
    ContextResponse,
    CreateEpisodicResult,
    CreateProceduralResult,
    CreateSemanticResult,
    CreateTaxonomicResult,
    CustomMemoryRetrieveResult,
    CustomMemorySaveResult,
    FormatStyle,
    MemoryChunk,
    SourceSpec,
    WriteTurnResult,
)


class MemoryRequestContext(BaseModel):
    """Immutable identity context carried across a memory call."""

    model_config = ConfigDict(frozen=True)

    user_id: str | None = Field(
        default=None, description="End user the memory operation is scoped to"
    )
    agent_id: str | None = Field(
        default=None, description="Agent performing the memory operation"
    )
    session_id: str | None = Field(
        default=None, description="Conversation session the operation belongs to"
    )


@runtime_checkable
class MemoryRuntime(Protocol):
    """Transport seam every memory backend implements.

    Identity is passed as resolved keyword arguments; tenancy is internal to
    each runtime.
    """

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
    ) -> WriteTurnResult: ...

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
    ) -> ContextResponse: ...

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
    ) -> ContextResponse: ...

    def search_semantic(
        self,
        *,
        query: str,
        user_id: str | None = None,
        visibility: str | None = None,
        top_k: int = 50,
    ) -> list[MemoryChunk]: ...

    def search_episodes(
        self,
        *,
        query: str,
        user_id: str | None = None,
        visibility: str | None = None,
        session_id: str | None = None,
        top_k: int = 50,
    ) -> list[MemoryChunk]: ...

    def search_taxonomic(
        self,
        *,
        query: str,
        user_id: str | None = None,
        domain: str | None = None,
        visibility: str | None = None,
        top_k: int = 50,
    ) -> list[MemoryChunk]: ...

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
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class AmbientIdentityRuntime(Protocol):
    """Optional capability: a runtime that can supply per-call ambient identity.

    Not all runtimes carry ambient identity — only app-bound runtimes running
    inside the platform stack have access to per-request contextvars. This is a
    separate Protocol (not merged into ``MemoryRuntime``) so the facade can
    detect the capability without requiring every runtime to implement it.
    """

    def request_context(self) -> MemoryRequestContext | None: ...


@runtime_checkable
class MemoryCrudClient(Protocol):
    """CRUD seam the facade's type-specific conveniences delegate to.

    Like ``MemoryRuntime``, tenancy is internal to each implementation: the
    method shapes mirror ``MemoryClient`` with ``org_id``/``project_id``
    stripped, so the raw HTTP client never satisfies this Protocol — each
    transport wires in an adapter that supplies tenancy itself.
    """

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
    ) -> CreateSemanticResult: ...

    def get_semantic(
        self,
        *,
        label: str,
        user_id: str | None = None,
        visibility: str | None = None,
    ) -> Any | None: ...

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
    ) -> CreateEpisodicResult: ...

    def list_episodic(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        visibility: str | None = None,
        limit: int = 20,
    ) -> list[Any]: ...

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
    ) -> CreateTaxonomicResult: ...

    def get_taxonomic(
        self,
        *,
        domain: str,
        term: str | None = None,
        user_id: str | None = None,
        visibility: str | None = None,
    ) -> Any | None: ...

    def get_distinct_domains(
        self,
        *,
        visibility: str | None = None,
    ) -> list[str]: ...

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
    ) -> CreateProceduralResult: ...

    def get_procedural(
        self,
        *,
        procedure: str | None = None,
        user_id: str | None = None,
        visibility: str | None = None,
        include_deleted: bool = False,
    ) -> Any | None: ...

    def create_custom(
        self,
        *,
        memory_type: str,
        content: str,
        tags: Mapping[str, Any] | None = None,
        contextual_metadata: dict[str, Any] | None = None,
    ) -> CustomMemorySaveResult:
        """Save a custom-type memory. Identity is stamped by the platform."""
        ...

    def retrieve_custom(
        self,
        *,
        memory_type: str,
        query: str,
        tags: Mapping[str, Any] | None = None,
        top_k: int = 10,
    ) -> CustomMemoryRetrieveResult:
        """Retrieve custom-type memories. Identity is stamped by the platform."""
        ...


__all__ = [
    "AmbientIdentityRuntime",
    "MemoryCrudClient",
    "MemoryRequestContext",
    "MemoryRuntime",
]
