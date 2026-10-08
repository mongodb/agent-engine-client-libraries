"""
Canonical Pydantic models for Memory API.

These models define the request/response types for the Memory Server HTTP API
and are used by both MemoryClient and the server routes.

Single source of truth for Memory API types.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import (  # Pydantic requires this on Python < 3.12
    NotRequired,
    TypedDict,
)

TagScalar = str | int | float | bool

# =============================================================================
# Result Models (Operation Results)
# =============================================================================


class WriteTurnResult(BaseModel):
    """Result of writing a conversation turn."""

    id: str = Field(..., description="Created document ID")
    session_id: str = Field(..., description="Session ID")
    turn_seq: int = Field(..., description="Turn sequence number")
    acknowledged: bool = Field(
        default=True, description="Whether operation was acknowledged"
    )
    has_embedding: bool = Field(
        default=False, description="Whether the stored turn has an embedding"
    )


class CreateSemanticResult(BaseModel):
    """Result of creating a semantic memory."""

    id: str = Field(..., description="Created document ID")
    label: str = Field(..., description="Label of the memory")
    has_embedding: bool = Field(..., description="Whether embedding was generated")
    acknowledged: bool = Field(
        default=True, description="Whether operation was acknowledged"
    )


class BulkCreateSemanticResult(BaseModel):
    """Result of bulk creating semantic memories."""

    created_count: int = Field(..., description="Number of memories created")
    skipped_count: int = Field(..., description="Number of duplicates skipped")
    created_ids: list[str] = Field(..., description="List of created document IDs")
    skipped_labels: list[str] = Field(..., description="List of skipped labels")
    has_embeddings: bool = Field(..., description="Whether embeddings were generated")
    acknowledged: bool = Field(
        default=True, description="Whether operation was acknowledged"
    )


class CreateEpisodicResult(BaseModel):
    """Result of creating an episodic memory."""

    id: str = Field(..., description="Created document ID")
    title: str = Field(..., description="Title of the memory")
    has_embedding: bool = Field(..., description="Whether embedding was generated")
    acknowledged: bool = Field(
        default=True, description="Whether operation was acknowledged"
    )


class CreateTaxonomicResult(BaseModel):
    """Result of creating a taxonomic memory."""

    id: str = Field(..., description="Created document ID")
    domain: str = Field(..., description="Domain of the taxonomic entry")
    term: str = Field(..., description="Term defined")
    has_embedding: bool = Field(..., description="Whether embedding was generated")
    acknowledged: bool = Field(
        default=True, description="Whether operation was acknowledged"
    )


class CreateProceduralResult(BaseModel):
    """Result of creating a procedural memory."""

    id: str = Field(..., description="Created document ID")
    procedure: str = Field(..., description="Procedure name")
    has_embedding: bool = Field(..., description="Whether embedding was generated")
    acknowledged: bool = Field(
        default=True, description="Whether operation was acknowledged"
    )


class CreateUserContextResult(BaseModel):
    """Result of creating a user context memory."""

    id: str = Field(..., description="Created document ID")
    context_key: str = Field(..., description="Context key")
    acknowledged: bool = Field(
        default=True, description="Whether operation was acknowledged"
    )


class CreateSnapshotResult(BaseModel):
    """Result of creating a snapshot memory."""

    id: str = Field(..., description="Created document ID")
    session_id: str = Field(..., description="Session ID")
    message_count: int = Field(..., description="Number of messages in snapshot")
    has_embedding: bool = Field(..., description="Whether embedding was generated")
    embedding_strategy: str | None = Field(None, description="Embedding strategy used")
    snapshot_reason: str = Field(default="manual", description="Reason for snapshot")
    acknowledged: bool = Field(
        default=True, description="Whether operation was acknowledged"
    )


class PromoteSnapshotResult(BaseModel):
    """Result of promoting STM to snapshot."""

    snapshot_id: str | None = Field(
        None, description="Snapshot ID (None if no promotion)"
    )
    session_id: str = Field(..., description="Session ID")
    promoted_count: int = Field(..., description="Number of turns promoted")
    reason: str = Field(..., description="Promotion reason")
    has_embedding: bool = Field(
        default=False, description="Whether embedding was generated"
    )
    acknowledged: bool = Field(
        default=True, description="Whether operation was acknowledged"
    )


class DeleteResult(BaseModel):
    """Result of a delete operation."""

    deleted_count: int = Field(..., description="Number of documents deleted")
    acknowledged: bool = Field(
        default=True, description="Whether operation was acknowledged"
    )


class InternalStateResult(BaseModel):
    """
    Result of updating internal state.

    Returns key information for verification:
    - session_id: Which session was updated
    - version: New version number (increments on each update)
    - token_count: Token count of the new content
    - content_hash: SHA256[:16] for integrity verification
    """

    session_id: str = Field(..., description="Session ID")
    version: int = Field(..., description="New version number")
    token_count: int = Field(..., description="Token count of the content")
    content_hash: str = Field(..., description="SHA256[:16] hash for integrity")
    previous_version: int | None = Field(None, description="Previous version number")
    was_noop: bool = Field(default=False, description="True if content_hash unchanged")
    acknowledged: bool = Field(
        default=True, description="Whether operation was acknowledged"
    )


class ISGenerationResult(BaseModel):
    """
    Result of automatic IS generation via generate_is_from_turn().

    Tracks whether the update happened, was skipped, or coalesced.
    """

    updated: bool = Field(..., description="True if IS was generated and stored")
    version: int | None = Field(None, description="New IS version number")
    token_count: int | None = Field(None, description="Token count of generated IS")
    content_hash: str | None = Field(
        None, description="SHA256[:16] of generated content"
    )
    previous_version: int | None = Field(None, description="Version before this update")
    was_noop: bool = Field(default=False, description="True if content_hash unchanged")
    coalesced: bool = Field(default=False, description="True if update was queued")
    skipped_reason: str | None = Field(None, description="Why update was skipped")
    next_update_at: str | None = Field(
        None, description="ISO timestamp for next update"
    )
    model_id: str | None = Field(None, description="Model used for generation")
    events_hash: str | None = Field(None, description="Hash of input events")


# =============================================================================
# Retrieval Models (Memory Chunks and Context)
# =============================================================================


class MemorySource(str, Enum):
    """Source type for memory chunks."""

    STM = "stm"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    TAXONOMIC = "taxonomic"
    PROCEDURAL = "procedural"


class SearchSource(str, Enum):
    """A memory source that ``Memory.search`` can query.

    Mirrors ``MemorySource`` minus ``STM`` (short-term turns are not a search
    target). Callers may pass either the enum or its string value.
    """

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    TAXONOMIC = "taxonomic"
    PROCEDURAL = "procedural"


class RetrievalMode(str, Enum):
    """How a single source is searched in per-source context building.

    ``TEXT`` is lexical (``$search``), ``SEMANTIC`` is vector (``$vectorSearch``),
    and ``HYBRID`` fuses both. Callers may pass either the enum or its string
    value.
    """

    TEXT = "text"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class FormatStyle(str, Enum):
    """Output format for built context.

    Mirrors the server's format styles: ``OPENAI`` is a chat-message list that
    keeps STM turn roles, ``CLAUDE`` an XML string, and ``JINJA2`` a markdown
    string. Callers may pass either the enum or its string value.
    """

    OPENAI = "openai"
    CLAUDE = "claude"
    JINJA2 = "jinja2"


class SourceSpec(BaseModel):
    """Per-source retrieval configuration for ``Memory.build_context_from_sources``.

    Each enabled source declares its own retrieval mode, metadata filter, and
    candidate count, unlike ``build_context`` where one filter and one mode apply
    to every source.

    A ``metadata_filter`` key is the *fully qualified index path*, not the bare
    attribute name from the project config. An attribute declared as ``tier`` is
    indexed at ``metadata.tier``, and that is what the filter must say -- nothing
    is prefixed for you::

        # project config declares metadata partition keys: tier, score
        SourceSpec(
            source="semantic",
            mode="hybrid",
            metadata_filter={"metadata.tier": "gold", "metadata.score": {"$gt": 0.8}},
        )

    Unlike ``build_context``'s single filter, these keys are validated: an
    undeclared or misspelled path raises ``MemoryBadRequestError`` listing the
    declared paths, so a typo fails at the call instead of silently narrowing the
    result set.
    """

    source: MemorySource = Field(..., description="Which memory source to search")
    mode: RetrievalMode = Field(
        default=RetrievalMode.SEMANTIC,
        description="How to search this source (text, semantic, or hybrid)",
    )
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Field-level metadata filter for this source only. Filterable fields "
            "must be declared for the memory type in the project's metadata "
            "partition configuration, and must be written as the fully qualified "
            "index path: an attribute declared as 'tier' is filtered as "
            "'metadata.tier'. Filtering on an undeclared field is rejected with "
            "MemoryBadRequestError naming the declared paths - a misspelled key "
            "fails at the call rather than quietly returning a wrong result set."
        ),
    )
    top_k: int = Field(
        default=20,
        ge=1,
        # Mirrors the server's per-source maximum (MAX_SOURCE_TOP_K = 200), which
        # it enforces by rejecting larger values (HTTP 422), not by clamping.
        # Bounding here turns that into a clear local ValueError.
        le=200,
        description=(
            "Number of candidates this source contributes (1-200). The server "
            "rejects values above its per-source maximum of 200."
        ),
    )

    model_config = {
        "use_enum_values": True,
    }


class MemoryChunk(BaseModel):
    """
    Unified representation of memory from any source (STM, episodic, semantic).

    This model provides a common interface for memory chunks retrieved from
    different memory types, enabling uniform processing in the retrieval pipeline.
    """

    id: str = Field(..., description="Memory document ID")
    content: str = Field(..., description="Text content of the memory chunk")
    source: MemorySource = Field(
        ..., description="Source type: stm, episodic, or semantic"
    )
    timestamp: datetime = Field(
        ..., description="Timestamp when the memory was created"
    )
    embedding: list[float] | None = Field(
        default=None, description="Vector embedding for the memory chunk"
    )
    similarity_score: float | None = Field(
        default=None,
        description=(
            "Relevance score, on the scale of the retrieval mode that produced "
            "this chunk: vector similarity for semantic mode, text relevance for "
            "text mode, and the fused rank-fusion score for hybrid mode. "
            "Comparable between chunks retrieved the same way; not comparable "
            "across modes, or across sources searched differently. None when the "
            "chunk did not come from a search."
        ),
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata about the memory chunk",
    )

    model_config = {
        "arbitrary_types_allowed": True,
        "use_enum_values": True,
    }

    @property
    def org_id(self) -> str | None:
        """Get org_id from metadata if available."""
        if self.metadata:
            return self.metadata.get("org_id")
        return None


class SourceOutcome(TypedDict):
    """One entry of ``ContextMetadata.source_outcomes`` — what happened for a
    single source on the per-source context path.

    The wire shape the server emits (per-source builder serializes its internal
    ``SourceOutcome`` dataclass to this dict). Enum-valued fields arrive as their
    string values. ``error`` is ``None`` unless the source failed; a
    ``requested_mode``/``effective_mode`` mismatch flags a source that ran in a
    degraded mode. ``retryable`` is true only for a first-use lazy index
    build — retry shortly rather than treat the empty contribution as
    "no memories". Absent from older servers, so treat a missing key as false.
    """

    source: str
    requested_mode: str
    effective_mode: str
    count: int
    error: str | None
    retryable: NotRequired[bool]


class ContextMetadata(BaseModel):
    """
    Metadata about the context building process.

    Provides information about token usage, memory counts, and timing.
    """

    token_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Token count of the formatted context output only; excludes the "
            "token_buffer formatting reserve"
        ),
    )
    memory_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Count of memories by source type",
    )
    timing: dict[str, float] = Field(
        default_factory=dict,
        description="Timing information in seconds",
    )
    ranking_strategy: str | None = Field(
        default=None,
        description=(
            "Ordering strategy that produced the final memory order, when the "
            "per-source context path built it (e.g. 'client_rrf', "
            "'voyage_rerank'). Unset for the legacy /retrieval/context path."
        ),
    )
    source_outcomes: list[SourceOutcome] | None = Field(
        default=None,
        description=(
            "Per-source retrieval outcomes for the per-source context path; "
            "unset for the legacy /retrieval/context path. Each entry is "
            "{source, requested_mode, effective_mode, count, error}, so a caller "
            "can distinguish a failed or degraded source from one that matched "
            "nothing. A partial failure still returns 200 with the failure here."
        ),
    )


class ContextResponse(BaseModel):
    """
    Final response model for context building.

    Contains the formatted context and metadata about the retrieval process.
    """

    formatted_context: str | list[dict[str, Any]] = Field(
        ...,
        description="Formatted context as string or list of dicts",
    )
    metadata: ContextMetadata = Field(
        ...,
        description="Metadata about the context building process",
    )
    selected_memories: list[MemoryChunk] | None = Field(
        default=None,
        description="Selected memory chunks (only if include_memories=True)",
    )

    model_config = {
        "arbitrary_types_allowed": True,
    }


# =============================================================================
# Custom-Type Models
# =============================================================================


class CustomMemorySaveResult(BaseModel):
    """Result of saving a custom-type memory."""

    id: str = Field(..., description="The stored memory's identifier")
    type: str = Field(..., description="The custom memory type name")
    tags: dict[str, TagScalar] = Field(..., description="The stored, echoed tag values")
    has_embedding: bool = Field(..., description="Whether the content was embedded")


class RetrievedCustomMemory(BaseModel):
    """One retrieved custom-type memory."""

    id: str = Field(..., description="The memory's identifier")
    type: str = Field(..., description="The custom memory type name")
    content: str = Field(..., description="The stored content")
    tags: dict[str, TagScalar] = Field(
        ..., description="The memory's stored tag values"
    )
    contextual_metadata: dict[str, Any] | None = Field(
        default=None,
        description="The memory's stored additional context, when present",
    )
    user_id: str | None = Field(
        default=None, description="User ID the memory was saved under"
    )
    agent_id: str | None = Field(
        default=None, description="Agent ID the memory was saved under"
    )


class CustomMemoryRetrieveResult(BaseModel):
    """Result of retrieving custom-type memories."""

    results: list[RetrievedCustomMemory] = Field(
        ..., description="Matching memories, best first"
    )
    count: int = Field(..., description="Number of memories returned")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Result models
    "WriteTurnResult",
    "CreateSemanticResult",
    "BulkCreateSemanticResult",
    "CreateEpisodicResult",
    "CreateTaxonomicResult",
    "CreateProceduralResult",
    "CreateUserContextResult",
    "CreateSnapshotResult",
    "PromoteSnapshotResult",
    "DeleteResult",
    "InternalStateResult",
    "ISGenerationResult",
    # Retrieval enums
    "MemorySource",
    "SearchSource",
    "RetrievalMode",
    "FormatStyle",
    # Retrieval models
    "SourceSpec",
    "MemoryChunk",
    "SourceOutcome",
    "ContextMetadata",
    "ContextResponse",
    # Custom-type models
    "TagScalar",
    "CustomMemorySaveResult",
    "RetrievedCustomMemory",
    "CustomMemoryRetrieveResult",
]
