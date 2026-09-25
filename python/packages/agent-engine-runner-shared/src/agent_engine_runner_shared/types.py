"""
Type definitions and protocols for Runner SDK.

Defines MemoryEngineProtocol and MemoryContextProtocol for type-safe
integration with the platform memory service without creating a hard
dependency on the server-side implementation.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class MemoryContextProtocol(Protocol):
    """Protocol for memory context response."""

    @property
    def formatted_context(self) -> str:
        """Get the formatted context string."""
        ...


@runtime_checkable
class MemoryEngineProtocol(Protocol):
    """
    Protocol defining the interface for external MemoryEngine.

    This allows type checking without hard dependency on the platform memory service.
    """

    def build_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        org_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MemoryContextProtocol:
        """Build context for a query."""
        ...

    def write_turn(
        self,
        session_id: Optional[str],
        role: str,
        org_id: Optional[str],
        user_id: Optional[str] = None,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        is_error: bool = False,
        model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Write a single conversation turn to memory.

        Args:
            session_id: Conversation thread identifier
            role: Message role ("user", "assistant", "tool", "system")
            org_id: Organization/tenant ID
            user_id: User ID of the message author
            content: Message content (required for user/tool, optional for assistant)
            tool_calls: List of tool calls (for assistant messages)
            tool_call_id: ID of the tool call this result is for (for tool messages)
            tool_name: Name of the tool (for tool messages)
            is_error: Whether the tool execution failed (for tool messages)
            model_name: LLM model identifier (for assistant messages)
            metadata: Arbitrary key-value metadata stored with the turn
        """
        ...

    def bootstrap(self) -> None:
        """Initialize database indexes and collections."""
        ...

    # Semantic Memory Methods

    def create_semantic(
        self,
        *,
        label: str,
        text: str,
        org_id: str,
        user_id: str,
        source: str = "agent",
        visibility: str = "private",
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        upsert: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Create a semantic memory (long-lived facts and knowledge).

        Args:
            label: Unique label/identifier for this memory
            text: The memory content
            org_id: Organization ID for multi-tenancy
            user_id: User ID who owns this memory
            source: Source of the memory
            visibility: Visibility scope (private, shared, org)
            embedding: Pre-computed embedding vector
            metadata: Optional caller-supplied metadata dict stored with the memory
            upsert: If True, update existing memory with same label
        """
        ...

    def fetch_semantic_memories(
        self,
        query: str,
        org_id: str,
        *,
        user_id: Optional[str] = None,
        top_k: int = 50,
        **kwargs: Any,
    ) -> List[Any]:
        """
        Fetch semantic memories by query.

        Args:
            query: Query text for semantic search
            org_id: Organization ID
            user_id: Optional user ID filter
            top_k: Maximum number of results
        """
        ...

    def get_semantic(
        self,
        org_id: str,
        *,
        label: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Any]:
        """
        Get a semantic memory by label.

        Args:
            org_id: Organization ID
            label: Memory label
        """
        ...

    def update_semantic(
        self,
        org_id: str,
        *,
        label: str,
        text: Optional[str] = None,
        source: Optional[str] = None,
        visibility: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Update a semantic memory.

        Args:
            org_id: Organization ID
            label: Memory label
            text: New text content
            source: New source
            visibility: New visibility
        """
        ...

    # Taxonomic Memory Methods

    def create_taxonomic(
        self,
        *,
        domain: str,
        term: str,
        definition: str,
        org_id: str,
        user_id: str,
        related_terms: Optional[List[str]] = None,
        query_expansion: bool = True,
        visibility: str = "org",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create a taxonomic memory (domain-specific term with definition).

        Args:
            domain: Domain/category for this term
            term: The term being defined
            definition: Definition of the term
            org_id: Organization ID for multi-tenancy
            user_id: User ID who owns this memory
            related_terms: Related terms for semantic linking
            query_expansion: Whether to use this term for query expansion
            visibility: Visibility scope (default: "org")
            metadata: Arbitrary key-value metadata stored with the memory
        """
        ...

    def fetch_taxonomic_memories(
        self,
        query: str,
        org_id: str,
        *,
        domain: Optional[str] = None,
        top_k: int = 50,
        **kwargs: Any,
    ) -> List[Any]:
        """
        Fetch taxonomic memories by query.

        Args:
            query: Query text for semantic search
            org_id: Organization ID
            domain: Optional domain filter
            top_k: Maximum number of results
        """
        ...

    def get_taxonomic(
        self,
        org_id: str,
        *,
        domain: Optional[str] = None,
        term: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Any]:
        """
        Get a taxonomic memory by domain and term.

        Args:
            org_id: Organization ID
            domain: Domain name
            term: Term name
        """
        ...

    def get_distinct_domains(
        self,
        org_id: str,
        **kwargs: Any,
    ) -> List[str]:
        """
        Get distinct taxonomic domains for an organization.

        Args:
            org_id: Organization ID
        """
        ...

    # Procedural Memory Methods

    def create_procedural(
        self,
        *,
        procedure: str,
        description: str,
        content: str,
        org_id: str,
        user_id: str,
        steps: Optional[List[Dict[str, Any]]] = None,
        resources: Optional[List[Dict[str, Any]]] = None,
        allowed_tools: Optional[List[str]] = None,
        compatibility: Optional[str] = None,
        license: Optional[str] = None,
        trigger_conditions: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        visibility: str = "private",
        project_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        extraction_source: Optional[str] = None,
        source_format: Optional[str] = None,
        source_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a procedural memory (reusable workflow/skill)."""
        ...

    def get_procedural(
        self,
        org_id: str,
        *,
        id: Optional[str] = None,
        procedure: Optional[str] = None,
        include_deleted: bool = False,
        **kwargs: Any,
    ) -> Optional[Any]:
        """Get a procedural memory by ID or procedure name."""
        ...

    def update_procedural(
        self,
        org_id: str,
        *,
        id: Optional[str] = None,
        procedure: Optional[str] = None,
        description: Optional[str] = None,
        content: Optional[str] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        resources: Optional[List[Dict[str, Any]]] = None,
        allowed_tools: Optional[List[str]] = None,
        trigger_conditions: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        visibility: Optional[str] = None,
        project_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Update a procedural memory."""
        ...

    def delete_procedural(
        self,
        org_id: str,
        *,
        id: Optional[str] = None,
        procedure: Optional[str] = None,
        soft: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Delete a procedural memory."""
        ...

    def discover_procedures(
        self,
        query: str,
        org_id: str,
        *,
        user_id: Optional[str] = None,
        visibility: Optional[str] = None,
        project_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Discover procedures matching a query (lightweight results)."""
        ...

    def fetch_procedural_memories(
        self,
        query: str,
        org_id: str,
        *,
        user_id: Optional[str] = None,
        visibility: Optional[str] = None,
        project_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        top_k: int = 50,
        **kwargs: Any,
    ) -> List[Any]:
        """Fetch procedural memories by query (full content)."""
        ...

    # Episodic Memory Methods

    def create_episodic(
        self,
        *,
        title: str,
        content: str,
        summary_text: str,
        org_id: str,
        user_id: str,
        session_id: Optional[str] = None,
        snapshot_ref_id: Optional[str] = None,
        summary_type: str = "llm",
        source_agent: str = "agent",
        participants: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        visibility: str = "private",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create an episodic memory (conversation summary or event).

        Args:
            title: Title/label for this episode
            content: Full content of the episode
            summary_text: Brief summary of the episode
            org_id: Organization ID for multi-tenancy
            user_id: User ID who owns this memory
            session_id: Session/thread ID this episode originated from
            snapshot_ref_id: Reference ID for related snapshot
            summary_type: Type of summary ("llm", "manual")
            source_agent: Source of the summary
            participants: List of participants
            tags: Tags for categorization
            visibility: Visibility scope
        """
        ...

    def fetch_episodic_memories(
        self,
        query: str,
        org_id: str,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k: int = 50,
        **kwargs: Any,
    ) -> List[Any]:
        """
        Fetch episodic memories by query.

        Args:
            query: Query text for semantic search
            org_id: Organization ID
            user_id: Optional user ID filter
            session_id: Optional session ID filter
            top_k: Maximum number of results
        """
        ...

    def list_episodic(
        self,
        org_id: str,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> List[Any]:
        """
        List episodic memories with optional filters.

        Args:
            org_id: Organization ID
            user_id: Optional user ID filter
            session_id: Optional session ID filter
            limit: Maximum number of results
        """
        ...

    # Custom-type Memory Methods

    def create_custom(
        self,
        *,
        memory_type: str,
        content: str,
        tags: Optional[Dict[str, Any]] = None,
        contextual_metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Save a memory of a declared custom type.

        Args:
            memory_type: Name of the declared custom memory type
            content: The memory content
            tags: Optional key-value tags for filtering
            contextual_metadata: Optional metadata dict
        """
        ...

    def retrieve_custom(
        self,
        *,
        memory_type: str,
        query: str,
        tags: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> Any:
        """
        Retrieve memories of a declared custom type by semantic query.

        Args:
            memory_type: Name of the declared custom memory type
            query: Search query text
            tags: Optional key-value tags for filtering
            top_k: Maximum number of results
        """
        ...
