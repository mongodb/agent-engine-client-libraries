"""HTTP client for the Memory Server API.

Implements MemoryEngineProtocol by making HTTP calls to the memory server,
providing a clean HTTP interface to the standalone memory-server service.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import quote

import httpx

from agent_engine_sdk_memory._transport import raise_for_status
from agent_engine_sdk_memory._wire import custom_retrieve_body, custom_save_body
from agent_engine_sdk_memory.models import (
    ContextResponse,
    CreateEpisodicResult,
    CreateProceduralResult,
    CreateSemanticResult,
    CreateTaxonomicResult,
    CustomMemoryRetrieveResult,
    CustomMemorySaveResult,
    DeleteResult,
    FormatStyle,
    MemoryChunk,
    SourceSpec,
    WriteTurnResult,
)
from agent_engine_sdk_memory.validation import (
    require_positive_max_tokens,
    require_session_for_stm,
    require_valid_format_style,
    require_valid_sources,
)

logger = logging.getLogger(__name__)

__all__ = ["MemoryClient"]

# Headers injected into every request so the memory server can identify
# the calling agent and the active execution. Both the current and legacy
# names are sent during the rename transition; the server prefers the
# X-Agent-Engine-* name.
_HEADER_AGENT_ID = "X-Agent-Engine-Agent-Id"
_HEADER_AGENT_ID_LEGACY = "X-Agentic-Agent-Id"
_HEADER_EXECUTION_ID = "X-Agent-Engine-Execution-Id"
_HEADER_EXECUTION_ID_LEGACY = "X-Agentic-Execution-Id"


def _require_session_id(session_id: str | None) -> str:
    """Return a non-empty session ID or raise before issuing an HTTP request."""
    if not session_id or not session_id.strip():
        raise ValueError("session_id is required for memory server requests")
    return session_id.strip()


def _quote_path_segment(value: object) -> str:
    """URL-encode a value for interpolation as a single URL path segment.

    Memory ids are caller- or server-controlled strings; encoding keeps '?'
    and '#' from breaking the request's query string and '..' segments from
    traversing outside the memory route. Bare dot segments are
    refused outright: quote() leaves dots untouched, so an id of exactly '.'
    or '..' would survive encoding and collapse into a parent-path rewrite
    when the URL is parsed.
    """
    s = str(value)
    if s in (".", ".."):
        raise ValueError(f"Invalid memory id: bare dot segment {s!r}")
    return quote(s, safe="")


_CREATE_EPISODIC_EXTRA_FIELDS = (
    "snapshot_ref_id",
    "summary_type",
    "source_agent",
    "participants",
    "tags",
    "extraction_source",
    "embedding",
    "skip_embedding",
)


class MemoryClient:
    """
    HTTP client for Memory Server that implements MemoryEngineProtocol.

    Usage:
        client = MemoryClient("http://127.0.0.1:8081")

        # Write conversation turn
        client.write_turn(
            session_id="thread_123",
            role="user",
            content="Hello",
            org_id="org_1",
            user_id="user_1",
        )

        # Build context
        context = client.build_context(
            query="What did we discuss?",
            session_id="thread_123",
            org_id="org_1",
            user_id="user_1",
        )
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        static_headers: dict[str, str] | None = None,
        api_prefix: str = "/api/v1/memory",
        execution_id_provider: Callable[[], str | None] | None = None,
        transport: httpx.BaseTransport | None = None,
        ca_bundle: str | bool | None = None,
        http_client: httpx.Client | None = None,
    ):
        """
        Initialize memory client.

        Args:
            base_url: Memory server base URL (e.g., "http://127.0.0.1:8081")
            timeout: Request timeout in seconds (default: 30.0)
            static_headers: Pod-level identity headers set at construction time
                (e.g. X-Agent-Engine-Agent-Id from APP_ID env var). Defensively
                copied so caller mutations after construction have no effect.
            api_prefix: URL path prefix for memory endpoints. Defaults to
                "/api/v1/memory" (OE proxy route). Use "/api/v1" when
                connecting directly to the memory server.
            execution_id_provider: Optional callable returning the current
                execution id (or None). Read on every request so per-request
                execution-id semantics are preserved without this package
                depending on any execution-context source.
            transport: Optional httpx transport. Defaults to None (the standard
                network transport); inject an httpx.MockTransport to unit-test
                CRUD without a live server. Ignored if http_client is provided.
            ca_bundle: Path to CA certificate bundle for verifying HTTPS connections.
                If None, uses system default CA bundle. Set to False to disable
                SSL verification (not recommended). Ignored if http_client is provided.
            http_client: Optional pre-configured httpx.Client. If provided, takes
                precedence over timeout/transport/ca_bundle parameters. Use this to
                inject a fully-configured mTLS client (e.g., from
                agent_engine_runner_shared.tls_client.create_httpx_client_with_tls).
        """
        self._base_url = base_url.rstrip("/")
        self._api_prefix = "/" + api_prefix.strip("/")

        # Use injected client if provided, otherwise create one
        if http_client is not None:
            self._client = http_client
        else:
            # Configure SSL verification: ca_bundle can be path (str),
            # False (disable), or None (system default)
            verify: bool | str = True if ca_bundle is None else ca_bundle
            self._client = httpx.Client(
                timeout=timeout, transport=transport, verify=verify
            )
        self._static_headers: dict[str, str] = dict(static_headers or {})
        self._execution_id_provider = execution_id_provider
        logger.info(f"MemoryClient initialized: {self._base_url}")

    def _request_headers(self) -> dict[str, str]:
        """Return identity headers for a single request.

        Static pod-level headers (set at construction) are merged with the
        execution_id from the optional caller-supplied provider. The provider
        is read on every request so per-request execution-id semantics are
        preserved; agent_engine_sdk_memory imports nothing from the platform stack.
        """
        headers = dict(self._static_headers)
        if self._execution_id_provider is not None:
            if execution_id := self._execution_id_provider():
                headers[_HEADER_EXECUTION_ID] = execution_id
                headers[_HEADER_EXECUTION_ID_LEGACY] = execution_id
        return headers

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> MemoryClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # =========================================================================
    # Short-Term Memory (STM)
    # =========================================================================

    def write_turn(
        self,
        session_id: str,
        role: str,
        org_id: str,
        user_id: str,
        project_id: str,
        content: str | None = None,
        tokens: int | None = None,
        stop_reason: str | None = None,
        model_name: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        is_error: bool = False,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> WriteTurnResult:
        """Write a conversation turn to short-term memory."""
        body: dict[str, Any] = {
            "session_id": _require_session_id(session_id),
            "role": role,
            "org_id": org_id,
            "user_id": user_id,
            "project_id": project_id,
            "content": content,
            "tokens": tokens,
            "stop_reason": stop_reason,
            "model_name": model_name,
            "tool_calls": tool_calls,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "is_error": is_error,
        }
        if metadata is not None:
            body["metadata"] = metadata

        resp = self._client.post(
            f"{self._base_url}{self._api_prefix}/stm/turns",
            json=body,
            headers=self._request_headers(),
        )
        resp.raise_for_status()
        return WriteTurnResult.model_validate(resp.json())

    # =========================================================================
    # Context Building
    # =========================================================================

    def build_context(
        self,
        query: str,
        session_id: str,
        org_id: str,
        user_id: str,
        project_id: str,
        visibility: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        enabled_sources: set[str] | None = None,
        format_style: FormatStyle | str | None = None,
        top_k: int = 50,
        max_tokens: int | None = None,
        include_memories: bool = False,
        **kwargs: Any,
    ) -> ContextResponse:
        """Build formatted context from memories."""
        require_positive_max_tokens(max_tokens)
        require_valid_format_style(format_style)
        body: dict[str, Any] = {
            "query": query,
            "session_id": _require_session_id(session_id),
            "org_id": org_id,
            "user_id": user_id,
            "visibility": visibility,
            "project_id": project_id,
            "enabled_sources": list(enabled_sources) if enabled_sources else None,
            "format_style": format_style,
            "top_k": top_k,
            "include_memories": include_memories,
        }
        if metadata_filter:
            body["metadata_filter"] = metadata_filter
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        resp = self._client.post(
            f"{self._base_url}{self._api_prefix}/retrieval/context",
            json=body,
            headers=self._request_headers(),
        )
        resp.raise_for_status()
        return ContextResponse.model_validate(resp.json())

    def build_context2(
        self,
        query: str,
        sources: list[SourceSpec],
        org_id: str,
        user_id: str,
        project_id: str,
        session_id: str | None = None,
        visibility: str | None = None,
        rerank: bool = False,
        max_tokens: int | None = None,
        format_style: FormatStyle | str | None = None,
        include_memories: bool = False,
        **kwargs: Any,
    ) -> ContextResponse:
        """Build context from an explicit, per-source-configured set of sources.

        Unlike ``build_context``, each source in ``sources`` carries its own
        retrieval mode, metadata filter, and candidate count; results are merged
        cross-source and optionally reranked. ``session_id`` is only required
        when the ``stm`` source is requested.
        """
        require_positive_max_tokens(max_tokens)
        require_valid_format_style(format_style)
        specs = require_valid_sources(sources)
        require_session_for_stm(specs, session_id)
        body: dict[str, Any] = {
            "query": query,
            "org_id": org_id,
            "user_id": user_id,
            "project_id": project_id,
            "sources": [spec.model_dump(exclude_none=True) for spec in specs],
            "rerank": rerank,
            "include_memories": include_memories,
        }
        if session_id is not None:
            body["session_id"] = session_id
        if visibility is not None:
            body["visibility"] = visibility
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if format_style is not None:
            body["format_style"] = format_style

        resp = self._client.post(
            f"{self._base_url}{self._api_prefix}/retrieval/context-from-sources",
            json=body,
            headers=self._request_headers(),
        )
        resp.raise_for_status()
        return ContextResponse.model_validate(resp.json())

    # =========================================================================
    # Semantic Memory
    # =========================================================================

    def create_semantic(
        self,
        label: str,
        text: str,
        org_id: str,
        user_id: str,
        project_id: str,
        source: str = "agent",
        visibility: str = "private",
        agent_id: str | None = None,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        upsert: bool = False,
        **kwargs: Any,
    ) -> CreateSemanticResult:
        """Create a semantic memory."""
        # Back-compat: `contextual_metadata` was renamed to `metadata`. Forward a
        # value passed under the old name so pre-rename callers keep working, and
        # warn so they migrate before the fallback is removed.
        if "contextual_metadata" in kwargs:
            warnings.warn(
                "`contextual_metadata` is deprecated for create_semantic; use "
                "`metadata`. The value is being forwarded to `metadata`.",
                DeprecationWarning,
                stacklevel=2,
            )
            if metadata is None:
                metadata = kwargs.pop("contextual_metadata")
            else:
                kwargs.pop("contextual_metadata")
        body = {
            "label": label,
            "text": text,
            "org_id": org_id,
            "user_id": user_id,
            "source": source,
            "visibility": visibility,
            "project_id": project_id,
            "agent_id": agent_id,
            "embedding": embedding,
            "metadata": metadata,
            "upsert": upsert,
        }

        resp = self._client.post(
            f"{self._base_url}{self._api_prefix}/semantic",
            json=body,
            headers=self._request_headers(),
        )
        resp.raise_for_status()
        return CreateSemanticResult.model_validate(resp.json())

    def fetch_semantic_memories(
        self,
        query: str,
        org_id: str,
        project_id: str,
        user_id: str | None = None,
        visibility: str | None = None,
        top_k: int = 50,
        **kwargs: Any,
    ) -> list[MemoryChunk]:
        """Search semantic memories."""
        body = {
            "query": query,
            "org_id": org_id,
            "project_id": project_id,
            "user_id": user_id,
            "visibility": visibility,
            "top_k": top_k,
        }

        resp = self._client.post(
            f"{self._base_url}{self._api_prefix}/retrieval/semantic",
            json=body,
            headers=self._request_headers(),
        )
        resp.raise_for_status()
        memories_data = resp.json().get("memories", [])
        return [MemoryChunk.model_validate(m) for m in memories_data]

    def get_semantic(
        self,
        org_id: str,
        project_id: str,
        label: str | None = None,
        id: str | None = None,
        user_id: str | None = None,
        visibility: str | None = None,
        **kwargs: Any,
    ) -> Any | None:
        """Get semantic memory by label or ID."""
        params = {"org_id": org_id, "project_id": project_id}
        if user_id:
            params["user_id"] = user_id
        if visibility:
            params["visibility"] = visibility
        if label:
            params["label"] = label

        # If ID provided, use direct endpoint
        if id:
            id_params: dict[str, Any] = {"org_id": org_id, "project_id": project_id}
            if user_id:
                id_params["user_id"] = user_id
            if visibility:
                id_params["visibility"] = visibility
            resp = self._client.get(
                f"{self._base_url}{self._api_prefix}/semantic/{_quote_path_segment(id)}",
                params=id_params,
                headers=self._request_headers(),
            )
        else:
            # Otherwise list with label filter
            resp = self._client.get(
                f"{self._base_url}{self._api_prefix}/semantic",
                params=params,
                headers=self._request_headers(),
            )

        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        data = resp.json()
        # If list endpoint, return first match
        if "entries" in data:
            entries = data["entries"]
            return entries[0] if entries else None
        return data

    def update_semantic(
        self,
        org_id: str,
        project_id: str,
        label: str,
        text: str | None = None,
        source: str | None = None,
        visibility: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Update semantic memory."""
        # First get the memory ID by label
        memory = self.get_semantic(org_id=org_id, project_id=project_id, label=label)
        if not memory:
            raise ValueError(f"Semantic memory not found: {label}")

        memory_id = memory.get("id") or memory.get("_id")

        body = {
            "org_id": org_id,
            "project_id": project_id,
            "text": text,
            "source": source,
            "visibility": visibility,
        }

        resp = self._client.patch(
            f"{self._base_url}{self._api_prefix}/semantic/{_quote_path_segment(memory_id)}",
            json=body,
            headers=self._request_headers(),
        )
        resp.raise_for_status()
        return resp.json()

    # =========================================================================
    # Episodic Memory
    # =========================================================================

    def create_episodic(
        self,
        title: str,
        content: str,
        summary_text: str,
        org_id: str,
        user_id: str,
        session_id: str,
        project_id: str,
        visibility: str = "private",
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> CreateEpisodicResult:
        """Create an episodic memory."""
        body: dict[str, Any] = {
            "title": title,
            "content": content,
            "summary_text": summary_text,
            "org_id": org_id,
            "user_id": user_id,
            "session_id": _require_session_id(session_id),
            "visibility": visibility,
            "project_id": project_id,
            "agent_id": agent_id,
        }
        if metadata is not None:
            body["metadata"] = metadata
        for field in _CREATE_EPISODIC_EXTRA_FIELDS:
            if field in kwargs:
                body[field] = kwargs[field]

        resp = self._client.post(
            f"{self._base_url}{self._api_prefix}/episodic",
            json=body,
            headers=self._request_headers(),
        )
        resp.raise_for_status()
        return CreateEpisodicResult.model_validate(resp.json())

    def fetch_episodic_memories(
        self,
        query: str,
        org_id: str,
        project_id: str,
        user_id: str | None = None,
        visibility: str | None = None,
        session_id: str | None = None,
        top_k: int = 50,
        **kwargs: Any,
    ) -> list[MemoryChunk]:
        """Search episodic memories."""
        body = {
            "query": query,
            "org_id": org_id,
            "project_id": project_id,
            "user_id": user_id,
            "visibility": visibility,
            "session_id": session_id,
            "top_k": top_k,
        }

        resp = self._client.post(
            f"{self._base_url}{self._api_prefix}/retrieval/episodic",
            json=body,
            headers=self._request_headers(),
        )
        resp.raise_for_status()
        memories_data = resp.json().get("memories", [])
        return [MemoryChunk.model_validate(m) for m in memories_data]

    def list_episodic(
        self,
        org_id: str,
        project_id: str,
        user_id: str | None = None,
        session_id: str | None = None,
        visibility: str | None = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> list[Any]:
        """List episodic memories."""
        # Omit optional filters when unset rather than sending them. httpx
        # serializes a None or blank value as an empty query param
        # (`session_id=`), not by dropping it, so an unset filter would still
        # reach the server as a blank value. A bound/ambient session is no longer
        # inherited for episodic reads, so session_id arrives unset here
        # unless the caller passed one explicitly; user_id is likewise unset on
        # broad-visibility reads. Blank or whitespace-only values count as unset,
        # matching identity._normalize, so omitting them keeps "unset"
        # unambiguous instead of a blank the server might treat as a filter.
        params: dict[str, Any] = {
            "org_id": org_id,
            "project_id": project_id,
            "limit": limit,
        }
        if user_id and user_id.strip():
            params["user_id"] = user_id
        if session_id and session_id.strip():
            params["session_id"] = session_id
        if visibility and visibility.strip():
            params["visibility"] = visibility

        resp = self._client.get(
            f"{self._base_url}{self._api_prefix}/episodic",
            params=params,
            headers=self._request_headers(),
        )
        resp.raise_for_status()
        return resp.json().get("entries", [])

    # =========================================================================
    # Taxonomic Memory
    # =========================================================================

    def create_taxonomic(
        self,
        domain: str,
        term: str,
        definition: str,
        org_id: str,
        user_id: str,
        project_id: str,
        related_terms: list[str] | None = None,
        query_expansion: bool = True,
        visibility: str = "org",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> CreateTaxonomicResult:
        """Create a taxonomic memory."""
        body: dict[str, Any] = {
            "domain": domain,
            "term": term,
            "definition": definition,
            "org_id": org_id,
            "user_id": user_id,
            "project_id": project_id,
            "related_terms": related_terms,
            "query_expansion": query_expansion,
            "visibility": visibility,
        }
        if metadata is not None:
            body["metadata"] = metadata

        resp = self._client.post(
            f"{self._base_url}{self._api_prefix}/taxonomic",
            json=body,
            headers=self._request_headers(),
        )
        resp.raise_for_status()
        return CreateTaxonomicResult.model_validate(resp.json())

    def fetch_taxonomic_memories(
        self,
        query: str,
        org_id: str,
        project_id: str,
        domain: str | None = None,
        visibility: str | None = None,
        top_k: int = 50,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> list[MemoryChunk]:
        """Search taxonomic memories."""
        body = {
            "query": query,
            "org_id": org_id,
            "project_id": project_id,
            "user_id": user_id,
            "domain": domain,
            "visibility": visibility,
            "top_k": top_k,
        }

        resp = self._client.post(
            f"{self._base_url}{self._api_prefix}/retrieval/taxonomic",
            json=body,
            headers=self._request_headers(),
        )
        resp.raise_for_status()
        memories_data = resp.json().get("memories", [])
        return [MemoryChunk.model_validate(m) for m in memories_data]

    def get_taxonomic(
        self,
        org_id: str,
        project_id: str,
        domain: str | None = None,
        term: str | None = None,
        user_id: str | None = None,
        visibility: str | None = None,
        **kwargs: Any,
    ) -> Any | None:
        """Get taxonomic memory by domain and term."""
        params: dict[str, Any] = {
            "org_id": org_id,
            "project_id": project_id,
            "domain": domain,
        }
        if user_id:
            params["user_id"] = user_id
        if visibility:
            params["visibility"] = visibility

        resp = self._client.get(
            f"{self._base_url}{self._api_prefix}/taxonomic",
            params=params,
            headers=self._request_headers(),
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        entries = resp.json().get("entries", [])
        # Filter by term if provided
        if term:
            for entry in entries:
                if entry.get("term") == term:
                    return entry
            return None
        return entries[0] if entries else None

    def get_distinct_domains(
        self,
        org_id: str,
        project_id: str,
        visibility: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Get distinct taxonomic domains."""
        params: dict[str, Any] = {"org_id": org_id, "project_id": project_id}
        if visibility:
            params["visibility"] = visibility
        resp = self._client.get(
            f"{self._base_url}{self._api_prefix}/taxonomic/domains",
            params=params,
            headers=self._request_headers(),
        )
        resp.raise_for_status()
        return resp.json().get("domains", [])

    # =========================================================================
    # Procedural Memory
    # =========================================================================

    def create_procedural(
        self,
        *,
        procedure: str,
        description: str,
        content: str,
        org_id: str,
        user_id: str,
        steps: list[dict[str, Any]] | None = None,
        resources: list[dict[str, Any]] | None = None,
        allowed_tools: list[str] | None = None,
        compatibility: str | None = None,
        license: str | None = None,
        trigger_conditions: list[str] | None = None,
        tags: list[str] | None = None,
        visibility: str = "private",
        project_id: str,
        agent_id: str | None = None,
        extraction_source: str | None = None,
        source_format: str | None = None,
        source_path: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> CreateProceduralResult:
        """Create a procedural memory.

        The server's create route always creates; it has no update-existing
        semantics, so this client does not carry an ``update_existing`` flag.
        Callers that need update-or-create do the lookup-then-update themselves.
        """
        body: dict[str, Any] = {
            "procedure": procedure,
            "description": description,
            "content": content,
            "org_id": org_id,
            "user_id": user_id,
            "steps": steps,
            "resources": resources,
            "allowed_tools": allowed_tools,
            "compatibility": compatibility,
            "license": license,
            "trigger_conditions": trigger_conditions,
            "tags": tags,
            "visibility": visibility,
            "project_id": project_id,
            "agent_id": agent_id,
            "extraction_source": extraction_source,
            "source_format": source_format,
            "source_path": source_path,
        }
        if metadata is not None:
            body["metadata"] = metadata

        resp = self._client.post(
            f"{self._base_url}{self._api_prefix}/procedural",
            json=body,
            headers=self._request_headers(),
        )
        resp.raise_for_status()
        return CreateProceduralResult.model_validate(resp.json())

    def get_procedural(
        self,
        org_id: str,
        project_id: str,
        *,
        id: str | None = None,
        procedure: str | None = None,
        user_id: str | None = None,
        visibility: str | None = None,
        include_deleted: bool = False,
        **kwargs: Any,
    ) -> Any | None:
        """Get a procedural memory by ID or procedure name."""
        if id:
            id_params: dict[str, Any] = {
                "org_id": org_id,
                "project_id": project_id,
                "include_deleted": include_deleted,
            }
            if user_id:
                id_params["user_id"] = user_id
            if visibility:
                id_params["visibility"] = visibility
            resp = self._client.get(
                f"{self._base_url}{self._api_prefix}/procedural/{_quote_path_segment(id)}",
                params=id_params,
                headers=self._request_headers(),
            )
        else:
            params: dict[str, Any] = {
                "org_id": org_id,
                "project_id": project_id,
                "include_deleted": include_deleted,
            }
            if user_id:
                params["user_id"] = user_id
            if visibility:
                params["visibility"] = visibility
            if procedure:
                params["procedure"] = procedure
            resp = self._client.get(
                f"{self._base_url}{self._api_prefix}/procedural",
                params=params,
                headers=self._request_headers(),
            )

        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        data = resp.json()
        if "entries" in data:
            entries = data["entries"]
            if procedure:
                for entry in entries:
                    if entry.get("procedure") == procedure:
                        return entry
                return None
            return entries[0] if entries else None
        return data

    def update_procedural(
        self,
        org_id: str,
        project_id: str,
        *,
        id: str | None = None,
        procedure: str | None = None,
        description: str | None = None,
        content: str | None = None,
        steps: list[dict[str, Any]] | None = None,
        resources: list[dict[str, Any]] | None = None,
        allowed_tools: list[str] | None = None,
        trigger_conditions: list[str] | None = None,
        tags: list[str] | None = None,
        visibility: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Update a procedural memory.

        ``user_id`` is required and must name the memory's owner, matching
        ``create_procedural``: procedural memories are user-owned, and writes
        without an owner identity are rejected.
        """
        if not user_id:
            raise ValueError("user_id is required to update procedural memory")
        memory_id = id
        if not memory_id and procedure:
            # Identity is org/project/owner/name. Resolve by name without a
            # visibility filter so an update can change visibility. OE still
            # requires user_id (or visibility) on proxied GETs.
            existing = self.get_procedural(
                org_id=org_id,
                project_id=project_id,
                procedure=procedure,
                user_id=user_id,
            )
            if not existing:
                raise ValueError(f"Procedural memory not found: {procedure}")
            memory_id = existing.get("id") or existing.get("_id")

        body: dict[str, Any] = {
            "org_id": org_id,
            "description": description,
            "content": content,
            "steps": steps,
            "resources": resources,
            "allowed_tools": allowed_tools,
            "trigger_conditions": trigger_conditions,
            "tags": tags,
            "visibility": visibility,
            "project_id": project_id,
            # OE enforces write identity from body fields; memory-server's
            # update model ignores this.
            "user_id": user_id,
        }
        if metadata is not None:
            body["metadata"] = metadata

        resp = self._client.patch(
            f"{self._base_url}{self._api_prefix}/procedural/{_quote_path_segment(memory_id)}",
            json=body,
            headers=self._request_headers(),
        )
        resp.raise_for_status()
        return resp.json()

    def delete_procedural(
        self,
        org_id: str,
        project_id: str,
        *,
        id: str | None = None,
        procedure: str | None = None,
        user_id: str | None = None,
        visibility: str | None = None,
        soft: bool = True,
        **kwargs: Any,
    ) -> DeleteResult:
        """Delete a procedural memory.

        ``user_id`` is required and must name the memory's owner, matching
        ``create_procedural``: procedural memories are user-owned, and writes
        without an owner identity are rejected.
        """
        if not user_id:
            raise ValueError("user_id is required to delete procedural memory")
        memory_id = id
        if not memory_id and procedure:
            # The OE proxy rejects identity-less lookups; scope the name
            # resolution the same way the caller scopes the delete.
            existing = self.get_procedural(
                org_id=org_id,
                project_id=project_id,
                procedure=procedure,
                user_id=user_id,
                visibility=visibility,
            )
            if not existing:
                raise ValueError(f"Procedural memory not found: {procedure}")
            memory_id = existing.get("id") or existing.get("_id")

        # OE applies its identity rule to DELETE query params too.
        params: dict[str, Any] = {
            "org_id": org_id,
            "project_id": project_id,
            "soft": soft,
            "user_id": user_id,
        }
        if visibility:
            params["visibility"] = visibility

        resp = self._client.delete(
            f"{self._base_url}{self._api_prefix}/procedural/{_quote_path_segment(memory_id)}",
            params=params,
            headers=self._request_headers(),
        )
        resp.raise_for_status()
        return DeleteResult.model_validate(resp.json())

    def discover_procedures(
        self,
        query: str,
        org_id: str,
        project_id: str,
        *,
        user_id: str | None = None,
        visibility: str | None = None,
        tags: list[str] | None = None,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Discover procedures matching a query (lightweight results)."""
        body: dict[str, Any] = {
            "query": query,
            "org_id": org_id,
            "user_id": user_id,
            "visibility": visibility,
            "project_id": project_id,
            "tags": tags,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
        }
        if metadata_filter:
            body["metadata_filter"] = metadata_filter

        resp = self._client.post(
            f"{self._base_url}{self._api_prefix}/retrieval/procedural",
            json=body,
            headers=self._request_headers(),
        )
        resp.raise_for_status()
        memories_data = resp.json().get("memories", [])
        return [
            {
                "procedure": m.get("metadata", {}).get("procedure", ""),
                "content": m.get("content", ""),
                "score": m.get("similarity_score", 0.0),
                **m.get("metadata", {}),
            }
            for m in memories_data
        ]

    def fetch_procedural_memories(
        self,
        query: str,
        org_id: str,
        project_id: str,
        *,
        user_id: str | None = None,
        visibility: str | None = None,
        tags: list[str] | None = None,
        top_k: int = 50,
        **kwargs: Any,
    ) -> list[MemoryChunk]:
        """Fetch procedural memories by query (full content)."""
        body = {
            "query": query,
            "org_id": org_id,
            "user_id": user_id,
            "visibility": visibility,
            "project_id": project_id,
            "tags": tags,
            "top_k": top_k,
        }

        resp = self._client.post(
            f"{self._base_url}{self._api_prefix}/retrieval/procedural",
            json=body,
            headers=self._request_headers(),
        )
        resp.raise_for_status()
        memories_data = resp.json().get("memories", [])
        return [MemoryChunk.model_validate(m) for m in memories_data]

    # =========================================================================
    # Custom Types
    # =========================================================================

    def create_custom(
        self,
        *,
        memory_type: str,
        content: str,
        tags: Mapping[str, Any] | None = None,
        contextual_metadata: dict[str, Any] | None = None,
    ) -> CustomMemorySaveResult:
        """Save a custom-type memory. Identity is stamped by the platform."""
        body = custom_save_body(
            content=content, tags=tags, contextual_metadata=contextual_metadata
        )

        resp = self._client.post(
            f"{self._base_url}{self._api_prefix}/types/{_quote_path_segment(memory_type)}",
            json=body,
            headers=self._request_headers(),
        )
        raise_for_status(resp)
        return CustomMemorySaveResult.model_validate(resp.json())

    def retrieve_custom(
        self,
        *,
        memory_type: str,
        query: str,
        tags: Mapping[str, Any] | None = None,
        top_k: int = 10,
    ) -> CustomMemoryRetrieveResult:
        """Retrieve custom-type memories by semantic query with equality tag filters."""
        body = custom_retrieve_body(query=query, top_k=top_k, tags=tags)

        resp = self._client.post(
            f"{self._base_url}{self._api_prefix}"
            f"/types/{_quote_path_segment(memory_type)}/retrieve",
            json=body,
            headers=self._request_headers(),
        )
        raise_for_status(resp)
        return CustomMemoryRetrieveResult.model_validate(resp.json())

    # =========================================================================
    # Bootstrap (no-op for HTTP client)
    # =========================================================================

    def bootstrap(self) -> None:
        """No-op for HTTP client - server handles bootstrap."""
        pass
