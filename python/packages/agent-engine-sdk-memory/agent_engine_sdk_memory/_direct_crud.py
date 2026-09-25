"""Empty-tenancy CRUD adapter for Gateway and OE modes.

Implements the ``MemoryCrudClient`` Protocol against the in-package
``_HttpTransport``, supplying ``org_id=""``/``project_id=""`` on every call.
Uses the same transport — and therefore the same connection pool and
typed-error mapping — as the ``MemoryRuntime`` workflow operations. Creates opt
out of the transport's retry loop (``retry=False``): they carry no idempotency
key, so retrying after the server committed could duplicate the record.

For Gateway mode the empty tenancy values are harmless: the Gateway
authenticates the API key and infers the real org/project from it before
routing to the memory server, so the SDK-supplied empty strings are overridden.

For OE mode the empty tenancy is intentional: the OE infers execution-scoped
tenancy from its own context and overrides any caller-supplied
org/project when a trusted execution context is present, so the SDK-supplied
empty strings are ignored by the serving layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from agent_engine_sdk_memory._transport import _HttpTransport
from agent_engine_sdk_memory._wire import custom_retrieve_body, custom_save_body
from agent_engine_sdk_memory.errors import MemoryNotSupportedError
from agent_engine_sdk_memory.models import (
    CreateEpisodicResult,
    CreateProceduralResult,
    CreateSemanticResult,
    CreateTaxonomicResult,
    CustomMemoryRetrieveResult,
    CustomMemorySaveResult,
)

__all__ = ["_EmptyTenancyCrudClient"]

# Operation suffixes appended to the runtime's resolved base_path, mirroring
# the core-loop suffixes in _http_runtime. The base path differs per backend
# (flat OE ``/api/v1/memory`` vs project-scoped Gateway
# ``/api/v1/projects/{project_id}/memory``); the suffixes do not.
_SEMANTIC_SUFFIX = "/semantic"
_EPISODIC_SUFFIX = "/episodic"
_TAXONOMIC_SUFFIX = "/taxonomic"
_DOMAINS_SUFFIX = "/taxonomic/domains"
_PROCEDURAL_SUFFIX = "/procedural"
_TYPES_SUFFIX = "/types"

_EMPTY_TENANCY: dict[str, str] = {"org_id": "", "project_id": ""}


class _EmptyTenancyCrudClient:
    """MemoryCrudClient adapter that injects empty org/project tenancy.

    Delegates directly to the shared ``_HttpTransport`` so CRUD calls share
    the same connection pool and typed-error mapping as the runtime workflow
    operations (record_turn, build_context, searches). CRUD paths hang off the
    runtime's resolved ``base_path``, so they follow the same route shape (flat
    OE or project-scoped Gateway) as the core loop. Creates are sent with
    ``retry=False`` because they carry no idempotency key.
    """

    def __init__(self, transport: _HttpTransport, base_path: str) -> None:
        self._t = transport
        self._base = base_path

    def close(self) -> None:
        """No-op: the transport is owned by the runtime, not this adapter."""

    # =========================================================================
    # Semantic
    # =========================================================================

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
        resp = self._t.post(
            self._base + _SEMANTIC_SUFFIX,
            {
                **_EMPTY_TENANCY,
                "label": label,
                "text": text,
                "user_id": user_id,
                "source": source,
                "visibility": visibility,
                "agent_id": agent_id,
                "metadata": metadata,
                "upsert": upsert,
            },
            retry=False,
        )
        return CreateSemanticResult.model_validate(
            self._t.json_object(resp, "create semantic")
        )

    def get_semantic(
        self,
        *,
        label: str,
        user_id: str | None = None,
        visibility: str | None = None,
    ) -> Any | None:
        params: dict[str, Any] = {**_EMPTY_TENANCY, "label": label}
        if user_id is not None:
            params["user_id"] = user_id
        if visibility is not None:
            params["visibility"] = visibility
        resp = self._t.get_optional(self._base + _SEMANTIC_SUFFIX, params)
        if resp is None:
            return None
        entries = self._t.json_object(resp, "get semantic").get("entries", [])
        return entries[0] if entries else None

    # =========================================================================
    # Episodic
    # =========================================================================

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
        body: dict[str, Any] = {
            **_EMPTY_TENANCY,
            "title": title,
            "content": content,
            "summary_text": summary_text or "",
            "user_id": user_id,
            "session_id": session_id,
            "visibility": visibility,
            "agent_id": agent_id,
            "participants": participants,
            "tags": tags,
        }
        if metadata is not None:
            body["metadata"] = metadata
        resp = self._t.post(
            self._base + _EPISODIC_SUFFIX,
            body,
            retry=False,
        )
        return CreateEpisodicResult.model_validate(
            self._t.json_object(resp, "create episodic")
        )

    def list_episodic(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        visibility: str | None = None,
        limit: int = 20,
    ) -> list[Any]:
        params: dict[str, Any] = {**_EMPTY_TENANCY, "limit": limit}
        if user_id is not None:
            params["user_id"] = user_id
        if session_id is not None:
            params["session_id"] = session_id
        if visibility is not None:
            params["visibility"] = visibility
        resp = self._t.get(self._base + _EPISODIC_SUFFIX, params)
        return self._t.json_object(resp, "list episodic").get("entries", [])

    # =========================================================================
    # Taxonomic
    # =========================================================================

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
        body: dict[str, Any] = {
            **_EMPTY_TENANCY,
            "domain": domain,
            "term": term,
            "definition": definition,
            "user_id": user_id,
            "related_terms": related_terms,
            "visibility": visibility,
        }
        if metadata is not None:
            body["metadata"] = metadata
        resp = self._t.post(
            self._base + _TAXONOMIC_SUFFIX,
            body,
            retry=False,
        )
        return CreateTaxonomicResult.model_validate(
            self._t.json_object(resp, "create taxonomic")
        )

    def get_taxonomic(
        self,
        *,
        domain: str,
        term: str | None = None,
        user_id: str | None = None,
        visibility: str | None = None,
    ) -> Any | None:
        params: dict[str, Any] = {**_EMPTY_TENANCY, "domain": domain}
        if user_id is not None:
            params["user_id"] = user_id
        if visibility is not None:
            params["visibility"] = visibility
        resp = self._t.get_optional(self._base + _TAXONOMIC_SUFFIX, params)
        if resp is None:
            return None
        entries = self._t.json_object(resp, "get taxonomic").get("entries", [])
        if term is not None:
            return next((e for e in entries if e.get("term") == term), None)
        return entries[0] if entries else None

    def get_distinct_domains(
        self,
        *,
        visibility: str | None = None,
    ) -> list[str]:
        params: dict[str, Any] = {**_EMPTY_TENANCY}
        if visibility is not None:
            params["visibility"] = visibility
        resp = self._t.get(self._base + _DOMAINS_SUFFIX, params)
        return self._t.json_object(resp, "list domains").get("domains", [])

    # =========================================================================
    # Procedural
    # =========================================================================

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
        if update_existing:
            # The HTTP memory backends serve only a create route that always
            # creates; there is no update-existing semantics over the wire, so
            # forwarding the flag would silently no-op. Surface it as a typed
            # error rather than create-instead-of-update. App-bound mode honors
            # update_existing through its own runtime.
            raise MemoryNotSupportedError(
                "update_existing is not supported by the HTTP memory backends; "
                "the create route always creates a new procedure"
            )
        body: dict[str, Any] = {
            **_EMPTY_TENANCY,
            "procedure": procedure,
            "description": description,
            "content": content,
            "user_id": user_id,
            "steps": steps,
            "resources": resources,
            "allowed_tools": allowed_tools,
            "compatibility": compatibility,
            "license": license,
            "trigger_conditions": trigger_conditions,
            "tags": tags,
            "visibility": visibility,
            "agent_id": agent_id,
            "extraction_source": extraction_source,
            "source_format": source_format,
            "source_path": source_path,
        }
        if metadata is not None:
            body["metadata"] = metadata
        resp = self._t.post(
            self._base + _PROCEDURAL_SUFFIX,
            body,
            retry=False,
        )
        return CreateProceduralResult.model_validate(
            self._t.json_object(resp, "create procedural")
        )

    def get_procedural(
        self,
        *,
        procedure: str | None = None,
        user_id: str | None = None,
        visibility: str | None = None,
        include_deleted: bool = False,
    ) -> Any | None:
        params: dict[str, Any] = {**_EMPTY_TENANCY, "include_deleted": include_deleted}
        if user_id is not None:
            params["user_id"] = user_id
        if visibility is not None:
            params["visibility"] = visibility
        if procedure is not None:
            params["procedure"] = procedure
        resp = self._t.get_optional(self._base + _PROCEDURAL_SUFFIX, params)
        if resp is None:
            return None
        entries = self._t.json_object(resp, "get procedural").get("entries", [])
        if procedure is not None:
            return next((e for e in entries if e.get("procedure") == procedure), None)
        return entries[0] if entries else None

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
        body = custom_save_body(
            content=content, tags=tags, contextual_metadata=contextual_metadata
        )
        resp = self._t.post(
            f"{self._base}{_TYPES_SUFFIX}/{quote(memory_type, safe='')}",
            body,
            retry=False,
        )
        return CustomMemorySaveResult.model_validate(
            self._t.json_object(resp, "create custom memory")
        )

    def retrieve_custom(
        self,
        *,
        memory_type: str,
        query: str,
        tags: Mapping[str, Any] | None = None,
        top_k: int = 10,
    ) -> CustomMemoryRetrieveResult:
        body = custom_retrieve_body(query=query, top_k=top_k, tags=tags)
        resp = self._t.post(
            f"{self._base}{_TYPES_SUFFIX}/{quote(memory_type, safe='')}/retrieve",
            body,
            retry=True,
        )
        return CustomMemoryRetrieveResult.model_validate(
            self._t.json_object(resp, "retrieve custom memories")
        )
