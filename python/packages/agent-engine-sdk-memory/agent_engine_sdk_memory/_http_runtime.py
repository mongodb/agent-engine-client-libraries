"""Single HTTP memory runtime, parameterized by endpoint and auth.

There is one runtime class. It speaks the public memory routes
(``/api/v1/memory/{turns,context,search}``) and takes three orthogonal inputs:
a ``base_url`` (which backend), an auth header (optional — ``api_key`` becomes a
bearer token, or no header at all), and an ``_EndpointProfile`` describing what
that backend can do. Auth is never a runtime subtype: pointing at a different
backend, or adding/removing a token, changes a value passed in, not the class.

A profile captures the capability differences between backends — currently the
hosted Gateway and a local Orchestration Engine — across a few axes:

* whether tool-call/model turn metadata is forwarded or rejected,
* whether taxonomic search is served at all,
* whether ranked semantic/procedural search is available, and
* whether type-specific CRUD is available.

The two profiles below are data, not subclasses. As the backends converge, the
profiles converge with them and collapse to one. This module is private and
exports nothing into the package root.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Any, cast
from urllib.parse import quote

import httpx

from agent_engine_sdk_memory._transport import _HttpTransport
from agent_engine_sdk_memory.errors import (
    MemoryBadRequestError,
    MemoryNotSupportedError,
    MemoryRouteNotFoundError,
)
from agent_engine_sdk_memory.models import (
    ContextResponse,
    FormatStyle,
    MemoryChunk,
    SourceSpec,
    WriteTurnResult,
)

# Operation suffixes appended to a profile's base_path. The base path differs
# per backend (flat OE vs project-scoped Gateway); the suffixes do not.
_TURNS_SUFFIX = "/turns"
_CONTEXT_SUFFIX = "/context"
# The per-source route is addressed by its native memory-server sub-path rather
# than a "/context-from-sources" alias: the OE proxy already forwards the whitelisted
# "retrieval/" prefix verbatim, so this reaches the server with no OE change.
_CONTEXT_FROM_SOURCES_SUFFIX = "/retrieval/context-from-sources"
_SEARCH_SUFFIX = "/search"


@dataclass(frozen=True)
class _EndpointProfile:
    """What a backend can do — capability data, not a runtime subtype.

    ``tool_turn_unsupported_reason`` is ``None`` when tool-call/model metadata
    is forwarded into the turn body; a non-empty string is the message of the
    ``MemoryNotSupportedError`` raised instead.

    ``taxonomic_unsupported_reason`` is ``None`` when taxonomic search works
    against this endpoint; a non-empty string is the message of the
    ``MemoryNotSupportedError`` raised by ``search_taxonomic`` before any HTTP
    call.

    ``ranked_search_unsupported_reason`` is ``None`` when ranked semantic and
    procedural search work against this endpoint; a non-empty string is the
    message raised by ``search_semantic``/``discover_procedures`` before any HTTP
    call.

    ``crud_unsupported_reason`` is ``None`` when type-specific CRUD
    (``save_*``/``get_*``/``list_*``) works against this endpoint; a non-empty
    string is the message raised when CRUD is attempted.

    ``base_path`` is the route prefix the operation suffixes hang off. The OE
    serves a flat surface (``/api/v1/memory``). The Gateway serves a
    project-scoped one (``/api/v1/projects/{project_id}/memory``). The
    ``{project_id}`` placeholder is filled at construction from the caller's
    ``project_id``. That presence is the route-shape selector (set means
    project-scoped Gateway, empty means flat OE), independent of auth.
    """

    tool_turn_unsupported_reason: str | None
    taxonomic_unsupported_reason: str | None
    ranked_search_unsupported_reason: str | None
    crud_unsupported_reason: str | None
    base_path: str


_GATEWAY_PROFILE = _EndpointProfile(
    tool_turn_unsupported_reason=None,
    taxonomic_unsupported_reason=None,
    ranked_search_unsupported_reason=None,
    crud_unsupported_reason=None,
    base_path="/api/v1/projects/{project_id}/memory",
)

_OE_PROFILE = _EndpointProfile(
    tool_turn_unsupported_reason=None,
    taxonomic_unsupported_reason=None,
    ranked_search_unsupported_reason=None,
    crud_unsupported_reason=None,
    base_path="/api/v1/memory",
)


class _HttpMemoryRuntime:
    """The six ``MemoryRuntime`` operations over one HTTP route map."""

    def __init__(
        self,
        *,
        base_url: str,
        profile: _EndpointProfile,
        project_id: str | None = None,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._transport = _HttpTransport(
            base_url=base_url,
            timeout=timeout,
            headers=headers,
            transport=transport,
        )
        self._profile = profile
        # Resolve the concrete route prefix once. A project-scoped base_path
        # (the Gateway) needs project_id; a flat one (the OE) ignores it. The
        # caller (memory.py) only pairs the Gateway profile with a project_id,
        # so a missing one here is a programming error surfaced at construction
        # rather than as an opaque request to ".../projects//memory/...".
        # Route-shape signal: a project-scoped profile carries the {project_id}
        # placeholder. Drives both the base-path fill below and the reactive-404
        # hint direction.
        self._project_scoped = "{project_id}" in profile.base_path
        base_path = profile.base_path
        if self._project_scoped:
            if project_id is None or not project_id.strip():
                raise ValueError(
                    "project_id is required for the project-scoped (Gateway) "
                    "endpoint profile; pass project_id or set "
                    "AGENTIC_MEMORY_PROJECT_ID to use project-scoped routes"
                )
            # Strip then URL-encode the segment. Stripping drops surrounding
            # whitespace that would otherwise be encoded into the path. Encoding
            # stops a project_id from injecting extra path segments or a query
            # string. A stray '/', '?', or '#' would otherwise misroute.
            base_path = base_path.format(project_id=quote(project_id.strip(), safe=""))
        self._base_path = base_path

    def _post(self, suffix: str, body: dict[str, Any]) -> httpx.Response:
        """POST a core-loop operation, enriching a 404 with a route-shape hint.

        A 404 on a core-loop route most often means the route shape (chosen from
        ``project_id`` presence) does not match the backend ``base_url`` points
        at. We surface that as a typed, directional, *likely-cause* hint rather
        than retrying on the other shape (silent fallback would hide the
        misconfiguration). Other 4xx/5xx pass through unchanged.
        """
        try:
            return self._transport.post(self._base_path + suffix, body)
        except MemoryBadRequestError as exc:
            if exc.status == 404:
                raise self._route_not_found_error(exc) from exc
            raise

    def _route_not_found_error(
        self, exc: MemoryBadRequestError
    ) -> MemoryRouteNotFoundError:
        if self._project_scoped:
            hint = (
                "project_id is set, so the SDK used a project-scoped memory "
                "route; if you are targeting a local or direct OE backend, unset "
                "project_id (or AGENTIC_MEMORY_PROJECT_ID)"
            )
        else:
            hint = (
                "project_id is empty, so the SDK used a flat memory route; if you "
                "are targeting the hosted Gateway, set project_id (or "
                "AGENTIC_MEMORY_PROJECT_ID)"
            )
        message = (
            "memory route not found (404). This often means the route shape does "
            f"not match the backend: {hint}. It can also mean base_url points at "
            "the wrong host."
        )
        return MemoryRouteNotFoundError(
            message,
            status=exc.status,
            code=exc.code,
            response_text=exc.response_text,
        )

    @property
    def transport(self) -> _HttpTransport:
        """Package-internal: the shared HTTP transport.

        Exposed without the leading underscore so cross-class composition
        (e.g. ``_EmptyTenancyCrudClient`` wired in ``memory.py``) satisfies
        pyright's ``reportPrivateUsage`` rule.
        """
        return self._transport

    @property
    def base_path(self) -> str:
        """Package-internal: the resolved route prefix (project_id already
        filled for the Gateway). The CRUD adapter hangs its operation suffixes
        off this so CRUD and the core loop share one route map."""
        return self._base_path

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> _HttpMemoryRuntime:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

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
        has_tool_metadata = (
            tool_calls is not None
            or tool_call_id is not None
            or tool_name is not None
            or model_name is not None
            or is_error
        )
        if has_tool_metadata and self._profile.tool_turn_unsupported_reason is not None:
            raise MemoryNotSupportedError(self._profile.tool_turn_unsupported_reason)

        body: dict[str, Any] = {
            "session_id": session_id,
            "user_id": user_id,
            "role": role,
            "content": content,
        }
        if agent_id is not None:
            body["agent_id"] = agent_id
        # Forwarded only when the mode supports it (guarded above); the turns
        # route accepts these fields, so a tool or assistant turn keeps its tool
        # calls, model attribution, and error flag.
        if model_name is not None:
            body["model_name"] = model_name
        if tool_calls is not None:
            body["tool_calls"] = tool_calls
        if tool_call_id is not None:
            body["tool_call_id"] = tool_call_id
        if tool_name is not None:
            body["tool_name"] = tool_name
        if is_error:
            body["is_error"] = is_error
        if metadata is not None:
            body["metadata"] = metadata
        if idempotency_key is not None:
            body["idempotency_key"] = idempotency_key
        resp = self._post(_TURNS_SUFFIX, body)
        return WriteTurnResult.model_validate(resp.json())

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
        """POST /memory/context.

        Omitted enabled_sources defaults to episodic and semantic.
        """
        body: dict[str, Any] = {
            "query": query,
            "user_id": user_id,
            "top_k": top_k,
        }
        if session_id is not None:
            body["session_id"] = session_id
        if visibility is not None:
            body["visibility"] = visibility
        if metadata_filter is not None:
            body["metadata_filter"] = metadata_filter
        if enabled_sources is not None:
            body["enabled_sources"] = sorted(enabled_sources)
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if format_style is not None:
            body["format_style"] = format_style
        if include_memories:
            body["include_memories"] = True
        resp = self._post(_CONTEXT_SUFFIX, body)
        return ContextResponse.model_validate(resp.json())

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
        """POST /retrieval/context-from-sources.

        Each source carries its own retrieval mode, metadata filter, and
        candidate count; tenancy is supplied by the runtime's route/headers.
        """
        body: dict[str, Any] = {
            "query": query,
            "sources": [spec.model_dump(exclude_none=True) for spec in sources],
            "rerank": rerank,
        }
        if user_id is not None:
            body["user_id"] = user_id
        if session_id is not None:
            body["session_id"] = session_id
        if visibility is not None:
            body["visibility"] = visibility
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if format_style is not None:
            body["format_style"] = format_style
        if include_memories:
            body["include_memories"] = True
        resp = self._post(_CONTEXT_FROM_SOURCES_SUFFIX, body)
        return ContextResponse.model_validate(resp.json())

    def _search(self, body: dict[str, Any]) -> list[MemoryChunk]:
        resp = self._post(_SEARCH_SUFFIX, body)
        return [MemoryChunk.model_validate(m) for m in self._transport.memories(resp)]

    def search_semantic(
        self,
        *,
        query: str,
        user_id: str | None = None,
        visibility: str | None = None,
        top_k: int = 50,
    ) -> list[MemoryChunk]:
        if self._profile.ranked_search_unsupported_reason is not None:
            raise MemoryNotSupportedError(
                self._profile.ranked_search_unsupported_reason
            )
        body: dict[str, Any] = {
            "type": "semantic",
            "query": query,
            "user_id": user_id,
            "top_k": top_k,
        }
        if visibility is not None:
            body["visibility"] = visibility
        return self._search(body)

    def search_episodes(
        self,
        *,
        query: str,
        user_id: str | None = None,
        visibility: str | None = None,
        session_id: str | None = None,
        top_k: int = 50,
    ) -> list[MemoryChunk]:
        body: dict[str, Any] = {
            "type": "episodic",
            "query": query,
            "user_id": user_id,
            "top_k": top_k,
        }
        if visibility is not None:
            body["visibility"] = visibility
        if session_id is not None:
            body["session_id"] = session_id
        return self._search(body)

    def search_taxonomic(
        self,
        *,
        query: str,
        user_id: str | None = None,
        domain: str | None = None,
        visibility: str | None = None,
        top_k: int = 50,
    ) -> list[MemoryChunk]:
        """Search the knowledge base by relevance to ``query``.

        Routes through the shared ``/search`` dispatch with ``type=taxonomic``,
        the same ranked vector search the other long-term types use, optionally
        filtered by ``domain``. Modes that do not support taxonomic search raise
        before reaching the route.
        """
        if self._profile.taxonomic_unsupported_reason is not None:
            raise MemoryNotSupportedError(self._profile.taxonomic_unsupported_reason)

        body: dict[str, Any] = {
            "type": "taxonomic",
            "query": query,
            "user_id": user_id,
            "top_k": top_k,
        }
        if visibility is not None:
            body["visibility"] = visibility
        if domain is not None:
            body["domain"] = domain
        return self._search(body)

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
        if self._profile.ranked_search_unsupported_reason is not None:
            raise MemoryNotSupportedError(
                self._profile.ranked_search_unsupported_reason
            )
        body: dict[str, Any] = {
            "type": "procedural",
            "query": query,
            "user_id": user_id,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
        }
        if visibility is not None:
            body["visibility"] = visibility
        if tags is not None:
            body["tags"] = tags
        if metadata_filter is not None:
            body["metadata_filter"] = metadata_filter
        resp = self._post(_SEARCH_SUFFIX, body)
        results: list[dict[str, Any]] = []
        for m in self._transport.memories(resp):
            raw_meta = m.get("metadata")
            meta: dict[str, Any] = (
                cast("dict[str, Any]", raw_meta) if isinstance(raw_meta, dict) else {}
            )
            results.append(
                {
                    **meta,
                    "id": m.get("id"),
                    "timestamp": (
                        m.get("timestamp") or m.get("created_at") or m.get("updated_at")
                    ),
                    "procedure": meta.get("procedure", ""),
                    "content": m.get("content", ""),
                    "score": m.get("similarity_score", 0.0),
                }
            )
        return results


__all__ = [
    "_HttpMemoryRuntime",
    "_EndpointProfile",
    "_GATEWAY_PROFILE",
    "_OE_PROFILE",
]
