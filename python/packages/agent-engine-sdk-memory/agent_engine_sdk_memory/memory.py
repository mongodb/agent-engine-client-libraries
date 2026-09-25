"""Transport-free public Memory facade.

`Memory` is the surface platform users interact with. It delegates the six
high-level workflow operations to an injected `MemoryRuntime` and the CRUD
conveniences to an injected `MemoryCrudClient`. Identity is resolved per call
via `resolve_identity`; tenancy never appears in any public signature.
"""

from __future__ import annotations

import contextlib
import os
import uuid
import warnings
from collections.abc import Generator, Mapping, Sequence
from types import TracebackType
from typing import Any, cast, overload

from agent_engine_sdk_memory._direct_crud import _EmptyTenancyCrudClient
from agent_engine_sdk_memory._http_runtime import (
    _GATEWAY_PROFILE,
    _OE_PROFILE,
    _EndpointProfile,
    _HttpMemoryRuntime,
)
from agent_engine_sdk_memory._tag_syntax import (
    validate_memory_type,
    validate_tag_syntax,
)
from agent_engine_sdk_memory.errors import (
    MemoryBadRequestError,
    MemoryNotSupportedError,
)
from agent_engine_sdk_memory.identity import resolve_identity
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
    SearchSource,
    SourceSpec,
    WriteTurnResult,
)
from agent_engine_sdk_memory.protocol import (
    AmbientIdentityRuntime,
    MemoryCrudClient,
    MemoryRequestContext,
    MemoryRuntime,
)
from agent_engine_sdk_memory.validation import (
    require_positive_max_tokens,
    require_session_for_stm,
    require_valid_format_style,
    require_valid_sources,
)

__all__ = ["Memory"]

_CUSTOM_TYPES_UNSUPPORTED_MSG = (
    "this platform version does not support custom memory types; "
    "save/retrieve requires a platform release with the custom memory type "
    "operations enabled"
)

_CUSTOM_TYPES_DISABLED_MSG = (
    "custom memory types are disabled on this deployment; "
    "save/retrieve requires the custom_memory_types feature to be enabled"
)

# The gateway's flag-off rejection reuses the generic INVALID_REQUEST code, so
# this message substring is the only stable discriminator for that 400.
_FLAG_OFF_MARKER = "custom_memory_types is not enabled"


def _map_custom_route_error(exc: MemoryBadRequestError) -> Exception:
    """Distinguish an absent route (old backend / feature flag off) from the
    server's structured unknown-type 404, which names the type in its body."""
    if exc.status in (404, 405) and "unknown custom memory type" not in str(exc):
        return MemoryNotSupportedError(_CUSTOM_TYPES_UNSUPPORTED_MSG)
    if exc.status == 400 and _FLAG_OFF_MARKER in str(exc):
        return MemoryNotSupportedError(_CUSTOM_TYPES_DISABLED_MSG)
    return exc


@contextlib.contextmanager
def _custom_route_errors() -> Generator[None, None, None]:
    """Remap custom-route rejections via ``_map_custom_route_error``."""
    try:
        yield
    except MemoryBadRequestError as exc:
        mapped = _map_custom_route_error(exc)
        if mapped is exc:
            # Re-raise untouched: chaining an exception to itself sets a
            # self-referential __cause__ and suppresses __context__.
            raise
        raise mapped from exc


_DEFAULT_GATEWAY_URL = "https://agentengine.mongodb.com"
_SERVICE_ACCOUNT_TOKEN_ENV_VAR = "AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN"
# Deprecated alias for _SERVICE_ACCOUNT_TOKEN_ENV_VAR, honored with a
# DeprecationWarning so existing deployments keep working.
_API_KEY_ENV_VAR = "AGENTIC_MEMORY_API_KEY"
_BASE_URL_ENV_VAR = "AGENTIC_MEMORY_BASE_URL"
_PROJECT_ID_ENV_VAR = "AGENTIC_MEMORY_PROJECT_ID"

# Default sources for the zero-argument search(): the two ranked similarity
# sources (taxonomic is a query-ignoring GET list).
_DEFAULT_SEARCH_SOURCES = (SearchSource.SEMANTIC, SearchSource.EPISODIC)


def _env_or_unset(name: str) -> str | None:
    """Return the env var's value, or ``None`` when unset or blank.

    Matches the ``base_url``/``project_id`` convention: an empty-exported
    placeholder (common in CI, Helm, and dotenv files) is not a provided
    credential, so it must neither authenticate nor count as a second input.
    """
    value = os.environ.get(name)
    return value if value is not None and value.strip() else None


def _normalize_sources(
    sources: SearchSource | str | Sequence[SearchSource | str] | None,
) -> list[SearchSource]:
    """Resolve requested sources to a deduped, order-preserving list.

    ``None`` selects the universally-supported default. A single source (a
    ``SearchSource`` or its string value) is accepted as well as a sequence of
    them; an unknown source raises ``ValueError``. A bare string is treated as
    one source, not iterated character by character.
    """
    if sources is None:
        return list(_DEFAULT_SEARCH_SOURCES)
    if isinstance(sources, (str, SearchSource)):
        sources = [sources]
    resolved: list[SearchSource] = []
    for source in sources:
        member = SearchSource(source)
        if member not in resolved:
            resolved.append(member)
    return resolved


def _procedure_to_chunk(proc: dict[str, Any]) -> MemoryChunk:
    """Adapt a ``discover_procedures`` dict into a ``MemoryChunk``.

    Procedural discovery returns loose dicts (no typed model yet), so ``search``
    presents them in the unified list by mapping content/description/name to the
    chunk content and preserving the whole dict under ``metadata``. The source
    ``timestamp`` is carried through when present, falling back to the epoch
    (not query-time ``now``) so an absent timestamp does not masquerade as a
    fresh result and distort recency.
    """
    raw_id = proc.get("id") or proc.get("_id") or ""
    content = (
        proc.get("content") or proc.get("description") or proc.get("procedure") or ""
    )
    score = proc.get("score")
    if score is None:
        score = proc.get("similarity_score")
    raw_ts = proc.get("timestamp") or proc.get("created_at") or proc.get("updated_at")
    return MemoryChunk.model_validate(
        {
            "id": str(raw_id),
            "content": str(content),
            "source": "procedural",
            "timestamp": raw_ts or "1970-01-01T00:00:00Z",
            "similarity_score": score,
            "metadata": proc,
        }
    )


class Memory:
    """Public, transport-free memory facade."""

    @overload
    def __init__(
        self,
        *,
        service_account_token: str | None = ...,
        api_key: str | None = ...,
        base_url: str | None = ...,
        project_id: str | None = ...,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        runtime: MemoryRuntime,
        client: MemoryCrudClient | None = None,
        _bound_ctx: MemoryRequestContext | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        runtime: MemoryRuntime | None = None,
        client: MemoryCrudClient | None = None,
        _bound_ctx: MemoryRequestContext | None = None,
        service_account_token: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        project_id: str | None = None,
    ) -> None:
        """Construct a Memory facade.

        Pass the inputs directly. Each one also falls back to an
        ``AGENTIC_MEMORY_*`` environment variable when the argument is omitted,
        so an explicit argument always wins over the environment.

        * ``service_account_token`` (``AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN``)
          is auth — a service-account access token minted from a service
          account's client ID and secret at ``POST /api/v1/oauth/token`` (the
          README's hosted guide walks through it). It becomes a bearer token
          when set, and there is no auth header when it is absent.
        * ``base_url`` (``AGENTIC_MEMORY_BASE_URL``) is the backend address.
        * ``project_id`` (``AGENTIC_MEMORY_PROJECT_ID``) selects the route shape.
          Set it to use project-scoped routes
          (``/api/v1/projects/{id}/memory/*``). Leave it empty to use flat routes
          (``/api/v1/memory/*``). Auth does not select the route shape, so an
          authenticated caller with no ``project_id`` still uses flat routes.

        ``api_key`` (``AGENTIC_MEMORY_API_KEY``) is a deprecated alias for
        ``service_account_token``: still accepted, emits a
        ``DeprecationWarning``. Setting both the new and the legacy input — in
        any mix of argument and environment variable — raises ``ValueError``.
        A blank environment variable counts as unset (matching ``base_url`` and
        ``project_id``), so an empty-exported placeholder never conflicts with
        an explicit argument. A blank argument still raises ``ValueError``.

        ``base_url`` resolves from the argument, then ``AGENTIC_MEMORY_BASE_URL``,
        then the hosted default when an auth token is set. With neither a
        resolvable URL nor an auth token, construction raises ``ValueError``.

        Type-specific CRUD is available on hosted (Gateway/``project_id`` set) and
        direct (OE/flat) HTTP modes. Per-backend gaps (including app-bound) are
        documented in ``docs/capability-matrix.md`` and raise
        ``MemoryNotSupportedError`` when hit. Application code does not construct
        app-bound mode; the platform supplies ``runtime`` / ``client`` injection
        instead.
        """
        if runtime is not None:
            self._runtime = runtime
            self._client = client
            self._bound_ctx = _bound_ctx
            self._owns_runtime = False
            self._crud_unsupported_reason = None
            return

        token = (
            service_account_token
            if service_account_token is not None
            else _env_or_unset(_SERVICE_ACCOUNT_TOKEN_ENV_VAR)
        )
        legacy_token = (
            api_key if api_key is not None else _env_or_unset(_API_KEY_ENV_VAR)
        )
        if token is not None and legacy_token is not None:
            raise ValueError(
                "pass only one of service_account_token "
                f"({_SERVICE_ACCOUNT_TOKEN_ENV_VAR}) and the deprecated api_key "
                f"({_API_KEY_ENV_VAR})"
            )
        auth_token = token if token is not None else legacy_token
        if auth_token is not None and not auth_token.strip():
            name = "service_account_token" if token is not None else "api_key"
            raise ValueError(f"{name} must be a non-empty string")
        # Validation precedes the warning so a rejected value never also warns.
        if legacy_token is not None:
            warnings.warn(
                "api_key (AGENTIC_MEMORY_API_KEY) is deprecated and will be "
                "removed in a future release; use service_account_token "
                "(AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN) instead",
                DeprecationWarning,
                stacklevel=2,
            )
        project_id = (
            project_id
            if project_id is not None
            else os.environ.get(_PROJECT_ID_ENV_VAR)
        )
        # Normalize project_id. A blank value (env left as "") is treated as
        # unset, since the discriminator is presence and an empty path segment
        # would misroute. A surrounding-whitespace value is stripped so it does
        # not end up percent-encoded into the route (" p1 " -> "%20p1%20").
        if project_id is not None:
            project_id = project_id.strip() or None

        resolved_url = self._resolve_url(base_url, auth_token=auth_token)
        if resolved_url is None:
            raise ValueError(
                "Memory requires base_url, the "
                f"{_BASE_URL_ENV_VAR} environment variable, or an auth token "
                "(service_account_token; deprecated alias api_key), which "
                f"falls back to the hosted default {_DEFAULT_GATEWAY_URL}"
            )

        # project_id presence, not auth, picks the route shape.
        profile = _GATEWAY_PROFILE if project_id else _OE_PROFILE
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else None
        self._init_http_runtime(
            base_url=resolved_url,
            profile=profile,
            project_id=project_id,
            headers=headers,
        )

    def _init_http_runtime(
        self,
        *,
        base_url: str,
        profile: _EndpointProfile,
        project_id: str | None,
        headers: dict[str, str] | None,
    ) -> None:
        """Build the self-owned HTTP runtime and its empty-tenancy CRUD adapter.

        ``project_id`` fills the project-scoped Gateway base path; it is ignored
        by the flat OE profile. The CRUD adapter rides the runtime's transport
        and base path, so workflow operations and CRUD share one connection
        pool, retry loop, typed-error mapping, and route map. A profile that
        does not support CRUD gets no adapter; ``_require_client`` then raises
        ``MemoryNotSupportedError`` with the profile's reason.
        """
        self._runtime = _HttpMemoryRuntime(
            base_url=base_url,
            profile=profile,
            project_id=project_id,
            headers=headers,
        )
        self._crud_unsupported_reason = profile.crud_unsupported_reason
        self._client = (
            None
            if profile.crud_unsupported_reason is not None
            else _EmptyTenancyCrudClient(
                self._runtime.transport, self._runtime.base_path
            )
        )
        self._bound_ctx = None
        self._owns_runtime = True

    @staticmethod
    def _resolve_url(base_url: str | None, *, auth_token: str | None) -> str | None:
        """Resolve the backend URL: the ``base_url`` argument, then the env var,
        then the hosted Gateway default when an auth token is present.

        Returns ``None`` when nothing resolves (no argument, no env var, no
        auth token), which the constructor turns into a ``ValueError``. A
        ``base_url`` passed explicitly but blank is an error, mirroring the
        auth-token non-blank check. A blank env var is treated as unset. The
        auth-token default lets an authenticated caller construct with no URL and
        reach the hosted Gateway. Auth does not pick the route shape.
        ``project_id`` does.
        """
        if base_url is not None and not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        resolved_url = base_url or os.environ.get(_BASE_URL_ENV_VAR)
        if resolved_url and resolved_url.strip():
            return resolved_url.strip()
        if auth_token:
            return _DEFAULT_GATEWAY_URL
        return None

    def bind(self, ctx: MemoryRequestContext) -> Memory:
        """Return a new Memory scoped to ``ctx`` without mutating this handle.

        The bound handle shares this handle's transport; closing either one
        closes the shared transport. Ownership propagates, so a bound handle of
        a self-constructed (api-key) Memory still releases the transport on
        close(), while a bound handle of an injected runtime leaves it to its
        owner.
        """
        bound = Memory(runtime=self._runtime, client=self._client, _bound_ctx=ctx)
        bound._owns_runtime = self._owns_runtime
        bound._crud_unsupported_reason = self._crud_unsupported_reason
        return bound

    def close(self) -> None:
        """Release the underlying transport(s) when this handle owns them.

        A runtime this Memory constructed — and, in direct mode, the CRUD
        adapter alongside it — is closed; an injected or bound runtime belongs
        to its provider and is left untouched.
        """
        if not self._owns_runtime:
            return
        close = getattr(self._runtime, "close", None)
        if callable(close):
            close()
        # _EmptyTenancyCrudClient.close() is a no-op (the adapter shares the
        # runtime's transport and does not own it); an injected client is left
        # to its provider.
        client_close = getattr(self._client, "close", None)
        if callable(client_close):
            client_close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _require_client(self) -> MemoryCrudClient:
        if self._client is None:
            raise MemoryNotSupportedError(
                self._crud_unsupported_reason
                or (
                    "type-specific save/get/list operations require a CRUD client; "
                    "provide one via the client= constructor arg when injecting a "
                    "custom runtime, or use record_turn, build_context, or search"
                )
            )
        return self._client

    def _runtime_ctx(self) -> MemoryRequestContext | None:
        # Only runtimes that declare the AmbientIdentityRuntime capability expose
        # per-call identity via request_context(); for all others (api-key, direct)
        # identity comes only from call args and bind(). Read fresh on every
        # resolve so a long-lived facade picks up context that varies per request.
        if not isinstance(self._runtime, AmbientIdentityRuntime):
            return None
        return self._runtime.request_context()

    def _resolve(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        required: tuple[str, ...] = (),
        visibility: str | None = None,
        suppress_inherited_session_id: bool = False,
    ) -> dict[str, str | None]:
        # This method maps an operation's intent onto the two resolve_identity
        # guards (see that docstring for why each exists).
        #
        # A broad visibility ("org") suppresses the runtime user_id, so a
        # cross-user read does not borrow the ambient principal. "private" keeps
        # it, since that is the user's own scope. Writes pass no visibility, so
        # their user_id resolves unchanged.
        #
        # Episodic reads set suppress_inherited_session_id, so a bound or ambient
        # session never filters memory that was stored without one. Callers set
        # it at search_episodes, list_episodes, and the episodic leg of search().
        # Writes and build_context leave it off and keep session inheritance.
        return resolve_identity(
            call_args={
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id,
            },
            bind_ctx=self._bound_ctx,
            runtime_ctx=self._runtime_ctx(),
            required=required,
            suppress_runtime_user_id=visibility is not None and visibility != "private",
            suppress_inherited_session_id=suppress_inherited_session_id,
        )

    # =========================================================================
    # Conversation turns
    # =========================================================================

    def record_turn(
        self,
        *,
        role: str,
        content: str | None = None,
        session_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        is_error: bool = False,
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> WriteTurnResult:
        """Record a single conversation turn (write-accepted semantics).

        Args:
            metadata: Arbitrary key-value metadata persisted on the turn and
                returned on retrieval in its own ``chunk.metadata["metadata"]``
                slot, kept apart from the platform's turn fields, so any key is
                safe to use. Matchable by the
                retrieval ``metadata_filter``, where the key is the qualified
                index path (``metadata.<name>``) because the filter addresses
                the stored document. Omitting it preserves prior behavior.
            idempotency_key: Deduplication key for the write. Auto-generated
                per call when omitted, which protects transport-level retries
                beneath this call only; callers that retry ``record_turn``
                itself must supply their own key for cross-call dedupe.

        Note:
            In app-bound mode the returned ``WriteTurnResult.id`` is ``""`` and
            ``turn_seq`` is ``0`` — the platform runtime fires the write
            asynchronously and returns no per-turn identifier or sequence number.
        """
        resolved = self._resolve(session_id=session_id)
        if idempotency_key is None:
            idempotency_key = str(uuid.uuid4())
        return self._runtime.record_turn(
            role=role,
            content=content,
            session_id=resolved["session_id"],
            user_id=resolved["user_id"],
            agent_id=resolved["agent_id"],
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            is_error=is_error,
            model_name=model_name,
            metadata=metadata,
            idempotency_key=idempotency_key,
        )

    # =========================================================================
    # Semantic memory
    # =========================================================================

    def save_semantic(
        self,
        *,
        text: str,
        label: str,
        user_id: str | None = None,
        source: str = "agent",
        visibility: str = "private",
        metadata: dict[str, Any] | None = None,
        upsert: bool = True,
        agent_id: str | None = None,
    ) -> CreateSemanticResult:
        """Save a semantic memory (fact, customer profile).

        Args:
            visibility: Access scope of the memory ("private" or "org").
            metadata: Optional caller-supplied metadata dictionary stored with
                the memory.
            upsert: When True (the default), replace an existing memory with
                the same label instead of creating a duplicate.

        Raises:
            MemoryIdentityError: If ``user_id`` is not supplied and cannot be
                resolved from the bound context.
            MemoryNotSupportedError: If this handle has no CRUD client configured.
        """
        resolved = self._resolve(
            user_id=user_id, agent_id=agent_id, required=("user_id",)
        )
        return self._require_client().create_semantic(
            label=label,
            text=text,
            user_id=cast(str, resolved["user_id"]),
            agent_id=resolved["agent_id"],
            source=source,
            visibility=visibility,
            metadata=metadata,
            upsert=upsert,
        )

    def search_semantic(
        self,
        query: str,
        user_id: str | None = None,
        visibility: str | None = None,
        top_k: int = 50,
    ) -> list[MemoryChunk]:
        """Search semantic memories.

        Args:
            visibility: When supplied, the ambient (runtime-context) principal
                is not auto-substituted for ``user_id``; the caller must supply
                one explicitly if a user-scoped filter is needed.
        """
        resolved = self._resolve(user_id=user_id, visibility=visibility)
        return self._runtime.search_semantic(
            query=query,
            user_id=resolved["user_id"],
            visibility=visibility,
            top_k=top_k,
        )

    def get_semantic(
        self,
        label: str,
        user_id: str | None = None,
        visibility: str | None = None,
    ) -> Any | None:
        """Get a semantic memory by label.

        Raises:
            MemoryNotSupportedError: If this handle has no CRUD client configured.
        """
        resolved = self._resolve(user_id=user_id, visibility=visibility)
        return self._require_client().get_semantic(
            label=label,
            user_id=resolved["user_id"],
            visibility=visibility,
        )

    # =========================================================================
    # Episodic memory
    # =========================================================================

    def save_episode(
        self,
        *,
        title: str,
        content: str,
        user_id: str | None = None,
        summary: str | None = None,
        session_id: str | None = None,
        participants: list[str] | None = None,
        tags: list[str] | None = None,
        visibility: str = "private",
        metadata: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> CreateEpisodicResult:
        """Save an episodic memory (conversation summary).

        Args:
            visibility: Access scope of the memory ("private" or "org").
            metadata: Optional caller-supplied metadata dictionary stored with
                the episode.

        Raises:
            MemoryIdentityError: If ``user_id`` or ``session_id`` is not
                supplied and cannot be resolved from the bound context.
            MemoryNotSupportedError: If this handle has no CRUD client configured.
        """
        resolved = self._resolve(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            required=("user_id", "session_id"),
        )
        return self._require_client().create_episodic(
            title=title,
            content=content,
            summary_text=summary,
            user_id=cast(str, resolved["user_id"]),
            agent_id=resolved["agent_id"],
            session_id=cast(str, resolved["session_id"]),
            participants=participants,
            tags=tags,
            visibility=visibility,
            metadata=metadata,
        )

    def search_episodes(
        self,
        query: str,
        user_id: str | None = None,
        visibility: str | None = None,
        session_id: str | None = None,
        top_k: int = 50,
    ) -> list[MemoryChunk]:
        """Search past episodes.

        Args:
            visibility: When supplied, the ambient (runtime-context) principal
                is not auto-substituted for ``user_id``; the caller must supply
                one explicitly if a user-scoped filter is needed.
            session_id: Episodic memory is stored without a session, so a bound
                or ambient ``session_id`` is not inherited here. Pass one
                explicitly to filter to a single conversation.
        """
        resolved = self._resolve(
            user_id=user_id,
            session_id=session_id,
            visibility=visibility,
            suppress_inherited_session_id=True,
        )
        return self._runtime.search_episodes(
            query=query,
            user_id=resolved["user_id"],
            visibility=visibility,
            session_id=resolved["session_id"],
            top_k=top_k,
        )

    def list_episodes(
        self,
        user_id: str | None = None,
        visibility: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[Any]:
        """List episodic memories with optional filters.

        Raises:
            MemoryNotSupportedError: If this handle has no CRUD client configured.
        """
        resolved = self._resolve(
            user_id=user_id,
            session_id=session_id,
            visibility=visibility,
            suppress_inherited_session_id=True,
        )
        return self._require_client().list_episodic(
            user_id=resolved["user_id"],
            visibility=visibility,
            session_id=resolved["session_id"],
            limit=limit,
        )

    # =========================================================================
    # Taxonomic memory
    # =========================================================================

    def save_taxonomic(
        self,
        *,
        domain: str,
        term: str,
        definition: str,
        related_terms: list[str] | None = None,
        visibility: str = "org",
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CreateTaxonomicResult:
        """Save a taxonomic term (knowledge-base entry).

        Args:
            visibility: Access scope of the entry ("org" by default so the
                term is shared knowledge; "private" restricts it to the user).
            metadata: Arbitrary key-value metadata persisted on the entry and
                returned on retrieval; matchable by the retrieval
                ``metadata_filter``. Omitting it preserves prior behavior.

        Raises:
            MemoryIdentityError: If ``user_id`` is not supplied and cannot be
                resolved from the bound context.
            MemoryNotSupportedError: If this handle has no CRUD client configured.
        """
        resolved = self._resolve(user_id=user_id, required=("user_id",))
        return self._require_client().create_taxonomic(
            domain=domain,
            term=term,
            definition=definition,
            related_terms=related_terms,
            visibility=visibility,
            user_id=cast(str, resolved["user_id"]),
            metadata=metadata,
        )

    def search_taxonomic(
        self,
        query: str,
        user_id: str | None = None,
        domain: str | None = None,
        visibility: str | None = None,
        top_k: int = 50,
    ) -> list[MemoryChunk]:
        """Search the knowledge base.

        Args:
            visibility: When supplied, the ambient (runtime-context) principal
                is not auto-substituted for ``user_id``; the caller must supply
                one explicitly if a user-scoped filter is needed.
        """
        resolved = self._resolve(user_id=user_id, visibility=visibility)
        return self._runtime.search_taxonomic(
            query=query,
            user_id=resolved["user_id"],
            domain=domain,
            visibility=visibility,
            top_k=top_k,
        )

    def get_taxonomic_term(
        self,
        domain: str,
        term: str,
        user_id: str | None = None,
        visibility: str | None = None,
    ) -> Any | None:
        """Get a specific knowledge-base term.

        Raises:
            MemoryNotSupportedError: If this handle has no CRUD client configured.
        """
        resolved = self._resolve(user_id=user_id, visibility=visibility)
        return self._require_client().get_taxonomic(
            domain=domain,
            term=term,
            user_id=resolved["user_id"],
            visibility=visibility,
        )

    def list_domains(self, visibility: str | None = None) -> list[str]:
        """List all knowledge-base domains.

        Raises:
            MemoryNotSupportedError: If this handle has no CRUD client configured.
        """
        return self._require_client().get_distinct_domains(visibility=visibility)

    # =========================================================================
    # Procedural memory
    # =========================================================================

    def discover_procedures(
        self,
        query: str,
        user_id: str | None = None,
        *,
        visibility: str | None = None,
        tags: list[str] | None = None,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Discover procedures matching a query without loading full content.

        Args:
            metadata_filter: Field-level filter on procedural metadata. Keys are
                fully qualified index paths, so an attribute declared as ``tier``
                in the project's metadata partition configuration is filtered
                as ``{"metadata.tier": "gold"}``; nothing is prefixed for you.
                Keys here are not validated against the declared set -- an
                undeclared or misspelled path is not rejected and silently
                yields no or degraded matches.
        """
        resolved = self._resolve(user_id=user_id, visibility=visibility)
        return self._runtime.discover_procedures(
            query=query,
            user_id=resolved["user_id"],
            visibility=visibility,
            tags=tags,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            metadata_filter=metadata_filter,
        )

    def get_procedure(
        self,
        procedure_name: str,
        *,
        user_id: str | None = None,
        visibility: str | None = None,
        include_deleted: bool = False,
    ) -> Any | None:
        """Get a full procedural memory by procedure name.

        Raises:
            MemoryNotSupportedError: If this handle has no CRUD client configured.
        """
        resolved = self._resolve(user_id=user_id, visibility=visibility)
        return self._require_client().get_procedural(
            procedure=procedure_name,
            user_id=resolved["user_id"],
            visibility=visibility,
            include_deleted=include_deleted,
        )

    def save_procedure(
        self,
        *,
        procedure: str,
        description: str,
        content: str,
        user_id: str | None = None,
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
        """Create or update a procedural memory.

        Args:
            visibility: Access scope of the procedure ("private" or "org").
            metadata: Arbitrary key-value metadata persisted on the procedure
                and returned on retrieval; matchable by the retrieval
                ``metadata_filter``. Applies to both create and (in app-bound
                mode) ``update_existing``. Omitting it preserves prior behavior.
            update_existing: When True, update the procedure with the same name
                instead of creating a duplicate. Honored only in app-bound mode;
                the HTTP backends serve a create-only route, so Gateway and OE
                modes raise ``MemoryNotSupportedError``.

        Raises:
            MemoryIdentityError: If ``user_id`` is not supplied and cannot be
                resolved from the bound context.
            MemoryNotSupportedError: If this handle has no CRUD client configured,
                or if ``update_existing`` is True against an HTTP backend.
        """
        resolved = self._resolve(
            user_id=user_id, agent_id=agent_id, required=("user_id",)
        )
        return self._require_client().create_procedural(
            procedure=procedure,
            description=description,
            content=content,
            user_id=cast(str, resolved["user_id"]),
            agent_id=resolved["agent_id"],
            steps=steps,
            resources=resources,
            allowed_tools=allowed_tools,
            compatibility=compatibility,
            license=license,
            trigger_conditions=trigger_conditions,
            tags=tags,
            visibility=visibility,
            extraction_source=extraction_source,
            source_format=source_format,
            source_path=source_path,
            metadata=metadata,
            update_existing=update_existing,
        )

    # =========================================================================
    # Custom memory types
    # =========================================================================

    def save(
        self,
        memory_type: str,
        content: str,
        *,
        tags: Mapping[str, Any] | None = None,
        contextual_metadata: dict[str, Any] | None = None,
    ) -> CustomMemorySaveResult:
        """Save a memory of a declared custom type.

        The platform stamps identity (org, project, user) and enforces the
        type's declared tag schema; this method validates only tag syntax
        client-side so mistakes fail fast with server-matching messages.

        Args:
            memory_type: A custom type declared in the project's memory
                configuration. Built-in types (semantic, episodic, taxonomic,
                procedural) are rejected — use their dedicated methods.
            content: The content to store and embed.
            tags: Declared tag keys with scalar values. Dotted keys
                ("profile.location") and one-level nested mappings are
                equivalent.
            contextual_metadata: Arbitrary additional context; stored as-is,
                never filterable.

        Raises:
            MemoryClientError: If the type name is built-in/empty or a tag
                fails syntax checks (a ``ValueError`` subclass).
            MemoryNotSupportedError: If this handle has no CRUD client, or the
                platform does not support custom memory types.
        """
        validate_memory_type(memory_type)
        if tags is not None:
            validate_tag_syntax(tags)
        client = self._require_client()
        with _custom_route_errors():
            return client.create_custom(
                memory_type=memory_type,
                content=content,
                tags=tags,
                contextual_metadata=contextual_metadata,
            )

    def retrieve(
        self,
        memory_type: str,
        query: str,
        *,
        tags: Mapping[str, Any] | None = None,
        top_k: int = 10,
    ) -> CustomMemoryRetrieveResult:
        """Retrieve memories of a declared custom type by semantic query.

        Filters are exact-match equality on declared tag keys; result ordering
        may improve between releases and is not contractual.

        Args:
            memory_type: A custom type declared in the project's memory
                configuration; one type per call.
            query: Query text for semantic search.
            tags: Equality filters on declared tag keys.
            top_k: Maximum number of results (server enforces 1-100).

        Raises:
            MemoryClientError: If the type name is built-in/empty or a tag
                fails syntax checks (a ``ValueError`` subclass).
            MemoryNotSupportedError: If this handle has no CRUD client, or the
                platform does not support custom memory types.
        """
        validate_memory_type(memory_type)
        if tags is not None:
            validate_tag_syntax(tags)
        client = self._require_client()
        with _custom_route_errors():
            return client.retrieve_custom(
                memory_type=memory_type, query=query, tags=tags, top_k=top_k
            )

    # =========================================================================
    # Context building
    # =========================================================================

    def build_context(
        self,
        query: str,
        user_id: str | None = None,
        session_id: str | None = None,
        visibility: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        enabled_sources: set[str] | None = None,
        *,
        thread_id: str | None = None,
        max_tokens: int | None = None,
        format_style: FormatStyle | str | None = None,
        include_memories: bool = False,
    ) -> ContextResponse:
        """Build a unified memory context across memory types.

        Omitted ``enabled_sources`` defaults to ``episodic`` and
        ``semantic``; pass ``{"stm", "episodic", "semantic"}`` to include
        recent ``record_turn`` content.

        ``thread_id`` is a deprecated alias for ``session_id``; when both are
        provided, a non-blank ``session_id`` wins.

        Args:
            enabled_sources: Memory sources to include. ``None`` uses the
                server default (episodic, semantic).
            metadata_filter: One field-level filter applied to every enabled
                source. Keys are fully qualified index paths, so an attribute
                declared as ``tier`` in the project's metadata partition
                configuration is filtered as ``{"metadata.tier": "gold"}``;
                nothing is prefixed for you. Unlike
                ``build_context_from_sources``, these keys are *not* validated
                against the declared set -- an undeclared or misspelled path is
                not rejected and silently yields no or degraded matches. Use
                ``build_context_from_sources`` when you want a bad key to fail
                loudly.
            visibility: When supplied, the ambient (runtime-context) principal
                is not auto-substituted for ``user_id``; the caller must supply
                one explicitly if a user-scoped filter is needed.
            max_tokens: Optional gross context-construction budget. Omitted
                (``None``) preserves prior behavior. Non-positive, non-integer,
                non-finite, and bool values raise ``ValueError`` before runtime
                delegation. After retrieval and ranking, the server subtracts a
                500-token formatting reserve, then greedily selects whole memory
                chunks that fit in the remainder. Positive values at or below 500
                leave no budget for memories. Values above 500 can still
                yield empty context when no chunk fits. ``metadata.token_count``
                reports formatted output only and excludes the reserve.
            format_style: Output format for the assembled context:
                ``"openai"`` (a chat-message list that keeps STM turn roles),
                ``"claude"`` (an XML string), or ``"jinja2"`` (a markdown
                string). Unknown values raise ``ValueError`` before any request
                is made. Omitted (``None``), the server infers the format from
                its configured model. Caveat: the server currently re-infers
                from its configured model when the explicit value matches its
                default model type, so an explicit ``"openai"`` is honored
                verbatim only on servers with an OpenAI-family model
                configured; ``"claude"`` and ``"jinja2"`` are always honored.
            include_memories: When ``True``, the response's
                ``selected_memories`` carries the post-budget ``MemoryChunk``
                list alongside ``formatted_context``, so callers can inspect
                exactly which memories were selected.
        """
        require_positive_max_tokens(max_tokens)
        require_valid_format_style(format_style)
        resolved = self._resolve(
            user_id=user_id,
            # A blank session_id is "unset" (identity resolution discards it),
            # so it must not suppress the thread_id alias either.
            session_id=session_id if session_id and session_id.strip() else thread_id,
            visibility=visibility,
        )
        # Omit unset optional kwargs from the runtime call so injected runtimes
        # that inspect key presence stay compatible with callers written before
        # these parameters existed.
        optional: dict[str, Any] = {}
        if max_tokens is not None:
            optional["max_tokens"] = max_tokens
        if format_style is not None:
            optional["format_style"] = format_style
        if include_memories:
            optional["include_memories"] = True
        return self._runtime.build_context(
            query=query,
            user_id=resolved["user_id"],
            session_id=resolved["session_id"],
            visibility=visibility,
            metadata_filter=metadata_filter,
            enabled_sources=enabled_sources,
            **optional,
        )

    def build_context_from_sources(
        self,
        query: str,
        sources: Sequence[SourceSpec],
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        visibility: str | None = None,
        rerank: bool = False,
        thread_id: str | None = None,
        max_tokens: int | None = None,
        format_style: FormatStyle | str | None = None,
        include_memories: bool = False,
    ) -> ContextResponse:
        """Build a memory context from an explicit, per-source-configured source set.

        Unlike :meth:`build_context`, each source in ``sources`` declares its own
        retrieval mode, metadata filter, and candidate count. Results are merged
        and de-duplicated across sources, optionally reordered by a relevance
        rerank, then formatted and budgeted like :meth:`build_context`.
        ``metadata.ranking_strategy`` and ``metadata.source_outcomes`` report how
        the final order was produced and how each source fared.

        ``thread_id`` is a deprecated alias for ``session_id``; when both are
        provided, a non-blank ``session_id`` wins. ``session_id`` is only needed
        when the ``stm`` source is requested.

        Args:
            sources: One :class:`SourceSpec` per memory source to search. Must be
                non-empty, and each source may appear at most once.
            rerank: When ``True``, reorder merged results by a relevance rerank
                if the service is available.
            max_tokens: Optional gross context-construction budget; see
                :meth:`build_context`.
            format_style: Output format for the assembled context; see
                :meth:`build_context`.
            include_memories: When ``True``, the response's
                ``selected_memories`` carries the post-budget ``MemoryChunk``
                list; see :meth:`build_context`.

        Raises:
            ValueError: If ``sources`` is empty or lists a source more than once,
                or if ``max_tokens`` is non-positive.
            MemoryIdentityError: If a ``stm`` source is requested but no non-blank
                ``session_id`` resolves from the argument or execution context.
        """
        require_positive_max_tokens(max_tokens)
        require_valid_format_style(format_style)
        specs = require_valid_sources(sources)
        resolved = self._resolve(
            user_id=user_id,
            # A blank session_id is "unset" (identity resolution discards it), so
            # it must not suppress the thread_id alias either.
            session_id=session_id if session_id and session_id.strip() else thread_id,
            visibility=visibility,
        )
        require_session_for_stm(specs, resolved["session_id"])
        # Same key-presence convention as build_context: forward the optional
        # response-shaping kwargs only when the caller set them.
        optional: dict[str, Any] = {}
        if format_style is not None:
            optional["format_style"] = format_style
        if include_memories:
            optional["include_memories"] = True
        return self._runtime.build_context_from_sources(
            query=query,
            sources=specs,
            user_id=resolved["user_id"],
            session_id=resolved["session_id"],
            visibility=visibility,
            rerank=rerank,
            max_tokens=max_tokens,
            **optional,
        )

    # =========================================================================
    # Unified search
    # =========================================================================

    def search(
        self,
        query: str,
        *,
        sources: SearchSource | str | Sequence[SearchSource | str] | None = None,
        top_k: int = 10,
        user_id: str | None = None,
        visibility: str | None = None,
        session_id: str | None = None,
        domain: str | None = None,
        tags: list[str] | None = None,
        similarity_threshold: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryChunk]:
        """Search one or more memory sources and return a single ranked list.

        ``sources`` defaults to ``[semantic, episodic]``, the two ranked
        similarity sources. It accepts a single source (``"semantic"`` or
        ``SearchSource.SEMANTIC``) or a sequence of them. Requesting a source the
        current backend does not serve raises ``MemoryNotSupportedError``; see
        ``docs/capability-matrix.md`` for current per-backend gaps.
        Per-source arguments are forwarded only to the sources that use them:
        ``session_id`` (episodic), ``domain`` (taxonomic), and ``tags`` /
        ``similarity_threshold`` / ``metadata_filter`` (procedural). Results are
        merged, sorted by ``similarity_score`` (descending, unscored last) and
        truncated to ``top_k``.

        ``metadata_filter`` keys are fully qualified index paths -- an attribute
        declared as ``tier`` is filtered as ``{"metadata.tier": "gold"}``, and
        nothing is prefixed for you. They are not validated against the declared
        set, so an undeclared or misspelled path is not rejected and silently
        yields no or degraded matches.

        Raises:
            MemoryNotSupportedError: If a requested source is unavailable in this
                mode. See ``docs/capability-matrix.md`` for current gaps.
        """
        # session_id feeds only the episodic leg below. The semantic, taxonomic,
        # and procedural legs ignore it. Setting suppress_inherited_session_id on
        # this one shared resolve is therefore safe for every leg, and it stops a
        # bound or ambient session from filtering episodic. An explicit
        # session_id= still filters.
        resolved = self._resolve(
            user_id=user_id,
            session_id=session_id,
            visibility=visibility,
            suppress_inherited_session_id=True,
        )
        rid = resolved["user_id"]
        rsession = resolved["session_id"]

        chunks: list[MemoryChunk] = []
        for source in _normalize_sources(sources):
            if source is SearchSource.SEMANTIC:
                chunks.extend(
                    self._runtime.search_semantic(
                        query=query, user_id=rid, visibility=visibility, top_k=top_k
                    )
                )
            elif source is SearchSource.EPISODIC:
                chunks.extend(
                    self._runtime.search_episodes(
                        query=query,
                        user_id=rid,
                        visibility=visibility,
                        session_id=rsession,
                        top_k=top_k,
                    )
                )
            elif source is SearchSource.TAXONOMIC:
                chunks.extend(
                    self._runtime.search_taxonomic(
                        query=query,
                        user_id=rid,
                        domain=domain,
                        visibility=visibility,
                        top_k=top_k,
                    )
                )
            elif source is SearchSource.PROCEDURAL:
                chunks.extend(
                    _procedure_to_chunk(proc)
                    for proc in self._runtime.discover_procedures(
                        query=query,
                        user_id=rid,
                        visibility=visibility,
                        tags=tags,
                        top_k=top_k,
                        similarity_threshold=similarity_threshold,
                        metadata_filter=metadata_filter,
                    )
                )

        chunks.sort(
            key=lambda c: (c.similarity_score is None, -(c.similarity_score or 0.0))
        )
        return chunks[:top_k]
