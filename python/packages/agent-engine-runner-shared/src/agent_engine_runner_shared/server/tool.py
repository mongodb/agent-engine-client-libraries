"""
Tool Executor server - Executes tool functions in isolated pods.

The Tool Executor is responsible for:
- Executing registered tool functions
- Running tools that need special permissions (network, secrets)
- Executing LLM calls routed from AER via SecureWrappedLLM
- Returning results to OE
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

from agent_engine_sdk.models import LLMStreamChunk, LLMToolSchema, Message, ToolCallChunk
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from agent_engine_runner_shared.error_reporting import _redact_text
from agent_engine_runner_shared.guardrails_evaluator import evaluate_guardrail_check
from agent_engine_runner_shared.hooks import (
    entrypoint_scope,
    get_llm_adapter_factory,
    get_named_llm,
    has_named_llms,
    register_llm,
    reset_llm_registry,
    snapshot_llm_registry,
)
from agent_engine_runner_shared.metrics import with_metrics
from agent_engine_runner_shared.models import (
    GuardrailCheckRequest,
    GuardrailCheckResponse,
    LLMPodInvokeRequest,
    LLMPodInvokeResponse,
    LLMPodStreamEvent,
    ToolPodExecuteRequest,
    ToolPodExecuteResponse,
    ToolsListResponse,
    merge_token_usage,
)
from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
from agent_engine_runner_shared.server.base import BaseServer
from agent_engine_runner_shared.server.chunk_types import (
    LLM_CREDENTIAL_REJECTED_ERROR_CODE,
)
from agent_engine_runner_shared.server.metadata import applied_metadata_env, merge_metadata_env
from agent_engine_runner_shared.server.oe_url import resolve_oe_url
from agent_engine_runner_shared.server.owner_url import resolve_owner_url
from agent_engine_runner_shared.span_kinds import OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind
from agent_engine_runner_shared.tool_api_error import (
    classify_tool_api_error,
    request_credential_values,
    request_named_credential_values,
)
from agent_engine_runner_shared.toolpod_handlers import BUILTIN_TOOL_NAMES
from agent_engine_runner_shared.utils import (
    LLM_BACKOFF_MULTIPLIER,
    LLM_INITIAL_BACKOFF,
    LLM_MAX_BACKOFF,
    LLM_MAX_RETRIES,
    format_llm_error,
    is_llm_credential_rejection,
    is_retryable_error,
    log_tool_request,
    normalize_tool_call_args,
    tenant_env_vars,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from starlette.datastructures import Headers

    from agent_engine_runner_shared.context import OwnerUrlFailureState
    from agent_engine_runner_shared.runtime import TenantRuntime


def _deserialize_messages(messages: list[Message]) -> list[Message]:
    """Return typed sdk-core Message objects validated by LLMPodInvokeRequest."""
    return list(messages)


def _coerce_stream_chunk(chunk: Any) -> LLMStreamChunk:
    """Normalize adapter or LangChain stream output into sdk-core's LLMStreamChunk."""
    if isinstance(chunk, LLMStreamChunk):
        return chunk

    if isinstance(chunk, dict):
        return LLMStreamChunk.model_validate(chunk)

    content = chunk.content if isinstance(getattr(chunk, "content", None), str) else None

    tool_calls: list[ToolCallChunk] = []
    for index, tool_call in enumerate(getattr(chunk, "tool_call_chunks", None) or []):
        if not isinstance(tool_call, dict):
            continue
        tool_calls.append(
            ToolCallChunk(
                id=tool_call.get("id"),
                name=tool_call.get("name"),
                args=normalize_tool_call_args(tool_call.get("args")),
                type=tool_call.get("type"),
                index=tool_call.get("index", index),
            )
        )

    if not tool_calls:
        for index, tool_call in enumerate(getattr(chunk, "tool_calls", None) or []):
            if not isinstance(tool_call, dict):
                continue
            tool_calls.append(
                ToolCallChunk(
                    id=tool_call.get("id"),
                    name=tool_call.get("name"),
                    args=normalize_tool_call_args(
                        tool_call.get("args", tool_call.get("arguments"))
                    ),
                    type=tool_call.get("type"),
                    index=tool_call.get("index", index),
                )
            )

    return LLMStreamChunk(
        content=content,
        tool_calls=tool_calls or None,
        usage=getattr(chunk, "usage", None),
        id=getattr(chunk, "id", None),
        name=getattr(chunk, "name", None),
        response_metadata=getattr(chunk, "response_metadata", None),
        additional_kwargs=getattr(chunk, "additional_kwargs", None),
    )


def _stream_chunk_has_payload(stream_chunk: LLMStreamChunk) -> bool:
    return bool(
        stream_chunk.content
        or stream_chunk.tool_calls
        or stream_chunk.id is not None
        or stream_chunk.name is not None
        or stream_chunk.response_metadata is not None
        or stream_chunk.additional_kwargs is not None
    )


# Bound on the entrypoint-derived text echoed to the caller. The text comes
# from the agent's own code, which makes it useful, but it is unbounded
# user-authored input on a caller-facing path. Mirrors _MAX_REASON_LEN's role
# in tool_api_error.py.
_MAX_ENTRYPOINT_ERROR_TEXT = 512


def _safe_entrypoint_error_text(error: Exception) -> str:
    """Render an entrypoint failure for a caller-facing error, redacted.

    The cause is the agent's own code failing, which is what makes it worth
    surfacing at all; but its text can echo a connection string, an API key, or
    a credential-bearing URL. Two layers, both used for caller-facing text in
    this package:

    * the shared pattern redactor (mongo/URL userinfo, bearer tokens, inline
      secrets), then
    * exact replacement of the credential values a name marks as secrets
      (``request_named_credential_values``).

    Both choices are load-bearing:

    * Patterns run on the pristine text. Replacing values verbatim first can
      destroy the ``://`` the userinfo patterns anchor on, leaving the
      credential in place.
    * Candidates are selected by variable *name*, not by value length. Tenant
      env is mostly not secret and includes single-character values ("1", "/"),
      and a length threshold cannot tell those from a short token -- it drops
      the token too. Selecting by name keeps ``/`` out (never redacted) while
      redacting a 7-character secret (always redacted).

    Deliberately renders ``type`` + ``str()`` rather than ``format_llm_error``:
    that helper unpacks a provider error's ``.details``/``.body`` wholesale,
    which is the highest-risk payload to echo, and the exception type plus
    message already carries the actionable part ("Missing credentials"). The
    full detail still reaches the pod log via the warning above.
    """
    text = _redact_text(f"{type(error).__name__}: {error}")
    for value in sorted(request_named_credential_values(), key=len, reverse=True):
        text = text.replace(value, "<redacted>")
    if len(text) > _MAX_ENTRYPOINT_ERROR_TEXT:
        text = text[: _MAX_ENTRYPOINT_ERROR_TEXT - 1] + "…"
    return text


def _llm_failure_error_code(error: Exception) -> Optional[str]:
    """Wire code for a provider credential rejection, else None.

    Deliberately narrow: only a rejection authenticated by the provider's own
    HTTP status earns the code, so a rate limit, outage, or agent-side bug
    keeps its existing generic classification instead of being relabeled as a
    customer-secret problem.
    """
    if is_llm_credential_rejection(error):
        return LLM_CREDENTIAL_REJECTED_ERROR_CODE
    return None


class LLMRegistryLoadError(RuntimeError):
    """Raised when an LLM lookup fails because the entrypoint load failed.

    The named-LLM registry is populated by running the user entrypoint.
    When that run raises, the registry is left empty (or holding only
    import-time registrations) and every lookup of an LLM the entrypoint
    would have registered fails with a bare ``KeyError`` about registration
    -- which points the agent developer at their ``app.llm()`` calls instead
    of at the transient cause that actually broke the load.

    The real cause is named in the message only. It is deliberately not
    attached as ``__cause__``, because the caller-facing response is rendered
    by ``format_llm_error``, which walks ``__cause__`` and would echo a raw
    provider ``.details``/``.body`` from there, bypassing the message
    redaction. The original ``KeyError`` remains the implicit ``__context__``.

    Unlike the bare ``KeyError`` it replaces, this is not a ``KeyError``
    subclass, so an existing ``except KeyError:`` handler will not catch it.
    """


class ToolExecution:
    """Transport-agnostic tool runtime shared by the Tool Pod server and function mode.

    Owns the tool work that is independent of how the invocation arrives:
    registering built-in tools and the named-LLM registry (``prepare``), and
    resolving + running a single tool under execution context (``invoke_tool``).
    Concrete classes provide ``self.runtime``: ``ToolServer`` via ``BaseServer``,
    ``ToolFunctionRunner`` (function mode) directly.
    """

    runtime: Any  # provided by the concrete class

    # All three provided by the concrete class's __init__ (ToolServer,
    # ToolFunctionRunner) -- declared here so the registry helpers below
    # type-check against the base class they are defined on.
    _llm_registry_lock: "asyncio.Lock"
    _llm_registry_loaded: bool
    # Why the last entrypoint run failed, or None after a successful load.
    # Read by _create_llm_for_pod to replace a downstream "not registered"
    # KeyError with the real cause. Defaulted at class level so an instance
    # built without __init__ degrades to the plain KeyError.
    _llm_registry_load_error: Optional[Exception] = None

    async def prepare(self) -> None:
        """Register built-in tool handlers."""
        from agent_engine_runner_shared import toolpod_handlers
        from agent_engine_runner_shared.connectors import CONNECTOR_BUNDLE_PATH_ENV

        # A pre-materialized bundle (env var) wins; otherwise the agent.yaml
        # `connectors` entries are materialized in-process (local files only).
        # Shape check: mocked configs auto-create truthy attributes.
        agent_connectors = getattr(self.runtime.agent_config, "connectors", None)
        has_connectors = isinstance(agent_connectors, (list, tuple)) and bool(agent_connectors)
        bundle = None
        if (os.environ.get(CONNECTOR_BUNDLE_PATH_ENV) or has_connectors) and getattr(
            self, "_connector_runtime", None
        ) is None:
            from agent_engine_runner_shared.connectors import resolve_runtime_bundle
            from agent_engine_runner_shared.mcp_tools import make_mcp_sdk_tool_name

            bundle = await asyncio.to_thread(resolve_runtime_bundle, self.runtime.agent_config)
            unavailable_names = set(self.runtime._tools) | BUILTIN_TOOL_NAMES
            for server_name, config in self.runtime.agent_config.mcp.servers.items():
                if config.allowed_tools is not None:
                    unavailable_names.update(
                        make_mcp_sdk_tool_name(server_name, tool_name)
                        for tool_name in config.allowed_tools
                    )
            bundle.require_available_names(unavailable_names)

        # ``features.deep_agent`` gates the built-in filesystem + shell
        # handlers. Tenants that don't opt in never get a file/shell attack
        # surface registered on their Tool-Pod, so the platform stays
        # agent-type-agnostic and tenants pay only for what they use.
        deep_agent_enabled = self.runtime.agent_config.feature_enabled("deep_agent", default=False)
        if deep_agent_enabled:
            toolpod_handlers.register_builtin_tools(self.runtime)

            missing = BUILTIN_TOOL_NAMES - set(self.runtime._tools.keys())
            if missing:
                raise RuntimeError(
                    f"ToolServer startup: missing built-in tool handlers: "
                    f"{sorted(missing)}. Expected all of "
                    f"{sorted(BUILTIN_TOOL_NAMES)} to be registered by "
                    f"register_builtin_tools()."
                )
            logger.info(
                "Deep-agent built-in tools registered (features.deep_agent=true): %s",
                sorted(BUILTIN_TOOL_NAMES),
            )
        else:
            logger.info(
                "Deep-agent built-in tools NOT registered "
                "(features.deep_agent is off in agent.yaml)."
            )

        if bundle is not None:
            from agent_engine_runner_shared.connectors import ConnectorToolRuntime

            connector_runtime = ConnectorToolRuntime(bundle)
            for registration in connector_runtime.registrations:
                name = registration.tool.operation.name
                self.runtime._tools[name] = registration.callable
                self.runtime._tool_definitions[name] = registration.metadata
            self._connector_runtime = connector_runtime

        tools = list(self.runtime._tools.keys())
        logger.info(f"Registered tools: {tools}")

        # Populate the registry while the guest warms, before the first model request.
        await self._ensure_llm_registry_loaded()

    async def close_connector_runtime(self) -> None:
        """Close pooled connector transports after invocations have finished."""
        connector_runtime = getattr(self, "_connector_runtime", None)
        if connector_runtime is None:
            return
        self._connector_runtime = None
        await asyncio.to_thread(connector_runtime.close)

    async def _ensure_llm_registry_loaded(self) -> None:
        """Populate the registry at startup; retry failed construction on later requests.

        Guarded by ``self._llm_registry_lock`` so subsequent preparation or
        request calls reuse the registry. Reset the registry first so
        import-time registrations don't collide if the
        entrypoint re-registers them; if the entrypoint registers nothing,
        restore the pre-reset (import-time) snapshot so import-time-only
        agents still work.
        """
        if self._llm_registry_loaded:
            return
        async with self._llm_registry_lock:
            if self._llm_registry_loaded:
                return
            graph_builder = self.runtime._graph_builder
            get_agent = (
                getattr(graph_builder, "get_agent", None) if graph_builder is not None else None
            )
            if get_agent is not None:
                pre_snapshot = snapshot_llm_registry()
                reset_llm_registry()
                try:
                    from agent_engine_runner_shared.context import customer_origin_scope

                    with customer_origin_scope():
                        get_agent()
                except Exception as exc:
                    logger.warning(
                        "Failed to run entrypoint for LLM registration; "
                        "/invoke_llm will fail if no LLM was registered",
                        exc_info=True,
                    )
                    # Discard partial registrations before restoring the last snapshot.
                    reset_llm_registry()
                    if pre_snapshot:
                        # Runtime-owned restoration, not a user app.llm() call --
                        # exempt from the entrypoint-scope requirement.
                        with entrypoint_scope():
                            for llm_id, llm in pre_snapshot.items():
                                register_llm(llm_id, llm)
                        logger.info(
                            "Entrypoint failed; restored %d import-time registration(s)",
                            len(pre_snapshot),
                        )

                    # Remember the cause so the caller's failed lookup can report
                    # it instead of a bare "not registered" KeyError.
                    self._llm_registry_load_error = exc
                    return  # Leave a failed attempt retryable by the next request.

                if has_named_llms():
                    logger.info("Named LLM registry populated from user entrypoint")
                elif pre_snapshot:
                    with entrypoint_scope():
                        for llm_id, llm in pre_snapshot.items():
                            register_llm(llm_id, llm)
                    logger.info(
                        "Entrypoint registered no LLMs; restored %d import-time registration(s)",
                        len(pre_snapshot),
                    )
                else:
                    logger.warning(
                        "Entrypoint ran but did not call app.llm(); "
                        "/invoke_llm will fail until an LLM is registered"
                    )
            # A successful load clears any cause recorded by an earlier failure,
            # so a later lookup miss is reported as a genuine registration error.
            self._llm_registry_load_error = None
            self._llm_registry_loaded = True

    async def invoke_tool(
        self,
        request: ToolPodExecuteRequest,
        *,
        owner_url_failure: Optional["OwnerUrlFailureState"] = None,
    ) -> ToolPodExecuteResponse:
        """Resolve a tool by name, run it under execution context, and return the result.

        The transport-agnostic invocation core shared by the HTTP ``/execute``
        route (``ToolServer._handle_execute``) and function mode
        (``ToolFunctionRunner``). It owns registry lookup, the MCP tool
        fallback, execution-context setup, and error capture; it never touches
        the FastAPI request/response cycle.
        """
        import socket

        pod_name = socket.gethostname()
        resolved = self._resolve_tool(request, pod_name)
        if isinstance(resolved, ToolPodExecuteResponse):
            return resolved
        return await self._run_resolved_tool(
            resolved, request, pod_name, owner_url_failure=owner_url_failure
        )

    def _resolve_tool(
        self, request: ToolPodExecuteRequest, pod_name: str
    ) -> Any | ToolPodExecuteResponse:
        """Resolve the callable for ``request`` or return an error response.

        Pure registry/config lookup with no I/O, so ``/execute`` can reject
        invalid requests without acquiring the serialization gate.
        """
        if not request.tool_name:
            return ToolPodExecuteResponse(
                status="error", error="Tool name cannot be empty", pod_name=pod_name
            )

        func = self.runtime._tools.get(request.tool_name)
        if not func:
            from agent_engine_runner_shared.mcp_tools import (
                MCPConfigError,
                is_configured_mcp_sdk_tool_name,
            )

            mcp_server_name = request.metadata.get("mcp_server")
            mcp_tool_name = request.metadata.get("mcp_tool")
            if (
                not isinstance(mcp_server_name, str)
                or mcp_server_name == ""
                or not isinstance(mcp_tool_name, str)
                or mcp_tool_name == ""
            ):
                if is_configured_mcp_sdk_tool_name(
                    self.runtime.agent_config.mcp,
                    request.tool_name,
                ):
                    return ToolPodExecuteResponse(
                        status="error",
                        error=(
                            f"mcp tool {request.tool_name!r} is missing discovered "
                            "MCP call metadata (mcp_server, mcp_tool). Ensure AER has "
                            "been upgraded to send MCP tool identity."
                        ),
                        pod_name=pod_name,
                    )
                return ToolPodExecuteResponse(
                    status="error", error=f"Unknown tool: {request.tool_name}", pod_name=pod_name
                )

            try:
                func = self._configured_mcp_tool(
                    request.tool_name,
                    mcp_server_name,
                    mcp_tool_name,
                )
            except MCPConfigError as exc:
                return ToolPodExecuteResponse(status="error", error=str(exc), pod_name=pod_name)
            except Exception as exc:
                logger.exception("Unexpected error resolving MCP tool %r", request.tool_name)
                return ToolPodExecuteResponse(status="error", error=str(exc), pod_name=pod_name)
            if not func:
                return ToolPodExecuteResponse(
                    status="error", error=f"Unknown tool: {request.tool_name}", pod_name=pod_name
                )

        return func

    async def _run_resolved_tool(
        self,
        func: Any,
        request: ToolPodExecuteRequest,
        pod_name: str,
        *,
        owner_url_failure: Optional["OwnerUrlFailureState"] = None,
    ) -> ToolPodExecuteResponse:
        """Run an already-resolved tool under request-scoped execution context."""
        from agent_engine_runner_shared.connectors.runtime import connector_environment
        from agent_engine_runner_shared.context import (
            clear_execution_context,
            get_current_execution_metadata,
            get_requested_suspend,
            set_execution_context,
        )
        from agent_engine_runner_shared.custom_events import (
            clear_custom_event_transport,
            install_tool_custom_event_transport,
        )

        # Pin the callback base to the runner's own configured OE (deploy-time
        # OE_URL) before it is used as the callback base or the owner-URL trust
        # anchor. The request field is only honoured when no OE_URL is stamped
        # (local dev / tests). Resolving once here means the owner URL is
        # validated against a trusted anchor, not the raw request value.
        oe_url = resolve_oe_url(request.oe_url or "")
        context_tokens = set_execution_context(
            trace_id=request.platform_trace_id,
            execution_id=request.execution_id,
            wrapper=None,
            oe_url=oe_url,
            oe_owner_url=resolve_owner_url(request.oe_owner_url, oe_url),
            owner_url_failure=owner_url_failure,
            user_id=request.user_id,
            session_id=request.session_id,
            authorization=request.authorization,
            custom_headers=request.custom_headers,
            payload=request.payload,
        )
        transport_token = install_tool_custom_event_transport(
            enabled=self.runtime.agent_config.feature_enabled("use_custom_parser", default=False)
        )

        try:
            from agent_engine_runner_shared.context import customer_origin_scope

            with connector_environment(tenant_env_vars()), customer_origin_scope():
                if asyncio.iscoroutinefunction(func):
                    result = await func(**request.arguments)
                else:
                    # Off-load sync handlers to the thread-pool so one
                    # blocking tool cannot stall the event loop.
                    result = await asyncio.to_thread(func, **request.arguments)
            metadata = get_current_execution_metadata()
            # Author-intended suspend is signaled out of band via
            # SuspendPayload.to_json, never inferred from result content — so
            # relayed untrusted data cannot forge a HITL suspend.
            suspend_marker = get_requested_suspend()
            if suspend_marker is not None:
                return ToolPodExecuteResponse(
                    status="suspend",
                    result=json.dumps(suspend_marker),
                    pod_name=pod_name,
                    metadata=metadata,
                    oob_suspend_supported=True,
                )
            return ToolPodExecuteResponse(
                status="success",
                result=result,
                pod_name=pod_name,
                metadata=metadata,
                oob_suspend_supported=True,
            )

        except Exception as e:
            metadata = get_current_execution_metadata()
            tool_def = self.runtime._tool_definitions.get(request.tool_name, {})
            provider_type = tool_def.get("provider_type") if isinstance(tool_def, dict) else None
            tae, err = classify_tool_api_error(
                e, provider_type, credentials=request_credential_values()
            ) or (None, str(e))
            return ToolPodExecuteResponse(
                status="error",
                error=err,
                pod_name=pod_name,
                metadata=metadata,
                tool_api_error=tae,
            )

        finally:
            clear_custom_event_transport(transport_token)
            clear_execution_context(context_tokens)

    def _configured_mcp_tool(
        self,
        sdk_tool_name: str,
        mcp_server_name: str,
        mcp_tool_name: str,
    ) -> Any | None:
        """Return a Tool Pod callable for an MCP tool configured in agent.yaml.

        Tool Pods intentionally skip remote ``tools/list`` discovery during startup.
        They only need to execute the server-prefixed tool name OE sends after AER
        has already discovered and bound the schema.
        """
        from agent_engine_runner_shared.mcp_tools import (
            make_mcp_tool_pod_callable,
            resolve_configured_mcp_tool_binding,
        )

        binding = resolve_configured_mcp_tool_binding(
            self.runtime.agent_config.mcp,
            sdk_tool_name,
            mcp_server_name,
            mcp_tool_name,
        )
        if binding is None:
            return None
        return make_mcp_tool_pod_callable(binding)


class ToolServer(ToolExecution, BaseServer):
    """
    Tool Executor server - executes tool functions.

    Responsibilities:
    - Execute registered tool functions
    - Provide isolation for tools that need network/secrets
    - Return results to caller
    """

    def __init__(self, runtime: "TenantRuntime") -> None:
        super().__init__(runtime)
        # Serialize the credential-bearing endpoints (/execute, /invoke_llm,
        # /invoke_llm/stream) on this pod: applied_metadata_env mutates
        # process-global os.environ, so per-call delegated credentials and
        # request context must never overlap. Restriction is off,
        # so the gate is skipped.
        self._execute_gate = asyncio.Lock()
        self._restriction_disabled = True
        # Guards one-time registry loading across startup and request paths.
        self._llm_registry_lock = asyncio.Lock()
        self._llm_registry_loaded = False
        self._llm_registry_load_error = None
        self._connector_runtime = None

    @property
    def mode_name(self) -> str:
        return "tool"

    @asynccontextmanager
    async def _credential_isolation(self) -> AsyncGenerator[None, None]:
        """Credential window for restricted vs unrestricted secret handling.

        Restricted (default): take ``_execute_gate`` and apply/restore metadata
        env. Unrestricted: merge metadata into ``os.environ`` without
        restore and skip the gate so concurrent tool calls can overlap.
        """
        if self._restriction_disabled:
            merge_metadata_env()
            yield
            return
        async with self._execute_gate:
            with applied_metadata_env():
                yield

    async def on_startup(self) -> None:
        """Wire the tool-runtime preparation into the server startup lifecycle."""
        await self.prepare()

    async def on_shutdown(self) -> None:
        """Close connector transports after the server has drained requests."""
        await self.close_connector_runtime()

    def get_health_details(self) -> Dict[str, Any]:
        return {
            "tools": list(self.runtime._tools.keys()),
            "tool_count": len(self.runtime._tools),
        }

    def register_routes(self, app: FastAPI) -> None:
        """Register Tool-specific routes."""
        server = self

        @app.exception_handler(RequestValidationError)
        async def request_validation_error(
            _request: Request, exc: RequestValidationError
        ) -> JSONResponse:
            """Return useful validation details without reflecting request values."""
            return JSONResponse(
                status_code=422,
                content={
                    "detail": [
                        {key: error[key] for key in ("type", "loc", "msg") if key in error}
                        for error in exc.errors()
                    ]
                },
            )

        @app.post("/execute", response_model=ToolPodExecuteResponse)
        async def execute_tool(
            request: ToolPodExecuteRequest, http_request: Request
        ) -> ToolPodExecuteResponse:
            """Execute a registered tool function in this isolated pod.

            Runs the named tool with the provided arguments. Tools are registered
            via the Runner SDK ``app.tool()`` decorator and execute with the pod's
            network and secret access.
            """
            return await server._run_with_inbound_trace_context(
                "tool.execute",
                OpenInferenceSpanKind.TOOL,
                http_request.headers,
                lambda: server._handle_execute(request),
            )

        @app.post("/invoke_llm", response_model=LLMPodInvokeResponse)
        async def invoke_llm(
            request: LLMPodInvokeRequest, http_request: Request
        ) -> LLMPodInvokeResponse:
            """Invoke an LLM and return the complete response.

            Collects the full streaming response from the configured LLM provider,
            including token usage. Used by SecureWrappedLLM when streaming is not
            needed by the caller.
            """
            return await server._run_with_inbound_trace_context(
                "tool.invoke_llm",
                OpenInferenceSpanKind.CHAIN,
                http_request.headers,
                lambda: server._handle_invoke_llm(request),
            )

        @app.post("/invoke_llm/stream")
        async def invoke_llm_stream(
            request: LLMPodInvokeRequest, http_request: Request
        ) -> StreamingResponse:
            """Stream an LLM response from this tool pod via Server-Sent Events.

            Returns a streaming response of ``LLMPodStreamEvent`` JSON objects,
            one per SSE ``data:`` line. The final event has ``done=true`` and
            includes token usage and timing metadata.
            """
            # Register before returning: StreamingResponse starts iterating
            # the body only after the handler returns, so a drain landing in
            # between must already see the work (and would otherwise answer
            # execution_not_found while the stream runs). The consuming task
            # attaches when iteration starts; if the client disconnects before
            # iteration ever starts, the attach grace releases the
            # registration rather than leaking phantom work.
            registry = server.drain_registry
            drain_handle = registry.begin_work(
                request.execution_id, expect_attach=True, step_number=request.step_number
            )

            async def _tracked_stream() -> AsyncGenerator[str, None]:
                task = asyncio.current_task()
                if not registry.attach_task(drain_handle, task):
                    # The attach grace already released this registration (the
                    # client stalled before iteration). Entering the provider
                    # stream now would run work no drain can see — a completed
                    # drain must remain trustworthy quiescence — so the stream
                    # ends empty instead.
                    logger.warning(
                        "Refusing to stream for execution %s: stream registration "
                        "was released before the consumer attached",
                        request.execution_id,
                    )
                    return
                try:
                    async for chunk in server._stream_with_inbound_trace_context(
                        "tool.invoke_llm.stream",
                        OpenInferenceSpanKind.CHAIN,
                        http_request.headers,
                        server._handle_invoke_llm_stream(request),
                    ):
                        yield chunk
                finally:
                    registry.end_work(drain_handle)

            return StreamingResponse(
                _tracked_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        @app.post("/guardrails/check", response_model=GuardrailCheckResponse)
        async def check_guardrails(
            request: GuardrailCheckRequest, http_request: Request
        ) -> GuardrailCheckResponse:
            """Evaluate selected guardrail policies for OE-controlled runtime content."""
            return await server._run_with_inbound_trace_context(
                "tool.guardrails_check",
                OpenInferenceSpanKind.TOOL,
                http_request.headers,
                lambda: server._handle_guardrails_check(request),
            )

        @app.get("/tools", response_model=ToolsListResponse)
        async def list_tools() -> ToolsListResponse:
            """List all tools registered with this Tool Pod instance.

            Returns the tool definitions that were registered via the Runner SDK
            ``app.tool()`` decorator during agent initialization.
            """
            return ToolsListResponse(
                tools=[
                    {"name": name, **defn}
                    for name, defn in server.runtime._tool_definitions.items()
                ],
                count=len(server.runtime._tool_definitions),
            )

    async def _run_with_inbound_trace_context(
        self,
        span_name: str,
        span_kind: OpenInferenceSpanKind,
        headers: "Headers",
        handler: "Callable[[], Awaitable[Any]]",
    ) -> Any:
        """Extract an inbound W3C traceparent (if present) before handling.

        The OE injects a traceparent on every tool-pod call; without this,
        the spans produced while a tool or routed LLM call runs (framework
        instrumentation, outbound httpx) root a disconnected trace instead of
        joining the invocation's. Mirrors the AER's inbound extraction and
        falls back to running untraced if the optional ``tracing`` extra
        isn't installed.

        ``span_kind`` is the OpenInference kind the span declares, so a backend
        receiving it can classify the call instead of showing no kind.
        """
        try:
            from opentelemetry.trace.propagation.tracecontext import (
                TraceContextTextMapPropagator,
            )

            from agent_engine_runner_shared.tracing import get_tracer
        except Exception:
            return await handler()

        carrier = dict(headers.items())
        parent_context = TraceContextTextMapPropagator().extract(carrier=carrier)
        tracer = get_tracer("agent-engine-runner-shared.tool")
        with tracer.start_as_current_span(
            span_name,
            context=parent_context,
            attributes={OPENINFERENCE_SPAN_KIND: span_kind.value},
        ):
            return await handler()

    async def _stream_with_inbound_trace_context(
        self,
        span_name: str,
        span_kind: OpenInferenceSpanKind,
        headers: "Headers",
        chunks: "AsyncIterator[str]",
    ) -> "AsyncIterator[str]":
        """Streaming variant of :meth:`_run_with_inbound_trace_context`.

        A ``StreamingResponse`` body runs after the route handler returns, so
        the span must wrap the generator itself: it starts under the
        extracted parent when iteration begins and ends when the stream is
        exhausted, errors, or the client disconnects.

        ``span_kind`` is declared on the span as in the non-streaming variant.
        """
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.trace.propagation.tracecontext import (
                TraceContextTextMapPropagator,
            )

            from agent_engine_runner_shared.tracing import get_tracer
        except Exception:
            async for chunk in chunks:
                yield chunk
            return

        carrier = dict(headers.items())
        parent_context = TraceContextTextMapPropagator().extract(carrier=carrier)
        tracer = get_tracer("agent-engine-runner-shared.tool")
        span = tracer.start_span(
            span_name,
            context=parent_context,
            attributes={OPENINFERENCE_SPAN_KIND: span_kind.value},
        )
        with otel_trace.use_span(span, end_on_exit=True):
            async for chunk in chunks:
                yield chunk

    @with_metrics("tool_pod_guardrails_check", record_errors=False)
    async def _handle_guardrails_check(
        self, request: GuardrailCheckRequest
    ) -> GuardrailCheckResponse:
        # Count-only registration: the evaluation is synchronous and fast, so
        # there is no cancellable await to signal; a drain waits it out.
        drain_handle = self.drain_registry.begin_work(request.execution_id)
        try:
            return evaluate_guardrail_check(request)
        finally:
            self.drain_registry.end_work(drain_handle)

    @with_metrics("tool_pod_execute", record_errors=False)
    async def _handle_execute(self, request: ToolPodExecuteRequest) -> ToolPodExecuteResponse:
        """Execute a tool function for an HTTP ``/execute`` call."""
        import socket

        log_tool_request(
            request.tool_name or "(empty)",
            request.arguments,
            step=0,
            prefix="TOOL_POD",
            # Read the registry dict directly (mirrors the TS toolRedactFields
            # lookup): duck-typed runtimes in tests carry _tool_definitions but
            # not the TenantRuntime accessor.
            fields_to_redact=(
                self.runtime._tool_definitions.get(request.tool_name or "") or {}
            ).get("redact_fields"),
        )
        pod_name = socket.gethostname()
        # Resolve before acquiring the gate so invalid requests fail fast
        # instead of queueing behind an in-flight call.
        resolved = self._resolve_tool(request, pod_name)
        if isinstance(resolved, ToolPodExecuteResponse):
            return resolved
        # Async tools are drained by cancelling the request task; sync tools
        # run on a thread the runtime cannot cancel, so they are tracked but
        # only waited out (an honest timed_out, never a claimed cancellation).
        drain_task = asyncio.current_task() if asyncio.iscoroutinefunction(resolved) else None
        # Raises 409 if the execution was drained after OE dispatched.
        drain_handle = self.drain_registry.begin_work(
            request.execution_id, drain_task, step_number=request.step_number
        )
        try:
            async with self._credential_isolation():
                return await self._run_resolved_tool(resolved, request, pod_name)
        finally:
            self.drain_registry.end_work(drain_handle)

    def _create_llm_for_pod(
        self,
        llm_id: str,
        tools: Optional[List[LLMToolSchema]] = None,
        tool_choice: Any = None,
    ) -> Any:
        """Create a BaseLLM-compatible instance for tool pod execution.

        Retrieves the named LLM from the registry (registered via
        ``app.llm(llm, llm_id=...)``) and wraps it through the registered
        adapter factory. No SecureWrappedLLM needed — OE approval already
        happened in AER.

        ``tool_choice`` forces a specific bound tool (e.g. the schema bound by
        ``with_structured_output``); it is forwarded to the adapter's
        ``bind_tools`` call so LangChain can translate it.
        """
        try:
            llm = get_named_llm(llm_id)
        except KeyError:
            load_error = self._llm_registry_load_error
            if load_error is None:
                raise
            # The lookup miss is a symptom of a failed entrypoint load, so
            # reporting it as a missing app.llm() call would misdirect the
            # agent developer. Surface the real cause instead -- redacted, since
            # this text is returned to the caller. This branch is reachable only
            # for a lookup that already failed, so it can never turn a working
            # call into a failure -- in particular the import-time-snapshot
            # restore in _ensure_llm_registry_loaded still resolves an llm_id
            # the snapshot holds even when the entrypoint run raised.
            #
            # Chained with `from None`, not `from load_error`: the caller-facing
            # error is rendered by format_llm_error, which walks `__cause__` and
            # would dump a provider error's raw `.details`/`.body` from there,
            # bypassing the redaction in the message.
            raise LLMRegistryLoadError(
                f"llm_id {llm_id!r} is not registered because the agent entrypoint "
                f"failed when the LLM registry was loaded. The next LLM call "
                f"re-runs the entrypoint, so this recovers once the underlying "
                f"cause clears. Underlying failure: "
                f"{_safe_entrypoint_error_text(load_error)}"
            ) from None
        kwargs: dict[str, Any] = {"tools": tools}
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return get_llm_adapter_factory()(llm, **kwargs)

    @with_metrics("llm_pod_invoke", record_errors=False)
    async def _handle_invoke_llm(self, request: LLMPodInvokeRequest) -> LLMPodInvokeResponse:
        """Execute an LLM call by collecting the streaming execution path."""
        import socket

        pod_name = socket.gethostname()
        start_time = time.time()

        from agent_engine_runner_shared.context import (
            clear_execution_context,
            set_execution_context,
        )

        # Raises 409 if the execution was drained after OE dispatched.
        drain_handle = self.drain_registry.begin_work(
            request.execution_id, asyncio.current_task(), step_number=request.step_number
        )
        tokens = set_execution_context(
            execution_id=request.execution_id,
            wrapper=None,
            oe_url="",
            trace_id=request.platform_trace_id,
        )
        try:
            # Usually a no-op after startup; keep any registry load inside the
            # request's credential isolation, as with model execution below.
            async with self._credential_isolation():
                await self._ensure_llm_registry_loaded()
                chunks = [chunk async for chunk in self._stream_llm_chunks(request)]
            llm_response = SecureLLMProxy.response_from_stream_chunks(chunks)
            duration_ms = (time.time() - start_time) * 1000
            return LLMPodInvokeResponse(
                status="success",
                result=llm_response,
                pod_name=pod_name,
                duration_ms=duration_ms,
                usage=llm_response.usage,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return LLMPodInvokeResponse(
                status="error",
                error=format_llm_error(e),
                error_code=_llm_failure_error_code(e),
                pod_name=pod_name,
                duration_ms=duration_ms,
            )
        finally:
            clear_execution_context(tokens)
            self.drain_registry.end_work(drain_handle)

    async def _stream_llm_chunks(
        self, request: LLMPodInvokeRequest
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Yield typed LLM stream chunks with retry-before-first-delta semantics."""
        import asyncio

        invoke_args = request.arguments
        llm = self._create_llm_for_pod(
            invoke_args.llm_id, invoke_args.tools, invoke_args.tool_choice
        )
        messages = _deserialize_messages(invoke_args.messages)
        extra_kwargs = invoke_args.options.to_model_kwargs() if invoke_args.options else {}

        last_error: Exception | None = None
        stream_started = False

        for attempt in range(LLM_MAX_RETRIES):
            usage = None
            try:
                stream = llm.astream(
                    messages,
                    stop=invoke_args.stop_sequences,
                    **extra_kwargs,
                )

                async for chunk in stream:
                    stream_chunk = _coerce_stream_chunk(chunk)
                    if _stream_chunk_has_payload(stream_chunk):
                        stream_started = True
                        yield stream_chunk
                    if stream_chunk.usage is not None:
                        usage = merge_token_usage(usage, stream_chunk.usage)

                if usage is not None:
                    yield LLMStreamChunk(usage=usage)
                return

            except Exception as e:
                last_error = e
                if not stream_started and is_retryable_error(e) and attempt < LLM_MAX_RETRIES - 1:
                    backoff = min(
                        LLM_INITIAL_BACKOFF * (LLM_BACKOFF_MULTIPLIER**attempt),
                        LLM_MAX_BACKOFF,
                    )
                    logger.warning(
                        f"LLM Pod Stream: Transient provider failure (attempt {attempt + 1}/{LLM_MAX_RETRIES}). "
                        f"Retrying in {backoff:.1f}s..."
                    )
                    await asyncio.sleep(backoff)
                    continue
                break

        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to create LLM stream")

    async def _handle_invoke_llm_stream(
        self, request: LLMPodInvokeRequest
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response as SSE events."""
        import socket

        from agent_engine_runner_shared.context import (
            clear_execution_context,
            set_execution_context,
        )

        pod_name = socket.gethostname()
        start_time = time.time()
        tokens = set_execution_context(
            execution_id=request.execution_id,
            wrapper=None,
            oe_url="",
            trace_id=request.platform_trace_id,
        )
        try:
            usage = None
            # See the gate + metadata-env ordering rationale in
            # _handle_invoke_llm above.
            async with self._credential_isolation():
                await self._ensure_llm_registry_loaded()
                async for stream_chunk in self._stream_llm_chunks(request):
                    if not _stream_chunk_has_payload(stream_chunk):
                        if stream_chunk.usage is not None:
                            usage = stream_chunk.usage
                        continue
                    chunk_event = LLMPodStreamEvent(
                        content=stream_chunk.content,
                        tool_call_chunks=stream_chunk.tool_calls,
                        id=stream_chunk.id,
                        name=stream_chunk.name,
                        response_metadata=stream_chunk.response_metadata,
                        additional_kwargs=stream_chunk.additional_kwargs,
                    )
                    yield f"data: {chunk_event.model_dump_json(exclude_none=True)}\n\n"

            # Final summary event
            duration_ms = (time.time() - start_time) * 1000
            summary = LLMPodStreamEvent(
                done=True,
                pod_name=pod_name,
                duration_ms=duration_ms,
                usage=usage,
            )
            yield f"data: {summary.model_dump_json(exclude_none=True)}\n\n"

        except Exception as e:
            error_event = LLMPodStreamEvent(
                error=format_llm_error(e),
                error_code=_llm_failure_error_code(e),
            )
            yield f"data: {error_event.model_dump_json(exclude_none=True)}\n\n"
        finally:
            clear_execution_context(tokens)
