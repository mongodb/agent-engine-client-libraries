"""
Agent Execution Runtime (AER) server - Framework-agnostic agent execution.

The AER is responsible for:
- Executing agents via the BaseAgent protocol (framework-neutral)
- Making LLM calls (with network access)
- Routing all calls through OE via SecureToolWrapper
- Handling HITL via StreamEvent suspend/resume pattern
- Reporting completion/suspension to OE
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, NamedTuple, Optional, TypedDict, Union, cast

import httpx
from agent_engine_sdk import (
    AgentInput,
    RequestContext,
    collect_message_artifact_metadata,
)
from agent_engine_sdk.models import (
    SessionMessagesResponse,
    SessionsSummaryResponse,
)
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, Response
from pydantic import JsonValue, ValidationError

from agent_engine_runner_shared.checkpoint_workspace import resolve_checkpoint_workspace_id
from agent_engine_runner_shared.context import (
    OwnerUrlFailureState,
    clear_execution_context,
    close_session_finish_latch,
    current_execution_id,
    customer_origin_scope,
    get_current_oe_owner_url,
    is_session_finish_requested,
    report_oe_owner_url_failure,
    set_execution_context,
)
from agent_engine_runner_shared.metrics import with_metrics
from agent_engine_runner_shared.models import (
    AERExecuteResponse,
    ExecuteRequest,
    ExecutorCallbackRequest,
    InterruptResult,
    PendingInterrupt,
    StreamingResult,
    SuspendPayload,
    ToolsListResponse,
)
from agent_engine_runner_shared.node_logger import NodeExecutionLogger
from agent_engine_runner_shared.secure_wrapper import (
    ExternalAPICallError,
    LLMInvocationError,
    PolicyDeniedException,
    SecureToolWrapper,
    ToolCallTimeoutError,
)
from agent_engine_runner_shared.server.base import BaseServer
from agent_engine_runner_shared.server.callback_delivery import CallbackDelivery
from agent_engine_runner_shared.server.chunk_types import (
    CUSTOM_EVENT,
    DONE,
    ERROR,
    LLM_CREDENTIAL_REJECTED_ERROR_CODE,
    LLM_INVOCATION_ERROR_CODE,
    LLM_INVOCATION_ERROR_SOURCE,
    POLICY_DENIED_ERROR_CODE,
    SUBAGENT_END,
    SUBAGENT_START,
    TEXT,
    TIMEOUT_ERROR_CODE,
    TOOL_CREDENTIAL_REJECTED_ERROR_CODE,
)
from agent_engine_runner_shared.server.graph_warm import GraphWarmer
from agent_engine_runner_shared.server.http_retry import post_json_with_retries
from agent_engine_runner_shared.server.oe_url import resolve_oe_url
from agent_engine_runner_shared.server.owner_url import resolve_owner_url
from agent_engine_runner_shared.span_kinds import OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind
from agent_engine_runner_shared.span_names import AER_BUILD_AGENT
from agent_engine_runner_shared.tls_client import create_async_httpx_client_with_tls
from agent_engine_runner_shared.utils import (
    filter_thinking_tokens,
    get_env_float,
    get_request_timeout,
    log_execution_callback,
    log_section,
    strip_thinking,
)
from agent_engine_runner_shared.workflow import AsyncWorkflowClient
from agent_engine_runner_shared.workflow.attempt import (
    attempt_start_request_from_execute,
    heartbeat_interval_seconds_from_attempt,
    new_durability_owner_id,
)
from agent_engine_runner_shared.workflow.context import reset_attempt_context, set_attempt_context
from agent_engine_runner_shared.workflow.heartbeat import AttemptHeartbeat
from agent_engine_runner_shared.workflow.memory import (
    DurableMemoryState,
    validate_durable_memory_identity,
)

if TYPE_CHECKING:
    from starlette.datastructures import Headers

    from agent_engine_runner_shared.runtime import TenantRuntime

logger = logging.getLogger(__name__)

# Whole-turn execution timeout. Matches the platform's stream deadline so a turn
# is not cut off short of the limit the caller was promised.
EXECUTION_TIMEOUT = get_env_float("RUNNER_EXECUTION_TIMEOUT", 600.0)

# Bounded-retry policy for AER -> OE chunk POSTs. Transient HTTP errors
# (connection error, 5xx, read/write/protocol timeouts) are retried with
# exponential backoff (0.1s, 0.2s, 0.4s ...). 4xx responses are not retried.
MAX_CHUNK_POST_ATTEMPTS = 3
CHUNK_POST_BASE_DELAY_S = 0.1
# After the SDK emits a terminal result/suspend event, ask the iterator for one
# more item so async generators can run their natural cleanup path instead of
# being closed by GeneratorExit from an in-loop return.
AGENT_STREAM_TERMINAL_DRAIN_TIMEOUT_S = get_env_float("AGENT_STREAM_TERMINAL_DRAIN_TIMEOUT", 1.0)
# Bounded wait for queued memory writes to drain before asking the OE to
# release a finished session's compute. The OE may kill this pod
# with zero grace once it honours the finish request, so a write still queued
# past this deadline is released anyway rather than held forever.
SESSION_FINISH_MEMORY_DRAIN_TIMEOUT_S = 10.0
CAPABILITY_ADVERTISE_TIMEOUT_S = 10.0


async def _aclose_stream_iter(stream_iter: Any, execution_id: str) -> None:
    """Close an async iterator when it supports generator-style ``aclose``."""
    aclose = getattr(stream_iter, "aclose", None)
    if aclose is None:
        return
    try:
        # Generator cleanup (finally blocks in customer agent code) runs
        # during aclose — keep it inside the customer scope for attribution.
        with customer_origin_scope():
            result = aclose()
            if inspect.isawaitable(result):
                await result
    except Exception:
        logger.warning(
            "Failed to close agent stream iterator for execution %s",
            execution_id,
            exc_info=True,
        )


class _SubagentBoundaryData(TypedDict, total=False):
    """Expected fields on subagent_start / subagent_end SDK events."""

    source: str
    subagent_name: str
    tool_call_id: str
    description: str  # present on subagent_start; forwarded in metadata
    summary: str  # present on subagent_end; forwarded in metadata


def _artifact_metadata_from_messages(messages: list[Any]) -> dict[str, JsonValue]:
    """Collect generic message artifact metadata from the final assistant message."""
    for message in reversed(messages):
        if getattr(message, "role", None) != "assistant":
            continue

        metadata = collect_message_artifact_metadata(
            getattr(message, "additional_kwargs", None),
        )
        return metadata.to_message_metadata() if metadata else {}

    return {}


def _policy_denied_metadata(exc: PolicyDeniedException) -> dict[str, JsonValue]:
    """Structured metadata for a policy-engine denial, attached to the ERROR
    chunk + terminal callback so the OE/UI can identify a "blocked by policy"
    outcome via ``error_code`` rather than string-matching the message. The
    guardrail identity is included when the denial came from a guardrail policy
    (plain tool/model/budget denials carry only the reason)."""
    meta: dict[str, JsonValue] = {
        "error_code": POLICY_DENIED_ERROR_CODE,
        "reason": exc.reason,
    }
    if exc.guardrail_meta is not None:
        meta["guardrail_id"] = exc.guardrail_meta.guardrail_id
        meta["guardrail_category"] = exc.guardrail_meta.guardrail_category
    return meta


def _llm_invocation_metadata(error_code: Optional[str] = None) -> dict[str, JsonValue]:
    """Structured metadata for an LLM-provider failure after OE approval.

    ``source`` is the gateway invoke-owner key; ``error_code`` is persisted
    on the execution so unary invokes can classify without the live stream
    chunk. ``code`` is the same value so stream consumers that read
    ``metadata.code`` (the gateway) match ``metadata.error_code`` (AER).

    When the tool pod stamped a specific classification (e.g. a credential
    rejection), it replaces the generic code and ``source`` is omitted: the
    gateway maps that bare code to client-owned, while ``llm`` attribution
    would read as provider flakiness for a failure the customer can fix.

    Only the pod's credential stamp is honored. The live relay forwards the
    pod's frames verbatim and LLMInvocationError is agent-raisable, so the
    value arriving here is workload-authored: an unrecognized code falls back
    to the generic classification rather than letting the workload rename its
    failure for the gateway's allowlist and owner attribution.
    """
    if error_code == LLM_CREDENTIAL_REJECTED_ERROR_CODE:
        return {
            "error_code": error_code,
            "code": error_code,
        }
    return {
        "error_code": LLM_INVOCATION_ERROR_CODE,
        "code": LLM_INVOCATION_ERROR_CODE,
        "source": LLM_INVOCATION_ERROR_SOURCE,
    }


def _credential_rejection_metadata(exc: Exception) -> Optional[dict[str, JsonValue]]:
    """Structured metadata when the run failed on a rejected credential.

    Two shapes qualify, and only these: a tool call whose classified
    external-API failure was an auth rejection (the tool's configured
    credential), and an LLMInvocationError carrying the pod's credential stamp
    the framework wrapped before it reached the dedicated catch. Frameworks
    may nest either under ``__cause__``, so the walk covers the chain;
    ``__context__`` is deliberately not walked, because an unrelated error
    raised while handling an auth failure must not inherit its classification.
    Any other failure shape returns None and the error stays unclassified.
    """
    seen: set[int] = set()
    stack: list[BaseException | None] = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        # Literal matches the closed classification the pod stamps
        # (tool_api_error's status map) and the OE allowlists.
        if (
            isinstance(current, ExternalAPICallError)
            and current.tool_api_error.classification == "AUTH_FAILED"
        ):
            return {
                "error_code": TOOL_CREDENTIAL_REJECTED_ERROR_CODE,
                "code": TOOL_CREDENTIAL_REJECTED_ERROR_CODE,
            }
        # The exception type is agent-raisable, so only the exact code the pod
        # can have stamped is honored; anything else falls back to generic.
        if (
            isinstance(current, LLMInvocationError)
            and current.error_code == LLM_CREDENTIAL_REJECTED_ERROR_CODE
        ):
            return {
                "error_code": current.error_code,
                "code": current.error_code,
            }
        stack.append(current.__cause__)
    return None


def _executor_callback_body(
    execution_id: str,
    status: str,
    suspend_generation: Optional[int] = None,
    result: Any = None,
    error: Optional[str] = None,
    suspend_reason: Optional[str] = None,
    suspend_context: Optional[Dict[str, Any]] = None,
    interrupts: Optional[list[PendingInterrupt]] = None,
    resume_schema: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> dict[str, Any]:
    callback = ExecutorCallbackRequest(
        execution_id=execution_id,
        status=status,
        suspend_generation=suspend_generation,
        result=result,
        error=error,
        suspend_reason=suspend_reason,
        suspend_context=suspend_context,
        interrupts=interrupts,
        resume_schema=resume_schema,
        metadata=metadata,
    )
    callback_body = callback.model_dump()
    if suspend_generation is None:
        callback_body.pop("suspend_generation", None)
    # Serialize before cleanup so validation and delivery use the same detached
    # payload even if agent-owned values are mutated while resources close.
    return json.loads(json.dumps(callback_body, allow_nan=False))


class _ResolvedParams(NamedTuple):
    session_id: str
    message: str


def _resolve_invocation_params(request: ExecuteRequest) -> _ResolvedParams:
    """Resolve the continuation id and message for this execution.

    ``session_id`` falls back to ``execution_id`` when the caller did not
    supply one. Framework resume state (e.g. a checkpoint id) is *not*
    resolved here — it is opaque state the adapter reads from ``metadata``.
    """
    req_payload = request.payload or {}

    session_id = request.session_id or request.execution_id

    _payload_message = req_payload.get("message")
    message = (_payload_message if isinstance(_payload_message, str) else None) or request.message

    return _ResolvedParams(
        session_id=session_id,
        message=message,
    )


def _resolve_workspace_id(request: ExecuteRequest) -> Optional[str]:
    """Resolve the workspace_id used to scope checkpoint thread_ids.

    Delegates to :func:`agent_engine_runner_shared.checkpoint_workspace.resolve_checkpoint_workspace_id`
    so the write path and the read-side query plugin cannot diverge.
    """
    return resolve_checkpoint_workspace_id(request.workspace_id)


def _resolve_resume_payload(request: ExecuteRequest) -> JsonValue | None:
    return cast("JsonValue | None", request.resume_data)


def _reject_ambiguous_message(request: ExecuteRequest) -> None:
    """Reject a request that carries ``message`` both top-level and in ``payload``.

    The AER is the only component that unpacks the opaque payload, so the
    collision is caught here and raised as a clean 400. Doing it here (rather
    than in a model validator) keeps the payload out of the error body and the
    logs — a 422 from a validator would echo the entire request, payload
    included, back to the caller.
    """
    payload_msg = request.payload.get("message") if request.payload else None
    if request.message and isinstance(payload_msg, str) and payload_msg:
        raise HTTPException(
            status_code=400,
            detail=(
                "Ambiguous request: 'message' was provided both at the top level "
                "and inside 'payload'. Send the message in exactly one place."
            ),
        )


class AERServer(BaseServer):
    """
    Agent Execution Runtime server - framework-agnostic agent execution.

    Responsibilities:
    - Execute agents via the BaseAgent protocol (framework-neutral)
    - Use SecureToolWrapper for all tool/LLM calls
    - Report completion or suspension to OE
    """

    def __init__(self, runtime: "TenantRuntime"):
        super().__init__(runtime)
        self._clients: Dict[str, httpx.AsyncClient] = {}
        self._client_cert_mtimes: Dict[str, Optional[tuple[float, float, float]]] = {}
        self._client_lock = asyncio.Lock()
        self._a2a_registered = False
        self._capabilities_advertised = False
        self._chunk_seq: Dict[str, int] = {}
        # Validated replica-specific OE owner callback URL per in-flight
        # execution. Resolved once in _handle_execute and read by the
        # /stream/chunk and /executor/callback transports so a call site added
        # later inherits owner preference without threading (mirrors how
        # platform_api_url is rebound once rather than passed at every use).
        self._owner_callback_url: Dict[str, str] = {}
        # Process-lifetime owner id sent on StartAttempt / heartbeats.
        self._durability_owner_id = new_durability_owner_id()
        # Execution ids whose SDK run called request_session_finish() and
        # completed normally; drained by the /execute route after the response
        # is sent.
        self._pending_session_finish: set[str] = set()
        self._callback_delivery = CallbackDelivery(lambda url: self._get_client(url))
        self._graph_warmer = GraphWarmer(self.runtime.warm_up_agent)

    @property
    def mode_name(self) -> str:
        return "aer"

    def _get_cert_mtimes(self) -> Optional[tuple[float, float, float]]:
        """Get modification times of TLS cert files for rotation detection."""
        try:
            cert_path = os.environ.get("TLS_CERT_PATH", "")
            key_path = os.environ.get("TLS_KEY_PATH", "")
            ca_path = os.environ.get("TLS_CA_CERT_PATH", "")

            # Only track mtimes if we're using file-based certs (container mode)
            if not (cert_path and key_path and ca_path):
                return None

            return (
                os.path.getmtime(cert_path),
                os.path.getmtime(key_path),
                os.path.getmtime(ca_path),
            )
        except (OSError, ValueError):
            return None

    async def _get_client(self, base_url: str) -> httpx.AsyncClient:
        """
        Get or create HTTP client for the given base URL (thread-safe).

        Configures mTLS when base_url uses HTTPS. Clients are cached per base URL
        to support scenarios where platform_api_url differs from OE_URL (e.g., tests
        or override configs).

        Automatically invalidates cached clients if TLS certificate files have been
        rotated (based on mtime). This ensures clients pick up renewed certificates
        without requiring a pod restart.

        Args:
            base_url: The target service URL (must include scheme, e.g., https://...)
        """
        # Normalize URL for cache key (strip trailing slash)
        cache_key = base_url.rstrip("/")

        # Check if certs have been rotated (container mode only) and create/rotate client.
        # CRITICAL: cert rotation check must be inside the lock to avoid race conditions
        # where two coroutines both close the same client and the second raises KeyError.
        current_mtimes = self._get_cert_mtimes()

        async with self._client_lock:
            # Check if client exists and certs have been rotated
            if cache_key in self._clients and current_mtimes is not None:
                if self._client_cert_mtimes.get(cache_key) != current_mtimes:
                    logger.info(
                        "TLS certificates rotated, invalidating cached HTTP client",
                        extra={
                            "base_url": cache_key,
                            "old_mtimes": self._client_cert_mtimes.get(cache_key),
                            "new_mtimes": current_mtimes,
                        },
                    )
                    # Use pop to safely remove even if another coroutine already removed it
                    old_client = self._clients.pop(cache_key, None)
                    if old_client is not None:
                        await old_client.aclose()
                    self._client_cert_mtimes.pop(cache_key, None)

            # Create client if missing (first call or post-rotation)
            if cache_key not in self._clients:
                self._clients[cache_key] = await create_async_httpx_client_with_tls(
                    base_url, get_request_timeout()
                )
                self._client_cert_mtimes[cache_key] = current_mtimes

        return self._clients[cache_key]

    async def on_startup(self) -> None:
        """Verify graph builder is ready and run startup registration hooks."""
        if self.runtime._graph_builder is None:
            raise RuntimeError(
                "Agent graph builder not initialized. "
                "Did you pass a graph_builder to register_and_run()?"
            )

        logger.info("AER graph builder ready")

        oe_url = os.environ.get("OE_URL", "").strip()
        workspace_id = os.environ.get("APP_ID", "").strip()
        if not oe_url or not workspace_id:
            raise RuntimeError("Capability registration requires OE_URL and APP_ID")
        await self._ensure_oe_registrations(oe_url, workspace_id)

    async def on_shutdown(self) -> None:
        """Clean up resources."""
        await self._callback_delivery.shutdown()
        async with self._client_lock:
            for client in self._clients.values():
                await client.aclose()
            self._clients.clear()

    async def _ensure_oe_registrations(self, oe_url: str, workspace_id: str) -> None:
        """Run pending one-shot OE registrations (capability advertise, A2A)."""
        if not self._capabilities_advertised:
            await self._advertise_capabilities(oe_url, workspace_id)
        if not self._a2a_registered:
            await self._register_a2a_config(oe_url, workspace_id)

    async def _advertise_capabilities(self, oe_url: str, workspace_id: str) -> None:
        """Push the current declaration to OE, failing startup if it is not accepted."""
        agent_cfg = self.runtime.agent_config
        org_id = self.runtime.org_id
        project_id = self.runtime.project_id
        # Platform always injects ORG_ID/PROJECT_ID (including local-dev). Refuse
        # to stamp blank scope: sticky lookup filters by org/project and would
        # permanently miss a blank row.
        if not org_id or not project_id:
            raise RuntimeError(
                "Capability registration requires ORG_ID and PROJECT_ID "
                f"for workspace {workspace_id}"
            )

        language = (agent_cfg.language or "").strip() or "python"
        framework = (agent_cfg.framework or "").strip()
        features = agent_cfg.features.explicit()
        # SDK-injected, not an agent.yaml flag: always true for this SDK
        # version, so OE can tell it's safe to emit owner callback URLs.
        features = {"owner_callback_fallback": True, **features}
        payload = {
            "workspace_id": workspace_id,
            "org_id": org_id,
            "project_id": project_id,
            "language": language,
            "framework": framework,
            "features": features,
        }

        client = await self._get_client(oe_url)
        async with asyncio.timeout(CAPABILITY_ADVERTISE_TIMEOUT_S):
            resp = await client.post(
                f"{oe_url}/agent/capabilities",
                json=payload,
                timeout=CAPABILITY_ADVERTISE_TIMEOUT_S,
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Capability registration was rejected for {workspace_id} "
                f"with status {resp.status_code}"
            )

        logger.info(
            "Agent capabilities advertised for workspace %s "
            "(language=%s framework=%s durable_workflow=%s)",
            workspace_id,
            language,
            framework or "-",
            features.get("durable_workflow"),
        )
        self._capabilities_advertised = True

    async def _register_a2a_config(self, oe_url: str, workspace_id: str) -> None:
        """Push agent.yaml A2A config to the OE (one-shot, fail-open)."""
        a2a_cfg = self.runtime.agent_config.a2a
        try:
            # Do not strip: OE HMAC-verifies with the env value as-is.
            a2a_secret = os.environ.get("A2A_JWT_SECRET", "")
            if a2a_cfg.enabled and not a2a_secret:
                logger.error(
                    "A2A_JWT_SECRET is not set; A2A registration will fail and "
                    "discover_available_agents will look like no peers are configured. "
                    "Locally, set the same A2A_JWT_SECRET in the project .env for every "
                    "agent. The deployed platform seeds this automatically."
                )
            client = await self._get_client(oe_url)
            payload = {
                "workspace_id": workspace_id,
                "name": self.runtime.app_name,
                "description": "",
                "org_id": os.environ.get("ORG_ID", "") or (self.runtime.org_id or ""),
                "project_id": os.environ.get("PROJECT_ID", ""),
                "a2a_enabled": a2a_cfg.enabled,
                "skills": [
                    {
                        "name": sk.name,
                        "description": sk.description,
                        "example_input": sk.example_input or "",
                        "example_output": sk.example_output or "",
                    }
                    for sk in a2a_cfg.skills
                ],
                "input_modes": a2a_cfg.input_modes,
                "output_modes": a2a_cfg.output_modes,
                "allowed_callers": a2a_cfg.allowed_callers,
                "aer_http_endpoint": self._build_aer_endpoint(oe_url, workspace_id),
                "tool_http_endpoint": self._build_component_endpoint(oe_url, workspace_id, "tool"),
            }
            headers: dict[str, str] = {}
            if a2a_secret:
                from agent_engine_runner_shared.a2a_auth import sign_registration_jwt
                from agent_engine_runner_shared.db_config import get_store_db_name

                platform_db = get_store_db_name()
                token = sign_registration_jwt(
                    a2a_secret,
                    workspace_id,
                    issuer=f"oe.{platform_db}",
                    org_id=os.environ.get("ORG_ID", "") or (self.runtime.org_id or ""),
                    project_id=os.environ.get("PROJECT_ID", ""),
                )
                headers["Authorization"] = f"Bearer {token}"

            resp = await client.post(
                f"{oe_url}/a2a/register", json=payload, headers=headers, timeout=10.0
            )
            if resp.status_code == 200:
                logger.info(
                    f"A2A config registered for workspace {workspace_id} (enabled={a2a_cfg.enabled})"
                )
                self._a2a_registered = True
            else:
                msg = f"A2A registration returned {resp.status_code}: {resp.text}"
                if a2a_cfg.enabled:
                    msg += (
                        ". A2A_JWT_SECRET may be missing or mismatched; locally set "
                        "the same value in the project .env for every agent. The "
                        "deployed platform seeds this automatically."
                    )
                logger.warning(msg)
        except httpx.HTTPError as e:
            logger.warning(f"A2A registration failed (will retry on first /execute): {e}")
        except Exception as e:  # noqa: BLE001 — non-network error, will retry on first /execute
            logger.error(f"A2A registration failed unexpectedly: {e}")

    @staticmethod
    def _build_component_endpoint(oe_url: str, workspace_id: str, component: str) -> str:
        """Build an in-cluster service URL for a workspace component.

        Extracts the namespace from the OE URL and reads the component's port
        from the Kubernetes service-discovery env var that the kubelet injects
        for every Service in the namespace.

        Args:
            oe_url: The OE's in-cluster URL (contains the namespace).
            workspace_id: The workspace identifier.
            component: The component suffix ("aer" or "tool").
        """
        from urllib.parse import urlparse

        parsed = urlparse(oe_url)
        hostname = parsed.hostname or ""
        parts = hostname.split(".")
        namespace = parts[1] if len(parts) >= 2 else ""

        # Read the port from the Kubernetes service-discovery env var:
        # {WS_ID}_{COMPONENT}_SERVICE_PORT (uppercased, hyphens → underscores)
        svc_env_prefix = f"{workspace_id}_{component}".upper().replace("-", "_")
        port = os.environ.get(f"{svc_env_prefix}_SERVICE_PORT", "")
        if not port:
            port = os.environ.get("APP_PORT", "8001")

        svc_name = f"{workspace_id}-{component}"
        if namespace:
            return f"http://{svc_name}.{namespace}.svc.cluster.local:{port}"
        return f"http://{svc_name}:{port}"

    @staticmethod
    def _build_aer_endpoint(oe_url: str, workspace_id: str) -> str:
        """Derive the AER's in-cluster service URL from the OE URL."""
        return AERServer._build_component_endpoint(oe_url, workspace_id, "aer")

    def get_health_details(self) -> Dict[str, Any]:
        return {
            "graph_builder_ready": self.runtime._graph_builder is not None,
            "tools_registered": list(self.runtime._tools.keys()),
        }

    def register_routes(self, app: FastAPI) -> None:
        """Register AER-specific routes."""
        server = self

        @app.post("/warm-up", status_code=204)
        async def warm_up() -> Response:
            """Run the optional bounded graph warm-up sequence."""
            await server._graph_warmer.run()
            return Response(status_code=204)

        @app.post("/execute", response_model=AERExecuteResponse)
        async def execute(
            request: ExecuteRequest,
            http_request: Request,
            background_tasks: BackgroundTasks,
        ) -> AERExecuteResponse:
            """Execute an agent workflow via the BaseAgent protocol.

            Accepts an execution request from the Orchestration Engine, runs the
            agent graph with SecureToolWrapper for tool/LLM routing, and reports
            completion or HITL suspension back to the OE via callback.
            """
            drain_registry = server.drain_registry
            # Raises 409 if the execution was drained after OE dispatched.
            drain_handle = drain_registry.begin_work(request.execution_id, asyncio.current_task())
            try:
                response = await server._execute_with_inbound_trace_context(
                    request, http_request.headers
                )
            finally:
                drain_registry.end_work(drain_handle)
            # Release the session's compute only as a background task, i.e.
            # after this response is sent: the OE may kill this pod with zero
            # grace once it honours the finish request.
            if request.execution_id in server._pending_session_finish:
                server._pending_session_finish.discard(request.execution_id)
                background_tasks.add_task(server._release_finished_session, request)
            return response

        @app.get("/tools", response_model=ToolsListResponse)
        async def list_tools() -> ToolsListResponse:
            """List all tools registered with this AER instance.

            Returns the tool definitions that were registered via the Runner SDK
            ``app.tool()`` decorator during agent initialization.
            """
            return ToolsListResponse(
                tools=[
                    {"name": name, **defn}
                    for name, defn in server.runtime._tool_definitions.items()
                ],
            )

        @app.get("/query/sessions", response_model=SessionsSummaryResponse)
        async def query_session_summaries(
            # The platform's session-list page caps at 50; 200 leaves
            # headroom for batched enrichment without letting a single
            # request fan out into a 100k-element $in scan if the OE proxy
            # is ever bypassed (the trust model the AER documents).
            session_ids: list[str] = Query(default_factory=list, max_length=200),
        ) -> SessionsSummaryResponse:
            """Return framework-derived summary for the given session_ids.

            Used by the platform to enrich its workspace-scoped sessions list
            with framework-specific data (timestamps, message counts, message
            preview). The platform supplies the session_ids; this endpoint
            does no workspace scoping of its own.
            """
            if not session_ids:
                return SessionsSummaryResponse(sessions=[])
            plugin = server.runtime.get_query_plugin()
            if plugin is None:
                raise HTTPException(
                    status_code=501,
                    detail="session queries are not supported by this framework",
                )
            return await plugin.get_summaries_for_sessions(session_ids)

        @app.get(
            "/query/sessions/{session_id}/messages",
            response_model=SessionMessagesResponse,
        )
        async def query_messages_for_session(
            session_id: str,
        ) -> SessionMessagesResponse:
            """Return the decoded conversation history for ``session_id``.

            Returns an empty message list if the session has no persisted
            state.
            """
            plugin = server.runtime.get_query_plugin()
            if plugin is None:
                raise HTTPException(
                    status_code=501,
                    detail="session queries are not supported by this framework",
                )
            return await plugin.get_messages_for_session(session_id)

    @staticmethod
    @contextlib.contextmanager
    def _traced_step(name: str, kind: OpenInferenceSpanKind = OpenInferenceSpanKind.CHAIN):
        """Open a named child span under the active trace, or no-op untraced.

        Mirrors ``_execute_with_inbound_trace_context``'s fallback: the
        optional ``tracing`` extra may not be installed, so this must not
        raise when OTel isn't available.
        """
        try:
            from agent_engine_runner_shared.tracing import get_tracer
        except Exception:
            yield None
            return

        tracer = get_tracer("agent-engine-runner-shared.aer")
        with tracer.start_as_current_span(
            name, attributes={OPENINFERENCE_SPAN_KIND: kind.value}
        ) as span:
            yield span

    async def _execute_with_inbound_trace_context(
        self, request: ExecuteRequest, headers: "Headers"
    ) -> AERExecuteResponse:
        """Extract an inbound W3C traceparent (if present) before executing.

        Without this, this AER's own spans — and the trace_id persisted onto
        ExecutionStep/tool-call records via get_current_trace_context() — start
        a disconnected root trace even though the caller's traceparent header
        now arrives correctly over the A2A hop. Falls back to
        running untraced if the optional `tracing` extra isn't installed.
        """
        try:
            from opentelemetry.trace.propagation.tracecontext import (
                TraceContextTextMapPropagator,
            )

            from agent_engine_runner_shared.tracing import get_tracer
        except Exception:
            return await self._handle_execute(request)

        carrier = dict(headers.items())
        parent_context = TraceContextTextMapPropagator().extract(carrier=carrier)
        tracer = get_tracer("agent-engine-runner-shared.aer")
        # This span is the outermost step of an invocation, so it declares AGENT
        # rather than leaving a backend to fall back on the span name.
        with tracer.start_as_current_span(
            "aer.execute",
            context=parent_context,
            attributes={OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKind.AGENT.value},
        ):
            return await self._handle_execute(request)

    @with_metrics("aer_execute")
    async def _handle_execute(self, request: ExecuteRequest) -> AERExecuteResponse:
        """
        Handle /execute request from OE.

        Executes the agent via BaseAgent.stream() with SecureToolWrapper.
        """
        log_section()
        logger.info("AER: Execute %s... (resume=%s)", request.execution_id[:8], request.resume)

        # Pin the callback base to the runner's own configured OE before any
        # of it is used. Every outbound call below derives from this field,
        # and the runner treats the responses as authoritative results and
        # policy decisions — so it must not be chosen by the request.
        # Rebinding here rather than at each use site means a call site added
        # later inherits the trusted value automatically.
        request.platform_api_url = resolve_oe_url(request.platform_api_url)
        # Owner-callback fallback: a request-supplied replica URL is
        # only honoured when it validates as a headless replica of the trusted
        # service URL above; otherwise None and every callback uses the service
        # URL. Resolved once here so every callback site inherits it.
        owner_url = resolve_owner_url(request.platform_api_owner_url, request.platform_api_url)
        owner_url_failure = OwnerUrlFailureState()
        logger.info("AER: OE callback URL: %s", request.platform_api_url)
        logger.debug(
            "AER: Message: %s",
            request.message[:50] + "..." if len(request.message) > 50 else request.message,
        )

        # Fail fast on a message supplied both top-level and inside payload,
        # before any side effects (A2A registration, execution).
        _reject_ambiguous_message(request)

        # Defensively enforce readiness registration if the startup lifecycle
        # was bypassed. Capability identity comes from trusted deployment
        # configuration, not the caller-controlled execute payload.
        registration_workspace_id = os.environ.get("APP_ID", "").strip()
        if registration_workspace_id:
            await self._ensure_oe_registrations(request.platform_api_url, registration_workspace_id)

        self.runtime.note_checkpoint_wire_workspace_id(request.workspace_id)

        resolved = _resolve_invocation_params(request)
        session_id = resolved.session_id
        message = resolved.message
        workspace_id = _resolve_workspace_id(request)

        # Create node execution logger callback (sends events to OE for logging)
        node_logger = NodeExecutionLogger(
            oe_url=request.platform_api_url,
            execution_id=request.execution_id,
            session_id=session_id,
            user_id=request.user_id,
            org_id=request.org_id,
            project_id=request.project_id,
        )
        logger.debug("Created NodeExecutionLogger for session %s", session_id[:8])

        # Build the agent before the durable-routing decision: materializing
        # the graph is what registers the workflow adapter (a graph without
        # the platform checkpointer must stay native), so the attempt gate
        # below sees the correct eligibility even on the first request.
        with self._traced_step(AER_BUILD_AGENT):
            agent = self.runtime.get_agent(callbacks=[node_logger])

        # Durable route: activate an OE attempt before tenant work. Missing
        # identity, an old OE (404), or an ineligible framework/graph stays
        # native; other start failures fail closed before any replayable
        # work runs.
        durable_attempt = await self._start_durable_attempt(request, session_id)

        if durable_attempt is not None and self.runtime.memory_enabled:
            validate_durable_memory_identity(
                durable_attempt.workflow_identity,
                request.user_id,
            )

        # Use contextvars for per-execution isolation
        wrapper = SecureToolWrapper(
            oe_url=request.platform_api_url,
            execution_id=request.execution_id,
            custom_headers=request.custom_headers,
            oe_owner_url=owner_url,
            durable_memory=(
                DurableMemoryState(message or None)
                if durable_attempt is not None and self.runtime.memory_enabled
                else None
            ),
            call_registry=self.drain_registry,
        )
        # Native resume continues after its checkpoint. Durable replay instead
        # restarts at zero so each activity reaches its original OE identity.
        if request.resume_from_step is not None and durable_attempt is None:
            # Raise the shared allocator watermark so the next mint continues
            # past the resumed step (not restarting at 1).
            wrapper.observe_operational_step(request.resume_from_step)

        heartbeat = None
        attempt_token = None
        pending_callback: Optional[dict[str, Any]] = None

        def stage_callback(
            *,
            status: str,
            **fields: Any,
        ) -> None:
            nonlocal pending_callback
            pending_callback = _executor_callback_body(
                request.execution_id,
                status,
                suspend_generation=request.suspend_generation,
                **fields,
            )

        if durable_attempt is not None:
            heartbeat = AttemptHeartbeat(
                attempt=durable_attempt,
                interval_seconds=heartbeat_interval_seconds_from_attempt(
                    durable_attempt.heartbeat_interval_ms
                ),
                oe_url=request.platform_api_url,
            )
            # Activate the lease before tenant code can perform replayable
            # work; start() is fail-closed and blocks, so keep it off the loop.
            try:
                await asyncio.to_thread(heartbeat.start)
            except Exception:
                await wrapper.close()
                raise

        # Every resource acquired from here on is released by the finally
        # below; nothing may run between lease activation and this guard.
        context_tokens = None
        execution_succeeded = False
        try:
            # Registered inside the try so the finally's pop always runs; the
            # first callback that consumes it happens after this point.
            if owner_url:
                self._owner_callback_url[request.execution_id] = owner_url
            if durable_attempt is not None:
                attempt_token = set_attempt_context(durable_attempt)
            context_tokens = set_execution_context(
                trace_id=request.platform_trace_id,
                execution_id=request.execution_id,
                wrapper=wrapper,
                oe_url=request.platform_api_url,
                oe_owner_url=owner_url,
                owner_url_failure=owner_url_failure,
                user_id=request.user_id,
                session_id=session_id,
                workspace_id=workspace_id,
                custom_headers=request.custom_headers,
                payload=request.payload,
            )
            ctx = RequestContext(
                execution_id=request.execution_id,
                session_id=session_id,
                user_id=request.user_id,
                workspace_id=workspace_id,
                request_headers=request.custom_headers,
                resume=request.resume,
                resume_data=_resolve_resume_payload(request),
                metadata=request.metadata,
                previous_execution_cancelled=request.previous_execution_cancelled,
            )
            # AgentInput.payload is the opaque caller blob, with the resolved
            # message normalized in. Platform plumbing (identity, resume,
            # checkpoint) travels on ctx, not here.
            caller_payload: Dict[str, Any] = dict(request.payload or {})
            caller_payload["message"] = message
            agent_input = AgentInput(payload=caller_payload)

            if request.resume:
                logger.info("AER: Resume execution for %s", request.execution_id[:8])

            logger.info(
                "AER /execute: execution_id=%s, session_id=%s",
                request.execution_id[:8],
                session_id[:8],
            )

            # Execute via BaseAgent.stream()
            try:
                execution_outcome = await asyncio.wait_for(
                    self._execute_via_agent_stream(
                        agent,
                        ctx,
                        agent_input,
                        request.platform_api_url,
                        request.execution_id,
                    ),
                    timeout=EXECUTION_TIMEOUT,
                )
            except asyncio.TimeoutError:
                error_msg = f"Execution timed out after {EXECUTION_TIMEOUT} seconds"
                try:
                    await self._send_stream_chunk(
                        request.platform_api_url,
                        request.execution_id,
                        chunk_type=ERROR,
                        error=error_msg,
                        metadata={"error_code": TIMEOUT_ERROR_CODE},
                    )
                except Exception as chunk_err:
                    logger.error(
                        "Failed to deliver ERROR chunk for execution %s after retries: %s",
                        request.execution_id,
                        chunk_err,
                        exc_info=True,
                    )
                stage_callback(
                    status="ERROR",
                    error=error_msg,
                    metadata={"error_code": TIMEOUT_ERROR_CODE},
                )
                raise HTTPException(status_code=504, detail=error_msg)

            # Check if the agent was suspended (HITL).
            # _execute_via_agent_stream returns InterruptResult for
            # suspensions and StreamingResult for normal completion.
            if isinstance(execution_outcome, InterruptResult):
                has_envelope = (
                    execution_outcome.interrupts is not None
                    and execution_outcome.resume_schema is not None
                )
                if has_envelope:
                    if not execution_outcome.interrupts:
                        # Report ERROR to OE before failing, like the legacy
                        # empty-payload branch below: without the callback the
                        # execution would stay "running" forever.
                        error_msg = "Agent produced an empty interrupt snapshot; cannot suspend."
                        log_execution_callback(
                            request.execution_id, "ERROR", error=error_msg, prefix="AER"
                        )
                        stage_callback(
                            status="ERROR",
                            error=error_msg,
                        )
                        raise HTTPException(status_code=500, detail=error_msg)
                    try:
                        payload = SuspendPayload.model_validate(
                            execution_outcome.interrupts[0].value
                        )
                    except ValidationError:
                        suspend_reason = "agent_interrupt"
                        suspend_context = None
                    else:
                        suspend_reason = payload.suspend_reason
                        suspend_context = payload.suspend_context
                else:
                    if not execution_outcome.suspend_payload:
                        error_msg = "Agent produced an empty suspend payload; cannot suspend."
                        log_execution_callback(
                            request.execution_id, "ERROR", error=error_msg, prefix="AER"
                        )
                        stage_callback(
                            status="ERROR",
                            error=error_msg,
                        )
                        raise HTTPException(status_code=500, detail=error_msg)
                    payload = SuspendPayload.model_validate(execution_outcome.suspend_payload)
                    suspend_reason = payload.suspend_reason
                    suspend_context = payload.suspend_context

                log_execution_callback(
                    request.execution_id,
                    "SUSPENDED",
                    suspend_reason=suspend_reason,
                    prefix="AER",
                )

                # Persist the pre-suspend portion of this turn to STM before
                # returning. The suspend branch otherwise writes nothing, so the
                # user prompt and any assistant/tool output produced before the
                # interrupt would be lost from memory. On a fresh
                # invoke (not a resume) the user turn is included; on a resume
                # leg there is no new user prompt, so it is omitted.
                if self.runtime._memory_writer and durable_attempt is None:
                    try:
                        self.runtime.write_turn_async(
                            message=message,
                            result_messages=execution_outcome.messages,
                            user_id=request.user_id,
                            session_id=session_id,
                            include_user_turn=not request.resume,
                        )
                        logger.info(f"Pre-suspend memory write queued for user {request.user_id}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to queue pre-suspend memory write: {e}", exc_info=True
                        )

                # The framework adapter owns resume semantics; its opaque
                # metadata is forwarded to the OE verbatim and round-tripped
                # back into RequestContext.metadata on resume.
                stage_callback(
                    status="SUSPENDED",
                    suspend_reason=suspend_reason,
                    suspend_context=suspend_context,
                    interrupts=execution_outcome.interrupts,
                    resume_schema=execution_outcome.resume_schema,
                    metadata=execution_outcome.metadata,
                )
                execution_succeeded = True
                return AERExecuteResponse(
                    status="suspended",
                    suspend_reason=suspend_reason,
                )

            # Normal completion — write conversation turn to memory (async, non-blocking).
            # include_user_turn=not request.resume: a resume leg carries no new
            # user prompt (the input was a Command(resume=...)), so the user turn
            # was already written when the turn first started.
            if (
                self.runtime._memory_writer
                and execution_outcome.messages
                and durable_attempt is None
            ):
                try:
                    self.runtime.write_turn_async(
                        message=message,
                        result_messages=execution_outcome.messages,
                        user_id=request.user_id,
                        session_id=session_id,
                        include_user_turn=not request.resume,
                    )
                    logger.info(f"Memory write queued for user {request.user_id}")
                except Exception as e:
                    logger.warning(f"Failed to queue memory write: {e}", exc_info=True)

            try:
                done_metadata = {
                    "status": "completed",
                    **_artifact_metadata_from_messages(execution_outcome.messages),
                }
                await self._send_stream_chunk(
                    request.platform_api_url,
                    request.execution_id,
                    chunk_type=DONE,
                    content=execution_outcome.content,
                    metadata=done_metadata,
                )
            except Exception as chunk_err:
                logger.error(
                    "Failed to deliver DONE chunk for execution %s after retries: %s",
                    request.execution_id,
                    chunk_err,
                    exc_info=True,
                )

            completion_fields: dict[str, Any] = {
                "status": "COMPLETED",
                "result": execution_outcome.content,
            }
            if execution_outcome.metadata is not None:
                completion_fields["metadata"] = execution_outcome.metadata
            stage_callback(
                **completion_fields,
            )
            if is_session_finish_requested():
                # Latched here, acted on after the response: the OE blocks on
                # this /execute call and turns a transport error into a 502
                # that overwrites the finished run's result, so the session's
                # pods must not be killed before the response is delivered.
                self._pending_session_finish.add(request.execution_id)
            execution_succeeded = True
            return AERExecuteResponse(status="completed", result=execution_outcome.content)

        except HTTPException:
            raise
        except PolicyDeniedException as e:
            # A policy denial is a deliberate, terminal outcome. Attach a
            # machine-readable discriminator (metadata.error_code="policy_denied"
            # plus the bare reason) so consumers can render a "blocked by policy"
            # result instead of string-matching the message.
            #
            # The human-readable `error` keeps the full exception text
            # ("Policy denied: <reason>") so consumers that don't branch on
            # metadata see no change vs. the generic catch-all below; the
            # discriminator and bare reason ride in metadata.
            #
            # The discriminator is guaranteed on the live stream ERROR chunk. It
            # is also attached to the terminal callback, but the OE currently
            # drops terminal error metadata (and the status / SSE-fallback paths
            # don't expose it), so non-streaming consumers can't rely on it until
            # the OE follow-up lands — sending it now is forward-compatible.
            policy_meta = _policy_denied_metadata(e)
            log_execution_callback(
                request.execution_id, "POLICY_DENIED", error=e.reason, prefix="AER"
            )
            try:
                await self._send_stream_chunk(
                    request.platform_api_url,
                    request.execution_id,
                    chunk_type=ERROR,
                    error=str(e),
                    metadata=policy_meta,
                )
            except Exception as chunk_err:
                logger.error(
                    "Failed to deliver policy-denied ERROR chunk for execution %s after retries: %s",
                    request.execution_id,
                    chunk_err,
                    exc_info=True,
                )
            stage_callback(
                status="ERROR",
                error=str(e),
                metadata=policy_meta,
            )
            raise HTTPException(status_code=500, detail=str(e))
        except ToolCallTimeoutError as e:
            # A deadline breach, not a crash: carry the same discriminator as the
            # whole-turn timeout so a consumer handles both the same way, and 504
            # rather than 500 for the same reason.
            timeout_meta: dict[str, JsonValue] = {
                "error_code": TIMEOUT_ERROR_CODE,
                "tool_name": e.tool_name,
                "elapsed_seconds": e.elapsed_seconds,
            }
            log_execution_callback(request.execution_id, "ERROR", error=str(e), prefix="AER")
            try:
                await self._send_stream_chunk(
                    request.platform_api_url,
                    request.execution_id,
                    chunk_type=ERROR,
                    error=str(e),
                    metadata=timeout_meta,
                )
            except Exception as chunk_err:
                logger.error(
                    "Failed to deliver timeout ERROR chunk for execution %s after retries: %s",
                    request.execution_id,
                    chunk_err,
                    exc_info=True,
                )
            stage_callback(
                status="ERROR",
                error=str(e),
                metadata=timeout_meta,
            )
            raise HTTPException(status_code=504, detail=str(e))
        except LLMInvocationError as e:
            # Stamp source=llm only for provider-owned failures. Relay,
            # truncation, and guardrail plumbing share this exception type
            # without that source and must stay uncoded.
            if e.source != LLM_INVOCATION_ERROR_SOURCE:
                log_execution_callback(request.execution_id, "ERROR", error=str(e), prefix="AER")
                stage_callback(
                    status="ERROR",
                    error=str(e),
                )
                raise HTTPException(status_code=500, detail=str(e))
            llm_meta = _llm_invocation_metadata(e.error_code)
            log_execution_callback(request.execution_id, "ERROR", error=str(e), prefix="AER")
            try:
                await self._send_stream_chunk(
                    request.platform_api_url,
                    request.execution_id,
                    chunk_type=ERROR,
                    error=str(e),
                    metadata=llm_meta,
                )
            except Exception as chunk_err:
                logger.error(
                    "Failed to deliver LLM-invocation ERROR chunk for execution %s after retries: %s",
                    request.execution_id,
                    chunk_err,
                    exc_info=True,
                )
            stage_callback(
                status="ERROR",
                error=str(e),
                metadata=llm_meta,
            )
            raise HTTPException(status_code=500, detail=str(e))
        except asyncio.CancelledError:
            # The drain (or pod teardown) cancelled this handler task: the
            # OE's cancel path owns the terminal Cancelled settlement, so the
            # held /execute must answer cleanly — a 500 here races that
            # settlement and marks the run ERROR. No terminal callback is
            # staged for the same reason.
            logger.debug(
                "AER: execution %s cancelled (drain or teardown)",
                request.execution_id[:8],
            )
            return AERExecuteResponse(status="cancelled")
        except Exception as e:
            log_execution_callback(request.execution_id, "ERROR", error=str(e), prefix="AER")
            stage_callback(
                status="ERROR",
                error=str(e),
                metadata=_credential_rejection_metadata(e),
            )
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            # Close the session-finish latch first, before anything below
            # awaits: `await wrapper.close()` yields control, and a task
            # scheduled during the turn (e.g. asyncio.create_task) that
            # resumes in that window would otherwise see the latch still
            # open and get REQUESTED - a request that already missed the
            # is_session_finish_requested() drain above and would never be
            # acted on. Closing here, before any await, closes
            # that gap: covers the success, error, policy-denied, and
            # suspend paths alike.
            close_session_finish_latch()
            self._chunk_seq.pop(request.execution_id, None)

            cleanup_error: Optional[BaseException] = None

            def remember_cleanup_error(phase: str, exc: BaseException) -> None:
                nonlocal cleanup_error
                if cleanup_error is None:
                    cleanup_error = exc
                    if execution_succeeded:
                        return
                logger.error(
                    "Additional AER cleanup failure during %s for execution %s: %s",
                    phase,
                    request.execution_id,
                    exc,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

            # Cancellation is a BaseException. Remember it long enough to
            # attempt every settlement phase, then re-raise the first failure.
            try:
                await wrapper.close()
            except BaseException as exc:
                remember_cleanup_error("wrapper close", exc)
            try:
                if context_tokens is not None:
                    clear_execution_context(context_tokens)
            except BaseException as exc:
                remember_cleanup_error("execution context reset", exc)
            try:
                if attempt_token is not None:
                    reset_attempt_context(attempt_token)
            except BaseException as exc:
                remember_cleanup_error("attempt context reset", exc)
            if heartbeat is not None:
                heartbeat_stop = asyncio.create_task(asyncio.to_thread(heartbeat.stop))
                while True:
                    try:
                        await asyncio.shield(heartbeat_stop)
                        break
                    except asyncio.CancelledError as exc:
                        remember_cleanup_error("heartbeat shutdown", exc)
                        if heartbeat_stop.done():
                            try:
                                heartbeat_stop.result()
                            except BaseException as stop_exc:
                                remember_cleanup_error("heartbeat shutdown", stop_exc)
                            break
                    except BaseException as exc:
                        remember_cleanup_error("heartbeat shutdown", exc)
                        break
            try:
                if pending_callback is not None:
                    await self._send_callback_body(
                        request.platform_api_url,
                        pending_callback,
                        owner_url=owner_url,
                        owner_url_failure=owner_url_failure,
                    )
            except BaseException as exc:
                remember_cleanup_error("callback delivery", exc)
            finally:
                self._owner_callback_url.pop(request.execution_id, None)
            if cleanup_error is not None and execution_succeeded:
                raise cleanup_error

    async def _start_durable_attempt(self, request: ExecuteRequest, session_id: str):
        """Ask OE to activate a fenced attempt, or return None for native routing.

        None covers incomplete tenant identity and an old OE without workflow
        routes (bare 404); any other failure propagates so the invocation
        fails closed rather than running unfenced.
        """
        start_request = attempt_start_request_from_execute(
            request,
            session_id,
            self._durability_owner_id,
            self.runtime.app_name,
            workflow_version=self.runtime.app_version,
            memory_enabled=self.runtime.memory_enabled,
        )
        if start_request is None:
            return None
        async with AsyncWorkflowClient(request.platform_api_url) as client:
            return await client.start_attempt(start_request)

    async def _post_chunk_with_retries(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: Dict[str, Any],
        *,
        max_attempts: int = MAX_CHUNK_POST_ATTEMPTS,
        base_delay: float = CHUNK_POST_BASE_DELAY_S,
        owner_url: Optional[str] = None,
        on_owner_failure: Optional[Callable[[], None]] = None,
    ) -> None:
        """POST a streaming chunk to OE with the shared bounded-retry policy.

        See ``agent_engine_runner_shared.server.http_retry.post_json_with_retries`` for the
        retried exception set, owner-URL fallback, and backoff behaviour.
        """
        await post_json_with_retries(
            client,
            url,
            payload,
            max_attempts=max_attempts,
            base_delay=base_delay,
            owner_url=owner_url,
            on_owner_failure=on_owner_failure,
        )

    def _discard_owner_callback_url(self, execution_id: str) -> None:
        """Forget a replica callback URL after a failed owner pre-attempt."""
        self._owner_callback_url.pop(execution_id, None)
        if current_execution_id.get() == execution_id:
            report_oe_owner_url_failure()

    def _get_owner_callback_url(self, execution_id: str) -> Optional[str]:
        """Return request-scoped owner state, with a direct-call fallback."""
        if current_execution_id.get() == execution_id:
            return get_current_oe_owner_url()
        return self._owner_callback_url.get(execution_id)

    async def _send_stream_chunk(
        self,
        oe_url: str,
        execution_id: str,
        chunk_type: str,
        content: str = "",
        error: str = "",
        metadata: Optional[Dict[str, JsonValue]] = None,
        custom_event: Optional[JsonValue] = None,
    ) -> None:
        """Send a streaming chunk to OE's /stream/chunk endpoint.

        ``metadata`` is forwarded verbatim to OE. See
        ``agent_engine_runner_shared.server.chunk_types`` for the canonical chunk_type
        values and the per-type metadata keys each event carries.

        ``custom_event`` carries author-shaped output-parser output; consumers
        gate on both its presence and ``chunk_type == CUSTOM_EVENT``.

        Transient HTTP failures are retried with bounded exponential backoff
        (see ``_post_chunk_with_retries``). Terminal chunks (``done`` /
        ``error``) re-raise on exhaustion so the caller can recover; the
        caller is responsible for falling through to ``_report_callback`` so
        OE's ``exec.Status`` fallback can still synthesize a terminal SSE
        chunk from ``exec.Result`` / ``exec.Error``. Non-terminal chunks
        (``text`` / ``subagent_start`` / ``subagent_end`` / ``custom_event``)
        log at WARNING on exhaustion and return; the run continues, the
        chunk is dropped.
        """
        seq = self._chunk_seq.get(execution_id, 0) + 1
        self._chunk_seq[execution_id] = seq

        client = await self._get_client(oe_url)
        chunk = {
            "execution_id": execution_id,
            "chunk_type": chunk_type,
            "content": content,
            "error": error,
            "metadata": metadata or {},
            "custom_event": custom_event,
            "seq": seq,
        }
        owner_url = self._get_owner_callback_url(execution_id)
        terminal = chunk_type in (DONE, ERROR)
        try:
            await self._post_chunk_with_retries(
                client,
                f"{oe_url}/stream/chunk",
                chunk,
                owner_url=f"{owner_url}/stream/chunk" if owner_url else None,
                on_owner_failure=(
                    (lambda: self._discard_owner_callback_url(execution_id)) if owner_url else None
                ),
            )
        except Exception as e:
            if terminal:
                self._chunk_seq.pop(execution_id, None)
                raise
            logger.warning(
                "Failed to deliver %s chunk (seq=%d) to OE after retries: %s",
                chunk_type,
                seq,
                e,
            )
            return

        if terminal:
            self._chunk_seq.pop(execution_id, None)

    async def _emit_subagent_boundary(
        self,
        oe_url: str,
        execution_id: str,
        chunk_type: str,
        content_key: str,
        event_data: _SubagentBoundaryData,
    ) -> None:
        """Forward a subagent_start/subagent_end event to OE as a stream chunk."""
        # description/summary go into metadata, not content, so text accumulators
        # (CLI, UI) that append chunk.content without inspecting chunk_type don't
        # pick up structural boundary text as part of the agent's response.
        meta: dict[str, JsonValue] = {
            "source": str(event_data.get("source", "") or ""),
            "subagent_name": str(event_data.get("subagent_name", "") or ""),
            "tool_call_id": str(event_data.get("tool_call_id", "") or ""),
            content_key: str(event_data.get(content_key, "") or ""),
        }
        await self._send_stream_chunk(
            oe_url,
            execution_id,
            chunk_type=chunk_type,
            content="",
            metadata=meta,
        )

    async def _execute_via_agent_stream(
        self,
        agent: Any,
        ctx: RequestContext,
        agent_input: AgentInput,
        oe_url: str,
        execution_id: str,
    ) -> Union[InterruptResult, StreamingResult]:
        """Execute via BaseAgent.stream() and handle StreamEvents.

        This method iterates over the agent's streaming events and:
        - Streams "token" events to OE (with thinking token filtering)
        - Returns InterruptResult for "suspend" events (HITL)
        - Returns StreamingResult for "result" events (completion)

        Thinking tokens (``<think>...</think>``) are suppressed during
        streaming and stripped from the final content.

        Args:
            agent: BaseAgent instance (e.g., LangGraphBaseAgent)
            agent_input: Framework-neutral agent input
            oe_url: OE callback URL for streaming chunks
            execution_id: Execution ID for logging

        Returns:
            ``StreamingResult`` for normal completion (with sanitized
            ``content`` and raw ``messages`` for memory), or
            ``InterruptResult`` when the agent is suspended.
        """
        # Per-stream <think> filter state keyed by (source, tool_call_id).
        # source alone is insufficient: two parallel dispatches of the same-named
        # subagent share a source string but have distinct tool_call_ids, so their
        # thinking buffers must be isolated to prevent token cross-contamination.
        thinking_state: dict[tuple[str, str], tuple[str, bool]] = {}

        if not isinstance(agent_input.payload, dict):
            raise TypeError(f"Expected dict payload, got {type(agent_input.payload)}")

        terminal_outcome: Union[InterruptResult, StreamingResult, None] = None
        # Only stream creation and iterator steps run customer code; the
        # rest of the loop is platform-owned.
        with customer_origin_scope():
            stream_iter = agent.execute(ctx, agent_input).__aiter__()

        try:
            while True:
                # Once the SDK yields result/suspend, drive the iterator one more
                # step so async generators can finish post-yield cleanup naturally.
                # Only that drain pass is bounded here; pre-terminal execution is
                # still governed by the normal request/execution timeout.
                draining_after_terminal = terminal_outcome is not None
                timeout_s = (
                    AGENT_STREAM_TERMINAL_DRAIN_TIMEOUT_S if draining_after_terminal else None
                )
                try:
                    with customer_origin_scope():
                        next_event = stream_iter.__anext__()
                        # ``asyncio.timeout`` (same-task), not ``wait_for``
                        # (child Task): finalizing the async generator in a
                        # child Context raises ValueError on ContextVar tokens
                        # and leaks the transport into the parent Context.
                        if timeout_s is not None:
                            async with asyncio.timeout(timeout_s):
                                stream_event = await next_event
                        else:
                            stream_event = await next_event
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    if draining_after_terminal:
                        logger.warning(
                            "Timed out draining agent stream after terminal event "
                            "for execution %s after %.1fs",
                            execution_id,
                            AGENT_STREAM_TERMINAL_DRAIN_TIMEOUT_S,
                        )
                        break
                    raise
                except Exception:
                    if draining_after_terminal:
                        logger.warning(
                            "Unexpected exception during post-terminal drain "
                            "for execution %s, ignoring",
                            execution_id,
                            exc_info=True,
                        )
                        break
                    raise

                if draining_after_terminal:
                    logger.warning(
                        "Ignoring agent stream event %s after terminal event for execution %s",
                        getattr(stream_event, "event", type(stream_event).__name__),
                        execution_id,
                    )
                    break

                # Parser events carry only custom_event (no platform data): forward
                # as a CUSTOM_EVENT chunk and skip the platform handling below.
                if stream_event.custom_event is not None and stream_event.event == CUSTOM_EVENT:
                    await self._send_stream_chunk(
                        oe_url,
                        execution_id,
                        chunk_type=CUSTOM_EVENT,
                        custom_event=stream_event.custom_event,
                    )
                    continue

                if not isinstance(stream_event.data, dict):
                    raise TypeError(f"Expected dict event data, got {type(stream_event.data)}")
                event_data: dict[str, Any] = stream_event.data

                if stream_event.event == "token":
                    token_content = event_data.get("content", "")
                    source = str(event_data.get("source", "") or "")
                    tool_call_id = str(event_data.get("tool_call_id", "") or "")
                    if token_content:
                        state_key = (source, tool_call_id)
                        buf, inside = thinking_state.get(state_key, ("", False))
                        streamable, buf, inside = filter_thinking_tokens(token_content, buf, inside)
                        thinking_state[state_key] = (buf, inside)
                        if streamable:
                            await self._send_stream_chunk(
                                oe_url,
                                execution_id,
                                chunk_type=TEXT,
                                content=streamable,
                                metadata={
                                    "source": source,
                                    "tool_call_id": tool_call_id,
                                },
                            )

                elif stream_event.event == "subagent_start":
                    await self._emit_subagent_boundary(
                        oe_url,
                        execution_id,
                        chunk_type=SUBAGENT_START,
                        content_key="description",
                        event_data=cast(_SubagentBoundaryData, event_data),
                    )

                elif stream_event.event == "subagent_end":
                    await self._emit_subagent_boundary(
                        oe_url,
                        execution_id,
                        chunk_type=SUBAGENT_END,
                        content_key="summary",
                        event_data=cast(_SubagentBoundaryData, event_data),
                    )
                    source = str(event_data.get("source", "") or "")
                    tool_call_id = str(event_data.get("tool_call_id", "") or "")
                    thinking_state.pop((source, tool_call_id), None)

                elif stream_event.event == "suspend":
                    # The framework adapter owns all resume semantics. It places
                    # whatever it needs to resume (checkpoint id, function-call
                    # correlation, etc.) into an opaque ``metadata`` dict that the
                    # AER round-trips to the OE without inspecting.
                    raw_metadata = event_data.get("metadata")
                    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
                    terminal_outcome = InterruptResult(
                        suspend_payload=event_data.get("suspend_payload", {}),
                        interrupts=event_data.get("interrupts"),
                        resume_schema=event_data.get("resume_schema"),
                        metadata=metadata,
                        # `or []` coerces a missing key *or* an explicit None to
                        # an empty list, so an adapter emitting messages: null
                        # cannot fail InterruptResult validation and turn a
                        # suspend into a 500.
                        messages=event_data.get("messages") or [],
                    )
                    continue

                elif stream_event.event == "result":
                    content = strip_thinking(event_data.get("response", "") or "")
                    terminal_outcome = StreamingResult(
                        content=content,
                        messages=event_data.get("messages", []),
                        metadata=event_data.get("metadata"),
                    )
                    continue
        finally:
            await _aclose_stream_iter(stream_iter, execution_id)

        if terminal_outcome is not None:
            return terminal_outcome

        # Should not reach here - stream should yield result or suspend
        raise RuntimeError(
            f"Agent stream ended without result or suspend (execution_id={execution_id})"
        )

    async def _report_callback(
        self,
        oe_url: str,
        execution_id: str,
        status: str,
        suspend_generation: Optional[int] = None,
        result: Any = None,
        error: Optional[str] = None,
        suspend_reason: Optional[str] = None,
        suspend_context: Optional[Dict[str, Any]] = None,
        interrupts: Optional[list[PendingInterrupt]] = None,
        resume_schema: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Report execution status to OE."""
        callback_body = _executor_callback_body(
            execution_id,
            status,
            suspend_generation=suspend_generation,
            result=result,
            error=error,
            suspend_reason=suspend_reason,
            suspend_context=suspend_context,
            interrupts=interrupts,
            resume_schema=resume_schema,
            metadata=metadata,
        )
        await self._send_callback_body(oe_url, callback_body)

    async def _send_callback_body(
        self,
        oe_url: str,
        callback_body: dict[str, Any],
        *,
        owner_url: Optional[str] = None,
        owner_url_failure: Optional[OwnerUrlFailureState] = None,
    ) -> None:
        """Send one already-validated executor callback body to OE."""
        execution_id = str(callback_body.get("execution_id", "") or "")
        if owner_url_failure is not None:
            owner_url = owner_url if not owner_url_failure.failed else None
        else:
            owner_url = self._get_owner_callback_url(execution_id)

        def discard_owner() -> None:
            self._discard_owner_callback_url(execution_id)
            if owner_url_failure is not None:
                owner_url_failure.failed = True

        await self._callback_delivery.send(
            oe_url,
            callback_body,
            owner_url=owner_url,
            on_owner_failure=(discard_owner if owner_url else None),
        )

    async def _release_finished_session(self, request: ExecuteRequest) -> None:
        """Ask the OE to free this session's compute now that the turn is over.

        Runs as a FastAPI background task, i.e. after the /execute response has
        been sent, because the OE will kill this pod with zero grace. A retained
        terminal callback and pending STM writes are drained first for the same
        reason: either would otherwise die with the pod. Best-effort throughout
        - a failure just leaves the pods to the idle sweep.
        """
        if not await self._callback_delivery.wait_for_terminal(request.execution_id):
            return

        writer = getattr(self.runtime, "_memory_writer", None)
        if writer is not None:
            deadline = time.monotonic() + SESSION_FINISH_MEMORY_DRAIN_TIMEOUT_S
            while writer.pending_writes and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            if writer.pending_writes:
                logger.warning(
                    "Session finish: %d memory write(s) still pending after %.0fs; releasing anyway",
                    writer.pending_writes,
                    SESSION_FINISH_MEMORY_DRAIN_TIMEOUT_S,
                )
        url = f"{request.platform_api_url.rstrip('/')}/executions/{request.execution_id}/finish"
        try:
            client = await self._get_client(request.platform_api_url)
            await post_json_with_retries(client, url, {})
        except Exception as exc:
            logger.warning("Session finish request failed for %s: %s", request.execution_id, exc)
