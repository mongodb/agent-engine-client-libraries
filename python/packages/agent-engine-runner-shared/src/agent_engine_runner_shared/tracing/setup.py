"""
Tracing setup for Runner SDK.

Provides a simple interface to initialize OpenTelemetry with popular
AI/LLM instrumentors like OpenInference, exporting traces to MongoDB and,
when configured, OTLP.

When enabled, traces are written to:
1. MongoDB collection (if configured)
2. OTLP endpoint (if configured)
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
from typing import TYPE_CHECKING, Callable, Optional

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from .exporters import (
    ContentPolicyOTLPSpanExporter,
    MongoDBSpanExporter,
    get_content_capture_mode,
)

if TYPE_CHECKING:
    from pymongo.collection import Collection
    from pymongo.mongo_client import MongoClient

logger = logging.getLogger(__name__)

# Track if tracing has been set up.
_tracer_provider: Optional[TracerProvider] = None

# MongoDB trace-store exporter state. The store connection can fail at pod
# startup (e.g. a transient Atlas TLS error); without retry the pod permanently
# lost database tracing for its whole lifetime — a single warning, no recovery.
# `_mongodb_exporter_attached` tracks whether a MongoDB span
# processor is live; `_mongodb_degraded` marks a configured-but-unreachable
# store so /health can surface DEGRADED. The retry thread re-attempts the
# connection in the background and attaches the exporter on recovery.
_mongodb_exporter_attached: bool = False
_mongodb_degraded: bool = False
_retry_stop_event: Optional[threading.Event] = None
_retry_thread: Optional[threading.Thread] = None
# Client retained only when tracing OWNS it — a retry-recovery attach, or an
# explicit take_client_ownership=True at setup — so shutdown_tracing can close
# the pool after flushing the provider. A caller-provided collection is never
# retained: its client may be shared with application code. Mirrors the
# TypeScript `tracingMongoClient` lifecycle, which only closes self-created
# clients.
_tracing_mongo_client: Optional["MongoClient"] = None

# Resource attribute -> platform-injected env var. These are process-level
# identity (set once at pod startup), unlike execution.id/session.id which
# vary per request and are attached as span attributes instead (see
# ExecutionContextSpanProcessor below). Absent env vars are simply omitted —
# not every runtime mode has all of these populated.
#
# Producer status on the AER/Tool Pod (where setup_tracing() actually runs),
# verified against ECP/OE/operator source rather than assumed:
# - org_id/project_id/workspace_id/runtime_mode: live today — ORG_ID/
#   PROJECT_ID/APP_ID/RUNNER_MODE are injected onto both the AER and Tool
#   containers by ECP's buildIdentityEnvWithAtlasPatch (controller.go) and,
#   in RPC/VM mode, by OE's buildWorkload (rpc_reservation.go).
# - deployment_id/agent_id: forward-provisioned, not yet live. No producer
#   sets DEPLOYMENT_ID or AGENT_ID anywhere in agentic-operator/,
#   executor-control-plane/, or orchestration-engine/ today — these two
#   attributes will be absent from every span until a companion ECP/operator
#   change wires them in.
# - environment: also not yet live, despite AGENT_ENGINE_ENVIRONMENT being a real,
#   actively-set var elsewhere — the operator's applyMAASEnv
#   (tenantenvironment_controller.go) stamps it only onto OE's own pod, never
#   onto the AER/Tool Pod this code runs in. The name is still correct (it's
#   OE/the operator's external contract, not ours to rename) — it's simply
#   not wired to reach this process yet.
_PLATFORM_RESOURCE_ENV_VARS: tuple[tuple[str, str], ...] = (
    ("agentic_platform.org_id", "ORG_ID"),
    ("agentic_platform.project_id", "PROJECT_ID"),
    ("agentic_platform.workspace_id", "APP_ID"),
    ("agentic_platform.deployment_id", "DEPLOYMENT_ID"),
    ("agentic_platform.agent_id", "AGENT_ID"),
    ("agentic_platform.runtime_mode", "RUNNER_MODE"),
    (
        "agentic_platform.environment",  # open-source-refs:ignore — data contract
        "AGENT_ENGINE_ENVIRONMENT",
    ),
)


def _build_resource_attributes(service_name: str) -> dict[str, str]:
    attributes: dict[str, str] = {"service.name": service_name}
    for resource_key, env_var in _PLATFORM_RESOURCE_ENV_VARS:
        value = os.getenv(env_var)
        if value:
            attributes[resource_key] = value
    return attributes


def _otlp_endpoint_configured() -> bool:
    """Whether an OTLP endpoint was explicitly configured.

    ``OTLPSpanExporter()`` defaults to ``http://localhost:4318`` when no
    endpoint env var is set, which would silently start exporting to a
    collector that doesn't exist in this cell architecture. Gate on explicit
    configuration so "no endpoint configured" truly means "no OTLP traffic."
    """
    return bool(
        os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    )


class ExecutionContextSpanProcessor(SpanProcessor):
    """Attach platform execution context to spans created during AER requests."""

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        from agent_engine_runner_shared.context import (
            get_current_execution_id,
            get_current_session_id,
            get_current_workspace_id,
        )

        session_id = get_current_session_id()
        attrs = {
            "execution.id": get_current_execution_id(),
            "session.id": session_id,
            # thread.id mirrors session.id — the AER adapter uses session_id
            # as the LangGraph checkpoint thread_id, and some OTLP-consuming
            # tools key off "thread.id" rather than "session.id".
            "thread.id": session_id,
            "workspace.id": get_current_workspace_id(),
        }
        for key, value in attrs.items():
            if value:
                span.set_attribute(key, value)

    def on_end(self, span: ReadableSpan) -> None:
        return

    def shutdown(self) -> None:
        return

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def setup_tracing(
    service_name: str = "runner-sdk",
    mongodb_collection: Optional["Collection"] = None,
    *,
    take_client_ownership: bool = False,
) -> None:
    """Set up OpenTelemetry tracing with OpenInference.

    OpenInference instrumentors will automatically
    send spans to exporters once this is called.

    Args:
        service_name: Name of your service (appears in traces).
        mongodb_collection: MongoDB collection for traces (optional).
        take_client_ownership: Whether shutdown_tracing() closes the MongoClient
            behind mongodb_collection. Default False — the caller keeps the
            client, which may be shared with application code, and shutdown
            leaves it open. Pass True only when the client was created solely
            for tracing (as TenantRuntime._setup_tracing does); it is then
            closed after the provider flush. Clients attached later by
            attach_mongodb_tracing() (trace-store retry recovery) are always
            owned — see its docstring.

    Example:
        setup_tracing(service_name="my-agent")

        # With MongoDB (the client stays yours — shutdown_tracing won't close it):
        setup_tracing(
            service_name="my-agent",
            mongodb_collection=client["mydb"]["traces"],
        )
    """
    global _tracer_provider
    global _mongodb_exporter_attached, _mongodb_degraded, _tracing_mongo_client

    if _tracer_provider is not None:
        logger.debug("Tracing already initialized")
        return

    # Create tracer provider.
    resource = Resource.create(_build_resource_attributes(service_name))
    provider = TracerProvider(resource=resource)
    _tracer_provider = provider

    provider.add_span_processor(ExecutionContextSpanProcessor())

    # Add the canonical customer trace-store exporter when configured.
    if mongodb_collection is not None:
        provider.add_span_processor(BatchSpanProcessor(MongoDBSpanExporter(mongodb_collection)))
        _mongodb_exporter_attached = True
        if take_client_ownership:
            _tracing_mongo_client = mongodb_collection.database.client
        _mongodb_degraded = False
        logger.info(
            f"Tracing: MongoDB output to {mongodb_collection.database.name}.{mongodb_collection.name}"
        )

    # Set as global tracer provider before attempting the OTLP block below —
    # a broken/misconfigured OTLP setup (missing extra, bad endpoint) must not
    # prevent MongoDB from receiving spans too.
    trace.set_tracer_provider(provider)

    # Add OTLP exporter (if an endpoint was explicitly configured). It is
    # additive and never replaces MongoDB. Content-capture
    # redaction (AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE) applies only to this path.
    # Guarded: an accidentally-set OTEL_EXPORTER_OTLP_* env var (or a missing
    # `tracing` extra) must not take down tracing setup entirely.
    if _otlp_endpoint_configured():
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            content_capture_mode = get_content_capture_mode()
            if content_capture_mode == "full":
                logger.warning(
                    "AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE=full — OTLP export includes "
                    "unredacted prompt/completion/tool content. This should "
                    "only be set intentionally (e.g. local debugging)."
                )
            otlp_exporter = ContentPolicyOTLPSpanExporter(
                OTLPSpanExporter(), content_capture_mode=content_capture_mode
            )
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"Tracing: OTLP output enabled (content-capture={content_capture_mode})")
        except Exception:
            logger.warning(
                "Failed to set up OTLP exporter — OTLP export disabled, MongoDB tracing unaffected",
                exc_info=True,
            )

    # Restrict the global propagator to W3C trace-context only. The API's
    # default (from OTEL_PROPAGATORS, "tracecontext,baggage") also propagates
    # Baggage, which would leak arbitrary unredacted key/value pairs onto
    # every outbound A2A call once something starts writing to it.
    set_global_textmap(TraceContextTextMapPropagator())

    # Enable framework-specific instrumentation if registered.
    _run_instrumentor()

    # Propagate W3C trace context (traceparent/tracestate) on every outbound
    # httpx request process-wide, so SDK -> OE calls carry the active span
    # without hand-rolling per-call-site header injection.
    _instrument_httpx()

    logger.info(f"Tracing enabled for '{service_name}'")


def _run_instrumentor() -> None:
    """Run the registered framework instrumentor, if any."""
    from agent_engine_runner_shared.hooks import get_instrumentor

    instrumentor = get_instrumentor()
    if instrumentor is not None:
        try:
            instrumentor()
            logger.info("Framework instrumentation enabled")
        except Exception:
            logger.warning("Framework instrumentor failed — tracing disabled", exc_info=True)
    else:
        logger.warning(
            "No instrumentor hook registered. "
            "Framework SDK should call register_instrumentor() to enable tracing."
        )


def _instrument_httpx() -> None:
    """Instrument httpx globally so every client injects traceparent/tracestate headers."""
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
        logger.info("httpx instrumentation enabled (traceparent propagation)")
    except Exception:
        logger.warning(
            "httpx instrumentation failed — trace headers will not propagate", exc_info=True
        )


def attach_mongodb_tracing(collection: "Collection") -> bool:
    """Attach the MongoDB span exporter to the active tracer provider.

    Called from the background retry loop once a failed trace-store connection
    recovers. Idempotent — a no-op if a MongoDB exporter is already
    live. New spans created after this call route to the store; spans emitted
    while the store was unreachable are not retroactively recovered.

    Returns True when ``collection`` was wired in as the export target — on
    success this module takes ownership of the collection's client and
    ``shutdown_tracing`` closes it. False means the call was a no-op (already
    attached, or provider gone) and the caller still owns the client.
    """
    global _mongodb_exporter_attached, _mongodb_degraded, _tracing_mongo_client
    if _mongodb_exporter_attached:
        return False
    provider = _tracer_provider
    if provider is None:
        logger.warning("Cannot attach MongoDB tracing: tracer provider not initialized")
        return False
    provider.add_span_processor(BatchSpanProcessor(MongoDBSpanExporter(collection)))
    _mongodb_exporter_attached = True
    _tracing_mongo_client = collection.database.client
    _mongodb_degraded = False
    logger.info(
        f"Tracing: MongoDB output attached to {collection.database.name}.{collection.name} "
        "(trace store recovered)"
    )
    return True


def tracing_status() -> dict[str, str]:
    """Snapshot of the database trace-store exporter for /health.

    Returns ``{"database_exporter": "attached" | "degraded" | "disabled"}``:

    * ``attached`` — a MongoDB span processor is live.
    * ``degraded`` — a store URI was configured but the startup connection
      failed and the background retry is in progress.
    * ``disabled`` — no store URI was configured; this is an intentional
      configuration, not a degradation. OTLP may still be configured.
    """
    if _mongodb_exporter_attached:
        return {"database_exporter": "attached"}
    if _mongodb_degraded:
        return {"database_exporter": "degraded"}
    return {"database_exporter": "disabled"}


# The run is [^"'] (any char except quotes): it still crosses embedded
# newlines in a password (a --stdin-set secret can carry one), but stops at
# a JSON string boundary — a credential echoed inside a compact JSON body
# must not fold later siblings into the mask.
_MONGO_URI_USERINFO_PATTERN = re.compile(
    r"(mongodb(?:\+srv)?://)((?:(?!mongodb(?:\+srv)?://)[^\"'])*)@(?=[^\s@/?#\"'][^\s@?\"']*(?:[/?#:\s\"']|$))",
    re.IGNORECASE,
)


def scrub_credentials(message: str) -> str:
    """Mask the userinfo of any ``mongodb://`` / ``mongodb+srv://`` URI in an
    error message. Driver parse/connect errors echo the raw connection string,
    and the trace-store URI carries customer database credentials.

    The userinfo run is GREEDY up to the LAST '@' that a host-shaped token
    follows, so passwords containing an unescaped '@', space, or '/' are still
    fully masked — the old ``://[^@\\s]+@`` pattern stopped at the first '@'
    and could not cross whitespace, leaking the password tail (or the whole
    userinfo) to the centralized log sink. The run is tempered so it never
    crosses into a second URI's scheme; when message text after the URI
    contains its own '@' the mask may extend to it — over-redaction is the
    fail-closed direction, a leak is not recoverable. Keep in sync with
    ``scrubCredentials`` in the TypeScript ``agent-engine-runner-shared/src/tracing/setup.ts``.
    """
    return _MONGO_URI_USERINFO_PATTERN.sub(r"\1***@", message)


def _get_retry_interval_seconds() -> float:
    """Retry interval (seconds) for the background trace-store reconnection.

    A positive finite AGENTIC_TRACE_STORE_RETRY_INTERVAL_SECONDS wins;
    anything else (unset, unparsable, non-positive, inf/nan) falls back to
    30s — the same semantics as the TypeScript getRetryIntervalSeconds().
    """
    try:
        raw = float(os.getenv("AGENTIC_TRACE_STORE_RETRY_INTERVAL_SECONDS", ""))
    except ValueError:
        return 30.0
    return raw if math.isfinite(raw) and raw > 0 else 30.0


def start_mongodb_tracing_retry(
    connect: "Callable[[], Optional[Collection]]",
    interval: Optional[float] = None,
) -> None:
    """Retry the trace-store connection in the background.

    ``connect`` builds and pings a fresh MongoClient, returning a live
    Collection on success or raising on failure (the retry loop swallows the
    exception and tries again). The first attempt is delayed by ``interval``
    seconds (default: AGENTIC_TRACE_STORE_RETRY_INTERVAL_SECONDS, or 30s) —
    the startup attempt already failed, so an immediate retry would just
    re-hit the same transient condition. The thread is a daemon, so it
    never blocks process exit; ``shutdown_tracing`` signals it to stop and
    joins it (bounded) before tearing down the provider. No-op if the
    exporter is already attached or a retry is already running.
    """
    global _retry_stop_event, _retry_thread, _mongodb_degraded
    if _mongodb_exporter_attached:
        return
    if _retry_thread is not None and _retry_thread.is_alive():
        return
    if interval is None:
        interval = _get_retry_interval_seconds()
    _mongodb_degraded = True
    _retry_stop_event = threading.Event()
    _retry_thread = threading.Thread(
        target=_retry_loop,
        args=(connect, interval, _retry_stop_event),
        name="agentic-trace-store-retry",
        daemon=True,
    )
    _retry_thread.start()


def _close_collection_client(collection: "Collection") -> None:
    """Best-effort close of the MongoClient backing an unclaimed Collection."""
    try:
        collection.database.client.close()
    except Exception:  # noqa: BLE001 - best-effort cleanup
        pass


def _retry_loop(
    connect: "Callable[[], Optional[Collection]]",
    interval: float,
    stop_event: threading.Event,
) -> None:
    while not stop_event.wait(interval):
        if _mongodb_exporter_attached:
            return
        try:
            collection = connect()
        except Exception as e:  # noqa: BLE001 - retry on any connection failure
            logger.debug("Trace store retry failed: %s", scrub_credentials(str(e)))
            continue
        # shutdown_tracing() may have raced the blocking connect. If it did,
        # close the client this attempt opened rather than attach to a
        # provider that is gone — or replaced by a later setup_tracing().
        if stop_event.is_set():
            if collection is not None:
                _close_collection_client(collection)
            return
        if collection is not None:
            if not attach_mongodb_tracing(collection):
                # Attach skipped (attached concurrently, or provider gone):
                # nothing else owns this attempt's client — close it.
                _close_collection_client(collection)
            return


def shutdown_tracing() -> None:
    """Shutdown tracing and flush pending spans.

    Closes the trace-store MongoClient only when tracing owns it
    (take_client_ownership=True at setup, or a retry-attached client).
    A caller-owned client passed to setup_tracing() stays open.
    """
    global _tracer_provider, _retry_stop_event, _retry_thread
    global _mongodb_exporter_attached, _mongodb_degraded, _tracing_mongo_client

    # Signal the retry thread to stop, then wait (bounded) for it to exit
    # before tearing down the provider it attaches exporters to. The loop
    # re-checks the stop event after a blocking connect, so it exits promptly
    # unless the connect itself is still blocked in the network — don't hang
    # shutdown on that case: the thread is a daemon and closes the client it
    # opened when it eventually wakes. The reference is kept while the thread
    # is alive so a later start_mongodb_tracing_retry() no-ops instead of
    # starting a duplicate loop against a half-torn-down provider.
    if _retry_stop_event is not None:
        _retry_stop_event.set()
        _retry_stop_event = None
    thread = _retry_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)
    if thread is None or not thread.is_alive():
        _retry_thread = None

    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        _tracer_provider = None
        _uninstrument_httpx()
        logger.info("Tracing shutdown complete")

    # Close the store client only after the provider flush — the
    # BatchSpanProcessor's final export still needs the pool.
    if _tracing_mongo_client is not None:
        try:
            _tracing_mongo_client.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
        _tracing_mongo_client = None

    # Reset store-exporter state so a fresh setup_tracing() in the same
    # process (e.g. tests) starts clean.
    _mongodb_exporter_attached = False
    _mongodb_degraded = False


def _uninstrument_httpx() -> None:
    """Undo httpx instrumentation, if it was applied."""
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        instrumentor = HTTPXClientInstrumentor()
        if instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
    except Exception:
        logger.warning("httpx uninstrumentation failed", exc_info=True)


def get_current_trace_context() -> tuple[Optional[str], Optional[str]]:
    """
    Get current trace_id and span_id from the active span.

    Returns:
        Tuple of (trace_id, span_id) or (None, None) if no active span
    """
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        ctx = span.get_span_context()
        return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")
    return None, None


def get_tracer(name: str = "runner-sdk") -> trace.Tracer:
    """
    Get a tracer for creating custom spans.

    Args:
        name: Name of the tracer (usually module name)

    Returns:
        OpenTelemetry Tracer instance
    """
    return trace.get_tracer(name)
