"""
Tracing setup for Runner SDK.

Provides a simple interface to initialize OpenTelemetry with popular
AI/LLM instrumentors like OpenInference.

When enabled, traces are sent to:
1. The Orchestration Engine trace endpoint (when OE_URL is set)
2. OTLP endpoint (if configured)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from .conversation import mark_recorded_by_caller
from .exporters import (
    ContentPolicyOTLPSpanExporter,
    PlatformSpanExporter,
    get_content_capture_mode,
)

logger = logging.getLogger(__name__)

# Track if tracing has been set up.
_tracer_provider: Optional[TracerProvider] = None

# Retained so /health can report recent export results and shutdown can reset it.
_platform_exporter: Optional[PlatformSpanExporter] = None

# Explicit BatchSpanProcessor sizing for the platform exporter. A larger queue
# than the OTel default absorbs the burst of spans emitted as a turn finishes,
# and the 2s schedule delay bounds how long a span waits before export.
_PLATFORM_BSP_MAX_QUEUE_SIZE = 2048
_PLATFORM_BSP_MAX_EXPORT_BATCH_SIZE = 512
_PLATFORM_BSP_SCHEDULE_DELAY_MILLIS = 2000

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

        mark_recorded_by_caller(span)

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


def setup_tracing(service_name: str = "runner-sdk") -> None:
    """Set up OpenTelemetry tracing with OpenInference.

    OpenInference instrumentors will automatically
    send spans to exporters once this is called.

    Args:
        service_name: Name of your service (appears in traces).

    Example:
        setup_tracing(service_name="my-agent")
    """
    global _tracer_provider

    if _tracer_provider is not None:
        logger.debug("Tracing already initialized")
        return

    # Create tracer provider.
    resource = Resource.create(_build_resource_attributes(service_name))
    provider = TracerProvider(resource=resource)
    _tracer_provider = provider

    provider.add_span_processor(ExecutionContextSpanProcessor())

    oe_url = os.getenv("OE_URL", "").strip()
    otlp_configured = _otlp_endpoint_configured()
    if oe_url:
        _attach_platform_exporter(provider, oe_url)
    elif not otlp_configured:
        logger.warning("Tracing: no trace destination configured (OE_URL unset, no OTLP endpoint)")

    # Set as global tracer provider before attempting the OTLP block below —
    # a broken/misconfigured OTLP setup (missing extra, bad endpoint) must not
    # prevent the platform exporter from receiving spans too.
    trace.set_tracer_provider(provider)

    # Add OTLP exporter (if an endpoint was explicitly configured). It is
    # additive to the platform exporter. Content-capture
    # redaction (AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE) applies only to this path.
    # Guarded: an accidentally-set OTEL_EXPORTER_OTLP_* env var (or a missing
    # `tracing` extra) must not take down tracing setup entirely.
    if otlp_configured:
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
                "Failed to set up OTLP exporter — OTLP export disabled, platform tracing unaffected",
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


def _attach_platform_exporter(provider: TracerProvider, oe_url: str) -> None:
    """Install the OE platform trace exporter on ``provider``."""
    global _platform_exporter
    exporter = PlatformSpanExporter(oe_url)
    provider.add_span_processor(
        BatchSpanProcessor(
            exporter,
            max_queue_size=_PLATFORM_BSP_MAX_QUEUE_SIZE,
            max_export_batch_size=_PLATFORM_BSP_MAX_EXPORT_BATCH_SIZE,
            schedule_delay_millis=_PLATFORM_BSP_SCHEDULE_DELAY_MILLIS,
        )
    )
    _platform_exporter = exporter
    logger.info("Tracing: platform exporter -> %s", exporter.url)


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


def tracing_status() -> dict[str, str]:
    """Snapshot of the platform trace exporter for /health.

    ``platform_exporter`` is ``attached`` (spans are sent to the OE),
    ``degraded`` (the most recent export did not fully succeed), or
    ``disabled`` (OE_URL unset; OTLP may still be configured).
    """
    if _platform_exporter is None:
        platform = "disabled"
    elif _platform_exporter.last_error is not None:
        platform = "degraded"
    else:
        platform = "attached"
    return {"platform_exporter": platform}


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
    and connection URIs carry customer database credentials.

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


def shutdown_tracing() -> None:
    global _tracer_provider, _platform_exporter

    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        _tracer_provider = None
        _uninstrument_httpx()
        logger.info("Tracing shutdown complete")

    # Reset exporter state so a fresh setup_tracing() starts clean.
    _platform_exporter = None


def force_flush_tracing(timeout_millis: int = 5000) -> bool:
    """Flush pending spans to their exporters, bounded by ``timeout_millis``.

    Called from the session-finish path before the OE releases the pod, so a
    turn's spans are not lost to a zero-grace teardown. Returns True when
    tracing is not initialized (nothing to flush) or the flush completed within
    the timeout; False when it timed out or raised.
    """
    provider = _tracer_provider
    if provider is None:
        return True
    try:
        return provider.force_flush(timeout_millis)
    except Exception:  # noqa: BLE001 - best-effort flush
        logger.debug("Tracing force_flush failed", exc_info=True)
        return False


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
