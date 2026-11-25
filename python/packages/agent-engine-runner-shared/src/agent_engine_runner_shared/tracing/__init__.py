"""
Tracing module for Runner SDK.

Uses OpenInference for automatic LLM instrumentation.
Spans are sent to the Orchestration Engine trace endpoint when ``OE_URL`` is
set, and additionally to OTLP when an endpoint is configured.

Usage:
    from agent_engine_runner_shared.tracing import setup_tracing, get_tracer, get_current_trace_context

    setup_tracing(service_name="my-agent")

    # OTLP export is enabled by setting OTEL_EXPORTER_OTLP_ENDPOINT or
    # OTEL_EXPORTER_OTLP_TRACES_ENDPOINT before calling setup_tracing() — no
    # code change required. AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE controls whether
    # OTLP spans carry full prompt/completion content or metadata only
    # (default: metadata-only).

    # Create a custom span within the current trace:
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("my-operation"):
        ...

    # Read the active trace/span IDs for log correlation:
    trace_id, span_id = get_current_trace_context()
"""

from .exporters import (
    ContentPolicyOTLPSpanExporter,
    PlatformSpanExporter,
    get_content_capture_mode,
)
from .setup import (
    force_flush_tracing,
    get_current_trace_context,
    get_tracer,
    scrub_credentials,
    setup_tracing,
    shutdown_tracing,
    tracing_status,
)

__all__ = [
    "setup_tracing",
    "shutdown_tracing",
    "force_flush_tracing",
    "tracing_status",
    "get_current_trace_context",
    "get_tracer",
    "get_content_capture_mode",
    "scrub_credentials",
    "PlatformSpanExporter",
    "ContentPolicyOTLPSpanExporter",
]
