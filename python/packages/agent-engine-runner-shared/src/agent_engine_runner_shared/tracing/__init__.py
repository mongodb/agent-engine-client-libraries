"""
Tracing module for Runner SDK.

Uses OpenInference for automatic LLM instrumentation.
Traces are written to MongoDB when a collection is configured and additionally
to OTLP when an endpoint is configured. Local JSONL export is available only
through explicit use of ``JSONLSpanExporter``.

Usage:
    from agent_engine_runner_shared.tracing import setup_tracing, get_tracer, get_current_trace_context

    # With MongoDB (the canonical customer trace store)
    setup_tracing(service_name="my-agent", mongodb_collection=collection)

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
    JSONLSpanExporter,
    MongoDBSpanExporter,
    get_content_capture_mode,
    get_trace_path,
)
from .setup import (
    attach_mongodb_tracing,
    get_current_trace_context,
    get_tracer,
    scrub_credentials,
    setup_tracing,
    shutdown_tracing,
    start_mongodb_tracing_retry,
    tracing_status,
)

__all__ = [
    "setup_tracing",
    "shutdown_tracing",
    "attach_mongodb_tracing",
    "start_mongodb_tracing_retry",
    "tracing_status",
    "get_current_trace_context",
    "get_tracer",
    "get_trace_path",
    "get_content_capture_mode",
    "scrub_credentials",
    "JSONLSpanExporter",
    "MongoDBSpanExporter",
    "ContentPolicyOTLPSpanExporter",
]
