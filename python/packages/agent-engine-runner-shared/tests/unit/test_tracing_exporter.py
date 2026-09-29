"""
Unit tests for the span exporters.

Tests that OpenTelemetry spans are correctly exported to:
- JSONL files (development/debugging)
- MongoDB collection (production)
- OTLP (opt-in, content-capture redacted)
"""

import http.server
import json
import sys
import threading
from unittest.mock import MagicMock, Mock

import httpx
import pytest
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.propagate import get_global_textmap, set_global_textmap
from opentelemetry.sdk.trace import Event, ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode, TraceFlags
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

import agent_engine_runner_shared.tracing.setup as tracing_setup
from agent_engine_runner_shared import hooks
from agent_engine_runner_shared.context import clear_execution_context, set_execution_context
from agent_engine_runner_shared.tls_client import (
    create_async_httpx_client_with_tls,
    create_httpx_client_with_tls,
)
from agent_engine_runner_shared.tracing import (
    ContentPolicyOTLPSpanExporter,
    JSONLSpanExporter,
    MongoDBSpanExporter,
)
from agent_engine_runner_shared.tracing.exporters import get_content_capture_mode
from agent_engine_runner_shared.tracing.setup import (
    ExecutionContextSpanProcessor,
    _build_resource_attributes,
    _run_instrumentor,
)


@pytest.fixture
def mock_span():
    """Create a mock OpenTelemetry span."""
    context = SpanContext(
        trace_id=0x12345678123456781234567812345678,
        span_id=0x1234567812345678,
        is_remote=False,
        trace_flags=TraceFlags.SAMPLED,
    )

    span = MagicMock(spec=ReadableSpan)
    span.get_span_context.return_value = context
    span.parent = None
    span.name = "test_span"
    span.kind = SpanKind.INTERNAL
    span.start_time = 1000
    span.end_time = 2000
    span.status = Status(StatusCode.OK)
    span.attributes = {"key": "value"}
    span.resource.attributes = {"service.name": "test_service"}
    span.events = []

    return span


# =============================================================================
# JSONL Exporter Tests
# =============================================================================


def test_jsonl_exporter_writes_span(mock_span, tmp_path):
    """Test that JSONL exporter writes span to file."""
    trace_file = tmp_path / "traces.jsonl"
    exporter = JSONLSpanExporter(path=trace_file)

    result = exporter.export([mock_span])

    assert result.name == "SUCCESS"
    assert trace_file.exists()

    with open(trace_file) as f:
        doc = json.loads(f.readline())

    assert doc["trace_id"] == "12345678123456781234567812345678"
    assert doc["span_id"] == "1234567812345678"
    assert doc["name"] == "test_span"
    assert doc["kind"] == "INTERNAL"
    assert doc["attributes"]["key"] == "value"


def test_jsonl_exporter_appends(mock_span, tmp_path):
    """Test that JSONL exporter appends to existing file."""
    trace_file = tmp_path / "traces.jsonl"
    exporter = JSONLSpanExporter(path=trace_file)

    exporter.export([mock_span])
    mock_span.name = "span_2"
    exporter.export([mock_span])

    with open(trace_file) as f:
        lines = f.readlines()

    assert len(lines) == 2
    assert json.loads(lines[0])["name"] == "test_span"
    assert json.loads(lines[1])["name"] == "span_2"


def test_jsonl_exporter_handles_empty(tmp_path):
    """Test that exporter handles empty span list."""
    trace_file = tmp_path / "traces.jsonl"
    exporter = JSONLSpanExporter(path=trace_file)

    result = exporter.export([])

    assert result.name == "SUCCESS"
    assert not trace_file.exists()


# =============================================================================
# MongoDB Exporter Tests
# =============================================================================


def test_mongodb_exporter_inserts_span(mock_span):
    """Test that MongoDB exporter inserts span."""
    collection = MagicMock()
    exporter = MongoDBSpanExporter(collection)

    result = exporter.export([mock_span])

    assert result.name == "SUCCESS"
    assert collection.insert_many.called

    docs = collection.insert_many.call_args[0][0]
    assert len(docs) == 1
    assert docs[0]["trace_id"] == "12345678123456781234567812345678"
    assert docs[0]["name"] == "test_span"


def test_mongodb_exporter_does_not_hoist_untrusted_root_context(mock_span):
    """Root observability fields are added by OE from executions, not span attributes."""
    mock_span.attributes = {
        "session.id": "child-session",
        "execution.id": "child-exec",
        "root.session.id": "forged-root-session",
        "root.execution.id": "forged-root-exec",
        "workspace.id": "forged-workspace",
        "org.id": "forged-org",
        "project.id": "forged-project",
    }
    collection = MagicMock()
    exporter = MongoDBSpanExporter(collection)

    result = exporter.export([mock_span])

    assert result.name == "SUCCESS"
    docs = collection.insert_many.call_args[0][0]
    assert docs[0]["session_id"] == "child-session"
    assert docs[0]["execution_id"] == "child-exec"
    assert "root_session_id" not in docs[0]
    assert "root_execution_id" not in docs[0]
    assert "workspace_id" not in docs[0]
    assert "org_id" not in docs[0]
    assert "project_id" not in docs[0]


def test_execution_context_span_processor_sets_correlation_attributes():
    """Span processor adds execution/session attributes used for OE-side joining."""
    context = SpanContext(
        trace_id=0x99999999999999999999999999999999,
        span_id=0x9999999999999999,
        is_remote=False,
        trace_flags=TraceFlags.SAMPLED,
    )
    span = MagicMock()
    span.get_span_context.return_value = context
    tokens = set_execution_context(
        "child-exec",
        wrapper=None,
        oe_url="http://oe:8000",
        session_id="child-session",
        workspace_id="ws-child",
    )
    try:
        ExecutionContextSpanProcessor().on_start(span)
    finally:
        clear_execution_context(tokens)

    span.set_attribute.assert_any_call("execution.id", "child-exec")
    span.set_attribute.assert_any_call("session.id", "child-session")
    # thread.id mirrors session.id — the AER adapter uses session_id as the
    # LangGraph checkpoint thread_id, and some OTLP consumers key
    # off "thread.id".
    span.set_attribute.assert_any_call("thread.id", "child-session")
    span.set_attribute.assert_any_call("workspace.id", "ws-child")


def test_mongodb_exporter_handles_empty():
    """Test that exporter handles empty span list."""
    collection = MagicMock()
    exporter = MongoDBSpanExporter(collection)

    result = exporter.export([])

    assert result.name == "SUCCESS"
    assert not collection.insert_many.called


def test_mongodb_exporter_handles_error(mock_span):
    """Test that exporter handles MongoDB errors gracefully."""
    collection = MagicMock()
    collection.insert_many.side_effect = Exception("Connection failed")
    exporter = MongoDBSpanExporter(collection)

    result = exporter.export([mock_span])

    assert result.name == "FAILURE"


# =============================================================================
# Instrumentor Hook Tests
# =============================================================================


class TestRunInstrumentor:
    @pytest.fixture(autouse=True)
    def _reset_hooks(self):
        yield
        hooks.reset_hooks()

    def test_calls_registered_instrumentor(self):
        mock_fn = Mock()
        hooks.register_instrumentor(mock_fn)
        _run_instrumentor()
        mock_fn.assert_called_once()

    def test_logs_warning_when_no_hook(self, caplog):
        _run_instrumentor()
        assert "No instrumentor hook registered" in caplog.text

    def test_instrumentor_failure_does_not_raise(self, caplog):
        hooks.register_instrumentor(Mock(side_effect=RuntimeError("boom")))
        _run_instrumentor()
        assert "Framework instrumentor failed" in caplog.text


# =============================================================================
# Resource Attribute Tests
# =============================================================================


def test_build_resource_attributes_includes_configured_agentic_platform_fields(monkeypatch):
    monkeypatch.setenv("ORG_ID", "org-1")
    monkeypatch.setenv("PROJECT_ID", "proj-1")
    monkeypatch.setenv("APP_ID", "app-1")
    monkeypatch.setenv("DEPLOYMENT_ID", "dep-1")
    monkeypatch.setenv("AGENT_ID", "agent-1")
    monkeypatch.setenv("RUNNER_MODE", "aer")
    monkeypatch.setenv("AGENT_ENGINE_ENVIRONMENT", "qa")

    attrs = _build_resource_attributes("my-service")

    assert attrs == {
        "service.name": "my-service",
        "agentic_platform.org_id": "org-1",
        "agentic_platform.project_id": "proj-1",
        "agentic_platform.workspace_id": "app-1",
        "agentic_platform.deployment_id": "dep-1",
        "agentic_platform.agent_id": "agent-1",
        "agentic_platform.runtime_mode": "aer",
        "agentic_platform.environment": "qa",
    }


def test_build_resource_attributes_omits_unset_fields(monkeypatch):
    for env_var in (
        "ORG_ID",
        "PROJECT_ID",
        "APP_ID",
        "DEPLOYMENT_ID",
        "AGENT_ID",
        "RUNNER_MODE",
        "AGENT_ENGINE_ENVIRONMENT",
    ):
        monkeypatch.delenv(env_var, raising=False)

    attrs = _build_resource_attributes("my-service")

    assert attrs == {"service.name": "my-service"}


# =============================================================================
# Content-Capture Redaction Tests
# =============================================================================


@pytest.fixture
def content_bearing_span():
    """A span carrying prompt/completion/secret-shaped attributes, as OpenInference
    instrumentors would emit them."""
    context = SpanContext(
        trace_id=0x1,
        span_id=0x1,
        is_remote=False,
        trace_flags=TraceFlags.SAMPLED,
    )
    span = MagicMock(spec=ReadableSpan)
    span.get_span_context.return_value = context
    span.context = context
    span.parent = None
    span.name = "llm_call"
    span.kind = SpanKind.INTERNAL
    span.start_time = 1000
    span.end_time = 2000
    span.status = Status(StatusCode.OK)
    span.resource.attributes = {"service.name": "test_service"}
    span.events = [
        Event(
            name="exception",
            attributes={
                "exception.type": "ValidationError",
                "tool.parameters": '{"query": "secret"}',
            },
            timestamp=1500,
        )
    ]
    span.links = []
    span.instrumentation_scope = None
    span.attributes = {
        "openinference.span.kind": "LLM",
        "llm.model_name": "gpt-4",
        "llm.input_messages.0.message.content": "what is the customer's SSN?",
        "llm.output_messages.0.message.content": "I can't share that.",
        "input.value": "what is the customer's SSN?",
        "output.value": "I can't share that.",
        "tool.parameters": '{"query": "secret"}',
        "retrieval.documents.0.document.content": "confidential doc text",
        "embedding.embeddings.0.embedding.vector": [0.1, 0.2],
        "api_key": "sk-should-not-leak",
        "http.url": "https://internal-oe.agentic-platform.local/v1/invoke?token=abc123",
        "server.address": "internal-oe.agentic-platform.local",
        "net.peer.name": "10.0.1.42",
        "session.id": "session-1",
    }
    return span


def test_content_capture_mode_defaults_to_metadata_only(monkeypatch):
    monkeypatch.delenv("AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE", raising=False)
    assert get_content_capture_mode() == "metadata-only"


def test_content_capture_mode_fails_closed_on_unrecognized_value(monkeypatch):
    monkeypatch.setenv("AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE", "yolo")
    assert get_content_capture_mode() == "metadata-only"


def test_content_capture_mode_full_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE", "FULL")
    assert get_content_capture_mode() == "full"


def test_otlp_exporter_redacts_content_by_default(monkeypatch, content_bearing_span):
    monkeypatch.delenv("AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE", raising=False)
    inner = MagicMock(spec=SpanExporter)
    inner.export.return_value = SpanExportResult.SUCCESS
    exporter = ContentPolicyOTLPSpanExporter(inner)

    result = exporter.export([content_bearing_span])

    assert result == SpanExportResult.SUCCESS
    exported = inner.export.call_args[0][0][0]
    for redacted_key in (
        "llm.input_messages.0.message.content",
        "llm.output_messages.0.message.content",
        "input.value",
        "output.value",
        "tool.parameters",
        "retrieval.documents.0.document.content",
        "embedding.embeddings.0.embedding.vector",
        "api_key",
        "http.url",
        "server.address",
        "net.peer.name",
    ):
        assert redacted_key not in exported.attributes
    # Non-content-bearing attributes still make it through.
    assert exported.attributes["llm.model_name"] == "gpt-4"
    assert exported.attributes["session.id"] == "session-1"
    # The span kind is how a receiving backend classifies the span, and carries
    # no request data, so metadata-only export must keep it.
    assert exported.attributes["openinference.span.kind"] == "LLM"
    # Event attributes are redacted the same way as span attributes.
    exported_event = exported.events[0]
    assert "tool.parameters" not in exported_event.attributes
    assert exported_event.attributes["exception.type"] == "ValidationError"
    # The original span (as goes to Mongo/JSONL) is untouched.
    assert content_bearing_span.attributes["input.value"] == "what is the customer's SSN?"
    assert content_bearing_span.attributes["api_key"] == "sk-should-not-leak"
    assert content_bearing_span.events[0].attributes["tool.parameters"] == '{"query": "secret"}'


def test_otlp_exporter_passes_through_content_when_full_capture(monkeypatch, content_bearing_span):
    monkeypatch.setenv("AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE", "full")
    inner = MagicMock(spec=SpanExporter)
    inner.export.return_value = SpanExportResult.SUCCESS
    exporter = ContentPolicyOTLPSpanExporter(inner)

    exporter.export([content_bearing_span])

    exported_spans = inner.export.call_args[0][0]
    assert exported_spans[0] is content_bearing_span
    assert exported_spans[0].attributes["input.value"] == "what is the customer's SSN?"


def test_otlp_exporter_fails_closed_when_redaction_errors(monkeypatch):
    monkeypatch.delenv("AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE", raising=False)
    span = MagicMock(spec=ReadableSpan)
    span.attributes = MagicMock()
    span.attributes.items.side_effect = RuntimeError("boom")
    inner = MagicMock(spec=SpanExporter)
    exporter = ContentPolicyOTLPSpanExporter(inner)

    result = exporter.export([span])

    assert result == SpanExportResult.FAILURE
    assert not inner.export.called


def test_otlp_exporter_delegates_shutdown_and_flush():
    inner = MagicMock(spec=SpanExporter)
    inner.force_flush.return_value = True
    exporter = ContentPolicyOTLPSpanExporter(inner)

    exporter.shutdown()
    assert exporter.force_flush(1234) is True

    inner.shutdown.assert_called_once()
    inner.force_flush.assert_called_once_with(1234)


def test_otlp_exporter_shutdown_survives_inner_exception():
    """The class docstring promises transport failures never propagate out of
    the exporter — shutdown() must honor that too, not just export()."""
    inner = MagicMock(spec=SpanExporter)
    inner.shutdown.side_effect = ConnectionError("collector unreachable")
    exporter = ContentPolicyOTLPSpanExporter(inner)

    exporter.shutdown()  # must not raise

    inner.shutdown.assert_called_once()


def test_otlp_exporter_force_flush_survives_inner_exception():
    inner = MagicMock(spec=SpanExporter)
    inner.force_flush.side_effect = ConnectionError("collector unreachable")
    exporter = ContentPolicyOTLPSpanExporter(inner)

    result = exporter.force_flush(1234)

    assert result is False
    inner.force_flush.assert_called_once_with(1234)


def test_otlp_exporter_handles_empty_span_list():
    inner = MagicMock(spec=SpanExporter)
    exporter = ContentPolicyOTLPSpanExporter(inner)

    result = exporter.export([])

    assert result == SpanExportResult.SUCCESS
    assert not inner.export.called


def _make_minimal_span(attributes):
    """A minimal real-ish span for exercising `_redact_span`, which
    reconstructs a `ReadableSpan` and needs every field populated."""
    context = SpanContext(
        trace_id=0x1, span_id=0x1, is_remote=False, trace_flags=TraceFlags.SAMPLED
    )
    span = MagicMock(spec=ReadableSpan)
    span.get_span_context.return_value = context
    span.context = context
    span.parent = None
    span.name = "span"
    span.kind = SpanKind.INTERNAL
    span.start_time = 1000
    span.end_time = 2000
    span.status = Status(StatusCode.OK)
    span.resource.attributes = {}
    span.events = []
    span.links = []
    span.instrumentation_scope = None
    span.attributes = attributes
    return span


@pytest.mark.parametrize(
    "key",
    [
        "llm.prompt_template.template",
        "llm.invocation_parameters",
        "metadata",
        "reranker.query",
        "exception.message",
        "exception.stacktrace",
        # Previously in _REDACTED_KEY_FRAGMENTS but never exercised by any
        # test — a future typo'd/dropped fragment would silently stop
        # redacting these with nothing in the suite catching it.
        "llm.prompts",
        "llm.prompt_template.variables",
        "message.content",
        "message.function_call_arguments_json",
        "tool_call.function.arguments",
        "embedding.text",
        "memory.content",
        "memory.value",
        "url.full",
        "net.sock.peer",
        "db.connection_string",
        "secret",
        "password",
        "credential",
        "access_token",
        "bearer",
        # Added in this round: CRITICAL/MUST/SHOULD findings from the
        # 2026-07-17 review (user.id, llm.function_call, llm.tools,
        # embedding.invocation_parameters).
        "user.id",
        "llm.function_call",
        "llm.tools",
        "embedding.invocation_parameters",
    ],
)
def test_otlp_exporter_redacts_newly_covered_fragments(monkeypatch, key):
    """Regression guard for the review-flagged denylist gaps."""
    monkeypatch.delenv("AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE", raising=False)
    span = _make_minimal_span({key: "sensitive value", "llm.model_name": "gpt-4"})
    inner = MagicMock(spec=SpanExporter)
    inner.export.return_value = SpanExportResult.SUCCESS
    exporter = ContentPolicyOTLPSpanExporter(inner)

    exporter.export([span])

    exported = inner.export.call_args[0][0][0]
    assert key not in exported.attributes
    assert exported.attributes["llm.model_name"] == "gpt-4"


def test_otlp_exporter_passes_through_new_fragments_in_full_mode(monkeypatch):
    monkeypatch.setenv("AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE", "full")
    span = _make_minimal_span(
        {
            "llm.prompt_template.template": "template text",
            "llm.invocation_parameters": '{"temperature": 0.7}',
            "metadata": '{"user": "abc"}',
            "reranker.query": "search text",
            "exception.message": "boom",
            "exception.stacktrace": "traceback...",
        }
    )
    inner = MagicMock(spec=SpanExporter)
    inner.export.return_value = SpanExportResult.SUCCESS
    exporter = ContentPolicyOTLPSpanExporter(inner)

    exporter.export([span])

    exported_spans = inner.export.call_args[0][0]
    assert exported_spans[0] is span


def test_otlp_exporter_returns_failure_when_inner_export_raises_in_full_mode(monkeypatch):
    """Transport failures on the wrapped exporter must not propagate — they'd
    crash BatchSpanProcessor's background worker thread and stop all
    subsequent exports."""
    monkeypatch.setenv("AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE", "full")
    span = _make_minimal_span({"llm.model_name": "gpt-4"})
    inner = MagicMock(spec=SpanExporter)
    inner.export.side_effect = ConnectionError("boom")
    exporter = ContentPolicyOTLPSpanExporter(inner)

    result = exporter.export([span])

    assert result == SpanExportResult.FAILURE


def test_otlp_exporter_returns_failure_when_inner_export_raises_on_redacted_batch(monkeypatch):
    monkeypatch.delenv("AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE", raising=False)
    span = _make_minimal_span({"llm.model_name": "gpt-4"})
    inner = MagicMock(spec=SpanExporter)
    inner.export.side_effect = ConnectionError("boom")
    exporter = ContentPolicyOTLPSpanExporter(inner)

    result = exporter.export([span])

    assert result == SpanExportResult.FAILURE


@pytest.mark.parametrize(
    "override,expect_full_passthrough",
    [
        ("full", True),
        ("FULL", True),
        ("  full  ", True),
        ("Full", True),
        ("metadata-only", False),
        ("bogus", False),
    ],
)
def test_content_capture_mode_override_is_normalized(override, expect_full_passthrough):
    """An explicit override must be normalized the same way
    `get_content_capture_mode()` normalizes the env var — a caller passing
    "FULL" or " full " should get full-capture behavior, not a silent
    fallback to metadata-only."""
    span = _make_minimal_span({"input.value": "secret prompt"})
    inner = MagicMock(spec=SpanExporter)
    inner.export.return_value = SpanExportResult.SUCCESS
    exporter = ContentPolicyOTLPSpanExporter(inner, content_capture_mode=override)

    exporter.export([span])

    exported_spans = inner.export.call_args[0][0]
    if expect_full_passthrough:
        assert exported_spans[0] is span
    else:
        assert "input.value" not in exported_spans[0].attributes


def test_redaction_preserves_dropped_attributes_and_events_counts():
    """_redact_span/_redact_events rebuild attributes/events as plain
    dict/tuple, which would otherwise silently zero
    ReadableSpan.dropped_attributes/dropped_events and
    Event.dropped_attributes on every redacted span — the OTLP wire format's
    "this span was truncated" signal. Uses a real Tracer with tight
    SpanLimits so the dropped counts are genuine, not mocked."""
    from opentelemetry.sdk.trace import SpanLimits, TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    captured: list[ReadableSpan] = []

    class _CapturingExporter(SpanExporter):
        def export(self, spans):
            captured.extend(spans)
            return SpanExportResult.SUCCESS

    limits = SpanLimits(max_span_attributes=1, max_events=1, max_event_attributes=1)
    provider = TracerProvider(span_limits=limits)
    provider.add_span_processor(SimpleSpanProcessor(_CapturingExporter()))
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("span-with-drops") as span:
        span.set_attribute("first", "kept")
        span.set_attribute("second", "dropped-by-limit")
        # max_events=1 means only one of these survives — give both more
        # attributes than max_event_attributes=1 allows so whichever event
        # remains has a genuine per-event attribute drop, regardless of
        # BoundedList's eviction order.
        span.add_event("event-one", {"a": "1", "b": "dropped-by-limit"})
        span.add_event("event-two", {"c": "2", "d": "dropped-by-limit"})

    original = captured[0]
    assert original.dropped_attributes > 0
    assert original.dropped_events > 0
    assert original.events[0].dropped_attributes > 0

    from agent_engine_runner_shared.tracing.exporters import _redact_span

    redacted = _redact_span(original)

    assert redacted.dropped_attributes == original.dropped_attributes
    assert redacted.dropped_events == original.dropped_events
    assert redacted.events[0].dropped_attributes == original.events[0].dropped_attributes


# =============================================================================
# setup_tracing() OTLP wiring tests
# =============================================================================


@pytest.fixture
def isolated_tracer_provider(monkeypatch):
    """Ensure setup_tracing()'s module-level singleton doesn't leak across tests."""
    monkeypatch.setattr(tracing_setup, "_tracer_provider", None)
    yield tracing_setup
    if tracing_setup._tracer_provider is not None:
        tracing_setup.shutdown_tracing()


def test_setup_tracing_omits_otlp_when_no_endpoint_configured(
    monkeypatch, isolated_tracer_provider, tmp_path
):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    mock_otlp_cls = MagicMock()
    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
        mock_otlp_cls,
    )

    isolated_tracer_provider.setup_tracing(service_name="svc-no-otlp")

    assert not mock_otlp_cls.called


def test_setup_tracing_does_not_write_local_jsonl(monkeypatch, isolated_tracer_provider, tmp_path):
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))

    isolated_tracer_provider.setup_tracing(service_name="svc-no-local-traces")
    provider = isolated_tracer_provider._tracer_provider
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("not-written-locally"):
        pass
    isolated_tracer_provider.shutdown_tracing()

    assert not (tmp_path / "traces.jsonl").exists()


def test_setup_tracing_dual_writes_to_mongo_and_otlp_when_configured(
    monkeypatch, isolated_tracer_provider, tmp_path
):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.internal:4318")
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    otlp_inner = MagicMock(spec=SpanExporter)
    otlp_inner.export.return_value = SpanExportResult.SUCCESS
    mock_otlp_cls = MagicMock(return_value=otlp_inner)
    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
        mock_otlp_cls,
    )
    collection = MagicMock()

    isolated_tracer_provider.setup_tracing(
        service_name="svc-dual-write", mongodb_collection=collection
    )
    provider = isolated_tracer_provider._tracer_provider
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("dual-write-span"):
        pass
    isolated_tracer_provider.shutdown_tracing()

    assert mock_otlp_cls.called
    assert collection.insert_many.called
    assert otlp_inner.export.called
    mongo_span_names = {
        doc["name"] for call in collection.insert_many.call_args_list for doc in call[0][0]
    }
    otlp_span_names = {
        span.name for call in otlp_inner.export.call_args_list for span in call[0][0]
    }
    assert "dual-write-span" in mongo_span_names
    assert "dual-write-span" in otlp_span_names


def test_setup_tracing_survives_otlp_construction_failure(
    monkeypatch, isolated_tracer_provider, tmp_path, caplog
):
    """A broken/misconfigured OTLP setup (missing extra, bad endpoint, ...)
    must not prevent MongoDB from being registered — previously the
    exception propagated before `trace.set_tracer_provider()` ran at all."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.internal:4318")
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
        MagicMock(side_effect=RuntimeError("boom")),
    )
    collection = MagicMock()

    isolated_tracer_provider.setup_tracing(
        service_name="svc-otlp-broken", mongodb_collection=collection
    )

    # set_tracer_provider() is only honored once per process by the OTel API,
    # so other tests in this run may already hold the global registration —
    # assert against the provider setup_tracing() actually built instead.
    provider = isolated_tracer_provider._tracer_provider
    assert provider is not None

    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("still-works"):
        pass
    isolated_tracer_provider.shutdown_tracing()

    assert collection.insert_many.called
    assert "Failed to set up OTLP exporter" in caplog.text


def test_setup_tracing_logs_warning_when_full_content_capture(
    monkeypatch, isolated_tracer_provider, tmp_path, caplog
):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.internal:4318")
    monkeypatch.setenv("AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE", "full")
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
        MagicMock(return_value=MagicMock(spec=SpanExporter)),
    )

    isolated_tracer_provider.setup_tracing(service_name="svc-full-capture")

    assert "AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE=full" in caplog.text


def test_setup_tracing_does_not_warn_when_metadata_only(
    monkeypatch, isolated_tracer_provider, tmp_path, caplog
):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.internal:4318")
    monkeypatch.delenv("AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE", raising=False)
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
        MagicMock(return_value=MagicMock(spec=SpanExporter)),
    )

    isolated_tracer_provider.setup_tracing(service_name="svc-metadata-only")

    assert "AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE=full" not in caplog.text


# =============================================================================
# httpx instrumentation + get_current_trace_context()
# =============================================================================


def test_get_current_trace_context_no_active_span():
    trace_id, span_id = tracing_setup.get_current_trace_context()
    assert trace_id is None
    assert span_id is None


def test_get_current_trace_context_returns_active_span_ids(
    isolated_tracer_provider, tmp_path, monkeypatch
):
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    isolated_tracer_provider.setup_tracing(service_name="svc-trace-ctx")
    tracer = isolated_tracer_provider.get_tracer("test")

    with tracer.start_as_current_span("span-1") as span:
        trace_id, span_id = tracing_setup.get_current_trace_context()

    ctx = span.get_span_context()
    assert trace_id == format(ctx.trace_id, "032x")
    assert span_id == format(ctx.span_id, "016x")


class _CapturingHeaderHandler(http.server.BaseHTTPRequestHandler):
    """Records the traceparent header off each request; a real HTTPTransport
    is required here since httpx.MockTransport is a distinct class the
    HTTPXClientInstrumentor doesn't patch."""

    received: list = []

    def do_GET(self):
        self.__class__.received.append(self.headers.get("traceparent"))
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass


def _serve_capturing_header_handler():
    _CapturingHeaderHandler.received = []
    server = http.server.HTTPServer(("127.0.0.1", 0), _CapturingHeaderHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_httpx_instrumentation_injects_traceparent_when_tracing_enabled(
    isolated_tracer_provider, tmp_path, monkeypatch
):
    """setup_tracing() globally instruments httpx, so any client constructed
    anywhere in the process propagates the active span via traceparent."""
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    isolated_tracer_provider.setup_tracing(service_name="svc-httpx-instr")

    server, thread = _serve_capturing_header_handler()
    port = server.server_address[1]
    try:
        tracer = isolated_tracer_provider.get_tracer("test")
        with tracer.start_as_current_span("outbound-call") as span:
            ctx = span.get_span_context()
            with httpx.Client() as client:
                client.get(f"http://127.0.0.1:{port}/")

        assert len(_CapturingHeaderHandler.received) == 1
        traceparent = _CapturingHeaderHandler.received[0]
        assert traceparent is not None
        # The instrumentor creates a child span for the outbound request, so only the
        # trace_id (not span_id) is guaranteed to match the parent "outbound-call" span.
        assert format(ctx.trace_id, "032x") in traceparent
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_platform_http_client_propagates_without_emitting_transport_span():
    """Platform RPCs carry their caller context without consuming trace storage."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = HTTPXClientInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    instrumentor.instrument(tracer_provider=provider)

    server, thread = _serve_capturing_header_handler()
    port = server.server_address[1]
    try:
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("logical-operation") as span:
            context = span.get_span_context()
            with create_httpx_client_with_tls(f"http://127.0.0.1:{port}", 5.0) as client:
                client.get(f"http://127.0.0.1:{port}/")

        provider.force_flush()
        assert [finished.name for finished in exporter.get_finished_spans()] == [
            "logical-operation"
        ]
        traceparent = _CapturingHeaderHandler.received[0]
        assert traceparent is not None
        assert traceparent.split("-")[:3] == [
            "00",
            f"{context.trace_id:032x}",
            f"{context.span_id:016x}",
        ]
    finally:
        instrumentor.uninstrument()
        provider.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_async_platform_http_client_propagates_without_emitting_transport_span():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = HTTPXClientInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    instrumentor.instrument(tracer_provider=provider)

    server, thread = _serve_capturing_header_handler()
    port = server.server_address[1]
    try:
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("logical-operation") as span:
            context = span.get_span_context()
            client = await create_async_httpx_client_with_tls(f"http://127.0.0.1:{port}", 5.0)
            async with client:
                await client.get(f"http://127.0.0.1:{port}/")

        provider.force_flush()
        assert [finished.name for finished in exporter.get_finished_spans()] == [
            "logical-operation"
        ]
        traceparent = _CapturingHeaderHandler.received[0]
        assert traceparent is not None
        assert traceparent.split("-")[:3] == [
            "00",
            f"{context.trace_id:032x}",
            f"{context.span_id:016x}",
        ]
    finally:
        instrumentor.uninstrument()
        provider.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_httpx_instrumentation_still_injects_header_without_a_parent_span(
    isolated_tracer_provider, tmp_path, monkeypatch
):
    """The instrumentor wraps every request in its own client span, so a
    traceparent is sent even with no application span in context — it just
    starts a fresh, uncorrelated trace rather than extending an existing one."""
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    isolated_tracer_provider.setup_tracing(service_name="svc-httpx-no-span")

    server, thread = _serve_capturing_header_handler()
    port = server.server_address[1]
    try:
        with httpx.Client() as client:
            client.get(f"http://127.0.0.1:{port}/")

        assert len(_CapturingHeaderHandler.received) == 1
        assert _CapturingHeaderHandler.received[0] is not None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_httpx_not_instrumented_when_tracing_never_configured(tmp_path, monkeypatch):
    """Matches the opt-in, no-op-when-unconfigured invariant: without a
    setup_tracing() call, httpx sends no traceparent and nothing crashes."""
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    instrumentor = HTTPXClientInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()

    server, thread = _serve_capturing_header_handler()
    port = server.server_address[1]
    try:
        with httpx.Client() as client:
            client.get(f"http://127.0.0.1:{port}/")

        assert _CapturingHeaderHandler.received == [None]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_shutdown_tracing_uninstruments_httpx(isolated_tracer_provider, tmp_path, monkeypatch):
    """After shutdown_tracing(), httpx clients stop injecting traceparent."""
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    isolated_tracer_provider.setup_tracing(service_name="svc-httpx-shutdown")
    isolated_tracer_provider.shutdown_tracing()

    server, thread = _serve_capturing_header_handler()
    port = server.server_address[1]
    try:
        with httpx.Client() as client:
            client.get(f"http://127.0.0.1:{port}/")

        assert _CapturingHeaderHandler.received == [None]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_instrument_httpx_missing_dependency_warns_instead_of_raising(monkeypatch, caplog):
    """A missing opentelemetry-instrumentation-httpx install must degrade to a
    warning-only no-op, not propagate ImportError out of _instrument_httpx()."""
    monkeypatch.setitem(sys.modules, "opentelemetry.instrumentation.httpx", None)

    with caplog.at_level("WARNING", logger=tracing_setup.logger.name):
        tracing_setup._instrument_httpx()

    assert "httpx instrumentation failed" in caplog.text


def test_uninstrument_httpx_missing_dependency_warns_instead_of_raising(monkeypatch, caplog):
    """Mirrors the _instrument_httpx() guard: _uninstrument_httpx() must not
    raise ImportError either, since shutdown_tracing() calls it unguarded."""
    monkeypatch.setitem(sys.modules, "opentelemetry.instrumentation.httpx", None)

    with caplog.at_level("WARNING", logger=tracing_setup.logger.name):
        tracing_setup._uninstrument_httpx()

    assert "httpx uninstrumentation failed" in caplog.text


# =============================================================================
# Global Propagator Tests
# =============================================================================


def test_setup_tracing_restricts_propagator_to_tracecontext(monkeypatch):
    """No Baggage propagator may be registered, or a future baggage writer
    would leak unredacted content on every outbound A2A call."""
    monkeypatch.setattr(tracing_setup, "_tracer_provider", None)
    prev_propagator = get_global_textmap()
    try:
        tracing_setup.setup_tracing(service_name="test-propagator")
        assert isinstance(get_global_textmap(), TraceContextTextMapPropagator)
    finally:
        set_global_textmap(prev_propagator)
