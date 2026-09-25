"""
Span Exporters for Runner SDK.

Simple exporters that work with OpenInference instrumentation:
- JSONLSpanExporter: Optional explicit local debugging exporter.
- MongoDBSpanExporter: Writes to MongoDB collection (when configured).
- ContentPolicyOTLPSpanExporter: Wraps an OTLP exporter, applying the
  AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE redaction policy.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Sequence

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

if TYPE_CHECKING:
    from pymongo.collection import Collection

logger = logging.getLogger(__name__)


def get_trace_path() -> Path:
    """Get the trace file path from env vars or default."""
    if file_env := os.getenv("AGENTIC_TRACES_FILE"):
        path = Path(file_env)
    elif base_dir := os.getenv("AGENTIC_OBSERVABILITY_DIR"):
        path = Path(base_dir) / "traces.jsonl"
    else:
        path = Path("observability") / "traces.jsonl"

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _span_to_dict(span: ReadableSpan) -> Dict[str, Any]:
    """Convert OpenTelemetry span to a dict for storage."""
    ctx = span.get_span_context()
    parent = span.parent

    # Extract attributes
    attributes = dict(span.attributes) if span.attributes else {}

    # These attributes are emitted by agent-engine-runner-shared's span processor for
    # correlation only. Trusted tenant/root scope is applied by OE from
    # Execution records when events are read.
    session_id = attributes.get("session.id") or attributes.get("session_id")
    execution_id = attributes.get("execution.id") or attributes.get("execution_id")

    doc = {
        "trace_id": format(ctx.trace_id, "032x"),  # type: ignore[union-attr]  # ctx is set by get_span_context()
        "span_id": format(ctx.span_id, "016x"),  # type: ignore[union-attr]
        "name": span.name,
        "kind": span.kind.name,
        "start_time_ns": span.start_time,
        "end_time_ns": span.end_time,
        "duration_ns": (span.end_time or 0) - (span.start_time or 0),
        "status": {
            "code": span.status.status_code.name,
            "description": span.status.description,
        },
        "attributes": attributes,
        "resource": dict(span.resource.attributes)
        if span.resource and span.resource.attributes
        else {},
    }

    # Add session_id as top-level field for easier querying
    if session_id:
        doc["session_id"] = session_id

    # Add execution_id as top-level field for easier querying
    if execution_id:
        doc["execution_id"] = execution_id

    if parent:
        doc["parent_span_id"] = format(parent.span_id, "016x")

    if span.events:
        doc["events"] = [
            {
                "name": e.name,
                "timestamp_ns": e.timestamp,
                "attributes": dict(e.attributes) if e.attributes else {},
            }
            for e in span.events
        ]

    return doc


class JSONLSpanExporter(SpanExporter):
    """Exports spans to a JSONL file."""

    def __init__(self, path: Path | None = None):
        self._path = path
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Write spans to the JSONL file."""
        if not spans:
            return SpanExportResult.SUCCESS

        path = self._path or get_trace_path()

        try:
            with self._lock:
                with path.open("a", encoding="utf-8") as f:
                    for span in spans:
                        f.write(json.dumps(_span_to_dict(span), default=str) + "\n")
            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.error(f"Failed to write traces to {path}: {e}", exc_info=True)
            return SpanExportResult.FAILURE


class MongoDBSpanExporter(SpanExporter):
    """Exports spans to a MongoDB collection."""

    def __init__(self, collection: "Collection"):
        self.collection = collection

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Insert spans as documents into MongoDB."""
        if not spans:
            return SpanExportResult.SUCCESS

        try:
            docs = [_span_to_dict(span) for span in spans]
            self.collection.insert_many(docs)
            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.error(f"Failed to write traces to MongoDB: {e}", exc_info=True)
            return SpanExportResult.FAILURE


# =============================================================================
# Content-capture policy
#
# Mongo stays unredacted (customer's own BYOC DB, canonical for the
# playground). This policy applies only to the OTLP path, which leaves the
# process bound for a third-party tool.
# =============================================================================

_CONTENT_CAPTURE_ENV = "AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE"
_CONTENT_CAPTURE_FULL = "full"
_CONTENT_CAPTURE_METADATA_ONLY = "metadata-only"

# Substrings matched case-insensitively against attribute keys. OpenInference
# flattens nested/indexed fields (e.g. "llm.input_messages.0.message.content",
# "retrieval.documents.2.document.content"), so this is fragment matching,
# not an exact-key set.
#
# This is a key-substring denylist, not a value scan: an attribute whose key
# doesn't match any fragment here passes through unredacted regardless of what
# it contains. New OpenInference/OTel semconv keys that carry content need a
# fragment added below to be covered — this list isn't self-maintaining.
#
# An allowlist (redact everything except a known-safe metadata set) would
# close this gap structurally, but was deliberately deferred: it requires
# enumerating every legitimate metadata-only key OpenInference/OTel currently
# emit (span kind, model name/provider, token counts, timing/status,
# agentic_platform.*/session/execution identity, ...), and getting that enumeration
# wrong silently hides wanted metadata rather than leaking unwanted content —
# the opposite failure mode of what metadata-only mode promises. Tracked as a
# follow-up rather than done here.
_REDACTED_KEY_FRAGMENTS: tuple[str, ...] = (
    # Prompts / completions
    "input.value",
    "output.value",
    "llm.input_messages",
    "llm.output_messages",
    "llm.prompts",
    "llm.prompt_template.variables",
    "llm.prompt_template.template",
    "llm.invocation_parameters",
    "message.content",
    "message.function_call_arguments_json",
    "tool_call.function.arguments",
    # Legacy OpenAI function-calling response shape (name + parsed args) —
    # the same content class as message.function_call_arguments_json above,
    # still emitted by langchain-openai for models using the older shape.
    "llm.function_call",
    # Tool args/results
    "tool.parameters",
    # Full tool definitions (name/description/JSON parameter schema) sent
    # alongside every LLM invocation — part of the request payload, distinct
    # from the per-call tool.parameters arguments above.
    "llm.tools",
    # Retrieved docs
    "retrieval.documents",
    "document.content",
    "reranker.query",
    # Embeddings
    "embedding.embeddings",
    "embedding.text",
    "embedding.vector",
    "embedding.invocation_parameters",
    # Memory contents (forward-compat with upcoming memory-span attributes)
    "memory.content",
    "memory.value",
    # Real end-user identifier — e.g. Google ADK's Runner(user_id=...),
    # propagated to every descendant span. Unlike session.id/execution.id/
    # workspace.id (opaque platform-generated correlation IDs, deliberately
    # left unredacted below), this can be an email/account ID a tenant
    # supplies directly.
    "user.id",
    # Free-form metadata / exceptions — OpenInference's top-level `metadata`
    # attribute and exception event fields can both carry arbitrary
    # user/tool-supplied content (e.g. an exception message echoing a bad
    # tool call's arguments).
    "metadata",
    "exception.message",
    "exception.stacktrace",
    # Network / infra (httpx instrumentation emits these on every outbound SDK
    # call; URLs can embed query-string secrets and hostnames/IPs leak
    # internal platform topology to the OTLP vendor)
    "http.url",
    "url.full",
    "net.peer",
    "server.address",
    "net.sock.peer",
    "db.connection_string",
    # Secrets — defense in depth for any attribute carrying credential-shaped data
    "secret",
    "password",
    "credential",
    "api_key",
    "access_token",
    "bearer",
)


def get_content_capture_mode() -> str:
    """Read the content-capture policy from AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE.

    Defaults to ``metadata-only``; any value other than ``full`` is treated
    as ``metadata-only`` so a typo'd env var fails closed, not open.
    """
    mode = os.getenv(_CONTENT_CAPTURE_ENV, _CONTENT_CAPTURE_METADATA_ONLY).strip().lower()
    return mode if mode == _CONTENT_CAPTURE_FULL else _CONTENT_CAPTURE_METADATA_ONLY


def _is_redacted_attribute_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in _REDACTED_KEY_FRAGMENTS)


def _redact_attributes(attributes: Any) -> Dict[str, Any]:
    if not attributes:
        return {}
    return {k: v for k, v in attributes.items() if not _is_redacted_attribute_key(k)}


class _RedactedEvent(Event):
    """An ``Event`` copy whose ``dropped_attributes`` is taken from the
    original event rather than recomputed from the (now plain-dict)
    attributes container. See ``_RedactedReadableSpan`` for why this matters.
    """

    def __init__(self, *, dropped_attributes: int, **kwargs: Any):
        super().__init__(**kwargs)
        self._dropped_attributes_count = dropped_attributes

    @property
    def dropped_attributes(self) -> int:
        return self._dropped_attributes_count


class _RedactedReadableSpan(ReadableSpan):
    """A ``ReadableSpan`` copy whose ``dropped_attributes``/``dropped_events``
    are taken from the original span rather than recomputed.

    ``ReadableSpan.dropped_attributes``/``dropped_events`` (and
    ``Event.dropped_attributes``) only return a nonzero count when the
    underlying container is the SDK's ``BoundedAttributes``/``BoundedList``
    wrapper — a plain dict/tuple (what redaction rebuilds attributes/events
    as) always reports 0. The OTLP proto encoder reads exactly these
    properties for the wire format's dropped_attributes_count/
    dropped_events_count fields, so without this the "this span was
    truncated" signal is silently lost on every redacted span.
    """

    def __init__(self, *, dropped_attributes: int, dropped_events: int, **kwargs: Any):
        super().__init__(**kwargs)
        self._dropped_attributes_count = dropped_attributes
        self._dropped_events_count = dropped_events

    @property
    def dropped_attributes(self) -> int:
        return self._dropped_attributes_count

    @property
    def dropped_events(self) -> int:
        return self._dropped_events_count


def _redact_events(events: Any) -> tuple[Event, ...]:
    if not events:
        return ()
    return tuple(
        _RedactedEvent(
            name=e.name,
            attributes=_redact_attributes(e.attributes),
            timestamp=e.timestamp,
            dropped_attributes=e.dropped_attributes,
        )
        for e in events
    )


def _redact_span(span: ReadableSpan) -> ReadableSpan:
    """Return a copy of ``span`` with content-bearing attributes stripped.

    Event attributes (e.g. an exception event that echoes a bad tool call's
    arguments) are redacted the same way as span attributes — an event is
    just another place a content-bearing key could show up.
    """
    return _RedactedReadableSpan(
        dropped_attributes=span.dropped_attributes,
        dropped_events=span.dropped_events,
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=_redact_attributes(span.attributes),
        events=_redact_events(span.events),
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class ContentPolicyOTLPSpanExporter(SpanExporter):
    """Wraps an OTLP SpanExporter, enforcing AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE.

    Redaction failures fail closed: a batch that can't be safely redacted is
    dropped rather than forwarded unredacted, matching the metadata-only
    default's privacy-first intent. Transport failures on the wrapped
    exporter never propagate into the caller's execution — SpanExporter runs
    on the BatchSpanProcessor's background thread.
    """

    def __init__(self, inner: SpanExporter, content_capture_mode: str | None = None):
        self._inner = inner
        if content_capture_mode is not None:
            normalized = content_capture_mode.strip().lower()
            self._content_capture_mode = (
                normalized
                if normalized == _CONTENT_CAPTURE_FULL
                else _CONTENT_CAPTURE_METADATA_ONLY
            )
        else:
            self._content_capture_mode = get_content_capture_mode()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not spans:
            return SpanExportResult.SUCCESS

        if self._content_capture_mode == _CONTENT_CAPTURE_FULL:
            return self._export_inner(spans)

        try:
            redacted = [_redact_span(span) for span in spans]
        except Exception:
            logger.error(
                "Content-capture redaction failed; dropping batch rather than "
                "risk exporting unredacted content",
                exc_info=True,
            )
            return SpanExportResult.FAILURE

        return self._export_inner(redacted)

    def _export_inner(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Delegate to the wrapped exporter, converting any raised exception
        into a FAILURE result. OTLP HTTP exporters can raise on
        connection/transport failures; letting that propagate would crash
        the BatchSpanProcessor's background worker thread and stop all
        subsequent exports."""
        try:
            return self._inner.export(spans)
        except Exception:
            logger.error("OTLP export failed", exc_info=True)
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """Delegate to the wrapped exporter, swallowing any exception it
        raises — same guarantee as ``_export_inner``, so a final flush
        failure (e.g. collector unreachable) can't propagate out of shutdown."""
        try:
            self._inner.shutdown()
        except Exception:
            logger.error("OTLP exporter shutdown failed", exc_info=True)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        try:
            return self._inner.force_flush(timeout_millis)
        except Exception:
            logger.error("OTLP exporter force_flush failed", exc_info=True)
            return False
