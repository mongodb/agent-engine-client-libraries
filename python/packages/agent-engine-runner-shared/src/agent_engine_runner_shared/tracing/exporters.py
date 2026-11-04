"""
Span Exporters for Runner SDK.

Simple exporters that work with OpenInference instrumentation:
- JSONLSpanExporter: Optional explicit local debugging exporter.
- MongoDBSpanExporter: Writes to MongoDB collection (when configured).
- PlatformSpanExporter: Sends spans to the Orchestration Engine, which stores
  them in the project traces collection.
- ContentPolicyOTLPSpanExporter: Wraps an OTLP exporter, applying the
  AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE redaction policy.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import datetime
import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Sequence

import httpx
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from agent_engine_runner_shared.tracing.conversation import (
    ConversationDeltaTracker,
    has_input_messages,
    trace_stream,
)

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


def _trim_conversation(doc: Dict[str, Any], tracker: ConversationDeltaTracker) -> Dict[str, Any]:
    """Keep only the input messages this call added (see tracing.conversation)."""
    attributes = doc.get("attributes")
    if not attributes or not has_input_messages(attributes):
        return doc
    doc["attributes"] = tracker.trim(
        attributes,
        trace_id=doc["trace_id"],
        stream_id=trace_stream(doc.get("resource") or {}),
    )
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
        self._conversations = ConversationDeltaTracker()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Insert spans as documents into MongoDB."""
        if not spans:
            return SpanExportResult.SUCCESS

        try:
            # Trim on a copy: a failed insert must not advance past messages the
            # store never received, or later calls would omit them.
            staged = self._conversations.copy()
            docs = [_trim_conversation(_span_to_dict(span), staged) for span in spans]
            self.collection.insert_many(docs)
            self._conversations = staged
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


# =============================================================================
# Platform trace exporter (Orchestration Engine /internal/traces)
#
# The guest sends OTLP-protobuf batches to the OE, which converts and stores
# them in the project traces collection. This replaces the guest's direct
# MongoDB write: it is the platform trace path, independent of customer OTLP
# export, and carries unredacted spans so the playground can show prompts and
# outputs as it did from Mongo.
# =============================================================================

_PLATFORM_TRACE_INGEST_PATH = "/internal/traces"
_OTLP_PROTOBUF_CONTENT_TYPE = "application/x-protobuf"

# The OE rejects a raw request body over its own configured limit with 413.
# Chunk under the same default so a large batch is split before it is sent,
# rather than after a rejection; a single span larger than this is dropped.
_DEFAULT_PLATFORM_MAX_REQUEST_BYTES = 8 << 20

# Overall deadline for one export() call. export() runs synchronously on the
# BatchSpanProcessor worker thread, so a bounded deadline keeps an unreachable
# or slow OE from stalling the export queue behind it.
_DEFAULT_PLATFORM_EXPORT_TIMEOUT_SECONDS = 10.0
_PLATFORM_PER_ATTEMPT_TIMEOUT_SECONDS = 3.0

# Retryable: the OE is rolling (404), shedding (503/429), or a gateway hiccup
# (502/504). Any other status is terminal for the batch.
_PLATFORM_RETRYABLE_STATUS = frozenset({404, 429, 502, 503, 504})

_PLATFORM_BACKOFF_BASE_SECONDS = 0.5
_PLATFORM_BACKOFF_MAX_SECONDS = 5.0


class _ExportDeadlineExceeded(Exception):
    """A platform trace POST did not finish within the remaining export budget."""


_platform_loop: "asyncio.AbstractEventLoop | None" = None
_platform_loop_lock = threading.Lock()


def _platform_event_loop() -> "asyncio.AbstractEventLoop":
    """Return the process-wide loop that runs cancellable platform trace POSTs.

    A synchronous httpx request cannot be aborted once it is in flight, so a
    total wall-clock export deadline can only be enforced by cancelling an async
    request. One daemon loop serves every exporter and lives for the process.
    """
    global _platform_loop
    with _platform_loop_lock:
        if _platform_loop is None:
            loop = asyncio.new_event_loop()

            def _run() -> None:
                asyncio.set_event_loop(loop)
                loop.run_forever()

            threading.Thread(target=_run, daemon=True, name="platform-trace-export").start()
            _platform_loop = loop
        return _platform_loop


def _default_platform_client_factory(base_url: str) -> "Awaitable[httpx.AsyncClient]":
    """Build the async mTLS client used for platform trace POSTs.

    The async twin of ``get_shared_client``: same platform mTLS and self-span
    suppression, but cancellable so the export deadline can abort a request.
    """
    from agent_engine_runner_shared.tls_client import create_async_httpx_client_with_tls
    from agent_engine_runner_shared.utils import get_request_timeout

    return create_async_httpx_client_with_tls(base_url, get_request_timeout())


class _PlatformTracePoster:
    """Cancellable transport for platform trace POSTs.

    A synchronous httpx request cannot be aborted once in flight, so the
    exporter's total wall-clock deadline is enforced by cancelling an async
    request. The client is cached and rebuilt when the mounted certificates
    rotate, so keep-alive and certificate rotation match the shared synchronous
    client.
    """

    def __init__(
        self,
        base_url: str,
        *,
        client_factory: Callable[[str], "Awaitable[httpx.AsyncClient]"] | None = None,
        cert_mtimes: Callable[[], tuple[float, float, float] | None] | None = None,
    ):
        self._base_url = base_url
        self._client_factory = client_factory or _default_platform_client_factory
        if cert_mtimes is None:
            from agent_engine_runner_shared.tls_client import _get_cert_mtimes

            cert_mtimes = _get_cert_mtimes
        self._cert_mtimes = cert_mtimes
        self._client: httpx.AsyncClient | None = None
        self._client_cert_mtimes: tuple[float, float, float] | None = None

    async def _client_for_request(self) -> httpx.AsyncClient:
        current = self._cert_mtimes()
        if self._client is not None and current is not None and self._client_cert_mtimes != current:
            await self._client.aclose()
            self._client = None
        if self._client is None:
            self._client = await self._client_factory(self._base_url)
            self._client_cert_mtimes = current
        return self._client

    async def _request(
        self, url: str, body: bytes, headers: dict[str, str], timeout_seconds: float
    ) -> httpx.Response:
        client = await self._client_for_request()
        return await client.post(url, content=body, headers=headers, timeout=timeout_seconds)

    def post(
        self, url: str, body: bytes, headers: dict[str, str], timeout_seconds: float
    ) -> httpx.Response:
        """POST ``body`` and return the response, bounding the whole request.

        Raises ``_ExportDeadlineExceeded`` when the request does not finish
        within ``timeout_seconds``, cancelling the in-flight request rather than
        abandoning it. Any other transport failure propagates unchanged.
        """
        future = asyncio.run_coroutine_threadsafe(
            self._request(url, body, headers, timeout_seconds), _platform_event_loop()
        )
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise _ExportDeadlineExceeded() from exc

    def close(self) -> None:
        """Close the cached client, if any (best-effort)."""
        client = self._client
        if client is None:
            return
        self._client = None
        try:
            asyncio.run_coroutine_threadsafe(client.aclose(), _platform_event_loop()).result(
                timeout=2.0
            )
        except Exception:  # noqa: BLE001 - closing is best-effort
            pass


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse a Retry-After header, accepting delay-seconds or an HTTP-date."""
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
        return seconds if seconds >= 0 else None
    except ValueError:
        pass

    from email.utils import parsedate_to_datetime

    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    # parsedate_to_datetime returns a naive datetime for an HTTP-date with no
    # timezone; treat it as UTC rather than raising on the subtraction.
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.timezone.utc)
    return max(0.0, (when - datetime.datetime.now(datetime.timezone.utc)).total_seconds())


class PlatformSpanExporter(SpanExporter):
    """Export spans to the Orchestration Engine's platform trace endpoint.

    The guest POSTs OTLP-protobuf batches to ``{oe_url}/internal/traces``; the
    OE converts and stores them. Requests go through the platform async mTLS
    client (with certificate rotation and self-span suppression), and each is
    cancellable so the export deadline is a real wall-clock bound.

    Delivery is best-effort: a batch is retried on transport errors and
    retryable statuses within a bounded deadline, split on 413, and dropped when
    the OE permanently refuses it. A single span larger than the OE body limit is
    dropped without failing the rest of its batch.
    """

    def __init__(
        self,
        base_url: str,
        *,
        client_factory: Callable[[str], "Awaitable[httpx.AsyncClient]"] | None = None,
        cert_mtimes: Callable[[], tuple[float, float, float] | None] | None = None,
        max_request_bytes: int = _DEFAULT_PLATFORM_MAX_REQUEST_BYTES,
        export_timeout_seconds: float = _DEFAULT_PLATFORM_EXPORT_TIMEOUT_SECONDS,
        per_attempt_timeout_seconds: float = _PLATFORM_PER_ATTEMPT_TIMEOUT_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._base_url = base_url.rstrip("/")
        self.url = f"{self._base_url}{_PLATFORM_TRACE_INGEST_PATH}"
        self._poster = _PlatformTracePoster(
            self._base_url, client_factory=client_factory, cert_mtimes=cert_mtimes
        )
        self._max_request_bytes = max_request_bytes
        self._export_timeout_seconds = export_timeout_seconds
        self._per_attempt_timeout_seconds = per_attempt_timeout_seconds
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._dropped_spans = 0
        self._failed_batches = 0
        self._last_error: str | None = None

    @property
    def dropped_spans(self) -> int:
        with self._lock:
            return self._dropped_spans

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not spans:
            return SpanExportResult.SUCCESS

        deadline = self._clock() + self._export_timeout_seconds
        dropped_before = self.dropped_spans
        ok = True
        for chunk in self._chunk_by_size(list(spans)):
            if not self._export_chunk(chunk, deadline):
                ok = False

        if ok:
            # A partial success drops spans without failing the batch, so only
            # a clean export clears the last error; /health stays degraded until
            # then rather than reporting attached while spans are being lost.
            if self.dropped_spans == dropped_before:
                with self._lock:
                    self._last_error = None
            return SpanExportResult.SUCCESS
        with self._lock:
            self._failed_batches += 1
        return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        self._poster.close()

    @staticmethod
    def _encode(spans: Sequence[ReadableSpan]) -> bytes:
        from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans

        return encode_spans(spans).SerializeToString()

    def _chunk_by_size(self, spans: list[ReadableSpan]) -> list[list[ReadableSpan]]:
        """Split spans into chunks whose encoded request fits the OE body limit.

        A single span that does not fit on its own is returned alone;
        _export_chunk drops it rather than sending a request the OE will reject.
        """
        if len(spans) <= 1:
            return [spans]
        if len(self._encode(spans)) <= self._max_request_bytes:
            return [spans]
        mid = len(spans) // 2
        return self._chunk_by_size(spans[:mid]) + self._chunk_by_size(spans[mid:])

    def _export_chunk(self, chunk: list[ReadableSpan], deadline: float) -> bool:
        if len(chunk) == 1 and len(self._encode(chunk)) > self._max_request_bytes:
            # A single span larger than the OE body limit; nothing to split.
            self._drop_oversized()
            return False
        attempt = 0
        while True:
            attempt += 1
            remaining = deadline - self._clock()
            if remaining <= 0:
                self._record_error("export deadline exceeded")
                return False
            try:
                # Cap the attempt by the remaining export budget and let the
                # poster cancel the request at that bound, so the whole export
                # honours its wall-clock deadline rather than just each httpx
                # operation.
                response = self._post(chunk, min(self._per_attempt_timeout_seconds, remaining))
            except _ExportDeadlineExceeded:
                # The attempt was cancelled at its own timeout. Retry while the
                # overall export budget still allows another attempt; fail only
                # once that budget is exhausted.
                if not self._backoff(attempt, deadline, None):
                    self._record_error("export deadline exceeded")
                    return False
                continue
            except Exception as exc:  # noqa: BLE001 - any transport error is retryable
                if not self._backoff(attempt, deadline, None):
                    self._record_error(f"transport error: {exc}")
                    return False
                continue

            if self._clock() > deadline:
                # The attempt overran the export budget; a completion past the
                # deadline is not a timely success.
                self._record_error("export deadline exceeded")
                return False

            status = response.status_code
            if 200 <= status < 300:
                self._handle_partial_success(response)
                return True
            if status == 413:
                return self._split_and_retry(chunk, deadline)
            if status in _PLATFORM_RETRYABLE_STATUS:
                retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
                if not self._backoff(attempt, deadline, retry_after):
                    self._record_error(f"status {status} after {attempt} attempt(s)")
                    return False
                continue

            self._record_error(f"terminal status {status}")
            logger.warning(
                "Trace export: OE returned terminal status %s; dropping batch of %d span(s)",
                status,
                len(chunk),
            )
            return False

    def _split_and_retry(self, chunk: list[ReadableSpan], deadline: float) -> bool:
        """Halve a batch the OE rejected as too large and export each half.

        Each half goes back through _export_chunk, so a half still too large is
        split again, and a single span that cannot be split is dropped.
        """
        if len(chunk) <= 1:
            self._drop_oversized()
            return False
        mid = len(chunk) // 2
        ok = True
        for half in (chunk[:mid], chunk[mid:]):
            if not self._export_chunk(half, deadline):
                ok = False
        return ok

    def _post(self, chunk: list[ReadableSpan], timeout_seconds: float) -> httpx.Response:
        return self._poster.post(
            self.url,
            self._encode(chunk),
            {"Content-Type": _OTLP_PROTOBUF_CONTENT_TYPE},
            timeout_seconds,
        )

    def _handle_partial_success(self, response: httpx.Response) -> None:
        content = response.content
        if not content:
            return
        try:
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
                ExportTraceServiceResponse,
            )

            parsed = ExportTraceServiceResponse()
            parsed.ParseFromString(content)
        except Exception:  # noqa: BLE001 - an unparsable body is not actionable
            return
        rejected = parsed.partial_success.rejected_spans
        if not rejected:
            return
        # The OE accepted the batch but discarded some spans. Delivery is
        # best-effort, so the batch is not retried (a retry would duplicate the
        # spans that were stored), but the loss is recorded so tracing_status()
        # and /health surface it.
        message = parsed.partial_success.error_message
        with self._lock:
            self._dropped_spans += rejected
        self._record_error(
            f"OE rejected {rejected} span(s): {message}"
            if message
            else f"OE rejected {rejected} span(s)"
        )
        logger.warning(
            "Trace export: OE rejected %d span(s) as malformed or unstorable%s",
            rejected,
            f": {message}" if message else "",
        )

    def _backoff(self, attempt: int, deadline: float, retry_after: float | None) -> bool:
        """Sleep before the next attempt, or return False when none is possible.

        Honours Retry-After when the OE sends one; otherwise uses exponential
        backoff with jitter. Returns False when the delay would run past the
        export deadline, so a retry is never attempted without time for it.
        """
        if retry_after is not None:
            delay = retry_after
        else:
            delay = min(
                _PLATFORM_BACKOFF_MAX_SECONDS,
                _PLATFORM_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)),
            )
            delay *= 0.5 + random.random() * 0.5
        if delay >= deadline - self._clock():
            return False
        self._sleep(delay)
        return True

    def _drop_oversized(self) -> None:
        with self._lock:
            self._dropped_spans += 1
        logger.warning("Trace export: dropping a span larger than the OE body limit")
        # A dropped span means the export did not fully succeed; record it so
        # tracing_status() and /health surface the loss.
        self._record_error("dropped a span larger than the OE body limit")

    def _record_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message
        logger.debug("Trace export failed: %s", message)
