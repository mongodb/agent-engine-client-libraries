"""Unit tests for PlatformSpanExporter and its setup_tracing() wiring.

The platform exporter sends OTLP-protobuf batches to the OE's /internal/traces
endpoint: retrying transient failures within a bounded deadline, splitting a
batch the OE rejects as too large, and dropping only what it cannot deliver.
"""

import asyncio
import threading
from unittest.mock import MagicMock

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import agent_engine_runner_shared.tracing.setup as tracing_setup
from agent_engine_runner_shared import TenantRuntime
from agent_engine_runner_shared.tracing.exporters import PlatformSpanExporter


def _make_spans(count: int, pad_bytes: int = 0):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    for i in range(count):
        with tracer.start_as_current_span(f"span-{i}") as span:
            if pad_bytes:
                span.set_attribute("pad", "x" * pad_bytes)
    return exporter.get_finished_spans()


def _span_count(body: bytes) -> int:
    request = ExportTraceServiceRequest()
    request.ParseFromString(body)
    return sum(
        len(scope.spans) for resource in request.resource_spans for scope in resource.scope_spans
    )


def _span_ids(body: bytes) -> list[bytes]:
    request = ExportTraceServiceRequest()
    request.ParseFromString(body)
    return [
        span.span_id
        for resource in request.resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    ]


class _Response:
    def __init__(self, status_code: int, *, headers=None, content: bytes = b""):
        self.status_code = status_code
        self.headers = headers or {}
        self.content = content


class _Client:
    """Scripted async client: ``handler(body)`` returns a response or raises."""

    def __init__(self, handler):
        self._handler = handler
        self.requests = []

    async def post(self, url, *, content, headers, timeout):
        self.requests.append(
            {"url": url, "content": content, "headers": headers, "timeout": timeout}
        )
        result = self._handler(content)
        if isinstance(result, Exception):
            raise result
        return result


def _exporter(handler, **kwargs):
    client = _Client(handler)

    async def _factory(_base: str):
        return client

    exporter = PlatformSpanExporter(
        "http://oe:8000",
        client_factory=_factory,
        cert_mtimes=lambda: None,
        **kwargs,
    )
    return exporter, client


class _Clock:
    """A controllable monotonic clock for deadline tests."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_export_posts_otlp_protobuf_to_ingest_path():
    exporter, client = _exporter(lambda _body: _Response(200))

    result = exporter.export(_make_spans(1))

    assert result == SpanExportResult.SUCCESS
    assert len(client.requests) == 1
    request = client.requests[0]
    assert request["url"] == "http://oe:8000/internal/traces"
    assert request["headers"]["Content-Type"] == "application/x-protobuf"
    assert _span_count(request["content"]) == 1


def test_export_empty_span_list_is_a_noop():
    exporter, client = _exporter(lambda _body: _Response(200))

    assert exporter.export([]) == SpanExportResult.SUCCESS
    assert client.requests == []


@pytest.mark.parametrize("status", [404, 429, 502, 503, 504])
def test_export_retries_retryable_status_then_succeeds(status):
    calls = {"n": 0}

    def handler(_body):
        calls["n"] += 1
        return _Response(status) if calls["n"] == 1 else _Response(200)

    exporter, client = _exporter(handler, sleep=lambda _s: None)

    assert exporter.export(_make_spans(1)) == SpanExportResult.SUCCESS
    assert len(client.requests) == 2


def test_export_retries_connection_error_then_succeeds():
    calls = {"n": 0}

    def handler(_body):
        calls["n"] += 1
        if calls["n"] == 1:
            return ConnectionError("refused")
        return _Response(200)

    exporter, client = _exporter(handler, sleep=lambda _s: None)

    assert exporter.export(_make_spans(1)) == SpanExportResult.SUCCESS
    assert len(client.requests) == 2


def test_export_terminal_status_drops_without_retry():
    exporter, client = _exporter(lambda _body: _Response(400))

    assert exporter.export(_make_spans(1)) == SpanExportResult.FAILURE
    assert len(client.requests) == 1
    assert exporter.last_error is not None


def test_export_honors_retry_after_header():
    calls = {"n": 0}
    sleeps = []

    def handler(_body):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Response(503, headers={"Retry-After": "2"})
        return _Response(200)

    exporter, _client = _exporter(handler, sleep=sleeps.append, export_timeout_seconds=30.0)

    assert exporter.export(_make_spans(1)) == SpanExportResult.SUCCESS
    assert sleeps == [2.0]


def test_export_accepts_retry_after_http_date_without_timezone():
    """A naive HTTP-date Retry-After must not raise out of export()."""
    calls = {"n": 0}
    sleeps = []

    def handler(_body):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Response(503, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00"})
        return _Response(200)

    exporter, _client = _exporter(handler, sleep=sleeps.append, export_timeout_seconds=30.0)

    assert exporter.export(_make_spans(1)) == SpanExportResult.SUCCESS
    assert sleeps == [0.0]


def test_export_gives_up_when_retry_after_exceeds_deadline():
    exporter, client = _exporter(
        lambda _body: _Response(503, headers={"Retry-After": "30"}),
        sleep=lambda _s: None,
        export_timeout_seconds=5.0,
    )

    assert exporter.export(_make_spans(1)) == SpanExportResult.FAILURE
    assert len(client.requests) == 1


def test_export_caps_attempt_timeout_by_remaining_budget():
    clock = _Clock(0.0)
    calls = {"n": 0}

    def handler(_body):
        calls["n"] += 1
        return _Response(503) if calls["n"] == 1 else _Response(200)

    exporter, client = _exporter(
        handler,
        clock=clock,
        sleep=lambda _s: clock.advance(8.0),
        export_timeout_seconds=10.0,
        per_attempt_timeout_seconds=3.0,
    )

    assert exporter.export(_make_spans(1)) == SpanExportResult.SUCCESS
    # The first attempt has the whole 10s budget, so it takes the 3s
    # per-attempt cap; the second starts with 2s left and is capped to that,
    # rather than getting a fresh 3s that would overrun the deadline.
    assert [r["timeout"] for r in client.requests] == [3.0, 2.0]


def test_export_does_not_count_a_late_completion_as_success():
    clock = _Clock(0.0)

    def handler(_body):
        clock.advance(10.5)
        return _Response(200)

    exporter, client = _exporter(handler, clock=clock, export_timeout_seconds=10.0)

    assert exporter.export(_make_spans(1)) == SpanExportResult.FAILURE
    assert len(client.requests) == 1
    assert exporter.last_error is not None


async def test_export_cancels_an_in_flight_request_at_the_deadline():
    started = threading.Event()
    cancelled = threading.Event()

    class _SlowClient:
        async def post(self, url, *, content, headers, timeout):
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            raise AssertionError("in-flight request was not cancelled")

    async def _factory(_base: str):
        return _SlowClient()

    exporter = PlatformSpanExporter(
        "http://oe:8000",
        client_factory=_factory,
        cert_mtimes=lambda: None,
        export_timeout_seconds=0.5,
        per_attempt_timeout_seconds=0.5,
    )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, exporter.export, _make_spans(1))

    assert result == SpanExportResult.FAILURE
    assert started.is_set()
    assert cancelled.wait(timeout=2.0), "in-flight request was not cancelled"


async def test_export_retries_after_an_attempt_timeout_within_the_budget():
    calls = {"n": 0}
    cancelled = threading.Event()

    class _StallingThenOkClient:
        async def post(self, url, *, content, headers, timeout):
            calls["n"] += 1
            if calls["n"] == 1:
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
                raise AssertionError("first request was not cancelled")
            return _Response(200)

    async def _factory(_base: str):
        return _StallingThenOkClient()

    exporter = PlatformSpanExporter(
        "http://oe:8000",
        client_factory=_factory,
        cert_mtimes=lambda: None,
        export_timeout_seconds=5.0,
        per_attempt_timeout_seconds=0.5,
        sleep=lambda _s: None,
    )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, exporter.export, _make_spans(1))

    # The first attempt stalls past the per-attempt cap and is cancelled, but
    # the export budget still allows a retry, which succeeds.
    assert result == SpanExportResult.SUCCESS
    assert calls["n"] == 2
    assert cancelled.wait(timeout=2.0), "in-flight request was not cancelled"
    assert exporter.last_error is None


def test_export_splits_batch_on_413():
    # The OE rejects a batch of more than two spans; each half then succeeds.
    accepted: list[bytes] = []

    def handler(body):
        if _span_count(body) > 2:
            return _Response(413)
        accepted.append(body)
        return _Response(200)

    exporter, client = _exporter(handler)
    spans = _make_spans(4)

    result = exporter.export(spans)

    assert result == SpanExportResult.SUCCESS
    assert len(client.requests) == 3
    # Only accepted responses count as delivered: every original span must
    # arrive exactly once. The rejected parent request contributes nothing, so
    # a splitter that duplicates one half and drops the other fails here.
    expected = sorted(span.context.span_id.to_bytes(8, "big") for span in spans)
    delivered = sorted(span_id for body in accepted for span_id in _span_ids(body))
    assert delivered == expected


def test_export_drops_single_oversized_span():
    exporter, client = _exporter(lambda _body: _Response(200), max_request_bytes=1)

    assert exporter.export(_make_spans(1)) == SpanExportResult.FAILURE
    assert client.requests == []
    assert exporter.dropped_spans == 1
    assert exporter.last_error is not None


def test_export_drops_single_span_rejected_with_413():
    # A single span the OE still rejects as too large cannot be split; it is
    # dropped and reported, not retried forever.
    exporter, client = _exporter(lambda _body: _Response(413))

    assert exporter.export(_make_spans(1)) == SpanExportResult.FAILURE
    assert len(client.requests) == 1
    assert exporter.dropped_spans == 1
    assert exporter.last_error is not None


def test_oversized_span_degrades_status_even_when_a_sibling_chunk_succeeds():
    exporter, client = _exporter(lambda _body: _Response(200), max_request_bytes=1000)

    # One span too large for the OE, one deliverable: the batch fails as a
    # whole even though a sibling chunk reached the OE.
    spans = _make_spans(1, pad_bytes=2000) + _make_spans(1)

    assert exporter.export(spans) == SpanExportResult.FAILURE
    assert exporter.dropped_spans == 1
    assert exporter.last_error is not None
    assert len(client.requests) == 1


def test_export_chunks_a_batch_larger_than_the_body_limit():
    exporter, client = _exporter(lambda _body: _Response(200), max_request_bytes=1500)

    result = exporter.export(_make_spans(4, pad_bytes=1000))

    assert result == SpanExportResult.SUCCESS
    assert len(client.requests) > 1
    delivered = sum(_span_count(r["content"]) for r in client.requests)
    assert delivered == 4


def test_export_partial_success_is_not_retried_but_records_the_loss():
    response = ExportTraceServiceResponse()
    response.partial_success.rejected_spans = 1
    response.partial_success.error_message = "spans rejected: malformed or unstorable"
    body = response.SerializeToString()

    exporter, client = _exporter(lambda _b: _Response(200, content=body))

    assert exporter.export(_make_spans(2)) == SpanExportResult.SUCCESS
    assert len(client.requests) == 1
    # Not retried, but the rejected spans are recorded so /health degrades.
    assert exporter.dropped_spans == 1
    assert exporter.last_error is not None


def test_default_client_factory_is_the_async_platform_client():
    """The default factory is the cancellable async mTLS client, which creates
    no span of its own and picks up certificate rotation."""
    from agent_engine_runner_shared.tracing import exporters

    exporter = PlatformSpanExporter("http://oe:8000")

    assert exporter._poster._client_factory is exporters._default_platform_client_factory


# ---------------------------------------------------------------------------
# setup_tracing() wiring
# ---------------------------------------------------------------------------


_PLATFORM_FLAG = tracing_setup._PLATFORM_TRACE_EXPORTER_ENV


@pytest.fixture
def isolated_tracing(monkeypatch):
    monkeypatch.setattr(tracing_setup, "_tracer_provider", None)
    monkeypatch.setattr(tracing_setup, "_mongodb_exporter_attached", False)
    monkeypatch.setattr(tracing_setup, "_mongodb_degraded", False)
    monkeypatch.setattr(tracing_setup, "_retry_stop_event", None)
    monkeypatch.setattr(tracing_setup, "_retry_thread", None)
    monkeypatch.setattr(tracing_setup, "_tracing_mongo_client", None)
    monkeypatch.setattr(tracing_setup, "_platform_exporter", None)
    monkeypatch.setattr(tracing_setup, "_platform_exporter_attached", False)
    monkeypatch.delenv("OE_URL", raising=False)
    monkeypatch.delenv(_PLATFORM_FLAG, raising=False)
    yield tracing_setup
    tracing_setup.shutdown_tracing()


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " true "])
def test_platform_trace_exporter_enabled_accepts_truthy_values(monkeypatch, value):
    monkeypatch.setenv(_PLATFORM_FLAG, value)
    assert tracing_setup.platform_trace_exporter_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "on"])
def test_platform_trace_exporter_disabled_by_default_and_on_other_values(monkeypatch, value):
    monkeypatch.delenv(_PLATFORM_FLAG, raising=False)
    assert tracing_setup.platform_trace_exporter_enabled() is False
    monkeypatch.setenv(_PLATFORM_FLAG, value)
    assert tracing_setup.platform_trace_exporter_enabled() is False


def test_setup_tracing_installs_platform_exporter_when_flagged(monkeypatch, isolated_tracing):
    monkeypatch.setenv("OE_URL", "http://oe:8000")
    monkeypatch.setenv(_PLATFORM_FLAG, "1")

    isolated_tracing.setup_tracing(service_name="svc-platform")

    assert isolated_tracing._platform_exporter_attached is True
    assert isolated_tracing._mongodb_exporter_attached is False
    assert isolated_tracing.tracing_status() == {
        "database_exporter": "disabled",
        "platform_exporter": "attached",
    }


def test_setup_tracing_flag_prefers_oe_url_over_a_mongodb_collection(monkeypatch, isolated_tracing):
    monkeypatch.setenv("OE_URL", "http://oe:8000")
    monkeypatch.setenv(_PLATFORM_FLAG, "1")
    collection = MagicMock()

    isolated_tracing.setup_tracing(service_name="svc-both", mongodb_collection=collection)

    assert isolated_tracing._platform_exporter_attached is True
    assert isolated_tracing._mongodb_exporter_attached is False


def test_setup_tracing_oe_url_without_flag_keeps_mongodb(monkeypatch, isolated_tracing):
    # The flag is off by default: a live runner with OE_URL set must keep
    # writing MongoDB directly, not silently switch to the platform exporter.
    monkeypatch.setenv("OE_URL", "http://oe:8000")
    collection = MagicMock()

    isolated_tracing.setup_tracing(service_name="svc-default", mongodb_collection=collection)

    assert isolated_tracing._platform_exporter_attached is False
    assert isolated_tracing._mongodb_exporter_attached is True
    assert isolated_tracing.tracing_status() == {
        "database_exporter": "attached",
        "platform_exporter": "disabled",
    }


def test_setup_tracing_without_oe_url_uses_mongodb(isolated_tracing):
    collection = MagicMock()

    isolated_tracing.setup_tracing(service_name="svc-mongo", mongodb_collection=collection)

    assert isolated_tracing._mongodb_exporter_attached is True
    assert isolated_tracing._platform_exporter_attached is False
    assert isolated_tracing.tracing_status() == {
        "database_exporter": "attached",
        "platform_exporter": "disabled",
    }


def test_platform_exporter_status_degrades_on_drop_and_recovers(isolated_tracing):
    exporter, _client = _exporter(lambda _body: _Response(200), max_request_bytes=1000)
    isolated_tracing._platform_exporter = exporter
    isolated_tracing._platform_exporter_attached = True

    assert isolated_tracing.tracing_status()["platform_exporter"] == "attached"

    assert exporter.export(_make_spans(1, pad_bytes=2000)) == SpanExportResult.FAILURE
    assert isolated_tracing.tracing_status()["platform_exporter"] == "degraded"

    assert exporter.export(_make_spans(1)) == SpanExportResult.SUCCESS
    assert isolated_tracing.tracing_status()["platform_exporter"] == "attached"


def test_platform_exporter_status_degrades_on_partial_success_and_recovers(isolated_tracing):
    partial = ExportTraceServiceResponse()
    partial.partial_success.rejected_spans = 1
    partial_body = partial.SerializeToString()
    calls = {"n": 0}

    def handler(_body):
        calls["n"] += 1
        return _Response(200, content=partial_body) if calls["n"] == 1 else _Response(200)

    exporter, _client = _exporter(handler)
    isolated_tracing._platform_exporter = exporter
    isolated_tracing._platform_exporter_attached = True

    assert isolated_tracing.tracing_status()["platform_exporter"] == "attached"

    # The OE accepted the batch but dropped a span: the export still succeeds,
    # yet /health must report the loss until a clean export clears it.
    assert exporter.export(_make_spans(2)) == SpanExportResult.SUCCESS
    assert isolated_tracing.tracing_status()["platform_exporter"] == "degraded"

    assert exporter.export(_make_spans(2)) == SpanExportResult.SUCCESS
    assert isolated_tracing.tracing_status()["platform_exporter"] == "attached"


def test_force_flush_tracing_is_true_when_uninitialized(isolated_tracing):
    assert isolated_tracing.force_flush_tracing() is True


# ---------------------------------------------------------------------------
# TenantRuntime._setup_tracing() wiring
# ---------------------------------------------------------------------------


def _capture_runtime_setup_calls(monkeypatch) -> list[dict]:
    """Capture the setup_tracing kwargs a TenantRuntime construction uses."""
    monkeypatch.delenv("MONGODB_URI", raising=False)
    calls: list[dict] = []
    monkeypatch.setattr(
        "agent_engine_runner_shared.tracing.setup_tracing", lambda **kw: calls.append(kw)
    )
    return calls


def test_runtime_setup_tracing_skips_the_mongo_store_when_flagged(monkeypatch):
    """With the platform exporter flagged on and OE_URL set, the guest routes
    spans to the OE and must not open the trace-store Mongo connection — the
    double-write this change exists to prevent."""
    import pymongo

    import agent_engine_runner_shared.tracing as tracing_pkg

    monkeypatch.setenv("OE_URL", "http://oe:8000")
    monkeypatch.setenv(_PLATFORM_FLAG, "1")

    setup_calls: list[dict] = []
    monkeypatch.setattr(tracing_pkg, "setup_tracing", lambda **kwargs: setup_calls.append(kwargs))

    mongo_calls = {"n": 0}

    def _unexpected_mongo_client(*_args, **_kwargs):
        mongo_calls["n"] += 1
        raise RuntimeError("opened a Mongo trace-store client despite the platform exporter")

    monkeypatch.setattr(pymongo, "MongoClient", _unexpected_mongo_client)

    runtime = TenantRuntime.__new__(TenantRuntime)
    runtime.app_name = "svc"
    runtime._mongodb_uri = "mongodb://trace-store:27017"
    runtime._database_name_override = None
    runtime._traces_collection_name = "traces"

    runtime._setup_tracing()

    assert mongo_calls["n"] == 0
    assert setup_calls == [{"service_name": "svc"}]


def test_runtime_keeps_mongodb_path_when_flag_off(monkeypatch):
    """A live runner (OE_URL set) without the flag must not take the OE path."""
    calls = _capture_runtime_setup_calls(monkeypatch)
    monkeypatch.setenv("OE_URL", "http://oe:8000")
    monkeypatch.delenv(_PLATFORM_FLAG, raising=False)

    TenantRuntime(app_name="svc-default")

    # The MongoDB path passes the collection kwarg; the OE path does not.
    assert len(calls) == 1
    assert calls[0]["mongodb_collection"] is None
    assert calls[0]["take_client_ownership"] is True
