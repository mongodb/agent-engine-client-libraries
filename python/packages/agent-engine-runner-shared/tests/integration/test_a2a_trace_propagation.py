"""Connected-trace-tree verification across an A2A hop.

Drives a real SDK-side A2A invocation through a loopback HTTP server
standing in for OE's /a2a/invoke, simulates the OE-side child span that
would be created from the inbound traceparent, and exports both spans to a
loopback OTLP/HTTP receiver double — proving the acceptance criterion
("Agent A and Agent B spans share trace lineage") without depending on
the OE tracing module's production OTLP intake pipeline.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.propagate import get_global_textmap, set_global_textmap
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from agent_engine_runner_shared.a2a import AgentToAgent

INVOKE_RESPONSE = {
    "status": "completed",
    "result": "Here is joke!",
    "error": None,
    "execution_id": "exec-123",
}


class _RecordingA2AHandler(BaseHTTPRequestHandler):
    """Stands in for OE's /a2a/invoke, recording the inbound headers."""

    received_headers: httpx.Headers | None = None
    response_status: int = 200

    def do_POST(self) -> None:
        type(self).received_headers = httpx.Headers(dict(self.headers))
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps(INVOKE_RESPONSE).encode()
        self.send_response(type(self).response_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


class _RecordingOTLPHandler(BaseHTTPRequestHandler):
    """Stands in for an OTLP/HTTP collector's /v1/traces endpoint."""

    exported_trace_ids: list[str] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        request = ExportTraceServiceRequest()
        request.ParseFromString(raw)
        for resource_span in request.resource_spans:
            for scope_span in resource_span.scope_spans:
                for span in scope_span.spans:
                    type(self).exported_trace_ids.append(span.trace_id.hex())

        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def fake_oe_server():
    _RecordingA2AHandler.received_headers = None
    _RecordingA2AHandler.response_status = 200
    server = HTTPServer(("127.0.0.1", 0), _RecordingA2AHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture
def fake_otlp_server():
    _RecordingOTLPHandler.exported_trace_ids = []
    server = HTTPServer(("127.0.0.1", 0), _RecordingOTLPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_a2a_hop_shares_trace_id_with_simulated_oe_span(
    monkeypatch, fake_oe_server, fake_otlp_server
):
    otlp_host, otlp_port = fake_otlp_server.server_address
    provider = TracerProvider()
    provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint=f"http://{otlp_host}:{otlp_port}/v1/traces"))
    )
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    prev_propagator = get_global_textmap()
    set_global_textmap(TraceContextTextMapPropagator())
    HTTPXClientInstrumentor().instrument()
    try:
        oe_host, oe_port = fake_oe_server.server_address
        client = AgentToAgent(oe_url=f"http://{oe_host}:{oe_port}")
        client.invoke_agent(agent_id="ws-target", message="hello")

        # Simulate OE's side: extract the inbound traceparent (this is what
        # OE's otelgin middleware does on /a2a/invoke) and start a child span
        # under that remote context, as OE's own tracer would.
        carrier = dict(_RecordingA2AHandler.received_headers)
        remote_ctx = TraceContextTextMapPropagator().extract(carrier=carrier)
        oe_tracer = provider.get_tracer("orchestration-engine.a2a")
        with oe_tracer.start_as_current_span("a2a.invoke", context=remote_ctx) as oe_span:
            oe_trace_id = format(oe_span.get_span_context().trace_id, "032x")

        provider.force_flush()
    finally:
        HTTPXClientInstrumentor().uninstrument()
        set_global_textmap(prev_propagator)

    assert _RecordingA2AHandler.received_headers is not None
    assert "traceparent" in _RecordingA2AHandler.received_headers

    exported_trace_ids = set(_RecordingOTLPHandler.exported_trace_ids)
    assert len(exported_trace_ids) == 1, (
        f"expected Agent A and Agent B spans to share one trace_id, got {exported_trace_ids}"
    )
    assert oe_trace_id in exported_trace_ids


def test_a2a_hop_shares_trace_id_with_simulated_oe_span_even_on_error(
    monkeypatch, fake_oe_server, fake_otlp_server
):
    """Trace lineage must survive an unhappy path: even when OE responds
    with an error, the inbound request it received still carries the
    caller's trace context."""
    _RecordingA2AHandler.response_status = 500
    otlp_host, otlp_port = fake_otlp_server.server_address
    provider = TracerProvider()
    provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint=f"http://{otlp_host}:{otlp_port}/v1/traces"))
    )
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    prev_propagator = get_global_textmap()
    set_global_textmap(TraceContextTextMapPropagator())
    HTTPXClientInstrumentor().instrument()
    try:
        oe_host, oe_port = fake_oe_server.server_address
        client = AgentToAgent(oe_url=f"http://{oe_host}:{oe_port}")

        with pytest.raises(httpx.HTTPStatusError):
            client.invoke_agent(agent_id="ws-target", message="hello")

        carrier = dict(_RecordingA2AHandler.received_headers)
        remote_ctx = TraceContextTextMapPropagator().extract(carrier=carrier)
        oe_tracer = provider.get_tracer("orchestration-engine.a2a")
        with oe_tracer.start_as_current_span("a2a.invoke", context=remote_ctx) as oe_span:
            oe_trace_id = format(oe_span.get_span_context().trace_id, "032x")

        provider.force_flush()
    finally:
        HTTPXClientInstrumentor().uninstrument()
        set_global_textmap(prev_propagator)

    assert _RecordingA2AHandler.received_headers is not None
    assert "traceparent" in _RecordingA2AHandler.received_headers

    exported_trace_ids = set(_RecordingOTLPHandler.exported_trace_ids)
    assert oe_trace_id in exported_trace_ids, (
        "trace lineage must survive OE returning an error response"
    )
