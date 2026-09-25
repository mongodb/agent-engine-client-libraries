"""
OTLP/HTTP wire-format integration test.

Makes a real HTTP call to a local receiver, so it lives here rather than in
tests/unit — see docs/testing.md ("Mock external dependencies (DB, LLM
providers, HTTP calls)").
"""

import http.server
import threading

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode, TraceFlags

from agent_engine_runner_shared.tracing import ContentPolicyOTLPSpanExporter


class _CapturingOTLPHandler(http.server.BaseHTTPRequestHandler):
    received: list[dict] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.__class__.received.append(
            {
                "path": self.path,
                "content_type": self.headers.get("Content-Type"),
                "body_len": len(body),
            }
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.end_headers()
        self.wfile.write(b"")

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass


def test_otlp_exporter_sends_real_http_post():
    """Integration-style test: a real OTLPSpanExporter POSTs OTLP/HTTP protobuf
    to a local receiver. Mirrors the Go module_integration_test.go pattern of
    asserting method/path/content-type against an httptest-style server."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    context = SpanContext(
        trace_id=0x1, span_id=0x1, is_remote=False, trace_flags=TraceFlags.SAMPLED
    )
    real_span = ReadableSpan(
        name="llm_call",
        context=context,
        parent=None,
        resource=Resource.create({"service.name": "test_service"}),
        attributes={"llm.model_name": "gpt-4", "input.value": "secret prompt"},
        events=(),
        links=(),
        kind=SpanKind.INTERNAL,
        status=Status(StatusCode.OK),
        start_time=1000,
        end_time=2000,
        instrumentation_scope=None,
    )

    _CapturingOTLPHandler.received = []
    server = http.server.HTTPServer(("127.0.0.1", 0), _CapturingOTLPHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        exporter = OTLPSpanExporter(endpoint=f"http://127.0.0.1:{port}/v1/traces")
        wrapped = ContentPolicyOTLPSpanExporter(exporter)

        result = wrapped.export([real_span])

        assert result == SpanExportResult.SUCCESS
        assert len(_CapturingOTLPHandler.received) == 1
        received = _CapturingOTLPHandler.received[0]
        assert received["path"] == "/v1/traces"
        assert received["content_type"] == "application/x-protobuf"
        assert received["body_len"] > 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
