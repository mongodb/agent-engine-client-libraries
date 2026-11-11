"""The Python platform exporter sends the payload the TypeScript one sends.

Both runner test suites decode their own request and compare it with the shared
fixture, so a field one runner drops or renames fails here instead of in the OE.
"""

from __future__ import annotations

import json
from pathlib import Path

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Span as ProtoSpan
from opentelemetry.proto.trace.v1.trace_pb2 import Status as ProtoStatus
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode, TraceFlags

from agent_engine_runner_shared.tracing.exporters import PlatformSpanExporter

_FIXTURE = (
    Path(__file__).resolve().parents[5] / "shared-test-fixtures" / "runner_trace_payload.json"
)


_PROJECT_KEY = "agentic_platform.project_id"  # open-source-refs:ignore — data contract


class _Response:
    status_code = 200
    headers: dict = {}
    content = b""


class _Client:
    def __init__(self) -> None:
        self.bodies: list[bytes] = []

    async def post(self, url, *, content, headers, timeout):
        self.bodies.append(content)
        return _Response()


def _fixed_span() -> ReadableSpan:
    return ReadableSpan(
        name="agent.run",
        context=SpanContext(
            trace_id=0x0AF7651916CD43DD8448EB211C80319C,
            span_id=0xB7AD6B7169203331,
            is_remote=False,
            trace_flags=TraceFlags(1),
        ),
        parent=SpanContext(
            trace_id=0x0AF7651916CD43DD8448EB211C80319C,
            span_id=0x00F067AA0BA902B7,
            is_remote=False,
        ),
        resource=Resource({"service.name": "runner-sdk", _PROJECT_KEY: "proj-1"}),
        attributes={
            "session.id": "sess-1",
            "execution.id": "exec-1",
            "thread.id": "sess-1",
            "workspace.id": "ws-1",
            "llm.token_count.total": 42,
            "custom.flag": True,
            "ratio": 0.5,
            "user.note": "contact jane@x.com",
        },
        kind=SpanKind.INTERNAL,
        status=Status(StatusCode.OK),
        start_time=1700000000000000001,
        end_time=1700000001500000002,
    )


def _any_value(value):
    return getattr(value, value.WhichOneof("value"))


def _attributes(pairs) -> dict:
    return {pair.key: _any_value(pair.value) for pair in pairs}


def _normalize(body: bytes) -> dict:
    request = ExportTraceServiceRequest()
    request.ParseFromString(body)
    spans = []
    for resource_spans in request.resource_spans:
        resource = _attributes(resource_spans.resource.attributes)
        for scope_spans in resource_spans.scope_spans:
            for span in scope_spans.spans:
                spans.append(
                    {
                        "trace_id": span.trace_id.hex(),
                        "span_id": span.span_id.hex(),
                        "parent_span_id": span.parent_span_id.hex(),
                        "name": span.name,
                        "kind": ProtoSpan.SpanKind.Name(span.kind),
                        "start_time_unix_nano": str(span.start_time_unix_nano),
                        "end_time_unix_nano": str(span.end_time_unix_nano),
                        "status_code": ProtoStatus.StatusCode.Name(span.status.code),
                        "attributes": _attributes(span.attributes),
                        "resource": resource,
                    }
                )
    return {"spans": spans}


def test_platform_payload_matches_the_shared_fixture():
    client = _Client()

    async def factory(_base: str):
        return client

    exporter = PlatformSpanExporter(
        "http://oe:8000", client_factory=factory, cert_mtimes=lambda: None
    )

    assert exporter.export([_fixed_span()]) == SpanExportResult.SUCCESS

    expected = json.loads(_FIXTURE.read_text())
    assert len(client.bodies) == 1
    assert _normalize(client.bodies[0]) == {"spans": expected["spans"]}
