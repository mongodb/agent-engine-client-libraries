"""AER-side inbound W3C trace-context extraction.

`_execute_with_inbound_trace_context` is the thin wrapper the /execute route
calls instead of `_handle_execute` directly. Without it, this AER's own spans
-- and the trace_id persisted onto ExecutionStep/tool-call records via
`get_current_trace_context()` -- start a disconnected root trace even though
the caller's traceparent header now arrives correctly over the A2A hop.
"""

from __future__ import annotations

import sys
from typing import Optional
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent_engine_runner_shared.models import AERExecuteResponse, ExecuteRequest
from agent_engine_runner_shared.server.aer import AERServer
from agent_engine_runner_shared.span_kinds import OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind
from agent_engine_runner_shared.tracing.setup import get_current_trace_context


def _make_server() -> AERServer:
    runtime = Mock()
    return AERServer(runtime)


def _make_request() -> ExecuteRequest:
    return ExecuteRequest(
        execution_id="exec-1",
        message="hello",
        platform_api_url="http://oe:8000",
    )


@pytest.fixture
def in_memory_tracer(monkeypatch):
    """Install a real in-memory TracerProvider so spans are actually created."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    return exporter


class TestExecuteWithInboundTraceContext:
    @pytest.mark.asyncio
    async def test_valid_traceparent_joins_callers_trace(self, in_memory_tracer):
        server = _make_server()
        captured: dict[str, Optional[str]] = {}

        async def fake_handle_execute(request: ExecuteRequest) -> AERExecuteResponse:
            trace_id, _ = get_current_trace_context()
            captured["trace_id"] = trace_id
            return AERExecuteResponse(status="completed")

        server._handle_execute = fake_handle_execute

        caller_trace_id = "0af7651916cd43dd8448eb211c80319c"
        headers = httpx.Headers({"traceparent": f"00-{caller_trace_id}-b7ad6b7169203331-01"})

        response = await server._execute_with_inbound_trace_context(_make_request(), headers)

        assert response.status == "completed"
        assert captured["trace_id"] == caller_trace_id, (
            "AER's own trace_id (used for ExecutionStep/tool-call persistence) "
            "must join the caller's trace, not start a disconnected one"
        )

    @pytest.mark.asyncio
    async def test_missing_traceparent_still_succeeds_with_fresh_root_span(self, in_memory_tracer):
        server = _make_server()
        captured: dict[str, Optional[str]] = {}

        async def fake_handle_execute(request: ExecuteRequest) -> AERExecuteResponse:
            trace_id, _ = get_current_trace_context()
            captured["trace_id"] = trace_id
            return AERExecuteResponse(status="completed")

        server._handle_execute = fake_handle_execute

        response = await server._execute_with_inbound_trace_context(
            _make_request(), httpx.Headers({})
        )

        assert response.status == "completed"
        assert captured["trace_id"] is not None, (
            "a missing traceparent must still root a valid trace"
        )

    @pytest.mark.asyncio
    async def test_span_declares_agent_kind(self, in_memory_tracer):
        """The AER span is the outermost step of an invocation, so it declares
        AGENT rather than leaving a receiving backend to guess from the name."""
        server = _make_server()
        server._handle_execute = AsyncMock(return_value=AERExecuteResponse(status="completed"))

        headers = httpx.Headers(
            {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"}
        )

        await server._execute_with_inbound_trace_context(_make_request(), headers)

        spans = in_memory_tracer.get_finished_spans()
        assert [s.name for s in spans] == ["aer.execute"]
        assert spans[0].attributes[OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKind.AGENT.value

    @pytest.mark.asyncio
    async def test_malformed_traceparent_does_not_crash(self, in_memory_tracer):
        server = _make_server()
        server._handle_execute = AsyncMock(return_value=AERExecuteResponse(status="completed"))

        headers = httpx.Headers({"traceparent": "not-a-valid-traceparent"})

        response = await server._execute_with_inbound_trace_context(_make_request(), headers)

        assert response.status == "completed"
        server._handle_execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_header_lookup_is_case_insensitive(self, in_memory_tracer):
        server = _make_server()
        captured: dict[str, Optional[str]] = {}

        async def fake_handle_execute(request: ExecuteRequest) -> AERExecuteResponse:
            trace_id, _ = get_current_trace_context()
            captured["trace_id"] = trace_id
            return AERExecuteResponse(status="completed")

        server._handle_execute = fake_handle_execute

        caller_trace_id = "0af7651916cd43dd8448eb211c80319c"
        # httpx.Headers is already case-insensitive; a plain dict with an
        # unusual casing exercises the same normalization the FastAPI route
        # relies on Starlette's Headers for.
        headers = httpx.Headers({"Traceparent": f"00-{caller_trace_id}-b7ad6b7169203331-01"})

        response = await server._execute_with_inbound_trace_context(_make_request(), headers)

        assert response.status == "completed"
        assert captured["trace_id"] == caller_trace_id

    @pytest.mark.asyncio
    async def test_falls_back_to_untraced_execution_when_opentelemetry_unavailable(
        self, monkeypatch
    ):
        # Simulate the optional `tracing` extra not being installed by
        # forcing the module import inside the try block to fail, without
        # touching the real (already-imported) opentelemetry package used by
        # the rest of this test module.
        monkeypatch.setitem(sys.modules, "opentelemetry.trace.propagation.tracecontext", None)

        server = _make_server()
        server._handle_execute = AsyncMock(return_value=AERExecuteResponse(status="completed"))

        headers = httpx.Headers(
            {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"}
        )

        response = await server._execute_with_inbound_trace_context(_make_request(), headers)

        assert response.status == "completed"
        server._handle_execute.assert_awaited_once()
