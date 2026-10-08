"""Tool-pod inbound W3C trace-context extraction.

`_run_with_inbound_trace_context` wraps the /execute, /invoke_llm, and
/guardrails/check handlers; `_stream_with_inbound_trace_context` wraps the
/invoke_llm/stream response generator. The OE injects a traceparent on every
tool-pod call; without these wrappers the spans produced while a tool or
routed LLM call runs (framework instrumentation, outbound httpx) root a
disconnected trace instead of joining the invocation's.
"""

from __future__ import annotations

import sys
from typing import Optional
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent_engine_runner_shared.server.tool import ToolServer
from agent_engine_runner_shared.span_kinds import OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind
from agent_engine_runner_shared.tracing.setup import get_current_trace_context

CALLER_TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
TRACEPARENT = f"00-{CALLER_TRACE_ID}-b7ad6b7169203331-01"


def _make_server() -> ToolServer:
    return ToolServer(Mock())


def _route_client() -> TestClient:
    """A tool pod with its real routes registered over a stubbed runtime."""
    runtime = Mock()
    runtime._tools = {}
    runtime._tool_definitions = {}
    # /execute resolves an unknown tool name against the MCP server list before
    # giving up, so the stub needs a real (empty) one to iterate.
    runtime.agent_config.mcp.servers = []
    app = FastAPI()
    ToolServer(runtime).register_routes(app)
    return TestClient(app)


_EXECUTE_PAYLOAD = {
    "execution_id": "exec-1",
    "tool_name": "unregistered_tool",
    "arguments": {},
    "session_id": "session-1",
}

_INVOKE_LLM_PAYLOAD = {
    "execution_id": "exec-1",
    "arguments": {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
    },
}

_GUARDRAILS_PAYLOAD = {
    "execution_id": "exec-1",
    "stage": "llm_output",
    "input": {"text": "hi", "metadata": {}},
    "context": {
        "org_id": "000000000000000000000001",
        "project_id": "000000000000000000000002",
        "workspace_id": "workspace-1",
        "session_id": "session-1",
        "user_id": "user-1",
    },
    "policies": [],
}


@pytest.fixture
def in_memory_tracer(monkeypatch):
    """Install a real in-memory TracerProvider so spans are actually created."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    return exporter


class TestRunWithInboundTraceContext:
    @pytest.mark.asyncio
    async def test_valid_traceparent_joins_callers_trace(self, in_memory_tracer):
        server = _make_server()
        captured: dict[str, Optional[str]] = {}

        async def handler() -> str:
            trace_id, _ = get_current_trace_context()
            captured["trace_id"] = trace_id
            return "ok"

        result = await server._run_with_inbound_trace_context(
            "tool.execute",
            OpenInferenceSpanKind.TOOL,
            httpx.Headers({"traceparent": TRACEPARENT}),
            handler,
        )

        assert result == "ok"
        assert captured["trace_id"] == CALLER_TRACE_ID, (
            "tool-pod spans must join the invocation's trace, not root a new one"
        )

    @pytest.mark.asyncio
    async def test_missing_traceparent_still_roots_a_valid_trace(self, in_memory_tracer):
        server = _make_server()
        captured: dict[str, Optional[str]] = {}

        async def handler() -> str:
            trace_id, _ = get_current_trace_context()
            captured["trace_id"] = trace_id
            return "ok"

        result = await server._run_with_inbound_trace_context(
            "tool.execute", OpenInferenceSpanKind.TOOL, httpx.Headers({}), handler
        )

        assert result == "ok"
        assert captured["trace_id"] is not None

    @pytest.mark.asyncio
    async def test_malformed_traceparent_does_not_crash(self, in_memory_tracer):
        server = _make_server()

        async def handler() -> str:
            return "ok"

        result = await server._run_with_inbound_trace_context(
            "tool.execute",
            OpenInferenceSpanKind.TOOL,
            httpx.Headers({"traceparent": "garbage"}),
            handler,
        )
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_falls_back_to_untraced_when_opentelemetry_unavailable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "opentelemetry.trace.propagation.tracecontext", None)
        server = _make_server()

        async def handler() -> str:
            return "ok"

        result = await server._run_with_inbound_trace_context(
            "tool.execute",
            OpenInferenceSpanKind.TOOL,
            httpx.Headers({"traceparent": TRACEPARENT}),
            handler,
        )
        assert result == "ok"


class TestDeclaredSpanKind:
    """Tool-pod spans declare their OpenInference kind.

    Without the attribute a receiving backend has only the span name to go on,
    so these spans arrive unclassified.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("span_name", "span_kind"),
        [
            ("tool.execute", OpenInferenceSpanKind.TOOL),
            ("tool.invoke_llm", OpenInferenceSpanKind.CHAIN),
            ("tool.guardrails_check", OpenInferenceSpanKind.TOOL),
        ],
    )
    async def test_handler_span_declares_its_kind(self, in_memory_tracer, span_name, span_kind):
        server = _make_server()

        async def handler() -> str:
            return "ok"

        await server._run_with_inbound_trace_context(
            span_name, span_kind, httpx.Headers({"traceparent": TRACEPARENT}), handler
        )

        spans = in_memory_tracer.get_finished_spans()
        assert [s.name for s in spans] == [span_name]
        assert spans[0].attributes[OPENINFERENCE_SPAN_KIND] == span_kind.value

    @pytest.mark.asyncio
    async def test_stream_span_declares_its_kind(self, in_memory_tracer):
        server = _make_server()

        async def chunks():
            yield "a"

        _ = [
            c
            async for c in server._stream_with_inbound_trace_context(
                "tool.invoke_llm.stream",
                OpenInferenceSpanKind.CHAIN,
                httpx.Headers({"traceparent": TRACEPARENT}),
                chunks(),
            )
        ]

        spans = in_memory_tracer.get_finished_spans()
        assert spans[0].attributes[OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKind.CHAIN.value


class TestRouteDeclaredSpanKind:
    """Each route exports the kind it registers.

    The tests above hand the kind to the trace-context helper themselves, so
    they can only show the helper copies its argument onto the span. Which kind
    each endpoint *picks* is the part that reaches a customer's backend, and
    swapping two of them would leave every other test in this file green.

    The handlers all fail on this stubbed runtime and return their error shape
    rather than raising; the span is recorded either way, and its kind is fixed
    at registration, so the failure is irrelevant to what is asserted here.
    """

    @pytest.mark.parametrize(
        ("path", "payload", "span_name", "expected_kind"),
        [
            ("/execute", _EXECUTE_PAYLOAD, "tool.execute", OpenInferenceSpanKind.TOOL),
            ("/invoke_llm", _INVOKE_LLM_PAYLOAD, "tool.invoke_llm", OpenInferenceSpanKind.CHAIN),
            (
                "/invoke_llm/stream",
                _INVOKE_LLM_PAYLOAD,
                "tool.invoke_llm.stream",
                OpenInferenceSpanKind.CHAIN,
            ),
            (
                "/guardrails/check",
                _GUARDRAILS_PAYLOAD,
                "tool.guardrails_check",
                OpenInferenceSpanKind.TOOL,
            ),
        ],
    )
    def test_route_span_declares_its_kind(
        self, in_memory_tracer, path, payload, span_name, expected_kind
    ):
        client = _route_client()

        response = client.post(path, json=payload, headers={"traceparent": TRACEPARENT})

        assert response.status_code == 200, response.text
        spans = {span.name: span for span in in_memory_tracer.get_finished_spans()}
        assert span_name in spans, f"{path} produced spans {sorted(spans)}"
        assert spans[span_name].attributes[OPENINFERENCE_SPAN_KIND] == expected_kind.value

    def test_no_route_declares_llm(self):
        """The /invoke_llm hops are transport, not model calls: the LangChain
        model they drive is already wrapped in an instrumented LLM span, as is
        the caller's SecureWrappedLLM. A third LLM span carrying no model or
        token attributes would inflate LLM counts in the customer's backend."""
        assert "LLM" not in {kind.value for kind in OpenInferenceSpanKind}


class TestStreamWithInboundTraceContext:
    @pytest.mark.asyncio
    async def test_chunks_pass_through_under_callers_trace(self, in_memory_tracer):
        server = _make_server()
        seen_trace_ids: list[Optional[str]] = []

        async def chunks():
            for chunk in ("a", "b"):
                trace_id, _ = get_current_trace_context()
                seen_trace_ids.append(trace_id)
                yield chunk

        got = [
            c
            async for c in server._stream_with_inbound_trace_context(
                "tool.invoke_llm.stream",
                OpenInferenceSpanKind.CHAIN,
                httpx.Headers({"traceparent": TRACEPARENT}),
                chunks(),
            )
        ]

        assert got == ["a", "b"]
        assert seen_trace_ids == [CALLER_TRACE_ID, CALLER_TRACE_ID], (
            "spans created while producing stream chunks must join the caller's trace"
        )

    @pytest.mark.asyncio
    async def test_span_ends_when_stream_is_exhausted(self, in_memory_tracer):
        server = _make_server()

        async def chunks():
            yield "only"

        _ = [
            c
            async for c in server._stream_with_inbound_trace_context(
                "tool.invoke_llm.stream",
                OpenInferenceSpanKind.CHAIN,
                httpx.Headers({"traceparent": TRACEPARENT}),
                chunks(),
            )
        ]

        spans = in_memory_tracer.get_finished_spans()
        assert [s.name for s in spans] == ["tool.invoke_llm.stream"]
        assert trace.format_trace_id(spans[0].context.trace_id) == CALLER_TRACE_ID

    @pytest.mark.asyncio
    async def test_generator_error_still_ends_the_span(self, in_memory_tracer):
        server = _make_server()

        async def chunks():
            yield "first"
            raise RuntimeError("provider blew up")

        with pytest.raises(RuntimeError, match="provider blew up"):
            async for _ in server._stream_with_inbound_trace_context(
                "tool.invoke_llm.stream",
                OpenInferenceSpanKind.CHAIN,
                httpx.Headers({"traceparent": TRACEPARENT}),
                chunks(),
            ):
                pass

        spans = in_memory_tracer.get_finished_spans()
        assert [s.name for s in spans] == ["tool.invoke_llm.stream"], (
            "an erroring stream must still end (and export) its span"
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_plain_passthrough_when_opentelemetry_unavailable(
        self, monkeypatch
    ):
        monkeypatch.setitem(sys.modules, "opentelemetry.trace.propagation.tracecontext", None)
        server = _make_server()

        async def chunks():
            yield "a"

        got = [
            c
            async for c in server._stream_with_inbound_trace_context(
                "tool.invoke_llm.stream",
                OpenInferenceSpanKind.CHAIN,
                httpx.Headers({"traceparent": TRACEPARENT}),
                chunks(),
            )
        ]
        assert got == ["a"]
