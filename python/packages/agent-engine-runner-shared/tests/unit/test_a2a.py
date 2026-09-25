"""Tests for the AgentToAgent client."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import httpx
import pytest
from opentelemetry import baggage, trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.context import attach, detach
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.propagate import get_global_textmap, set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from agent_engine_runner_shared.a2a import AgentResponse, AgentToAgent

DISCOVER_RESPONSE = {
    "agents": [
        {
            "agent_id": "ws-jokes",
            "name": "Joke Agent",
            "description": "Tells jokes",
            "project_id": "proj-1",
            "skills": [
                {
                    "name": "tell-joke",
                    "description": "Tell a joke",
                    "example_input": '{"topic": "python"}',
                    "example_output": "Why do pythons...",
                }
            ],
            "capabilities": ["humor"],
            "input_modes": ["text/plain"],
            "output_modes": ["text/plain"],
        }
    ]
}

INVOKE_RESPONSE = {
    "status": "completed",
    "result": "Here is a joke!",
    "error": None,
    "execution_id": "exec-123",
}


class TestAgentToAgentInit:
    def test_strips_trailing_slash(self):
        client = AgentToAgent(oe_url="http://localhost:8000/", auth_token="tok")
        assert client._oe_url == "http://localhost:8000"

    def test_sets_auth_header(self):
        client = AgentToAgent(oe_url="http://localhost:8000", auth_token="my-jwt")
        http_client = client._ensure_client()
        assert http_client.headers["authorization"] == "Bearer my-jwt"

    def test_no_auth_header_when_empty(self):
        client = AgentToAgent(oe_url="http://localhost:8000")
        http_client = client._ensure_client()
        assert "authorization" not in http_client.headers

    def test_close(self):
        client = AgentToAgent(oe_url="http://localhost:8000")
        client._ensure_client()
        assert client._client is not None
        client.close()
        assert client._client is None


class TestDiscoverAgents:
    def test_parses_response(self):
        client = AgentToAgent(oe_url="http://test:8000")
        mock_resp = MagicMock()
        mock_resp.json.return_value = DISCOVER_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = MagicMock()
            mock_http.get.return_value = mock_resp
            mock_ensure.return_value = mock_http

            agents = client.discover_agents()

        assert len(agents) == 1
        assert agents[0].agent_id == "ws-jokes"
        assert agents[0].name == "Joke Agent"
        assert agents[0].project_id == "proj-1"
        assert len(agents[0].skills) == 1
        assert agents[0].skills[0].name == "tell-joke"
        assert agents[0].skills[0].example_input == '{"topic": "python"}'
        assert agents[0].capabilities == ["humor"]

    def test_passes_filters_as_params(self):
        client = AgentToAgent(oe_url="http://test:8000")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"agents": []}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = MagicMock()
            mock_http.get.return_value = mock_resp
            mock_ensure.return_value = mock_http

            client.discover_agents(
                project_id="proj-x",
                skills=["summarize"],
                capabilities=["nlp"],
                input_modes=["text/plain"],
                limit=10,
            )

            call_kwargs = mock_http.get.call_args
            params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params")
            assert params["project_id"] == "proj-x"
            assert params["skills"] == ["summarize"]
            assert params["capabilities"] == ["nlp"]
            assert params["input_modes"] == ["text/plain"]
            assert params["limit"] == 10

    def test_returns_empty_list_when_no_agents(self):
        client = AgentToAgent(oe_url="http://test:8000")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"agents": []}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = MagicMock()
            mock_http.get.return_value = mock_resp
            mock_ensure.return_value = mock_http

            agents = client.discover_agents()

        assert agents == []

    def test_raises_on_http_error(self):
        client = AgentToAgent(oe_url="http://test:8000")

        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = MagicMock()
            mock_http.get.side_effect = httpx.HTTPStatusError(
                "401", request=MagicMock(), response=MagicMock()
            )
            mock_ensure.return_value = mock_http

            with pytest.raises(httpx.HTTPStatusError):
                client.discover_agents()


class TestInvokeAgent:
    def test_parses_completed_response(self):
        client = AgentToAgent(oe_url="http://test:8000")
        mock_resp = MagicMock()
        mock_resp.json.return_value = INVOKE_RESPONSE
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = MagicMock()
            mock_http.post.return_value = mock_resp
            mock_ensure.return_value = mock_http

            result = client.invoke_agent(agent_id="ws-jokes", message="tell me a joke")

        assert isinstance(result, AgentResponse)
        assert result.status == "completed"
        assert result.result == "Here is a joke!"
        assert result.error is None
        assert result.execution_id == "exec-123"

    def test_sends_correct_body(self):
        client = AgentToAgent(oe_url="http://test:8000")
        mock_resp = MagicMock()
        mock_resp.json.return_value = INVOKE_RESPONSE
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = MagicMock()
            mock_http.post.return_value = mock_resp
            mock_ensure.return_value = mock_http

            client.invoke_agent(
                agent_id="ws-target", message="hello", skill="summarize", timeout=60.0
            )

            call_kwargs = mock_http.post.call_args
            body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert body["target_agent_id"] == "ws-target"
            assert body["message"] == "hello"
            assert body["skill"] == "summarize"
            assert body["timeout_seconds"] == 60.0

    def test_omits_optional_fields_when_empty(self):
        client = AgentToAgent(oe_url="http://test:8000")
        mock_resp = MagicMock()
        mock_resp.json.return_value = INVOKE_RESPONSE
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = MagicMock()
            mock_http.post.return_value = mock_resp
            mock_ensure.return_value = mock_http

            client.invoke_agent(agent_id="ws-target", message="hello")

            call_kwargs = mock_http.post.call_args
            body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert "skill" not in body
            assert "timeout_seconds" not in body

    def test_parses_error_response(self):
        client = AgentToAgent(oe_url="http://test:8000")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": "failed",
            "error": "agent crashed",
            "execution_id": "exec-456",
        }
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = MagicMock()
            mock_http.post.return_value = mock_resp
            mock_ensure.return_value = mock_http

            result = client.invoke_agent(agent_id="ws-target", message="hello")

        assert result.status == "failed"
        assert result.error == "agent crashed"
        assert result.result is None

    def test_raises_on_http_error(self):
        client = AgentToAgent(oe_url="http://test:8000")

        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = MagicMock()
            mock_http.post.side_effect = httpx.HTTPStatusError(
                "502", request=MagicMock(), response=MagicMock()
            )
            mock_ensure.return_value = mock_http

            with pytest.raises(httpx.HTTPStatusError):
                client.invoke_agent(agent_id="ws-target", message="hello")

    def test_sends_custom_headers(self):
        client = AgentToAgent(oe_url="http://test:8000")
        mock_resp = MagicMock()
        mock_resp.json.return_value = INVOKE_RESPONSE
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = MagicMock()
            mock_http.post.return_value = mock_resp
            mock_ensure.return_value = mock_http

            client.invoke_agent(
                agent_id="ws-target",
                message="hello",
                custom_headers={"oauth-token": "abc", "x-req-id": "123"},
            )

            call_kwargs = mock_http.post.call_args
            body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert body["custom_headers"] == {"oauth-token": "abc", "x-req-id": "123"}

    def test_omits_custom_headers_when_none(self):
        client = AgentToAgent(oe_url="http://test:8000")
        mock_resp = MagicMock()
        mock_resp.json.return_value = INVOKE_RESPONSE
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = MagicMock()
            mock_http.post.return_value = mock_resp
            mock_ensure.return_value = mock_http

            client.invoke_agent(agent_id="ws-target", message="hello")

            call_kwargs = mock_http.post.call_args
            body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert "custom_headers" not in body


class _RecordingA2AHandler(BaseHTTPRequestHandler):
    """Stands in for OE's /a2a/invoke — records inbound headers so tests can
    assert on what the real httpx transport put on the wire."""

    received_headers: httpx.Headers | None = None
    response_status: int = 200
    invoke_fail_times: int = 0
    invoke_hits: int = 0
    invoke_omit_retry_after: bool = False
    fail_502_with_retry_after: bool = False

    def do_POST(self) -> None:
        type(self).received_headers = httpx.Headers(dict(self.headers))
        type(self).invoke_hits += 1
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if type(self).fail_502_with_retry_after:
            body = json.dumps({"error": "gateway"}).encode()
            self.send_response(502)
            self.send_header("Retry-After", "1")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if type(self).invoke_fail_times > 0:
            type(self).invoke_fail_times -= 1
            body = json.dumps({"error": "busy"}).encode()
            self.send_response(503)
            if not type(self).invoke_omit_retry_after:
                self.send_header("Retry-After", "1")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = json.dumps(INVOKE_RESPONSE).encode()
        self.send_response(type(self).response_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def fake_oe_server():
    """A real loopback HTTP server standing in for OE.

    A real server is required here (not httpx.MockTransport):
    HTTPXClientInstrumentor patches httpx.HTTPTransport.handle_request /
    AsyncHTTPTransport.handle_async_request specifically, and
    httpx.MockTransport subclasses BaseTransport directly rather than
    HTTPTransport, so a MockTransport-backed client never reaches the
    patched methods and would never get a traceparent header injected.
    """
    _RecordingA2AHandler.received_headers = None
    _RecordingA2AHandler.response_status = 200
    _RecordingA2AHandler.invoke_fail_times = 0
    _RecordingA2AHandler.invoke_hits = 0
    _RecordingA2AHandler.invoke_omit_retry_after = False
    _RecordingA2AHandler.fail_502_with_retry_after = False
    server = HTTPServer(("127.0.0.1", 0), _RecordingA2AHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


class TestInvokeAgentTraceContext:
    """A2A platform RPCs carry W3C context without propagating Baggage."""

    def setup_method(self):
        HTTPXClientInstrumentor().instrument()

    def teardown_method(self):
        HTTPXClientInstrumentor().uninstrument()

    def test_injects_traceparent_on_outbound_request(self, monkeypatch, fake_oe_server):
        provider = TracerProvider()
        monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
        prev_propagator = get_global_textmap()
        set_global_textmap(TraceContextTextMapPropagator())
        try:
            host, port = fake_oe_server.server_address
            client = AgentToAgent(oe_url=f"http://{host}:{port}")

            tracer = provider.get_tracer("test")
            with tracer.start_as_current_span("test-parent"):
                client.invoke_agent(agent_id="ws-target", message="hello")
        finally:
            set_global_textmap(prev_propagator)

        assert _RecordingA2AHandler.received_headers is not None
        assert "traceparent" in _RecordingA2AHandler.received_headers

    def test_no_baggage_header_leaks_when_baggage_is_set(self, monkeypatch, fake_oe_server):
        """Platform RPCs enforce W3C-only propagation at their own boundary."""
        provider = TracerProvider()
        monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
        prev_propagator = get_global_textmap()
        set_global_textmap(
            CompositePropagator([TraceContextTextMapPropagator(), W3CBaggagePropagator()])
        )
        ctx = baggage.set_baggage("tenant.secret", "should-not-leak")
        token = attach(ctx)
        try:
            host, port = fake_oe_server.server_address
            client = AgentToAgent(oe_url=f"http://{host}:{port}")
            tracer = provider.get_tracer("test")
            with tracer.start_as_current_span("test-parent"):
                client.invoke_agent(agent_id="ws-target", message="hello")
        finally:
            detach(token)
            set_global_textmap(prev_propagator)

        assert _RecordingA2AHandler.received_headers is not None
        assert "traceparent" in _RecordingA2AHandler.received_headers
        assert "baggage" not in _RecordingA2AHandler.received_headers

    def test_injects_traceparent_even_when_oe_returns_error(self, monkeypatch, fake_oe_server):
        """Trace context must reach OE's /a2a/invoke before raise_for_status
        raises, so the resulting error is still traceable to its caller."""
        _RecordingA2AHandler.response_status = 500
        provider = TracerProvider()
        monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
        prev_propagator = get_global_textmap()
        set_global_textmap(TraceContextTextMapPropagator())
        try:
            host, port = fake_oe_server.server_address
            client = AgentToAgent(oe_url=f"http://{host}:{port}")

            tracer = provider.get_tracer("test")
            with tracer.start_as_current_span("test-parent"):
                with pytest.raises(httpx.HTTPStatusError):
                    client.invoke_agent(agent_id="ws-target", message="hello")
        finally:
            set_global_textmap(prev_propagator)

        assert _RecordingA2AHandler.received_headers is not None
        assert "traceparent" in _RecordingA2AHandler.received_headers

    def test_platform_client_propagates_without_global_httpx_instrumentation(
        self, monkeypatch, fake_oe_server
    ):
        """Platform RPC propagation does not depend on span-producing instrumentation."""
        HTTPXClientInstrumentor().uninstrument()
        try:
            provider = TracerProvider()
            monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
            prev_propagator = get_global_textmap()
            set_global_textmap(TraceContextTextMapPropagator())
            try:
                host, port = fake_oe_server.server_address
                client = AgentToAgent(oe_url=f"http://{host}:{port}")

                tracer = provider.get_tracer("test")
                with tracer.start_as_current_span("test-parent"):
                    client.invoke_agent(agent_id="ws-target", message="hello")
            finally:
                set_global_textmap(prev_propagator)
        finally:
            HTTPXClientInstrumentor().instrument()

        assert _RecordingA2AHandler.received_headers is not None
        assert "traceparent" in _RecordingA2AHandler.received_headers


class TestInvokeAgentRetry:
    def test_retries_503_honoring_retry_after(self, fake_oe_server, monkeypatch):
        host, port = fake_oe_server.server_address
        _RecordingA2AHandler.invoke_fail_times = 1
        sleeps: list[float] = []
        monkeypatch.setattr("agent_engine_runner_shared.a2a.time.sleep", sleeps.append)
        client = AgentToAgent(oe_url=f"http://{host}:{port}")
        res = client.invoke_agent(agent_id="ws-target", message="hello")
        assert res.status == "completed"
        assert res.result == "Here is a joke!"
        assert _RecordingA2AHandler.invoke_hits == 2
        assert sleeps == [1.0], "the Retry-After hint must be honored"

    def test_raises_after_exhausted_503s(self, fake_oe_server, monkeypatch):
        host, port = fake_oe_server.server_address
        _RecordingA2AHandler.invoke_fail_times = 2
        sleeps: list[float] = []
        monkeypatch.setattr("agent_engine_runner_shared.a2a.time.sleep", sleeps.append)
        client = AgentToAgent(oe_url=f"http://{host}:{port}")
        with pytest.raises(httpx.HTTPStatusError):
            client.invoke_agent(agent_id="ws-target", message="hello")
        assert _RecordingA2AHandler.invoke_hits == 2
        assert sleeps == [1.0], "only the cooperative wait is taken; the final attempt raises"

    def test_does_not_retry_4xx(self, fake_oe_server):
        host, port = fake_oe_server.server_address
        _RecordingA2AHandler.response_status = 404
        client = AgentToAgent(oe_url=f"http://{host}:{port}")
        with pytest.raises(httpx.HTTPStatusError):
            client.invoke_agent(agent_id="ws-target", message="hello")
        assert _RecordingA2AHandler.invoke_hits == 1

    def test_does_not_retry_503_without_retry_after(self, fake_oe_server):
        host, port = fake_oe_server.server_address
        _RecordingA2AHandler.invoke_fail_times = 1
        _RecordingA2AHandler.invoke_omit_retry_after = True
        client = AgentToAgent(oe_url=f"http://{host}:{port}")
        with pytest.raises(httpx.HTTPStatusError):
            client.invoke_agent(agent_id="ws-target", message="hello")
        assert _RecordingA2AHandler.invoke_hits == 1

    def test_does_not_retry_502_even_with_retry_after(self, fake_oe_server):
        host, port = fake_oe_server.server_address
        _RecordingA2AHandler.fail_502_with_retry_after = True
        client = AgentToAgent(oe_url=f"http://{host}:{port}")
        with pytest.raises(httpx.HTTPStatusError):
            client.invoke_agent(agent_id="ws-target", message="hello")
        assert _RecordingA2AHandler.invoke_hits == 1, "a 502 is never retried"

    def test_does_not_retry_read_error_after_delivery(self):
        client = AgentToAgent(oe_url="http://test:8000")
        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_http = MagicMock()
            mock_http.post.side_effect = httpx.ReadError("connection lost")
            mock_ensure.return_value = mock_http
            with pytest.raises(httpx.ReadError):
                client.invoke_agent(agent_id="ws-target", message="hello")
        assert mock_http.post.call_count == 1


class TestInvokeAgentRetrySpanStatus:
    """Exhausted A2A failures must record span ERROR, or the trace UI shows
    successful subagent calls for failed relays."""

    def setup_method(self):
        HTTPXClientInstrumentor().instrument()

    def teardown_method(self):
        HTTPXClientInstrumentor().uninstrument()

    def test_exhausted_503_records_error(self, monkeypatch, fake_oe_server):
        provider = TracerProvider()
        monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
        in_mem = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(in_mem))
        host, port = fake_oe_server.server_address
        _RecordingA2AHandler.invoke_fail_times = 2
        sleeps: list[float] = []
        monkeypatch.setattr("agent_engine_runner_shared.a2a.time.sleep", sleeps.append)
        client = AgentToAgent(oe_url=f"http://{host}:{port}")
        with pytest.raises(httpx.HTTPStatusError):
            client.invoke_agent(agent_id="ws-target", message="hello")
        spans = in_mem.get_finished_spans()
        a2a_spans = [s for s in spans if s.name == "a2a.invoke_agent"]
        assert a2a_spans, "the a2a span must be emitted on failure"
        assert a2a_spans[-1].status.status_code == trace.StatusCode.ERROR
