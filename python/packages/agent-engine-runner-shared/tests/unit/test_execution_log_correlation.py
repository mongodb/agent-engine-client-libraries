"""Execution-ID attribution of AER and Tool Pod logs outside the execution context.

Customers filter runner logs by ``executionId``, so a line written while the
runtime works for a known execution must carry that ID even before
``set_execution_context`` or after ``clear_execution_context``: setup,
cleanup, metrics, background work and side requests. Each test finds its
lines by message text, not by ID, so a missing ID cannot hide a line.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Callable, Iterator, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from agent_engine_sdk.models import LLMStreamChunk
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_engine_runner_shared.context import (
    bind_log_execution_id,
    clear_execution_context,
    get_current_execution_id,
    request_session_finish,
    set_execution_context,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import WorkflowIdentity
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.models import (
    ExecuteRequest,
    LLMPodInvokeRequest,
    StreamingResult,
)
from agent_engine_runner_shared.server import callback_delivery as callback_delivery_module
from agent_engine_runner_shared.server.aer import AERServer
from agent_engine_runner_shared.server.callback_delivery import CallbackDelivery
from agent_engine_runner_shared.server.drain import DrainRegistry
from agent_engine_runner_shared.server.tool import ToolServer
from agent_engine_runner_shared.structured_logging import StructuredJSONFormatter
from agent_engine_runner_shared.workflow.heartbeat import AttemptHeartbeat

_AUTH = {"Authorization": "Bearer s3cret"}


class _JsonLogs(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(StructuredJSONFormatter(mode="aer"))
        self._records: List[dict] = []
        self._arrived = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def emit(self, record: logging.LogRecord) -> None:
        self._records.append(json.loads(self.format(record)))
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._arrived.set)

    def __call__(self) -> List[dict]:
        return list(self._records)

    async def wait_for(self, prefix: str) -> None:
        self._loop = asyncio.get_running_loop()
        async with asyncio.timeout(5):
            while True:
                self._arrived.clear()
                if any(r["message"].startswith(prefix) for r in self._records):
                    return
                await self._arrived.wait()


@pytest.fixture
def json_logs() -> Iterator[_JsonLogs]:
    handler = _JsonLogs()
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)


def _ids(records: List[dict], prefix: str) -> set:
    matching = [r for r in records if r["message"].startswith(prefix)]
    assert matching, f"no record starting with {prefix!r}"
    return {r["executionId"] for r in matching}


def _aer_server() -> AERServer:
    runtime = MagicMock()
    runtime.build_graph = AsyncMock(return_value=None)
    runtime._memory_writer = None
    runtime.org_id = "org-1"
    runtime.get_agent = MagicMock(return_value=MagicMock())

    server = AERServer.__new__(AERServer)
    server.runtime = runtime  # type: ignore[attr-defined]
    server._client_lock = asyncio.Lock()  # type: ignore[attr-defined]
    server._a2a_registered = False  # type: ignore[attr-defined]
    server._chunk_seq = {}
    server._durable_ack_senders = {}  # type: ignore[attr-defined]
    server._owner_callback_url = {}  # type: ignore[attr-defined]
    server._durability_owner_id = "owner-test"  # type: ignore[attr-defined]
    server._pending_session_finish = {}  # type: ignore[attr-defined]
    server._callback_delivery = CallbackDelivery(  # type: ignore[attr-defined]
        lambda url: server._get_client(url)  # type: ignore[attr-defined]
    )
    server._send_stream_chunk = AsyncMock()  # type: ignore[assignment]
    return server


def _aer_app(server: AERServer) -> FastAPI:
    app = FastAPI()
    server.register_routes(app)
    return app


def _execute_payload(execution_id: str) -> dict:
    return ExecuteRequest(
        execution_id=execution_id,
        message="hi",
        platform_api_url="http://oe",
        user_id="u-1",
        custom_headers={},
        resume=False,
    ).model_dump()


def test_aer_execute_tags_setup_cleanup_metric_and_session_finish_logs(json_logs) -> None:
    execution_id = "exec-lifecycle-1"
    server = _aer_server()
    context_at_callback: list = []

    async def agent_stream(*_args: Any, **_kwargs: Any) -> Any:
        request_session_finish()
        logging.getLogger("agent_probe").info("agent body line")
        return StreamingResult(content="ok", messages=[])

    async def callback_probe(*_args: Any, **_kwargs: Any) -> None:
        context_at_callback.append(get_current_execution_id())
        logging.getLogger("callback_probe").info("callback delivery line")

    async def failing_finish_post(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("oe unreachable")

    server._execute_via_agent_stream = agent_stream  # type: ignore[assignment]
    server._send_callback_body = callback_probe  # type: ignore[assignment]
    with (
        patch.object(AERServer, "_get_client", new=AsyncMock(return_value=MagicMock())),
        patch(
            "agent_engine_runner_shared.server.aer.post_json_with_retries",
            new=failing_finish_post,
        ),
    ):
        response = TestClient(_aer_app(server)).post(
            "/execute", json=_execute_payload(execution_id)
        )
    logging.getLogger("after_probe").info("after request")

    assert response.status_code == 200
    # The callback probe runs after the execution context has cleared.
    assert context_at_callback == [None]
    records = json_logs()
    for prefix in (
        "AER: Execute",
        "agent body line",
        "callback delivery line",
        "METRIC latency",
        "Session finish request failed",
    ):
        assert _ids(records, prefix) == {execution_id}, prefix
    assert _ids(records, "after request") == {None}


def test_aer_setup_failure_log_carries_the_execution_id_without_credentials(
    json_logs, monkeypatch
) -> None:
    monkeypatch.setenv("PAYMENTS_API_TOKEN", "tenant-token-value-123")
    server = _aer_server()
    server.runtime.build_graph = AsyncMock(
        side_effect=RuntimeError(
            "cannot reach mongodb+srv://agent:Pa55w0rd@cluster0.example.net/db "
            "with tenant-token-value-123"
        )
    )

    with patch.object(AERServer, "_get_client", new=AsyncMock(return_value=MagicMock())):
        response = TestClient(_aer_app(server), raise_server_exceptions=False).post(
            "/execute", json=_execute_payload("exec-setup-1")
        )

    assert response.status_code == 500
    prefix = "AER: execution exec-set setup failed"
    assert _ids(json_logs(), prefix) == {"exec-setup-1"}
    message = next(r["message"] for r in json_logs() if r["message"].startswith(prefix))
    assert "cluster0.example.net" in message
    assert "Pa55w0rd" not in message
    assert "tenant-token-value-123" not in message


async def test_concurrent_aer_executions_never_carry_each_other_id(json_logs) -> None:
    server = _aer_server()
    both_running = asyncio.Barrier(2)

    async def agent_stream(*_args: Any, **_kwargs: Any) -> Any:
        await both_running.wait()
        return StreamingResult(content="ok", messages=[])

    async def callback_probe(_oe_url: str, body: dict, **_kwargs: Any) -> None:
        await asyncio.sleep(0)
        logging.getLogger("callback_probe").info("callback for %s", body["execution_id"])

    server._execute_via_agent_stream = agent_stream  # type: ignore[assignment]
    server._send_callback_body = callback_probe  # type: ignore[assignment]
    transport = httpx.ASGITransport(app=_aer_app(server))
    with patch.object(AERServer, "_get_client", new=AsyncMock(return_value=MagicMock())):
        async with httpx.AsyncClient(transport=transport, base_url="http://aer") as client:
            responses = await asyncio.gather(
                client.post("/execute", json=_execute_payload("execA-concurrent")),
                client.post("/execute", json=_execute_payload("execB-concurrent")),
            )

    assert [r.status_code for r in responses] == [200, 200]
    records = json_logs()
    for execution_id in ("execA-concurrent", "execB-concurrent"):
        assert _ids(records, f"AER: Execute {execution_id[:8]}") == {execution_id}
        assert _ids(records, f"callback for {execution_id}") == {execution_id}


def test_heartbeat_thread_warning_carries_the_originating_execution_id(json_logs) -> None:
    sends = 0
    failed = threading.Event()

    def send(_request: Any) -> None:
        nonlocal sends
        sends += 1
        if sends > 1:
            failed.set()
            raise ConnectionError("OE unreachable")

    attempt = AttemptContext(
        attempt_id="attempt-1",
        fencing_token=7,
        owner_id="aer-1",
        workflow_identity=WorkflowIdentity(session_id="s-1", execution_id="exec-heartbeat-1"),
        heartbeat_interval_ms=1000,
    )
    heartbeat = AttemptHeartbeat(attempt=attempt, interval_seconds=0.01, send=send)
    tokens = set_execution_context("exec-heartbeat-1", None, "http://oe")
    try:
        heartbeat.start()
    finally:
        clear_execution_context(tokens)
    try:
        assert failed.wait(timeout=5.0)
    finally:
        heartbeat.stop()

    assert _ids(json_logs(), "Failed to renew durable attempt lease") == {"exec-heartbeat-1"}


async def test_callback_redelivery_names_the_retained_execution_not_the_caller(
    json_logs, monkeypatch
) -> None:
    async def unreachable(*_args: Any, **_kwargs: Any) -> None:
        raise httpx.ConnectError("oe down")

    monkeypatch.setattr(callback_delivery_module, "post_json_with_retries", unreachable)
    monkeypatch.setattr(callback_delivery_module, "CALLBACK_POST_BASE_DELAY_S", 0.01)
    delivery = CallbackDelivery(AsyncMock(return_value=MagicMock()))

    with bind_log_execution_id("exec-unrelated"):
        await delivery.send("http://oe", {"execution_id": "exec-retained", "status": "COMPLETED"})
    await json_logs.wait_for("Callback redelivery")
    await delivery.shutdown()

    assert _ids(json_logs(), "Callback redelivery for execution exec-retained") == {"exec-retained"}


async def test_drain_and_call_interrupt_logs_carry_the_execution_id(json_logs, monkeypatch) -> None:
    monkeypatch.setenv("RUNNER_AUTH_TOKEN", "s3cret")
    server = _tool_server()
    registry = server.drain_registry
    handle = registry.begin_work("exec-drained", step_number=1)
    transport = httpx.ASGITransport(app=server.create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://tool") as client:
        interrupt = await client.post(
            "/interrupt/call",
            json={"execution_id": "exec-drained", "step_number": 1},
            headers=_AUTH,
        )
        drain = await client.post(
            "/drain",
            json={
                "request_id": "drain-aaaa",
                "execution_id": "exec-drained",
                "reason": "execution_cancelled",
                "deadline_at_ms": int(time.time() * 1000) + 5000,
            },
            headers=_AUTH,
        )
        registry.end_work(handle)
        await json_logs.wait_for("Drain completed")

    assert (interrupt.status_code, drain.status_code) == (200, 202)
    records = json_logs()
    for prefix in ("Call interrupt", "Drain accepted", "Drain completed"):
        assert _ids(records, prefix) == {"exec-drained"}, prefix


def _tool_server() -> ToolServer:
    runtime = Mock()
    runtime._tools = {}
    runtime._tool_definitions = {}
    runtime.agent_config.mcp.servers = []
    return ToolServer(runtime)


def _tool_app(server: ToolServer) -> FastAPI:
    app = FastAPI()
    server.register_routes(app)
    return app


def _llm_payload(execution_id: str) -> dict:
    return {
        "execution_id": execution_id,
        "arguments": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        },
    }


def _stub_llm(server: ToolServer) -> None:
    async def chunks(_request: Any):
        yield LLMStreamChunk(content="hello")

    server._ensure_llm_registry_loaded = AsyncMock()  # type: ignore[method-assign]
    server._stream_llm_chunks = chunks  # type: ignore[method-assign]


def test_tool_execute_tags_setup_resolution_failure_and_metric_logs(json_logs, monkeypatch) -> None:
    def fail_resolution(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("bad mcp url")

    monkeypatch.setattr(ToolServer, "_configured_mcp_tool", fail_resolution)
    response = TestClient(_tool_app(_tool_server())).post(
        "/execute",
        json={
            "execution_id": "exec-tool-1",
            "tool_name": "github__search_issues",
            "arguments": {},
            "session_id": "session-1",
            "metadata": {"mcp_server": "github", "mcp_tool": "search_issues"},
        },
    )

    assert response.json()["status"] == "error"
    records = json_logs()
    for prefix in (
        "TOOL_POD: Step 0 - github__search_issues",
        "Unexpected error resolving MCP tool",
        "METRIC latency",
    ):
        assert _ids(records, prefix) == {"exec-tool-1"}, prefix


def test_tool_invoke_llm_metric_after_the_handler_carries_the_execution_id(json_logs) -> None:
    server = _tool_server()
    _stub_llm(server)

    response = TestClient(_tool_app(server)).post("/invoke_llm", json=_llm_payload("exec-llm-1"))

    assert response.json()["status"] == "success"
    assert _ids(json_logs(), "METRIC latency") == {"exec-llm-1"}


def _guardrail_check(execution_id: str) -> dict:
    return {
        "execution_id": execution_id,
        "stage": "llm_output",
        "input": {"text": "hi", "metadata": {}},
        "context": {
            "org_id": "000000000000000000000001",
            "project_id": "000000000000000000000002",
            "workspace_id": "workspace-1",
            "session_id": "session-1",
            "user_id": "user-1",
        },
        # An uncompilable pattern makes the evaluator warn and fail closed.
        "policies": [
            {
                "id": "000000000000000000000003",
                "type": "output_validation",
                "status": "active",
                "action": "block",
                "stage_filter": ["llm_output"],
                "config": {"regex_patterns": ["("]},
            }
        ],
    }


async def test_guardrail_warnings_carry_only_their_own_check_execution_id(json_logs) -> None:
    transport = httpx.ASGITransport(app=_tool_app(_tool_server()))
    async with httpx.AsyncClient(transport=transport, base_url="http://tool") as client:
        checks = await asyncio.gather(
            client.post("/guardrails/check", json=_guardrail_check("exec-guard-a")),
            client.post("/guardrails/check", json=_guardrail_check("exec-guard-b")),
        )
        # A check with no execution must not inherit the caller's binding.
        with bind_log_execution_id("exec-stale"):
            unbound = await client.post("/guardrails/check", json=_guardrail_check(""))

    assert [c.json()["decision"] for c in (*checks, unbound)] == ["block"] * 3
    warnings = [
        r["executionId"]
        for r in json_logs()
        if r["message"].startswith("Invalid guardrail regex pattern")
    ]
    assert sorted(warnings, key=str) == [None, "exec-guard-a", "exec-guard-b"]


def _stream_endpoint(app: FastAPI) -> Callable[..., Any]:
    return next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/invoke_llm/stream")


async def test_unattached_stream_release_warning_carries_only_its_execution_id(
    json_logs,
) -> None:
    server = _tool_server()
    server._drain_registry = DrainRegistry(attach_grace_s=0.02)
    endpoint = _stream_endpoint(_tool_app(server))
    http_request = Mock(headers={})

    # The consumer never iterates either response, so each registration is
    # released by its attach grace.
    await endpoint(LLMPodInvokeRequest(**_llm_payload("exec-unattached-a")), http_request)
    await endpoint(LLMPodInvokeRequest(**_llm_payload("exec-unattached-b")), http_request)
    await json_logs.wait_for("Releasing stream registration for execution exec-unattached-a")
    await json_logs.wait_for("Releasing stream registration for execution exec-unattached-b")

    records = json_logs()
    for execution_id in ("exec-unattached-a", "exec-unattached-b"):
        prefix = f"Releasing stream registration for execution {execution_id}"
        assert _ids(records, prefix) == {execution_id}


@pytest.mark.parametrize("cancel_mid_stream", [False, True])
async def test_stream_close_logs_carry_the_execution_id(json_logs, cancel_mid_stream) -> None:
    server = _tool_server()
    first_chunk_sent = asyncio.Event()

    async def chunks(_request: Any):
        yield LLMStreamChunk(content="hello")
        first_chunk_sent.set()
        if cancel_mid_stream:
            # A client disconnect cancels the response task while the
            # provider is still producing.
            await asyncio.Event().wait()

    server._ensure_llm_registry_loaded = AsyncMock()  # type: ignore[method-assign]
    server._stream_llm_chunks = chunks  # type: ignore[method-assign]
    registry = server.drain_registry
    end_work = registry.end_work

    def end_work_probe(handle: Any) -> None:
        logging.getLogger("stream_probe").info(
            "stream registration ended (context=%s)", get_current_execution_id()
        )
        end_work(handle)

    registry.end_work = end_work_probe  # type: ignore[method-assign]
    endpoint = _stream_endpoint(_tool_app(server))
    response = await endpoint(
        LLMPodInvokeRequest(**_llm_payload("exec-stream-1")), Mock(headers={})
    )

    async def consume() -> None:
        async for _chunk in response.body_iterator:
            pass

    consumer = asyncio.create_task(consume())
    if cancel_mid_stream:
        await first_chunk_sent.wait()
        consumer.cancel()
    await asyncio.gather(consumer, return_exceptions=True)

    assert _ids(json_logs(), "stream registration ended (context=None)") == {"exec-stream-1"}


async def test_client_disconnect_mid_stream_closes_the_stream_under_its_execution_id(
    json_logs,
) -> None:
    server = _tool_server()

    async def chunks(_request: Any):
        yield LLMStreamChunk(content="hello")
        yield LLMStreamChunk(content="never sent")

    server._ensure_llm_registry_loaded = AsyncMock()  # type: ignore[method-assign]
    server._stream_llm_chunks = chunks  # type: ignore[method-assign]
    registry = server.drain_registry
    end_work = registry.end_work

    def end_work_probe(handle: Any) -> None:
        logging.getLogger("stream_probe").info(
            "stream registration ended (context=%s)", get_current_execution_id()
        )
        end_work(handle)

    registry.end_work = end_work_probe  # type: ignore[method-assign]
    endpoint = _stream_endpoint(_tool_app(server))
    response = await endpoint(
        LLMPodInvokeRequest(**_llm_payload("exec-disconnect-1")), Mock(headers={})
    )
    disconnected = asyncio.Event()

    async def receive() -> dict:
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        if message["type"] == "http.response.body" and message.get("more_body"):
            # The client is gone, so this send never completes.
            disconnected.set()
            await asyncio.Event().wait()

    await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)

    assert _ids(json_logs(), "stream registration ended (context=None)") == {"exec-disconnect-1"}
