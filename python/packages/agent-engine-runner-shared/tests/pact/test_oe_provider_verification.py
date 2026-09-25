"""Real provider verification of the committed OE→AER and OE→Tool Pod pacts.

The Orchestration Engine is the consumer; the AER and Tool Pod (this package's
FastAPI servers) are the providers. These tests stand up the *real* AERServer /
ToolServer FastAPI apps and replay every interaction in
``pacts/orchestration-engine-{aer,tool-pod}.json`` against them via pact-python's
verifier — the "real python service" verification the Go-side stub cannot do.

External I/O that is not part of the contract is mocked at the narrowest seam:
the agent/LLM execution and the OE-ward callbacks. Everything on the
request→response boundary (routing, request-model binding, response-model
serialization, status codes) is the production code path.

Provider states put the app into the state each interaction's ``given`` declares
(happy data, or an injected error), so error responses are produced by the state
handler rather than by fabricating malformed requests.

pact-python is not a committed dependency (its FFI wheel would churn the shared
workspace lock), so run these with an ephemeral install:

    uv run --with pact-python python -m pytest tests/pact/

(from client-libraries/packages/python/packages/agent-engine-runner-shared). A plain ``uv run pytest`` skips
this suite — see conftest.py.
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from agent_engine_sdk import StreamEvent  # type: ignore[import-not-found]
from agent_engine_sdk.models import (  # type: ignore[import-not-found]
    SessionMessage,
    SessionMessagesResponse,
)
from fastapi import FastAPI, HTTPException
from pact import Verifier  # type: ignore[import-not-found]  # ephemeral dep; see conftest

import agent_engine_runner_shared.server.tool as tool_mod
from agent_engine_runner_shared.server.aer import AERServer
from agent_engine_runner_shared.server.tool import ToolServer

# tests/pact/<file> -> repo root: pact,tests,agent-engine-runner-shared,packages,python,packages,client-libraries,<root>
REPO_ROOT = Path(__file__).resolve().parents[7]
# OE_PACTS_DIR points verification at archived pact snapshots (the AP-4639
# cross-version matrix extracts pacts from older release tags); unset, the
# committed current pacts are verified, unchanged. Mirrors the Go stub's
# OE_PACT_FILE override.
PACTS_DIR = (
    Path(os.environ.get("OE_PACTS_DIR", ""))
    if os.environ.get("OE_PACTS_DIR")
    else REPO_ROOT / "pacts"
)


# --------------------------------------------------------------------------- #
# Shared, mutable provider state. State handlers flip ``mode`` before each
# interaction is replayed; the fakes and the injected seams read it.
# --------------------------------------------------------------------------- #
_aer_state = {"mode": "happy"}
_tool_state = {"mode": "happy"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _running_app(app: FastAPI):
    """Serve ``app`` on a background uvicorn thread; yield its base URL."""
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        assert server.started, "provider app failed to start"
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


# --------------------------------------------------------------------------- #
# AER provider
# --------------------------------------------------------------------------- #
class _FakeAgent:
    """Yields a single terminal ``result`` StreamEvent (a clean completion), or
    raises when the state asks for an internal failure."""

    def __init__(self, mode: str):
        self._mode = mode

    async def execute(self, ctx: Any, agent_input: Any):
        if self._mode == "execute_fails":
            raise RuntimeError("agent blew up")
        yield StreamEvent(
            event="result",
            data={"response": "done", "messages": [], "resumed": False},
        )


class _FakeQueryPlugin:
    async def get_summaries_for_sessions(self, session_ids: list[str]):
        from agent_engine_sdk.models import SessionsSummaryResponse  # type: ignore

        return SessionsSummaryResponse(sessions=[])

    async def get_messages_for_session(self, session_id: str) -> SessionMessagesResponse:
        if _aer_state["mode"] == "query_fails":
            raise RuntimeError("session query blew up")
        from agent_engine_sdk.models import LLMToolCall  # type: ignore

        # Fully populated (including the optional tool_calls / tool_call_id /
        # additional_kwargs) so the contract pins every serialized field.
        return SessionMessagesResponse(
            messages=[
                SessionMessage(
                    id="msg-1",
                    role="user",
                    content="hello",
                    timestamp="2024-01-01T00:00:00Z",
                    session_id="sess-123",
                    name="user",
                    tool_calls=[
                        LLMToolCall(
                            id="call-1", name="lookup", args={"q": "x"}, type="function", index=0
                        )
                    ],
                    tool_call_id="call-1",
                    additional_kwargs={"finish_reason": "stop"},
                )
            ]
        )


class _FakeAERRuntime:
    app_name = "pact-test-agent"
    app_version = "0.0.0"
    memory_enabled = False
    _tool_definitions = {"search": {"is_local": True}}
    _tools = {"search": lambda **kw: None}
    _memory_writer = None
    _graph_builder = None

    def note_checkpoint_wire_workspace_id(self, workspace_id: Any) -> None:
        return None

    def get_query_plugin(self):
        if _aer_state["mode"] == "query_unsupported":
            return None
        return _FakeQueryPlugin()

    def get_agent(self, callbacks: Any = None):
        return _FakeAgent(_aer_state["mode"])

    def warm_up_agent(self) -> bool:
        return True


def _build_aer_app() -> FastAPI:
    # Real __init__ (attribute-only, no side effects) so newly added server
    # attributes are always present; then disable the outbound one-shot side
    # effects (A2A registration, capabilities advertisement).
    server = AERServer(_FakeAERRuntime())  # type: ignore[arg-type]
    server._a2a_registered = True
    server._capabilities_advertised = True
    app = FastAPI()
    server._register_common_routes(app)  # /health, /, /metrics (matches create_app)
    server.register_routes(app)
    return app


# Inject the request-shape errors (400 ambiguous, 422 validation) at the route
# seam: the pact sends OE's real (valid) request, so these can't arise naturally
# and the provider *state* is responsible for producing them.
_orig_aer_exec = AERServer._execute_with_inbound_trace_context


async def _patched_aer_exec(self, request, headers):  # type: ignore[no-untyped-def]
    if _aer_state["mode"] == "reject_invalid":
        raise HTTPException(status_code=400, detail="message was ambiguous")
    if _aer_state["mode"] == "cannot_process":
        raise HTTPException(
            status_code=422,
            detail=[
                {"loc": ["body", "message"], "msg": "field required", "type": "value_error.missing"}
            ],
        )
    return await _orig_aer_exec(self, request, headers)


# Silence OE-ward callbacks/streaming (not part of the contract, and the pact's
# platform_api_url is unreachable).
async def _noop(*args: Any, **kwargs: Any) -> None:
    return None


AER_STATES = {
    "the AER does not expose query endpoints": lambda: _aer_state.update(mode="query_unsupported"),
    "the AER session query fails": lambda: _aer_state.update(mode="query_fails"),
    "the AER fails internally": lambda: _aer_state.update(mode="execute_fails"),
    "the AER rejects the request as invalid": lambda: _aer_state.update(mode="reject_invalid"),
    "the AER cannot process the request": lambda: _aer_state.update(mode="cannot_process"),
}


def _aer_state_handler(state: str = "", **_: Any) -> None:
    _aer_state["mode"] = "happy"
    setter = AER_STATES.get(state)
    if setter:
        setter()


def test_aer_provider_contract(monkeypatch: pytest.MonkeyPatch):
    pact = PACTS_DIR / "orchestration-engine-aer.json"
    assert pact.exists(), f"missing {pact} — regenerate the OE consumer pacts first"

    monkeypatch.setattr(AERServer, "_execute_with_inbound_trace_context", _patched_aer_exec)
    monkeypatch.setattr(AERServer, "_send_callback_body", _noop, raising=False)
    monkeypatch.setattr(AERServer, "_send_stream_chunk", _noop, raising=False)

    _aer_state["mode"] = "happy"
    app = _build_aer_app()
    with _running_app(app) as base_url:
        (
            Verifier("aer", "127.0.0.1")
            .add_transport(url=base_url)
            .add_source(str(pact))
            .state_handler(_aer_state_handler, teardown=False)
            .verify()
        )


# --------------------------------------------------------------------------- #
# Tool Pod provider
# --------------------------------------------------------------------------- #
class _RaisingDefs:
    """Stands in for runtime._tool_definitions when the list must fail (500)."""

    def items(self):
        raise RuntimeError("tool registry unavailable")


def _search_tool(**kwargs: Any) -> dict[str, Any]:
    return {"output": "sunny"}


def _build_tool_app() -> tuple[FastAPI, ToolServer]:
    from unittest.mock import MagicMock

    server = ToolServer(MagicMock())
    server.runtime.app_name = "pact-test-agent"  # str for HealthResponse.component
    server._tool_definitions = {"search": {"is_local": True}}  # type: ignore[attr-defined]
    server._tools = {"search": _search_tool}  # type: ignore[attr-defined]
    app = FastAPI()
    server._register_common_routes(app)  # /health, /, /metrics (matches create_app)
    server.register_routes(app)
    return app, server


def _apply_tool_state(server: ToolServer) -> None:
    """Reconfigure the running Tool Pod server for the active state."""
    mode = _tool_state["mode"]
    if mode == "list_unavailable":
        server.runtime._tool_definitions = _RaisingDefs()  # type: ignore[attr-defined]
    else:
        server.runtime._tool_definitions = {"search": {"is_local": True}}  # type: ignore[attr-defined]
    server._tools = {"search": _search_tool}  # type: ignore[attr-defined]


TOOL_STATES = {
    "the tool pod list is unavailable": "list_unavailable",
    "the tool pod tool execution fails": "tool_fails",
    "the tool pod external API call is rate limited": "tool_fails_classified",
    "the tool pod llm invocation fails": "llm_fails",
    "the tool pod guardrail check is unavailable": "guardrail_fails",
}


def test_tool_pod_provider_contract(monkeypatch: pytest.MonkeyPatch):
    pact = PACTS_DIR / "orchestration-engine-tool-pod.json"
    assert pact.exists(), f"missing {pact} — regenerate the OE consumer pacts first"

    app, server = _build_tool_app()

    # invoke_llm: stub only the LLM I/O so the real _handle_invoke_llm assembles
    # the response envelope (the contracted part). A raising stub drives the
    # 200 status="error" failure path.
    async def _noop_registry(self):  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(ToolServer, "_ensure_llm_registry_loaded", _noop_registry, raising=False)

    async def _fake_stream_chunks(self, request):  # type: ignore[no-untyped-def]
        if _tool_state["mode"] == "llm_fails":
            raise RuntimeError("llm blew up")
        from agent_engine_sdk.models import LLMStreamChunk  # type: ignore

        yield LLMStreamChunk(content="hi")

    monkeypatch.setattr(ToolServer, "_stream_llm_chunks", _fake_stream_chunks, raising=False)

    # /invoke_llm/stream handshake: the real route returns a StreamingResponse
    # (200 + text/event-stream); stub the body generator so the handshake is
    # produced without a live LLM.
    async def _fake_stream_response(self, request):  # type: ignore[no-untyped-def]
        yield "data: {}\n\n"

    monkeypatch.setattr(
        ToolServer, "_handle_invoke_llm_stream", _fake_stream_response, raising=False
    )

    # A registered tool does not reliably produce a status="error" run (the tool
    # protocol is opaque), so the tool-failure state injects the contracted error
    # response at the handler seam; real routing + request-model validation +
    # response-model serialization still run.
    from agent_engine_runner_shared.models import ToolPodExecuteResponse

    _orig_tool_execute = ToolServer._handle_execute

    async def _patched_tool_execute(self, request):  # type: ignore[no-untyped-def]
        if _tool_state["mode"] == "tool_fails":
            return ToolPodExecuteResponse(status="error", error="tool execution failed")
        if _tool_state["mode"] == "tool_fails_classified":
            from agent_engine_runner_shared.models import ToolAPIError

            return ToolPodExecuteResponse(
                status="error",
                error='External API call to "github" was rate limited (HTTP 429).',
                tool_api_error=ToolAPIError(
                    provider_type="github",
                    classification="RATE_LIMITED",
                    http_status=429,
                    retryable=True,
                ),
            )
        return await _orig_tool_execute(self, request)

    monkeypatch.setattr(ToolServer, "_handle_execute", _patched_tool_execute, raising=False)

    _orig_guardrail = tool_mod.evaluate_guardrail_check

    def _patched_guardrail(request):  # type: ignore[no-untyped-def]
        if _tool_state["mode"] == "guardrail_fails":
            raise RuntimeError("guardrail evaluator unavailable")
        return _orig_guardrail(request)

    monkeypatch.setattr(tool_mod, "evaluate_guardrail_check", _patched_guardrail)

    def _tool_state_handler(state: str = "", **_: Any) -> None:
        _tool_state["mode"] = TOOL_STATES.get(state, "happy")
        _apply_tool_state(server)

    _tool_state["mode"] = "happy"
    _apply_tool_state(server)
    with _running_app(app) as base_url:
        (
            Verifier("tool-pod", "127.0.0.1")
            .add_transport(url=base_url)
            .add_source(str(pact))
            .state_handler(_tool_state_handler, teardown=False)
            .verify()
        )
