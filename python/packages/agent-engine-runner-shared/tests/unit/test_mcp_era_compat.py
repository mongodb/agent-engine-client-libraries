"""Dual-era compatibility tests for the runner's MCP client.

These start real loopback MCP servers instead of patching the SDK:

* a ``2026-07-28`` server built with ``mcp.server.mcpserver.MCPServer``;
* a legacy-only server that rejects ``server/discover`` with ``-32601`` and
  then answers the ``initialize`` handshake, which is how a pre-2026 server
  behaves.

The same runner code (``discover_mcp_tools_async`` and ``_call_mcp_tool``)
must list and call a tool against both. That is the backward-compatibility
gate for the SDK upgrade: if a future release stops falling back, these tests
fail.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import uvicorn
from mcp import MCPError
from mcp.server.mcpserver import MCPServer

from agent_engine_runner_shared.agent_config import RuntimeMCPConfig, RuntimeMCPServerConfig
from agent_engine_runner_shared.mcp_tools import (
    MCPConfigError,
    MCPToolBinding,
    MCPToolError,
    _call_mcp_tool,
    discover_mcp_tools_async,
)

_RECORDED_METHODS = {
    "server/discover",
    "initialize",
    "notifications/initialized",
    "tools/list",
    "tools/call",
}


class _RecordingMCP:
    """Record JSON-RPC methods and headers, and optionally reject discovery.

    ``reject_discover`` reproduces a legacy-only server: the modern probe gets
    a Method-not-found error, so a 2.x client must fall back to ``initialize``.
    """

    def __init__(
        self,
        app: Any,
        *,
        reject_discover: bool,
        fail_methods: dict[str, int] | None = None,
        error_methods: dict[str, str] | None = None,
        fail_delete: bool = False,
    ) -> None:
        self._app = app
        self._reject_discover = reject_discover
        self._fail_methods = fail_methods or {}
        self._error_methods = error_methods or {}
        self._fail_delete = fail_delete
        self.methods: list[str] = []
        self.http_methods: list[str] = []
        self.headers: list[tuple[str, str]] = []

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        self.http_methods.append(scope["method"])

        chunks: list[bytes] = []
        more = True
        while more:
            message = await receive()
            chunks.append(message.get("body", b""))
            more = message.get("more_body", False)
        body = b"".join(chunks)

        method: str | None = None
        request_id: Any = None
        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                method = payload.get("method")
                request_id = payload.get("id")
        except (TypeError, ValueError):
            pass

        if method is not None:
            if method in _RECORDED_METHODS:
                self.methods.append(method)
            for name, value in scope.get("headers", []):
                self.headers.append((name.decode("latin-1").lower(), value.decode("latin-1")))

        if scope["method"] == "DELETE" and self._fail_delete:
            await self._plain_response(send, 405)
            return

        if method is not None and method in self._fail_methods:
            await self._plain_response(send, self._fail_methods[method])
            return

        if method is not None and method in self._error_methods:
            response = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32602, "message": self._error_methods[method]},
                }
            ).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": response})
            return

        if method == "server/discover" and self._reject_discover:
            response = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "Method not found"},
                }
            ).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": response})
            return

        replayed = False

        async def replay() -> dict[str, Any]:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            # The Starlette response may keep watching for a disconnect; hand
            # the real transport's next message through instead of inventing one.
            return await receive()

        await self._app(scope, replay, send)

    def header_values(self, name: str) -> list[str]:
        """Return the values sent for one header name."""

        return [value for header, value in self.headers if header == name]

    @staticmethod
    async def _plain_response(send: Any, status: int) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"failure"})


def _add(a: int, b: int) -> dict[str, int]:
    """Add two integers."""
    return {"sum": a + b}


@contextmanager
def _serve(
    *,
    reject_discover: bool,
    fail_methods: dict[str, int] | None = None,
    error_methods: dict[str, str] | None = None,
    fail_delete: bool = False,
) -> Iterator[tuple[str, _RecordingMCP]]:
    server = MCPServer("era-fixture")
    server.add_tool(_add, name="add")
    recorder = _RecordingMCP(
        server.streamable_http_app(),
        reject_discover=reject_discover,
        fail_methods=fail_methods,
        error_methods=error_methods,
        fail_delete=fail_delete,
    )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]

    uvicorn_server = uvicorn.Server(uvicorn.Config(recorder, log_level="warning"))
    thread = threading.Thread(
        target=uvicorn_server.run,
        kwargs={"sockets": [listener]},
        daemon=True,
    )
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not uvicorn_server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("MCP fixture server did not start")
            time.sleep(0.01)
        yield f"http://127.0.0.1:{port}/mcp", recorder
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=5)


def _server_config(url: str, *, headers: dict[str, str] | None = None) -> RuntimeMCPServerConfig:
    payload: dict[str, Any] = {"url": url, "timeout_seconds": 10}
    if headers is not None:
        payload["headers"] = headers
    return RuntimeMCPServerConfig.model_validate(payload)


def _binding(config: RuntimeMCPServerConfig) -> MCPToolBinding:
    return MCPToolBinding(
        server_name="era",
        tool_name="add",
        sdk_tool_name="era__add",
        description="Add two integers",
        input_schema={"type": "object", "properties": {}},
        server_config=config,
    )


def test_modern_server_lists_and_calls_through_the_runner() -> None:
    with _serve(reject_discover=False) as (url, recorder):
        config = _server_config(url)

        bindings = asyncio.run(discover_mcp_tools_async(_mcp_config(url)))
        result = asyncio.run(_call_mcp_tool(_binding(config), {"a": 2, "b": 3}))

    assert [binding.tool_name for binding in bindings] == ["add"]
    assert bindings[0].sdk_tool_name == "era__add"
    assert bindings[0].input_schema["type"] == "object"
    assert '"sum": 5' in result.content[0]["text"]
    # The probe proves the client negotiated the modern era rather than
    # silently using the legacy handshake.
    assert "server/discover" in recorder.methods
    assert "initialize" not in recorder.methods


def test_legacy_only_server_falls_back_and_still_works() -> None:
    with _serve(reject_discover=True) as (url, recorder):
        config = _server_config(url)

        bindings = asyncio.run(discover_mcp_tools_async(_mcp_config(url)))
        result = asyncio.run(_call_mcp_tool(_binding(config), {"a": 4, "b": 6}))

    assert [binding.tool_name for binding in bindings] == ["add"]
    assert '"sum": 10' in result.content[0]["text"]
    # Evidence of the exact fallback path: probe, handshake, then work.
    assert recorder.methods[:2] == ["server/discover", "initialize"]
    assert recorder.methods.index("tools/list") > recorder.methods.index("initialize")
    assert recorder.methods.index("tools/call") > recorder.methods.index("tools/list")


def test_configured_headers_reach_the_server() -> None:
    with _serve(reject_discover=False) as (url, recorder):
        config = _server_config(url, headers={"X-MCP-Readonly": "true"})
        asyncio.run(_call_mcp_tool(_binding(config), {"a": 1, "b": 1}))

    assert "true" in recorder.header_values("x-mcp-readonly")


def _mcp_config(url: str) -> RuntimeMCPConfig:
    return RuntimeMCPConfig.model_validate(
        {"servers": {"era": {"url": url, "timeout_seconds": 10}}}
    )


def test_failed_call_reports_its_status_not_the_cleanup_status() -> None:
    """A tolerated session DELETE must not replace the failed call's status."""

    with _serve(
        reject_discover=True,
        fail_methods={"tools/call": 503},
        fail_delete=True,
    ) as (url, recorder):
        with pytest.raises(MCPToolError, match="HTTP 503 Service Unavailable"):
            asyncio.run(_call_mcp_tool(_binding(_server_config(url)), {"a": 1, "b": 1}))

    # The legacy handshake establishes a session, so teardown actually sends
    # the DELETE that returns 405.
    assert "DELETE" in recorder.http_methods


def test_recovered_probe_status_does_not_replace_the_operation_error() -> None:
    """A recovered discovery failure must not label a later tools/list error."""

    with _serve(
        reject_discover=False,
        fail_methods={"server/discover": 400},
        error_methods={"tools/list": "Invalid params"},
    ) as (url, _recorder):
        with pytest.raises(MCPConfigError, match="Invalid params") as exc_info:
            asyncio.run(discover_mcp_tools_async(_mcp_config(url)))

    assert "HTTP 400" not in str(exc_info.value)


def test_status_is_captured_before_context_teardown() -> None:
    """A cleanup request after the failure cannot replace the status."""

    from contextlib import asynccontextmanager

    from agent_engine_runner_shared import mcp_tools

    class FailingClient:
        async def call_tool(self, _name, _arguments):
            raise MCPError(-32603, "Server returned an error response")

    @asynccontextmanager
    async def fake_open(config, *, headers, auth, responses):
        _ = (config, headers, auth)
        responses.append(mcp_tools._MCPHTTPStatus(503, "Service Unavailable"))
        try:
            yield FailingClient()
        finally:
            # Simulate a cleanup token refresh succeeding after the failure.
            responses.clear()
            responses.append(mcp_tools._MCPHTTPStatus(200, "OK"))

    original = mcp_tools._open_mcp_client
    mcp_tools._open_mcp_client = fake_open
    try:
        with pytest.raises(MCPToolError, match="HTTP 503 Service Unavailable"):
            asyncio.run(
                _call_mcp_tool(_binding(_server_config("http://127.0.0.1:1/mcp")), {"a": 1, "b": 1})
            )
    finally:
        mcp_tools._open_mcp_client = original
