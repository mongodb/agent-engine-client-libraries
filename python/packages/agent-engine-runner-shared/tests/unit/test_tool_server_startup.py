"""Unit tests for ``ToolServer.on_startup``.

Guards against the regression where a missing built-in handler made
every tool call fail with ``Unknown tool: filesystem_*``.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import httpx
import pytest
from pydantic import ValidationError

import agent_engine_runner_shared.hooks as hooks
from agent_engine_runner_shared.agent_config import RuntimeMCPConfig
from agent_engine_runner_shared.connectors import CONNECTOR_BUNDLE_PATH_ENV, ConnectorBundleError
from agent_engine_runner_shared.context import (
    get_current_authorization,
    get_current_session_id,
    record_current_memory_metadata,
)
from agent_engine_runner_shared.mcp_tools import MCPToolResult
from agent_engine_runner_shared.models import ToolPodExecuteRequest
from agent_engine_runner_shared.server.tool import ToolServer
from agent_engine_runner_shared.toolpod_handlers import BUILTIN_TOOL_NAMES

EXPECTED_TOOLS = set(BUILTIN_TOOL_NAMES)

FIXTURE = Path(__file__).parent / "connectors" / "fixtures" / "jira.tool_defs.yaml"

TOOL_YAML_TEXT = """\
name: jira
tool_defs: jira.tool_defs.yaml
source:
  base_url: https://jira.example/rest
expose:
  allow_all: true
auth:
  type: bearer
  env: JIRA_TOKEN
"""


def _write_connector_bundle(tmp_path):
    definitions_path = tmp_path / "jiradc.tool_defs.yaml"
    definitions_path.write_text(
        """source:
  name: jiradc
  base_url: https://jira.example/rest
auth:
  type: bearer
  env: JIRA_TOKEN
tools:
  - name: jiradc_getComments
    description: Returns comments for an issue.
    method: GET
    path: /api/2/issue/{issueIdOrKey}/comment
    params:
      path: [issueIdOrKey]
    inputSchema:
      type: object
      properties:
        issueIdOrKey: {type: string}
      required: [issueIdOrKey]
  - name: jiradc_addComment
    description: Adds a comment to an issue.
    method: POST
    path: /api/2/issue/{issueIdOrKey}/comment
    params:
      path: [issueIdOrKey]
      body: [body]
    inputSchema:
      type: object
      properties:
        issueIdOrKey: {type: string}
        body: {type: string}
      required: [issueIdOrKey, body]
""",
        encoding="utf-8",
    )
    index_path = tmp_path / "bundle.yaml"
    index_path.write_text(
        "version: 1\nconnectors:\n  - name: jiradc\n    tool_defs: jiradc.tool_defs.yaml\n",
        encoding="utf-8",
    )
    return index_path


def _make_runtime(
    deep_agent_enabled: bool = True,
    mcp_config: RuntimeMCPConfig | None = None,
) -> SimpleNamespace:
    agent_config = SimpleNamespace(
        feature_enabled=lambda name, default=False: (
            deep_agent_enabled if name == "deep_agent" else default
        ),
        mcp=mcp_config or RuntimeMCPConfig(),
    )
    return SimpleNamespace(
        _tools={},
        _tool_definitions={},
        _graph_builder=None,
        agent_config=agent_config,
    )


@pytest.fixture(autouse=True)
def _isolate_llm_factory():
    hooks.reset_hooks()
    yield
    hooks.reset_hooks()


class TestToolServerStartup:
    """The contract: after ``on_startup``, all 8 built-in handlers are registered."""

    @pytest.mark.asyncio
    async def test_registers_all_eight_builtin_handlers(self):
        runtime = _make_runtime()
        server = ToolServer(runtime)

        await server.on_startup()

        assert set(runtime._tools.keys()) == EXPECTED_TOOLS

    @pytest.mark.asyncio
    async def test_each_handler_is_callable(self):
        """Registered handlers must be callables, not placeholder metadata."""
        runtime = _make_runtime()
        server = ToolServer(runtime)

        await server.on_startup()

        for name, handler in runtime._tools.items():
            assert callable(handler), f"{name} is not callable: {handler!r}"

    @pytest.mark.asyncio
    async def test_registers_tool_definitions_with_names(self):
        """``_tool_definitions`` must mirror ``_tools`` so metadata lookups work."""
        runtime = _make_runtime()
        server = ToolServer(runtime)

        await server.on_startup()

        assert set(runtime._tool_definitions.keys()) == EXPECTED_TOOLS
        for name, defn in runtime._tool_definitions.items():
            assert isinstance(defn, dict)
            assert defn.get("name") == name

    @pytest.mark.asyncio
    async def test_registers_and_executes_bundled_connectors(self, monkeypatch, tmp_path):
        monkeypatch.setenv(CONNECTOR_BUNDLE_PATH_ENV, str(_write_connector_bundle(tmp_path)))
        runtime = _make_runtime(deep_agent_enabled=False)
        calls = []
        closed = []

        class FakeExecutor:
            def __init__(self, definitions):
                assert definitions.source.name == "jiradc"

            def execute(self, name, arguments, *, env):
                calls.append((name, arguments, env.get("JIRA_TOKEN")))
                return "connector result"

            def close(self):
                closed.append(True)

        monkeypatch.setattr(
            "agent_engine_runner_shared.connectors.runtime.ConnectorExecutor", FakeExecutor
        )
        server = ToolServer(runtime)

        await server.on_startup()
        response = await server._handle_execute(
            ToolPodExecuteRequest(
                execution_id="exec-1",
                tool_name="jiradc_getComments",
                arguments={"issueIdOrKey": "DEMO-1"},
                session_id="session-1",
            )
        )
        await server.on_shutdown()

        assert set(runtime._tools) == {"jiradc_getComments", "jiradc_addComment"}
        assert runtime._tool_definitions["jiradc_getComments"] == {
            "name": "jiradc_getComments",
            "description": "Returns comments for an issue.",
            "is_local": False,
            "provider_type": None,
            "scopes": [],
            "network": ["jira.example"],
            "timeout_seconds": 30,
            "redact_fields": [],
            "connector": "jiradc",
        }
        assert response.status == "success"
        assert response.result == "connector result"
        assert calls == [("jiradc_getComments", {"issueIdOrKey": "DEMO-1"}, None)]
        assert closed == [True]

    @pytest.mark.asyncio
    async def test_connector_collision_does_not_partially_register_bundle(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv(CONNECTOR_BUNDLE_PATH_ENV, str(_write_connector_bundle(tmp_path)))
        runtime = _make_runtime(deep_agent_enabled=False)
        runtime._tools["jiradc_getComments"] = lambda: "existing"
        runtime._tool_definitions["jiradc_getComments"] = {"name": "jiradc_getComments"}
        server = ToolServer(runtime)

        with pytest.raises(ConnectorBundleError, match="jiradc_getComments"):
            await server.on_startup()

        assert set(runtime._tools) == {"jiradc_getComments"}

    @pytest.mark.asyncio
    async def test_materializes_connectors_from_agent_yaml_traversal(self, monkeypatch, tmp_path):
        monkeypatch.delenv(CONNECTOR_BUNDLE_PATH_ENV, raising=False)
        connectors = tmp_path / "connectors" / "jira"
        connectors.mkdir(parents=True)
        (connectors / "jira.tool_defs.yaml").write_text(
            FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (connectors / "tool.yaml").write_text(TOOL_YAML_TEXT, encoding="utf-8")
        agent_config = SimpleNamespace(
            feature_enabled=lambda name, default=False: default,
            mcp=RuntimeMCPConfig(),
            path=tmp_path / "agent.yaml",
            connectors=[SimpleNamespace(source="connectors/jira/tool.yaml")],
        )
        runtime = SimpleNamespace(
            _tools={},
            _tool_definitions={},
            _graph_builder=None,
            agent_config=agent_config,
        )
        server = ToolServer(runtime)

        await server.on_startup()
        await server.close_connector_runtime()

        assert set(runtime._tools) == {"jiradc_getComments", "jiradc_addComment"}
        assert runtime._tool_definitions["jiradc_getComments"]["connector"] == "jira"

    @pytest.mark.asyncio
    async def test_pre_materialized_bundle_wins_over_agent_yaml_traversal(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv(CONNECTOR_BUNDLE_PATH_ENV, str(_write_connector_bundle(tmp_path)))
        runtime = _make_runtime(deep_agent_enabled=False)
        runtime.agent_config = SimpleNamespace(
            feature_enabled=lambda name, default=False: default,
            mcp=RuntimeMCPConfig(),
            path=tmp_path / "agent.yaml",
            connectors=[SimpleNamespace(source="connectors/jira/tool.yaml")],
        )
        server = ToolServer(runtime)

        await server.on_startup()
        await server.close_connector_runtime()

        assert runtime._tool_definitions["jiradc_getComments"]["connector"] == "jiradc"

    @pytest.mark.asyncio
    async def test_execute_returns_memory_metadata_without_reclassifying_tool(self):
        runtime = _make_runtime()

        def save_memory():
            record_current_memory_metadata(
                action="write",
                memory_type="semantic",
                content="Customer name: Jane Smith",
            )
            return "ok"

        runtime._tools["save_memory"] = save_memory

        response = await ToolServer(runtime)._handle_execute(
            ToolPodExecuteRequest(
                execution_id="exec-1",
                tool_name="save_memory",
                arguments={},
                session_id="session-1",
            )
        )

        assert response.status == "success"
        assert response.kind is None
        assert response.metadata["memory"] == {
            "action": "write",
            "type": "semantic",
            "content": "Customer name: Jane Smith",
        }

    @pytest.mark.asyncio
    async def test_execute_sets_session_context(self):
        runtime = _make_runtime()

        def current_session():
            return get_current_session_id()

        runtime._tools["current_session"] = current_session

        response = await ToolServer(runtime)._handle_execute(
            ToolPodExecuteRequest(
                execution_id="exec-1",
                tool_name="current_session",
                arguments={},
                session_id="session-42",
            )
        )

        assert response.status == "success"
        assert response.result == "session-42"

    @pytest.mark.asyncio
    async def test_custom_tool_error_redacts_request_credential(self, monkeypatch):
        monkeypatch.setenv("CUSTOM_API_KEY", "sk-live-123")
        runtime = _make_runtime()

        def call_provider():
            request = httpx.Request("GET", "https://api.example.com/resource")
            response = httpx.Response(
                401,
                json={"message": "Rejected credential sk-live-123"},
                request=request,
            )
            raise httpx.HTTPStatusError("HTTP 401", request=request, response=response)

        runtime._tools["call_provider"] = call_provider

        response = await ToolServer(runtime)._handle_execute(
            ToolPodExecuteRequest(
                execution_id="exec-1",
                tool_name="call_provider",
                arguments={},
                session_id="session-1",
            )
        )

        assert response.status == "error"
        assert "sk-live-123" not in (response.error or "")
        assert response.tool_api_error is not None
        reason = response.tool_api_error.reason or ""
        # Exact text can vary with ambient tenant env values; the credential
        # and its adjacent marker are what this regression protects.
        assert "sk-live-123" not in reason
        assert "<redacted>" in reason

    @pytest.mark.asyncio
    async def test_custom_tool_error_drops_url_shaped_echo(self):
        runtime = _make_runtime()

        def call_provider():
            request = httpx.Request("GET", "https://api.example.com/resource")
            response = httpx.Response(
                400,
                json={"message": "see https://internal.example/private rejected"},
                request=request,
            )
            raise httpx.HTTPStatusError("HTTP 400", request=request, response=response)

        runtime._tools["call_provider"] = call_provider

        response = await ToolServer(runtime)._handle_execute(
            ToolPodExecuteRequest(
                execution_id="exec-1",
                tool_name="call_provider",
                arguments={},
                session_id="session-1",
            )
        )

        assert response.status == "error"
        assert "internal.example" not in (response.error or "")
        assert response.tool_api_error is not None
        assert response.tool_api_error.reason is None

    @pytest.mark.asyncio
    async def test_custom_tool_error_redacts_delegated_authorization_token(self):
        runtime = _make_runtime()

        def call_provider():
            request = httpx.Request("GET", "https://api.example.com/resource")
            response = httpx.Response(
                401,
                json={"message": "Rejected delegated-abc123"},
                request=request,
            )
            raise httpx.HTTPStatusError("HTTP 401", request=request, response=response)

        runtime._tools["call_provider"] = call_provider

        response = await ToolServer(runtime)._handle_execute(
            ToolPodExecuteRequest(
                execution_id="exec-1",
                tool_name="call_provider",
                arguments={},
                session_id="session-1",
                authorization={
                    "token": "delegated-abc123",
                    "expires_at": 1_735_689_600,
                },
            )
        )

        assert response.status == "error"
        assert "delegated-abc123" not in (response.error or "")
        assert response.tool_api_error is not None
        assert "delegated-abc123" not in (response.tool_api_error.reason or "")

    @pytest.mark.asyncio
    async def test_execute_sets_authorization_context(self):
        runtime = _make_runtime()

        def current_authorization():
            auth = get_current_authorization()
            if auth is None:
                return None
            return {"token": auth.token, "expires_at": auth.expires_at}

        runtime._tools["current_authorization"] = current_authorization

        response = await ToolServer(runtime)._handle_execute(
            ToolPodExecuteRequest(
                execution_id="exec-1",
                tool_name="current_authorization",
                arguments={},
                session_id="session-42",
                authorization={"token": "broker-token", "expires_at": 1_735_689_600},
            )
        )

        assert response.status == "success"
        assert response.result == {"token": "broker-token", "expires_at": 1_735_689_600}

    @pytest.mark.asyncio
    async def test_concurrent_execute_on_same_server_can_overlap(self):
        """Concurrent /execute calls on one ToolServer may overlap."""
        import asyncio

        runtime = _make_runtime()
        in_flight = 0
        max_in_flight = 0
        started_first = asyncio.Event()
        started_second = asyncio.Event()
        release = asyncio.Event()

        async def slow_tool(**_: Any) -> Dict[str, int]:
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            if not started_first.is_set():
                started_first.set()
            else:
                started_second.set()
            await release.wait()
            in_flight -= 1
            return {"max_in_flight": max_in_flight}

        runtime._tools["slow_tool"] = slow_tool
        server = ToolServer(runtime)
        assert server._restriction_disabled is True

        first = asyncio.create_task(
            server._handle_execute(
                ToolPodExecuteRequest(
                    execution_id="exec-1",
                    tool_name="slow_tool",
                    arguments={},
                    session_id="session-1",
                )
            )
        )
        await started_first.wait()
        second = asyncio.create_task(
            server._handle_execute(
                ToolPodExecuteRequest(
                    execution_id="exec-2",
                    tool_name="slow_tool",
                    arguments={},
                    session_id="session-1",
                )
            )
        )
        await started_second.wait()
        assert max_in_flight == 2
        release.set()
        first_resp, second_resp = await asyncio.gather(first, second)
        assert first_resp.status == "success"
        assert second_resp.status == "success"

    @pytest.mark.asyncio
    async def test_concurrent_connector_calls_keep_request_local_secrets(
        self, monkeypatch, tmp_path
    ):
        """A worker must use the secret captured for its own request."""
        import asyncio
        import threading

        import agent_engine_runner_shared.server.tool as tool_server
        from agent_engine_runner_shared.connectors import ConnectorExecutor
        from agent_engine_runner_shared.utils import tenant_env_vars

        bundle_path = _write_connector_bundle(tmp_path)
        monkeypatch.setenv(CONNECTOR_BUNDLE_PATH_ENV, str(bundle_path))

        merged_secrets = iter(("token-a", "token-b"))

        def merge_secret() -> None:
            monkeypatch.setenv("JIRA_TOKEN", next(merged_secrets))

        monkeypatch.setattr(tool_server, "merge_metadata_env", merge_secret)

        first_started = threading.Event()
        release_first = threading.Event()
        observed: dict[str, str | None] = {}

        def execute(
            _executor,
            _tool_name: str,
            arguments: dict[str, Any],
            *,
            env: dict[str, str] | None = None,
        ) -> str:
            issue = arguments["issueIdOrKey"]
            if issue == "FIRST-1":
                first_started.set()
                assert release_first.wait(timeout=2)
            observed[issue] = (env if env is not None else tenant_env_vars()).get("JIRA_TOKEN")
            return issue

        monkeypatch.setattr(ConnectorExecutor, "execute", execute)

        runtime = _make_runtime(deep_agent_enabled=False)
        server = ToolServer(runtime)
        await server.on_startup()
        try:
            first = asyncio.create_task(
                server._handle_execute(
                    ToolPodExecuteRequest(
                        execution_id="exec-1",
                        tool_name="jiradc_getComments",
                        arguments={"issueIdOrKey": "FIRST-1"},
                        session_id="session-1",
                    )
                )
            )
            assert await asyncio.to_thread(first_started.wait, 2)

            second = await server._handle_execute(
                ToolPodExecuteRequest(
                    execution_id="exec-2",
                    tool_name="jiradc_getComments",
                    arguments={"issueIdOrKey": "SECOND-2"},
                    session_id="session-2",
                )
            )
            release_first.set()
            first_response = await first

            assert first_response.status == "success"
            assert second.status == "success"
            assert observed == {"FIRST-1": "token-a", "SECOND-2": "token-b"}
        finally:
            release_first.set()
            await server.on_shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_execute_ignores_restriction_env_false(self, monkeypatch):
        """AGENTIC_PLATFORM_SECRET_RESTRICTION_DISABLED=false cannot re-enable restriction. (open-source-refs:ignore — runtime env-var interface name)"""
        import asyncio

        monkeypatch.setenv("AGENTIC_PLATFORM_SECRET_RESTRICTION_DISABLED", "false")
        runtime = _make_runtime()
        in_flight = 0
        max_in_flight = 0
        started_first = asyncio.Event()
        started_second = asyncio.Event()
        release = asyncio.Event()

        async def slow_tool(**_: Any) -> Dict[str, int]:
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            if not started_first.is_set():
                started_first.set()
            else:
                started_second.set()
            await release.wait()
            in_flight -= 1
            return {"max_in_flight": max_in_flight}

        runtime._tools["slow_tool"] = slow_tool
        server = ToolServer(runtime)
        assert server._restriction_disabled is True

        first = asyncio.create_task(
            server._handle_execute(
                ToolPodExecuteRequest(
                    execution_id="exec-1",
                    tool_name="slow_tool",
                    arguments={},
                    session_id="session-1",
                )
            )
        )
        await started_first.wait()
        second = asyncio.create_task(
            server._handle_execute(
                ToolPodExecuteRequest(
                    execution_id="exec-2",
                    tool_name="slow_tool",
                    arguments={},
                    session_id="session-1",
                )
            )
        )
        await started_second.wait()
        assert max_in_flight == 2
        release.set()
        first_resp, second_resp = await asyncio.gather(first, second)
        assert first_resp.status == "success"
        assert second_resp.status == "success"

    @pytest.mark.asyncio
    async def test_concurrent_execute_on_different_servers_can_overlap(self):
        """Separate ToolServer instances represent separate pods and stay concurrent."""
        import asyncio

        runtime_a = _make_runtime()
        runtime_b = _make_runtime()
        in_flight = 0
        max_in_flight = 0
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_tool(**_: Any) -> Dict[str, int]:
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            started.set()
            await release.wait()
            in_flight -= 1
            return {"max_in_flight": max_in_flight}

        runtime_a._tools["slow_tool"] = slow_tool
        runtime_b._tools["slow_tool"] = slow_tool
        server_a = ToolServer(runtime_a)
        server_b = ToolServer(runtime_b)

        first = asyncio.create_task(
            server_a._handle_execute(
                ToolPodExecuteRequest(
                    execution_id="exec-a",
                    tool_name="slow_tool",
                    arguments={},
                    session_id="session-1",
                )
            )
        )
        await started.wait()
        started.clear()
        second = asyncio.create_task(
            server_b._handle_execute(
                ToolPodExecuteRequest(
                    execution_id="exec-b",
                    tool_name="slow_tool",
                    arguments={},
                    session_id="session-1",
                )
            )
        )
        await started.wait()
        assert max_in_flight == 2
        release.set()
        await asyncio.gather(first, second)

    @pytest.mark.asyncio
    async def test_execute_gate_released_after_tool_error(self):
        """A failed tool must not leave the execute gate locked."""
        import asyncio

        runtime = _make_runtime()

        def failing_tool(**_: Any) -> None:
            raise RuntimeError("boom")

        runtime._tools["failing_tool"] = failing_tool
        server = ToolServer(runtime)

        error_resp = await server._handle_execute(
            ToolPodExecuteRequest(
                execution_id="exec-err",
                tool_name="failing_tool",
                arguments={},
                session_id="session-1",
            )
        )
        assert error_resp.status == "error"

        done = asyncio.Event()

        async def quick_tool(**_: Any) -> str:
            done.set()
            return "ok"

        runtime._tools["quick_tool"] = quick_tool
        ok_resp = await server._handle_execute(
            ToolPodExecuteRequest(
                execution_id="exec-ok",
                tool_name="quick_tool",
                arguments={},
                session_id="session-1",
            )
        )
        assert ok_resp.status == "success"
        assert done.is_set()

    @pytest.mark.asyncio
    async def test_invalid_execute_does_not_wait_for_gate(self):
        """Unknown-tool requests are rejected without queueing behind the gate."""
        import asyncio

        runtime = _make_runtime()
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_tool(**_: Any) -> str:
            started.set()
            await release.wait()
            return "done"

        runtime._tools["slow_tool"] = slow_tool
        server = ToolServer(runtime)

        first = asyncio.create_task(
            server._handle_execute(
                ToolPodExecuteRequest(
                    execution_id="exec-1",
                    tool_name="slow_tool",
                    arguments={},
                    session_id="session-1",
                )
            )
        )
        await started.wait()

        resp = await asyncio.wait_for(
            server._handle_execute(
                ToolPodExecuteRequest(
                    execution_id="exec-2",
                    tool_name="missing_tool",
                    arguments={},
                    session_id="session-1",
                )
            ),
            timeout=1.0,
        )
        assert resp.status == "error"
        assert "Unknown tool" in (resp.error or "")

        release.set()
        first_resp = await first
        assert first_resp.status == "success"

    @pytest.mark.asyncio
    async def test_invoke_llm_overlaps_with_execute(self):
        """unrestricted pods skip the execute gate so LLM and tool calls overlap."""
        import asyncio

        from agent_engine_sdk.models import LLMStreamChunk

        from agent_engine_runner_shared.models import InvokeLLMRequestArguments, LLMPodInvokeRequest

        runtime = _make_runtime()
        server = ToolServer(runtime)

        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_chunks(_request):
            started.set()
            await release.wait()
            yield LLMStreamChunk(content="hi")

        server._stream_llm_chunks = slow_chunks

        llm_task = asyncio.create_task(
            server._handle_invoke_llm(
                LLMPodInvokeRequest(
                    execution_id="exec-llm",
                    arguments=InvokeLLMRequestArguments(
                        model="test-model",
                        llm_id="primary",
                        messages=[{"role": "user", "content": "Hi"}],
                    ),
                )
            )
        )
        await started.wait()

        ran = asyncio.Event()

        async def quick_tool(**_: Any) -> str:
            ran.set()
            return "ok"

        runtime._tools["quick_tool"] = quick_tool
        exec_task = asyncio.create_task(
            server._handle_execute(
                ToolPodExecuteRequest(
                    execution_id="exec-1",
                    tool_name="quick_tool",
                    arguments={},
                    session_id="session-1",
                )
            )
        )
        # Drain the scheduler (no wall-clock wait): the unrestricted path
        # skips the gate, so execute must already have entered the tool body.
        for _ in range(10):
            await asyncio.sleep(0)
        assert ran.is_set(), "execute must overlap an in-flight LLM invoke"

        release.set()
        llm_resp, exec_resp = await asyncio.gather(llm_task, exec_task)
        assert llm_resp.status == "success"
        assert exec_resp.status == "success"
        assert ran.is_set()

    def test_execute_request_rejects_empty_session_id(self):
        with pytest.raises(ValidationError):
            ToolPodExecuteRequest(
                execution_id="exec-1",
                tool_name="current_session",
                arguments={},
                session_id="",
            )

    def test_validation_response_omits_rejected_arguments(self, monkeypatch):
        from fastapi.testclient import TestClient

        import agent_engine_runner_shared.models as models

        monkeypatch.setattr(models, "MAX_TOOL_ARGUMENT_BYTES", 20)
        secret = "sensitive-tool-argument"
        app = ToolServer(_make_runtime(deep_agent_enabled=False)).create_app()

        response = TestClient(app).post(
            "/execute",
            json={
                "execution_id": "exec-1",
                "tool_name": "write",
                "arguments": {"content": secret},
                "session_id": "session-1",
            },
        )

        assert response.status_code == 422
        assert secret not in response.text
        assert response.json() == {
            "detail": [
                {
                    "type": "value_error",
                    "loc": ["body", "arguments"],
                    "msg": "Value error, tool arguments exceed 20 bytes",
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_startup_is_idempotent(self):
        """Running startup twice must not duplicate entries or crash."""
        runtime = _make_runtime()
        server = ToolServer(runtime)

        await server.on_startup()
        await server.on_startup()

        assert set(runtime._tools.keys()) == EXPECTED_TOOLS

    @pytest.mark.asyncio
    async def test_connector_startup_is_idempotent(self, monkeypatch, tmp_path):
        bundle_path = _write_connector_bundle(tmp_path)
        monkeypatch.setenv(CONNECTOR_BUNDLE_PATH_ENV, str(bundle_path))
        runtime = _make_runtime(deep_agent_enabled=False)
        server = ToolServer(runtime)

        await server.on_startup()
        first_callable = runtime._tools["jiradc_getComments"]
        await server.on_startup()

        assert set(runtime._tools) == {"jiradc_getComments", "jiradc_addComment"}
        assert runtime._tools["jiradc_getComments"] is first_callable
        await server.on_shutdown()

    @pytest.mark.asyncio
    async def test_existing_user_tools_are_preserved(self):
        """A user-registered tool must survive the builtin registration pass."""
        runtime = _make_runtime()

        def my_tool(**_: Any) -> Dict[str, Any]:
            return {"ok": True}

        runtime._tools["my_tool"] = my_tool
        runtime._tool_definitions["my_tool"] = {"name": "my_tool"}

        server = ToolServer(runtime)
        await server.on_startup()

        assert "my_tool" in runtime._tools
        assert EXPECTED_TOOLS.issubset(set(runtime._tools.keys()))

    @pytest.mark.asyncio
    async def test_user_tool_colliding_with_builtin_name_is_overridden_with_warning(self, caplog):
        """Built-in wins over user @app.tool() with the same name, and warns loudly."""
        import logging as _logging

        import agent_engine_runner_shared.toolpod_handlers as toolpod_handlers

        runtime = _make_runtime()

        def user_ls(**_: Any) -> Dict[str, Any]:
            return {"from_user": True}

        runtime._tools["filesystem_ls"] = user_ls
        runtime._tool_definitions["filesystem_ls"] = {"name": "filesystem_ls"}

        with caplog.at_level(_logging.WARNING, logger=toolpod_handlers.__name__):
            server = ToolServer(runtime)
            await server.on_startup()

        assert runtime._tools["filesystem_ls"] is toolpod_handlers.filesystem_ls
        assert any(
            "overriding user-registered tool" in rec.message and "filesystem_ls" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_startup_fails_loudly_when_handler_missing(self, monkeypatch):
        """FACT-07: missing built-in handler raises RuntimeError, not silent skip."""
        import agent_engine_runner_shared.toolpod_handlers as toolpod_handlers

        def incomplete_register(runtime):
            # Register everything except filesystem_download
            from agent_engine_runner_shared.toolpod_handlers import (
                filesystem_edit,
                filesystem_glob,
                filesystem_grep,
                filesystem_ls,
                filesystem_read,
                filesystem_write,
                shell_execute,
            )

            partial = {
                "filesystem_ls": filesystem_ls,
                "filesystem_read": filesystem_read,
                "filesystem_write": filesystem_write,
                "filesystem_edit": filesystem_edit,
                "filesystem_glob": filesystem_glob,
                "filesystem_grep": filesystem_grep,
                "shell_execute": shell_execute,
            }
            for name, fn in partial.items():
                runtime._tools[name] = fn
                runtime._tool_definitions[name] = {"name": name}

        monkeypatch.setattr(toolpod_handlers, "register_builtin_tools", incomplete_register)

        runtime = _make_runtime()
        server = ToolServer(runtime)

        with pytest.raises(RuntimeError, match="missing built-in tool"):
            await server.on_startup()


class TestWorkspaceDirValidation:
    """Startup warns when WORKSPACE_DIR is outside the known writable mounts."""

    @pytest.mark.asyncio
    async def test_out_of_allowlist_workspace_emits_warning(self, monkeypatch, caplog):
        """Warning fires before os.makedirs so it is captured even if makedirs fails."""
        import logging

        import agent_engine_runner_shared.toolpod_handlers as toolpod_handlers

        monkeypatch.setattr(toolpod_handlers, "WORKSPACE_DIR", "/home/agent/workspace")
        runtime = _make_runtime()
        with caplog.at_level(logging.WARNING, logger=toolpod_handlers.__name__):
            # makedirs will raise on most CI hosts because /home/agent doesn't
            # exist and can't be created. The warning fires first (before makedirs),
            # so we catch the OS error and verify the log entry.
            try:
                await ToolServer(runtime).on_startup()
            except OSError:
                pass

        assert any(
            "WORKSPACE_DIR" in rec.message and "writable" in rec.message for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_known_good_workspace_no_warning(self, monkeypatch, caplog):
        import logging

        import agent_engine_runner_shared.toolpod_handlers as toolpod_handlers

        monkeypatch.setattr(toolpod_handlers, "WORKSPACE_DIR", "/tmp/my-workspace")
        runtime = _make_runtime()
        with caplog.at_level(logging.WARNING, logger=toolpod_handlers.__name__):
            await ToolServer(runtime).on_startup()

        assert not any(
            "WORKSPACE_DIR" in rec.message and "writable" in rec.message for rec in caplog.records
        )


class TestToolServerDeepAgentGate:
    """``features.deep_agent`` opt-in controls built-in-handler registration."""

    @pytest.mark.asyncio
    async def test_flag_off_does_not_register_builtin_handlers(self):
        runtime = _make_runtime(deep_agent_enabled=False)
        await ToolServer(runtime).on_startup()
        assert runtime._tools == {}
        assert runtime._tool_definitions == {}

    @pytest.mark.asyncio
    async def test_flag_off_preserves_user_registered_tools(self):
        runtime = _make_runtime(deep_agent_enabled=False)
        runtime._tools["my_tool"] = lambda: "user"
        runtime._tool_definitions["my_tool"] = {"name": "my_tool"}
        await ToolServer(runtime).on_startup()
        assert set(runtime._tools.keys()) == {"my_tool"}


class TestToolServerMCPExecution:
    """Tool Pods resolve configured MCP tools per request without startup discovery."""

    @pytest.mark.asyncio
    async def test_executes_configured_mcp_tool_when_not_registered(self, monkeypatch):
        mcp_config = RuntimeMCPConfig.model_validate(
            {
                "servers": {
                    "github": {
                        "url": "https://api.githubcopilot.com/mcp/",
                        "allowed_tools": ["search_issues"],
                    }
                }
            }
        )
        runtime = _make_runtime(deep_agent_enabled=False, mcp_config=mcp_config)

        async def fake_call_mcp_tool(binding, arguments):
            assert binding.server_name == "github"
            assert binding.tool_name == "search_issues"
            assert arguments == {"query": "status:open"}
            return MCPToolResult(content=[{"type": "text", "text": "ok"}])

        monkeypatch.setattr(
            "agent_engine_runner_shared.mcp_tools._call_mcp_tool", fake_call_mcp_tool
        )

        response = await ToolServer(runtime)._handle_execute(
            ToolPodExecuteRequest(
                execution_id="exec-1",
                tool_name="github__search_issues",
                arguments={"query": "status:open"},
                session_id="session-1",
                metadata={"mcp_server": "github", "mcp_tool": "search_issues"},
            )
        )

        assert response.status == "success"
        assert response.result == MCPToolResult(content=[{"type": "text", "text": "ok"}])

    @pytest.mark.asyncio
    async def test_executes_discovered_mcp_tool_original_name(self, monkeypatch):
        mcp_config = RuntimeMCPConfig.model_validate(
            {
                "servers": {
                    "tableau": {
                        "url": "https://tableau.example.com/mcp",
                    }
                }
            }
        )
        runtime = _make_runtime(deep_agent_enabled=False, mcp_config=mcp_config)

        async def fake_call_mcp_tool(binding, arguments):
            assert binding.server_name == "tableau"
            assert binding.tool_name == "get-view-data"
            assert arguments == {"view_id": "view-1"}
            return MCPToolResult(content=[{"type": "text", "text": "ok"}])

        monkeypatch.setattr(
            "agent_engine_runner_shared.mcp_tools._call_mcp_tool", fake_call_mcp_tool
        )

        response = await ToolServer(runtime)._handle_execute(
            ToolPodExecuteRequest(
                execution_id="exec-1",
                tool_name="tableau__get_view_data",
                arguments={"view_id": "view-1"},
                session_id="session-1",
                metadata={"mcp_server": "tableau", "mcp_tool": "get-view-data"},
            )
        )

        assert response.status == "success"
        assert response.result == MCPToolResult(content=[{"type": "text", "text": "ok"}])

    @pytest.mark.parametrize(
        "metadata",
        [
            {},
            {"mcp_server": "tableau"},
            {"mcp_server": "", "mcp_tool": ""},
        ],
    )
    @pytest.mark.asyncio
    async def test_rejects_mcp_tool_without_complete_discovered_metadata(self, metadata):
        mcp_config = RuntimeMCPConfig.model_validate(
            {
                "servers": {
                    "tableau": {
                        "url": "https://tableau.example.com/mcp",
                    }
                }
            }
        )
        runtime = _make_runtime(deep_agent_enabled=False, mcp_config=mcp_config)

        response = await ToolServer(runtime)._handle_execute(
            ToolPodExecuteRequest(
                execution_id="exec-1",
                tool_name="tableau__get_view_data",
                arguments={"view_id": "view-1"},
                session_id="session-1",
                metadata=metadata,
            )
        )

        assert response.status == "error"
        assert response.error is not None
        assert "missing discovered MCP call metadata" in response.error
        assert "mcp_server, mcp_tool" in response.error
        assert "Ensure AER has been upgraded" in response.error

    @pytest.mark.asyncio
    async def test_rejects_mcp_tool_metadata_for_unconfigured_server(self):
        mcp_config = RuntimeMCPConfig.model_validate(
            {
                "servers": {
                    "tableau": {
                        "url": "https://tableau.example.com/mcp",
                    }
                }
            }
        )
        runtime = _make_runtime(deep_agent_enabled=False, mcp_config=mcp_config)

        response = await ToolServer(runtime)._handle_execute(
            ToolPodExecuteRequest(
                execution_id="exec-1",
                tool_name="evil_server__any_tool",
                arguments={"view_id": "view-1"},
                session_id="session-1",
                metadata={"mcp_server": "evil_server", "mcp_tool": "any-tool"},
            )
        )

        assert response.status == "error"
        assert response.error is not None
        assert "Unknown tool: evil_server__any_tool" in response.error

    @pytest.mark.asyncio
    async def test_rejects_mcp_tool_metadata_outside_allowed_tools(self):
        mcp_config = RuntimeMCPConfig.model_validate(
            {
                "servers": {
                    "tableau": {
                        "url": "https://tableau.example.com/mcp",
                        "allowed_tools": ["get-view-data"],
                    }
                }
            }
        )
        runtime = _make_runtime(deep_agent_enabled=False, mcp_config=mcp_config)

        response = await ToolServer(runtime)._handle_execute(
            ToolPodExecuteRequest(
                execution_id="exec-1",
                tool_name="tableau__forbidden_tool",
                arguments={"view_id": "view-1"},
                session_id="session-1",
                metadata={"mcp_server": "tableau", "mcp_tool": "forbidden-tool"},
            )
        )

        assert response.status == "error"
        assert response.error is not None
        assert (
            "mcp server 'tableau' tool 'forbidden-tool' is not in allowed_tools" in response.error
        )

    @pytest.mark.asyncio
    async def test_executes_oauth_mcp_tool(self, monkeypatch):
        mcp_config = RuntimeMCPConfig.model_validate(
            {
                "servers": {
                    "sentry": {
                        "url": "https://mcp.sentry.dev/mcp",
                        "auth": {"type": "oauth"},
                        "allowed_tools": ["search_events"],
                    }
                }
            }
        )
        runtime = _make_runtime(deep_agent_enabled=False, mcp_config=mcp_config)

        async def fake_call_mcp_tool(binding, arguments):
            assert binding.server_name == "sentry"
            assert binding.tool_name == "search_events"
            assert arguments == {"query": "level:error"}
            return MCPToolResult(content=[{"type": "text", "text": "ok"}])

        monkeypatch.setattr(
            "agent_engine_runner_shared.mcp_tools._call_mcp_tool", fake_call_mcp_tool
        )

        response = await ToolServer(runtime)._handle_execute(
            ToolPodExecuteRequest(
                execution_id="exec-1",
                tool_name="sentry__search_events",
                arguments={"query": "level:error"},
                session_id="session-1",
                metadata={"mcp_server": "sentry", "mcp_tool": "search_events"},
            )
        )

        assert response.status == "success"
        assert response.result == MCPToolResult(content=[{"type": "text", "text": "ok"}])


class TestBuildUvicornLogConfig:
    """build_uvicorn_log_config must level-route uvicorn logs: INFO to stdout,
    WARNING+ to stderr."""

    def test_default_handler_uses_stdout(self):
        from agent_engine_runner_shared.server.base import build_uvicorn_log_config

        config = build_uvicorn_log_config(structured=False)
        default_handler = config["handlers"]["default"]
        assert default_handler["stream"] == "ext://sys.stdout", (
            "Uvicorn's 'default' handler must write to stdout so informational "
            "messages like 'Application startup complete.' are not misread as errors"
        )

    def test_warnings_and_errors_stay_on_stderr(self):
        """The stdout handler is level-capped and a stderr sibling takes WARNING+."""
        from agent_engine_runner_shared.server.base import build_uvicorn_log_config

        config = build_uvicorn_log_config(structured=False)
        assert "max_info_level" in config["handlers"]["default"]["filters"]

        stderr_handler = config["handlers"]["default_stderr"]
        assert stderr_handler["stream"] == "ext://sys.stderr"
        assert stderr_handler["level"] == "WARNING"

        uvicorn_handlers = config["loggers"]["uvicorn"]["handlers"]
        assert "default" in uvicorn_handlers and "default_stderr" in uvicorn_handlers

    def test_info_to_stdout_warning_to_stderr_end_to_end(self, monkeypatch):
        """Applying the config sends INFO to stdout and WARNING/ERROR to stderr."""
        import io
        import logging.config

        from agent_engine_runner_shared.server.base import build_uvicorn_log_config

        out, err = io.StringIO(), io.StringIO()
        monkeypatch.setattr("sys.stdout", out)
        monkeypatch.setattr("sys.stderr", err)
        uvicorn_logger = logging.getLogger("uvicorn")
        try:
            logging.config.dictConfig(build_uvicorn_log_config(structured=False))
            uvicorn_logger.info("application startup complete")
            uvicorn_logger.warning("a real warning")
            uvicorn_logger.error("a real error")
        finally:
            # Drop handlers bound to the StringIO buffers so later tests aren't affected.
            uvicorn_logger.handlers.clear()

        assert "application startup complete" in out.getvalue()
        assert "application startup complete" not in err.getvalue()
        assert "a real warning" in err.getvalue()
        assert "a real error" in err.getvalue()
        assert "a real warning" not in out.getvalue()

    def test_access_handler_unchanged(self):
        from agent_engine_runner_shared.server.base import build_uvicorn_log_config

        config = build_uvicorn_log_config(structured=False)
        access_handler = config["handlers"]["access"]
        assert access_handler["stream"] == "ext://sys.stdout"

    def test_does_not_mutate_uvicorn_defaults(self):
        from uvicorn.config import LOGGING_CONFIG as UVICORN_LOGGING_CONFIG

        from agent_engine_runner_shared.server.base import build_uvicorn_log_config

        before = deepcopy(UVICORN_LOGGING_CONFIG)
        build_uvicorn_log_config(structured=False)
        assert UVICORN_LOGGING_CONFIG == before, (
            "build_uvicorn_log_config must not mutate uvicorn's global LOGGING_CONFIG"
        )

        # The structured=True branch takes a different path — rebinding
        # each logger key via a dict-unpack, then a separate setdefault()
        # for the access-filter attachment — and is only actually safe
        # because that setdefault() finds the fresh dict the loop above
        # already rebound "uvicorn.access" to, not because it can't alias
        # the original. Exercise it too, not just the non-structured path.
        build_uvicorn_log_config(structured=True)
        assert UVICORN_LOGGING_CONFIG == before, (
            "build_uvicorn_log_config(structured=True) must not mutate uvicorn's "
            "global LOGGING_CONFIG either"
        )


def _snapshot_uvicorn_logging_state():
    """Snapshot the global state dictConfig() and install_structured_logging()
    mutate: root handlers, the three uvicorn loggers, sys.stdout/stderr, and
    the two excepthooks. The hooks install only once per process, so a test
    that installs without restoring leaks a patched hook into every later test.
    """
    import logging
    import sys
    import threading

    root = logging.getLogger()
    uvicorn_state = {
        name: {
            "handlers": list(logging.getLogger(name).handlers),
            "propagate": logging.getLogger(name).propagate,
            "filters": list(logging.getLogger(name).filters),
        }
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access")
    }
    return {
        "root_handlers": list(root.handlers),
        "root_level": root.level,
        "uvicorn": uvicorn_state,
        "stdout": sys.stdout,
        "stderr": sys.stderr,
        "excepthook": sys.excepthook,
        "thread_excepthook": threading.excepthook,
    }


def _restore_uvicorn_logging_state(snapshot) -> None:
    import logging
    import sys
    import threading

    root = logging.getLogger()
    root.handlers = snapshot["root_handlers"]
    root.setLevel(snapshot["root_level"])
    for name, state in snapshot["uvicorn"].items():
        lg = logging.getLogger(name)
        lg.handlers = state["handlers"]
        lg.propagate = state["propagate"]
        lg.filters = state["filters"]
    sys.stdout, sys.stderr = snapshot["stdout"], snapshot["stderr"]
    sys.excepthook = snapshot["excepthook"]
    threading.excepthook = snapshot["thread_excepthook"]


class TestBuildUvicornLogConfigStructured:
    """build_uvicorn_log_config(structured=True) routes uvicorn/uvicorn.error/
    uvicorn.access through the structured JSON pipeline instead of uvicorn's
    own plain-text handlers. Without it a route exception surfaced as
    level=WARNING with "ERROR:    " in the message text and no
    fields.exc_traceback.
    """

    def test_default_arg_is_derived_from_structured_logging_env(self, monkeypatch):
        """The existing call site (BaseServer.run) passes no argument, so the
        default must resolve from STRUCTURED_LOGGING — matching setup_logging's
        own gate — not silently stay False forever."""
        from agent_engine_runner_shared.server.base import build_uvicorn_log_config

        monkeypatch.setenv("STRUCTURED_LOGGING", "true")
        config = build_uvicorn_log_config()
        assert config["loggers"]["uvicorn"]["handlers"] == []
        assert config["loggers"]["uvicorn"]["propagate"] is True

        monkeypatch.delenv("STRUCTURED_LOGGING", raising=False)
        config = build_uvicorn_log_config()
        assert config["handlers"]["default"]["stream"] == "ext://sys.stdout"

    def test_structured_removes_handlers_and_enables_propagation(self):
        from agent_engine_runner_shared.server.base import build_uvicorn_log_config

        config = build_uvicorn_log_config(structured=True)
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            logger_cfg = config["loggers"][name]
            assert logger_cfg["handlers"] == [], f"{name} must have zero handlers of its own"
            assert logger_cfg["propagate"] is True, f"{name} must propagate to root"
        # uvicorn.error's own explicit level (INFO) must be preserved, not
        # dropped — LOG_LEVEL gating for these records happens at the
        # *originating* logger's own level, not at root.
        assert config["loggers"]["uvicorn.error"]["level"] == "INFO"

    def test_structured_attaches_access_filter_to_logger_not_handler(self):
        """No "access" handler exists anymore in structured mode, so the
        noise-suppression filter must move to the uvicorn.access logger
        itself rather than being silently dropped."""
        from agent_engine_runner_shared.server.base import build_uvicorn_log_config

        config = build_uvicorn_log_config(structured=True)
        assert "suppress_runner_access_noise" in config["loggers"]["uvicorn.access"].get(
            "filters", []
        )

    def test_structured_preserves_and_reuses_existing_root_handler(self):
        """Applying uvicorn's dictConfig must not close or replace a
        handler install_structured_logging() already put on root — uvicorn's
        config has no "root" key, so dictConfig should leave root alone
        entirely, but this is verified empirically rather than assumed."""
        import io
        import logging
        import logging.config

        from agent_engine_runner_shared.server.base import build_uvicorn_log_config

        snapshot = _snapshot_uvicorn_logging_state()
        root = logging.getLogger()
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        root.handlers = [handler]
        root.setLevel(logging.INFO)
        try:
            logging.config.dictConfig(build_uvicorn_log_config(structured=True))
            assert root.handlers == [handler], "root's handler list must be untouched"
            assert not buf.closed, "root's handler stream must not be closed"
            root.info("still works after dictConfig")
            assert "still works after dictConfig" in buf.getvalue()
        finally:
            _restore_uvicorn_logging_state(snapshot)

    def test_structured_end_to_end_uvicorn_error_reaches_root_with_exc_info(self):
        """The actual regression case: an uvicorn.error record logged with
        exc_info (mirroring uvicorn's own `self.logger.error(msg,
        exc_info=exc)` ASGI-exception path) must reach root with its real
        level and exception fields intact, not get stuck behind uvicorn's
        own propagate=False."""
        import io
        import json
        import logging
        import logging.config

        from agent_engine_runner_shared.server.base import build_uvicorn_log_config
        from agent_engine_runner_shared.structured_logging import install_structured_logging

        snapshot = _snapshot_uvicorn_logging_state()
        buf = io.StringIO()
        try:
            install_structured_logging(stream=buf, mode="aer")
            logging.config.dictConfig(build_uvicorn_log_config(structured=True))

            try:
                raise RuntimeError("route handler blew up")
            except RuntimeError as exc:
                logging.getLogger("uvicorn.error").error(
                    "Exception in ASGI application\n", exc_info=exc
                )

            records = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
            error_recs = [r for r in records if r["logger"] == "uvicorn.error"]
            assert len(error_recs) == 1, records
            rec = error_recs[0]
            assert rec["level"] == "ERROR", (
                "must be the real level, not the plain-text 'ERROR:' prefix uvicorn's "
                "own formatter bakes into a message string"
            )
            assert rec["fields"]["exc_type"] == "RuntimeError"
            assert "route handler blew up" in rec["fields"]["exc_traceback"]
        finally:
            _restore_uvicorn_logging_state(snapshot)

    def test_structured_informational_line_has_no_baked_in_levelprefix(self):
        """An ordinary INFO line (e.g. "Application startup complete.") must
        arrive as level=INFO with a clean message — no "INFO:    " prefix
        baked in — which is what proves the record went through our JSON
        formatter rather than uvicorn's own DefaultFormatter (whose whole
        job is to prepend %(levelprefix)s to plain text). The exception
        test above only proves the ERROR path; this proves the ordinary,
        far more common, non-exception path wasn't broken by removing
        uvicorn's own handler."""
        import io
        import json
        import logging
        import logging.config

        from agent_engine_runner_shared.server.base import build_uvicorn_log_config
        from agent_engine_runner_shared.structured_logging import install_structured_logging

        snapshot = _snapshot_uvicorn_logging_state()
        buf = io.StringIO()
        try:
            install_structured_logging(stream=buf, mode="aer")
            logging.config.dictConfig(build_uvicorn_log_config(structured=True))

            logging.getLogger("uvicorn.error").info("Application startup complete.")

            records = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
            rec = next(r for r in records if r["logger"] == "uvicorn.error")
            assert rec["level"] == "INFO"
            assert rec["message"] == "Application startup complete.", (
                "message must be the raw text uvicorn passed to .info() — a leading "
                "'INFO:    ' would mean uvicorn's own DefaultFormatter still ran"
            )
        finally:
            _restore_uvicorn_logging_state(snapshot)

    def test_structured_access_noise_filter_still_suppresses_health_checks(self):
        """End-to-end: a /health access-log line is still suppressed, and a
        real request path still comes through, once the filter lives on the
        logger instead of the (now-removed) access handler."""
        import io
        import json
        import logging
        import logging.config

        from agent_engine_runner_shared.server.base import build_uvicorn_log_config
        from agent_engine_runner_shared.structured_logging import install_structured_logging

        snapshot = _snapshot_uvicorn_logging_state()
        buf = io.StringIO()
        try:
            install_structured_logging(stream=buf, mode="aer")
            logging.config.dictConfig(build_uvicorn_log_config(structured=True))

            access_logger = logging.getLogger("uvicorn.access")
            access_logger.info(
                '%s - "%s %s HTTP/%s" %d',
                "127.0.0.1:1234",
                "GET",
                "/health",
                "1.1",
                200,
            )
            access_logger.info(
                '%s - "%s %s HTTP/%s" %d',
                "127.0.0.1:1234",
                "GET",
                "/execute",
                "1.1",
                200,
            )

            records = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
            messages = [r["message"] for r in records]
            assert not any("/health" in m for m in messages), messages
            assert any("/execute" in m for m in messages), messages
        finally:
            _restore_uvicorn_logging_state(snapshot)
