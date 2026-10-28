"""Tests for ``agent.yaml`` remote MCP tool support in the ADK adapter.

Parity with the LangGraph adapter: the AER discovers each configured server's
tools at App construction, exposes them to ADK with the server's own name,
description, and input JSON schema, and routes every call through the secure
wrapper with the MCP identity OE forwards to the Tool Pod.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_engine_runner_shared.agent_config import RuntimeMCPServerConfig
from agent_engine_runner_shared.mcp_tools import (
    MCPConfigError,
    MCPToolBinding,
    MCPToolResult,
)

_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query"},
    },
    "required": ["query"],
}


def _write_mcp_agent_yaml(
    tmp_path: Any,
    *,
    server_name: str = "github",
    tool_name: str = "search_issues",
    timeout: int = 45,
) -> None:
    (tmp_path / "agent.yaml").write_text(
        "\n".join(
            [
                "name: mcp-agent",
                "mcp:",
                "  servers:",
                f"    {server_name}:",
                "      url: https://api.githubcopilot.com/mcp/x/issues/readonly",
                "      allowed_tools:",
                f"        - {tool_name}",
                f"      timeout_seconds: {timeout}",
            ]
        ),
        encoding="utf-8",
    )


def _fake_list_server_tools(
    tool_name: str = "search_issues",
    input_schema: dict[str, Any] | None = None,
) -> Any:
    async def fake_list_server_tools(_server_name: str, _config: Any) -> list[Any]:
        return [
            SimpleNamespace(
                name=tool_name,
                description="Search issues.",
                input_schema=dict(input_schema or _SEARCH_SCHEMA),
            )
        ]

    return fake_list_server_tools


@pytest.fixture
def mcp_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    """A real App whose agent.yaml configures one remote MCP server."""
    monkeypatch.chdir(tmp_path)
    _write_mcp_agent_yaml(tmp_path)
    monkeypatch.setattr(
        "agent_engine_runner_shared.mcp_tools._list_server_tools",
        _fake_list_server_tools(),
    )
    from agent_engine_sdk_adk.runtime import App

    return App(app_name="MCP Agent")


def _binding(
    *,
    server_name: str = "github",
    tool_name: str = "search_issues",
    description: str = "Search issues.",
) -> MCPToolBinding:
    return MCPToolBinding(
        server_name=server_name,
        tool_name=tool_name,
        sdk_tool_name=f"{server_name}__{tool_name}",
        description=description,
        input_schema=dict(_SEARCH_SCHEMA),
        server_config=RuntimeMCPServerConfig(
            url="https://api.githubcopilot.com/mcp/x/issues/readonly"
        ),
    )


class TestMCPRegistration:
    def test_init_registers_mcp_tools_from_agent_yaml(self, mcp_app: Any) -> None:
        """AER discovers agent.yaml MCP servers and registers them as remote tools."""
        metadata = mcp_app._runtime.get_tool_metadata("github__search_issues")

        assert metadata["description"] == "Search issues."
        assert metadata["is_local"] is False
        assert metadata["network"] == ["api.githubcopilot.com"]
        assert metadata["timeout_seconds"] == 45
        assert metadata["redact_fields"] == []
        assert metadata["mcp_server"] == "github"
        assert metadata["mcp_tool"] == "search_issues"
        assert "github__search_issues" in mcp_app._raw_tools

        (tool_def,) = mcp_app.get_tool_definitions()
        assert tool_def.name == "github__search_issues"
        assert tool_def.description == "Search issues."
        assert tool_def.args_schema == _SEARCH_SCHEMA
        assert tool_def.remote is True
        assert tool_def.network == ["api.githubcopilot.com"]
        assert tool_def.timeout_seconds == 45

    def test_mcp_tool_names_are_sanitized(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Punctuation in server/tool names becomes the SDK-visible identifier."""
        monkeypatch.chdir(tmp_path)
        _write_mcp_agent_yaml(tmp_path, tool_name="get-view-data")
        monkeypatch.setattr(
            "agent_engine_runner_shared.mcp_tools._list_server_tools",
            _fake_list_server_tools("get-view-data"),
        )
        from agent_engine_sdk_adk.runtime import App

        app = App(app_name="MCP Agent")

        assert "github__get_view_data" in app._raw_tools
        assert app._runtime.get_tool_metadata("github__get_view_data")["mcp_tool"] == (
            "get-view-data"
        )

    def test_init_skips_mcp_discovery_outside_aer_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Only the AER discovers schemas; Tool Pods resolve calls from config."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("RUNNER_MODE", "tool")
        _write_mcp_agent_yaml(tmp_path)

        with patch("agent_engine_sdk_adk.runtime.discover_mcp_tools") as mock_discover:
            from agent_engine_sdk_adk.runtime import App

            app = App(app_name="MCP Tool Pod")

        mock_discover.assert_not_called()
        assert app._raw_tools == {}
        assert app._mcp_tools == {}

    def test_duplicate_mcp_tool_registration_fails_fast(self, mcp_app: Any) -> None:
        """A second registration of the same MCP tool is a configuration error."""
        with pytest.raises(MCPConfigError, match="already registered"):
            mcp_app._register_mcp_tool(_binding())

    def test_user_tool_colliding_with_mcp_tool_fails_fast(self, mcp_app: Any) -> None:
        """A user tool may not shadow an already-registered MCP tool."""
        with pytest.raises(ValueError, match="already registered"):

            @mcp_app.tool()
            def github__search_issues(query: str) -> str:
                """Colliding user tool."""
                return query

    def test_mcp_tool_colliding_with_user_tool_fails_fast(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """MCP registration may not shadow an already-registered SDK tool."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.yaml").write_text(
            "name: collision-agent\n", encoding="utf-8"
        )
        from agent_engine_sdk_adk.runtime import App

        app = App(app_name="Collision Agent")

        @app.tool()
        def github__search_issues(query: str) -> str:
            """Existing tool."""
            return query

        with pytest.raises(MCPConfigError, match="already registered"):
            app._register_mcp_tool(_binding())


class TestMCPToolDeclaration:
    def test_tools_expose_mcp_name_description_and_json_schema(
        self, mcp_app: Any
    ) -> None:
        """The model-facing declaration is the server's own name, text, and schema."""
        (tool,) = mcp_app.tools()

        declaration = tool._get_declaration()
        assert declaration.name == "github__search_issues"
        assert declaration.description == "Search issues."
        assert declaration.parameters_json_schema == _SEARCH_SCHEMA

    def test_declaration_schema_survives_the_secure_llm_bridge(
        self, mcp_app: Any
    ) -> None:
        """The provider request must carry the MCP tool's arguments.

        The platform's secure bridge (``tools_dict_to_schemas``) and the ADK
        request builder read only ``parameters_json_schema``; a declaration
        that carried the schema anywhere else would reach the provider with no
        arguments. The feature override disables ADK's JSON-schema flag so this
        test exercises the configuration in which the removed Gemini-Schema
        fallback lost the schema — restoring that fallback fails here.
        """
        from google.adk.features import FeatureName

        # Private ADK hook: the feature flag has no public override, so the
        # test uses this one deliberately; it restores the prior state on exit.
        # A rename in ADK should fail this import loudly rather than skip — the
        # regression signal below is the point of the test.
        from google.adk.features._feature_registry import (  # pyright: ignore[reportPrivateImportUsage]
            temporary_feature_override,
        )

        from agent_engine_sdk_adk.llm_adapter import ADKLLMAdapter
        from agent_engine_sdk_adk.messages import tools_dict_to_schemas

        (tool,) = mcp_app.tools()
        with temporary_feature_override(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, False):
            schemas = tools_dict_to_schemas({tool.name: tool})
            assert [schema.parameters for schema in schemas] == [_SEARCH_SCHEMA]

            mock_llm = MagicMock()
            mock_llm.model = "gemini-2.5-flash"
            request = ADKLLMAdapter(llm=mock_llm, tools=schemas)._build_request([])

        declaration = request.config.tools[0].function_declarations[0]
        assert declaration.name == "github__search_issues"
        assert declaration.parameters_json_schema == _SEARCH_SCHEMA


class TestMCPToolExecution:
    async def test_run_async_invokes_secure_wrapper_with_call_id(self) -> None:
        """ADK's function_call_id is forwarded as the stable tool_call_id."""
        from agent_engine_sdk_adk.mcp import RemoteMCPTool

        calls: list[dict[str, Any]] = []

        def invoke(**kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {"content": [], "is_error": False}

        tool = RemoteMCPTool(binding=_binding(), invoke=invoke)
        context = SimpleNamespace(function_call_id="fc-42")

        result = await tool.run_async(args={"query": "hello"}, tool_context=context)

        assert result == {"content": [], "is_error": False}
        assert calls == [{"tool_call_id": "fc-42", "query": "hello"}]

    async def test_run_async_consumes_reserved_tool_call_id_argument(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A schema property named tool_call_id must not collide with the wrapper.

        The model-supplied value is dropped — the platform's call identity must
        win — and the drop is logged so the collision is debuggable.
        """
        from agent_engine_sdk_adk.mcp import RemoteMCPTool

        calls: list[dict[str, Any]] = []

        def invoke(**kwargs: Any) -> str:
            calls.append(kwargs)
            return "ok"

        tool = RemoteMCPTool(binding=_binding(), invoke=invoke)
        context = SimpleNamespace(function_call_id="fc-1")

        with caplog.at_level(logging.WARNING):
            await tool.run_async(
                args={"query": "q", "tool_call_id": "model-supplied"},
                tool_context=context,
            )

        assert calls == [{"tool_call_id": "fc-1", "query": "q"}]
        assert "tool_call_id" in caplog.text

    async def test_run_async_normalizes_mcp_result_for_adk(self) -> None:
        """Direct execution returns MCPToolResult; ADK needs a JSON dict."""
        from agent_engine_sdk_adk.mcp import RemoteMCPTool

        def invoke(**_kwargs: Any) -> MCPToolResult:
            return MCPToolResult(
                content=[{"type": "text", "text": "found"}],
                structured_content={"total": 1},
                is_error=False,
            )

        tool = RemoteMCPTool(binding=_binding(), invoke=invoke)

        result = await tool.run_async(
            args={"query": "q"},
            tool_context=SimpleNamespace(function_call_id=None),
        )

        assert result == {
            "content": [{"type": "text", "text": "found"}],
            "structured_content": {"total": 1},
            "is_error": False,
        }

    async def test_run_async_does_not_block_event_loop(self) -> None:
        """The OE round trip runs on a worker thread, keeping the loop responsive.

        The stub's wait is finite so an inline-invocation regression fails the
        test with a clear assertion instead of parking the loop past every
        timeout (``wait_for`` cannot fire while the loop is blocked) and
        hanging the job.
        """
        import threading

        from agent_engine_sdk_adk.mcp import RemoteMCPTool

        unblock = threading.Event()
        watcher_ran = False

        def invoke(**_kwargs: Any) -> str:
            assert unblock.wait(timeout=30), "invocation blocked the event loop"
            return "done"

        tool = RemoteMCPTool(binding=_binding(), invoke=invoke)

        async def watcher() -> str:
            nonlocal watcher_ran
            await asyncio.sleep(0)
            watcher_ran = True
            unblock.set()
            return "health-ok"

        result, watcher_result = await asyncio.wait_for(
            asyncio.gather(
                tool.run_async(
                    args={"query": "q"},
                    tool_context=SimpleNamespace(function_call_id=None),
                ),
                watcher(),
            ),
            timeout=5.0,
        )

        assert result == "done"
        assert watcher_result == "health-ok"
        assert watcher_ran

    def test_tools_wire_secure_wrapper_with_mcp_identity(self, mcp_app: Any) -> None:
        """AER forwards discovered MCP names for Tool Pod calls, like LangGraph."""
        with patch(
            "agent_engine_sdk_adk.runtime.create_secure_tool_function"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            tools = mcp_app.tools()

        mock_create.assert_called_once()
        kwargs = mock_create.call_args.kwargs
        assert kwargs["tool_name"] == "github__search_issues"
        assert kwargs["metadata"] == {
            "mcp_server": "github",
            "mcp_tool": "search_issues",
        }
        assert kwargs["is_local"] is False
        assert kwargs["redact_fields"] == []
        (tool,) = tools
        assert tool.name == "github__search_issues"


class TestMCPHelpers:
    def test_description_falls_back_to_remote_label(self) -> None:
        """A server tool without a description still gets a usable label."""
        from agent_engine_sdk_adk.mcp import mcp_tool_description

        assert mcp_tool_description(_binding(description="")) == (
            "Remote MCP tool search_issues"
        )
