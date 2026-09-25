from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages.tool import ToolCall
from langchain_core.utils.function_calling import convert_to_openai_tool
from agent_engine_sdk import BaseApp

from agent_engine_sdk_langgraph import App, LangGraphOutputParser, Memory
from agent_engine_sdk_langgraph.memory_appbound import (
    AppBoundCrudClient,
    AppBoundRuntime,
)
from agent_engine_runner_shared.context import (
    clear_execution_context,
    set_execution_context,
)
from agent_engine_runner_shared.connectors import (
    CONNECTOR_BUNDLE_PATH_ENV,
    ConnectorBundleError,
)
from agent_engine_runner_shared.mcp_tools import MCPConfigError


def _write_connector_bundle(
    tmp_path: Path, *, tool_names: list[str] | None = None
) -> Path:
    names = tool_names or ["jiradc_getComments", "jiradc_addComment"]
    definitions = {
        "source": {"name": "jiradc", "base_url": "https://jira.example/rest"},
        "auth": {"type": "bearer", "env": "JIRA_TOKEN"},
        "tools": [
            {
                "name": name,
                "description": f"Description for {name}.",
                "method": "GET",
                "path": f"/api/2/issue/{{issueIdOrKey}}/{index}",
                "params": {"path": ["issueIdOrKey"]},
                "inputSchema": {
                    "type": "object",
                    "properties": {"issueIdOrKey": {"type": "string"}},
                    "required": ["issueIdOrKey"],
                },
            }
            for index, name in enumerate(names)
        ],
    }
    definitions_path = tmp_path / "jiradc.tool_defs.yaml"
    import yaml

    definitions_path.write_text(
        yaml.safe_dump(definitions, sort_keys=False), encoding="utf-8"
    )
    index_path = tmp_path / "bundle.yaml"
    index_path.write_text(
        "version: 1\nconnectors:\n  - name: jiradc\n    tool_defs: jiradc.tool_defs.yaml\n",
        encoding="utf-8",
    )
    return index_path


class TestApp:
    """Tests for App class."""

    def test_is_base_app_subclass(self):
        """App is a BaseApp subclass."""
        assert issubclass(App, BaseApp)

    def test_init_creates_tenant_runtime(self):
        """App initialization creates underlying TenantRuntime."""
        app = App(app_name="Test Agent")

        assert app.name == "Test Agent"
        assert app._runtime is not None
        assert app._builder_fn is None

    def test_deprecation_warning_on_org_id(self, monkeypatch):
        """App(org_id=...) emits a DeprecationWarning and the env var wins."""
        monkeypatch.setenv("ORG_ID", "env_org")
        with pytest.warns(DeprecationWarning, match="org_id"):
            app = App(app_name="Test Agent", org_id="hardcoded_org")
        assert app._runtime.org_id == "env_org"

    def test_init_reads_agent_config_from_agent_yaml(self, monkeypatch, tmp_path):
        """App exposes the parsed agent.yaml loaded from the working directory."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.yaml").write_text(
            "\n".join(
                [
                    "name: config-agent",
                    "entrypoint: config.agent:app",
                    "features:",
                    "  memory: true",
                ]
            ),
            encoding="utf-8",
        )

        app = App(app_name="Configured Agent")

        assert app.agent_config.path == tmp_path / "agent.yaml"
        assert app.agent_config.entrypoint == "config.agent:app"
        assert app.agent_config.feature_enabled("memory") is True

    def test_init_registers_bundled_connector_tools_as_remote(
        self, monkeypatch, tmp_path
    ):
        index_path = _write_connector_bundle(tmp_path)
        monkeypatch.setenv(CONNECTOR_BUNDLE_PATH_ENV, str(index_path))

        app = App(app_name="Connector Agent")

        tool = app._lc_tools["jiradc_getComments"]
        assert convert_to_openai_tool(tool)["function"]["parameters"] == {
            "type": "object",
            "properties": {"issueIdOrKey": {"type": "string"}},
            "required": ["issueIdOrKey"],
        }
        metadata = app._runtime.get_tool_metadata("jiradc_getComments")
        assert metadata == {
            "name": "jiradc_getComments",
            "description": "Description for jiradc_getComments.",
            "is_local": False,
            "provider_type": None,
            "scopes": [],
            "network": ["jira.example"],
            "timeout_seconds": 30,
            "redact_fields": [],
            "connector": "jiradc",
        }
        assert app.get_tool_definitions()[0].remote is True

    def test_init_materializes_connectors_from_agent_yaml_traversal(
        self, monkeypatch, tmp_path
    ):
        """No pre-materialized bundle: agent.yaml `connectors` entries are
        materialized at init and registered as ordinary remote tools."""
        monkeypatch.delenv(CONNECTOR_BUNDLE_PATH_ENV, raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.yaml").write_text(
            "name: connector-agent\n"
            "entrypoint: config.agent:app\n"
            "connectors:\n"
            "  - source: connectors/jira/tool.yaml\n",
            encoding="utf-8",
        )
        connectors = tmp_path / "connectors" / "jira"
        connectors.mkdir(parents=True)
        import yaml

        definitions = {
            "source": {"name": "jiradc", "base_url": "https://jira.example/rest"},
            "auth": {"type": "bearer", "env": "JIRA_TOKEN"},
            "tools": [
                {
                    "name": "jiradc_getComments",
                    "description": "Returns comments for an issue.",
                    "method": "GET",
                    "path": "/api/2/issue/{issueIdOrKey}/comment",
                    "params": {"path": ["issueIdOrKey"]},
                    "inputSchema": {
                        "type": "object",
                        "properties": {"issueIdOrKey": {"type": "string"}},
                        "required": ["issueIdOrKey"],
                    },
                }
            ],
        }
        (connectors / "jira.tool_defs.yaml").write_text(
            yaml.safe_dump(definitions, sort_keys=False), encoding="utf-8"
        )
        (connectors / "tool.yaml").write_text(
            "name: jira\n"
            "tool_defs: jira.tool_defs.yaml\n"
            "source:\n"
            "  base_url: https://jira.example/rest\n"
            "expose:\n"
            "  allow_all: true\n"
            "auth:\n"
            "  type: bearer\n"
            "  env: JIRA_TOKEN\n",
            encoding="utf-8",
        )

        app = App(app_name="Connector Agent")

        assert "jiradc_getComments" in app._runtime._tools
        metadata = app._runtime.get_tool_metadata("jiradc_getComments")
        assert metadata["connector"] == "jira"
        assert metadata["is_local"] is False

    @patch("agent_engine_runner_shared.secure_wrapper.create_secure_tool_function")
    def test_bundled_connector_tool_uses_secure_remote_wrapper(
        self, mock_create_secure, monkeypatch, tmp_path
    ):
        monkeypatch.setenv(
            CONNECTOR_BUNDLE_PATH_ENV, str(_write_connector_bundle(tmp_path))
        )
        mock_create_secure.return_value = lambda **_kwargs: "wrapped"

        app = App(app_name="Connector Agent")
        app.get_tools()

        calls = {
            call.kwargs["tool_name"]: call.kwargs
            for call in mock_create_secure.call_args_list
        }
        assert calls["jiradc_getComments"]["is_local"] is False
        assert calls["jiradc_getComments"]["metadata"] is None

    def test_connector_collision_is_rejected_before_any_connector_registration(
        self, monkeypatch, tmp_path
    ):
        app = App(app_name="Connector Agent")

        def existing_tool() -> str:
            return "existing"

        app._register_tool_definition(
            name="jiradc_getComments",
            func=existing_tool,
            description="Existing tool",
            args_schema={},
            is_local=True,
            provider_type=None,
            scopes=[],
            network=[],
            timeout_seconds=30,
            redact_fields=[],
            langchain_tool=MagicMock(),
        )
        monkeypatch.setenv(
            CONNECTOR_BUNDLE_PATH_ENV, str(_write_connector_bundle(tmp_path))
        )

        with pytest.raises(ConnectorBundleError, match="jiradc_getComments"):
            app._register_connector_tools_from_bundle()

        assert "jiradc_addComment" not in app._runtime._tools

    def test_tool_mode_does_not_load_connector_schemas(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RUNNER_MODE", "tool")
        monkeypatch.setenv(CONNECTOR_BUNDLE_PATH_ENV, str(tmp_path / "missing.yaml"))

        app = App(app_name="Connector Tool Pod")

        assert app._lc_tools == {}

    @pytest.mark.parametrize(
        (
            "server_name",
            "url",
            "auth_lines",
            "tool_name",
            "expected_network",
            "timeout",
        ),
        [
            (
                "github",
                "https://api.githubcopilot.com/mcp/x/issues/readonly",
                [],
                "search_issues",
                ["api.githubcopilot.com"],
                45,
            ),
            (
                "sentry",
                "https://mcp.sentry.dev/mcp",
                ["      auth:", "        type: oauth"],
                "search_events",
                ["mcp.sentry.dev"],
                30,
            ),
        ],
    )
    def test_init_registers_mcp_tools_from_agent_yaml(
        self,
        monkeypatch,
        tmp_path,
        server_name,
        url,
        auth_lines,
        tool_name,
        expected_network,
        timeout,
    ):
        """App discovers configured MCP tools and registers them like SDK tools."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.yaml").write_text(
            "\n".join(
                [
                    "name: mcp-agent",
                    "mcp:",
                    "  servers:",
                    f"    {server_name}:",
                    f"      url: {url}",
                    *auth_lines,
                    "      allowed_tools:",
                    f"        - {tool_name}",
                    f"      timeout_seconds: {timeout}",
                ]
            ),
            encoding="utf-8",
        )

        async def fake_list_server_tools(actual_server_name, config):
            assert actual_server_name == server_name
            assert config.timeout_seconds == timeout
            return [
                SimpleNamespace(
                    name=tool_name,
                    description="",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                        },
                        "required": ["query"],
                    },
                )
            ]

        monkeypatch.setattr(
            "agent_engine_runner_shared.mcp_tools._list_server_tools",
            fake_list_server_tools,
        )
        app = App(app_name="MCP Agent")
        sdk_tool_name = f"{server_name}__{tool_name}"

        assert sdk_tool_name in app._lc_tools
        assert sdk_tool_name in app._runtime._tools
        expected_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        }
        langchain_tool = app._lc_tools[sdk_tool_name]
        assert langchain_tool.args_schema == expected_schema
        assert langchain_tool.description == f"Remote MCP tool {tool_name}"
        openai_tool = convert_to_openai_tool(langchain_tool)
        assert openai_tool["function"]["parameters"] == expected_schema

        metadata = app._runtime.get_tool_metadata(sdk_tool_name)
        assert metadata["description"] == f"Remote MCP tool {tool_name}"
        assert metadata["is_local"] is False
        assert metadata["network"] == expected_network
        assert metadata["timeout_seconds"] == timeout
        assert metadata["redact_fields"] == []
        assert metadata["mcp_server"] == server_name
        assert metadata["mcp_tool"] == tool_name

        tool_defs = app.get_tool_definitions()
        assert len(tool_defs) == 1
        assert tool_defs[0].name == sdk_tool_name
        assert tool_defs[0].description == f"Remote MCP tool {tool_name}"
        assert tool_defs[0].remote is True
        assert tool_defs[0].args_schema == expected_schema
        assert tool_defs[0].network == expected_network

    @patch("agent_engine_runner_shared.secure_wrapper.create_secure_tool_function")
    def test_get_tools_passes_mcp_original_name_metadata(
        self,
        mock_create_secure,
        monkeypatch,
        tmp_path,
    ):
        """AER forwards discovered MCP names for Tool Pod calls."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.yaml").write_text(
            "\n".join(
                [
                    "name: mcp-agent",
                    "mcp:",
                    "  servers:",
                    "    tableau:",
                    "      url: https://tableau.example.com/mcp",
                ]
            ),
            encoding="utf-8",
        )

        async def fake_list_server_tools(actual_server_name, _config):
            assert actual_server_name == "tableau"
            return [
                SimpleNamespace(
                    name="get-view-data",
                    description="",
                    inputSchema={"type": "object", "properties": {}},
                )
            ]

        monkeypatch.setattr(
            "agent_engine_runner_shared.mcp_tools._list_server_tools",
            fake_list_server_tools,
        )
        mock_create_secure.return_value = lambda **kwargs: "wrapped"

        app = App(app_name="MCP Agent")
        app.get_tools()

        mock_create_secure.assert_called_once()
        assert (
            mock_create_secure.call_args.kwargs["tool_name"] == "tableau__get_view_data"
        )
        assert mock_create_secure.call_args.kwargs["metadata"] == {
            "mcp_server": "tableau",
            "mcp_tool": "get-view-data",
        }

    def test_init_skips_mcp_discovery_outside_aer_mode(self, monkeypatch, tmp_path):
        """Only AER discovers MCP schemas."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("RUNNER_MODE", "tool")
        (tmp_path / "agent.yaml").write_text(
            "\n".join(
                [
                    "name: mcp-agent",
                    "mcp:",
                    "  servers:",
                    "    github:",
                    "      url: https://api.githubcopilot.com/mcp/x/issues/readonly",
                    "      allowed_tools:",
                    "        - search_issues",
                ]
            ),
            encoding="utf-8",
        )

        with patch(
            "agent_engine_sdk_langgraph.runtime.discover_mcp_tools"
        ) as mock_discover:
            app = App(app_name="MCP Tool Pod")

        mock_discover.assert_not_called()
        assert app._lc_tools == {}
        assert app._runtime._tools == {}

    def test_register_tool_definition_rejects_duplicate_mcp_tool(self):
        """MCP tool registration should fail fast."""
        app = App(app_name="Test Agent")

        def existing_tool() -> str:
            return "ok"

        app._register_tool_definition(
            name="github__search_issues",
            func=existing_tool,
            description="Existing tool",
            args_schema={},
            is_local=True,
            provider_type=None,
            scopes=[],
            network=[],
            timeout_seconds=30,
            redact_fields=[],
            langchain_tool=MagicMock(),
        )

        with pytest.raises(MCPConfigError, match="already registered"):
            app._register_tool_definition(
                name="github__search_issues",
                func=existing_tool,
                description="MCP tool",
                args_schema={},
                is_local=False,
                provider_type=None,
                scopes=[],
                network=["api.githubcopilot.com"],
                timeout_seconds=30,
                redact_fields=[],
                source="mcp",
            )

    def test_register_tool_definition_rejects_duplicate_user_tool(self):
        """User tools with duplicate names should fail."""
        app = App(app_name="Test Agent")

        @app.tool()
        def duplicate_tool() -> str:
            """First."""
            return "first"

        def duplicate_tool_again() -> str:
            """Second."""
            return "second"

        duplicate_tool_again.__name__ = "duplicate_tool"
        with pytest.raises(ValueError, match="already registered"):
            app.tool()(duplicate_tool_again)

    def test_register_tool_definition_strips_description(self):
        """Metadata and ToolDefinition descriptions should match."""
        app = App(app_name="Test Agent")

        def spaced_tool() -> str:
            return "ok"

        app._register_tool_definition(
            name="spaced_tool",
            func=spaced_tool,
            description="  Spaced description  ",
            args_schema={},
            is_local=True,
            provider_type=None,
            scopes=[],
            network=[],
            timeout_seconds=30,
            redact_fields=[],
            langchain_tool=MagicMock(),
        )

        assert app._runtime.get_tool_metadata("spaced_tool")["description"] == (
            "Spaced description"
        )
        assert app.get_tool_definitions()[0].description == "Spaced description"

    def test_memory_property_returns_memory_instance(self):
        """App.memory returns the re-exported facade over the app-bound adapters."""
        app = App(app_name="Test Agent")

        assert isinstance(app.memory, Memory)
        assert isinstance(app.memory._runtime, AppBoundRuntime)
        assert isinstance(app.memory._client, AppBoundCrudClient)

    def test_entrypoint_decorator_stores_builder_function(self):
        """@app.entrypoint decorator stores the builder function."""
        app = App(app_name="Test Agent")

        def build_agent():
            return "test_graph"

        decorated = app.entrypoint(build_agent)

        assert app._builder_fn is build_agent
        assert decorated is build_agent  # Should return original function

    def test_prepare_agent_input_decorator_stores_fn(self):
        """@app.prepare_agent_input stores the hook and returns the original fn."""
        app = App(app_name="Test Agent")

        def prepare(input, ctx):
            return {"messages": []}

        decorated = app.prepare_agent_input(prepare)

        assert app._prepare_input_fn is prepare
        assert decorated is prepare

    def test_get_agent_passes_prepare_input_to_adapter(self):
        """get_agent wires the registered prepare_input hook into the adapter."""
        app = App(app_name="Test Agent")
        app._builder_fn = lambda: MagicMock()

        def prepare(input, ctx):
            return {"messages": []}

        app.prepare_agent_input(prepare)
        agent = app.get_agent()

        assert agent._prepare_input is prepare

    def test_get_agent_without_prepare_input_leaves_hook_none(self):
        """When no hook is registered, the adapter's prepare_input is None."""
        app = App(app_name="Test Agent")
        app._builder_fn = lambda: MagicMock()

        agent = app.get_agent()

        assert agent._prepare_input is None

    def test_resolve_thread_id_decorator_stores_fn(self):
        """@app.resolve_thread_id stores the hook and returns the original fn."""
        app = App(app_name="Test Agent")

        def resolve(ctx):
            return f"{ctx.session_id}__actor"

        decorated = app.resolve_thread_id(resolve)

        assert app._resolve_thread_id_fn is resolve
        assert decorated is resolve

    def test_get_agent_passes_resolve_thread_id_to_adapter(self):
        """get_agent wires the registered resolve_thread_id hook into the adapter."""
        app = App(app_name="Test Agent")
        app._builder_fn = lambda: MagicMock()

        def resolve(ctx):
            return f"{ctx.session_id}__actor"

        app.resolve_thread_id(resolve)
        agent = app.get_agent()

        assert agent._resolve_thread_id is resolve

    def test_get_agent_without_resolve_thread_id_leaves_hook_none(self):
        """When no hook is registered, the adapter's resolve_thread_id is None."""
        app = App(app_name="Test Agent")
        app._builder_fn = lambda: MagicMock()

        agent = app.get_agent()

        assert agent._resolve_thread_id is None

    def test_output_parser_decorator_stores_cls(self):
        """@app.output_parser stores the subclass and returns it unchanged."""
        app = App(app_name="Test Agent")

        @app.output_parser
        class Parser(LangGraphOutputParser):
            async def parse(self, item, ctx):
                return
                yield  # pragma: no cover - unreachable; makes this an async generator

            async def on_stream_error(self, ctx, error):
                return None

        assert app._output_parser_cls is Parser

    def test_output_parser_decorator_rejects_non_parser(self):
        """Registering a non-OutputParser fails at decoration time."""
        app = App(app_name="Test Agent")

        class NotAParser:
            pass

        # Deliberately wrong type: assert the runtime issubclass guard fires.
        with pytest.raises(TypeError):
            app.output_parser(NotAParser)  # pyright: ignore[reportArgumentType]

    def test_get_agent_threads_output_parser_to_adapter(self):
        """get_agent wires the registered parser class into the adapter."""
        app = App(app_name="Test Agent")
        app._builder_fn = lambda: MagicMock()

        @app.output_parser
        class Parser(LangGraphOutputParser):
            async def parse(self, item, ctx):
                return
                yield  # pragma: no cover - unreachable; makes this an async generator

            async def on_stream_error(self, ctx, error):
                return None

        assert app.get_agent()._output_parser is Parser

    def test_get_agent_without_output_parser_leaves_none(self):
        """When no parser is registered, the adapter's output_parser is None."""
        app = App(app_name="Test Agent")
        app._builder_fn = lambda: MagicMock()

        assert app.get_agent()._output_parser is None

    @staticmethod
    def _set_use_custom_parser(app: App, value: bool) -> None:
        """Override the agent.yaml opt-in the runtime loaded at construction."""
        from agent_engine_runner_shared.agent_config import (
            AgentFeatureConfig,
            RuntimeAgentConfig,
        )

        app._runtime._agent_config = RuntimeAgentConfig(
            features=AgentFeatureConfig(use_custom_parser=value)
        )

    def test_get_agent_activates_parser_when_flag_true(self):
        """use_custom_parser=true arms the adapter to run the registered parser."""
        app = App(app_name="Test Agent")
        app._builder_fn = lambda: MagicMock()
        self._set_use_custom_parser(app, True)

        @app.output_parser
        class Parser(LangGraphOutputParser):
            async def parse(self, item, ctx):
                return
                yield  # pragma: no cover - unreachable; makes this an async generator

            async def on_stream_error(self, ctx, error):
                return None

        agent = app.get_agent()
        assert agent._output_parser is Parser
        assert agent._use_custom_parser is True

    def test_get_agent_parser_inert_when_flag_off(self):
        """A registered parser stays threaded but inert when the flag is off."""
        app = App(app_name="Test Agent")
        app._builder_fn = lambda: MagicMock()
        self._set_use_custom_parser(app, False)

        @app.output_parser
        class Parser(LangGraphOutputParser):
            async def parse(self, item, ctx):
                return
                yield  # pragma: no cover - unreachable; makes this an async generator

            async def on_stream_error(self, ctx, error):
                return None

        agent = app.get_agent()
        assert agent._output_parser is Parser
        assert agent._use_custom_parser is False

    def test_get_agent_raises_when_flag_true_without_parser(self):
        """Opting in without a registered parser fails fast at agent build."""
        app = App(app_name="Test Agent")
        app._builder_fn = lambda: MagicMock()
        self._set_use_custom_parser(app, True)

        with pytest.raises(RuntimeError, match="use_custom_parser"):
            app.get_agent()

    def test_langgraph_output_parser_rejects_bare_str_stream_modes(self):
        """A bare-str stream_modes char-splits under list(); reject at definition."""
        with pytest.raises(TypeError):

            class BadModes(LangGraphOutputParser):
                stream_modes = "messages"

                async def parse(self, item, ctx):
                    return
                    yield  # pragma: no cover - unreachable

                async def on_stream_error(self, ctx, error):
                    return None

    def test_tool_decorator_registers_on_runtime(self):
        """@app.tool() calls runtime.register_tool() with raw function and metadata."""
        app = App(app_name="Test Agent")
        app._runtime.register_tool = MagicMock()

        @app.tool(is_local=False, timeout=60)
        def my_tool(query: str) -> str:
            """Search for stuff."""
            return query

        app._runtime.register_tool.assert_called_once()
        call_kwargs = app._runtime.register_tool.call_args
        assert call_kwargs[1]["name"] == "my_tool"
        assert call_kwargs[1]["func"] is my_tool
        assert call_kwargs[1]["metadata"]["is_local"] is False
        assert call_kwargs[1]["metadata"]["timeout_seconds"] == 60

    def test_tool_decorator_creates_lc_tool(self):
        """@app.tool() creates a LangChain tool in App._lc_tools."""
        app = App(app_name="Test Agent")

        @app.tool()
        def my_tool(query: str) -> str:
            """Search for stuff."""
            return query

        assert "my_tool" in app._lc_tools
        assert app._lc_tools["my_tool"].name == "my_tool"

    def test_tool_decorator_captures_tool_definition(self):
        """@app.tool() populates get_tool_definitions()."""
        app = App(app_name="Test Agent")

        @app.tool(is_local=False, timeout=60)
        def my_tool(query: str) -> str:
            """Search for stuff."""
            return query

        defs = app.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0].name == "my_tool"
        assert defs[0].description == "Search for stuff."
        assert defs[0].remote is True
        assert defs[0].timeout_seconds == 60

    def test_tool_decorator_derives_args_schema(self):
        """@app.tool() derives a non-empty args_schema from the signature."""
        app = App(app_name="Test Agent")

        @app.tool()
        def my_tool(query: str, limit: int = 10) -> str:
            """Search for stuff."""
            return query

        schema = app.get_tool_definitions()[0].args_schema
        assert schema.get("type") == "object"
        assert set(schema.get("properties", {})) == {"query", "limit"}
        assert schema["properties"]["query"]["type"] == "string"
        assert schema["properties"]["limit"]["type"] == "integer"
        assert schema.get("required") == ["query"]

    def test_tool_decorator_no_params_registers_object_schema(self):
        """A parameterless tool still registers cleanly with an object schema."""
        app = App(app_name="Test Agent")

        @app.tool()
        def ping() -> str:
            """No args."""
            return "pong"

        schema = app.get_tool_definitions()[0].args_schema
        assert schema.get("type") == "object"
        assert schema.get("properties", {}) == {}

    def test_tool_decorator_schema_derivation_failure_falls_back(self):
        """A schema-derivation error degrades to {} instead of crashing import."""
        from agent_engine_sdk_langgraph import runtime as runtime_module

        class _Boom:
            name = "boom_tool"

            @property
            def args_schema(self):
                raise RuntimeError("cannot build schema")

        assert runtime_module._derive_args_schema(_Boom()) == {}

    def test_tools_returns_lc_tools(self):
        """App.tools() returns LangChain tools from App._lc_tools."""
        app = App(app_name="Test Agent")

        @app.tool()
        def my_tool(query: str) -> str:
            """Search."""
            return query

        result = app.tools()
        assert len(result) == 1
        assert result[0].name == "my_tool"

    @patch("agent_engine_runner_shared.secure_wrapper.create_secure_tool_function")
    def test_get_tools_wraps_in_aer_mode(self, mock_create_secure):
        """get_tools() wraps tools with SecureToolWrapper in AER mode."""
        from agent_engine_runner_shared import RuntimeMode

        app = App(app_name="Test Agent")
        app._runtime.mode = RuntimeMode.AER

        @app.tool(is_local=True)
        def local_tool(x: str) -> str:
            """A local tool."""
            return x

        @app.tool(is_local=False)
        def remote_tool(x: str) -> str:
            """A remote tool."""
            return x

        mock_create_secure.return_value = lambda **kwargs: "wrapped"

        result = app.get_tools()

        assert len(result) == 2
        assert mock_create_secure.call_count == 2
        assert all(tool.handle_tool_error is True for tool in result)

        calls = {call.kwargs["tool_name"] for call in mock_create_secure.call_args_list}
        assert calls == {"local_tool", "remote_tool"}

    @patch("agent_engine_runner_shared.secure_wrapper.create_secure_tool_function")
    def test_get_tools_forwards_redact_fields(self, mock_create_secure):
        from agent_engine_runner_shared import RuntimeMode

        app = App(app_name="Test Agent")
        app._runtime.mode = RuntimeMode.AER

        @app.tool(redact_fields=["card_number"])
        def charge_customer(card_number: str) -> str:
            """Charge a customer."""
            return "ok"

        mock_create_secure.return_value = lambda **kwargs: "wrapped"

        app.get_tools()

        mock_create_secure.assert_called_once()
        assert mock_create_secure.call_args.kwargs["redact_fields"] == ["card_number"]

    def test_get_tools_injects_tool_call_id_to_oe(self):
        """A ToolCall id reaches SecureToolWrapper.execute_tool as tool_call_id.

        End-to-end through real create_secure_tool_function + LangChain's
        InjectedToolCallId path: ToolNode invokes the wrapped tool with a full
        ToolCall (carrying the LLM id), which the wrapper forwards to OE for the
        execution-log join key. The id must stay out of the
        model-facing schema and out of the tool arguments.
        """
        from agent_engine_runner_shared import RuntimeMode

        app = App(app_name="Test Agent")
        app._runtime.mode = RuntimeMode.AER

        @app.tool(is_local=True)
        def get_weather(city: str) -> str:
            """Get the weather."""
            return f"weather in {city}"

        wrapped = app.get_tools()[0]

        # Injected id is hidden from the model-facing schema; real args remain.
        model_fields = wrapped.tool_call_schema.model_fields
        assert "tool_call_id" not in model_fields
        assert "city" in model_fields

        mock_wrapper = MagicMock()
        mock_wrapper.execute_tool.return_value = "weather in Tokyo"
        tokens = set_execution_context(
            execution_id="exec-123",
            wrapper=mock_wrapper,
            oe_url="http://localhost:8080",
        )
        try:
            wrapped.invoke(
                {
                    "name": "get_weather",
                    "args": {"city": "Tokyo"},
                    "id": "call_xyz",
                    "type": "tool_call",
                }
            )
        finally:
            clear_execution_context(tokens)

        mock_wrapper.execute_tool.assert_called_once()
        call_kwargs = mock_wrapper.execute_tool.call_args.kwargs
        assert call_kwargs["tool_call_id"] == "call_xyz"
        assert call_kwargs["arguments"] == {"city": "Tokyo"}

    def test_get_tools_assigns_durable_tool_result_identity(self):
        """Platform tool results keep stable, step-scoped IDs across replay."""
        from langchain_core.messages import ToolMessage

        from agent_engine_runner_shared import RuntimeMode
        from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
            WorkflowIdentity,
        )
        from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import (
            AttemptContext,
        )
        from agent_engine_runner_shared.workflow.context import (
            advance_step_ordinal,
            attempt_context_scope,
        )

        app = App(app_name="Test Agent")
        app._runtime.mode = RuntimeMode.AER

        @app.tool(is_local=True, response_format="content_and_artifact")
        def get_sections(ticker: str) -> tuple[str, dict]:
            """Fetch sections."""
            return f"summary {ticker}", {"ticker": ticker}

        wrapped = app.get_tools()[0]
        tool_call = {
            "name": "get_sections",
            "args": {"ticker": "AAPL"},
            "id": "call_1",
            "type": "tool_call",
        }
        mock_wrapper = MagicMock()
        mock_wrapper.execute_tool.return_value = [
            "summary AAPL",
            {"ticker": "AAPL"},
        ]
        tokens = set_execution_context(
            execution_id="exec-1",
            wrapper=mock_wrapper,
            oe_url="http://localhost:8080",
        )
        try:
            with attempt_context_scope(
                AttemptContext(
                    attempt_id="attempt-1",
                    workflow_identity=WorkflowIdentity(execution_id="exec-1"),
                )
            ):
                first = wrapped.invoke(tool_call)
                advance_step_ordinal(1)
                repeated = wrapped.invoke(tool_call)

            with attempt_context_scope(
                AttemptContext(
                    attempt_id="attempt-2",
                    replay_mode=True,
                    workflow_identity=WorkflowIdentity(execution_id="exec-1"),
                )
            ):
                replay = wrapped.invoke(tool_call)
        finally:
            clear_execution_context(tokens)

        assert isinstance(first, ToolMessage)
        assert isinstance(repeated, ToolMessage)
        assert isinstance(replay, ToolMessage)
        assert first.id == "durable-tool-result:exec-1:1:call_1"
        assert replay.id == first.id
        assert repeated.id == "durable-tool-result:exec-1:2:call_1"
        assert first.content == "summary AAPL"
        assert first.artifact == {"ticker": "AAPL"}

    def test_tool_decorator_defaults_response_format_to_content(self):
        """@app.tool() without response_format keeps LangChain's default."""
        app = App(app_name="Test Agent")

        @app.tool()
        def my_tool(query: str) -> str:
            """Search."""
            return query

        assert app._lc_tools["my_tool"].response_format == "content"

    def test_tool_decorator_forwards_response_format(self):
        """@app.tool(response_format=...) forwards to the LangChain tool."""
        app = App(app_name="Test Agent")

        @app.tool(response_format="content_and_artifact")
        def my_tool(query: str) -> tuple[str, dict]:
            """Search."""
            return query, {"raw": query}

        assert app._lc_tools["my_tool"].response_format == "content_and_artifact"

    def test_get_tools_preserves_response_format(self):
        """AER re-wrap keeps response_format so ToolNode splits the tuple."""
        from agent_engine_runner_shared import RuntimeMode

        app = App(app_name="Test Agent")
        app._runtime.mode = RuntimeMode.AER

        @app.tool(is_local=True, response_format="content_and_artifact")
        def my_tool(query: str) -> tuple[str, dict]:
            """Search."""
            return query, {"raw": query}

        assert app.get_tools()[0].response_format == "content_and_artifact"

    def test_get_tools_content_and_artifact_reconstructs_tuple(self):
        """OE returns the tuple as a JSON list; ToolNode still sees content+artifact.

        The OE round trip serializes the tool's (content, artifact) tuple to a
        JSON list. The wrapper coerces it back to a tuple so LangChain splits it
        into ToolMessage.content and ToolMessage.artifact.
        """
        from agent_engine_runner_shared import RuntimeMode

        app = App(app_name="Test Agent")
        app._runtime.mode = RuntimeMode.AER

        @app.tool(is_local=True, response_format="content_and_artifact")
        def get_sections(ticker: str) -> tuple[str, dict]:
            """Fetch sections."""
            return f"summary {ticker}", {"ticker": ticker}

        wrapped = app.get_tools()[0]

        mock_wrapper = MagicMock()
        # OE serializes the tuple to a JSON list on the way back.
        mock_wrapper.execute_tool.return_value = ["summary AAPL", {"ticker": "AAPL"}]
        tokens = set_execution_context(
            execution_id="exec-1",
            wrapper=mock_wrapper,
            oe_url="http://localhost:8080",
        )
        try:
            message = wrapped.invoke(
                {
                    "name": "get_sections",
                    "args": {"ticker": "AAPL"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            )
        finally:
            clear_execution_context(tokens)

        assert message.content == "summary AAPL"
        assert message.artifact == {"ticker": "AAPL"}
        assert message.id is None

    def test_get_tools_content_and_artifact_non_pair_does_not_crash(self):
        """An HITL resume returns a bare string, not a (content, artifact) pair.

        A content_and_artifact tool whose result is not a 2-element sequence (the
        HITL resume path returns the human-decision JSON string) must still yield a
        ToolMessage rather than the ValueError ToolNode raises on a non-tuple —
        which handle_tool_error does not catch.
        """
        from agent_engine_runner_shared import RuntimeMode

        app = App(app_name="Test Agent")
        app._runtime.mode = RuntimeMode.AER

        @app.tool(is_local=True, response_format="content_and_artifact")
        def approve(amount: int) -> tuple[str, dict]:
            """Approve an amount."""
            return "approved", {"amount": amount}

        wrapped = app.get_tools()[0]

        mock_wrapper = MagicMock()
        # HITL resume: execute_tool returns the human-decision JSON string.
        mock_wrapper.execute_tool.return_value = '{"approved": true}'
        tokens = set_execution_context(
            execution_id="exec-1",
            wrapper=mock_wrapper,
            oe_url="http://localhost:8080",
        )
        try:
            message = wrapped.invoke(
                {
                    "name": "approve",
                    "args": {"amount": 100},
                    "id": "call_1",
                    "type": "tool_call",
                }
            )
        finally:
            clear_execution_context(tokens)

        assert message.status != "error"
        assert message.content == '{"approved": true}'
        assert message.artifact is None

    def test_get_tools_injects_tool_call_id_for_dict_schema_mcp_tool(self):
        """MCP tools register a raw JSON-schema dict args_schema; the id must still
        be injected and forwarded for them (LangChain skips injection on dict
        schemas, so the wrapper must wrap them in a model first).
        """
        from langchain_core.tools import StructuredTool

        from agent_engine_runner_shared import RuntimeMode

        app = App(app_name="Test Agent")
        app._runtime.mode = RuntimeMode.AER

        # Mirror _register_mcp_tool: a StructuredTool whose args_schema is a raw
        # JSON-schema dict rather than a pydantic model.
        app._lc_tools["mcp_tool"] = StructuredTool.from_function(
            func=lambda **kwargs: "raw",
            name="mcp_tool",
            description="An MCP tool.",
            args_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )

        wrapped = app.get_tools()[0]
        # Injected id is hidden from the model-facing schema.
        assert "tool_call_id" not in wrapped.tool_call_schema.model_fields

        mock_wrapper = MagicMock()
        mock_wrapper.execute_tool.return_value = "ok"
        tokens = set_execution_context(
            execution_id="exec-123",
            wrapper=mock_wrapper,
            oe_url="http://localhost:8080",
        )
        try:
            wrapped.invoke(
                {
                    "name": "mcp_tool",
                    "args": {"city": "Tokyo"},
                    "id": "call_mcp1",
                    "type": "tool_call",
                }
            )
        finally:
            clear_execution_context(tokens)

        mock_wrapper.execute_tool.assert_called_once()
        call_kwargs = mock_wrapper.execute_tool.call_args.kwargs
        assert call_kwargs["tool_call_id"] == "call_mcp1"
        assert call_kwargs["arguments"] == {"city": "Tokyo"}

    def test_checkpointer_wraps_mongodb_saver_in_aer_mode(self):
        """App.checkpointer() returns a PlatformCheckpointer over the Mongo saver."""
        from agent_engine_sdk_langgraph.platform_checkpointer import (
            PlatformCheckpointer,
        )
        from agent_engine_runner_shared import RuntimeMode

        app = App(app_name="Test Agent")
        mock_runtime = MagicMock()
        mock_runtime.mode = RuntimeMode.AER
        mock_runtime.mongodb_uri = "mongodb://localhost:27017"
        app._runtime = mock_runtime

        mock_saver = MagicMock()
        with (
            patch("pymongo.MongoClient", return_value=MagicMock()),
            patch(
                "langgraph.checkpoint.mongodb.MongoDBSaver",
                return_value=mock_saver,
            ),
        ):
            result = app.checkpointer()

        assert isinstance(result, PlatformCheckpointer)
        assert result.native is mock_saver

    def test_checkpointer_uses_per_project_scoped_db(self):
        """checkpointer() resolves the per-project store DB for MongoDBSaver."""
        import os

        from agent_engine_runner_shared import RuntimeMode, db_config

        app = App(app_name="Test Agent")
        mock_runtime = MagicMock()
        mock_runtime.mode = RuntimeMode.AER
        mock_runtime.mongodb_uri = "mongodb://localhost:27017"
        app._runtime = mock_runtime

        fake_client = MagicMock()
        fake_client.list_database_names.return_value = ["mdb_agentic_store_proj1"]
        captured = {}

        def _capture_saver(client, db_name):  # noqa: ANN001
            captured["db_name"] = db_name
            return MagicMock()

        prev = os.environ.get("PROJECT_ID")
        prev_ckpt = os.environ.get("CHECKPOINT_DB_NAME")
        os.environ["PROJECT_ID"] = "proj1"
        os.environ.pop("CHECKPOINT_DB_NAME", None)
        db_config.reset_store_db_cache()
        try:
            with (
                patch("pymongo.MongoClient", return_value=fake_client),
                patch(
                    "langgraph.checkpoint.mongodb.MongoDBSaver",
                    side_effect=_capture_saver,
                ),
            ):
                app.checkpointer()
        finally:
            db_config.reset_store_db_cache()
            if prev is None:
                os.environ.pop("PROJECT_ID", None)
            else:
                os.environ["PROJECT_ID"] = prev
            if prev_ckpt is None:
                os.environ.pop("CHECKPOINT_DB_NAME", None)
            else:
                os.environ["CHECKPOINT_DB_NAME"] = prev_ckpt

        assert captured["db_name"] == "mdb_agentic_store_proj1"

    def test_checkpointer_uses_exact_checkpoint_db_name_override(self):
        """CHECKPOINT_DB_NAME is used verbatim with no project scoping."""
        import os

        from agent_engine_runner_shared import RuntimeMode, db_config

        app = App(app_name="Test Agent")
        mock_runtime = MagicMock()
        mock_runtime.mode = RuntimeMode.AER
        mock_runtime.mongodb_uri = "mongodb://localhost:27017"
        app._runtime = mock_runtime

        fake_client = MagicMock()
        captured = {}

        def _capture_saver(client, db_name):  # noqa: ANN001
            captured["db_name"] = db_name
            return MagicMock()

        prev_project = os.environ.get("PROJECT_ID")
        prev_ckpt = os.environ.get("CHECKPOINT_DB_NAME")
        prev_store = os.environ.get("MDB_AGENTIC_STORE_DB")
        os.environ["PROJECT_ID"] = "proj1"
        os.environ["CHECKPOINT_DB_NAME"] = "external_checkpoints_agent"
        os.environ.pop("MDB_AGENTIC_STORE_DB", None)
        db_config.reset_store_db_cache()
        try:
            with (
                patch("pymongo.MongoClient", return_value=fake_client),
                patch(
                    "langgraph.checkpoint.mongodb.MongoDBSaver",
                    side_effect=_capture_saver,
                ),
            ):
                app.checkpointer()
        finally:
            db_config.reset_store_db_cache()
            if prev_project is None:
                os.environ.pop("PROJECT_ID", None)
            else:
                os.environ["PROJECT_ID"] = prev_project
            if prev_ckpt is None:
                os.environ.pop("CHECKPOINT_DB_NAME", None)
            else:
                os.environ["CHECKPOINT_DB_NAME"] = prev_ckpt
            if prev_store is None:
                os.environ.pop("MDB_AGENTIC_STORE_DB", None)
            else:
                os.environ["MDB_AGENTIC_STORE_DB"] = prev_store

        assert captured["db_name"] == "external_checkpoints_agent"
        fake_client.list_database_names.assert_not_called()

    # ------------------------------------------------------------------
    # App.run() — LangGraphQueryPlugin registration
    #
    # The registration block is the integration point that makes
    # /query/sessions* work in production. We exercise the three branches:
    # happy registration, checkpointer() raises (swallow + warning), and
    # non-AER mode (registration entirely skipped).
    # ------------------------------------------------------------------

    def test_run_registers_query_plugin_in_aer_mode_when_checkpointer_available(self):
        """AER mode + working checkpointer → LangGraphQueryPlugin registered."""
        from agent_engine_sdk_langgraph.query import LangGraphQueryPlugin
        from agent_engine_runner_shared import RuntimeMode

        app = App(app_name="Test Agent")
        app._builder_fn = lambda: MagicMock()  # bypass the no-entrypoint guard

        mock_runtime = MagicMock()
        mock_runtime.mode = RuntimeMode.AER
        app._runtime = mock_runtime

        mock_saver = MagicMock()
        with (
            patch.object(App, "_register_hooks"),
            patch("agent_engine_runner_shared.tracing.setup._run_instrumentor"),
            patch.object(App, "checkpointer", return_value=mock_saver),
        ):
            app.run()

        mock_runtime.register_query_plugin.assert_called_once()
        registered_plugin = mock_runtime.register_query_plugin.call_args[0][0]
        assert isinstance(registered_plugin, LangGraphQueryPlugin)
        mock_runtime.register_and_run.assert_called_once()

    def test_run_skips_query_plugin_when_checkpointer_returns_none(self):
        """No checkpointer (e.g. MONGODB_URI unset) → registration skipped,
        /query/sessions* will return 501."""
        from agent_engine_runner_shared import RuntimeMode

        app = App(app_name="Test Agent")
        app._builder_fn = lambda: MagicMock()

        mock_runtime = MagicMock()
        mock_runtime.mode = RuntimeMode.AER
        app._runtime = mock_runtime

        with (
            patch.object(App, "_register_hooks"),
            patch("agent_engine_runner_shared.tracing.setup._run_instrumentor"),
            patch.object(App, "checkpointer", return_value=None),
        ):
            app.run()

        mock_runtime.register_query_plugin.assert_not_called()
        mock_runtime.register_and_run.assert_called_once()

    def test_run_swallows_checkpointer_construction_failure(self):
        """Checkpointer construction raising (bad URI, auth, etc.) must not
        take down App.run() — the rest of the AER still starts; query
        routes degrade to 501."""
        from agent_engine_runner_shared import RuntimeMode

        app = App(app_name="Test Agent")
        app._builder_fn = lambda: MagicMock()

        mock_runtime = MagicMock()
        mock_runtime.mode = RuntimeMode.AER
        app._runtime = mock_runtime

        with (
            patch.object(App, "_register_hooks"),
            patch("agent_engine_runner_shared.tracing.setup._run_instrumentor"),
            patch.object(
                App,
                "checkpointer",
                side_effect=RuntimeError("bad MONGODB_URI"),
            ),
        ):
            app.run()  # must not raise

        mock_runtime.register_query_plugin.assert_not_called()
        mock_runtime.register_and_run.assert_called_once()

    def test_run_skips_query_plugin_registration_outside_aer_mode(self):
        """In TOOL or memory-server mode the plugin is irrelevant; the
        registration block must not call checkpointer() at all."""
        from agent_engine_runner_shared import RuntimeMode

        app = App(app_name="Test Agent")
        app._builder_fn = lambda: MagicMock()

        mock_runtime = MagicMock()
        mock_runtime.mode = RuntimeMode.TOOL
        app._runtime = mock_runtime

        with (
            patch.object(App, "_register_hooks"),
            patch("agent_engine_runner_shared.tracing.setup._run_instrumentor"),
            patch.object(App, "checkpointer") as mock_checkpointer,
        ):
            app.run()

        mock_checkpointer.assert_not_called()
        mock_runtime.register_query_plugin.assert_not_called()
        mock_runtime.register_and_run.assert_called_once()

    def test_get_tool_schemas_returns_lc_tools(self):
        """App.get_tool_schemas() returns original LangChain tools."""
        app = App(app_name="Test Agent")

        @app.tool()
        def my_tool(query: str) -> str:
            """Search."""
            return query

        result = app.get_tool_schemas()
        assert len(result) == 1
        assert result[0].name == "my_tool"

    def test_validate_llm_response_returns_response_unchanged(self):
        """Returns response unchanged when response has no string content."""
        app = App(app_name="Test Agent")
        mock_response = MagicMock()

        result = app.validate_llm_response(mock_response)

        assert result is mock_response

    def test_validate_llm_response_passthrough_when_no_content(self):
        """Returns response unchanged when response has no string content."""
        app = App(app_name="Test Agent")
        mock_response = MagicMock(spec=[])  # no content attr

        result = app.validate_llm_response(mock_response)

        assert result is mock_response

    def test_validate_llm_response_passthrough_when_has_tool_calls(self):
        """Returns response unchanged when response has tool_calls."""
        app = App(app_name="Test Agent")
        mock_response = MagicMock()
        mock_response.content = "Some content"
        mock_response.tool_calls = [{"name": "my_tool", "args": {}}]

        result = app.validate_llm_response(mock_response)

        assert result is mock_response

    def test_validate_llm_response_returns_unchanged_when_content_same(self):
        """Returns original response when validate_output makes no changes."""
        app = App(app_name="Test Agent")
        app._runtime.validate_output = MagicMock(return_value="Hello world")
        mock_response = MagicMock()
        mock_response.content = "Hello world"
        mock_response.tool_calls = []

        result = app.validate_llm_response(mock_response)

        assert result is mock_response

    def test_validate_llm_response_returns_new_message_when_modified(self):
        """Returns new AIMessage when validate_output modifies content."""
        app = App(app_name="Test Agent")
        app._runtime.validate_output = MagicMock(return_value="[REDACTED]")
        mock_response = MagicMock()
        mock_response.content = "sensitive data"
        mock_response.tool_calls = []
        mock_response.response_metadata = {"model": "test"}

        result = app.validate_llm_response(mock_response)

        assert result is not mock_response
        assert result.content == "[REDACTED]"

    def test_validate_output_delegates_to_runtime(self):
        """App.validate_output() delegates to TenantRuntime."""
        app = App(app_name="Test Agent")
        app._runtime.validate_output = MagicMock(return_value="validated_text")

        result = app.validate_output("test text")

        app._runtime.validate_output.assert_called_once_with("test text")
        assert result == "validated_text"

    def test_get_current_user_id_delegates_to_runtime(self):
        """App.get_current_user_id() delegates to TenantRuntime."""
        app = App(app_name="Test Agent")
        app._runtime.get_current_user_id = MagicMock(return_value="user_123")

        result = app.get_current_user_id()

        app._runtime.get_current_user_id.assert_called_once()
        assert result == "user_123"


class TestAppCheckpointer:
    """Tests for App.checkpointer() branches."""

    def test_get_agent_registers_workflow_adapter_only_for_platform_checkpointer(
        self, monkeypatch
    ):
        """Durable eligibility follows the materialized graph's checkpointer."""
        import agent_engine_runner_shared.hooks as hooks
        from agent_engine_sdk_langgraph.platform_checkpointer import (
            PlatformCheckpointer,
        )

        # Platform-checkpointed graph: registers the adapter.
        monkeypatch.setattr(hooks, "_workflow_adapter", None)
        app = App(app_name="Durable Agent")
        graph = MagicMock()
        graph.checkpointer = PlatformCheckpointer(native=None)
        app._builder_fn = lambda: graph
        agent = app.get_agent()
        assert hooks.get_workflow_adapter() is not None
        # The graph remains durable, but an empty compiled-child topology must
        # retain the existing root operation path (including Deep Agent graphs).
        assert agent._durable_subgraphs is None

        # Custom-saver graph: stays native (no registration) — and clears a
        # previously registered adapter so eligibility follows the latest
        # materialized graph.
        app2 = App(app_name="Custom Saver Agent")
        graph2 = MagicMock()
        graph2.checkpointer = MagicMock()  # arbitrary BaseCheckpointSaver
        app2._builder_fn = lambda: graph2
        agent2 = app2.get_agent()
        assert hooks.get_workflow_adapter() is None
        assert agent2._durable_subgraphs is None

    def test_checkpointer_returns_none_when_aer_mode_but_no_uri(self):
        """Without a Mongo URI agents keep compiling a stateless graph."""
        from agent_engine_runner_shared import RuntimeMode

        app = App(app_name="Test Agent")
        mock_runtime = MagicMock()
        mock_runtime.mode = RuntimeMode.AER
        mock_runtime.mongodb_uri = None
        app._runtime = mock_runtime

        assert app.checkpointer() is None

    def test_checkpointer_returns_none_outside_aer_mode(self):
        """App.checkpointer() returns None and logs a warning in non-AER modes."""
        from agent_engine_runner_shared import RuntimeMode

        for mode in [RuntimeMode.TOOL]:
            app = App(app_name="Test Agent")
            mock_runtime = MagicMock()
            mock_runtime.mode = mode
            app._runtime = mock_runtime

            result = app.checkpointer()

            assert result is None

    def test_checkpointer_is_cached_on_second_call(self):
        """App.checkpointer() returns the same object on subsequent calls."""
        from agent_engine_runner_shared import RuntimeMode

        app = App(app_name="Test Agent")
        mock_runtime = MagicMock()
        mock_runtime.mode = RuntimeMode.AER
        mock_runtime.mongodb_uri = "mongodb://localhost:27017"
        app._runtime = mock_runtime

        mock_saver = MagicMock()
        with (
            patch("pymongo.MongoClient", return_value=MagicMock()),
            patch("langgraph.checkpoint.mongodb.MongoDBSaver", return_value=mock_saver),
        ):
            first = app.checkpointer()
            second = app.checkpointer()

        assert first is second

    def test_checkpointer_uses_env_configured_mongo_timeouts(self, monkeypatch):
        """MongoDB checkpointer timeouts are operator-tunable in seconds."""
        from agent_engine_runner_shared import RuntimeMode

        monkeypatch.setenv("CHECKPOINTER_SERVER_SELECTION_TIMEOUT", "2.5")
        monkeypatch.setenv("CHECKPOINTER_CONNECT_TIMEOUT", "3")
        monkeypatch.setenv("CHECKPOINTER_SOCKET_TIMEOUT", "4.25")

        app = App(app_name="Test Agent")
        mock_runtime = MagicMock()
        mock_runtime.mode = RuntimeMode.AER
        mock_runtime.mongodb_uri = "mongodb://localhost:27017"
        app._runtime = mock_runtime

        mock_saver = MagicMock()
        with (
            patch("pymongo.MongoClient", return_value=MagicMock()) as mock_client,
            patch("langgraph.checkpoint.mongodb.MongoDBSaver", return_value=mock_saver),
        ):
            result = app.checkpointer()

        assert result is not None
        assert result.native is mock_saver
        mock_client.assert_called_once_with(
            "mongodb://localhost:27017",
            serverSelectionTimeoutMS=2500,
            connectTimeoutMS=3000,
            socketTimeoutMS=4250,
        )

    def test_close_releases_mongo_client(self):
        """App.close() calls close() on the cached MongoClient."""
        from agent_engine_runner_shared import RuntimeMode

        app = App(app_name="Test Agent")
        mock_runtime = MagicMock()
        mock_runtime.mode = RuntimeMode.AER
        mock_runtime.mongodb_uri = "mongodb://localhost:27017"
        app._runtime = mock_runtime

        mock_client = MagicMock()
        with (
            patch("pymongo.MongoClient", return_value=mock_client),
            patch(
                "langgraph.checkpoint.mongodb.MongoDBSaver", return_value=MagicMock()
            ),
        ):
            app.checkpointer()

        app.close()

        mock_client.close.assert_called_once()
        assert app._mongo_client is None
        assert app._checkpointer is None

    def test_checkpointer_closes_client_when_saver_construction_fails(self):
        """If MongoDBSaver(...) raises, the freshly-opened MongoClient must be
        closed before the exception propagates — otherwise every failed
        init would leak a pool connection. The runtime.py try/except/raise
        is there specifically to prevent that leak; keep a test locked on
        it so regressions don't slip past review.
        """
        from agent_engine_runner_shared import RuntimeMode

        app = App(app_name="Test Agent")
        mock_runtime = MagicMock()
        mock_runtime.mode = RuntimeMode.AER
        mock_runtime.mongodb_uri = "mongodb://localhost:27017"
        app._runtime = mock_runtime

        mock_client = MagicMock()
        with (
            patch("pymongo.MongoClient", return_value=mock_client),
            patch(
                "langgraph.checkpoint.mongodb.MongoDBSaver",
                side_effect=RuntimeError("boom"),
            ),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                app.checkpointer()

        mock_client.close.assert_called_once()
        # The freshly-opened client must NOT be cached on the App when
        # construction failed — otherwise the next call would return None
        # for the checkpointer but hold a dead client reference.
        assert app._mongo_client is None
        assert app._checkpointer is None


class TestAppSuspend:
    """Tests for App.suspend()."""

    def test_suspend_returns_json_string(self):
        """App.suspend() returns a JSON string from SuspendPayload."""
        app = App(app_name="Test Agent")

        mock_payload = MagicMock()
        mock_payload.to_json.return_value = '{"suspend_reason": "needs_approval"}'

        with patch(
            "agent_engine_sdk_langgraph.runtime.SuspendPayload",
            return_value=mock_payload,
        ) as mock_cls:
            result = app.suspend(
                reason="needs_approval", context={"ticket": "TICKET-123"}
            )

        mock_cls.assert_called_once_with(
            suspend_reason="needs_approval",
            suspend_context={"ticket": "TICKET-123"},
        )
        mock_payload.to_json.assert_called_once()
        assert result == '{"suspend_reason": "needs_approval"}'

    def test_supported_tool_wait_becomes_durable_activity_suspension(self):
        from agent_engine_sdk_langgraph.runtime import _suspend_durable_activity
        from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import (
            AttemptContext,
        )
        from agent_engine_runner_shared.workflow import (
            DurableActivitySuspended,
            attempt_context_scope,
        )

        with (
            attempt_context_scope(AttemptContext(attempt_id="attempt-1")),
            pytest.raises(DurableActivitySuspended) as raised,
        ):
            _suspend_durable_activity(
                {
                    "suspend_reason": "awaiting_human_review",
                    "suspend_context": {
                        "allowed_decisions": ["approve", "reject"],
                        "claim_id": "claim-1",
                    },
                }
            )

        assert raised.value.reason == "awaiting_human_review"
        assert raised.value.context == {
            "allowed_decisions": ["approve", "reject"],
            "claim_id": "claim-1",
        }

    def test_native_suspend_is_available_in_a_durable_attempt(self):
        from agent_engine_sdk_langgraph.runtime import _suspend
        from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import (
            AttemptContext,
        )
        from agent_engine_runner_shared.workflow import attempt_context_scope

        payload = {
            "suspend_reason": "guardrail_require_review",
            "suspend_context": {},
        }
        with attempt_context_scope(AttemptContext(attempt_id="attempt-1")):
            with patch(
                "langgraph.types.interrupt", return_value="approved"
            ) as interrupt:
                assert _suspend(payload) == "approved"

        interrupt.assert_called_once_with(payload)

    @pytest.mark.anyio
    async def test_async_tool_node_does_not_convert_durable_suspension_to_tool_error(
        self,
    ):
        from langchain_core.messages import AIMessage
        from langchain_core.tools import tool
        from langgraph.prebuilt import ToolNode
        from langgraph.prebuilt.tool_node import ToolRuntime

        from agent_engine_sdk_langgraph.runtime import _suspend_durable_activity
        from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import (
            AttemptContext,
        )
        from agent_engine_runner_shared.workflow import (
            DurableActivitySuspended,
            attempt_context_scope,
        )

        @tool
        async def wait_for_review() -> str:
            """Wait for human review."""
            _suspend_durable_activity(
                {
                    "suspend_reason": "awaiting_human_review",
                    "suspend_context": {},
                }
            )
            return "unreachable"

        node = ToolNode([wait_for_review], handle_tool_errors=True)
        call: ToolCall = {
            "name": "wait_for_review",
            "args": {},
            "id": "call-1",
            "type": "tool_call",
        }
        runtime = ToolRuntime(
            state={"messages": [AIMessage(content="", tool_calls=[call])]},
            context=None,
            config={},
            stream_writer=lambda _: None,
            tool_call_id="call-1",
            store=None,
        )

        with (
            attempt_context_scope(AttemptContext(attempt_id="attempt-1")),
            pytest.raises(DurableActivitySuspended),
        ):
            await node._arun_one(call, "dict", runtime)


class TestAppLlm:
    """Tests for App.llm()."""

    @pytest.fixture(autouse=True)
    def _reset_hooks(self):
        from agent_engine_runner_shared.hooks import entrypoint_scope, reset_hooks

        reset_hooks()
        # app.llm() is only callable inside the entrypoint's dynamic extent;
        # these tests call App.llm() directly (not via get_agent()).
        with entrypoint_scope():
            yield
        reset_hooks()

    def test_llm_wraps_with_secure_wrapped_llm(self):
        """App.llm() returns a SecureWrappedLLM wrapping the provided model."""
        from agent_engine_runner_shared import RuntimeMode

        app = App(app_name="Test Agent")
        app._runtime.mode = RuntimeMode.AER
        mock_llm = MagicMock()
        mock_wrapper_fn = MagicMock()
        mock_secure_llm = MagicMock()

        llm_path = "agent_engine_sdk_langgraph.runtime.SecureWrappedLLM"
        wrapper_path = "agent_engine_sdk_langgraph.runtime.get_current_wrapper"
        with (
            patch(wrapper_path, mock_wrapper_fn),
            patch(llm_path, return_value=mock_secure_llm) as mock_cls,
        ):
            result = app.llm(mock_llm)

        mock_cls.assert_called_once_with(
            llm=mock_llm, get_wrapper=mock_wrapper_fn, llm_id="__default__"
        )
        assert result is mock_secure_llm

    def test_llm_registers_into_named_registry_in_tool_mode(self):
        """In TOOL mode, App.llm() registers under '__default__' and returns it raw."""
        from agent_engine_runner_shared import RuntimeMode
        from agent_engine_runner_shared.hooks import get_named_llm, reset_hooks

        app = App(app_name="Test Agent")
        app._runtime.mode = RuntimeMode.TOOL
        mock_llm = MagicMock()

        try:
            result = app.llm(mock_llm)

            assert result is mock_llm
            assert get_named_llm("__default__") is mock_llm
        finally:
            reset_hooks()


class TestAppGetAgent:
    """Tests for App.get_agent()."""

    @pytest.fixture(autouse=True)
    def _reset_hooks(self):
        from agent_engine_runner_shared.hooks import reset_hooks

        reset_hooks()
        yield
        reset_hooks()

    def test_get_agent_returns_langgraph_base_agent(self):
        """App.get_agent() returns a LangGraphBaseAgent wrapping the built graph."""
        from agent_engine_sdk_langgraph.agent import LangGraphBaseAgent

        app = App(app_name="Test Agent")
        mock_graph = MagicMock()

        @app.entrypoint
        def build_graph():
            return mock_graph

        agent = app.get_agent()

        assert isinstance(agent, LangGraphBaseAgent)
        assert agent._graph is mock_graph

    def test_get_agent_wraps_execution_callbacks(self):
        """App.get_agent() wraps BaseExecutionCallback."""
        from agent_engine_sdk_langgraph.node_logger_adapter import (
            LangGraphCallbackAdapter,
        )

        app = App(app_name="Test Agent")
        mock_graph = MagicMock()

        # Create a mock BaseExecutionCallback (has on_node_start but not on_chain_start)
        execution_callback = MagicMock(
            spec=["on_node_start", "on_node_end", "on_node_error", "on_node_suspend"],
        )

        @app.entrypoint
        def build_graph():
            return mock_graph

        agent = app.get_agent(callbacks=[execution_callback])

        assert len(agent._callbacks) == 1
        assert isinstance(agent._callbacks[0], LangGraphCallbackAdapter)

    def test_get_agent_cache_hit_does_not_reset_llm_registry(self):
        """get_agent() caches the built graph, so a second call with the
        same adapter version is a cache hit -- the entrypoint does not
        re-run and the registry keeps the first build's LLM."""
        from agent_engine_runner_shared import RuntimeMode
        from agent_engine_runner_shared.hooks import get_named_llm

        app = App(app_name="Test Agent")
        app._runtime.mode = RuntimeMode.AER
        first_llm = MagicMock(name="first_llm")
        second_llm = MagicMock(name="second_llm")
        llms = iter([first_llm, second_llm])

        @app.entrypoint
        def build_graph():
            app.llm(next(llms), llm_id="primary")
            return MagicMock()

        app.get_agent()
        assert get_named_llm("primary") is first_llm

        app.get_agent()
        assert get_named_llm("primary") is first_llm, (
            "a cache hit must not re-run the entrypoint or reset the registry"
        )

    def test_get_agent_rebuilds_and_resets_llm_registry_on_version_change(self):
        """A runtime/adapter version change invalidates the graph cache, so
        the entrypoint re-runs and can register the same llm_id again."""
        from unittest.mock import patch

        from agent_engine_runner_shared import RuntimeMode
        from agent_engine_runner_shared.hooks import get_named_llm

        app = App(app_name="Test Agent")
        app._runtime.mode = RuntimeMode.AER
        first_llm = MagicMock(name="first_llm")
        second_llm = MagicMock(name="second_llm")
        llms = iter([first_llm, second_llm])

        @app.entrypoint
        def build_graph():
            app.llm(next(llms), llm_id="primary")
            return MagicMock()

        with patch(
            "agent_engine_sdk_langgraph.runtime._adapter_version", return_value="v1"
        ):
            app.get_agent()
        assert get_named_llm("primary") is first_llm

        with patch(
            "agent_engine_sdk_langgraph.runtime._adapter_version", return_value="v2"
        ):
            app.get_agent()
        assert get_named_llm("primary") is second_llm

    def test_get_agent_raises_without_entrypoint(self):
        """App.get_agent() raises RuntimeError if no entrypoint registered."""
        app = App(app_name="Test Agent")

        with pytest.raises(RuntimeError, match="No entrypoint registered"):
            app.get_agent()


class TestAppDeepAgent:
    """Tests for App.deep_agent() checkpointer resolution and feature-flag guard."""

    @pytest.fixture(autouse=True)
    def _reset_hooks(self):
        # deep_agent() registers the orchestrator LLM under "__default__" via
        # self.llm(). The registry is process-global, so reset it around every
        # test — otherwise a second test hits "llm_id already registered".
        from agent_engine_runner_shared.hooks import reset_hooks

        reset_hooks()
        yield
        reset_hooks()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_app(self, mode, mongodb_uri="mongodb://localhost:27017"):
        """Return an App wired with a mock runtime in the given mode."""
        app = App(app_name="Test Agent")
        mock_runtime = MagicMock()
        mock_runtime.mode = mode
        mock_runtime.mongodb_uri = mongodb_uri
        mock_runtime.agent_config.path = None
        mock_runtime.agent_config.feature_enabled.return_value = True
        app._runtime = mock_runtime
        return app

    def _deep_agent_patches(self):
        """Return a patch.dict that satisfies deep_agent()'s optional imports."""
        import sys

        mock_deep_agent_mod = MagicMock()
        self._mock_create = MagicMock(return_value=MagicMock())
        mock_deep_agent_mod.create_agent_engine_deep_agent = self._mock_create

        mock_toolpod_mod = MagicMock()
        self._mock_backend_cls = MagicMock(return_value=MagicMock())
        mock_toolpod_mod.AgentEngineToolPodBackend = self._mock_backend_cls

        return patch.dict(
            sys.modules,
            {
                "agent_engine_sdk_langgraph.deep_agent": mock_deep_agent_mod,
                "agent_engine_sdk_langgraph.backends.toolpod": mock_toolpod_mod,
            },
        )

    def _call_deep_agent(self, app, **kwargs):
        """Call app.deep_agent() with all optional dependencies mocked out."""
        from agent_engine_runner_shared.hooks import entrypoint_scope

        mock_llm = MagicMock()
        with (
            self._deep_agent_patches(),
            patch("agent_engine_sdk_langgraph.runtime.SecureWrappedLLM"),
            patch("agent_engine_sdk_langgraph.runtime.get_current_wrapper"),
            # deep_agent() calls self.llm(...) internally, which is only
            # callable inside the entrypoint's dynamic extent.
            entrypoint_scope(),
        ):
            app.deep_agent(mock_llm, backend=MagicMock(), **kwargs)
        return self._mock_create

    # ------------------------------------------------------------------
    # Feature-flag guard
    # ------------------------------------------------------------------

    def test_deep_agent_raises_when_feature_flag_disabled(self):
        """App.deep_agent() raises RuntimeError when deep_agent feature flag is off."""
        from agent_engine_runner_shared import RuntimeMode

        app = App(app_name="Test Agent")
        mock_runtime = MagicMock()
        mock_runtime.mode = RuntimeMode.AER
        mock_runtime.agent_config.feature_enabled.return_value = False
        app._runtime = mock_runtime

        with (
            self._deep_agent_patches(),
            pytest.raises(RuntimeError, match="features.deep_agent: true"),
        ):
            app.deep_agent(MagicMock())

    # ------------------------------------------------------------------
    # Checkpointer sentinel resolution — default (_UNSET)
    # ------------------------------------------------------------------

    def test_deep_agent_uses_checkpointer_in_aer_mode(self):
        """Default checkpointer resolves to app.checkpointer() in AER mode."""
        from agent_engine_runner_shared import RuntimeMode
        from agent_engine_runner_shared.hooks import entrypoint_scope

        app = self._make_app(RuntimeMode.AER)
        mock_saver = MagicMock()

        with (
            self._deep_agent_patches(),
            patch("agent_engine_sdk_langgraph.runtime.SecureWrappedLLM"),
            patch("agent_engine_sdk_langgraph.runtime.get_current_wrapper"),
            patch("pymongo.MongoClient"),
            patch("langgraph.checkpoint.mongodb.MongoDBSaver", return_value=mock_saver),
            entrypoint_scope(),
        ):
            app.deep_agent(MagicMock(), backend=MagicMock())

        from agent_engine_sdk_langgraph.platform_checkpointer import (
            PlatformCheckpointer,
        )

        _, kwargs = self._mock_create.call_args
        assert isinstance(kwargs["checkpointer"], PlatformCheckpointer)
        assert kwargs["checkpointer"].native is mock_saver

    def test_deep_agent_disables_checkpointer_in_tool_mode(self):
        """Default checkpointer resolves to None in TOOL mode (no crash)."""
        from agent_engine_runner_shared import RuntimeMode

        app = self._make_app(RuntimeMode.TOOL)
        mock_create = self._call_deep_agent(app)

        _, kwargs = mock_create.call_args
        assert kwargs["checkpointer"] is None

    def test_deep_agent_passes_agent_config_dir_as_skills_base_dir(self, tmp_path):
        """Relative skill paths resolve from the deployed agent.yaml directory."""
        from agent_engine_runner_shared import RuntimeMode

        app = self._make_app(RuntimeMode.TOOL)
        app._runtime.agent_config.path = tmp_path / "agent.yaml"

        mock_create = self._call_deep_agent(app)

        _, kwargs = mock_create.call_args
        assert kwargs["skills_base_dir"] == tmp_path

    def test_deep_agent_resolves_relative_agent_config_dir(self, monkeypatch, tmp_path):
        """Relative agent.yaml paths still produce absolute skill paths."""
        from agent_engine_runner_shared import RuntimeMode

        monkeypatch.chdir(tmp_path)
        app = self._make_app(RuntimeMode.TOOL)
        app._runtime.agent_config.path = Path("agent.yaml")

        mock_create = self._call_deep_agent(app)

        _, kwargs = mock_create.call_args
        assert kwargs["skills_base_dir"] == tmp_path

    def test_deep_agent_uses_custom_skills_dir_env(self, monkeypatch, tmp_path):
        """AGENTIC_SKILLS_DIR can move skill resolution within the agent root."""
        from agent_engine_runner_shared import RuntimeMode

        skills_root = tmp_path / "src" / "reviewer" / "skills"
        app = self._make_app(RuntimeMode.TOOL)
        app._runtime.agent_config.path = tmp_path / "agent.yaml"
        monkeypatch.setenv("AGENTIC_SKILLS_DIR", "src/reviewer/skills")

        mock_create = self._call_deep_agent(app)

        _, kwargs = mock_create.call_args
        assert kwargs["skills_base_dir"] == skills_root

    def test_deep_agent_rejects_absolute_custom_skills_dir_env(
        self,
        monkeypatch,
    ):
        """AGENTIC_SKILLS_DIR must be relative to the agent source root."""
        from agent_engine_runner_shared import RuntimeMode

        app = self._make_app(RuntimeMode.TOOL)
        monkeypatch.setenv("AGENTIC_SKILLS_DIR", "/app/skills")

        with pytest.raises(ValueError, match="AGENTIC_SKILLS_DIR must be relative"):
            self._call_deep_agent(app)

    def test_deep_agent_rejects_custom_skills_dir_outside_agent_root(
        self,
        monkeypatch,
        tmp_path,
    ):
        """AGENTIC_SKILLS_DIR cannot traverse above the agent source root."""
        from agent_engine_runner_shared import RuntimeMode

        app = self._make_app(RuntimeMode.TOOL)
        app._runtime.agent_config.path = tmp_path / "agent.yaml"
        monkeypatch.setenv("AGENTIC_SKILLS_DIR", "../shared-skills")

        with pytest.raises(ValueError, match="AGENTIC_SKILLS_DIR must stay within"):
            self._call_deep_agent(app)

    def test_deep_agent_warns_when_relative_skills_have_no_base_dir(
        self,
    ):
        """Relative skills need agent.yaml or AGENTIC_SKILLS_DIR for a stable root."""
        from agent_engine_runner_shared import RuntimeMode

        app = self._make_app(RuntimeMode.TOOL)
        app._runtime.agent_config.path = None

        with patch("agent_engine_sdk_langgraph.runtime.logger.warning") as mock_warning:
            mock_create = self._call_deep_agent(
                app,
                skills=["skills/review-skill", "/app/skills/absolute-skill"],
            )

        _, kwargs = mock_create.call_args
        assert kwargs["skills_base_dir"] is None
        mock_warning.assert_any_call(
            "Relative skill paths %s cannot be resolved because agent.yaml "
            "was not found and AGENTIC_SKILLS_DIR is unset",
            ["skills/review-skill"],
        )

    # ------------------------------------------------------------------
    # Checkpointer sentinel resolution — explicit values
    # ------------------------------------------------------------------

    def test_deep_agent_passes_explicit_none_checkpointer(self):
        """Explicit checkpointer=None disables checkpointing even in AER mode."""
        from agent_engine_runner_shared import RuntimeMode

        app = self._make_app(RuntimeMode.AER)
        mock_create = self._call_deep_agent(app, checkpointer=None)

        _, kwargs = mock_create.call_args
        assert kwargs["checkpointer"] is None

    def test_deep_agent_passes_explicit_checkpointer_instance(self):
        """Explicit checkpointer instance is forwarded as-is, regardless of mode."""
        from agent_engine_runner_shared import RuntimeMode
        from agent_engine_runner_shared.hooks import reset_llm_registry

        custom_checkpointer = MagicMock()
        for mode in [RuntimeMode.AER, RuntimeMode.TOOL]:
            # Each iteration builds a fresh deep agent, which registers the
            # orchestrator under "__default__"; clear the process-global
            # registry between builds so the second call doesn't collide.
            reset_llm_registry()
            app = self._make_app(mode)
            mock_create = self._call_deep_agent(app, checkpointer=custom_checkpointer)

            _, kwargs = mock_create.call_args
            assert kwargs["checkpointer"] is custom_checkpointer

    # ------------------------------------------------------------------
    # Named-LLM registry preservation
    # ------------------------------------------------------------------

    def test_deep_agent_preserves_named_subagent_registrations(self):
        """deep_agent() must not clear the named-LLM registry.

        A subagent model built inline as ``app.llm(model, "researcher")`` in
        the ``subagents=[...]`` argument registers before the method body runs
        (Python evaluates arguments first). If deep_agent() reset the registry
        it would wipe that entry and the Tool Pod's /invoke_llm could no longer
        resolve the subagent LLM by id.
        """
        from agent_engine_runner_shared import RuntimeMode
        from agent_engine_runner_shared.hooks import (
            entrypoint_scope,
            get_named_llm,
            register_llm,
        )

        subagent_llm = MagicMock(name="subagent_llm")
        # entrypoint_scope() is framework-internal; user code never calls it.
        # It stands in here for the SDK evaluating @app.entrypoint, during
        # which the inline app.llm(model, "researcher") argument registers.
        with entrypoint_scope():
            register_llm("researcher", subagent_llm)

        app = self._make_app(RuntimeMode.TOOL)
        self._call_deep_agent(app)

        assert get_named_llm("researcher") is subagent_llm
