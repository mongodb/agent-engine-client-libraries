from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from mcp.types import CallToolResult, TextContent

from agent_engine_runner_shared import mcp_tools
from agent_engine_runner_shared.agent_config import RuntimeMCPConfig, RuntimeMCPServerConfig
from agent_engine_runner_shared.mcp_tools import (
    MCPConfigError,
    MCPToolError,
    MCPToolResult,
    discover_mcp_tools,
    discover_mcp_tools_async,
    is_configured_mcp_sdk_tool_name,
    make_mcp_sdk_tool_name,
    make_sync_mcp_tool_callable,
    mcp_server_network_hosts,
    normalize_mcp_tool_result,
    resolve_configured_mcp_tool_binding,
    resolve_mcp_auth,
    resolve_mcp_headers,
)
from agent_engine_runner_shared.models import ToolPodExecuteResponse


class _FakeStreamableHTTPClient:
    async def __aenter__(self):
        return "read", "write", lambda: None

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _FailingStreamableHTTPClient:
    def __init__(self, exc: BaseException):
        self._exc = exc

    async def __aenter__(self):
        raise self._exc

    async def __aexit__(self, exc_type, exc, tb):
        return None


def _http_status_exception_group(status_code: int) -> ExceptionGroup:
    request = httpx.Request("POST", "https://api.githubcopilot.com/mcp/")
    response = httpx.Response(status_code, request=request)
    return ExceptionGroup(
        "unhandled errors in a TaskGroup",
        [httpx.HTTPStatusError("request failed", request=request, response=response)],
    )


def test_make_mcp_sdk_tool_name_prefixes_server_and_sanitizes_provider_names():
    assert make_mcp_sdk_tool_name("devprod-gateway", "jira.search-issues") == (
        "devprod_gateway__jira_search_issues"
    )


def test_resolve_mcp_headers_adds_bearer_token(monkeypatch):
    config = RuntimeMCPServerConfig.model_validate(
        {
            "url": "https://api.githubcopilot.com/mcp/",
            "headers": {"X-MCP-Readonly": "true"},
            "auth": {"type": "bearer_env", "token_env": "GITHUB_MCP_TOKEN"},
        }
    )
    monkeypatch.setenv("GITHUB_MCP_TOKEN", "token-123")

    assert resolve_mcp_headers("github", config) == {
        "X-MCP-Readonly": "true",
        "Authorization": "Bearer token-123",
    }


def test_resolve_mcp_headers_leaves_oauth_tokens_to_sdk():
    config = RuntimeMCPServerConfig.model_validate(
        {
            "url": "https://api.githubcopilot.com/mcp/",
            "headers": {"X-MCP-Readonly": "true"},
            "auth": {"type": "oauth"},
        }
    )

    assert resolve_mcp_headers("github", config) == {"X-MCP-Readonly": "true"}


def test_resolve_mcp_headers_leaves_client_credentials_to_sdk_auth_provider():
    config = RuntimeMCPServerConfig.model_validate(
        {
            "url": "https://cloud-dev.mongodb.com/api/private/mcp",
            "headers": {"X-MCP-Readonly": "true"},
            "auth": {
                "type": "client_credentials",
                "token_url": "https://cloud-dev.mongodb.com/api/oauth/token",
                "client_id_env": "ATLAS_MCP_CLIENT_ID",
                "client_secret_env": "ATLAS_MCP_CLIENT_SECRET",
            },
        }
    )

    assert resolve_mcp_headers("atlas", config) == {"X-MCP-Readonly": "true"}


def test_resolve_mcp_auth_creates_oauth_provider(monkeypatch):
    calls = []

    def fake_make_mcp_oauth_auth(server_name, config):
        calls.append((server_name, config.url))
        return "oauth-provider"

    monkeypatch.setattr(
        "agent_engine_runner_shared.mcp_oauth.make_mcp_oauth_auth", fake_make_mcp_oauth_auth
    )
    config = RuntimeMCPServerConfig.model_validate(
        {
            "url": "https://api.githubcopilot.com/mcp/",
            "auth": {"type": "oauth"},
        }
    )

    assert resolve_mcp_auth("github", config) == "oauth-provider"
    assert calls == [("github", "https://api.githubcopilot.com/mcp/")]


def test_resolve_mcp_auth_creates_client_credentials_provider(monkeypatch):
    calls = []

    def fake_make_mcp_client_credentials_auth(server_name, config):
        calls.append((server_name, config.url))
        return "client-credentials-provider"

    monkeypatch.setattr(
        "agent_engine_runner_shared.mcp_oauth.make_mcp_client_credentials_auth",
        fake_make_mcp_client_credentials_auth,
    )
    config = RuntimeMCPServerConfig.model_validate(
        {
            "url": "https://cloud-dev.mongodb.com/api/private/mcp",
            "auth": {
                "type": "client_credentials",
                "token_url": "https://cloud-dev.mongodb.com/api/oauth/token",
                "client_id_env": "ATLAS_MCP_CLIENT_ID",
                "client_secret_env": "ATLAS_MCP_CLIENT_SECRET",
            },
        }
    )

    assert resolve_mcp_auth("atlas", config) == "client-credentials-provider"
    assert calls == [("atlas", "https://cloud-dev.mongodb.com/api/private/mcp")]


def test_resolve_mcp_auth_allows_oauth_in_tool_pod(monkeypatch):
    monkeypatch.setenv("RUNNER_MODE", "tool")
    calls = []

    def fake_make_mcp_oauth_auth(server_name, config):
        calls.append((server_name, config.url))
        return "oauth-provider"

    monkeypatch.setattr(
        "agent_engine_runner_shared.mcp_oauth.make_mcp_oauth_auth", fake_make_mcp_oauth_auth
    )
    config = RuntimeMCPServerConfig.model_validate(
        {
            "url": "https://api.githubcopilot.com/mcp/",
            "auth": {"type": "oauth"},
        }
    )

    assert resolve_mcp_auth("github", config) == "oauth-provider"
    assert calls == [("github", "https://api.githubcopilot.com/mcp/")]


def test_mcp_server_network_hosts_derives_host_from_url():
    config = RuntimeMCPServerConfig.model_validate(
        {
            "url": "https://api.githubcopilot.com/mcp/x/issues/readonly",
        }
    )

    assert mcp_server_network_hosts(config) == ["api.githubcopilot.com"]


def test_mcp_server_network_hosts_includes_direct_token_url_host():
    config = RuntimeMCPServerConfig.model_validate(
        {
            "url": "https://mcp.example.com/mcp",
            "auth": {
                "type": "client_credentials",
                "token_url": "https://auth.example.com/oauth/token",
                "client_id_env": "MCP_CLIENT_ID",
                "client_secret_env": "MCP_CLIENT_SECRET",
            },
        }
    )

    assert mcp_server_network_hosts(config) == ["mcp.example.com", "auth.example.com"]


def test_discover_mcp_tools_filters_allowed_tools(monkeypatch):
    async def fake_list_server_tools(server_name, config):
        return [
            SimpleNamespace(
                name="search_issues",
                description="Search issues",
                inputSchema={"type": "object", "properties": {}},
            ),
            SimpleNamespace(
                name="create_issue",
                description="Create issue",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    monkeypatch.setattr(mcp_tools, "_list_server_tools", fake_list_server_tools)
    config = RuntimeMCPConfig.model_validate(
        {
            "servers": {
                "github": {
                    "url": "https://api.githubcopilot.com/mcp/",
                    "allowed_tools": ["search_issues"],
                }
            }
        }
    )

    bindings = discover_mcp_tools(config)

    assert [binding.sdk_tool_name for binding in bindings] == ["github__search_issues"]
    assert bindings[0].tool_name == "search_issues"


def test_discover_mcp_tools_rejects_active_event_loop(monkeypatch):
    async def fake_list_server_tools(server_name, config):
        return [
            SimpleNamespace(
                name="search_issues",
                description="Search issues",
                inputSchema={"type": "object", "properties": {}},
            )
        ]

    monkeypatch.setattr(mcp_tools, "_list_server_tools", fake_list_server_tools)
    config = RuntimeMCPConfig.model_validate(
        {
            "servers": {
                "github": {
                    "url": "https://api.githubcopilot.com/mcp/",
                }
            }
        }
    )

    async def run_discovery():
        return discover_mcp_tools(config)

    with pytest.raises(MCPConfigError, match="running event loop"):
        asyncio.run(run_discovery())


def test_discover_mcp_tools_async_runs_from_active_event_loop(monkeypatch):
    async def fake_list_server_tools(server_name, config):
        return [
            SimpleNamespace(
                name="search_issues",
                description="Search issues",
                inputSchema={"type": "object", "properties": {}},
            )
        ]

    monkeypatch.setattr(mcp_tools, "_list_server_tools", fake_list_server_tools)
    config = RuntimeMCPConfig.model_validate(
        {
            "servers": {
                "github": {
                    "url": "https://api.githubcopilot.com/mcp/",
                }
            }
        }
    )

    async def run_discovery():
        return await discover_mcp_tools_async(config)

    bindings = asyncio.run(run_discovery())

    assert [binding.sdk_tool_name for binding in bindings] == ["github__search_issues"]


def test_list_server_tools_pages_until_next_cursor_is_empty(monkeypatch):
    class FakeClientSession:
        def __init__(self, read_stream, write_stream):
            self.cursors: list[str | None] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def initialize(self):
            return None

        async def list_tools(self, cursor=None):
            self.cursors.append(cursor)
            if cursor is None:
                return SimpleNamespace(
                    tools=[
                        SimpleNamespace(
                            name="search_issues",
                            description="Search issues",
                            inputSchema={"type": "object", "properties": {}},
                        )
                    ],
                    nextCursor="page-2",
                )
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="get_issue",
                        description="Get issue",
                        inputSchema={"type": "object", "properties": {}},
                    )
                ],
                nextCursor=None,
            )

    captured_client: FakeClientSession | None = None
    captured_stream_kwargs: dict[str, object] = {}

    def fake_streamablehttp_client(url, **kwargs):
        captured_stream_kwargs.update(kwargs)
        return _FakeStreamableHTTPClient()

    def fake_client_session(read_stream, write_stream):
        nonlocal captured_client
        captured_client = FakeClientSession(read_stream, write_stream)
        return captured_client

    monkeypatch.setattr(
        "mcp.client.streamable_http.streamablehttp_client",
        fake_streamablehttp_client,
    )
    monkeypatch.setattr("mcp.ClientSession", fake_client_session)
    config = RuntimeMCPServerConfig.model_validate(
        {
            "url": "https://api.githubcopilot.com/mcp/",
            "timeout_seconds": 12,
        }
    )

    tools = asyncio.run(mcp_tools._list_server_tools("github", config))

    assert [tool.name for tool in tools] == ["search_issues", "get_issue"]
    assert captured_client is not None
    assert captured_client.cursors == [None, "page-2"]
    assert captured_stream_kwargs["timeout"] == 12
    assert captured_stream_kwargs["sse_read_timeout"] == 12


def test_sync_mcp_tool_callable_uses_configured_sse_timeout(monkeypatch):
    class FakeClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def initialize(self):
            return None

        async def call_tool(self, tool_name, arguments):
            assert tool_name == "search_issues"
            assert arguments == {"query": "status:open"}
            return CallToolResult(content=[TextContent(type="text", text="ok")])

    captured_stream_kwargs: dict[str, object] = {}

    def fake_streamablehttp_client(url, **kwargs):
        captured_stream_kwargs.update(kwargs)
        return _FakeStreamableHTTPClient()

    monkeypatch.setattr(
        "mcp.client.streamable_http.streamablehttp_client",
        fake_streamablehttp_client,
    )
    monkeypatch.setattr("mcp.ClientSession", lambda read_stream, write_stream: FakeClientSession())
    server_config = RuntimeMCPServerConfig.model_validate(
        {
            "url": "https://api.githubcopilot.com/mcp/",
            "timeout_seconds": 7,
        }
    )
    binding = mcp_tools.MCPToolBinding(
        server_name="github",
        tool_name="search_issues",
        sdk_tool_name="github__search_issues",
        description="Search issues",
        input_schema={"type": "object", "properties": {}},
        server_config=server_config,
    )

    result = make_sync_mcp_tool_callable(binding)(query="status:open")

    assert result == MCPToolResult(content=[{"type": "text", "text": "ok"}])
    assert captured_stream_kwargs["timeout"] == 7
    assert captured_stream_kwargs["sse_read_timeout"] == 7


def test_sync_mcp_tool_callable_rejects_active_event_loop(monkeypatch):
    class FakeClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def initialize(self):
            return None

        async def call_tool(self, tool_name, arguments):
            return CallToolResult(content=[TextContent(type="text", text=arguments["query"])])

    def fake_streamablehttp_client(url, **kwargs):
        return _FakeStreamableHTTPClient()

    monkeypatch.setattr(
        "mcp.client.streamable_http.streamablehttp_client",
        fake_streamablehttp_client,
    )
    monkeypatch.setattr("mcp.ClientSession", lambda read_stream, write_stream: FakeClientSession())
    server_config = RuntimeMCPServerConfig.model_validate(
        {
            "url": "https://api.githubcopilot.com/mcp/",
        }
    )
    binding = mcp_tools.MCPToolBinding(
        server_name="github",
        tool_name="search_issues",
        sdk_tool_name="github__search_issues",
        description="Search issues",
        input_schema={"type": "object", "properties": {}},
        server_config=server_config,
    )

    async def invoke_tool():
        return make_sync_mcp_tool_callable(binding)(query="status:open")

    with pytest.raises(MCPToolError, match="running event loop"):
        asyncio.run(invoke_tool())


def test_list_server_tools_unwraps_http_errors_from_mcp_exception_group(monkeypatch):
    def fake_streamablehttp_client(url, **kwargs):
        return _FailingStreamableHTTPClient(_http_status_exception_group(401))

    monkeypatch.setattr(
        "mcp.client.streamable_http.streamablehttp_client",
        fake_streamablehttp_client,
    )
    config = RuntimeMCPServerConfig.model_validate(
        {
            "url": "https://api.githubcopilot.com/mcp/",
        }
    )

    with pytest.raises(MCPConfigError, match="HTTP 401 Unauthorized"):
        asyncio.run(mcp_tools._list_server_tools("github", config))


def test_mcp_tool_callable_unwraps_http_errors_from_mcp_exception_group(monkeypatch):
    def fake_streamablehttp_client(url, **kwargs):
        return _FailingStreamableHTTPClient(_http_status_exception_group(401))

    monkeypatch.setattr(
        "mcp.client.streamable_http.streamablehttp_client",
        fake_streamablehttp_client,
    )
    server_config = RuntimeMCPServerConfig.model_validate(
        {
            "url": "https://api.githubcopilot.com/mcp/",
        }
    )
    binding = mcp_tools.MCPToolBinding(
        server_name="github",
        tool_name="search_issues",
        sdk_tool_name="github__search_issues",
        description="Search issues",
        input_schema={"type": "object", "properties": {}},
        server_config=server_config,
    )

    with pytest.raises(MCPToolError, match="HTTP 401 Unauthorized"):
        make_sync_mcp_tool_callable(binding)(query="status:open")


def test_list_server_tools_propagates_cancellation(monkeypatch):
    def fake_streamablehttp_client(url, **kwargs):
        return _FailingStreamableHTTPClient(asyncio.CancelledError())

    monkeypatch.setattr(
        "mcp.client.streamable_http.streamablehttp_client",
        fake_streamablehttp_client,
    )
    config = RuntimeMCPServerConfig.model_validate(
        {
            "url": "https://api.githubcopilot.com/mcp/",
        }
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(mcp_tools._list_server_tools("github", config))


def test_mcp_tool_callable_propagates_cancellation(monkeypatch):
    def fake_streamablehttp_client(url, **kwargs):
        return _FailingStreamableHTTPClient(asyncio.CancelledError())

    monkeypatch.setattr(
        "mcp.client.streamable_http.streamablehttp_client",
        fake_streamablehttp_client,
    )
    server_config = RuntimeMCPServerConfig.model_validate(
        {
            "url": "https://api.githubcopilot.com/mcp/",
        }
    )
    binding = mcp_tools.MCPToolBinding(
        server_name="github",
        tool_name="search_issues",
        sdk_tool_name="github__search_issues",
        description="Search issues",
        input_schema={"type": "object", "properties": {}},
        server_config=server_config,
    )

    with pytest.raises(asyncio.CancelledError):
        make_sync_mcp_tool_callable(binding)(query="status:open")


def test_discover_mcp_tools_rejects_missing_allowed_tool(monkeypatch):
    async def fake_list_server_tools(server_name, config):
        return [
            SimpleNamespace(
                name="search_issues",
                description="Search issues",
                inputSchema={"type": "object", "properties": {}},
            )
        ]

    monkeypatch.setattr(mcp_tools, "_list_server_tools", fake_list_server_tools)
    config = RuntimeMCPConfig.model_validate(
        {
            "servers": {
                "github": {
                    "url": "https://api.githubcopilot.com/mcp/",
                    "allowed_tools": ["missing_tool"],
                }
            }
        }
    )

    with pytest.raises(MCPConfigError, match="missing_tool"):
        discover_mcp_tools(config)


def test_resolve_configured_mcp_tool_binding_uses_allowed_tool_original_name():
    config = RuntimeMCPConfig.model_validate(
        {
            "servers": {
                "devprod-gateway": {
                    "url": "https://devprod.example.com/mcp",
                    "allowed_tools": ["jira.search-issues"],
                }
            }
        }
    )

    binding = resolve_configured_mcp_tool_binding(
        config,
        "devprod_gateway__jira_search_issues",
        "devprod-gateway",
        "jira.search-issues",
    )

    assert binding is not None
    assert binding.server_name == "devprod-gateway"
    assert binding.tool_name == "jira.search-issues"


def test_is_configured_mcp_sdk_tool_name_matches_configured_server_prefix():
    config = RuntimeMCPConfig.model_validate(
        {
            "servers": {
                "tableau": {
                    "url": "https://tableau.example.com/mcp",
                }
            }
        }
    )

    assert is_configured_mcp_sdk_tool_name(config, "tableau__get_view_data") is True


def test_is_configured_mcp_sdk_tool_name_ignores_non_mcp_tool():
    config = RuntimeMCPConfig.model_validate(
        {
            "servers": {
                "tableau": {
                    "url": "https://tableau.example.com/mcp",
                }
            }
        }
    )

    assert is_configured_mcp_sdk_tool_name(config, "regular_tool") is False


def test_resolve_configured_mcp_tool_binding_uses_discovered_original_name():
    config = RuntimeMCPConfig.model_validate(
        {
            "servers": {
                "tableau": {
                    "url": "https://tableau.example.com/mcp",
                }
            }
        }
    )

    binding = resolve_configured_mcp_tool_binding(
        config,
        "tableau__get_view_data",
        "tableau",
        "get-view-data",
    )

    assert binding is not None
    assert binding.server_name == "tableau"
    assert binding.tool_name == "get-view-data"


def test_resolve_configured_mcp_tool_binding_rejects_mismatched_discovery_metadata():
    config = RuntimeMCPConfig.model_validate(
        {
            "servers": {
                "tableau": {
                    "url": "https://tableau.example.com/mcp",
                }
            }
        }
    )

    with pytest.raises(MCPConfigError, match="does not match"):
        resolve_configured_mcp_tool_binding(
            config,
            "tableau__get_view_data",
            "tableau",
            "list-views",
        )


def test_resolve_configured_mcp_tool_binding_enforces_allowed_tools_with_metadata():
    config = RuntimeMCPConfig.model_validate(
        {
            "servers": {
                "tableau": {
                    "url": "https://tableau.example.com/mcp",
                    "allowed_tools": ["list-views"],
                }
            }
        }
    )

    with pytest.raises(MCPConfigError, match="not in allowed_tools"):
        resolve_configured_mcp_tool_binding(
            config,
            "tableau__get_view_data",
            "tableau",
            "get-view-data",
        )


def test_list_server_tools_rejects_too_many_pages(monkeypatch):
    class FakeClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def initialize(self):
            return None

        async def list_tools(self, cursor=None):
            return SimpleNamespace(tools=[], nextCursor="again")

    def fake_streamablehttp_client(url, **kwargs):
        return _FakeStreamableHTTPClient()

    monkeypatch.setattr(mcp_tools, "_MAX_LIST_TOOL_PAGES", 2)
    monkeypatch.setattr(
        "mcp.client.streamable_http.streamablehttp_client",
        fake_streamablehttp_client,
    )
    monkeypatch.setattr("mcp.ClientSession", lambda read_stream, write_stream: FakeClientSession())
    config = RuntimeMCPServerConfig.model_validate({"url": "https://api.githubcopilot.com/mcp/"})

    with pytest.raises(MCPConfigError, match="exceeded 2 pages"):
        asyncio.run(mcp_tools._list_server_tools("github", config))


def test_list_server_tools_rejects_too_many_tools(monkeypatch):
    class FakeClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def initialize(self):
            return None

        async def list_tools(self, cursor=None):
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="search_issues",
                        description="Search issues",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    SimpleNamespace(
                        name="get_issue",
                        description="Get issue",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                ],
                nextCursor=None,
            )

    def fake_streamablehttp_client(url, **kwargs):
        return _FakeStreamableHTTPClient()

    monkeypatch.setattr(mcp_tools, "_MAX_TOOLS_PER_SERVER", 1)
    monkeypatch.setattr(
        "mcp.client.streamable_http.streamablehttp_client",
        fake_streamablehttp_client,
    )
    monkeypatch.setattr("mcp.ClientSession", lambda read_stream, write_stream: FakeClientSession())
    config = RuntimeMCPServerConfig.model_validate({"url": "https://api.githubcopilot.com/mcp/"})

    with pytest.raises(MCPConfigError, match="more than 1 tools"):
        asyncio.run(mcp_tools._list_server_tools("github", config))


def test_normalize_mcp_tool_result_returns_json_safe_shape():
    result = CallToolResult(
        content=[TextContent(type="text", text="hello")],
        structuredContent={"ok": True},
    )

    normalized = normalize_mcp_tool_result(result)

    assert normalized == MCPToolResult(
        content=[{"type": "text", "text": "hello"}],
        structured_content={"ok": True},
    )
    assert ToolPodExecuteResponse(status="success", result=normalized).model_dump(mode="json")[
        "result"
    ] == {
        "content": [{"type": "text", "text": "hello"}],
        "structured_content": {"ok": True},
        "is_error": False,
    }


def test_normalize_mcp_tool_result_raises_for_mcp_error():
    result = CallToolResult(
        content=[TextContent(type="text", text="bad input")],
        isError=True,
    )

    with pytest.raises(MCPToolError, match="bad input"):
        normalize_mcp_tool_result(result)


def test_normalize_mcp_tool_result_rejects_non_object_structured_content():
    result = SimpleNamespace(
        content=[TextContent(type="text", text="hello")],
        structuredContent=["not", "an", "object"],
        isError=False,
    )

    with pytest.raises(MCPToolError, match="not an object"):
        normalize_mcp_tool_result(result)


def test_normalize_mcp_tool_result_rejects_non_json_structured_content():
    result = SimpleNamespace(
        content=[TextContent(type="text", text="hello")],
        structuredContent={"bad": object()},
        isError=False,
    )

    with pytest.raises(MCPToolError, match="non-JSON-serializable structuredContent"):
        normalize_mcp_tool_result(result)
