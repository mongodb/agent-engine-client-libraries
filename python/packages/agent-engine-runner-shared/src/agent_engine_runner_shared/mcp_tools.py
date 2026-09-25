"""Framework-neutral helpers for remote MCP tool discovery and execution."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar, cast
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field

from agent_engine_runner_shared.agent_config import (
    RuntimeMCPConfig,
    RuntimeMCPServerConfig,
)

_UNSAFE_TOOL_NAME_CHARS_RE = re.compile(r"[^a-zA-Z0-9_]")
_MAX_LIST_TOOL_PAGES = 100
_MAX_TOOLS_PER_SERVER = 10_000
_T = TypeVar("_T")


class MCPConfigError(ValueError):
    """Raised when remote MCP configuration cannot be resolved."""


class MCPToolError(RuntimeError):
    """Raised when a remote MCP tool returns an MCP error result."""


@dataclass(frozen=True)
class MCPToolBinding:
    """Discovered MCP tool bound to a configured server."""

    server_name: str
    tool_name: str
    sdk_tool_name: str
    description: str
    input_schema: dict[str, Any]
    server_config: RuntimeMCPServerConfig


class MCPToolResult(BaseModel):
    """JSON/BSON-safe result returned by an MCP ``tools/call`` invocation."""

    model_config = ConfigDict(frozen=True)

    content: list[Any] = Field(
        default_factory=list,
        description="JSON-safe MCP content blocks returned by the remote tool.",
    )
    structured_content: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured MCP result content.",
    )
    is_error: bool = Field(
        default=False,
        description="Whether the MCP tool returned an error result.",
    )


def make_mcp_sdk_tool_name(server_name: str, tool_name: str) -> str:
    """Return the server-prefixed SDK name for a remote MCP tool."""

    server_part = _sanitize_tool_name_part(server_name)
    tool_part = _sanitize_tool_name_part(tool_name)
    return f"{server_part}__{tool_part}"


def resolve_mcp_headers(server_name: str, config: RuntimeMCPServerConfig) -> dict[str, str]:
    """Resolve headers for a remote MCP server without exposing secret values."""

    headers = dict(config.headers)
    if config.auth.type in {"none", "oauth", "client_credentials"}:
        return headers

    token_env = config.auth.token_env
    if token_env is None:
        raise MCPConfigError(f"mcp server {server_name!r} auth.token_env is required")

    token = os.environ.get(token_env)
    if token is None or token.strip() == "":
        raise MCPConfigError(f"mcp server {server_name!r} references missing env var {token_env}")
    headers["Authorization"] = f"Bearer {token.strip()}"
    return headers


def resolve_mcp_auth(server_name: str, config: RuntimeMCPServerConfig) -> Any | None:
    """Resolve an MCP SDK auth provider for the configured auth mode."""

    if config.auth.type == "none" or config.auth.type == "bearer_env":
        return None

    from agent_engine_runner_shared.mcp_oauth import (
        make_mcp_client_credentials_auth,
        make_mcp_oauth_auth,
    )

    try:
        if config.auth.type == "oauth":
            return make_mcp_oauth_auth(server_name, config)
        if config.auth.type == "client_credentials":
            return make_mcp_client_credentials_auth(server_name, config)
    except ValueError as exc:
        raise MCPConfigError(str(exc)) from exc

    raise MCPConfigError(f"mcp server {server_name!r} has unsupported auth.type {config.auth.type}")


def mcp_server_network_hosts(config: RuntimeMCPServerConfig) -> list[str]:
    """Return the outbound hosts needed to reach a configured MCP server."""

    hosts = [cast(str, urlparse(config.url).hostname)]
    if config.auth.token_url is not None:
        token_host = cast(str, urlparse(config.auth.token_url).hostname)
        if token_host not in hosts:
            hosts.append(token_host)
    return hosts


def discover_mcp_tools(config: RuntimeMCPConfig) -> list[MCPToolBinding]:
    """Discover remote MCP tools for the synchronous ``App`` constructor path.

    This mirrors common MCP client adapters: initialize each server, call
    ``tools/list``, then expose the returned schemas as framework-native tools.
    """

    return _run_coroutine_sync(
        lambda: discover_mcp_tools_async(config),
        active_loop_error=MCPConfigError(
            "cannot discover MCP tools from a running event loop; construct App outside "
            "async startup or use discover_mcp_tools_async"
        ),
    )


async def discover_mcp_tools_async(config: RuntimeMCPConfig) -> list[MCPToolBinding]:
    """Async implementation because the MCP Python client is async-first."""

    bindings: list[MCPToolBinding] = []
    used_names: dict[str, str] = {}

    for server_name, server_config in config.servers.items():
        tools = await _list_server_tools(server_name, server_config)
        available_tool_names = {tool.name for tool in tools}
        allowed_tools = server_config.allowed_tools

        if allowed_tools is not None:
            missing = sorted(set(allowed_tools) - available_tool_names)
            if missing:
                missing_list = ", ".join(missing)
                raise MCPConfigError(
                    f"mcp server {server_name!r} did not expose configured "
                    f"allowed_tools: {missing_list}"
                )
            tools = [tool for tool in tools if tool.name in allowed_tools]

        for tool in tools:
            sdk_tool_name = make_mcp_sdk_tool_name(server_name, tool.name)
            previous = used_names.get(sdk_tool_name)
            if previous is not None:
                raise MCPConfigError(
                    f"mcp tool name collision for {sdk_tool_name!r}: "
                    f"{previous} and {server_name}.{tool.name}"
                )
            used_names[sdk_tool_name] = f"{server_name}.{tool.name}"
            bindings.append(
                MCPToolBinding(
                    server_name=server_name,
                    tool_name=tool.name,
                    sdk_tool_name=sdk_tool_name,
                    description=tool.description or "",
                    input_schema=dict(tool.inputSchema),
                    server_config=server_config,
                )
            )

    return bindings


def resolve_configured_mcp_tool_binding(
    config: RuntimeMCPConfig,
    sdk_tool_name: str,
    mcp_server_name: str,
    mcp_tool_name: str,
) -> MCPToolBinding | None:
    """Map an SDK-visible tool name back to configured MCP call metadata.

    AER discovers remote MCP schemas at startup and exposes server-prefixed
    names such as ``github__search_issues`` to the LLM. Tool Pods skip
    ``tools/list`` startup discovery, so call-time execution resolves that SDK
    name against ``agent.yaml`` and calls the original MCP tool name.
    """

    server_config = config.servers.get(mcp_server_name)
    if server_config is None:
        return None

    expected_sdk_tool_name = make_mcp_sdk_tool_name(mcp_server_name, mcp_tool_name)
    if expected_sdk_tool_name != sdk_tool_name:
        raise MCPConfigError(
            f"mcp call metadata for {mcp_server_name!r}.{mcp_tool_name!r} "
            f"does not match {sdk_tool_name!r}"
        )

    if server_config.allowed_tools is not None and mcp_tool_name not in server_config.allowed_tools:
        raise MCPConfigError(
            f"mcp server {mcp_server_name!r} tool {mcp_tool_name!r} is not in allowed_tools"
        )

    return MCPToolBinding(
        server_name=mcp_server_name,
        tool_name=mcp_tool_name,
        sdk_tool_name=sdk_tool_name,
        description="",
        input_schema={},
        server_config=server_config,
    )


def is_configured_mcp_sdk_tool_name(config: RuntimeMCPConfig, sdk_tool_name: str) -> bool:
    """Return whether ``sdk_tool_name`` belongs to a configured MCP server."""

    for server_name in config.servers:
        server_prefix = _sanitize_tool_name_part(server_name) + "__"
        if sdk_tool_name.startswith(server_prefix) and len(sdk_tool_name) > len(server_prefix):
            return True
    return False


async def _call_mcp_tool(binding: MCPToolBinding, arguments: dict[str, Any]) -> MCPToolResult:
    """Execute a remote MCP tool and normalize its result for Go OE persistence."""

    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    headers = resolve_mcp_headers(binding.server_name, binding.server_config)
    auth = resolve_mcp_auth(binding.server_name, binding.server_config)
    timeout = binding.server_config.timeout_seconds

    try:
        # TODO: Reuse MCP sessions once OAuth token lifecycle support can
        # safely own long-lived remote clients across calls.
        async with streamablehttp_client(
            binding.server_config.url,
            headers=headers,
            timeout=timeout,
            sse_read_timeout=timeout,
            auth=auth,
        ) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(binding.tool_name, arguments)
                return normalize_mcp_tool_result(result)
    except MCPToolError:
        raise
    except MCPConfigError:
        raise
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        operation = f"tool {binding.tool_name!r}"
        message = _mcp_transport_error_message(binding.server_name, operation, exc)
        raise MCPToolError(message) from exc


def make_mcp_tool_pod_callable(binding: MCPToolBinding) -> Callable[..., Any]:
    """Create the async callable executed by the Tool Pod for MCP tool calls."""

    async def remote_mcp_tool(**kwargs: Any) -> Any:
        return await _call_mcp_tool(binding, kwargs)

    remote_mcp_tool.__name__ = binding.sdk_tool_name
    remote_mcp_tool.__doc__ = binding.description
    return remote_mcp_tool


def make_sync_mcp_tool_callable(binding: MCPToolBinding) -> Callable[..., Any]:
    """Create the sync callable LangChain uses for direct ``tool.invoke()`` calls."""

    def direct_mcp_tool(**kwargs: Any) -> Any:
        return _run_coroutine_sync(
            lambda: _call_mcp_tool(binding, kwargs),
            active_loop_error=MCPToolError(
                "cannot invoke sync MCP tool from a running event loop; use the "
                "LangChain async tool path instead"
            ),
        )

    direct_mcp_tool.__name__ = binding.sdk_tool_name
    direct_mcp_tool.__doc__ = binding.description
    return direct_mcp_tool


def normalize_mcp_tool_result(result: Any) -> MCPToolResult:
    """Convert an MCP ``CallToolResult`` into a JSON/BSON-safe model."""

    content = [
        block.model_dump(mode="json", by_alias=False, exclude_none=True)
        if hasattr(block, "model_dump")
        else block
        for block in result.content
    ]
    _ensure_json_safe("content", content)

    structured_content = getattr(result, "structuredContent", None)
    if structured_content is not None:
        if not isinstance(structured_content, dict):
            raise MCPToolError("MCP tool returned structuredContent that is not an object")
        _ensure_json_safe("structuredContent", structured_content)

    normalized = MCPToolResult(
        content=content,
        structured_content=structured_content,
        is_error=bool(result.isError),
    )

    if result.isError:
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    text_parts.append(text)
        if text_parts:
            raise MCPToolError("MCP tool returned error: " + "\n".join(text_parts))
        raise MCPToolError("MCP tool returned an error")

    return normalized


async def _list_server_tools(server_name: str, config: RuntimeMCPServerConfig) -> list[Any]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    headers = resolve_mcp_headers(server_name, config)
    auth = resolve_mcp_auth(server_name, config)
    timeout = config.timeout_seconds

    try:
        async with streamablehttp_client(
            config.url,
            headers=headers,
            timeout=timeout,
            sse_read_timeout=timeout,
            auth=auth,
        ) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools: list[Any] = []
                cursor: str | None = None
                for _ in range(_MAX_LIST_TOOL_PAGES):
                    result = await session.list_tools(cursor=cursor)
                    tools.extend(result.tools)
                    if len(tools) > _MAX_TOOLS_PER_SERVER:
                        raise MCPConfigError(
                            f"mcp server {server_name!r} returned more than "
                            f"{_MAX_TOOLS_PER_SERVER} tools"
                        )
                    cursor = result.nextCursor
                    if cursor is None:
                        return tools
                raise MCPConfigError(
                    f"mcp server {server_name!r} tools/list exceeded {_MAX_LIST_TOOL_PAGES} pages"
                )
    except MCPConfigError:
        raise
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        message = _mcp_transport_error_message(server_name, "tools/list", exc)
        raise MCPConfigError(message) from exc


def _run_coroutine_sync(
    coroutine_factory: Callable[[], Coroutine[Any, Any, _T]],
    *,
    active_loop_error: Exception,
) -> _T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine_factory())
    raise active_loop_error


def _ensure_json_safe(field_name: str, value: Any) -> None:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise MCPToolError(f"MCP tool returned non-JSON-serializable {field_name}") from exc


def _sanitize_tool_name_part(value: str) -> str:
    # Provider names can contain punctuation that downstream LLM tool-call APIs
    # reject, so the SDK exposes a stable identifier and keeps the original MCP
    # tool name only for the outbound tools/call request.
    sanitized = _UNSAFE_TOOL_NAME_CHARS_RE.sub("_", value.strip())
    sanitized = sanitized.strip("_")
    if sanitized == "":
        raise MCPConfigError("mcp server and tool names must contain alphanumeric characters")
    if sanitized[0].isdigit():
        sanitized = f"mcp_{sanitized}"
    return sanitized


def _mcp_transport_error_message(server_name: str, operation: str, exc: BaseException) -> str:
    http_error = _find_http_status_error(exc)
    if http_error is not None:
        response = http_error.response
        status = response.status_code
        reason = response.reason_phrase or "HTTP error"
        return f"mcp server {server_name!r} {operation} failed: HTTP {status} {reason}"

    leaf_messages = _leaf_exception_messages(exc)
    if leaf_messages:
        return f"mcp server {server_name!r} {operation} failed: {'; '.join(leaf_messages)}"
    return f"mcp server {server_name!r} {operation} failed"


def _find_http_status_error(exc: BaseException) -> httpx.HTTPStatusError | None:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        for child in exc.exceptions:
            found = _find_http_status_error(child)
            if found is not None:
                return found
    return None


def _leaf_exception_messages(exc: BaseException) -> list[str]:
    if isinstance(exc, BaseExceptionGroup):
        messages: list[str] = []
        for child in exc.exceptions:
            messages.extend(_leaf_exception_messages(child))
        return messages

    message = str(exc).strip()
    if message:
        return [message]
    return [type(exc).__name__]


__all__ = [
    "MCPConfigError",
    "MCPToolBinding",
    "MCPToolError",
    "MCPToolResult",
    "discover_mcp_tools",
    "discover_mcp_tools_async",
    "is_configured_mcp_sdk_tool_name",
    "make_mcp_sdk_tool_name",
    "make_mcp_tool_pod_callable",
    "make_sync_mcp_tool_callable",
    "mcp_server_network_hosts",
    "normalize_mcp_tool_result",
    "resolve_mcp_auth",
    "resolve_mcp_headers",
    "resolve_configured_mcp_tool_binding",
]
