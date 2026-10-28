"""Framework-neutral helpers for remote MCP tool discovery and execution."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypeVar, cast
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from agent_engine_runner_shared.agent_config import (
    RuntimeMCPConfig,
    RuntimeMCPServerConfig,
)

_UNSAFE_TOOL_NAME_CHARS_RE = re.compile(r"[^a-zA-Z0-9_]")
_MAX_LIST_TOOL_PAGES = 100
_MAX_TOOLS_PER_SERVER = 10_000
# Transport messages that add nothing beyond an HTTP status we already report.
_GENERIC_MCP_ERRORS = frozenset({"Server returned an error response"})
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
    annotations: dict[str, Any] | None = None


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


@dataclass(frozen=True)
class _MCPHTTPStatus:
    """The HTTP status of a failed response, kept for the caller's error text.

    The 2.x transport reports a non-2xx response as a JSON-RPC error and drops
    the status, so the response hook is the only remaining source for messages
    such as ``HTTP 401 Unauthorized``.
    """

    status_code: int
    reason_phrase: str


def _make_mcp_http_client(
    config: RuntimeMCPServerConfig,
    *,
    headers: dict[str, str],
    auth: Any | None,
    responses: list[_MCPHTTPStatus],
) -> Any:
    """Build the httpx2 client the MCP transport uses for one operation.

    Redirect following stays off so the transport keeps owning same-origin
    redirects, and the response hook records failed statuses for error text.
    """

    import httpx2

    timeout = float(config.timeout_seconds)

    async def record_status(response: httpx2.Response) -> None:
        # httpx2 awaits response hooks, so this must be a coroutine.
        #
        # Only JSON-RPC POSTs describe the operation. The SDK also sends a
        # session DELETE while the client closes and tolerates its 405; taking
        # that status would report the cleanup instead of the failed call.
        if response.request.method != "POST":
            return
        if response.status_code >= 400:
            responses.append(_MCPHTTPStatus(response.status_code, response.reason_phrase))
        else:
            # A later successful POST means an earlier probe or authentication
            # failure recovered, so it must not be reused as this call's status.
            responses.clear()

    return httpx2.AsyncClient(
        headers=headers,
        timeout=httpx2.Timeout(timeout, read=timeout),
        auth=auth,
        event_hooks={"response": [record_status]},
    )


@asynccontextmanager
async def _open_mcp_client(
    config: RuntimeMCPServerConfig,
    *,
    headers: dict[str, str],
    auth: Any | None,
    responses: list[_MCPHTTPStatus],
) -> AsyncIterator[Any]:
    """Open one MCP client session with the SDK's automatic era negotiation.

    ``mode="auto"`` probes ``server/discover`` and falls back to the legacy
    ``initialize`` handshake, so servers on ``2025-11-25`` and older keep
    working while ``2026-07-28`` servers are reached directly.
    """

    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    timeout = float(config.timeout_seconds)
    http_client = _make_mcp_http_client(
        config,
        headers=headers,
        auth=auth,
        responses=responses,
    )
    # The transport never closes a caller-owned client, so the session owns it.
    async with http_client:
        client = Client(
            streamable_http_client(config.url, http_client=http_client),
            read_timeout_seconds=timeout,
            mode="auto",
        )
        async with client as connected:
            yield connected


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
                    input_schema=dict(tool.input_schema),
                    server_config=server_config,
                    annotations=normalize_mcp_tool_annotations(getattr(tool, "annotations", None)),
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

    headers = resolve_mcp_headers(binding.server_name, binding.server_config)
    auth = resolve_mcp_auth(binding.server_name, binding.server_config)
    responses: list[_MCPHTTPStatus] = []

    failure: tuple[BaseException, _MCPHTTPStatus | None] | None = None
    operation = f"tool {binding.tool_name!r}"
    try:
        # TODO: Reuse MCP sessions once OAuth token lifecycle support can
        # safely own long-lived remote clients across calls.
        async with _open_mcp_client(
            binding.server_config,
            headers=headers,
            auth=auth,
            responses=responses,
        ) as client:
            try:
                result = await client.call_tool(binding.tool_name, arguments)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Read the status here, before the client's context exits: a
                # cleanup token refresh or session DELETE can otherwise replace
                # the failed operation's status.
                failure = (exc, _last_http_status(responses))
            else:
                return normalize_mcp_tool_result(result)
    except MCPToolError:
        raise
    except MCPConfigError:
        raise
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # The client's task group can regroup the captured failure with a
        # teardown error; prefer the operation's own failure when we have one.
        cause, status = failure if failure is not None else (exc, _last_http_status(responses))
        raise MCPToolError(
            _mcp_transport_error_message(binding.server_name, operation, cause, status)
        ) from cause

    # Raise outside the client context so the SDK cannot regroup the error.
    if failure is not None:
        cause, status = failure
        raise MCPToolError(
            _mcp_transport_error_message(binding.server_name, operation, cause, status)
        ) from cause
    raise MCPToolError(f"mcp server {binding.server_name!r} {operation} failed")


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
        # by_alias keeps the wire field names (``mimeType``, not ``mime_type``)
        # that the stored contract and the TypeScript runner use.
        block.model_dump(mode="json", by_alias=True, exclude_none=True)
        if hasattr(block, "model_dump")
        else block
        for block in result.content
    ]
    _ensure_json_safe("content", content)

    structured_content = getattr(result, "structured_content", None)
    if structured_content is not None:
        if not isinstance(structured_content, dict):
            raise MCPToolError("MCP tool returned structured_content that is not an object")
        _ensure_json_safe("structured_content", structured_content)

    normalized = MCPToolResult(
        content=content,
        structured_content=structured_content,
        is_error=bool(result.is_error),
    )

    if result.is_error:
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
    headers = resolve_mcp_headers(server_name, config)
    auth = resolve_mcp_auth(server_name, config)
    responses: list[_MCPHTTPStatus] = []

    failure: tuple[BaseException, _MCPHTTPStatus | None] | None = None
    try:
        async with _open_mcp_client(
            config,
            headers=headers,
            auth=auth,
            responses=responses,
        ) as client:
            tools: list[Any] = []
            cursor: str | None = None
            for _ in range(_MAX_LIST_TOOL_PAGES):
                try:
                    result = await client.list_tools(cursor=cursor)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # Capture before context exit; see _call_mcp_tool.
                    failure = (exc, _last_http_status(responses))
                    break
                tools.extend(result.tools)
                if len(tools) > _MAX_TOOLS_PER_SERVER:
                    raise MCPConfigError(
                        f"mcp server {server_name!r} returned more than "
                        f"{_MAX_TOOLS_PER_SERVER} tools"
                    )
                cursor = result.next_cursor
                if cursor is None:
                    return tools
            if failure is None:
                raise MCPConfigError(
                    f"mcp server {server_name!r} tools/list exceeded {_MAX_LIST_TOOL_PAGES} pages"
                )
    except MCPConfigError:
        raise
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        cause, status = failure if failure is not None else (exc, _last_http_status(responses))
        raise MCPConfigError(
            _mcp_transport_error_message(server_name, "tools/list", cause, status)
        ) from cause

    # Raise outside the client context so the SDK cannot regroup the error.
    if failure is not None:
        cause, status = failure
        raise MCPConfigError(
            _mcp_transport_error_message(server_name, "tools/list", cause, status)
        ) from cause
    raise MCPConfigError(f"mcp server {server_name!r} tools/list failed")


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


def normalize_mcp_tool_annotations(value: Any) -> dict[str, Any] | None:
    """Return a JSON-safe copy of MCP ``Tool.annotations``, or ``None``.

    Missing or malformed values stay missing. The SDK must not apply MCP client
    defaults (``readOnlyHint: false``, ``destructiveHint: true``, …); applications
    fail closed when hints are absent. Pydantic models are dumped with
    ``exclude_unset=True`` so SDK field defaults are not invented as server hints.
    """

    if value is None:
        return None
    dumped: Any = value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json", by_alias=True, exclude_unset=True)
        except Exception:
            return None
    if not isinstance(dumped, dict):
        return None
    try:
        parsed = json.loads(json.dumps(dumped, allow_nan=False))
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


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


def _mcp_transport_error_message(
    server_name: str,
    operation: str,
    exc: BaseException,
    http_status: _MCPHTTPStatus | None = None,
) -> str:
    prefix = f"mcp server {server_name!r} {operation} failed"
    leaf_messages = _leaf_exception_messages(exc)

    if http_status is not None:
        reason = http_status.reason_phrase or "HTTP error"
        message = f"{prefix}: HTTP {http_status.status_code} {reason}"
        # Keep provider detail such as an OAuth token error's scope explanation.
        # The transport's own "Server returned an error response" says nothing
        # beyond the status we already report.
        actionable = [item for item in leaf_messages if item not in _GENERIC_MCP_ERRORS]
        if actionable:
            message = f"{message}: {'; '.join(actionable)}"
        return message

    if leaf_messages:
        return f"{prefix}: {'; '.join(leaf_messages)}"
    return prefix


def _last_http_status(responses: list[_MCPHTTPStatus]) -> _MCPHTTPStatus | None:
    """Return the last failed HTTP status seen during one MCP operation."""

    return responses[-1] if responses else None


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
    "normalize_mcp_tool_annotations",
    "normalize_mcp_tool_result",
    "resolve_mcp_auth",
    "resolve_mcp_headers",
    "resolve_configured_mcp_tool_binding",
]
