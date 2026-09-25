"""One-shot MCP OAuth login and status command used by ``agentengine dev mcp auth``."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO
from urllib.parse import ParseResult, parse_qs, urlparse

import httpx

from agent_engine_runner_shared.agent_config import (
    RuntimeMCPServerConfig,
    load_runtime_agent_config,
)
from agent_engine_runner_shared.mcp_oauth import (
    DEFAULT_MCP_OAUTH_REDIRECT_URI,
    FileOAuthTokenStorage,
    make_interactive_mcp_oauth_auth,
    make_mcp_oauth_auth,
)
from agent_engine_runner_shared.mcp_tools import resolve_mcp_headers
from agent_engine_runner_shared.utils import tenant_env_vars

OPEN_URL_EVENT_PREFIX = "__AGENTIC_MCP_OAUTH_OPEN_URL__"


@dataclass(frozen=True)
class OAuthCacheStatus:
    path: Path
    exists: bool
    expired: bool = False
    expires_at: datetime | None = None
    has_refresh_token: bool = False


def load_oauth_server_config(agent_path: str | Path, server_name: str) -> RuntimeMCPServerConfig:
    """Load the named MCP server and ensure it is configured for OAuth."""

    config = load_runtime_agent_config(agent_path, env_vars=tenant_env_vars())
    server = config.mcp.servers.get(server_name)
    if server is None:
        raise ValueError(f"mcp server {server_name!r} is not configured")
    if server.auth.type != "oauth":
        raise ValueError(
            f"mcp server {server_name!r} uses auth.type {server.auth.type!r}, not oauth"
        )
    return server


def format_open_url_event(url: str, *, server_name: str) -> str:
    """Format the machine-readable line used to open a browser."""

    payload = json.dumps(
        {"server_name": server_name, "url": url},
        sort_keys=True,
    )
    return f"{OPEN_URL_EVENT_PREFIX} {payload}"


async def login_oauth_server(
    server_name: str,
    config: RuntimeMCPServerConfig,
    *,
    cache_dir: Path,
    redirect_uri: str,
    out: TextIO | None = None,
) -> list[str]:
    """Run the interactive OAuth flow and persist refreshed credentials."""

    output = out or sys.stdout
    parsed_redirect = urlparse(redirect_uri)
    async with LoopbackCallbackServer(parsed_redirect) as callback_server:

        async def redirect_handler(authorization_url: str) -> None:
            print(
                format_open_url_event(authorization_url, server_name=server_name),
                file=output,
                flush=True,
            )

        auth = make_interactive_mcp_oauth_auth(
            server_name,
            config,
            cache_dir=cache_dir,
            redirect_uri=redirect_uri,
            redirect_handler=redirect_handler,
            callback_handler=callback_server.wait,
        )
        return await list_oauth_tools(server_name, config, auth=auth)


async def status_oauth_server(
    server_name: str,
    config: RuntimeMCPServerConfig,
    *,
    cache_dir: Path,
    display_cache_dir: Path | None = None,
    check: bool = False,
    out: TextIO | None = None,
) -> None:
    """Print cache status, optionally validating cached credentials with ``tools/list``."""

    output = out or sys.stdout
    status = read_cache_status(server_name, config.url, cache_dir, now=datetime.now(UTC))
    print(
        format_cache_status(server_name, status, display_cache_dir=display_cache_dir), file=output
    )
    if not check:
        return
    if not status.exists:
        raise RuntimeError(
            f"{server_name} check failed: run 'agentengine dev mcp auth login {server_name}'"
        )
    if status.expired and not status.has_refresh_token:
        raise RuntimeError(
            f"{server_name} check failed: access token expired and no refresh token is cached; "
            f"run 'agentengine dev mcp auth login {server_name}'"
        )

    auth = make_mcp_oauth_auth(server_name, config, cache_dir=cache_dir)
    tools = await list_oauth_tools(server_name, config, auth=auth)
    print(f"{server_name} check ok ({len(tools)} tool(s))", file=output)


def read_cache_status(
    server_name: str,
    server_url: str,
    cache_dir: Path,
    *,
    now: datetime,
) -> OAuthCacheStatus:
    path = FileOAuthTokenStorage(server_name, server_url, cache_dir=cache_dir).path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return OAuthCacheStatus(path=path, exists=False)
    if not isinstance(payload, dict):
        raise ValueError(f"reading MCP OAuth cache {path}: expected a JSON object")

    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        return OAuthCacheStatus(path=path, exists=True)

    expires_at = _parse_cache_datetime(tokens.get("expiry"))
    expired = expires_at is not None and expires_at <= now
    refresh_token = tokens.get("refresh_token")
    return OAuthCacheStatus(
        path=path,
        exists=True,
        expired=expired,
        expires_at=expires_at,
        has_refresh_token=isinstance(refresh_token, str) and refresh_token.strip() != "",
    )


def format_cache_status(
    server_name: str,
    status: OAuthCacheStatus,
    *,
    display_cache_dir: Path | None = None,
) -> str:
    path = status.path
    if display_cache_dir is not None:
        path = display_cache_dir / status.path.name
    if not status.exists:
        return f"{server_name} not authenticated (cache: {path})"
    if status.expired and status.has_refresh_token and status.expires_at is not None:
        return (
            f"{server_name} access token expired at {_format_cache_datetime(status.expires_at)}; "
            f"refresh token cached (cache: {path})"
        )
    if status.expired and status.expires_at is not None:
        return (
            f"{server_name} expired at {_format_cache_datetime(status.expires_at)} (cache: {path})"
        )
    if status.expires_at is None:
        return f"{server_name} authenticated (cache: {path})"
    return f"{server_name} authenticated until {_format_cache_datetime(status.expires_at)} (cache: {path})"


async def list_oauth_tools(
    server_name: str,
    config: RuntimeMCPServerConfig,
    *,
    auth: Any,
) -> list[str]:
    """Initialize the MCP session and return tool names."""

    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    headers = resolve_mcp_headers(server_name, config)
    timeout = config.timeout_seconds
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
            result = await session.list_tools()
            return [tool.name for tool in result.tools]


class LoopbackCallbackServer:
    """Small loopback callback server for the MCP OAuth authorization code."""

    def __init__(self, redirect: ParseResult) -> None:
        if redirect.scheme != "http":
            raise ValueError("OAuth redirect URI must use http loopback")
        if not _is_supported_loopback_host(redirect.hostname):
            raise ValueError(
                "OAuth redirect URI host must be localhost or an IPv4 loopback address"
            )
        if redirect.port is None:
            raise ValueError("OAuth redirect URI must include an explicit port")
        self._callback_path = redirect.path or "/"
        # Docker publishes the host loopback port into this container; binding
        # only to container-local 127.0.0.1 would make that forwarded callback
        # unreachable. The host-side bind remains loopback-only.
        self._server = ThreadingHTTPServer(("0.0.0.0", redirect.port), self._handler_class())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._future: asyncio.Future[tuple[str, str | None]] | None = None

    async def __aenter__(self) -> "LoopbackCallbackServer":
        self._loop = asyncio.get_running_loop()
        self._future = self._loop.create_future()
        self._thread.start()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    async def wait(self) -> tuple[str, str | None]:
        if self._future is None:
            raise RuntimeError("OAuth callback server is not running")
        return await self._future

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        owner = self

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                owner._handle_callback(self)

            def log_message(self, format: str, *_args: Any) -> None:
                _ = format
                return

        return CallbackHandler

    def _handle_callback(self, handler: BaseHTTPRequestHandler) -> None:
        if self._future is None or self._loop is None:
            _send_callback_response(
                handler, HTTPStatus.SERVICE_UNAVAILABLE, "OAuth helper is not ready."
            )
            return
        if self._future.done():
            _send_callback_response(handler, HTTPStatus.OK, "OAuth callback already received.")
            return

        parsed = urlparse(handler.path)
        if parsed.path != self._callback_path:
            _send_callback_response(handler, HTTPStatus.NOT_FOUND, "Unknown OAuth callback path.")
            return

        query = parse_qs(parsed.query)
        error = _first_query_value(query, "error")
        if error:
            description = _first_query_value(query, "error_description") or error
            self._loop.call_soon_threadsafe(
                self._future.set_exception,
                RuntimeError(f"OAuth authorization failed: {description}"),
            )
            _send_callback_response(handler, HTTPStatus.BAD_REQUEST, "OAuth authorization failed.")
            return

        code = _first_query_value(query, "code")
        if code is None:
            _send_callback_response(handler, HTTPStatus.BAD_REQUEST, "Missing OAuth code.")
            return

        state = _first_query_value(query, "state")
        self._loop.call_soon_threadsafe(self._future.set_result, (code, state))
        _send_callback_response(handler, HTTPStatus.OK, "OAuth complete. You can close this tab.")


def _first_query_value(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    return values[0]


def _is_supported_loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return ip.version == 4 and ip.is_loopback


def _send_callback_response(
    handler: BaseHTTPRequestHandler,
    status: HTTPStatus,
    message: str,
) -> None:
    body = f"{message}\n".encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _parse_cache_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or value == "":
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_cache_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        server = load_oauth_server_config(args.agent, args.server)
        if args.command == "login":
            redirect_uri = (
                args.redirect_uri or server.auth.redirect_uri or DEFAULT_MCP_OAUTH_REDIRECT_URI
            )
            tools = asyncio.run(
                login_oauth_server(
                    args.server,
                    server,
                    cache_dir=args.cache_dir,
                    redirect_uri=redirect_uri,
                )
            )
            cache_path = FileOAuthTokenStorage(args.server, server.url, args.cache_dir).path
            display_cache_path = cache_path
            if args.display_cache_dir is not None:
                display_cache_path = args.display_cache_dir / cache_path.name
            print(f"Successfully authenticated MCP server {args.server!r}.")
            print(f"Discovered {len(tools)} tool(s).")
            print(f"Credentials saved to {display_cache_path}.")
            return 0

        if args.command == "status":
            asyncio.run(
                status_oauth_server(
                    args.server,
                    server,
                    cache_dir=args.cache_dir,
                    display_cache_dir=args.display_cache_dir,
                    check=args.check,
                )
            )
            return 0

        raise ValueError(f"unsupported command {args.command!r}")
    except Exception as exc:
        message = f"MCP OAuth {args.command} for {args.server!r} failed"
        leaf_errors: list[str] = []
        pending: list[BaseException] = [exc]
        while pending:
            current = pending.pop(0)
            if isinstance(current, BaseExceptionGroup):
                children = [
                    child for child in current.exceptions if isinstance(child, BaseException)
                ]
                pending = children + pending
                continue

            if isinstance(current, httpx.HTTPStatusError):
                response = current.response
                status_detail = f"HTTP {response.status_code} {response.reason_phrase}"
                try:
                    response_body = " ".join(response.text.split())
                except httpx.ResponseNotRead:
                    response_body = ""
                if response_body:
                    status_detail = f"{status_detail}: {response_body}"
                leaf_errors.append(status_detail)
                continue

            detail = str(current).strip() or type(current).__name__
            leaf_errors.append(" ".join(detail.split()))

        unique_errors: list[str] = []
        for detail in leaf_errors:
            if len(detail) > 500:
                detail = detail[:497].rstrip() + "..."
            if detail and detail not in unique_errors:
                unique_errors.append(detail)
        if unique_errors:
            message = f"{message}: {'; '.join(unique_errors)}"
        print(f"Error: {message}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agent_engine_runner_shared.mcp_oauth_cli")
    subcommands = parser.add_subparsers(dest="command", required=True)

    login = subcommands.add_parser("login")
    login.add_argument("--agent", required=True, type=Path)
    login.add_argument("--server", required=True)
    login.add_argument("--cache-dir", required=True, type=Path)
    login.add_argument("--display-cache-dir", type=Path)
    login.add_argument("--redirect-uri")

    status = subcommands.add_parser("status")
    status.add_argument("--agent", required=True, type=Path)
    status.add_argument("--server", required=True)
    status.add_argument("--cache-dir", required=True, type=Path)
    status.add_argument("--display-cache-dir", type=Path)
    status.add_argument("--check", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OPEN_URL_EVENT_PREFIX",
    "OAuthCacheStatus",
    "format_cache_status",
    "format_open_url_event",
    "list_oauth_tools",
    "load_oauth_server_config",
    "login_oauth_server",
    "main",
    "read_cache_status",
    "status_oauth_server",
]
