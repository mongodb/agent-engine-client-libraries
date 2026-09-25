"""OAuth provider and file-backed token storage for remote MCP servers."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import AsyncGenerator
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, cast

import httpx
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.auth import oauth2 as mcp_oauth2
from mcp.client.auth.extensions.client_credentials import ClientCredentialsOAuthProvider
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    create_oauth_metadata_request,
    extract_field_from_www_auth,
    extract_scope_from_www_auth,
    handle_auth_metadata_response,
)
from mcp.client.auth.utils import (
    get_client_metadata_scopes as _sdk_get_client_metadata_scopes,
)
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthMetadata,
    OAuthToken,
    ProtectedResourceMetadata,
)

from agent_engine_runner_shared.agent_config import RuntimeMCPServerConfig
from agent_engine_runner_shared.mcp_oauth_cache_lock import mcp_oauth_cache_file_lock
from agent_engine_runner_shared.mcp_oauth_cache_name import mcp_oauth_cache_name

DEFAULT_MCP_OAUTH_CLIENT_NAME = "Atlas Agent Engine Dev MCP Client"
DEFAULT_MCP_OAUTH_REDIRECT_URI = "http://127.0.0.1:8765/callback"
MCP_OAUTH_CACHE_DIR = Path(
    os.environ.get("AGENTIC_MCP_OAUTH_DIR", Path.home() / ".agentic" / "mcp-oauth")
)
_CACHE_SCHEMA_VERSION = 1
_CONFIGURED_OAUTH_SCOPE: ContextVar[str | None] = ContextVar(
    "configured_mcp_oauth_scope",
    default=None,
)


def mcp_oauth_cache_dir() -> Path:
    """Resolve the cache directory after startup secret materialization."""

    return Path(os.environ.get("AGENTIC_MCP_OAUTH_DIR", MCP_OAUTH_CACHE_DIR))


def _agentic_get_client_metadata_scopes(
    www_authenticate_scope: str | None,
    protected_resource_metadata: ProtectedResourceMetadata | None,
    authorization_server_metadata: OAuthMetadata | None = None,
) -> str | None:
    configured_scope = _CONFIGURED_OAUTH_SCOPE.get()
    if configured_scope:
        return configured_scope
    return _sdk_get_client_metadata_scopes(
        www_authenticate_scope,
        protected_resource_metadata,
        authorization_server_metadata,
    )


# The MCP SDK's scope selection strategy can replace explicit YAML scopes with
# server-advertised scopes. Keep omitted scopes on the SDK default path, but make
# configured scopes authoritative for this provider.
setattr(mcp_oauth2, "get_client_metadata_scopes", _agentic_get_client_metadata_scopes)


class FileOAuthTokenStorage(TokenStorage):
    """Token storage backed by the shared ``agentengine dev`` cache file."""

    def __init__(
        self,
        server_name: str,
        server_url: str,
        cache_dir: Path | None = None,
        client_id: str | None = None,
        configured_scope: str | None = None,
    ) -> None:
        self.server_name = server_name
        self.server_url = server_url
        self.cache_dir = cache_dir or mcp_oauth_cache_dir()
        self.client_id = client_id
        self.configured_scope = configured_scope

    @property
    def path(self) -> Path:
        return self.cache_dir / f"{mcp_oauth_cache_name(self.server_name, self.server_url)}.json"

    async def get_tokens(self) -> OAuthToken | None:
        payload = self._read_payload()
        if not self._payload_matches(payload):
            return None
        token_data = payload.get("tokens")
        if not isinstance(token_data, dict) or not token_data.get("access_token"):
            return None
        expires_in = _seconds_until_expiry(token_data.get("expiry"))
        if expires_in is None and isinstance(token_data.get("expires_in"), int):
            expires_in = token_data["expires_in"]
        return OAuthToken(
            access_token=str(token_data["access_token"]),
            token_type=token_data.get("token_type") or "Bearer",
            refresh_token=token_data.get("refresh_token"),
            expires_in=expires_in,
            scope=token_data.get("scope"),
        )

    async def set_tokens(self, tokens: OAuthToken) -> None:
        await asyncio.to_thread(self._update_payload, tokens=_dump_tokens(tokens))

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        payload = self._read_payload()
        if not self._payload_matches(payload):
            return None
        client_data = payload.get("client")
        if not isinstance(client_data, dict) or not client_data.get("client_id"):
            return None
        return OAuthClientInformationFull.model_validate(
            {
                "redirect_uris": client_data.get("redirect_uris")
                or [DEFAULT_MCP_OAUTH_REDIRECT_URI],
                "token_endpoint_auth_method": client_data.get("token_endpoint_auth_method"),
                "grant_types": client_data.get("grant_types")
                or ["authorization_code", "refresh_token"],
                "response_types": client_data.get("response_types") or ["code"],
                "scope": client_data.get("scope"),
                "client_name": client_data.get("client_name"),
                "client_id": client_data.get("client_id"),
                "client_secret": client_data.get("client_secret"),
                "client_id_issued_at": client_data.get("client_id_issued_at"),
                "client_secret_expires_at": client_data.get("client_secret_expires_at"),
            }
        )

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        await asyncio.to_thread(
            self._update_payload,
            client=client_info.model_dump(mode="json", exclude_none=True),
        )

    def _read_payload(self) -> dict[str, Any]:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload

    def _update_payload(self, **values: Any) -> None:
        """Read-modify-write under an advisory file lock.

        The final write is atomic (tmp file + replace), but without a lock around
        the read-modify-write as a whole, concurrent writers (e.g. two Tool Pod
        invocations refreshing the same MCP server) can clobber each other.
        """
        self.cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        with mcp_oauth_cache_file_lock(self.path):
            payload = self._read_payload()
            payload.update(
                {
                    "schema_version": _CACHE_SCHEMA_VERSION,
                    "server_name": self.server_name,
                    "server_url": self.server_url,
                    "updated_at": _format_datetime(datetime.now(UTC)),
                }
            )
            if self.client_id is not None:
                payload["client_id"] = self.client_id
                payload["configured_scope"] = self.configured_scope
            payload.update(values)
            self._write_payload_unlocked(payload)

    def _payload_matches(self, payload: dict[str, Any]) -> bool:
        cached_server_url = payload.get("server_url")
        if cached_server_url != self.server_url:
            return False
        if self.client_id is not None:
            return (
                payload.get("client_id") == self.client_id
                and payload.get("configured_scope") == self.configured_scope
            )
        return True

    def _write_payload_unlocked(self, payload: dict[str, Any]) -> None:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".tmp-{mcp_oauth_cache_name(self.server_name, self.server_url)}-",
            dir=self.cache_dir,
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, self.path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass


class CachedOAuthClientProvider(OAuthClientProvider):
    """OAuth provider that restores cached expiry before refresh decisions.

    This intentionally reaches into MCP SDK OAuth provider internals; keep the
    `mcp` dependency pinned and the refresh-flow unit tests updated with SDK
    upgrades.
    """

    def __init__(
        self,
        *args: Any,
        configured_scope: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._configured_scope = configured_scope

    async def async_auth_flow(
        self,
        request: httpx.Request,
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        token = None
        if self._configured_scope:
            token = _CONFIGURED_OAUTH_SCOPE.set(self._configured_scope)
        upstream = super().async_auth_flow(request)
        send_response = False
        response: httpx.Response | None = None
        try:
            while True:
                if send_response:
                    if response is None:
                        raise RuntimeError("MCP OAuth flow did not receive an HTTP response")
                    next_request = await upstream.asend(response)
                else:
                    next_request = await anext(upstream)
                    send_response = True
                response = yield next_request
        except StopAsyncIteration:
            return
        finally:
            await upstream.aclose()
            if token is not None:
                _CONFIGURED_OAUTH_SCOPE.reset(token)

    async def _initialize(self) -> None:
        await super()._initialize()
        if self.context.current_tokens is not None:
            self.context.update_token_expiry(self.context.current_tokens)

    async def _refresh_token(self) -> httpx.Request:
        await self._load_oauth_metadata()
        return await super()._refresh_token()

    async def _load_oauth_metadata(self) -> None:
        if self.context.oauth_metadata is not None and self.context.oauth_metadata.token_endpoint:
            return

        urls = build_oauth_authorization_server_metadata_discovery_urls(
            self.context.auth_server_url,
            self.context.server_url,
        )
        async with httpx.AsyncClient(timeout=self.context.timeout) as client:
            for metadata_url in urls:
                response = await client.send(create_oauth_metadata_request(metadata_url))
                ok, metadata = await handle_auth_metadata_response(response)
                if metadata is not None:
                    self.context.oauth_metadata = metadata
                    self.context.auth_server_url = str(metadata.issuer)
                    return
                if not ok:
                    continue


class CachedClientCredentialsOAuthProvider(ClientCredentialsOAuthProvider):
    """SDK client-credentials provider with Agent Engine cache and scope behavior."""

    def __init__(
        self,
        *args: Any,
        scopes: str | None = None,
        timeout: float = 300.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, scopes=scopes, **kwargs)
        self._configured_scope = scopes
        self.context.timeout = timeout

    async def _perform_authorization(self) -> httpx.Request:
        if self._configured_scope:
            # MCP SDK discovery can replace constructor scopes before the token
            # request. Keep explicit agent.yaml scopes authoritative.
            self.context.client_metadata.scope = self._configured_scope
        return await super()._perform_authorization()

    async def _initialize(self) -> None:
        await super()._initialize()
        if self.context.current_tokens is not None:
            self.context.update_token_expiry(self.context.current_tokens)


class DirectClientCredentialsOAuthProvider(CachedClientCredentialsOAuthProvider):
    """SDK client-credentials provider for servers that publish a token URL out of band.

    Atlas's current Remote MCP service-account flow gives clients the token
    endpoint directly instead of requiring the discovery-first 401 flow. Keep
    the token request/response handling in the MCP SDK; this adapter only
    supplies the configured endpoint and obtains the first token proactively.
    """

    def __init__(
        self,
        *,
        server_url: str,
        token_url: str,
        storage: TokenStorage,
        client_id: str,
        client_secret: str,
        scopes: str | None,
        timeout: float,
    ) -> None:
        super().__init__(
            server_url=server_url,
            storage=storage,
            client_id=client_id,
            client_secret=client_secret,
            token_endpoint_auth_method="client_secret_basic",
            scopes=scopes,
            timeout=timeout,
        )
        self._token_url = token_url

    def _get_token_endpoint(self) -> str:
        return self._token_url

    async def async_auth_flow(
        self,
        request: httpx.Request,
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        token_request: httpx.Request | None = None
        async with self.context.lock:
            if not self._initialized:
                await self._initialize()

            if self.context.is_token_valid():
                self._add_auth_header(request)
            else:
                token_request = await self._perform_authorization()

        if token_request is not None:
            token_response = yield token_request
            async with self.context.lock:
                await self._handle_token_response(token_response)
                self._add_auth_header(request)

        response = yield request

        if response.status_code == 401:
            async with self.context.lock:
                self.context.clear_tokens()
                token_request = await self._perform_authorization()

            token_response = yield token_request
            async with self.context.lock:
                await self._handle_token_response(token_response)
                self._add_auth_header(request)
            yield request
        elif (
            response.status_code == 403
            and extract_field_from_www_auth(response, "error") == "insufficient_scope"
        ):
            async with self.context.lock:
                self.context.client_metadata.scope = _agentic_get_client_metadata_scopes(
                    extract_scope_from_www_auth(response),
                    self.context.protected_resource_metadata,
                )
                token_request = await self._perform_authorization()

            token_response = yield token_request
            async with self.context.lock:
                await self._handle_token_response(token_response)
                self._add_auth_header(request)
            yield request


def make_mcp_oauth_auth(
    server_name: str,
    config: RuntimeMCPServerConfig,
    *,
    cache_dir: Path | None = None,
) -> OAuthClientProvider:
    """Return a non-interactive OAuth provider for a configured MCP server."""

    login_required = _raise_interactive_login_required(server_name)
    return _make_oauth_provider(
        server_name,
        config,
        cache_dir=cache_dir,
        redirect_uri=config.auth.redirect_uri or DEFAULT_MCP_OAUTH_REDIRECT_URI,
        redirect_handler=login_required,
        callback_handler=login_required,
    )


def make_mcp_client_credentials_auth(
    server_name: str,
    config: RuntimeMCPServerConfig,
    *,
    cache_dir: Path | None = None,
) -> OAuthClientProvider:
    """Return a client-credentials OAuth provider for a configured MCP server."""

    auth = config.auth
    client_id = _required_env_value(server_name, "client_id_env", cast(str, auth.client_id_env))
    client_secret = _required_env_value(
        server_name,
        "client_secret_env",
        cast(str, auth.client_secret_env),
    )
    storage = FileOAuthTokenStorage(
        server_name,
        config.url,
        cache_dir=cache_dir,
        client_id=client_id,
        configured_scope=auth.scope,
    )

    if auth.token_url is None:
        return CachedClientCredentialsOAuthProvider(
            server_url=config.url,
            storage=storage,
            client_id=client_id,
            client_secret=client_secret,
            token_endpoint_auth_method="client_secret_basic",
            scopes=auth.scope,
            timeout=float(config.timeout_seconds),
        )

    return DirectClientCredentialsOAuthProvider(
        server_url=config.url,
        token_url=cast(str, auth.token_url),
        storage=storage,
        client_id=client_id,
        client_secret=client_secret,
        scopes=auth.scope,
        timeout=float(config.timeout_seconds),
    )


def make_interactive_mcp_oauth_auth(
    server_name: str,
    config: RuntimeMCPServerConfig,
    *,
    cache_dir: Path,
    redirect_uri: str,
    redirect_handler: Callable[[str], Awaitable[None]],
    callback_handler: Callable[[], Awaitable[tuple[str, str | None]]],
) -> OAuthClientProvider:
    """Return an interactive OAuth provider used by the one-shot auth helper."""

    return _make_oauth_provider(
        server_name,
        config,
        cache_dir=cache_dir,
        redirect_uri=redirect_uri,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )


def _make_oauth_provider(
    server_name: str,
    config: RuntimeMCPServerConfig,
    *,
    cache_dir: Path | None,
    redirect_uri: str,
    redirect_handler: Any,
    callback_handler: Any,
) -> OAuthClientProvider:
    storage = FileOAuthTokenStorage(server_name, config.url, cache_dir=cache_dir)
    return CachedOAuthClientProvider(
        server_url=config.url,
        client_metadata=_client_metadata(config, redirect_uri),
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
        timeout=float(config.timeout_seconds),
        configured_scope=config.auth.scope,
    )


def _client_metadata(config: RuntimeMCPServerConfig, redirect_uri: str) -> OAuthClientMetadata:
    return OAuthClientMetadata.model_validate(
        {
            "redirect_uris": [redirect_uri],
            "client_name": config.auth.client_name or DEFAULT_MCP_OAUTH_CLIENT_NAME,
            "scope": config.auth.scope,
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            # A loopback client cannot keep a client secret. Omitting this lets
            # servers apply RFC 7591's client_secret_basic default and mint a
            # secret the token exchange then has no way to send.
            "token_endpoint_auth_method": "none",
        }
    )


def _dump_tokens(tokens: OAuthToken) -> dict[str, Any]:
    token_data = tokens.model_dump(mode="json", exclude_none=True)
    expires_in = token_data.get("expires_in")
    if isinstance(expires_in, int):
        expiry = datetime.now(UTC) + timedelta(seconds=expires_in)
        token_data["expiry"] = _format_datetime(expiry)
        token_data.pop("expires_in", None)
    return token_data


def _seconds_until_expiry(raw_expiry: Any) -> int | None:
    if not isinstance(raw_expiry, str) or raw_expiry == "":
        return None
    expiry = _parse_datetime(raw_expiry)
    if expiry is None:
        return None
    return max(0, int((expiry - datetime.now(UTC)).total_seconds()))


def _parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _raise_interactive_login_required(server_name: str) -> Callable[..., Awaitable[Any]]:
    async def handler(*_args: Any) -> Any:
        raise RuntimeError(
            f"mcp server {server_name!r} requires cached OAuth credentials; "
            f"run 'agentengine dev mcp auth login {server_name}'"
        )

    return handler


def _required_env_value(server_name: str, field_name: str, env_name: str) -> str:
    value = os.environ.get(env_name)
    if value is None or value.strip() == "":
        raise ValueError(
            f"mcp server {server_name!r} auth.{field_name} references missing env var {env_name}"
        )
    return value.strip()


__all__ = [
    "DEFAULT_MCP_OAUTH_CLIENT_NAME",
    "DEFAULT_MCP_OAUTH_REDIRECT_URI",
    "CachedOAuthClientProvider",
    "CachedClientCredentialsOAuthProvider",
    "DirectClientCredentialsOAuthProvider",
    "FileOAuthTokenStorage",
    "MCP_OAUTH_CACHE_DIR",
    "mcp_oauth_cache_dir",
    "make_mcp_client_credentials_auth",
    "make_interactive_mcp_oauth_auth",
    "make_mcp_oauth_auth",
    "mcp_oauth_cache_name",
]
