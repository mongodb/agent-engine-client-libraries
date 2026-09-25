from __future__ import annotations

import asyncio
import base64
import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

import agent_engine_runner_shared.mcp_oauth as mcp_oauth
import agent_engine_runner_shared.mcp_oauth_cache_lock as mcp_oauth_cache_lock
from agent_engine_runner_shared.agent_config import RuntimeMCPServerConfig
from agent_engine_runner_shared.mcp_oauth import (
    FileOAuthTokenStorage,
    make_interactive_mcp_oauth_auth,
    make_mcp_client_credentials_auth,
    make_mcp_oauth_auth,
)

_GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"


@dataclass
class _InterleavedRmwSync:
    """Force overlapping read-modify-write windows for deterministic race tests."""

    read_barrier: threading.Barrier = field(default_factory=lambda: threading.Barrier(2))
    write_barrier: threading.Barrier = field(default_factory=lambda: threading.Barrier(2))

    def after_read(self) -> None:
        self.read_barrier.wait(timeout=5)

    def before_write(self) -> None:
        self.write_barrier.wait(timeout=5)


class _NoOpFileLock:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def __enter__(self) -> _NoOpFileLock:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


def _github_storage(cache_dir: Path) -> FileOAuthTokenStorage:
    return FileOAuthTokenStorage(
        "github",
        _GITHUB_MCP_URL,
        cache_dir=cache_dir,
    )


def test_file_oauth_token_storage_path_includes_server_url(tmp_path: Path):
    first = FileOAuthTokenStorage("github", "https://mcp.example.com/one", cache_dir=tmp_path)
    second = FileOAuthTokenStorage("github", "https://mcp.example.com/two", cache_dir=tmp_path)

    assert first.path != second.path
    assert first.path.name == (
        "github-34fcf0941fd4db94f72e35ac8d58114ce17a82c9949c883fa6f2767b5c095c2b.json"
    )


def test_mcp_oauth_cache_name_golden_vectors():
    # Mirrored by the Go (mcpname) and TypeScript parity tests; a drift in any
    # implementation breaks the shared cache layout.
    url = "https://mcp.example.com/one"
    url_hash = "34fcf0941fd4db94f72e35ac8d58114ce17a82c9949c883fa6f2767b5c095c2b"
    vectors = {
        "github": f"github-{url_hash}",
        "github_enterprise": f"github_enterprise-{url_hash}",
        "GitHub": f"GitHub-{url_hash}",
        "github.com": f"github-com-3aeb002460381c6f-{url_hash}",
        "github com": f"github-com-7950713f209fb3ca-{url_hash}",
        " /// ": f"663b6e4b48f7158c-{url_hash}",
        "GitHub Enterprise / Team": f"GitHub-Enterprise-Team-5616f7fc218b4465-{url_hash}",
        "a" * 65: f"{'a' * 40}-635361c48bb9eab1-{url_hash}",
    }
    for server_name, expected in vectors.items():
        assert mcp_oauth.mcp_oauth_cache_name(server_name, url) == expected, server_name


def test_mcp_oauth_cache_name_separates_sanitization_collisions():
    url = "https://mcp.example.com/one"

    assert mcp_oauth.mcp_oauth_cache_name("github.com", url) != mcp_oauth.mcp_oauth_cache_name(
        "github com", url
    )
    assert mcp_oauth.mcp_oauth_cache_name("github.com", url) != mcp_oauth.mcp_oauth_cache_name(
        "github/com", url
    )


async def _write_github_tokens_and_client_info(storage: FileOAuthTokenStorage) -> None:
    await asyncio.gather(
        storage.set_tokens(OAuthToken(access_token="token-a", token_type="Bearer")),
        storage.set_client_info(
            OAuthClientInformationFull.model_validate(
                {
                    "client_id": "client-xyz",
                    "redirect_uris": ["http://127.0.0.1:8765/callback"],
                }
            )
        ),
    )


def _load_github_cache_payload(cache_dir: Path) -> dict[str, Any]:
    cache_name = mcp_oauth.mcp_oauth_cache_name("github", _GITHUB_MCP_URL)
    return json.loads((cache_dir / f"{cache_name}.json").read_text(encoding="utf-8"))


def _patch_interleaved_rmw(monkeypatch: pytest.MonkeyPatch, sync: _InterleavedRmwSync) -> None:
    original_read = FileOAuthTokenStorage._read_payload
    original_write = FileOAuthTokenStorage._write_payload_unlocked

    def patched_read(self: FileOAuthTokenStorage) -> dict[str, Any]:
        payload = original_read(self)
        sync.after_read()
        return payload

    def patched_write(self: FileOAuthTokenStorage, payload: dict[str, Any]) -> None:
        sync.before_write()
        original_write(self, payload)

    monkeypatch.setattr(FileOAuthTokenStorage, "_read_payload", patched_read)
    monkeypatch.setattr(FileOAuthTokenStorage, "_write_payload_unlocked", patched_write)


def _patch_pause_after_token_read(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tokens_read_done: threading.Event,
    release_token_write: threading.Event,
) -> None:
    original_read = FileOAuthTokenStorage._read_payload

    def patched_read(self: FileOAuthTokenStorage) -> dict[str, Any]:
        payload = original_read(self)
        tokens_read_done.set()
        release_token_write.wait(timeout=5)
        return payload

    monkeypatch.setattr(FileOAuthTokenStorage, "_read_payload", patched_read)


@pytest.mark.asyncio
async def test_file_oauth_token_storage_reads_shared_helper_cache(tmp_path: Path):
    expiry = datetime.now(UTC) + timedelta(hours=1)
    storage = _github_storage(tmp_path)
    cache_file = storage.path
    cache_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "server_name": "github",
                "server_url": "https://api.githubcopilot.com/mcp/",
                "client": {
                    "client_id": "client-id",
                    "client_secret": "client-secret",
                    "redirect_uris": ["http://127.0.0.1:1234/callback"],
                    "grant_types": ["authorization_code", "refresh_token"],
                    "response_types": ["code"],
                    "scope": "repo",
                    "client_name": "Atlas Agent Engine Dev MCP Client",
                },
                "tokens": {
                    "access_token": "access-token",
                    "token_type": "Bearer",
                    "refresh_token": "refresh-token",
                    "expires_in": 999999,
                    "expiry": expiry.isoformat().replace("+00:00", "Z"),
                    "scope": "repo",
                },
            }
        ),
        encoding="utf-8",
    )

    tokens = await storage.get_tokens()
    assert tokens is not None
    assert tokens.access_token == "access-token"
    assert tokens.token_type == "Bearer"
    assert tokens.refresh_token == "refresh-token"
    assert tokens.expires_in == pytest.approx(3600, abs=5)
    assert tokens.scope == "repo"
    client = await storage.get_client_info()
    assert client is not None
    assert client.client_id == "client-id"
    assert client.client_secret == "client-secret"
    assert client.redirect_uris is not None
    assert [str(uri) for uri in client.redirect_uris] == ["http://127.0.0.1:1234/callback"]


@pytest.mark.asyncio
async def test_file_oauth_token_storage_ignores_different_server_url(tmp_path: Path):
    expiry = datetime.now(UTC) + timedelta(hours=1)
    old_storage = FileOAuthTokenStorage("github", "https://old.example.com/mcp", cache_dir=tmp_path)
    old_storage.path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "server_name": "github",
                "server_url": "https://old.example.com/mcp",
                "client": {"client_id": "client-id"},
                "tokens": {
                    "access_token": "access-token",
                    "token_type": "Bearer",
                    "expiry": expiry.isoformat().replace("+00:00", "Z"),
                },
            }
        ),
        encoding="utf-8",
    )
    storage = FileOAuthTokenStorage(
        "github",
        "https://api.githubcopilot.com/mcp/",
        cache_dir=tmp_path,
    )

    assert await storage.get_tokens() is None
    assert await storage.get_client_info() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload_identity",
    [
        pytest.param({}, id="missing"),
        pytest.param({"server_url": 42}, id="non-string"),
    ],
)
async def test_file_oauth_token_storage_ignores_invalid_server_url(
    tmp_path: Path,
    payload_identity: dict[str, Any],
):
    expiry = datetime.now(UTC) + timedelta(hours=1)
    payload = {
        "schema_version": 1,
        "server_name": "github",
        "client": {"client_id": "client-id"},
        "tokens": {
            "access_token": "access-token",
            "token_type": "Bearer",
            "expiry": expiry.isoformat().replace("+00:00", "Z"),
        },
    }
    payload.update(payload_identity)
    storage = _github_storage(tmp_path)
    storage.path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    assert await storage.get_tokens() is None
    assert await storage.get_client_info() is None


@pytest.mark.asyncio
async def test_file_oauth_token_storage_writes_refresh_token_update(tmp_path: Path):
    storage = FileOAuthTokenStorage(
        "github",
        "https://api.githubcopilot.com/mcp/",
        cache_dir=tmp_path,
    )
    await storage.set_client_info(
        OAuthClientInformationFull.model_validate(
            {
                "client_id": "client-id",
                "redirect_uris": ["http://127.0.0.1:1234/callback"],
            }
        )
    )
    await storage.set_tokens(
        OAuthToken(
            access_token="new-access-token",
            refresh_token="new-refresh-token",
            expires_in=60,
        )
    )

    payload = json.loads(storage.path.read_text(encoding="utf-8"))
    assert payload["client"]["client_id"] == "client-id"
    assert payload["tokens"]["access_token"] == "new-access-token"
    assert payload["tokens"]["refresh_token"] == "new-refresh-token"
    assert payload["tokens"]["expiry"].endswith("Z")
    assert "expires_in" not in payload["tokens"]


@pytest.mark.asyncio
async def test_concurrent_set_tokens_and_set_client_info_do_not_clobber(tmp_path: Path):
    await _write_github_tokens_and_client_info(_github_storage(tmp_path))

    payload = _load_github_cache_payload(tmp_path)
    assert payload["tokens"]["access_token"] == "token-a"
    assert payload["client"]["client_id"] == "client-xyz"


@pytest.mark.asyncio
async def test_interleaved_rmw_without_file_lock_loses_an_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(mcp_oauth_cache_lock, "FileLock", _NoOpFileLock)
    _patch_interleaved_rmw(monkeypatch, _InterleavedRmwSync())
    await _write_github_tokens_and_client_info(_github_storage(tmp_path))

    payload = _load_github_cache_payload(tmp_path)
    has_tokens = payload.get("tokens", {}).get("access_token") == "token-a"
    has_client = payload.get("client", {}).get("client_id") == "client-xyz"
    assert has_tokens ^ has_client


@pytest.mark.asyncio
async def test_interleaved_materialize_and_token_update_without_cache_lock_loses_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from agent_engine_runner_shared.mcp_oauth_secret import materialize_mcp_oauth_secret_cache

    cache_dir = tmp_path / "cache"
    monkeypatch.delenv("AGENTIC_MCP_OAUTH_DIR", raising=False)
    monkeypatch.setattr(mcp_oauth_cache_lock, "FileLock", _NoOpFileLock)
    tokens_read_done = threading.Event()
    release_token_write = threading.Event()
    _patch_pause_after_token_read(
        monkeypatch,
        tokens_read_done=tokens_read_done,
        release_token_write=release_token_write,
    )

    def materialize_after_token_read() -> None:
        tokens_read_done.wait(timeout=5)
        materialize_mcp_oauth_secret_cache(default_cache_dir=cache_dir)
        release_token_write.set()

    secret_payload = {
        "schema_version": 1,
        "server_name": "github",
        "server_url": "https://api.githubcopilot.com/mcp/",
        "client": {
            "client_id": "client-from-secret",
            "redirect_uris": ["http://127.0.0.1:8765/callback"],
        },
    }
    monkeypatch.setenv(
        "AGENTIC_MCP_OAUTH_B64_GITHUB",
        base64.b64encode(json.dumps(secret_payload).encode()).decode(),
    )
    await asyncio.gather(
        _github_storage(cache_dir).set_tokens(
            OAuthToken(access_token="token-a", token_type="Bearer")
        ),
        asyncio.to_thread(materialize_after_token_read),
    )

    payload = _load_github_cache_payload(cache_dir)
    has_tokens = payload.get("tokens", {}).get("access_token") == "token-a"
    has_client = payload.get("client", {}).get("client_id") == "client-from-secret"
    assert has_tokens != has_client


@pytest.mark.asyncio
async def test_token_update_after_materialize_preserves_client(tmp_path: Path, monkeypatch):
    from agent_engine_runner_shared.mcp_oauth_secret import materialize_mcp_oauth_secret_cache

    cache_dir = tmp_path / "cache"
    monkeypatch.delenv("AGENTIC_MCP_OAUTH_DIR", raising=False)
    secret_payload = {
        "schema_version": 1,
        "server_name": "github",
        "server_url": "https://api.githubcopilot.com/mcp/",
        "client": {
            "client_id": "client-from-secret",
            "redirect_uris": ["http://127.0.0.1:8765/callback"],
        },
    }
    monkeypatch.setenv(
        "AGENTIC_MCP_OAUTH_B64_GITHUB",
        base64.b64encode(json.dumps(secret_payload).encode()).decode(),
    )

    await asyncio.to_thread(materialize_mcp_oauth_secret_cache, default_cache_dir=cache_dir)
    await _github_storage(cache_dir).set_tokens(
        OAuthToken(access_token="token-a", token_type="Bearer")
    )

    payload = _load_github_cache_payload(cache_dir)
    assert payload["tokens"]["access_token"] == "token-a"
    assert payload["client"]["client_id"] == "client-from-secret"


def _concurrent_oauth_cache_writer(cache_dir: str, write_kind: str) -> None:
    import asyncio
    from pathlib import Path

    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    from agent_engine_runner_shared.mcp_oauth import FileOAuthTokenStorage

    storage = FileOAuthTokenStorage(
        "github",
        "https://api.githubcopilot.com/mcp/",
        cache_dir=Path(cache_dir),
    )
    if write_kind == "tokens":
        asyncio.run(storage.set_tokens(OAuthToken(access_token="token-a", token_type="Bearer")))
        return
    asyncio.run(
        storage.set_client_info(
            OAuthClientInformationFull.model_validate(
                {
                    "client_id": "client-xyz",
                    "redirect_uris": ["http://127.0.0.1:8765/callback"],
                }
            )
        )
    )


def test_cross_process_cache_writes_do_not_clobber(tmp_path: Path):
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_concurrent_oauth_cache_writer, str(tmp_path), "tokens"),
            executor.submit(_concurrent_oauth_cache_writer, str(tmp_path), "client"),
        ]
        for future in futures:
            future.result(timeout=10)

    payload = _load_github_cache_payload(tmp_path)
    assert payload["tokens"]["access_token"] == "token-a"
    assert payload["client"]["client_id"] == "client-xyz"


def test_make_mcp_oauth_auth_uses_non_interactive_provider():
    config = RuntimeMCPServerConfig.model_validate(
        {
            "url": "https://api.githubcopilot.com/mcp/",
            "auth": {
                "type": "oauth",
                "redirect_uri": "http://127.0.0.1:8765/callback",
                "client_name": "Test Client",
                "scope": "repo",
            },
            "timeout_seconds": 5,
        }
    )

    provider = make_mcp_oauth_auth("github", config)

    assert provider is not None


def test_make_mcp_client_credentials_auth_requires_env_vars(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("ATLAS_MCP_CLIENT_ID", raising=False)
    monkeypatch.setenv("ATLAS_MCP_CLIENT_SECRET", "client-secret")

    with pytest.raises(ValueError, match="ATLAS_MCP_CLIENT_ID"):
        make_mcp_client_credentials_auth("atlas", _atlas_client_credentials_config())


@pytest.mark.asyncio
async def test_client_credentials_provider_uses_standard_sdk_discovery_flow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("ATLAS_MCP_CLIENT_ID", "client-id")
    monkeypatch.setenv("ATLAS_MCP_CLIENT_SECRET", "client-secret")
    provider = make_mcp_client_credentials_auth(
        "atlas",
        _atlas_client_credentials_config(
            server_url="https://mcp.example.com/mcp",
            scope="ORG_MCP_ACCESS",
            token_url=None,
        ),
        cache_dir=tmp_path,
    )
    assert provider.context.timeout == 5.0

    request = httpx.Request("POST", "https://mcp.example.com/mcp")
    flow = provider.async_auth_flow(request)
    initial_request = await anext(flow)
    assert "Authorization" not in initial_request.headers

    protected_resource_request = await flow.asend(
        httpx.Response(
            401,
            headers={
                "WWW-Authenticate": (
                    'Bearer resource_metadata="https://mcp.example.com/.well-known/'
                    'oauth-protected-resource/mcp"'
                )
            },
            request=initial_request,
        )
    )
    assert (
        str(protected_resource_request.url)
        == "https://mcp.example.com/.well-known/oauth-protected-resource/mcp"
    )

    oauth_metadata_request = await flow.asend(
        httpx.Response(
            200,
            json={
                "resource": "https://mcp.example.com/mcp",
                "authorization_servers": ["https://auth.example.com"],
            },
            request=protected_resource_request,
        )
    )
    assert str(oauth_metadata_request.url) == (
        "https://auth.example.com/.well-known/oauth-authorization-server"
    )

    token_request = await flow.asend(
        httpx.Response(
            200,
            json={
                "issuer": "https://auth.example.com",
                "authorization_endpoint": "https://auth.example.com/oauth/authorize",
                "token_endpoint": "https://auth.example.com/oauth/token",
            },
            request=oauth_metadata_request,
        )
    )

    token_payload = parse_qs(token_request.content.decode("utf-8"))
    basic_header = token_request.headers["Authorization"].removeprefix("Basic ")
    decoded_basic_header = base64.b64decode(basic_header).decode("utf-8")
    assert str(token_request.url) == "https://auth.example.com/oauth/token"
    assert token_payload == {
        "grant_type": ["client_credentials"],
        "resource": ["https://mcp.example.com/mcp"],
        "scope": ["ORG_MCP_ACCESS"],
    }
    assert decoded_basic_header == "client-id:client-secret"

    authed_request = await flow.asend(
        httpx.Response(
            200,
            json={
                "access_token": "standard-access-token",
                "token_type": "Bearer",
                "expires_in": 3600,
            },
            request=token_request,
        )
    )
    await flow.aclose()

    assert authed_request.headers["Authorization"] == "Bearer standard-access-token"


@pytest.mark.asyncio
async def test_client_credentials_provider_uses_sdk_flow_with_configured_token_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("ATLAS_MCP_CLIENT_ID", "client-id")
    monkeypatch.setenv("ATLAS_MCP_CLIENT_SECRET", "client-secret")
    provider = make_mcp_client_credentials_auth(
        "atlas",
        _atlas_client_credentials_config(scope="ORG_MCP_ACCESS"),
        cache_dir=tmp_path,
    )

    request = httpx.Request("POST", "https://cloud-dev.mongodb.com/api/private/mcp")
    flow = provider.async_auth_flow(request)
    token_request = await anext(flow)

    token_payload = parse_qs(token_request.content.decode("utf-8"))
    basic_header = token_request.headers["Authorization"].removeprefix("Basic ")
    decoded_basic_header = base64.b64decode(basic_header).decode("utf-8")
    assert str(token_request.url) == "https://cloud-dev.mongodb.com/api/oauth/token"
    assert token_payload == {
        "grant_type": ["client_credentials"],
        "scope": ["ORG_MCP_ACCESS"],
    }
    assert decoded_basic_header == "client-id:client-secret"

    authed_request = await flow.asend(
        httpx.Response(
            200,
            json={
                "access_token": "ingress-access-token",
                "token_type": "Bearer",
                "expires_in": 3600,
            },
            request=token_request,
        )
    )
    await flow.aclose()

    assert authed_request.headers["Authorization"] == "Bearer ingress-access-token"
    payload = json.loads(_atlas_cache_path(tmp_path).read_text(encoding="utf-8"))
    assert payload["client_id"] == "client-id"
    assert payload["configured_scope"] == "ORG_MCP_ACCESS"
    assert payload["tokens"]["access_token"] == "ingress-access-token"
    assert payload["tokens"]["expiry"].endswith("Z")


@pytest.mark.asyncio
async def test_client_credentials_provider_retries_direct_flow_on_insufficient_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("ATLAS_MCP_CLIENT_ID", "client-id")
    monkeypatch.setenv("ATLAS_MCP_CLIENT_SECRET", "client-secret")
    provider = make_mcp_client_credentials_auth(
        "atlas",
        _atlas_client_credentials_config(scope=None),
        cache_dir=tmp_path,
    )

    request = httpx.Request("POST", "https://cloud-dev.mongodb.com/api/private/mcp")
    flow = provider.async_auth_flow(request)
    token_request = await anext(flow)
    assert parse_qs(token_request.content.decode("utf-8")) == {
        "grant_type": ["client_credentials"],
    }

    authed_request = await flow.asend(
        httpx.Response(
            200,
            json={
                "access_token": "initial-token",
                "token_type": "Bearer",
                "expires_in": 3600,
            },
            request=token_request,
        )
    )
    assert authed_request.headers["Authorization"] == "Bearer initial-token"

    scoped_token_request = await flow.asend(
        httpx.Response(
            403,
            headers={
                "WWW-Authenticate": 'Bearer error="insufficient_scope", scope="GROUP_MCP_ACCESS"'
            },
            request=authed_request,
        )
    )
    assert parse_qs(scoped_token_request.content.decode("utf-8")) == {
        "grant_type": ["client_credentials"],
        "scope": ["GROUP_MCP_ACCESS"],
    }

    retried_request = await flow.asend(
        httpx.Response(
            200,
            json={
                "access_token": "scoped-token",
                "token_type": "Bearer",
                "expires_in": 3600,
            },
            request=scoped_token_request,
        )
    )
    await flow.aclose()

    assert retried_request.headers["Authorization"] == "Bearer scoped-token"


@pytest.mark.asyncio
async def test_client_credentials_provider_ignores_cached_token_for_different_client_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("ATLAS_MCP_CLIENT_ID", "client-id")
    monkeypatch.setenv("ATLAS_MCP_CLIENT_SECRET", "client-secret")
    _atlas_cache_path(tmp_path).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "server_name": "atlas",
                "server_url": "https://cloud-dev.mongodb.com/api/private/mcp",
                "client_id": "other-client-id",
                "tokens": {
                    "access_token": "cached-access-token",
                    "token_type": "Bearer",
                    "expiry": (datetime.now(UTC) + timedelta(hours=1))
                    .isoformat()
                    .replace("+00:00", "Z"),
                },
            }
        ),
        encoding="utf-8",
    )
    provider = make_mcp_client_credentials_auth(
        "atlas",
        _atlas_client_credentials_config(),
        cache_dir=tmp_path,
    )

    request = httpx.Request("POST", "https://cloud-dev.mongodb.com/api/private/mcp")
    flow = provider.async_auth_flow(request)
    token_request = await anext(flow)
    await flow.aclose()

    assert str(token_request.url) == "https://cloud-dev.mongodb.com/api/oauth/token"
    assert "Authorization" not in request.headers


@pytest.mark.asyncio
async def test_client_credentials_provider_ignores_cached_token_for_different_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("ATLAS_MCP_CLIENT_ID", "client-id")
    monkeypatch.setenv("ATLAS_MCP_CLIENT_SECRET", "client-secret")
    _atlas_cache_path(tmp_path).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "server_name": "atlas",
                "server_url": "https://cloud-dev.mongodb.com/api/private/mcp",
                "client_id": "client-id",
                "configured_scope": "OLD_SCOPE",
                "tokens": {
                    "access_token": "cached-access-token",
                    "token_type": "Bearer",
                    "expiry": (datetime.now(UTC) + timedelta(hours=1))
                    .isoformat()
                    .replace("+00:00", "Z"),
                },
            }
        ),
        encoding="utf-8",
    )
    provider = make_mcp_client_credentials_auth(
        "atlas",
        _atlas_client_credentials_config(scope="NEW_SCOPE"),
        cache_dir=tmp_path,
    )

    request = httpx.Request("POST", "https://cloud-dev.mongodb.com/api/private/mcp")
    flow = provider.async_auth_flow(request)
    token_request = await anext(flow)
    await flow.aclose()

    assert str(token_request.url) == "https://cloud-dev.mongodb.com/api/oauth/token"
    assert "Authorization" not in request.headers


@pytest.mark.asyncio
async def test_interactive_provider_preserves_configured_scope_during_login(
    tmp_path: Path,
):
    requested_scope = "org:read project:read team:read event:read"
    advertised_scope = "org:read project:write team:write event:write"
    redirect_uri = "http://127.0.0.1:8765/callback"
    captured_state = ""
    authorization_urls: list[str] = []

    async def redirect_handler(url: str) -> None:
        nonlocal captured_state
        authorization_urls.append(url)
        captured_state = parse_qs(urlparse(url).query)["state"][0]

    async def callback_handler() -> tuple[str, str | None]:
        return "auth-code", captured_state

    provider = make_interactive_mcp_oauth_auth(
        "sentry",
        _sentry_config(scope=requested_scope),
        cache_dir=tmp_path,
        redirect_uri=redirect_uri,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )

    request = httpx.Request("POST", "https://mcp.sentry.dev/mcp")
    flow = provider.async_auth_flow(request)
    initial_request = await anext(flow)

    protected_resource_request = await flow.asend(
        httpx.Response(
            401,
            headers={
                "WWW-Authenticate": (
                    'Bearer resource_metadata="https://mcp.sentry.dev/.well-known/'
                    f'oauth-protected-resource/mcp", scope="{advertised_scope}"'
                )
            },
            request=initial_request,
        )
    )
    assert (
        str(protected_resource_request.url)
        == "https://mcp.sentry.dev/.well-known/oauth-protected-resource/mcp"
    )

    oauth_metadata_request = await flow.asend(
        httpx.Response(
            200,
            json={
                "resource": "https://mcp.sentry.dev/mcp",
                "authorization_servers": ["https://mcp.sentry.dev"],
                "scopes_supported": advertised_scope.split(),
            },
            request=protected_resource_request,
        )
    )
    assert (
        str(oauth_metadata_request.url)
        == "https://mcp.sentry.dev/.well-known/oauth-authorization-server"
    )

    registration_request = await flow.asend(
        httpx.Response(
            200,
            json={
                "issuer": "https://mcp.sentry.dev",
                "authorization_endpoint": "https://mcp.sentry.dev/oauth/authorize",
                "token_endpoint": "https://mcp.sentry.dev/oauth/token",
                "registration_endpoint": "https://mcp.sentry.dev/oauth/register",
            },
            request=oauth_metadata_request,
        )
    )
    registration_payload = json.loads(registration_request.content.decode("utf-8"))
    assert registration_payload["scope"] == requested_scope
    assert registration_payload["token_endpoint_auth_method"] == "none"

    token_request = await flow.asend(
        httpx.Response(
            201,
            json={
                "client_id": "client-id",
                "redirect_uris": [redirect_uri],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
            request=registration_request,
        )
    )

    authorization_scope = parse_qs(urlparse(authorization_urls[0]).query)["scope"][0]
    assert authorization_scope == requested_scope
    token_payload = token_request.content.decode("utf-8")
    assert "grant_type=authorization_code" in token_payload

    authed_request = await flow.asend(
        httpx.Response(
            200,
            json={
                "access_token": "access-token",
                "token_type": "Bearer",
                "refresh_token": "refresh-token",
                "expires_in": 3600,
                "scope": requested_scope,
            },
            request=token_request,
        )
    )
    await flow.aclose()

    assert authed_request.headers["Authorization"] == "Bearer access-token"
    payload = json.loads(next(tmp_path.glob("sentry-*.json")).read_text(encoding="utf-8"))
    assert payload["tokens"]["scope"] == requested_scope


@pytest.mark.asyncio
async def test_make_mcp_oauth_auth_marks_expired_cached_tokens_refreshable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _write_sentry_cache(tmp_path)
    monkeypatch.delenv("AGENTIC_MCP_OAUTH_DIR", raising=False)
    monkeypatch.setattr(mcp_oauth, "MCP_OAUTH_CACHE_DIR", tmp_path)

    provider = make_mcp_oauth_auth("sentry", _sentry_config())
    await provider._initialize()

    assert provider.context.current_tokens is not None
    assert provider.context.current_tokens.refresh_token == "refresh-token"
    assert provider.context.can_refresh_token()
    assert not provider.context.is_token_valid()


@pytest.mark.asyncio
async def test_cached_oauth_provider_discovers_token_endpoint_before_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _write_sentry_cache(tmp_path)
    requested_urls: list[str] = []

    monkeypatch.delenv("AGENTIC_MCP_OAUTH_DIR", raising=False)
    monkeypatch.setattr(mcp_oauth, "MCP_OAUTH_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        mcp_oauth.httpx, "AsyncClient", _metadata_client(requested_urls=requested_urls)
    )

    provider = make_mcp_oauth_auth("sentry", _sentry_config())
    await provider._initialize()
    request = await provider._refresh_token()

    assert requested_urls == ["https://mcp.sentry.dev/.well-known/oauth-authorization-server"]
    assert str(request.url) == "https://mcp.sentry.dev/oauth/token"


@pytest.mark.asyncio
async def test_cached_oauth_provider_tries_second_metadata_url_after_404(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _write_sentry_cache(tmp_path)
    metadata_urls = [
        "https://mcp.sentry.dev/.well-known/openid-configuration",
        "https://mcp.sentry.dev/.well-known/oauth-authorization-server",
    ]
    requested_urls: list[str] = []

    monkeypatch.setattr(
        mcp_oauth,
        "build_oauth_authorization_server_metadata_discovery_urls",
        lambda *_args: metadata_urls,
    )
    monkeypatch.setattr(
        mcp_oauth.httpx,
        "AsyncClient",
        _metadata_client(requested_urls=requested_urls, not_found_url=metadata_urls[0]),
    )

    provider = make_mcp_oauth_auth("sentry", _sentry_config(), cache_dir=tmp_path)
    await provider._initialize()
    request = await provider._refresh_token()

    assert requested_urls == metadata_urls
    assert str(request.url) == "https://mcp.sentry.dev/oauth/token"


@pytest.mark.asyncio
async def test_cached_oauth_provider_refreshes_expired_cached_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cache_file = _write_sentry_cache(tmp_path)
    monkeypatch.setattr(mcp_oauth.httpx, "AsyncClient", _metadata_client())
    provider = make_mcp_oauth_auth("sentry", _sentry_config(), cache_dir=tmp_path)

    request = httpx.Request("GET", "https://mcp.sentry.dev/mcp")
    flow = provider.async_auth_flow(request)
    refresh_request = await anext(flow)
    assert str(refresh_request.url) == "https://mcp.sentry.dev/oauth/token"

    authed_request = await flow.asend(
        httpx.Response(
            200,
            json={
                "access_token": "refreshed-access-token",
                "token_type": "Bearer",
                "refresh_token": "next-refresh-token",
                "expires_in": 3600,
            },
            request=refresh_request,
        )
    )
    await flow.aclose()

    assert authed_request.headers["Authorization"] == "Bearer refreshed-access-token"
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    assert payload["tokens"]["access_token"] == "refreshed-access-token"
    assert payload["tokens"]["refresh_token"] == "next-refresh-token"
    assert payload["tokens"]["expiry"].endswith("Z")


@pytest.mark.asyncio
async def test_cached_oauth_provider_refreshes_public_client_without_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """The cache a public-client login leaves behind must refresh without a secret."""

    cache_file = _write_sentry_cache(
        tmp_path,
        client={
            "client_id": "public-client-id",
            "redirect_uris": ["http://127.0.0.1:1234/callback"],
            "token_endpoint_auth_method": "none",
        },
    )
    monkeypatch.setattr(mcp_oauth.httpx, "AsyncClient", _metadata_client())
    provider = make_mcp_oauth_auth("sentry", _sentry_config(), cache_dir=tmp_path)

    request = httpx.Request("GET", "https://mcp.sentry.dev/mcp")
    flow = provider.async_auth_flow(request)
    refresh_request = await anext(flow)
    assert str(refresh_request.url) == "https://mcp.sentry.dev/oauth/token"
    assert provider.context.client_info is not None
    assert provider.context.client_info.token_endpoint_auth_method == "none"
    assert provider.context.client_info.client_secret is None
    assert "Authorization" not in refresh_request.headers
    refresh_payload = parse_qs(refresh_request.content.decode("utf-8"))
    assert refresh_payload["grant_type"] == ["refresh_token"]
    assert refresh_payload["refresh_token"] == ["refresh-token"]
    assert refresh_payload["client_id"] == ["public-client-id"]
    assert "client_secret" not in refresh_payload

    authed_request = await flow.asend(
        httpx.Response(
            200,
            json={
                "access_token": "refreshed-access-token",
                "token_type": "Bearer",
                "refresh_token": "next-refresh-token",
                "expires_in": 3600,
            },
            request=refresh_request,
        )
    )
    await flow.aclose()

    assert authed_request.headers["Authorization"] == "Bearer refreshed-access-token"
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    assert payload["tokens"]["access_token"] == "refreshed-access-token"
    assert payload["tokens"]["refresh_token"] == "next-refresh-token"


def _sentry_config(scope: str | None = None) -> RuntimeMCPServerConfig:
    return RuntimeMCPServerConfig.model_validate(
        {
            "url": "https://mcp.sentry.dev/mcp",
            "auth": {"type": "oauth", "scope": scope},
            "timeout_seconds": 5,
        }
    )


def _atlas_client_credentials_config(
    server_url: str = "https://cloud-dev.mongodb.com/api/private/mcp",
    scope: str | None = None,
    token_url: str | None = "https://cloud-dev.mongodb.com/api/oauth/token",
) -> RuntimeMCPServerConfig:
    auth = {
        "type": "client_credentials",
        "client_id_env": "ATLAS_MCP_CLIENT_ID",
        "client_secret_env": "ATLAS_MCP_CLIENT_SECRET",
    }
    if token_url is not None:
        auth["token_url"] = token_url
    if scope is not None:
        auth["scope"] = scope

    return RuntimeMCPServerConfig.model_validate(
        {
            "url": server_url,
            "auth": auth,
            "timeout_seconds": 5,
        }
    )


def _atlas_cache_path(cache_dir: Path) -> Path:
    server_url = _atlas_client_credentials_config().url
    cache_name = mcp_oauth.mcp_oauth_cache_name("atlas", server_url)
    return cache_dir / f"{cache_name}.json"


def _write_sentry_cache(cache_dir: Path, *, client: dict[str, Any] | None = None) -> Path:
    server_url = _sentry_config().url
    cache_name = mcp_oauth.mcp_oauth_cache_name("sentry", server_url)
    cache_file = cache_dir / f"{cache_name}.json"
    cache_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "server_name": "sentry",
                "server_url": "https://mcp.sentry.dev/mcp",
                "client": client
                if client is not None
                else {
                    "client_id": "client-id",
                    "client_secret": "client-secret",
                    "redirect_uris": ["http://127.0.0.1:1234/callback"],
                    "token_endpoint_auth_method": "client_secret_basic",
                },
                "tokens": {
                    "access_token": "expired-access-token",
                    "token_type": "Bearer",
                    "refresh_token": "refresh-token",
                    "expiry": (datetime.now(UTC) - timedelta(minutes=5))
                    .isoformat()
                    .replace("+00:00", "Z"),
                },
            }
        ),
        encoding="utf-8",
    )
    return cache_file


def _metadata_client(
    *,
    requested_urls: list[str] | None = None,
    not_found_url: str | None = None,
) -> type[Any]:
    class MetadataClient:
        def __init__(self, **_kwargs: Any):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args: Any):
            return None

        async def send(self, request: httpx.Request) -> httpx.Response:
            if requested_urls is not None:
                requested_urls.append(str(request.url))
            if str(request.url) == not_found_url:
                return httpx.Response(404, request=request)
            return _metadata_response(request)

    return MetadataClient


def _metadata_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "issuer": "https://mcp.sentry.dev",
            "authorization_endpoint": "https://mcp.sentry.dev/oauth/authorize",
            "token_endpoint": "https://mcp.sentry.dev/oauth/token",
            "registration_endpoint": "https://mcp.sentry.dev/oauth/register",
            "response_types_supported": ["code"],
        },
        request=request,
    )
