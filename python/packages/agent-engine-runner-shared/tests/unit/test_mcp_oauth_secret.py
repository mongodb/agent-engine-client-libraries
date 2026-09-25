import base64
import json
import os
import stat

import pytest

from agent_engine_runner_shared.mcp_oauth import FileOAuthTokenStorage, mcp_oauth_cache_name
from agent_engine_runner_shared.mcp_oauth_secret import materialize_mcp_oauth_secret_cache


def test_materialize_mcp_oauth_secret_cache_sets_runtime_cache_without_secret(
    monkeypatch,
    tmp_path,
):
    cache_dir = tmp_path / "cache"
    monkeypatch.delenv("AGENTIC_MCP_OAUTH_DIR", raising=False)

    materialized = materialize_mcp_oauth_secret_cache(default_cache_dir=cache_dir)

    assert materialized == 0
    assert os.environ["AGENTIC_MCP_OAUTH_DIR"] == str(cache_dir)
    assert cache_dir.is_dir()


def test_materialize_mcp_oauth_secret_cache_writes_decoded_cache(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    payload = {
        "schema_version": 1,
        "server_name": "github-enterprise-team",
        "server_url": "https://github.example.com/mcp/",
        "tokens": {"access_token": "access-token", "refresh_token": "refresh-token"},
    }
    raw_payload = json.dumps(payload, indent=2).encode()
    monkeypatch.setenv(
        "AGENTIC_MCP_OAUTH_B64_GITHUB_ENTERPRISE_TEAM",
        base64.b64encode(raw_payload).decode(),
    )
    monkeypatch.delenv("AGENTIC_MCP_OAUTH_DIR", raising=False)

    materialized = materialize_mcp_oauth_secret_cache(default_cache_dir=cache_dir)

    cache_file = (
        cache_dir / f"{mcp_oauth_cache_name(payload['server_name'], payload['server_url'])}.json"
    )
    assert materialized == 1
    assert os.environ["AGENTIC_MCP_OAUTH_DIR"] == str(cache_dir)
    assert json.loads(cache_file.read_text()) == payload
    assert stat.S_IMODE(cache_file.stat().st_mode) == 0o600


def test_materialized_cache_is_the_default_runtime_reader_path(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    server_url = "https://github.example.com/mcp/"
    payload = {
        "server_name": "github",
        "server_url": server_url,
        "tokens": {"access_token": "access-token"},
    }
    monkeypatch.delenv("AGENTIC_MCP_OAUTH_DIR", raising=False)
    monkeypatch.setenv(
        "AGENTIC_MCP_OAUTH_B64_GITHUB",
        base64.b64encode(json.dumps(payload).encode()).decode(),
    )

    materialize_mcp_oauth_secret_cache(default_cache_dir=cache_dir)
    storage = FileOAuthTokenStorage("github", server_url)

    assert storage.path.parent == cache_dir
    assert json.loads(storage.path.read_text()) == payload


def test_materialize_mcp_oauth_secret_cache_can_use_secret_name(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    payload = {
        "server_url": "https://github.example.com/mcp/",
        "tokens": {"refresh_token": "refresh-token"},
    }
    raw_payload = json.dumps(payload).encode()
    monkeypatch.delenv("AGENTIC_MCP_OAUTH_DIR", raising=False)
    monkeypatch.setenv(
        "AGENTIC_MCP_OAUTH_B64_GITHUB_ENTERPRISE",
        base64.b64encode(raw_payload).decode(),
    )

    materialized = materialize_mcp_oauth_secret_cache(default_cache_dir=cache_dir)

    assert materialized == 1
    cache_name = mcp_oauth_cache_name("github_enterprise", payload["server_url"])
    assert json.loads((cache_dir / f"{cache_name}.json").read_text()) == payload


def test_materialize_mcp_oauth_secret_cache_rejects_invalid_secret(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENTIC_MCP_OAUTH_DIR", raising=False)
    monkeypatch.setenv("AGENTIC_MCP_OAUTH_B64_GITHUB", "not base64!!!")

    with pytest.raises(RuntimeError) as exc_info:
        materialize_mcp_oauth_secret_cache(default_cache_dir=tmp_path)

    message = str(exc_info.value)
    assert "AGENTIC_MCP_OAUTH_B64_GITHUB" in message
    assert "not base64" not in message


def test_materialize_mcp_oauth_secret_cache_preserves_legacy_server_name_identity(
    monkeypatch, tmp_path
):
    # A payload minted before any naming rule can carry an arbitrary alias; the
    # materialized file must land exactly where the runtime reader for that
    # same alias looks.
    payload = {
        "server_name": "GitHub Enterprise",
        "server_url": "https://github.example.com/mcp/",
        "tokens": {"refresh_token": "token"},
    }
    monkeypatch.delenv("AGENTIC_MCP_OAUTH_DIR", raising=False)
    monkeypatch.setenv(
        "AGENTIC_MCP_OAUTH_B64_GITHUB_ENTERPRISE",
        base64.b64encode(json.dumps(payload).encode()).decode(),
    )

    cache_dir = tmp_path / "cache"
    materialize_mcp_oauth_secret_cache(default_cache_dir=cache_dir)

    storage = FileOAuthTokenStorage(payload["server_name"], payload["server_url"])
    assert storage.path.parent == cache_dir
    assert json.loads(storage.path.read_text()) == payload


def test_materialize_mcp_oauth_secret_cache_separates_sanitization_collisions(
    monkeypatch, tmp_path
):
    # Aliases that sanitize to the same readable form must not
    # share a materialized cache file.
    server_url = "https://github.example.com/mcp/"
    monkeypatch.delenv("AGENTIC_MCP_OAUTH_DIR", raising=False)
    for env_suffix, server_name in (("GITHUB_COM", "github.com"), ("GITHUB_COM_2", "github com")):
        payload = {
            "server_name": server_name,
            "server_url": server_url,
            "tokens": {"refresh_token": "token"},
        }
        monkeypatch.setenv(
            f"AGENTIC_MCP_OAUTH_B64_{env_suffix}",
            base64.b64encode(json.dumps(payload).encode()).decode(),
        )

    cache_dir = tmp_path / "cache"
    materialized = materialize_mcp_oauth_secret_cache(default_cache_dir=cache_dir)

    assert materialized == 2
    assert len(list(cache_dir.glob("*.json"))) == 2
