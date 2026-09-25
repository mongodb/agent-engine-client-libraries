from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest

from agent_engine_runner_shared import mcp_oauth_cli
from agent_engine_runner_shared.mcp_oauth import mcp_oauth_cache_name

_SENTRY_MCP_URL = "https://mcp.sentry.dev/mcp"


def test_load_oauth_server_config_returns_named_server(tmp_path: Path):
    agent_path = tmp_path / "agent.yaml"
    agent_path.write_text(
        "\n".join(
            [
                "entrypoint: my_agent:app",
                "mcp:",
                "  servers:",
                "    sentry:",
                "      url: https://mcp.sentry.dev/mcp",
                "      auth:",
                "        type: oauth",
                "        client_name: Atlas Agent Engine Dev MCP Client",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = mcp_oauth_cli.load_oauth_server_config(agent_path, "sentry")

    assert config.url == "https://mcp.sentry.dev/mcp"
    assert config.auth.type == "oauth"
    assert config.auth.client_name == "Atlas Agent Engine Dev MCP Client"


def test_load_oauth_server_config_rejects_non_oauth_server(tmp_path: Path):
    agent_path = tmp_path / "agent.yaml"
    agent_path.write_text(
        "\n".join(
            [
                "entrypoint: my_agent:app",
                "mcp:",
                "  servers:",
                "    github:",
                "      url: https://api.githubcopilot.com/mcp/",
                "      auth:",
                "        type: bearer_env",
                "        token_env: GITHUB_TOKEN",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="uses auth.type 'bearer_env', not oauth"):
        mcp_oauth_cli.load_oauth_server_config(agent_path, "github")


def test_open_url_event_line_is_machine_readable():
    line = mcp_oauth_cli.format_open_url_event(
        "https://example.test/authorize?client_id=abc",
        server_name="sentry",
    )

    prefix, payload = line.split(" ", 1)
    assert prefix == mcp_oauth_cli.OPEN_URL_EVENT_PREFIX
    assert json.loads(payload) == {
        "server_name": "sentry",
        "url": "https://example.test/authorize?client_id=abc",
    }


def test_loopback_callback_server_rejects_non_loopback_redirect():
    with pytest.raises(ValueError, match="localhost or an IPv4 loopback"):
        mcp_oauth_cli.LoopbackCallbackServer(urlparse("http://example.com:8765/callback"))


def test_read_cache_status_reports_refreshable_expired_token(tmp_path: Path):
    expiry = datetime.now(UTC) - timedelta(minutes=5)
    cache_name = mcp_oauth_cache_name("sentry", _SENTRY_MCP_URL)
    (tmp_path / f"{cache_name}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tokens": {
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "expiry": expiry.isoformat().replace("+00:00", "Z"),
                },
            }
        ),
        encoding="utf-8",
    )

    status = mcp_oauth_cli.read_cache_status(
        "sentry", _SENTRY_MCP_URL, tmp_path, now=datetime.now(UTC)
    )

    assert status.exists
    assert status.expired
    assert status.has_refresh_token
    assert f"{cache_name}.json" in str(status.path)


def test_main_status_check_delegates_after_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    agent_path = tmp_path / "agent.yaml"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    agent_path.write_text(
        "\n".join(
            [
                "entrypoint: my_agent:app",
                "mcp:",
                "  servers:",
                "    sentry:",
                "      url: https://mcp.sentry.dev/mcp",
                "      auth:",
                "        type: oauth",
                "",
            ]
        ),
        encoding="utf-8",
    )
    cache_name = mcp_oauth_cache_name("sentry", _SENTRY_MCP_URL)
    (cache_dir / f"{cache_name}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tokens": {
                    "access_token": "access-token",
                    "expiry": (datetime.now(UTC) + timedelta(minutes=5))
                    .isoformat()
                    .replace("+00:00", "Z"),
                },
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_auth(server_name, config, *, cache_dir):
        captured["server_name"] = server_name
        captured["url"] = config.url
        captured["cache_dir"] = cache_dir
        return "cached-auth"

    async def fake_list_tools(server_name, config, *, auth):
        captured["list_server_name"] = server_name
        captured["list_url"] = config.url
        captured["auth"] = auth
        return ["issue_search"]

    monkeypatch.setattr(mcp_oauth_cli, "make_mcp_oauth_auth", fake_auth)
    monkeypatch.setattr(mcp_oauth_cli, "list_oauth_tools", fake_list_tools)

    exit_code = mcp_oauth_cli.main(
        [
            "status",
            "--agent",
            str(agent_path),
            "--server",
            "sentry",
            "--cache-dir",
            str(cache_dir),
            "--check",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert captured == {
        "server_name": "sentry",
        "url": "https://mcp.sentry.dev/mcp",
        "cache_dir": cache_dir,
        "list_server_name": "sentry",
        "list_url": "https://mcp.sentry.dev/mcp",
        "auth": "cached-auth",
    }
    assert "sentry authenticated until" in output
    assert "sentry check ok (1 tool(s))" in output


def test_main_unwraps_oauth_exception_group_with_response_detail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    agent_path = tmp_path / "agent.yaml"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    agent_path.write_text(
        "\n".join(
            [
                "entrypoint: my_agent:app",
                "mcp:",
                "  servers:",
                "    glean:",
                "      url: https://mongodb-be.glean.com/mcp/default",
                "      auth:",
                "        type: oauth",
                "        scope: SEARCH DOCUMENTS ENTITIES",
                "",
            ]
        ),
        encoding="utf-8",
    )

    async def fake_login(*_args, **_kwargs):
        request = httpx.Request("POST", "https://mongodb-be.glean.com/oauth/token")
        response = httpx.Response(
            400,
            json={
                "error": "invalid_scope",
                "error_description": "Unsupported scope: SEARCH",
            },
            request=request,
        )
        raise ExceptionGroup(
            "unhandled errors in a TaskGroup",
            [
                httpx.HTTPStatusError(
                    "request failed",
                    request=request,
                    response=response,
                )
            ],
        )

    monkeypatch.setattr(mcp_oauth_cli, "login_oauth_server", fake_login)

    exit_code = mcp_oauth_cli.main(
        [
            "login",
            "--agent",
            str(agent_path),
            "--server",
            "glean",
            "--cache-dir",
            str(cache_dir),
            "--redirect-uri",
            "http://127.0.0.1:8765/callback",
        ]
    )

    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert "MCP OAuth login for 'glean' failed: HTTP 400 Bad Request" in stderr
    assert "invalid_scope" in stderr
    assert "Unsupported scope: SEARCH" in stderr
    assert "Glean advertises lowercase OAuth scopes" not in stderr
    assert "unhandled errors in a TaskGroup" not in stderr


def test_main_handles_unread_http_response_body(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    agent_path = tmp_path / "agent.yaml"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    agent_path.write_text(
        "\n".join(
            [
                "entrypoint: my_agent:app",
                "mcp:",
                "  servers:",
                "    glean:",
                "      url: https://mongodb-be.glean.com/mcp/default",
                "      auth:",
                "        type: oauth",
                "        scope: SEARCH DOCUMENTS ENTITIES",
                "",
            ]
        ),
        encoding="utf-8",
    )

    async def fake_login(*_args, **_kwargs):
        request = httpx.Request("POST", "https://mongodb-be.glean.com/oauth/token")
        response = httpx.Response(
            400,
            stream=httpx.ByteStream(b'{"error":"invalid_scope"}'),
            request=request,
        )
        raise ExceptionGroup(
            "unhandled errors in a TaskGroup",
            [
                httpx.HTTPStatusError(
                    "request failed",
                    request=request,
                    response=response,
                )
            ],
        )

    monkeypatch.setattr(mcp_oauth_cli, "login_oauth_server", fake_login)

    exit_code = mcp_oauth_cli.main(
        [
            "login",
            "--agent",
            str(agent_path),
            "--server",
            "glean",
            "--cache-dir",
            str(cache_dir),
            "--redirect-uri",
            "http://127.0.0.1:8765/callback",
        ]
    )

    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert "MCP OAuth login for 'glean' failed: HTTP 400 Bad Request" in stderr
    assert "ResponseNotRead" not in stderr
    assert "Attempted to access streaming response content" not in stderr


def test_main_caps_oauth_error_detail_length(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    agent_path = tmp_path / "agent.yaml"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    agent_path.write_text(
        "\n".join(
            [
                "entrypoint: my_agent:app",
                "mcp:",
                "  servers:",
                "    glean:",
                "      url: https://mongodb-be.glean.com/mcp/default",
                "      auth:",
                "        type: oauth",
                "        scope: SEARCH DOCUMENTS ENTITIES",
                "",
            ]
        ),
        encoding="utf-8",
    )

    async def fake_login(*_args, **_kwargs):
        raise RuntimeError("x" * 600)

    monkeypatch.setattr(mcp_oauth_cli, "login_oauth_server", fake_login)

    exit_code = mcp_oauth_cli.main(
        [
            "login",
            "--agent",
            str(agent_path),
            "--server",
            "glean",
            "--cache-dir",
            str(cache_dir),
            "--redirect-uri",
            "http://127.0.0.1:8765/callback",
        ]
    )

    stderr = capsys.readouterr().err
    error_detail = stderr.removeprefix("Error: MCP OAuth login for 'glean' failed: ").strip()
    assert exit_code == 1
    assert len(error_detail) == 500
    assert error_detail.endswith("...")
