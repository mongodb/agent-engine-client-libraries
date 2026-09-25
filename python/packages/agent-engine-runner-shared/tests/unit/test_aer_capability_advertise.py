"""Tests for AER capability advertise flow (_advertise_capabilities)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from agent_engine_runner_shared.agent_config import load_runtime_agent_config
from agent_engine_runner_shared.server.aer import AERServer


def _make_aer_server_from_workdir(workdir: Path) -> AERServer:
    runtime = MagicMock()
    runtime.app_name = "test-agent"
    runtime.org_id = "org-from-runtime"
    runtime.project_id = "proj-from-runtime"
    runtime.agent_config = load_runtime_agent_config(workdir)
    runtime._graph_builder = MagicMock()
    return AERServer(runtime)


class _NoTextResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code
        self.text_accessed = False

    @property
    def text(self) -> str:
        self.text_accessed = True
        raise AssertionError("resp.text should not be read")


class TestAdvertiseCapabilities:
    @pytest.mark.asyncio
    async def test_advertises_language_framework_features_from_agent_yaml(
        self, monkeypatch, tmp_path: Path
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.yaml").write_text(
            "\n".join(
                [
                    "entrypoint: config.agent:app",
                    "language: python",
                    "framework: langgraph",
                    "features:",
                    "  durable_workflow: true",
                    "  memory: false",
                ]
            ),
            encoding="utf-8",
        )

        captured: dict[str, Any] = {}

        async def mock_post(url, **kwargs):
            captured["url"] = str(url)
            captured["json"] = kwargs.get("json")
            captured["timeout"] = kwargs.get("timeout")
            resp = MagicMock()
            resp.status_code = 200
            return resp

        server = _make_aer_server_from_workdir(tmp_path)
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client

        await server._advertise_capabilities("http://oe:8000", "ws-test-agent")

        assert captured["url"] == "http://oe:8000/agent/capabilities"
        assert captured["timeout"] == 10.0
        body = captured["json"]
        assert body["workspace_id"] == "ws-test-agent"
        assert body["org_id"] == "org-from-runtime"
        assert body["project_id"] == "proj-from-runtime"
        assert body["language"] == "python"
        assert body["framework"] == "langgraph"
        assert body["features"]["durable_workflow"] is True
        assert body["features"]["memory"] is False
        assert body["features"]["owner_callback_fallback"] is True
        assert server._capabilities_advertised is True

    @pytest.mark.asyncio
    async def test_omitted_durable_workflow_not_in_features(self, monkeypatch, tmp_path: Path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.yaml").write_text(
            "entrypoint: config.agent:app\nfeatures:\n  memory: true\n",
            encoding="utf-8",
        )

        captured: dict[str, Any] = {}

        async def mock_post(url, **kwargs):
            captured["json"] = kwargs.get("json")
            resp = MagicMock()
            resp.status_code = 200
            return resp

        server = _make_aer_server_from_workdir(tmp_path)
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client

        await server._advertise_capabilities("http://oe:8000", "ws-test")

        assert "durable_workflow" not in captured["json"]["features"]
        assert captured["json"]["features"]["memory"] is True
        assert captured["json"]["features"]["owner_callback_fallback"] is True
        assert captured["json"]["language"] == "python"

    @pytest.mark.asyncio
    async def test_owner_callback_fallback_injected_without_mutating_config(
        self, monkeypatch, tmp_path: Path
    ):
        """SDK injects owner_callback_fallback even when agent.yaml has no features,
        and the injection must not mutate the loaded agent config in place."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.yaml").write_text(
            "entrypoint: config.agent:app\nlanguage: python\nframework: langgraph\n",
            encoding="utf-8",
        )

        captured: dict[str, Any] = {}

        async def mock_post(url, **kwargs):
            captured["json"] = kwargs.get("json")
            resp = MagicMock()
            resp.status_code = 200
            return resp

        server = _make_aer_server_from_workdir(tmp_path)
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client

        assert server.runtime.agent_config.features.explicit() == {}

        await server._advertise_capabilities("http://oe:8000", "ws-test")

        body = captured["json"]
        assert set(body.keys()) == {
            "workspace_id",
            "org_id",
            "project_id",
            "language",
            "framework",
            "features",
        }
        assert body["features"] == {"owner_callback_fallback": True}
        # The agent config object itself is untouched by the injection.
        assert server.runtime.agent_config.features.explicit() == {}

    @pytest.mark.asyncio
    async def test_missing_scope_rejects_capability_registration(self, monkeypatch, tmp_path: Path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.yaml").write_text(
            "entrypoint: config.agent:app\n",
            encoding="utf-8",
        )

        posted = {"n": 0}

        async def mock_post(url, **kwargs):
            posted["n"] += 1
            resp = MagicMock()
            resp.status_code = 200
            return resp

        server = _make_aer_server_from_workdir(tmp_path)
        server.runtime.org_id = None
        server.runtime.project_id = "proj-from-runtime"
        server._a2a_registered = True
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client

        with pytest.raises(RuntimeError, match="requires ORG_ID and PROJECT_ID"):
            await server._advertise_capabilities("http://oe:8000", "ws-test")

        assert posted["n"] == 0
        assert server._capabilities_advertised is False

    @pytest.mark.asyncio
    async def test_network_error_propagates(self, monkeypatch, tmp_path: Path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.yaml").write_text(
            "entrypoint: config.agent:app\n",
            encoding="utf-8",
        )

        async def mock_post(url, **kwargs):
            raise httpx.ConnectError("connection refused")

        server = _make_aer_server_from_workdir(tmp_path)
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client

        with pytest.raises(httpx.ConnectError, match="connection refused"):
            await server._advertise_capabilities("http://oe:8000", "ws-test")

        assert server._capabilities_advertised is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [307, 404, 503])
    async def test_non_success_response_rejects_registration(
        self, monkeypatch, tmp_path: Path, status_code: int
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.yaml").write_text(
            "entrypoint: config.agent:app\n",
            encoding="utf-8",
        )

        calls = {"n": 0}
        captured: dict[str, Any] = {}

        async def mock_post(url, **kwargs):
            calls["n"] += 1
            resp = _NoTextResponse(status_code)
            captured["resp"] = resp
            return resp

        server = _make_aer_server_from_workdir(tmp_path)
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client
        server._a2a_registered = True

        with pytest.raises(RuntimeError, match="registration was rejected"):
            await server._advertise_capabilities("http://oe:8000", "ws-test")
        assert server._capabilities_advertised is False
        assert calls["n"] == 1
        assert captured["resp"].text_accessed is False

    @pytest.mark.asyncio
    async def test_on_startup_advertises_when_oe_and_app_id_set(self, monkeypatch, tmp_path: Path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.yaml").write_text(
            "entrypoint: config.agent:app\nlanguage: python\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("OE_URL", "http://oe:8000")
        monkeypatch.setenv("APP_ID", "ws-startup")

        captured: dict[str, Any] = {}

        async def mock_post(url, **kwargs):
            captured["url"] = str(url)
            resp = MagicMock()
            resp.status_code = 200
            return resp

        server = _make_aer_server_from_workdir(tmp_path)
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client
        server._a2a_registered = True

        await server.on_startup()

        assert captured["url"] == "http://oe:8000/agent/capabilities"
        assert server._capabilities_advertised is True

    @pytest.mark.asyncio
    async def test_startup_propagates_capability_advertisement_failure(
        self, monkeypatch, tmp_path: Path
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.yaml").write_text(
            "entrypoint: config.agent:app\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("OE_URL", "http://oe:8000")
        monkeypatch.setenv("APP_ID", "ws-startup")

        calls = {"n": 0}

        async def mock_post(url, **kwargs):
            calls["n"] += 1
            raise httpx.ConnectError("connection refused")

        server = _make_aer_server_from_workdir(tmp_path)
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client
        server._a2a_registered = True

        with pytest.raises(httpx.ConnectError, match="connection refused"):
            await server.on_startup()

        assert calls["n"] == 1
        assert server._capabilities_advertised is False

    @pytest.mark.asyncio
    async def test_startup_bounds_the_complete_registration_request(
        self, monkeypatch, tmp_path: Path
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.yaml").write_text(
            "entrypoint: config.agent:app\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("OE_URL", "http://oe:8000")
        monkeypatch.setenv("APP_ID", "ws-startup")
        monkeypatch.setattr(
            "agent_engine_runner_shared.server.aer.CAPABILITY_ADVERTISE_TIMEOUT_S", 0.05
        )

        cancelled = asyncio.Event()

        async def mock_post(url, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        server = _make_aer_server_from_workdir(tmp_path)
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client
        server._a2a_registered = True

        started = asyncio.get_running_loop().time()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(server.on_startup(), timeout=0.5)

        assert asyncio.get_running_loop().time() - started < 0.25
        assert cancelled.is_set()
        assert server._capabilities_advertised is False

    @pytest.mark.asyncio
    async def test_startup_requires_registration_configuration(self, monkeypatch, tmp_path: Path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.yaml").write_text(
            "entrypoint: config.agent:app\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("OE_URL", raising=False)
        monkeypatch.delenv("APP_ID", raising=False)

        server = _make_aer_server_from_workdir(tmp_path)

        with pytest.raises(RuntimeError, match="requires OE_URL and APP_ID"):
            await server.on_startup()

    @pytest.mark.asyncio
    async def test_startup_rejects_terminal_registration_failure(self, monkeypatch, tmp_path: Path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.yaml").write_text(
            "entrypoint: config.agent:app\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("OE_URL", "http://oe:8000")
        monkeypatch.setenv("APP_ID", "ws-startup")

        calls = {"n": 0}

        async def mock_post(url, **kwargs):
            calls["n"] += 1
            return _NoTextResponse(404)

        server = _make_aer_server_from_workdir(tmp_path)
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client

        with pytest.raises(RuntimeError, match="registration was rejected"):
            await server.on_startup()

        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_startup_preserves_invalid_tls_configuration_error(
        self, monkeypatch, tmp_path: Path
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.yaml").write_text(
            "entrypoint: config.agent:app\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("OE_URL", "https://oe:8443")
        monkeypatch.setenv("APP_ID", "ws-startup")
        for name in (
            "TLS_CERT_PATH",
            "TLS_KEY_PATH",
            "TLS_CA_CERT_PATH",
            "TLS_CERT_PEM",
            "TLS_KEY_PEM",
            "TLS_CA_CERT_PEM",
        ):
            monkeypatch.delenv(name, raising=False)

        server = _make_aer_server_from_workdir(tmp_path)

        with pytest.raises(ValueError, match="HTTPS URL requires mTLS configuration"):
            await server.on_startup()

        assert server._capabilities_advertised is False
