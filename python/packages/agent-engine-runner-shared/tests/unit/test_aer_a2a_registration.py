"""Tests for AER A2A registration flow (_register_a2a_config)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from agent_engine_runner_shared.agent_config import A2AConfig, A2ASkillConfig
from agent_engine_runner_shared.server.aer import AERServer


def _jwt_signature_matches(token: str, secret: str) -> bool:
    header, payload, signature = token.split(".")
    signing_input = f"{header}.{payload}"
    expected = (
        base64.urlsafe_b64encode(
            hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
        )
        .rstrip(b"=")
        .decode("ascii")
    )
    return signature == expected


def _make_aer_server(a2a_config: A2AConfig | None = None) -> AERServer:
    """Create a minimal AERServer with mocked runtime for testing."""
    runtime = MagicMock()
    runtime.app_name = "test-agent"
    runtime.org_id = "org-from-runtime"
    runtime.agent_config.a2a = a2a_config or A2AConfig(
        enabled=True,
        skills=[A2ASkillConfig(name="test-skill", description="A test skill")],
        input_modes=["text/plain"],
        output_modes=["text/plain"],
        allowed_callers=["ws-allowed"],
    )
    runtime._graph_builder = MagicMock()
    server = AERServer(runtime)
    return server


class TestRegisterA2AConfig:
    @pytest.mark.asyncio
    async def test_sends_correct_payload(self, monkeypatch):
        monkeypatch.setenv("PROJECT_ID", "proj-123")
        monkeypatch.setenv("ORG_ID", "org-456")
        monkeypatch.delenv("A2A_JWT_SECRET", raising=False)

        captured: dict[str, Any] = {}

        async def mock_post(url, **kwargs):
            captured["url"] = str(url)
            captured["json"] = kwargs.get("json")
            captured["headers"] = kwargs.get("headers", {})
            resp = MagicMock()
            resp.status_code = 200
            return resp

        server = _make_aer_server()
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client

        await server._register_a2a_config("http://oe:8000", "ws-test-agent")

        assert captured["url"] == "http://oe:8000/a2a/register"
        body = captured["json"]
        assert body["workspace_id"] == "ws-test-agent"
        assert body["name"] == "test-agent"
        assert body["org_id"] == "org-456"
        assert body["project_id"] == "proj-123"
        assert body["a2a_enabled"] is True
        assert len(body["skills"]) == 1
        assert body["skills"][0]["name"] == "test-skill"
        assert body["input_modes"] == ["text/plain"]
        assert body["allowed_callers"] == ["ws-allowed"]

    @pytest.mark.asyncio
    async def test_sends_jwt_when_secret_set(self, monkeypatch):
        monkeypatch.setenv("A2A_JWT_SECRET", "test-secret")
        monkeypatch.setenv("PROJECT_ID", "proj-1")
        monkeypatch.setenv("ORG_ID", "org-1")

        captured_headers: dict[str, str] = {}

        async def mock_post(url, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            resp = MagicMock()
            resp.status_code = 200
            return resp

        server = _make_aer_server()
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client

        await server._register_a2a_config("http://oe:8000", "ws-test")

        assert "Authorization" in captured_headers
        assert captured_headers["Authorization"].startswith("Bearer ")
        token = captured_headers["Authorization"].removeprefix("Bearer ")
        assert len(token.split(".")) == 3
        assert _jwt_signature_matches(token, "test-secret")

    @pytest.mark.asyncio
    async def test_no_jwt_when_secret_empty(self, monkeypatch):
        monkeypatch.delenv("A2A_JWT_SECRET", raising=False)

        captured_headers: dict[str, str] = {}

        async def mock_post(url, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            resp = MagicMock()
            resp.status_code = 200
            return resp

        server = _make_aer_server()
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client

        await server._register_a2a_config("http://oe:8000", "ws-test")

        assert "Authorization" not in captured_headers

    @pytest.mark.asyncio
    async def test_marks_registered_on_success(self, monkeypatch):
        monkeypatch.delenv("A2A_JWT_SECRET", raising=False)

        async def mock_post(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            return resp

        server = _make_aer_server()
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client
        assert server._a2a_registered is False

        await server._register_a2a_config("http://oe:8000", "ws-test")

        assert server._a2a_registered is True

    @pytest.mark.asyncio
    async def test_does_not_mark_registered_on_failure(self, monkeypatch):
        monkeypatch.delenv("A2A_JWT_SECRET", raising=False)

        async def mock_post(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 401
            resp.text = "unauthorized"
            return resp

        server = _make_aer_server()
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client

        await server._register_a2a_config("http://oe:8000", "ws-test")

        assert server._a2a_registered is False

    @pytest.mark.asyncio
    async def test_does_not_mark_registered_on_exception(self, monkeypatch):
        monkeypatch.delenv("A2A_JWT_SECRET", raising=False)

        async def mock_post(url, **kwargs):
            raise httpx.ConnectError("connection refused")

        server = _make_aer_server()
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client

        await server._register_a2a_config("http://oe:8000", "ws-test")

        assert server._a2a_registered is False

    @pytest.mark.asyncio
    async def test_jwt_contains_issuer_org_and_project(self, monkeypatch):
        monkeypatch.setenv("A2A_JWT_SECRET", "test-secret")
        monkeypatch.setenv("PROJECT_ID", "proj-xyz")
        monkeypatch.setenv("ORG_ID", "org-abc")
        monkeypatch.delenv("MDB_AGENTIC_STORE_DB", raising=False)

        captured_headers: dict[str, str] = {}

        async def mock_post(url, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            resp = MagicMock()
            resp.status_code = 200
            return resp

        server = _make_aer_server()
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client

        await server._register_a2a_config("http://oe:8000", "ws-test")

        import base64

        token = captured_headers["Authorization"].removeprefix("Bearer ")
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))

        assert claims["iss"] == "oe.mdb_store"
        assert claims["source_agent"] == "ws-test"
        assert claims["org_id"] == "org-abc"
        assert claims["project_id"] == "proj-xyz"

    @pytest.mark.asyncio
    async def test_org_id_falls_back_to_runtime(self, monkeypatch):
        monkeypatch.delenv("ORG_ID", raising=False)
        monkeypatch.delenv("A2A_JWT_SECRET", raising=False)

        captured: dict[str, Any] = {}

        async def mock_post(url, **kwargs):
            captured["json"] = kwargs.get("json")
            resp = MagicMock()
            resp.status_code = 200
            return resp

        server = _make_aer_server()
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client

        await server._register_a2a_config("http://oe:8000", "ws-test")

        assert captured["json"]["org_id"] == "org-from-runtime"

    @pytest.mark.asyncio
    async def test_logs_error_when_enabled_and_secret_missing(self, monkeypatch, caplog):
        monkeypatch.delenv("A2A_JWT_SECRET", raising=False)

        async def mock_post(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            return resp

        server = _make_aer_server()
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client

        with caplog.at_level("ERROR"):
            await server._register_a2a_config("http://oe:8000", "ws-test")

        assert "A2A_JWT_SECRET" in caplog.text
        assert "no peers" in caplog.text.lower() or "look like" in caplog.text

    @pytest.mark.asyncio
    async def test_does_not_log_missing_secret_when_a2a_disabled(self, monkeypatch, caplog):
        monkeypatch.delenv("A2A_JWT_SECRET", raising=False)

        async def mock_post(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            return resp

        server = _make_aer_server(A2AConfig(enabled=False))
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client

        with caplog.at_level("ERROR"):
            await server._register_a2a_config("http://oe:8000", "ws-test")

        assert "A2A_JWT_SECRET" not in caplog.text

    @pytest.mark.asyncio
    async def test_non_200_warning_mentions_a2a_jwt_secret(self, monkeypatch, caplog):
        monkeypatch.setenv("A2A_JWT_SECRET", "test-secret")

        async def mock_post(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 401
            resp.text = "unauthorized"
            return resp

        server = _make_aer_server()
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client

        with caplog.at_level("WARNING"):
            await server._register_a2a_config("http://oe:8000", "ws-test")

        assert server._a2a_registered is False
        assert "A2A_JWT_SECRET" in caplog.text
        assert "401" in caplog.text
        assert "test-secret" not in caplog.text

    @pytest.mark.asyncio
    async def test_jwt_hmac_uses_untrimmed_a2a_secret(self, monkeypatch, caplog):
        padded = "  test-secret  "
        monkeypatch.setenv("A2A_JWT_SECRET", padded)
        monkeypatch.setenv("PROJECT_ID", "proj-1")
        monkeypatch.setenv("ORG_ID", "org-1")

        captured_headers: dict[str, str] = {}

        async def mock_post(url, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            resp = MagicMock()
            resp.status_code = 200
            return resp

        server = _make_aer_server()
        mock_client = MagicMock()
        mock_client.post = mock_post

        async def mock_get_client(url):
            return mock_client

        server._get_client = mock_get_client

        with caplog.at_level("ERROR"):
            await server._register_a2a_config("http://oe:8000", "ws-test")

        token = captured_headers["Authorization"].removeprefix("Bearer ")
        assert _jwt_signature_matches(token, padded)
        assert not _jwt_signature_matches(token, padded.strip())
        assert "not set" not in caplog.text.lower()
        assert padded not in caplog.text
        assert padded.strip() not in caplog.text


class TestBuildComponentEndpoint:
    def test_extracts_namespace_from_oe_url(self):
        url = AERServer._build_component_endpoint(
            "http://oe.my-namespace.svc.cluster.local:8000", "ws-123", "aer"
        )
        assert url == "http://ws-123-aer.my-namespace.svc.cluster.local:8001"

    def test_no_namespace_in_url(self):
        url = AERServer._build_component_endpoint("http://localhost:8000", "ws-123", "tool")
        assert url == "http://ws-123-tool:8001"

    def test_uses_service_port_env_var(self, monkeypatch):
        monkeypatch.setenv("WS_123_AER_SERVICE_PORT", "9999")
        url = AERServer._build_component_endpoint(
            "http://oe.ns.svc.cluster.local:8000", "ws-123", "aer"
        )
        assert ":9999" in url

    def test_build_aer_endpoint_delegates(self):
        url = AERServer._build_aer_endpoint("http://oe.ns.svc.cluster.local:8000", "ws-abc")
        assert "ws-abc-aer" in url
        assert "ns" in url
