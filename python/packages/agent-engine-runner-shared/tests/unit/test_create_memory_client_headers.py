"""Tests for create_memory_client APP_ID env-var-to-header wiring."""

import os
from unittest.mock import Mock, patch

import pytest
from agent_engine_sdk_memory._client import _HEADER_AGENT_ID, _HEADER_AGENT_ID_LEGACY

from agent_engine_runner_shared.memory import create_memory_client


@pytest.fixture
def mock_httpx():
    with patch("agent_engine_sdk_memory._client.httpx.Client"):
        yield


class TestCreateMemoryClientHeaders:
    def test_app_id_becomes_agent_id_header(self, mock_httpx):
        with patch.dict(os.environ, {"APP_ID": "agent-abc"}, clear=False):
            client = create_memory_client("http://localhost:8081")
        assert client is not None
        assert client._static_headers[_HEADER_AGENT_ID] == "agent-abc"
        # Legacy name is dual-sent during the rename transition.
        assert client._static_headers[_HEADER_AGENT_ID_LEGACY] == "agent-abc"

    def test_missing_app_id_not_included(self, mock_httpx):
        env = {k: v for k, v in os.environ.items() if k != "APP_ID"}
        with patch.dict(os.environ, env, clear=True):
            client = create_memory_client("http://localhost:8081")
        assert client is not None
        assert _HEADER_AGENT_ID not in client._static_headers

    def test_blank_app_id_not_included(self, mock_httpx):
        with patch.dict(os.environ, {"APP_ID": "  "}, clear=False):
            client = create_memory_client("http://localhost:8081")
        assert client is not None
        assert _HEADER_AGENT_ID not in client._static_headers

    def test_returns_none_when_memory_package_not_installed(self):
        with patch.dict(
            "sys.modules",
            {
                "agent_engine_sdk_memory": None,
                "agent_engine_sdk_memory._client": None,
            },
        ):
            result = create_memory_client("http://localhost:8081")
        assert result is None


class TestCreateMemoryClientExecutionId:
    def test_execution_id_flows_through_provider(self, mock_httpx):
        with patch(
            "agent_engine_runner_shared.memory.get_current_execution_id", return_value="exec-123"
        ):
            client = create_memory_client("http://localhost:8081")
            headers = client._request_headers()
        assert headers["X-Agent-Engine-Execution-Id"] == "exec-123"
        # Legacy name is dual-sent during the rename transition.
        assert headers["X-Agentic-Execution-Id"] == "exec-123"

    def test_none_execution_id_omits_header(self, mock_httpx):
        with patch("agent_engine_runner_shared.memory.get_current_execution_id", return_value=None):
            client = create_memory_client("http://localhost:8081")
            headers = client._request_headers()
        assert "X-Agent-Engine-Execution-Id" not in headers
        assert "X-Agentic-Execution-Id" not in headers

    def test_execution_id_is_read_per_request(self, mock_httpx):
        with patch("agent_engine_runner_shared.memory.get_current_execution_id") as mock_get:
            client = create_memory_client("http://localhost:8081")
            mock_get.return_value = "exec-a"
            first = client._request_headers()
            mock_get.return_value = "exec-b"
            second = client._request_headers()
        assert first["X-Agent-Engine-Execution-Id"] == "exec-a"
        assert second["X-Agent-Engine-Execution-Id"] == "exec-b"


class TestCreateMemoryClientTLS:
    def test_http_url_uses_platform_client(self):
        """HTTP memory requests suppress transport spans while propagating context."""
        with patch("agent_engine_runner_shared.memory.get_current_execution_id", return_value=None):
            fake_client = Mock()
            with patch(
                "agent_engine_runner_shared.tls_client.create_httpx_client_with_tls",
                return_value=fake_client,
            ) as mock_tls_client:
                with patch("agent_engine_sdk_memory._client.MemoryClient") as mock_client_class:
                    create_memory_client("http://localhost:8081")
                    mock_tls_client.assert_called_once_with("http://localhost:8081", 30.0)
                    call_kwargs = mock_client_class.call_args[1]
                    assert call_kwargs["http_client"] is fake_client

    def test_https_url_creates_mtls_client(self):
        """HTTPS URLs should call create_httpx_client_with_tls and pass result."""
        with patch("agent_engine_runner_shared.memory.get_current_execution_id", return_value=None):
            fake_client = Mock()
            with patch(
                "agent_engine_runner_shared.tls_client.create_httpx_client_with_tls",
                return_value=fake_client,
            ) as mock_tls_client:
                with patch("agent_engine_sdk_memory._client.MemoryClient") as mock_client_class:
                    create_memory_client("https://localhost:8443")
                    # Verify create_httpx_client_with_tls was called with URL and timeout
                    mock_tls_client.assert_called_once_with("https://localhost:8443", 30.0)
                    # Verify the mTLS client was passed to MemoryClient
                    call_kwargs = mock_client_class.call_args[1]
                    assert call_kwargs["http_client"] is fake_client
