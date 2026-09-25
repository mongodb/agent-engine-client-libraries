"""Tests for MemoryClient identity header injection and the execution-id provider.

The client sources execution-id via an injected per-request provider and imports
nothing from the platform stack; static identity headers are merged on every call.
"""

import sys
from unittest.mock import Mock

from agent_engine_sdk_memory._client import (
    _HEADER_AGENT_ID,
    _HEADER_EXECUTION_ID,
    MemoryClient,
)


def test_client_does_not_import_platform_stack() -> None:
    assert "agent_engine_runner_shared" not in sys.modules


class TestMemoryClientInit:
    def test_static_headers_defensively_copied(self, mock_httpx_client):
        original = {_HEADER_AGENT_ID: "agent-1"}
        client = MemoryClient("http://localhost:8081", static_headers=original)
        original[_HEADER_AGENT_ID] = "mutated"
        assert client._static_headers[_HEADER_AGENT_ID] == "agent-1"

    def test_no_static_headers_defaults_to_empty(self, mock_httpx_client):
        client = MemoryClient("http://localhost:8081")
        assert client._static_headers == {}


class TestRequestHeaders:
    def test_no_static_no_provider_returns_empty(self, mock_httpx_client):
        client = MemoryClient("http://localhost:8081")
        assert client._request_headers() == {}

    def test_static_agent_id_included(self, mock_httpx_client):
        client = MemoryClient(
            "http://localhost:8081",
            static_headers={_HEADER_AGENT_ID: "agent-abc"},
        )
        headers = client._request_headers()
        assert headers[_HEADER_AGENT_ID] == "agent-abc"

    def test_no_provider_omits_execution_id(self, mock_httpx_client):
        client = MemoryClient(
            "http://localhost:8081",
            static_headers={_HEADER_AGENT_ID: "agent-1"},
        )
        headers = client._request_headers()
        assert headers == {_HEADER_AGENT_ID: "agent-1"}
        assert _HEADER_EXECUTION_ID not in headers


class TestExecutionIdProvider:
    def test_merges_execution_id_from_provider(self, mock_httpx_client):
        client = MemoryClient(
            "http://localhost:8081",
            static_headers={_HEADER_AGENT_ID: "agent-1"},
            execution_id_provider=lambda: "exec-123",
        )
        headers = client._request_headers()
        assert headers[_HEADER_AGENT_ID] == "agent-1"
        assert headers[_HEADER_EXECUTION_ID] == "exec-123"

    def test_omits_execution_id_when_provider_returns_none(self, mock_httpx_client):
        client = MemoryClient(
            "http://localhost:8081",
            execution_id_provider=lambda: None,
        )
        assert _HEADER_EXECUTION_ID not in client._request_headers()


class TestHeadersOnRequests:
    def _make_response(self, data):
        resp = Mock()
        resp.json.return_value = data
        return resp

    def test_write_turn_sends_static_headers(self, mock_httpx_client):
        mock_httpx_client.post.return_value = self._make_response(
            {"id": "t1", "session_id": "s1", "turn_seq": 1, "acknowledged": True}
        )
        client = MemoryClient(
            "http://localhost:8081",
            static_headers={_HEADER_AGENT_ID: "agent-1"},
        )
        client.write_turn(
            session_id="s1",
            role="user",
            org_id="org1",
            user_id="u1",
            project_id="proj1",
        )
        headers = mock_httpx_client.post.call_args[1]["headers"]
        assert headers[_HEADER_AGENT_ID] == "agent-1"
        assert _HEADER_EXECUTION_ID not in headers

    def test_build_context_sends_static_headers(self, mock_httpx_client):
        mock_httpx_client.post.return_value = self._make_response(
            {"formatted_context": "ctx", "metadata": {}}
        )
        client = MemoryClient(
            "http://localhost:8081",
            static_headers={_HEADER_AGENT_ID: "agent-1"},
        )
        client.build_context(
            query="q",
            session_id="s1",
            org_id="org1",
            user_id="u1",
            project_id="proj1",
        )
        headers = mock_httpx_client.post.call_args[1]["headers"]
        assert headers[_HEADER_AGENT_ID] == "agent-1"
        assert _HEADER_EXECUTION_ID not in headers
