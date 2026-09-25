"""Tests for MemoryClient HTTP wrapper."""

from unittest.mock import Mock, patch

import pytest
from agent_engine_sdk_memory._client import MemoryClient
from agent_engine_sdk_memory.errors import MemoryIdentityError
from agent_engine_sdk_memory.models import (
    ContextResponse,
    CreateEpisodicResult,
    CreateSemanticResult,
    CreateTaxonomicResult,
    SourceSpec,
    WriteTurnResult,
)


class TestMemoryClientInit:
    def test_init_strips_trailing_slash(self, mock_httpx_client):
        client = MemoryClient("http://localhost:8081/")
        assert client._base_url == "http://localhost:8081"

    def test_init_without_trailing_slash(self, mock_httpx_client):
        client = MemoryClient("http://localhost:8081")
        assert client._base_url == "http://localhost:8081"

    def test_init_with_ca_bundle_path(self):
        """Test that ca_bundle parameter is passed to httpx.Client as verify."""
        with patch("agent_engine_sdk_memory._client.httpx.Client") as mock_client:
            MemoryClient("https://localhost:8443", ca_bundle="/path/to/ca.crt")
            mock_client.assert_called_once()
            call_kwargs = mock_client.call_args[1]
            assert call_kwargs["verify"] == "/path/to/ca.crt"

    def test_init_with_ca_bundle_false(self):
        """Test that ca_bundle=False disables SSL verification."""
        with patch("agent_engine_sdk_memory._client.httpx.Client") as mock_client:
            MemoryClient("https://localhost:8443", ca_bundle=False)
            mock_client.assert_called_once()
            call_kwargs = mock_client.call_args[1]
            assert call_kwargs["verify"] is False

    def test_init_with_ca_bundle_none(self):
        """Test that ca_bundle=None uses system default CA bundle."""
        with patch("agent_engine_sdk_memory._client.httpx.Client") as mock_client:
            MemoryClient("https://localhost:8443", ca_bundle=None)
            mock_client.assert_called_once()
            call_kwargs = mock_client.call_args[1]
            assert call_kwargs["verify"] is True

    def test_init_with_http_client(self):
        """Test that http_client parameter bypasses internal client creation."""
        import httpx

        injected_client = httpx.Client(timeout=60.0)
        with patch("agent_engine_sdk_memory._client.httpx.Client") as mock_client:
            client = MemoryClient("https://localhost:8443", http_client=injected_client)
            # Verify httpx.Client was NOT called (injected client used instead)
            mock_client.assert_not_called()
            # Verify the injected client is used
            assert client._client is injected_client


class TestWriteTurn:
    def test_write_turn_success(self, mock_httpx_client):
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "turn123",
            "session_id": "sess1",
            "turn_seq": 1,
            "acknowledged": True,
        }
        mock_httpx_client.post.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        result = client.write_turn(
            session_id="sess1",
            role="user",
            org_id="org1",
            user_id="user1",
            project_id="proj1",
            content="Hello",
        )

        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        assert call_args[0][0] == "http://localhost:8081/api/v1/memory/stm/turns"
        assert call_args[1]["json"]["session_id"] == "sess1"
        assert call_args[1]["json"]["role"] == "user"
        assert isinstance(result, WriteTurnResult)

    def test_write_turn_requires_session_id(self, mock_httpx_client):
        client = MemoryClient("http://localhost:8081")

        with pytest.raises(ValueError, match="session_id is required"):
            client.write_turn(
                session_id=None,
                role="user",
                org_id="org1",
                user_id="user1",
                project_id="proj1",
                content="Hello",
            )

        mock_httpx_client.post.assert_not_called()


class TestSemanticMemory:
    def test_create_semantic_success(self, mock_httpx_client):
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "mem123",
            "label": "test",
            "text": "Test memory",
            "org_id": "org1",
            "user_id": "user1",
            "visibility": "private",
            "has_embedding": True,
        }
        mock_httpx_client.post.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        result = client.create_semantic(
            label="test",
            text="Test memory",
            org_id="org1",
            user_id="user1",
            project_id="proj1",
        )

        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        assert call_args[0][0] == "http://localhost:8081/api/v1/memory/semantic"
        assert isinstance(result, CreateSemanticResult)

    def test_create_semantic_sends_upsert_flag(self, mock_httpx_client):
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "mem123",
            "label": "test",
            "has_embedding": True,
            "acknowledged": True,
        }
        mock_httpx_client.post.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        client.create_semantic(
            label="test",
            text="Test memory",
            org_id="org1",
            user_id="user1",
            project_id="proj1",
            upsert=True,
        )

        call_args = mock_httpx_client.post.call_args
        assert call_args[1]["json"]["upsert"] is True

    def test_create_semantic_includes_metadata_when_provided(self, mock_httpx_client):
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "mem123",
            "label": "test",
            "has_embedding": True,
            "acknowledged": True,
        }
        mock_httpx_client.post.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        client.create_semantic(
            label="test",
            text="Test memory",
            org_id="org1",
            user_id="user1",
            project_id="proj1",
            metadata={"channel": "web"},
        )

        call_args = mock_httpx_client.post.call_args
        assert call_args[1]["json"]["metadata"] == {"channel": "web"}

    def test_create_semantic_forwards_deprecated_contextual_metadata(
        self, mock_httpx_client
    ):
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "mem123",
            "label": "test",
            "has_embedding": True,
            "acknowledged": True,
        }
        mock_httpx_client.post.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        with pytest.warns(DeprecationWarning, match="contextual_metadata"):
            client.create_semantic(
                label="test",
                text="Test memory",
                org_id="org1",
                user_id="user1",
                project_id="proj1",
                contextual_metadata={"channel": "web"},
            )

        call_args = mock_httpx_client.post.call_args
        assert call_args[1]["json"]["metadata"] == {"channel": "web"}

    def test_create_semantic_prefers_metadata_over_deprecated_alias(
        self, mock_httpx_client
    ):
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "mem123",
            "label": "test",
            "has_embedding": True,
            "acknowledged": True,
        }
        mock_httpx_client.post.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        with pytest.warns(DeprecationWarning, match="contextual_metadata"):
            client.create_semantic(
                label="test",
                text="Test memory",
                org_id="org1",
                user_id="user1",
                project_id="proj1",
                metadata={"channel": "web"},
                contextual_metadata={"channel": "legacy"},
            )

        call_args = mock_httpx_client.post.call_args
        assert call_args[1]["json"]["metadata"] == {"channel": "web"}

    def test_fetch_semantic_memories_uses_correct_endpoint(self, mock_httpx_client):
        mock_response = Mock()
        mock_response.json.return_value = {"memories": []}
        mock_httpx_client.post.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        result = client.fetch_semantic_memories(
            query="test query",
            org_id="org1",
            project_id="proj1",
            user_id="user1",
        )

        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        assert (
            call_args[0][0] == "http://localhost:8081/api/v1/memory/retrieval/semantic"
        )
        assert call_args[1]["json"]["query"] == "test query"
        assert isinstance(result, list)

    def test_get_semantic_by_label(self, mock_httpx_client):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"entries": [{"id": "mem1", "label": "test"}]}
        mock_httpx_client.get.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        result = client.get_semantic(org_id="org1", project_id="proj1", label="test")

        mock_httpx_client.get.assert_called_once()
        call_args = mock_httpx_client.get.call_args
        assert call_args[0][0] == "http://localhost:8081/api/v1/memory/semantic"
        assert result["label"] == "test"

    def test_get_semantic_not_found(self, mock_httpx_client):
        mock_response = Mock()
        mock_response.status_code = 404
        mock_httpx_client.get.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        result = client.get_semantic(org_id="org1", project_id="proj1", label="missing")

        assert result is None


class TestEpisodicMemory:
    def test_create_episodic_success(self, mock_httpx_client):
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "ep123",
            "title": "Test episode",
            "has_embedding": True,
            "acknowledged": True,
        }
        mock_httpx_client.post.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        result = client.create_episodic(
            title="Test episode",
            content="Content",
            summary_text="Summary",
            org_id="org1",
            user_id="user1",
            session_id="sess1",
            project_id="proj1",
            snapshot_ref_id="snap1",
            summary_type="llm",
            source_agent="agent",
            participants=["Customer", "Alex"],
            tags=["policy_created"],
        )

        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        assert call_args[0][0] == "http://localhost:8081/api/v1/memory/episodic"
        assert call_args[1]["json"]["session_id"] == "sess1"
        assert call_args[1]["json"]["snapshot_ref_id"] == "snap1"
        assert call_args[1]["json"]["summary_type"] == "llm"
        assert call_args[1]["json"]["source_agent"] == "agent"
        assert call_args[1]["json"]["participants"] == ["Customer", "Alex"]
        assert call_args[1]["json"]["tags"] == ["policy_created"]
        # metadata omitted -> absent from body (unchanged behavior for callers)
        assert "metadata" not in call_args[1]["json"]
        assert isinstance(result, CreateEpisodicResult)

    def test_create_episodic_includes_metadata_when_provided(self, mock_httpx_client):
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "ep123",
            "title": "Test episode",
            "has_embedding": True,
            "acknowledged": True,
        }
        mock_httpx_client.post.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        client.create_episodic(
            title="Test episode",
            content="Content",
            summary_text="Summary",
            org_id="org1",
            user_id="user1",
            session_id="sess1",
            project_id="proj1",
            metadata={"channel": "web"},
        )

        call_args = mock_httpx_client.post.call_args
        assert call_args[1]["json"]["metadata"] == {"channel": "web"}

    def test_create_episodic_requires_session_id(self, mock_httpx_client):
        client = MemoryClient("http://localhost:8081")

        with pytest.raises(ValueError, match="session_id is required"):
            client.create_episodic(
                title="Test episode",
                content="Content",
                summary_text="Summary",
                org_id="org1",
                user_id="user1",
                session_id=None,
                project_id="proj1",
            )

        mock_httpx_client.post.assert_not_called()

    def test_fetch_episodic_memories_uses_correct_endpoint(self, mock_httpx_client):
        mock_response = Mock()
        mock_response.json.return_value = {"memories": []}
        mock_httpx_client.post.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        result = client.fetch_episodic_memories(
            query="test query",
            org_id="org1",
            project_id="proj1",
        )

        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        assert (
            call_args[0][0] == "http://localhost:8081/api/v1/memory/retrieval/episodic"
        )
        assert isinstance(result, list)

    def test_list_episodic(self, mock_httpx_client):
        mock_response = Mock()
        mock_response.json.return_value = {"entries": [{"id": "ep1"}, {"id": "ep2"}]}
        mock_httpx_client.get.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        result = client.list_episodic(
            org_id="org1", project_id="proj1", user_id="user1"
        )

        mock_httpx_client.get.assert_called_once()
        call_args = mock_httpx_client.get.call_args
        assert call_args[0][0] == "http://localhost:8081/api/v1/memory/episodic"
        assert len(result) == 2

    def test_list_episodic_omits_unset_session_id(self, mock_httpx_client):
        # A None session_id must not be sent as a query param (which
        # would filter to the session-unscoped null bucket server-side).
        mock_response = Mock()
        mock_response.json.return_value = {"entries": []}
        mock_httpx_client.get.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        client.list_episodic(org_id="org1", project_id="proj1", user_id="user1")

        params = mock_httpx_client.get.call_args[1]["params"]
        assert "session_id" not in params

    def test_list_episodic_includes_explicit_session_id(self, mock_httpx_client):
        mock_response = Mock()
        mock_response.json.return_value = {"entries": []}
        mock_httpx_client.get.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        client.list_episodic(
            org_id="org1", project_id="proj1", user_id="user1", session_id="sess1"
        )

        params = mock_httpx_client.get.call_args[1]["params"]
        assert params["session_id"] == "sess1"

    def test_list_episodic_omits_blank_session_id(self, mock_httpx_client):
        # A blank/whitespace session_id counts as unset (matches _normalize), so
        # it must not be sent as an empty query param either.
        mock_response = Mock()
        mock_response.json.return_value = {"entries": []}
        mock_httpx_client.get.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        client.list_episodic(
            org_id="org1", project_id="proj1", user_id="user1", session_id="   "
        )

        params = mock_httpx_client.get.call_args[1]["params"]
        assert "session_id" not in params


class TestTaxonomicMemory:
    def test_create_taxonomic_success(self, mock_httpx_client):
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "tax123",
            "domain": "insurance",
            "term": "premium",
            "has_embedding": True,
            "acknowledged": True,
        }
        mock_httpx_client.post.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        result = client.create_taxonomic(
            domain="insurance",
            term="premium",
            definition="Amount paid",
            org_id="org1",
            user_id="user1",
            project_id="proj1",
        )

        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        assert call_args[0][0] == "http://localhost:8081/api/v1/memory/taxonomic"
        assert isinstance(result, CreateTaxonomicResult)

    def test_fetch_taxonomic_memories_uses_correct_endpoint(self, mock_httpx_client):
        """Verify taxonomic fetch uses retrieval/taxonomic NOT procedural."""
        mock_response = Mock()
        mock_response.json.return_value = {"memories": []}
        mock_httpx_client.post.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        result = client.fetch_taxonomic_memories(
            query="test query",
            org_id="org1",
            project_id="proj1",
            domain="insurance",
        )

        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        # This is the critical assertion - should be taxonomic, not procedural
        assert (
            call_args[0][0] == "http://localhost:8081/api/v1/memory/retrieval/taxonomic"
        )
        assert call_args[1]["json"]["query"] == "test query"
        assert call_args[1]["json"]["domain"] == "insurance"
        assert isinstance(result, list)

    def test_fetch_taxonomic_memories_forwards_user_id(self, mock_httpx_client):
        """Without user_id the server matches every user's taxonomic entries."""
        mock_response = Mock()
        mock_response.json.return_value = {"memories": []}
        mock_httpx_client.post.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        client.fetch_taxonomic_memories(
            query="test query",
            org_id="org1",
            project_id="proj1",
            user_id="user-b",
        )

        assert mock_httpx_client.post.call_args[1]["json"]["user_id"] == "user-b"

    def test_get_taxonomic_by_term(self, mock_httpx_client):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "entries": [
                {"term": "premium", "definition": "Amount paid"},
                {"term": "deductible", "definition": "Out of pocket"},
            ]
        }
        mock_httpx_client.get.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        result = client.get_taxonomic(
            org_id="org1",
            project_id="proj1",
            domain="insurance",
            term="premium",
        )

        mock_httpx_client.get.assert_called_once()
        call_args = mock_httpx_client.get.call_args
        assert call_args[0][0] == "http://localhost:8081/api/v1/memory/taxonomic"
        assert result["term"] == "premium"

    def test_get_distinct_domains(self, mock_httpx_client):
        mock_response = Mock()
        mock_response.json.return_value = {"domains": ["insurance", "banking"]}
        mock_httpx_client.get.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        result = client.get_distinct_domains(org_id="org1", project_id="proj1")

        mock_httpx_client.get.assert_called_once()
        call_args = mock_httpx_client.get.call_args
        assert (
            call_args[0][0] == "http://localhost:8081/api/v1/memory/taxonomic/domains"
        )
        assert result == ["insurance", "banking"]


class TestContextBuilding:
    def test_build_context_success(self, mock_httpx_client):
        mock_response = Mock()
        mock_response.json.return_value = {
            "formatted_context": "Context text",
            "metadata": {"total_chunks": 5},
        }
        mock_httpx_client.post.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        result = client.build_context(
            query="test query",
            session_id="sess1",
            org_id="org1",
            user_id="user1",
            project_id="proj1",
        )

        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        assert (
            call_args[0][0] == "http://localhost:8081/api/v1/memory/retrieval/context"
        )
        assert isinstance(result, ContextResponse)

    def test_build_context_requires_session_id(self, mock_httpx_client):
        client = MemoryClient("http://localhost:8081")

        with pytest.raises(ValueError, match="session_id is required"):
            client.build_context(
                query="test query",
                session_id=None,
                org_id="org1",
                user_id="user1",
                project_id="proj1",
            )

        mock_httpx_client.post.assert_not_called()

    def test_build_context_serializes_max_tokens_when_explicit(self, mock_httpx_client):
        mock_response = Mock()
        mock_response.json.return_value = {
            "formatted_context": "Context text",
            "metadata": {},
        }
        mock_httpx_client.post.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        client.build_context(
            query="test query",
            session_id="sess1",
            org_id="org1",
            user_id="user1",
            project_id="proj1",
            max_tokens=2048,
        )

        body = mock_httpx_client.post.call_args[1]["json"]
        assert body["max_tokens"] == 2048

    def test_build_context_omits_max_tokens_when_absent(self, mock_httpx_client):
        mock_response = Mock()
        mock_response.json.return_value = {
            "formatted_context": "Context text",
            "metadata": {},
        }
        mock_httpx_client.post.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        client.build_context(
            query="test query",
            session_id="sess1",
            org_id="org1",
            user_id="user1",
            project_id="proj1",
        )

        body = mock_httpx_client.post.call_args[1]["json"]
        assert "max_tokens" not in body

    def test_build_context_serializes_format_style_and_include_memories(
        self, mock_httpx_client
    ):
        mock_response = Mock()
        mock_response.json.return_value = {
            "formatted_context": "Context text",
            "metadata": {},
        }
        mock_httpx_client.post.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        client.build_context(
            query="test query",
            session_id="sess1",
            org_id="org1",
            user_id="user1",
            project_id="proj1",
            format_style="openai",
            include_memories=True,
        )

        body = mock_httpx_client.post.call_args[1]["json"]
        assert body["format_style"] == "openai"
        assert body["include_memories"] is True

    def test_build_context_defaults_include_memories_false(self, mock_httpx_client):
        mock_response = Mock()
        mock_response.json.return_value = {
            "formatted_context": "Context text",
            "metadata": {},
        }
        mock_httpx_client.post.return_value = mock_response

        client = MemoryClient("http://localhost:8081")
        client.build_context(
            query="test query",
            session_id="sess1",
            org_id="org1",
            user_id="user1",
            project_id="proj1",
        )

        body = mock_httpx_client.post.call_args[1]["json"]
        assert body["include_memories"] is False

    @pytest.mark.parametrize("format_style", ["jinj2", "", "markdown"])
    def test_build_context_rejects_invalid_format_style(
        self, mock_httpx_client, format_style
    ):
        client = MemoryClient("http://localhost:8081")

        with pytest.raises(ValueError, match="^format_style must be one of:"):
            client.build_context(
                query="test query",
                session_id="sess1",
                org_id="org1",
                user_id="user1",
                project_id="proj1",
                format_style=format_style,
            )

        mock_httpx_client.post.assert_not_called()

    @pytest.mark.parametrize(
        "max_tokens",
        [0, -1, True, False, 1.5, float("nan"), float("inf"), float("-inf")],
    )
    def test_build_context_rejects_invalid_max_tokens(
        self, mock_httpx_client, max_tokens
    ):
        client = MemoryClient("http://localhost:8081")

        with pytest.raises(ValueError, match="^max_tokens must be a positive integer$"):
            client.build_context(
                query="test query",
                session_id="sess1",
                org_id="org1",
                user_id="user1",
                project_id="proj1",
                max_tokens=max_tokens,
            )

        mock_httpx_client.post.assert_not_called()


class TestContext2Building:
    def _ok_response(self):
        mock_response = Mock()
        mock_response.json.return_value = {
            "formatted_context": "Context text",
            "metadata": {},
        }
        return mock_response

    def test_build_context2_success_posts_to_context2_route(self, mock_httpx_client):
        mock_httpx_client.post.return_value = self._ok_response()

        client = MemoryClient("http://localhost:8081")
        result = client.build_context2(
            query="test query",
            sources=[SourceSpec(source="semantic", mode="hybrid", top_k=20)],
            org_id="org1",
            user_id="user1",
            project_id="proj1",
        )

        call_args = mock_httpx_client.post.call_args
        assert (
            call_args[0][0]
            == "http://localhost:8081/api/v1/memory/retrieval/context-from-sources"
        )
        assert isinstance(result, ContextResponse)

    def test_build_context2_serializes_sources_and_rerank(self, mock_httpx_client):
        mock_httpx_client.post.return_value = self._ok_response()

        client = MemoryClient("http://localhost:8081")
        client.build_context2(
            query="q",
            sources=[
                SourceSpec(source="semantic", mode="hybrid", top_k=15),
                SourceSpec(source="episodic", metadata_filter={"topic": "x"}),
            ],
            org_id="org1",
            user_id="user1",
            project_id="proj1",
            rerank=True,
        )

        body = mock_httpx_client.post.call_args[1]["json"]
        assert body["rerank"] is True
        assert body["sources"] == [
            {"source": "semantic", "mode": "hybrid", "top_k": 15},
            {
                "source": "episodic",
                "mode": "semantic",
                "metadata_filter": {"topic": "x"},
                "top_k": 20,
            },
        ]

    def test_build_context2_omits_session_id_when_absent(self, mock_httpx_client):
        mock_httpx_client.post.return_value = self._ok_response()

        client = MemoryClient("http://localhost:8081")
        client.build_context2(
            query="q",
            sources=[SourceSpec(source="semantic")],
            org_id="org1",
            user_id="user1",
            project_id="proj1",
        )

        body = mock_httpx_client.post.call_args[1]["json"]
        assert "session_id" not in body

    def test_build_context2_includes_session_id_when_given(self, mock_httpx_client):
        mock_httpx_client.post.return_value = self._ok_response()

        client = MemoryClient("http://localhost:8081")
        client.build_context2(
            query="q",
            sources=[SourceSpec(source="stm")],
            org_id="org1",
            user_id="user1",
            project_id="proj1",
            session_id="sess1",
        )

        body = mock_httpx_client.post.call_args[1]["json"]
        assert body["session_id"] == "sess1"

    def test_build_context2_serializes_max_tokens_when_explicit(
        self, mock_httpx_client
    ):
        mock_httpx_client.post.return_value = self._ok_response()

        client = MemoryClient("http://localhost:8081")
        client.build_context2(
            query="q",
            sources=[SourceSpec(source="semantic")],
            org_id="org1",
            user_id="user1",
            project_id="proj1",
            max_tokens=2048,
        )

        body = mock_httpx_client.post.call_args[1]["json"]
        assert body["max_tokens"] == 2048

    def test_build_context2_omits_max_tokens_when_absent(self, mock_httpx_client):
        mock_httpx_client.post.return_value = self._ok_response()

        client = MemoryClient("http://localhost:8081")
        client.build_context2(
            query="q",
            sources=[SourceSpec(source="semantic")],
            org_id="org1",
            user_id="user1",
            project_id="proj1",
        )

        body = mock_httpx_client.post.call_args[1]["json"]
        assert "max_tokens" not in body

    def test_build_context2_serializes_format_style_and_include_memories(
        self, mock_httpx_client
    ):
        mock_httpx_client.post.return_value = self._ok_response()

        client = MemoryClient("http://localhost:8081")
        client.build_context2(
            query="q",
            sources=[SourceSpec(source="semantic")],
            org_id="org1",
            user_id="user1",
            project_id="proj1",
            format_style="jinja2",
            include_memories=True,
        )

        body = mock_httpx_client.post.call_args[1]["json"]
        assert body["format_style"] == "jinja2"
        assert body["include_memories"] is True

    def test_build_context2_omits_format_style_when_absent(self, mock_httpx_client):
        mock_httpx_client.post.return_value = self._ok_response()

        client = MemoryClient("http://localhost:8081")
        client.build_context2(
            query="q",
            sources=[SourceSpec(source="semantic")],
            org_id="org1",
            user_id="user1",
            project_id="proj1",
        )

        body = mock_httpx_client.post.call_args[1]["json"]
        assert "format_style" not in body
        assert body["include_memories"] is False

    @pytest.mark.parametrize("format_style", ["jinj2", "", "markdown"])
    def test_build_context2_rejects_invalid_format_style(
        self, mock_httpx_client, format_style
    ):
        client = MemoryClient("http://localhost:8081")

        with pytest.raises(ValueError, match="^format_style must be one of:"):
            client.build_context2(
                query="q",
                sources=[SourceSpec(source="semantic")],
                org_id="org1",
                user_id="user1",
                project_id="proj1",
                format_style=format_style,
            )

        mock_httpx_client.post.assert_not_called()

    @pytest.mark.parametrize(
        "max_tokens",
        [0, -1, True, False, 1.5, float("nan"), float("inf"), float("-inf")],
    )
    def test_build_context2_rejects_invalid_max_tokens(
        self, mock_httpx_client, max_tokens
    ):
        client = MemoryClient("http://localhost:8081")

        with pytest.raises(ValueError, match="^max_tokens must be a positive integer$"):
            client.build_context2(
                query="q",
                sources=[SourceSpec(source="semantic")],
                org_id="org1",
                user_id="user1",
                project_id="proj1",
                max_tokens=max_tokens,
            )

        mock_httpx_client.post.assert_not_called()

    def test_build_context2_rejects_empty_sources(self, mock_httpx_client):
        client = MemoryClient("http://localhost:8081")

        with pytest.raises(ValueError, match="at least one SourceSpec"):
            client.build_context2(
                query="q",
                sources=[],
                org_id="org1",
                user_id="user1",
                project_id="proj1",
            )

        mock_httpx_client.post.assert_not_called()

    def test_build_context2_rejects_duplicate_source(self, mock_httpx_client):
        client = MemoryClient("http://localhost:8081")

        with pytest.raises(ValueError, match="at most once"):
            client.build_context2(
                query="q",
                sources=[
                    SourceSpec(source="semantic", mode="text"),
                    SourceSpec(source="semantic", mode="hybrid"),
                ],
                org_id="org1",
                user_id="user1",
                project_id="proj1",
            )

        mock_httpx_client.post.assert_not_called()

    @pytest.mark.parametrize("session_id", [None, "", "   "])
    def test_build_context2_rejects_stm_without_session(
        self, mock_httpx_client, session_id
    ):
        client = MemoryClient("http://localhost:8081")

        with pytest.raises(MemoryIdentityError, match="stm"):
            client.build_context2(
                query="q",
                sources=[SourceSpec(source="stm")],
                org_id="org1",
                user_id="user1",
                project_id="proj1",
                session_id=session_id,
            )

        mock_httpx_client.post.assert_not_called()


class TestContextManager:
    def test_context_manager_closes_client(self, mock_httpx_client):
        with MemoryClient("http://localhost:8081") as client:
            assert client._base_url == "http://localhost:8081"

        mock_httpx_client.close.assert_called_once()


class TestRequestBodyFields:
    """Verify that project_id, user_id, and visibility are sent correctly in request
    bodies."""

    def _ok_response(self, mock_httpx_client, data):
        resp = Mock()
        resp.json.return_value = data
        mock_httpx_client.post.return_value = resp
        mock_httpx_client.get.return_value = resp
        return resp

    def test_write_turn_includes_project_id_and_user_id(self, mock_httpx_client):
        self._ok_response(
            mock_httpx_client,
            {"id": "t1", "session_id": "s1", "turn_seq": 1, "acknowledged": True},
        )
        client = MemoryClient("http://localhost:8081")
        client.write_turn(
            session_id="s1",
            role="user",
            org_id="org1",
            user_id="user1",
            project_id="proj1",
            content="hi",
        )
        body = mock_httpx_client.post.call_args[1]["json"]
        assert body["user_id"] == "user1"
        assert body["project_id"] == "proj1"
        assert body["org_id"] == "org1"

    def test_create_semantic_includes_project_id_user_id_visibility(
        self, mock_httpx_client
    ):
        self._ok_response(
            mock_httpx_client,
            {
                "id": "m1",
                "label": "l",
                "text": "t",
                "org_id": "o",
                "user_id": "u",
                "visibility": "private",
                "has_embedding": False,
            },
        )
        client = MemoryClient("http://localhost:8081")
        client.create_semantic(
            label="l",
            text="t",
            org_id="org1",
            user_id="user1",
            project_id="proj1",
            visibility="private",
        )
        body = mock_httpx_client.post.call_args[1]["json"]
        assert body["user_id"] == "user1"
        assert body["project_id"] == "proj1"
        assert body["visibility"] == "private"
        assert body["org_id"] == "org1"

    def test_create_episodic_includes_project_id_user_id_visibility(
        self, mock_httpx_client
    ):
        self._ok_response(
            mock_httpx_client,
            {"id": "e1", "title": "t", "has_embedding": False},
        )
        client = MemoryClient("http://localhost:8081")
        client.create_episodic(
            title="t",
            content="c",
            summary_text="s",
            org_id="org1",
            user_id="user1",
            session_id="sess1",
            project_id="proj1",
            visibility="private",
        )
        body = mock_httpx_client.post.call_args[1]["json"]
        assert body["user_id"] == "user1"
        assert body["project_id"] == "proj1"
        assert body["visibility"] == "private"

    def test_create_taxonomic_includes_project_id_user_id(self, mock_httpx_client):
        self._ok_response(
            mock_httpx_client,
            {"id": "x1", "domain": "d", "term": "t", "has_embedding": False},
        )
        client = MemoryClient("http://localhost:8081")
        client.create_taxonomic(
            domain="d",
            term="t",
            definition="def",
            org_id="org1",
            user_id="user1",
            project_id="proj1",
        )
        body = mock_httpx_client.post.call_args[1]["json"]
        assert body["user_id"] == "user1"
        assert body["project_id"] == "proj1"

    def test_write_turn_forwards_metadata(self, mock_httpx_client):
        self._ok_response(
            mock_httpx_client,
            {"id": "t1", "session_id": "s1", "turn_seq": 1, "acknowledged": True},
        )
        client = MemoryClient("http://localhost:8081")
        client.write_turn(
            session_id="s1",
            role="user",
            org_id="org1",
            user_id="user1",
            project_id="proj1",
            content="hi",
            metadata={"channel": "slack"},
        )
        body = mock_httpx_client.post.call_args[1]["json"]
        assert body["metadata"] == {"channel": "slack"}

    def test_write_turn_omits_metadata_when_absent(self, mock_httpx_client):
        self._ok_response(
            mock_httpx_client,
            {"id": "t1", "session_id": "s1", "turn_seq": 1, "acknowledged": True},
        )
        client = MemoryClient("http://localhost:8081")
        client.write_turn(
            session_id="s1",
            role="user",
            org_id="org1",
            user_id="user1",
            project_id="proj1",
            content="hi",
        )
        body = mock_httpx_client.post.call_args[1]["json"]
        assert "metadata" not in body

    def test_create_taxonomic_forwards_metadata(self, mock_httpx_client):
        self._ok_response(
            mock_httpx_client,
            {"id": "x1", "domain": "d", "term": "t", "has_embedding": False},
        )
        client = MemoryClient("http://localhost:8081")
        client.create_taxonomic(
            domain="d",
            term="t",
            definition="def",
            org_id="org1",
            user_id="user1",
            project_id="proj1",
            metadata={"k": "v"},
        )
        body = mock_httpx_client.post.call_args[1]["json"]
        assert body["metadata"] == {"k": "v"}

    def test_create_procedural_forwards_metadata(self, mock_httpx_client):
        self._ok_response(
            mock_httpx_client,
            {"id": "p1", "procedure": "p", "has_embedding": False},
        )
        client = MemoryClient("http://localhost:8081")
        client.create_procedural(
            procedure="p",
            description="d",
            content="c",
            org_id="org1",
            user_id="user1",
            project_id="proj1",
            metadata={"k": "v"},
        )
        body = mock_httpx_client.post.call_args[1]["json"]
        assert body["metadata"] == {"k": "v"}

    def test_fetch_semantic_includes_project_id_user_id_visibility(
        self, mock_httpx_client
    ):
        self._ok_response(mock_httpx_client, {"memories": []})
        client = MemoryClient("http://localhost:8081")
        client.fetch_semantic_memories(
            query="q",
            org_id="org1",
            project_id="proj1",
            user_id="user1",
            visibility="private",
        )
        body = mock_httpx_client.post.call_args[1]["json"]
        assert body["user_id"] == "user1"
        assert body["project_id"] == "proj1"
        assert body["visibility"] == "private"

    def test_fetch_episodic_includes_project_id_user_id_visibility(
        self, mock_httpx_client
    ):
        self._ok_response(mock_httpx_client, {"memories": []})
        client = MemoryClient("http://localhost:8081")
        client.fetch_episodic_memories(
            query="q",
            org_id="org1",
            project_id="proj1",
            user_id="user1",
            visibility="org",
        )
        body = mock_httpx_client.post.call_args[1]["json"]
        assert body["user_id"] == "user1"
        assert body["project_id"] == "proj1"
        assert body["visibility"] == "org"

    def test_build_context_includes_project_id_user_id(self, mock_httpx_client):
        self._ok_response(mock_httpx_client, {"formatted_context": "", "metadata": {}})
        client = MemoryClient("http://localhost:8081")
        client.build_context(
            query="q",
            session_id="s1",
            org_id="org1",
            user_id="user1",
            project_id="proj1",
        )
        body = mock_httpx_client.post.call_args[1]["json"]
        assert body["user_id"] == "user1"
        assert body["project_id"] == "proj1"


class TestProceduralLookupByName:
    """Regression tests for update/delete by procedure name forwarding project_id."""

    def _ok_response(self, mock_httpx_client, data, status=200):
        mock_response = Mock()
        mock_response.status_code = status
        mock_response.json.return_value = data
        return mock_response

    def test_update_procedural_by_name_passes_project_id(self, mock_httpx_client):
        """update_procedural by name must forward project_id to get_procedural."""
        existing = {"id": "mem-1", "procedure": "my-proc"}
        updated = {"id": "mem-1", "procedure": "my-proc", "content": "new"}

        # First call: GET for get_procedural; second call: PATCH for update
        mock_httpx_client.get.return_value = self._ok_response(
            mock_httpx_client, {"entries": [existing]}
        )
        mock_httpx_client.patch.return_value = self._ok_response(
            mock_httpx_client, updated
        )

        client = MemoryClient("http://localhost:8081")
        client.update_procedural(
            "org-1",
            "proj-1",
            procedure="my-proc",
            content="new",
            user_id="user-1",
        )

        get_params = mock_httpx_client.get.call_args[1]["params"]
        assert get_params["project_id"] == "proj-1"
        assert get_params["org_id"] == "org-1"

    def test_delete_procedural_by_name_passes_project_id(self, mock_httpx_client):
        """delete_procedural by name must forward project_id to get_procedural."""
        existing = {"id": "mem-1", "procedure": "my-proc"}
        delete_result = {"deleted_count": 1, "acknowledged": True}

        mock_httpx_client.get.return_value = self._ok_response(
            mock_httpx_client, {"entries": [existing]}
        )
        mock_httpx_client.delete.return_value = self._ok_response(
            mock_httpx_client, delete_result
        )

        client = MemoryClient("http://localhost:8081")
        client.delete_procedural(
            "org-1", "proj-1", procedure="my-proc", user_id="user-1"
        )

        get_params = mock_httpx_client.get.call_args[1]["params"]
        assert get_params["project_id"] == "proj-1"
        assert get_params["org_id"] == "org-1"

        delete_params = mock_httpx_client.delete.call_args[1]["params"]
        assert delete_params["project_id"] == "proj-1"

    def test_update_procedural_by_name_forwards_identity(self, mock_httpx_client):
        """The by-name lookup must carry user_id so OE does not reject it, but
        must not filter by visibility so an update can change visibility."""
        existing = {"id": "mem-1", "procedure": "my-proc", "visibility": "private"}
        updated = {
            "id": "mem-1",
            "procedure": "my-proc",
            "content": "new",
            "visibility": "org",
        }

        mock_httpx_client.get.return_value = self._ok_response(
            mock_httpx_client, {"entries": [existing]}
        )
        mock_httpx_client.patch.return_value = self._ok_response(
            mock_httpx_client, updated
        )

        client = MemoryClient("http://localhost:8081")
        client.update_procedural(
            "org-1",
            "proj-1",
            procedure="my-proc",
            content="new",
            user_id="user-1",
            visibility="org",
        )

        get_params = mock_httpx_client.get.call_args[1]["params"]
        assert get_params["user_id"] == "user-1"
        assert "visibility" not in get_params

        patch_body = mock_httpx_client.patch.call_args[1]["json"]
        assert patch_body["user_id"] == "user-1"
        assert patch_body["visibility"] == "org"

    def test_update_procedural_with_id_skips_lookup(self, mock_httpx_client):
        """When id is supplied, no identity-less by-name GET is issued."""
        updated = {"id": "mem-1", "procedure": "my-proc", "content": "new"}
        mock_httpx_client.patch.return_value = self._ok_response(
            mock_httpx_client, updated
        )

        client = MemoryClient("http://localhost:8081")
        client.update_procedural(
            "org-1",
            "proj-1",
            id="mem-1",
            content="new",
            user_id="user-1",
            visibility="org",
        )

        mock_httpx_client.get.assert_not_called()
        patch_body = mock_httpx_client.patch.call_args[1]["json"]
        assert patch_body["user_id"] == "user-1"

    def test_update_procedural_forwards_metadata(self, mock_httpx_client):
        updated = {"id": "mem-1", "procedure": "my-proc"}
        mock_httpx_client.patch.return_value = self._ok_response(
            mock_httpx_client, updated
        )

        client = MemoryClient("http://localhost:8081")
        client.update_procedural(
            "org-1",
            "proj-1",
            id="mem-1",
            user_id="user-1",
            metadata={"k": "v"},
        )

        patch_body = mock_httpx_client.patch.call_args[1]["json"]
        assert patch_body["metadata"] == {"k": "v"}

    def test_update_procedural_omits_metadata_when_absent(self, mock_httpx_client):
        updated = {"id": "mem-1", "procedure": "my-proc"}
        mock_httpx_client.patch.return_value = self._ok_response(
            mock_httpx_client, updated
        )

        client = MemoryClient("http://localhost:8081")
        client.update_procedural(
            "org-1",
            "proj-1",
            id="mem-1",
            user_id="user-1",
            content="new",
        )

        patch_body = mock_httpx_client.patch.call_args[1]["json"]
        assert "metadata" not in patch_body

    def test_delete_procedural_forwards_identity(self, mock_httpx_client):
        """user_id/visibility must reach both the by-name lookup GET and the
        DELETE query (the OE identity rule applies to GET and DELETE)."""
        existing = {"id": "mem-1", "procedure": "my-proc"}
        delete_result = {"deleted_count": 1, "acknowledged": True}

        mock_httpx_client.get.return_value = self._ok_response(
            mock_httpx_client, {"entries": [existing]}
        )
        mock_httpx_client.delete.return_value = self._ok_response(
            mock_httpx_client, delete_result
        )

        client = MemoryClient("http://localhost:8081")
        client.delete_procedural(
            "org-1",
            "proj-1",
            procedure="my-proc",
            user_id="user-1",
            visibility="org",
        )

        get_params = mock_httpx_client.get.call_args[1]["params"]
        assert get_params["user_id"] == "user-1"
        assert get_params["visibility"] == "org"

        delete_params = mock_httpx_client.delete.call_args[1]["params"]
        assert delete_params["user_id"] == "user-1"
        assert delete_params["visibility"] == "org"

    def test_update_procedural_requires_user_id(self, mock_httpx_client):
        """Writes without user_id are rejected client-side: they would reach
        memory-server with no identity for the OE write policy to check."""
        client = MemoryClient("http://localhost:8081")
        with pytest.raises(ValueError, match="user_id"):
            client.update_procedural("org-1", "proj-1", id="mem-1", content="new")

        mock_httpx_client.get.assert_not_called()
        mock_httpx_client.patch.assert_not_called()

    def test_delete_procedural_requires_user_id(self, mock_httpx_client):
        client = MemoryClient("http://localhost:8081")
        with pytest.raises(ValueError, match="user_id"):
            client.delete_procedural("org-1", "proj-1", id="mem-1")

        mock_httpx_client.get.assert_not_called()
        mock_httpx_client.delete.assert_not_called()

    def test_update_procedural_by_name_not_found_raises(self, mock_httpx_client):
        """A name that resolves to nothing must raise, never PATCH."""
        mock_httpx_client.get.return_value = self._ok_response(
            mock_httpx_client, {"entries": []}
        )

        client = MemoryClient("http://localhost:8081")
        with pytest.raises(ValueError, match="not found"):
            client.update_procedural(
                "org-1", "proj-1", procedure="ghost", content="new", user_id="user-1"
            )

        mock_httpx_client.patch.assert_not_called()

    def test_delete_procedural_by_name_not_found_raises(self, mock_httpx_client):
        """A name that resolves to nothing must raise, never DELETE."""
        mock_httpx_client.get.return_value = self._ok_response(
            mock_httpx_client, {"entries": []}
        )

        client = MemoryClient("http://localhost:8081")
        with pytest.raises(ValueError, match="not found"):
            client.delete_procedural(
                "org-1", "proj-1", procedure="ghost", user_id="user-1"
            )

        mock_httpx_client.delete.assert_not_called()


class TestMemoryIdUrlEncoding:
    """Memory ids must be URL-encoded in path segments so '?', '#', and '../'
    cannot forge requests to other endpoints."""

    def _ok_response(self, data, status=200):
        mock_response = Mock()
        mock_response.status_code = status
        mock_response.json.return_value = data
        return mock_response

    def test_get_semantic_encodes_query_and_fragment_in_id(self, mock_httpx_client):
        mock_httpx_client.get.return_value = self._ok_response({"id": "mem-1"})

        client = MemoryClient("http://localhost:8081")
        client.get_semantic("org-1", "proj-1", id="fake-id?org_id=evil#frag")

        url = mock_httpx_client.get.call_args[0][0]
        assert url == (
            "http://localhost:8081/api/v1/memory/semantic/"
            "fake-id%3Forg_id%3Devil%23frag"
        )
        # Tenant scoping stays in the query dict, not swallowed by the id.
        params = mock_httpx_client.get.call_args[1]["params"]
        assert params["org_id"] == "org-1"
        assert params["project_id"] == "proj-1"

    def test_get_procedural_encodes_query_and_fragment_in_id(self, mock_httpx_client):
        mock_httpx_client.get.return_value = self._ok_response({"id": "mem-1"})

        client = MemoryClient("http://localhost:8081")
        client.get_procedural("org-1", "proj-1", id="fake-id?org_id=evil#frag")

        url = mock_httpx_client.get.call_args[0][0]
        assert url == (
            "http://localhost:8081/api/v1/memory/procedural/"
            "fake-id%3Forg_id%3Devil%23frag"
        )

    def test_update_semantic_encodes_server_derived_id(self, mock_httpx_client):
        mock_httpx_client.get.return_value = self._ok_response(
            {"entries": [{"id": "../../evil", "label": "api-docs"}]}
        )
        mock_httpx_client.patch.return_value = self._ok_response({"id": "x"})

        client = MemoryClient("http://localhost:8081")
        client.update_semantic("org-1", "proj-1", label="api-docs", text="new")

        url = mock_httpx_client.patch.call_args[0][0]
        assert url == "http://localhost:8081/api/v1/memory/semantic/..%2F..%2Fevil"

    def test_update_procedural_encodes_traversal_id(self, mock_httpx_client):
        mock_httpx_client.patch.return_value = self._ok_response({"id": "x"})

        client = MemoryClient("http://localhost:8081")
        client.update_procedural(
            "org-1", "proj-1", id="../../admin/endpoint", user_id="user-1"
        )

        url = mock_httpx_client.patch.call_args[0][0]
        assert url == (
            "http://localhost:8081/api/v1/memory/procedural/..%2F..%2Fadmin%2Fendpoint"
        )

    def test_delete_procedural_encodes_traversal_id(self, mock_httpx_client):
        mock_httpx_client.delete.return_value = self._ok_response(
            {"deleted_count": 1, "acknowledged": True}
        )

        client = MemoryClient("http://localhost:8081")
        client.delete_procedural(
            "org-1", "proj-1", id="../../admin/endpoint", user_id="user-1"
        )

        url = mock_httpx_client.delete.call_args[0][0]
        assert url == (
            "http://localhost:8081/api/v1/memory/procedural/..%2F..%2Fadmin%2Fendpoint"
        )

    @pytest.mark.parametrize("bad_id", [".", ".."])
    def test_bare_dot_segment_ids_rejected_before_any_request(
        self, mock_httpx_client, bad_id
    ):
        """quote() leaves dots untouched, so '.'/'..' would survive encoding
        and collapse into a parent-path rewrite — refuse them."""
        client = MemoryClient("http://localhost:8081")

        with pytest.raises(ValueError, match="bare dot segment"):
            client.get_semantic("org-1", "proj-1", id=bad_id)
        with pytest.raises(ValueError, match="bare dot segment"):
            client.get_procedural("org-1", "proj-1", id=bad_id)
        with pytest.raises(ValueError, match="bare dot segment"):
            client.delete_procedural("org-1", "proj-1", id=bad_id, user_id="user-1")

        mock_httpx_client.get.assert_not_called()
        mock_httpx_client.delete.assert_not_called()
