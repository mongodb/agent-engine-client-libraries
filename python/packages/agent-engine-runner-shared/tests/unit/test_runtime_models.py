"""Unit tests for agent_engine_runner_shared API models."""

import pytest

from agent_engine_runner_shared.models import (
    HealthResponse,
    HealthStatus,
    InvokeRequest,
    InvokeResponse,
    StreamChunk,
)

# =============================================================================
# InvokeRequest Model Tests
# =============================================================================


def test_invoke_request_model():
    """Test InvokeRequest model."""
    request = InvokeRequest(
        message="Hello",
        session_id="session-123",
        user_id="user-456",
    )

    assert request.message == "Hello"
    assert request.session_id == "session-123"
    assert request.user_id == "user-456"


def test_invoke_request_defaults():
    """Test InvokeRequest default values."""
    request = InvokeRequest(message="Hello")

    assert request.message == "Hello"
    assert request.session_id is None
    assert request.user_id is None
    assert request.wait is True


def test_invoke_request_model_dump():
    """Test InvokeRequest serialization."""
    request = InvokeRequest(
        message="Test message",
        session_id="s-123",
        user_id="u-456",
    )

    data = request.model_dump()

    assert data["message"] == "Test message"
    assert data["session_id"] == "s-123"
    assert data["user_id"] == "u-456"


# =============================================================================
# InvokeResponse Model Tests
# =============================================================================


def test_invoke_response_model():
    """Test InvokeResponse model."""
    response = InvokeResponse(
        result="Hello back!",
        session_id="session-123",
        user_id="user-456",
    )

    assert response.result == "Hello back!"
    assert response.session_id == "session-123"
    assert response.user_id == "user-456"


def test_invoke_response_with_dict_result():
    """Test InvokeResponse with dict result."""
    response = InvokeResponse(
        result={"key": "value", "count": 42},
        session_id="session-123",
    )

    assert response.result == {"key": "value", "count": 42}


def test_invoke_response_optional_fields():
    """Test InvokeResponse optional fields can be None."""
    response = InvokeResponse(result="Result only")

    assert response.result == "Result only"
    assert response.session_id is None
    assert response.user_id is None


# =============================================================================
# HealthResponse Model Tests
# =============================================================================


def test_health_response_model():
    """Test HealthResponse model."""
    response = HealthResponse(
        status=HealthStatus.HEALTHY,
        component="Test Agent",
        mode="orchestrator",
        version="1.0.0",
    )

    assert response.status == HealthStatus.HEALTHY
    assert response.component == "Test Agent"
    assert response.mode == "orchestrator"
    assert response.version == "1.0.0"


def test_health_response_with_details():
    """Test HealthResponse with details."""
    response = HealthResponse(
        status=HealthStatus.DEGRADED,
        component="Test Agent",
        mode="aer",
        details={"memory": "low", "cpu": "high"},
    )

    assert response.status == HealthStatus.DEGRADED
    assert response.details == {"memory": "low", "cpu": "high"}


# =============================================================================
# StreamChunk Model Tests
# =============================================================================


def test_stream_chunk_text():
    """Test StreamChunk with text content."""
    chunk = StreamChunk(
        chunk_type="text",
        content="Hello, world!",
    )

    assert chunk.chunk_type == "text"
    assert chunk.content == "Hello, world!"
    assert chunk.metadata == {}
    assert chunk.error is None


def test_stream_chunk_metadata():
    """Test StreamChunk with metadata."""
    chunk = StreamChunk(
        chunk_type="metadata",
        metadata={"thread_id": "123", "user_id": "456"},
    )

    assert chunk.chunk_type == "metadata"
    assert chunk.metadata == {"thread_id": "123", "user_id": "456"}
    assert chunk.content == ""


def test_stream_chunk_metadata_allows_artifact_arrays():
    """Structured artifact metadata should survive stream chunk validation."""
    chunk = StreamChunk(
        chunk_type="done",
        metadata={
            "status": "completed",
            "artifacts": [{"id": "chart-1", "kind": "chart"}],
        },
    )

    assert chunk.metadata["artifacts"] == [{"id": "chart-1", "kind": "chart"}]


def test_stream_chunk_error():
    """Test StreamChunk with error."""
    chunk = StreamChunk(
        chunk_type="error",
        error="Something went wrong",
    )

    assert chunk.chunk_type == "error"
    assert chunk.error == "Something went wrong"
    assert chunk.content == ""


def test_stream_chunk_tool_call():
    """Test StreamChunk for tool calls."""
    chunk = StreamChunk(
        chunk_type="tool_call",
        tool_name="my_tool",
        tool_call_id="tc-123",
        content='{"arg": "value"}',
    )

    assert chunk.chunk_type == "tool_call"
    assert chunk.tool_name == "my_tool"
    assert chunk.tool_call_id == "tc-123"


def test_stream_chunk_done():
    """Test StreamChunk done marker."""
    chunk = StreamChunk(chunk_type="done")

    assert chunk.chunk_type == "done"
    assert chunk.content == ""
    assert chunk.metadata == {}


# =============================================================================
# TenantRuntime.get_agent() Tests
# =============================================================================


class TestGetAgent:
    """Tests for TenantRuntime.get_agent() dispatch logic."""

    def test_raises_when_no_app_registered(self):
        """get_agent() raises RuntimeError when no App registered."""
        from agent_engine_runner_shared.runtime import TenantRuntime

        rt = TenantRuntime.__new__(TenantRuntime)
        rt._graph_builder = None

        with pytest.raises(RuntimeError, match="No App registered"):
            rt.get_agent()

    def test_delegates_to_app_get_agent(self):
        """get_agent() delegates to App.get_agent() when _graph_builder has that method."""
        from unittest.mock import Mock, sentinel

        from agent_engine_runner_shared.runtime import TenantRuntime

        rt = TenantRuntime.__new__(TenantRuntime)
        mock_app = Mock()
        mock_app.get_agent.return_value = sentinel.agent
        rt._graph_builder = mock_app

        result = rt.get_agent(callbacks=["cb1"])

        assert result is sentinel.agent
        mock_app.get_agent.assert_called_once_with(callbacks=["cb1"])

    def test_plain_callable_raises(self):
        """get_agent() raises RuntimeError for plain callables (legacy path removed)."""
        from agent_engine_runner_shared.runtime import TenantRuntime

        rt = TenantRuntime.__new__(TenantRuntime)
        rt._graph_builder = lambda: "graph"

        with pytest.raises(RuntimeError, match="must be a BaseApp instance"):
            rt.get_agent()
