"""Tests for type definitions."""

from datetime import datetime, timezone

from agent_engine_runner_shared.logging import ExecutionLog, ExecutionStatus


class TestExecutionStatus:
    """Tests for ExecutionStatus enum."""

    def test_values(self):
        """Test enum values."""
        assert ExecutionStatus.STARTED == "started"
        assert ExecutionStatus.SUCCESS == "success"
        assert ExecutionStatus.ERROR == "error"
        assert ExecutionStatus.SUSPENDED == "suspend"

    def test_string_comparison(self):
        """Test string comparison."""
        assert ExecutionStatus.STARTED == "started"
        assert "success" == ExecutionStatus.SUCCESS


class TestExecutionLog:
    """Tests for ExecutionLog."""

    def test_minimal(self):
        """Test minimal log entry."""
        log = ExecutionLog(
            id="test-123",
            execution_id="exec-123",
            tool="my_tool",
            status=ExecutionStatus.STARTED,
            timestamp=datetime.now(timezone.utc),
        )

        assert log.id == "test-123"
        assert log.execution_id == "exec-123"
        assert log.tool == "my_tool"
        assert log.status == ExecutionStatus.STARTED
        assert log.inputs is None
        assert log.output is None
        assert log.error is None

    def test_with_inputs_and_output(self):
        """Test log with inputs and output."""
        log = ExecutionLog(
            id="test-456",
            execution_id="exec-456",
            tool="add",
            status=ExecutionStatus.SUCCESS,
            timestamp=datetime.now(timezone.utc),
            inputs={"x": 1, "y": 2},
            output=3,
            duration_ms=1.5,
        )

        assert log.inputs == {"x": 1, "y": 2}
        assert log.output == 3
        assert log.duration_ms == 1.5

    def test_with_error(self):
        """Test log with error."""
        log = ExecutionLog(
            id="test-789",
            execution_id="exec-789",
            tool="failing_tool",
            status=ExecutionStatus.ERROR,
            timestamp=datetime.now(timezone.utc),
            error="Something went wrong",
            duration_ms=10.0,
        )

        assert log.status == ExecutionStatus.ERROR
        assert log.error == "Something went wrong"

    def test_with_context(self):
        """Test log with session/user/thread context."""
        log = ExecutionLog(
            id="test-context",
            execution_id="exec-context",
            tool="tool",
            status=ExecutionStatus.STARTED,
            timestamp=datetime.now(timezone.utc),
            session_id="session-123",
            user_id="user-456",
        )

        assert log.session_id == "session-123"
        assert log.user_id == "user-456"

    def test_model_dump(self):
        """Test serialization."""
        log = ExecutionLog(
            id="test-dump",
            execution_id="exec-dump",
            tool="tool",
            status=ExecutionStatus.SUCCESS,
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            output={"result": "ok"},
        )

        data = log.model_dump(mode="json")

        assert data["id"] == "test-dump"
        assert data["execution_id"] == "exec-dump"
        assert data["status"] == "success"
        assert data["output"] == {"result": "ok"}
