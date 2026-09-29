"""Unit tests for NodeExecutionLogger (framework-neutral)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from agent_engine_sdk.interfaces import BaseExecutionCallback

from agent_engine_runner_shared.node_logger import NodeExecutionLogger


class TestNodeExecutionLoggerProtocol:
    """Verify NodeExecutionLogger satisfies BaseExecutionCallback."""

    def test_satisfies_base_execution_callback(self):
        """NodeExecutionLogger is structurally compatible with BaseExecutionCallback."""
        logger = NodeExecutionLogger(
            oe_url="http://localhost:8080",
            execution_id="exec-1",
        )
        assert isinstance(logger, BaseExecutionCallback)


class TestOnNodeStart:
    """Tests for on_node_start."""

    def test_sends_post_with_correct_payload(self):
        """Sends HTTP POST to OE /node/execution with started status."""
        logger = NodeExecutionLogger(
            oe_url="http://oe:8080",
            execution_id="exec-1",
            session_id="sess-1",
            user_id="user-1",
            org_id="org-1",
            project_id="group-1",
        )

        with patch(
            "agent_engine_runner_shared.node_logger.create_httpx_client_with_tls"
        ) as mock_create:
            mock_client = MagicMock()
            mock_create.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_create.return_value.__exit__ = MagicMock(return_value=False)

            logger.on_node_start(
                node_name="agent_node",
                inputs={"messages": ["hello"]},
                run_id="run-123",
            )

            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://oe:8080/node/execution"
            payload = call_args[1]["json"]
            assert payload["execution_id"] == "exec-1"
            assert payload["node_name"] == "agent_node"
            assert payload["status"] == "started"
            assert payload["run_id"] == "run-123"
            assert payload["session_id"] == "sess-1"
            assert payload["user_id"] == "user-1"
            assert payload["org_id"] == "org-1"
            assert payload["project_id"] == "group-1"
            assert "inputs" in payload

    def test_records_start_time(self):
        """Records start time for duration calculation."""
        logger = NodeExecutionLogger(oe_url="http://oe:8080", execution_id="exec-1")

        with patch("agent_engine_runner_shared.node_logger.create_httpx_client_with_tls"):
            logger.on_node_start(
                node_name="agent_node",
                inputs={},
                run_id="run-123",
            )

        assert "run-123" in logger._node_start_times


class TestOnNodeEnd:
    """Tests for on_node_end."""

    def test_calculates_duration_from_start(self):
        """Calculates duration_ms from recorded start time."""
        logger = NodeExecutionLogger(oe_url="http://oe:8080", execution_id="exec-1")
        # Manually set a start time
        logger._node_start_times["run-123"] = datetime.now(timezone.utc) - timedelta(milliseconds=1)

        with patch(
            "agent_engine_runner_shared.node_logger.create_httpx_client_with_tls"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            logger.on_node_end(
                node_name="agent_node",
                outputs={"result": "done"},
                run_id="run-123",
            )

            payload = mock_client.post.call_args[1]["json"]
            assert payload["status"] == "success"
            assert payload["duration_ms"] is not None
            assert payload["duration_ms"] > 0
            assert "run-123" not in logger._node_start_times  # cleaned up

    def test_uses_provided_duration_ms(self):
        """Uses provided duration_ms if given."""
        logger = NodeExecutionLogger(oe_url="http://oe:8080", execution_id="exec-1")

        with patch(
            "agent_engine_runner_shared.node_logger.create_httpx_client_with_tls"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            logger.on_node_end(
                node_name="agent_node",
                outputs={},
                run_id="run-456",
                duration_ms=42.5,
            )

            payload = mock_client.post.call_args[1]["json"]
            assert payload["duration_ms"] == 42.5

    def test_handles_missing_start_time(self):
        """Works even if no start time was recorded."""
        logger = NodeExecutionLogger(oe_url="http://oe:8080", execution_id="exec-1")

        with patch(
            "agent_engine_runner_shared.node_logger.create_httpx_client_with_tls"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            logger.on_node_end(
                node_name="agent_node",
                outputs={},
                run_id="run-unknown",
            )

            payload = mock_client.post.call_args[1]["json"]
            assert payload["status"] == "success"


class TestOnNodeError:
    """Tests for on_node_error."""

    def test_sends_error_payload(self):
        """Sends error status with error message."""
        logger = NodeExecutionLogger(oe_url="http://oe:8080", execution_id="exec-1")

        with patch(
            "agent_engine_runner_shared.node_logger.create_httpx_client_with_tls"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            logger.on_node_error(
                node_name="agent_node",
                error="Something went wrong",
                run_id="run-123",
            )

            payload = mock_client.post.call_args[1]["json"]
            assert payload["status"] == "error"
            assert payload["error"] == "Something went wrong"
            assert payload["node_name"] == "agent_node"

    def test_cleans_up_start_time(self):
        """Removes start time entry on error."""
        logger = NodeExecutionLogger(oe_url="http://oe:8080", execution_id="exec-1")
        logger._node_start_times["run-123"] = datetime.now(timezone.utc)

        with patch("agent_engine_runner_shared.node_logger.create_httpx_client_with_tls"):
            logger.on_node_error(
                node_name="agent_node",
                error="fail",
                run_id="run-123",
            )

        assert "run-123" not in logger._node_start_times


class TestOnNodeSuspend:
    """Tests for on_node_suspend."""

    def test_sends_suspend_payload(self):
        """Sends suspend status with no error field."""
        logger = NodeExecutionLogger(oe_url="http://oe:8080", execution_id="exec-1")

        with patch(
            "agent_engine_runner_shared.node_logger.create_httpx_client_with_tls"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            logger.on_node_suspend(
                node_name="agent_node",
                run_id="run-123",
            )

            payload = mock_client.post.call_args[1]["json"]
            assert payload["status"] == "suspend"
            assert payload["node_name"] == "agent_node"
            assert "error" not in payload

    def test_computes_duration_from_start_time(self):
        """Computes duration_ms when a start time was recorded."""
        logger = NodeExecutionLogger(oe_url="http://oe:8080", execution_id="exec-1")
        logger._node_start_times["run-123"] = datetime.now(timezone.utc) - timedelta(milliseconds=1)

        with patch(
            "agent_engine_runner_shared.node_logger.create_httpx_client_with_tls"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            logger.on_node_suspend(node_name="agent_node", run_id="run-123")

            payload = mock_client.post.call_args[1]["json"]
            assert payload["duration_ms"] is not None
            assert payload["duration_ms"] > 0


class TestOnNodeInterrupted:
    """Tests for on_node_interrupted."""

    def test_sends_interrupted_payload_with_partial_duration(self):
        """Interrupted is terminal-not-failure: partial duration, no error."""
        logger = NodeExecutionLogger(oe_url="http://oe:8080", execution_id="exec-1")
        logger._node_start_times["run-123"] = datetime.now(timezone.utc) - timedelta(
            milliseconds=150
        )

        with patch(
            "agent_engine_runner_shared.node_logger.create_httpx_client_with_tls"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            logger.on_node_interrupted(
                node_name="agent_node",
                run_id="run-123",
            )

            payload = mock_client.post.call_args[1]["json"]
            assert payload["status"] == "interrupted"
            assert payload["node_name"] == "agent_node"
            assert "error" not in payload
            assert payload["duration_ms"] >= 100
            assert "run-123" not in logger._node_start_times

    def test_duration_none_when_no_start_time(self):
        """duration_ms is absent from the payload when no start time was recorded."""
        logger = NodeExecutionLogger(oe_url="http://oe:8080", execution_id="exec-1")

        with patch(
            "agent_engine_runner_shared.node_logger.create_httpx_client_with_tls"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            logger.on_node_suspend(node_name="agent_node", run_id="run-unknown")

            payload = mock_client.post.call_args[1]["json"]
            assert "duration_ms" not in payload

    def test_cleans_up_start_time(self):
        """Removes start time entry on suspend."""
        logger = NodeExecutionLogger(oe_url="http://oe:8080", execution_id="exec-1")
        logger._node_start_times["run-123"] = datetime.now(timezone.utc)

        with patch("agent_engine_runner_shared.node_logger.create_httpx_client_with_tls"):
            logger.on_node_suspend(
                node_name="agent_node",
                run_id="run-123",
            )

        assert "run-123" not in logger._node_start_times


class TestHttpFailureSilence:
    """Tests that HTTP failures are silently caught."""

    def test_http_error_does_not_raise(self):
        """HTTP errors are caught and logged, not raised."""
        logger = NodeExecutionLogger(oe_url="http://oe:8080", execution_id="exec-1")

        with patch(
            "agent_engine_runner_shared.node_logger.create_httpx_client_with_tls"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.side_effect = Exception("Connection refused")
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            # Should not raise
            logger.on_node_start(
                node_name="agent_node",
                inputs={},
                run_id="run-123",
            )


class TestSerializeForJson:
    """Tests for _serialize_for_json."""

    def test_handles_primitives(self):
        """Passes through strings, ints, floats, bools, None."""
        logger = NodeExecutionLogger(oe_url="http://oe:8080", execution_id="exec-1")

        assert logger._serialize_for_json("hello") == "hello"
        assert logger._serialize_for_json(42) == 42
        assert logger._serialize_for_json(3.14) == 3.14
        assert logger._serialize_for_json(True) is True
        assert logger._serialize_for_json(None) is None

    def test_handles_dicts(self):
        """Recursively serializes dicts."""
        logger = NodeExecutionLogger(oe_url="http://oe:8080", execution_id="exec-1")
        result = logger._serialize_for_json({"key": "value", "nested": {"a": 1}})
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_handles_lists(self):
        """Recursively serializes lists."""
        logger = NodeExecutionLogger(oe_url="http://oe:8080", execution_id="exec-1")
        result = logger._serialize_for_json([1, "two", {"three": 3}])
        assert result == [1, "two", {"three": 3}]

    def test_handles_pydantic_model(self):
        """Converts Pydantic model via dict()."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            x: int = 1

        logger = NodeExecutionLogger(oe_url="http://oe:8080", execution_id="exec-1")
        result = logger._serialize_for_json(MyModel())
        assert result == {"x": 1}

    def test_falls_back_to_str(self):
        """Falls back to str() for unknown types."""
        logger = NodeExecutionLogger(oe_url="http://oe:8080", execution_id="exec-1")
        result = logger._serialize_for_json(object())
        assert isinstance(result, str)
