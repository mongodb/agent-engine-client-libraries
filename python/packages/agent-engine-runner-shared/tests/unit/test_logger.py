"""Tests for the execution logger."""

import json
from datetime import datetime, timezone

import pytest

from agent_engine_runner_shared.logging import (
    ExecutionLog,
    ExecutionStatus,
    get_log_path,
    log_entry,
    set_tool_logs_collection,
)


@pytest.fixture(autouse=True)
def temp_log_dir(tmp_path, monkeypatch):
    """Use temporary directory for logs."""
    monkeypatch.setenv("AGENTIC_OBSERVABILITY_DIR", str(tmp_path))
    # Reset MongoDB collection
    set_tool_logs_collection(None)
    return tmp_path


def test_get_log_path_default(temp_log_dir):
    """Test default log path."""
    path = get_log_path()

    assert path.parent == temp_log_dir
    assert path.name == "executions.jsonl"


def test_get_log_path_custom_file(tmp_path, monkeypatch):
    """Test custom log file path."""
    custom_path = tmp_path / "custom" / "logs.jsonl"
    monkeypatch.setenv("AGENTIC_EXECUTIONS_FILE", str(custom_path))

    path = get_log_path()

    assert path == custom_path


def test_log_entry_writes_jsonl(temp_log_dir):
    """Test that log_entry writes JSONL."""
    log = ExecutionLog(
        id="test-123",
        execution_id="exec-123",
        tool="test_tool",
        status=ExecutionStatus.STARTED,
        timestamp=datetime.now(timezone.utc),
        inputs={"x": 1, "y": 2},
    )

    log_entry(log)

    path = get_log_path()
    assert path.exists()

    with path.open() as f:
        line = f.readline()

    data = json.loads(line)
    assert data["id"] == "test-123"
    assert data["execution_id"] == "exec-123"
    assert data["tool"] == "test_tool"
    assert data["status"] == "started"
    assert data["inputs"] == {"x": 1, "y": 2}


def test_log_entry_appends(temp_log_dir):
    """Test that multiple logs are appended."""
    for i in range(3):
        log = ExecutionLog(
            id=f"test-{i}",
            execution_id=f"exec-{i}",
            tool="test_tool",
            status=ExecutionStatus.SUCCESS,
            timestamp=datetime.now(timezone.utc),
        )
        log_entry(log)

    path = get_log_path()
    with path.open() as f:
        lines = f.readlines()

    assert len(lines) == 3


def test_log_entry_with_context(temp_log_dir):
    """Test logging with session/user/thread context."""
    log = ExecutionLog(
        id="test-context",
        execution_id="exec-context",
        tool="test_tool",
        status=ExecutionStatus.SUCCESS,
        timestamp=datetime.now(timezone.utc),
        session_id="session-123",
        user_id="user-456",
    )

    log_entry(log)

    path = get_log_path()
    with path.open() as f:
        data = json.loads(f.readline())

    assert data["session_id"] == "session-123"
    assert data["user_id"] == "user-456"


def test_log_entry_with_trace_context(temp_log_dir):
    """Test logging with trace context fields."""
    log = ExecutionLog(
        id="test-trace",
        execution_id="exec-trace",
        tool="test_tool",
        status=ExecutionStatus.SUCCESS,
        timestamp=datetime.now(timezone.utc),
        trace_id="abcd1234abcd1234abcd1234abcd1234",
        span_id="1234567890abcdef",
    )

    log_entry(log)

    path = get_log_path()
    with path.open() as f:
        data = json.loads(f.readline())

    assert data["trace_id"] == "abcd1234abcd1234abcd1234abcd1234"
    assert data["span_id"] == "1234567890abcdef"


def test_log_entry_without_trace_context(temp_log_dir):
    """Test that logs work without trace context (fields are None)."""
    log = ExecutionLog(
        id="test-no-trace",
        execution_id="exec-no-trace",
        tool="test_tool",
        status=ExecutionStatus.SUCCESS,
        timestamp=datetime.now(timezone.utc),
    )

    log_entry(log)

    path = get_log_path()
    with path.open() as f:
        data = json.loads(f.readline())

    assert "trace_id" in data
    assert "span_id" in data
    assert data["trace_id"] is None
    assert data["span_id"] is None
