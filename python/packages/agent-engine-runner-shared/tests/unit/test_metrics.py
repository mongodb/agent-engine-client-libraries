"""Unit tests for log_tool_result in metrics.py."""

import logging
from unittest.mock import patch

import pytest

from agent_engine_runner_shared.metrics import log_tool_result


@pytest.mark.parametrize("status", ["success", "suspend", "cached"])
def test_log_tool_result_info_statuses(status: str) -> None:
    with (
        patch("agent_engine_runner_shared.metrics.logger") as mock_logger,
        patch("agent_engine_runner_shared.metrics.Metrics"),
    ):
        log_tool_result(
            execution_id="exec-1",
            step_number=1,
            tool_name="my_tool",
            status=status,
            duration_ms=10.0,
        )
        level = mock_logger.log.call_args[0][0]
        assert level == logging.INFO, f"expected INFO for status={status!r}, got {level}"


@pytest.mark.parametrize("status", ["error", "timeout", "unknown"])
def test_log_tool_result_error_statuses(status: str) -> None:
    with (
        patch("agent_engine_runner_shared.metrics.logger") as mock_logger,
        patch("agent_engine_runner_shared.metrics.Metrics"),
    ):
        log_tool_result(
            execution_id="exec-1",
            step_number=1,
            tool_name="my_tool",
            status=status,
            duration_ms=10.0,
        )
        level = mock_logger.log.call_args[0][0]
        assert level == logging.ERROR, f"expected ERROR for status={status!r}, got {level}"
