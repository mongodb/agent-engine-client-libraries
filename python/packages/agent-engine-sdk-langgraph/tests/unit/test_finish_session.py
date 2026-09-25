"""Tests for App.finish_session()."""

from agent_engine_sdk_langgraph import App, SessionFinishStatus
from agent_engine_runner_shared import clear_execution_context, set_execution_context


def test_finish_session_reports_requested_then_already_requested():
    """Test finish_session returns REQUESTED first, then ALREADY_REQUESTED."""
    app = App(app_name="t", app_version="1.0.0")
    # A non-None wrapper simulates an AER context (the only context where
    # finish_session is available) — server/aer.py always builds a real
    # SecureToolWrapper before calling set_execution_context.
    tokens = set_execution_context(
        execution_id="exec-1",
        wrapper=object(),
        oe_url="http://oe",
        session_id="sess-1",
    )
    try:
        result1 = app.finish_session()
        result2 = app.finish_session()
        assert result1 is SessionFinishStatus.REQUESTED
        assert result2 is SessionFinishStatus.ALREADY_REQUESTED
    finally:
        clear_execution_context(tokens)


def test_finish_session_outside_a_run_is_unavailable():
    """Test finish_session returns UNAVAILABLE outside of a session context."""
    app = App(app_name="t", app_version="1.0.0")
    result = app.finish_session()
    assert result is SessionFinishStatus.UNAVAILABLE
