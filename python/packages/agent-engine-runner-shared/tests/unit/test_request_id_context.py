"""Legacy request identity stays independent of platform trace correlation."""

import agent_engine_runner_shared as sdk
from agent_engine_runner_shared.context import get_current_session_id, get_current_user_id


def test_legacy_positional_arguments_and_nested_context_restore() -> None:
    trace_id = "0123456789abcdef0123456789abcdef"
    tokens = sdk.set_execution_context(
        "e", None, "http://oe", "legacy-request", "user", "session", trace_id=trace_id
    )
    try:
        assert sdk.get_current_request_id() == "legacy-request"
        assert get_current_user_id() == "user"
        assert get_current_session_id() == "session"
        assert sdk.get_current_trace_id() == trace_id
        nested = sdk.set_execution_context("nested", None, "http://oe")
        try:
            nested_id = sdk.get_current_request_id()
            assert nested_id is not None
            assert nested_id.startswith("req-")
            assert sdk.get_current_trace_id() is None
        finally:
            sdk.clear_execution_context(nested)
        assert sdk.get_current_request_id() == "legacy-request"
        assert sdk.get_current_trace_id() == trace_id
    finally:
        sdk.clear_execution_context(tokens)
    assert sdk.current_request_id.get() is None
    assert sdk.get_current_trace_id() is None
