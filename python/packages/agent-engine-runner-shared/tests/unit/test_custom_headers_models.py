"""Tests for custom_headers fields on ExecuteRequest and InvokeRequest models."""

from agent_engine_runner_shared.models import ExecuteRequest, InvokeRequest, ToolExecuteRequest

# Required fields for model construction
_EXECUTE_REQUIRED = {
    "execution_id": "exec-001",
    "message": "test message",
    "platform_api_url": "http://localhost:8000",
}
_INVOKE_REQUIRED = {
    "message": "test message",
}
_TOOL_EXECUTE_REQUIRED = {
    "execution_id": "exec-001",
    "tool_name": "get_custom_headers",
    "arguments": {},
    "step_number": 1,
}


# ---------------------------------------------------------------------------
# ExecuteRequest tests
# ---------------------------------------------------------------------------


def test_execute_request_accepts_custom_headers():
    """ExecuteRequest accepts a custom_headers dict and stores it."""
    headers = {"authorization": "Bearer secret", "x-tenant-id": "tenant-42"}
    req = ExecuteRequest(**_EXECUTE_REQUIRED, custom_headers=headers)

    assert req.custom_headers == headers


def test_execute_request_round_trips_custom_headers():
    """ExecuteRequest custom_headers survives a model_dump / model_validate round-trip."""
    headers = {"x-request-id": "req-999", "x-feature-flag": "dark-mode"}
    req = ExecuteRequest(**_EXECUTE_REQUIRED, custom_headers=headers)

    dumped = req.model_dump()
    restored = ExecuteRequest.model_validate(dumped)

    assert restored.custom_headers == headers


def test_execute_request_defaults_custom_headers_to_none():
    """ExecuteRequest defaults custom_headers to None when not provided."""
    req = ExecuteRequest(**_EXECUTE_REQUIRED)

    assert req.custom_headers is None


def test_execute_request_custom_headers_omitted_when_none_in_serialization():
    """ExecuteRequest with custom_headers=None omits the key from model_dump(exclude_none=True)."""
    req = ExecuteRequest(**_EXECUTE_REQUIRED)

    dumped = req.model_dump(exclude_none=True)

    assert "custom_headers" not in dumped


def test_execute_request_custom_headers_included_when_set_in_serialization():
    """ExecuteRequest with custom_headers set includes the key in model_dump output."""
    headers = {"x-tenant-id": "tenant-xyz"}
    req = ExecuteRequest(**_EXECUTE_REQUIRED, custom_headers=headers)

    dumped = req.model_dump()

    assert dumped["custom_headers"] == headers


def test_execute_request_accepts_empty_custom_headers_dict():
    """ExecuteRequest accepts an empty dict for custom_headers (distinct from None)."""
    req = ExecuteRequest(**_EXECUTE_REQUIRED, custom_headers={})

    assert req.custom_headers == {}


# ---------------------------------------------------------------------------
# ToolExecuteRequest tests
# ---------------------------------------------------------------------------


def test_tool_execute_request_round_trips_custom_headers():
    """Request-local headers survive the AER to OE protocol boundary."""
    headers = {"authorization": "Bearer request-token", "x-tenant-id": "tenant-42"}
    req = ToolExecuteRequest(**_TOOL_EXECUTE_REQUIRED, custom_headers=headers)

    restored = ToolExecuteRequest.model_validate(req.model_dump())

    assert restored.custom_headers == headers


def test_tool_execute_request_custom_headers_omitted_when_none():
    req = ToolExecuteRequest(**_TOOL_EXECUTE_REQUIRED)

    assert "custom_headers" not in req.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# InvokeRequest tests
# ---------------------------------------------------------------------------


def test_invoke_request_accepts_custom_headers():
    """InvokeRequest accepts a custom_headers dict and stores it."""
    headers = {"authorization": "Bearer invoke-token", "x-correlation-id": "corr-1"}
    req = InvokeRequest(**_INVOKE_REQUIRED, custom_headers=headers)

    assert req.custom_headers == headers


def test_invoke_request_round_trips_custom_headers():
    """InvokeRequest custom_headers survives a model_dump / model_validate round-trip."""
    headers = {"x-user-id": "user-7", "x-session-id": "sess-88"}
    req = InvokeRequest(**_INVOKE_REQUIRED, custom_headers=headers)

    dumped = req.model_dump()
    restored = InvokeRequest.model_validate(dumped)

    assert restored.custom_headers == headers


def test_invoke_request_defaults_custom_headers_to_none():
    """InvokeRequest defaults custom_headers to None when not provided."""
    req = InvokeRequest(**_INVOKE_REQUIRED)

    assert req.custom_headers is None


def test_invoke_request_custom_headers_omitted_when_none_in_serialization():
    """InvokeRequest with custom_headers=None omits the key from model_dump(exclude_none=True)."""
    req = InvokeRequest(**_INVOKE_REQUIRED)

    dumped = req.model_dump(exclude_none=True)

    assert "custom_headers" not in dumped


def test_invoke_request_custom_headers_included_when_set_in_serialization():
    """InvokeRequest with custom_headers set includes the key in model_dump output."""
    headers = {"x-tenant-id": "tenant-abc"}
    req = InvokeRequest(**_INVOKE_REQUIRED, custom_headers=headers)

    dumped = req.model_dump()

    assert dumped["custom_headers"] == headers


def test_invoke_request_accepts_empty_custom_headers_dict():
    """InvokeRequest accepts an empty dict for custom_headers (distinct from None)."""
    req = InvokeRequest(**_INVOKE_REQUIRED, custom_headers={})

    assert req.custom_headers == {}
