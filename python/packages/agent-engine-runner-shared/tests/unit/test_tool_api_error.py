import json
import sys
import types

import httpx
import pytest

from agent_engine_runner_shared.tool_api_error import classify_tool_api_error


def _fake_requests(monkeypatch: pytest.MonkeyPatch):
    timeout = type("Timeout", (Exception,), {})
    conn = type("ConnectionError", (Exception,), {})
    exceptions = types.SimpleNamespace(Timeout=timeout, ConnectionError=conn)
    fake = types.ModuleType("requests")
    fake.exceptions = exceptions
    monkeypatch.setitem(sys.modules, "requests", fake)
    return exceptions


def _status_error(
    status_code: int,
    json_body: dict | None = None,
    text: str | None = None,
) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://api.example.com/resource")
    if json_body is not None:
        content = json.dumps(json_body).encode()
        headers = {"content-type": "application/json"}
    else:
        content = (text or f"HTTP {status_code}").encode()
        headers = {}
    response = httpx.Response(
        status_code,
        content=content,
        headers=headers,
        request=request,
    )
    return httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=request,
        response=response,
    )


@pytest.mark.parametrize(
    ("exc", "classification", "retryable", "status"),
    [
        (_status_error(401), "AUTH_FAILED", False, 401),
        (_status_error(403), "AUTH_FAILED", False, 403),
        (_status_error(429), "RATE_LIMITED", True, 429),
        (_status_error(503), "PROVIDER_UNAVAILABLE", True, 503),
        (httpx.TimeoutException("secret timeout"), "TIMEOUT", True, None),
        (httpx.ConnectError("secret connection"), "CONNECTION_ERROR", True, None),
        (_status_error(400), "UNKNOWN", False, 400),
    ],
)
def test_classifies_external_api_failures(exc, classification, retryable, status):
    result = classify_tool_api_error(exc, "atlas")
    assert result is not None
    error, message = result
    assert error.classification == classification
    assert error.retryable is retryable
    assert error.http_status == status
    assert "secret" not in message


def test_atlas_envelope_extracts_redacted_error_code_and_reason():
    exc = _status_error(
        403,
        json_body={
            "errorCode": "AtlasError",
            "reason": "Forbidden access",
            "detail": "Bearer sk-secret-token-123",
            "parameters": {"url": "https://user:password@evil.com"},
        },
    )
    result = classify_tool_api_error(exc, "atlas")
    assert result is not None
    error, message = result

    assert error.error_code == "AtlasError"
    assert error.reason == "Forbidden access"
    assert error.classification == "AUTH_FAILED"

    json_str = error.model_dump_json()
    assert "sk-secret-token-123" not in json_str
    assert "password" not in json_str
    assert "Bearer" not in json_str
    assert "detail" not in json_str
    assert "parameters" not in json_str

    assert "sk-secret-token-123" not in message
    assert "password" not in message


def test_atlas_envelope_redacts_secrets_in_error_code_and_reason():
    exc = _status_error(
        429,
        json_body={
            "errorCode": "token=sk-secret-key",
            "reason": "password=hunter2 blocked",
        },
    )
    result = classify_tool_api_error(exc, "atlas")
    assert result is not None
    error, _ = result

    assert "sk-secret-key" not in (error.error_code or "")
    assert "hunter2" not in (error.reason or "")
    assert "<redacted>" in (error.error_code or "")
    assert "<redacted>" in (error.reason or "")


def test_atlas_envelope_drops_urls_in_error_code_and_reason():
    exc = _status_error(
        403,
        json_body={
            "errorCode": "see https://evil.example/x",
            "reason": "docs at http://provider.example/help",
        },
    )
    result = classify_tool_api_error(exc, "atlas")
    assert result is not None
    error, _ = result
    assert error.error_code is None
    assert error.reason is None


def test_atlas_envelope_truncates_long_values():
    long_code = "E" * 200
    long_reason = "R" * 300
    exc = _status_error(
        403,
        json_body={"errorCode": long_code, "reason": long_reason},
    )
    result = classify_tool_api_error(exc, "atlas")
    assert result is not None
    error, _ = result
    assert len(error.error_code or "") <= 128
    assert len(error.reason or "") <= 256


def test_jira_envelope_surfaces_the_provider_explanation():
    exc = _status_error(
        400,
        json_body={
            "errorMessages": ["The value 'OpenJira' does not exist for the field 'project'."],
            "errors": {},
        },
    )
    result = classify_tool_api_error(exc, "jiradc")
    assert result is not None
    error, message = result

    assert error.classification == "UNKNOWN"
    assert error.reason == "The value 'OpenJira' does not exist for the field 'project'."
    assert message.endswith("— The value 'OpenJira' does not exist for the field 'project'.")


def test_jira_errors_map_used_when_messages_are_empty():
    exc = _status_error(
        400,
        json_body={"errorMessages": [], "errors": {"project": "Project 'X' is unknown"}},
    )
    result = classify_tool_api_error(exc, "jiradc")
    assert result is not None
    error, _ = result
    assert error.reason == "Project 'X' is unknown"


def test_generic_and_nested_envelopes_are_recognized():
    generic = classify_tool_api_error(
        _status_error(422, json_body={"message": "Validation failed"}), "svc"
    )
    assert generic is not None
    assert generic[0].reason == "Validation failed"

    nested = classify_tool_api_error(
        _status_error(
            400, json_body={"error": {"message": "Invalid model", "code": "invalid_request"}}
        ),
        "openai",
    )
    assert nested is not None
    error, message = nested
    assert error.reason == "Invalid model"
    assert error.error_code == "invalid_request"
    assert "Invalid model" in message


def test_unrecognized_envelope_keeps_the_plain_message():
    exc = _status_error(400, json_body={"detail": "not a recognized field"})
    result = classify_tool_api_error(exc, "svc")
    assert result is not None
    error, message = result
    assert error.reason is None
    assert message == "svc API call failed: HTTP 400 UNKNOWN"


def test_control_split_url_is_not_surfaced():
    exc = _status_error(400, json_body={"message": "https:/\u0000/evil.example"})
    result = classify_tool_api_error(exc, "svc")
    assert result is not None
    error, message = result
    assert error.reason is None
    assert message == "svc API call failed: HTTP 400 UNKNOWN"


def test_envelope_text_drops_control_characters():
    exc = _status_error(400, json_body={"message": "bad\u009bmessage\u0007"})
    result = classify_tool_api_error(exc, "svc")
    assert result is not None
    error, message = result
    assert error.reason == "badmessage"
    assert "\u009b" not in message
    assert "\u0007" not in message


def test_credentials_redact_an_unlabeled_provider_echo():
    exc = _status_error(400, json_body={"message": "Rejected credential sk-live-123"})
    result = classify_tool_api_error(exc, "svc", credentials=["sk-live-123"])
    assert result is not None
    error, message = result
    assert error.reason == "Rejected credential <redacted>"
    assert "sk-live-123" not in message


def test_request_credential_values_include_delegated_token(monkeypatch):
    from agent_engine_runner_shared.context import clear_execution_context, set_execution_context
    from agent_engine_runner_shared.models import ToolAuthorization
    from agent_engine_runner_shared.tool_api_error import request_credential_values

    monkeypatch.setenv("TENANT_SECRET", "tenant-value")
    tokens = set_execution_context(
        execution_id="exec-1",
        wrapper=None,
        oe_url="http://localhost:8080",
        authorization=ToolAuthorization(token="delegated-abc123", expires_at=1_735_689_600),
    )
    try:
        values = request_credential_values()
    finally:
        clear_execution_context(tokens)

    assert "tenant-value" in values
    assert "delegated-abc123" in values


def test_urls_are_not_surfaced():
    for body in (
        {"message": "see https://internal.example/private"},
        {"errorMessages": ["target //api.internal.example/private"]},
    ):
        exc = _status_error(400, json_body=body)
        result = classify_tool_api_error(exc, "svc")
        assert result is not None
        error, message = result
        assert error.reason is None
        assert message == "svc API call failed: HTTP 400 UNKNOWN"


def test_double_slash_prose_is_still_surfaced():
    exc = _status_error(400, json_body={"message": "retry // later"})
    result = classify_tool_api_error(exc, "svc")
    assert result is not None
    error, _ = result
    assert error.reason == "retry // later"


def test_oversized_buffered_body_does_not_set_error_code():
    exc = _status_error(
        403,
        json_body={"errorCode": "AtlasError", "reason": "Forbidden", "pad": "x" * 5000},
    )
    result = classify_tool_api_error(exc, "atlas")
    assert result is not None
    error, _ = result
    assert error.error_code is None
    assert error.reason is None


def test_complete_json_object_with_padding_does_not_set_error_code():
    request = httpx.Request("GET", "https://api.example.com/resource")
    content = json.dumps({"errorCode": "AtlasError", "reason": "Forbidden"}).encode() + b" " * 5000
    response = httpx.Response(
        403,
        content=content,
        headers={"content-type": "application/json"},
        request=request,
    )
    exc = httpx.HTTPStatusError("HTTP 403", request=request, response=response)
    result = classify_tool_api_error(exc, "atlas")
    assert result is not None
    error, _ = result
    assert error.error_code is None
    assert error.reason is None


def test_unbuffered_body_does_not_set_error_code():
    request = httpx.Request("GET", "https://api.example.com/resource")
    response = httpx.Response(403, request=request)
    exc = httpx.HTTPStatusError("HTTP 403", request=request, response=response)
    result = classify_tool_api_error(exc, "atlas")
    assert result is not None
    error, _ = result
    assert error.error_code is None
    assert error.reason is None


def test_non_object_json_body_does_not_set_error_code():
    exc = _status_error(403, text='["not", "an", "object"]')
    result = classify_tool_api_error(exc, "atlas")
    assert result is not None
    error, _ = result
    assert error.error_code is None
    assert error.reason is None


def test_non_string_error_code_ignored():
    exc = _status_error(
        403,
        json_body={"errorCode": 42, "reason": "ok"},
    )
    result = classify_tool_api_error(exc, "atlas")
    assert result is not None
    error, _ = result
    assert error.error_code is None
    assert error.reason == "ok"


def test_no_json_body_does_not_set_error_code():
    exc = _status_error(403, text="plain text body")
    result = classify_tool_api_error(exc, "atlas")
    assert result is not None
    error, _ = result
    assert error.error_code is None
    assert error.reason is None


def test_non_http_exception_returns_none():
    assert classify_tool_api_error(ValueError("boom"), "atlas") is None


def test_requests_timeout_classified(monkeypatch: pytest.MonkeyPatch):
    exceptions = _fake_requests(monkeypatch)
    result = classify_tool_api_error(exceptions.Timeout("secret requests timeout"), "atlas")
    assert result is not None
    error, message = result
    assert error.classification == "TIMEOUT"
    assert error.retryable is True
    assert "secret" not in message


def test_requests_connection_error_classified(monkeypatch: pytest.MonkeyPatch):
    exceptions = _fake_requests(monkeypatch)
    result = classify_tool_api_error(exceptions.ConnectionError("secret requests conn"), "atlas")
    assert result is not None
    error, message = result
    assert error.classification == "CONNECTION_ERROR"
    assert error.retryable is True
    assert "secret" not in message


def test_provider_type_propagated_to_tool_api_error():
    exc = httpx.TimeoutException("timed out")
    result = classify_tool_api_error(exc, "openai")
    assert result is not None
    error, _ = result
    assert error.provider_type == "openai"


def test_unknown_http_status_is_unknown_not_retryable():
    exc = _status_error(500)
    result = classify_tool_api_error(exc, "atlas")
    assert result is not None
    error, _ = result
    assert error.classification == "UNKNOWN"
    assert error.retryable is False
    assert error.http_status == 500
