"""Typed transport-error hierarchy for the api-key Gateway transport."""

import pytest
from agent_engine_sdk_memory.errors import (
    MemoryAPIError,
    MemoryAuthError,
    MemoryBadRequestError,
    MemoryClientError,
    MemoryConnectionError,
    MemoryIdentityError,
    MemoryNotProvisionedError,
    MemoryNotSupportedError,
    MemoryServerError,
)


def test_base_exposes_attributes():
    err = MemoryAPIError("forbidden", status=403, code="DENIED", response_text="raw")
    assert err.message == "forbidden"
    assert err.status == 403
    assert err.code == "DENIED"
    assert err.response_text == "raw"


def test_base_defaults_are_none():
    err = MemoryAPIError("boom")
    assert err.status is None
    assert err.code is None
    assert err.response_text is None


def test_str_includes_message_and_code():
    err = MemoryAPIError("forbidden", status=403, code="DENIED")
    text = str(err)
    assert "forbidden" in text
    assert "DENIED" in text


@pytest.mark.parametrize(
    "cls",
    [
        MemoryAuthError,
        MemoryNotProvisionedError,
        MemoryBadRequestError,
        MemoryServerError,
        MemoryConnectionError,
    ],
)
def test_subclasses_inherit_base(cls):
    assert issubclass(cls, MemoryAPIError)
    err = cls("msg", status=400, code="X")
    assert isinstance(err, MemoryAPIError)
    assert err.message == "msg"


def test_chaining_preserves_cause():
    original = ValueError("underlying")
    try:
        try:
            raise original
        except ValueError as exc:
            raise MemoryServerError("server blew up", status=500) from exc
    except MemoryServerError as caught:
        assert caught.__cause__ is original


def test_identity_error_unchanged():
    assert issubclass(MemoryIdentityError, ValueError)
    assert not issubclass(MemoryIdentityError, MemoryAPIError)


def test_not_supported_is_client_error_not_api_error():
    # Pre-HTTP capability check — no status code or response body.
    assert issubclass(MemoryNotSupportedError, MemoryClientError)
    assert not issubclass(MemoryNotSupportedError, MemoryAPIError)
    err = MemoryNotSupportedError("unsupported in this mode")
    assert str(err) == "unsupported in this mode"
