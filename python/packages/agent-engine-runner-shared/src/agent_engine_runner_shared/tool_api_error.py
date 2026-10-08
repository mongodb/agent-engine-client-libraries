"""Classify external API call failures into structured, safe metadata."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from typing import Any

import httpx
from pydantic import BaseModel, Field

from agent_engine_runner_shared.context import get_current_authorization
from agent_engine_runner_shared.error_reporting import _redact_text
from agent_engine_runner_shared.utils import tenant_env_vars

_MAX_ERROR_CODE_LEN = 128
_MAX_REASON_LEN = 256
_MAX_ENVELOPE_BODY = 4096
# Rudimentary URL drop: a scheme-bearing URL or a protocol-relative reference
# with a non-whitespace authority. Encoded or nested spellings are accepted
# risk, not chased.
_PROTOCOL_RELATIVE_URL = re.compile(r"(?<![A-Za-z0-9])//(?=\S)")


def request_credential_values() -> list[str]:
    """Request-local credential values applied for this call.

    Combines the tenant environment secrets with the delegated authorization
    token installed for this execution; both are values a provider can echo.

    Reads pod-level tenant env; the Tool Pod runs with secret restriction
    disabled (merge, no restore), so these values persist for the request. If
    per-request apply/restore is ever re-enabled, capture the values inside the
    credential window instead of reading them at classification time.
    """
    values = [value for value in tenant_env_vars().values() if value]
    authorization = get_current_authorization()
    token = getattr(authorization, "token", None) if authorization is not None else None
    if token and token not in values:
        values.append(token)
    return values


# Name segments marking a tenant env var's value as credential-bearing. Env
# names are conventionally SCREAMING_SNAKE with the credential kind last
# (OPENAI_API_KEY, GITHUB_TOKEN, DB_PASSWORD), so the name is what selects
# credential values. Selecting on the value's shape instead -- its length, or
# whether it looks alphanumeric -- cannot separate a 7-character token from the
# working directory or the string "1", and redacting those shreds the message
# the redaction exists to make readable.
_CREDENTIAL_NAME_MARKERS = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "CREDS",
    "AUTH",
)


def request_named_credential_values() -> list[str]:
    """Credential values a name marks as secrets, at any length.

    A narrower view of :func:`request_credential_values`: ordinary tenant env
    (``PATH``, ``HOSTNAME``, ``SHLVL``) is excluded, while a value under a
    credential-named variable is treated as a credential however short it is.
    Use this where the value is inserted into text a human reads, so that
    redaction cannot corrupt the message; use
    :func:`request_credential_values` where the sink tolerates a blunt
    replacement. The delegated authorization token counts as credential-named.
    """
    from agent_engine_runner_shared.context import get_current_authorization
    from agent_engine_runner_shared.utils import tenant_env_vars

    values = [
        value
        for name, value in tenant_env_vars().items()
        if value and any(marker in name.upper() for marker in _CREDENTIAL_NAME_MARKERS)
    ]
    authorization = get_current_authorization()
    token = getattr(authorization, "token", None) if authorization is not None else None
    if token and token not in values:
        values.append(token)
    return values


_STATUS_CLASSIFICATIONS: dict[int, tuple[str, bool]] = {
    401: ("AUTH_FAILED", False),
    403: ("AUTH_FAILED", False),
    429: ("RATE_LIMITED", True),
    503: ("PROVIDER_UNAVAILABLE", True),
}


class ToolAPIError(BaseModel):
    provider_type: str | None = Field(default=None, description="Provider identity when known")
    classification: str = Field(
        description=(
            "AUTH_FAILED, RATE_LIMITED, PROVIDER_UNAVAILABLE, TIMEOUT, CONNECTION_ERROR, or UNKNOWN"
        )
    )
    http_status: int | None = Field(
        default=None, description="HTTP status when the failure was an HTTP response"
    )
    retryable: bool = Field(
        default=False,
        description=(
            "Provider-condition guidance only. Must not trigger replay of an "
            "already-dispatched tool call."
        ),
    )
    error_code: str | None = Field(default=None, description="Bounded redacted provider error code")
    reason: str | None = Field(
        default=None, description="Bounded redacted provider reason; URLs dropped"
    )


def classify_tool_api_error(
    exc: BaseException,
    provider_type: str | None,
    *,
    credentials: Sequence[str] = (),
) -> tuple[ToolAPIError, str] | None:
    """Return structured metadata and a safe message, or None.

    ``credentials`` are the request-local secret values applied for this call;
    exact occurrences in recognized provider text are replaced with a marker.
    """
    if isinstance(exc, httpx.TimeoutException):
        return _result(provider_type, "TIMEOUT", None, True), _msg(provider_type, "TIMEOUT")
    if isinstance(exc, httpx.ConnectError):
        return (
            _result(provider_type, "CONNECTION_ERROR", None, True),
            _msg(provider_type, "CONNECTION_ERROR"),
        )
    if isinstance(exc, httpx.HTTPStatusError):
        return _classify_http(exc.response.status_code, exc.response, provider_type, credentials)

    try:
        import requests
    except ImportError:
        return None

    if isinstance(exc, requests.exceptions.Timeout):
        return _result(provider_type, "TIMEOUT", None, True), _msg(provider_type, "TIMEOUT")
    if isinstance(exc, requests.exceptions.ConnectionError):
        return (
            _result(provider_type, "CONNECTION_ERROR", None, True),
            _msg(provider_type, "CONNECTION_ERROR"),
        )
    if isinstance(exc, requests.exceptions.HTTPError):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        return _classify_http(status, response, provider_type, credentials)

    return None


def _result(
    provider_type: str | None,
    classification: str,
    http_status: int | None,
    retryable: bool,
    error_code: str | None = None,
    reason: str | None = None,
) -> ToolAPIError:
    return ToolAPIError(
        provider_type=provider_type,
        classification=classification,
        http_status=http_status,
        retryable=retryable,
        error_code=error_code,
        reason=reason,
    )


def _msg(provider_type: str | None, classification: str) -> str:
    return f"{provider_type or 'External'} API call failed: {classification}"


def _classify_http(
    status: int | None,
    response: Any,
    provider_type: str | None,
    credentials: Sequence[str] = (),
) -> tuple[ToolAPIError, str]:
    classification, retryable = _STATUS_CLASSIFICATIONS.get(status or 0, ("UNKNOWN", False))
    error_code, reason = _extract_envelope(response, credentials)
    error = _result(provider_type, classification, status, retryable, error_code, reason)
    detail = f"HTTP {status} {classification}" if status is not None else classification
    message = _msg(provider_type, detail)
    if reason:
        # The provider's explanation is the fastest path to the cause; it
        # reaches the user and the agent, and is bounded/redacted above.
        message = f"{message} — {reason}"
    return error, message


def _extract_envelope(
    response: Any, credentials: Sequence[str] = ()
) -> tuple[str | None, str | None]:
    if response is None:
        return None, None
    raw = getattr(response, "_content", None)
    if not isinstance(raw, (bytes, bytearray)):
        return None, None
    if len(raw) > _MAX_ENVELOPE_BODY:
        return None, None
    try:
        body = json.loads(bytes(raw))
    except Exception:
        return None, None
    if not isinstance(body, dict):
        return None, None
    return _envelope_error_code(body, credentials), _envelope_reason(body, credentials)


def _envelope_error_code(body: dict[str, Any], credentials: Sequence[str] = ()) -> str | None:
    """The provider's error code from a recognized envelope."""
    raw_code = body.get("errorCode")
    if isinstance(raw_code, str):
        return _safe_envelope_text(raw_code, _MAX_ERROR_CODE_LEN, credentials)
    nested = body.get("error")
    if isinstance(nested, dict) and isinstance(nested.get("code"), str):
        return _safe_envelope_text(nested["code"], _MAX_ERROR_CODE_LEN, credentials)
    return None


def _envelope_reason(body: dict[str, Any], credentials: Sequence[str] = ()) -> str | None:
    """The provider's own explanation from a recognized error envelope.

    Shapes follow the common provider conventions: Atlas ``reason``, Jira
    ``errorMessages``/``errors``, a top-level ``message``, and a nested
    ``error.message`` (OpenAI/Stripe/Anthropic style).
    """
    candidates: list[str] = []
    raw_reason = body.get("reason")
    if isinstance(raw_reason, str):
        candidates.append(raw_reason)
    messages = body.get("errorMessages")
    if isinstance(messages, list):
        candidates.extend(message for message in messages if isinstance(message, str))
    errors = body.get("errors")
    if isinstance(errors, dict):
        candidates.extend(value for value in errors.values() if isinstance(value, str))
    raw_message = body.get("message")
    if isinstance(raw_message, str):
        candidates.append(raw_message)
    nested = body.get("error")
    if isinstance(nested, dict) and isinstance(nested.get("message"), str):
        candidates.append(nested["message"])
    for candidate in candidates:
        safe = _safe_envelope_text(candidate, _MAX_REASON_LEN, credentials)
        if safe:
            return safe
    return None


def _safe_envelope_text(text: str, max_len: int, credentials: Sequence[str] = ()) -> str | None:
    """Rudimentary hygiene for one provider-authored string.

    Best-effort, not a guarantee: control characters are dropped, URL-shaped
    text is rejected, exact credential values are redacted, and the result is
    bounded. Other encodings and provider-specific content are accepted risk.
    """
    printable = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    if "://" in printable or _PROTOCOL_RELATIVE_URL.search(printable):
        return None
    for value in sorted((value for value in credentials if value), key=len, reverse=True):
        printable = printable.replace(value, "<redacted>")
    redacted = _redact_text(printable)[:max_len]
    return redacted or None
