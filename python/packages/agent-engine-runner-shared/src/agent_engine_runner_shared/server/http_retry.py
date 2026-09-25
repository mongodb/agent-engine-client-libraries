"""Bounded-retry JSON POST shared by the AER chunk stream and function mode.

Both the AER's streaming-chunk delivery and function mode need to POST a JSON
body to the Orchestration Engine and tolerate transient transport failures.
This module holds the single retry policy both paths use so the behaviour
stays identical.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

# Default retry budget. AER overrides these via its own module constants; the
# values live here so a caller that just wants "the platform default" can omit
# them.
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_DELAY_S = 0.1

# Ceiling on an honored ``Retry-After`` wait: a cooperative 503 hint is worth a
# short pause, but the runner must not park a callback for an OE-chosen duration.
RETRY_AFTER_MAX_WAIT_S = 10.0

# Preserve the pre-terminal-redelivery retry surface for callers that share
# this helper. Established-connection read/write/close failures are only safe
# to retry for idempotent terminal callbacks because delivery may have occurred.
_DEFAULT_RETRYABLE_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
)


async def _post_without_reading_response_body(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
) -> httpx.Response:
    """POST without consuming the body, preserving the received status on close errors."""
    response: httpx.Response | None = None
    try:
        async with client.stream("POST", url, json=payload, follow_redirects=False) as response:
            pass
    except Exception:
        if response is None:
            raise
        # Headers already arrived, so delivery status is authoritative. A
        # teardown failure must not cause a duplicate callback attempt.
    return response


def parse_retry_after_seconds(header_value: Optional[str]) -> Optional[float]:
    """Parse a delta-seconds ``Retry-After`` header; ignore the HTTP-date form.

    Returns the non-negative integer seconds as a float, or ``None`` when the
    header is absent or not in delta-seconds form (an HTTP-date, or garbage).
    Callers fall back to their normal backoff when this returns ``None``.
    """
    if header_value is None:
        return None
    value = header_value.strip()
    if not value.isascii() or not value.isdigit():
        return None
    return float(value)


async def _owner_delivered(
    client: httpx.AsyncClient,
    owner_url: str,
    service_url: str,
    payload: Dict[str, Any],
) -> bool:
    """Best-effort single POST to the replica-specific owner URL.

    Async twin of :func:`agent_engine_runner_shared.owner_callback.owner_delivered` — the
    two implement the same owner-callback policy and must stay in sync.

    Returns True only on a 2xx response (delivered; the caller is done). Any
    failure — a transport error OR any non-2xx response — marks the replica
    unusable, is logged, and returns False so the caller runs the trusted
    service loop. The owner is never retried and never raises.
    """
    try:
        response = await _post_without_reading_response_body(client, owner_url, payload)
        if response.is_success:
            return True
        failure = f"HTTP {response.status_code}"
    except Exception as exc:  # noqa: BLE001 - owner is best-effort; any failure falls back
        failure = str(exc) or type(exc).__name__
    logger.warning(
        "Owner callback URL %s unusable (%s); falling back to service URL %s",
        owner_url,
        failure,
        service_url,
    )
    return False


def _is_default_retryable_http_error(exc: BaseException) -> bool:
    """Return whether a shared-helper failure matches its historical policy."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, _DEFAULT_RETRYABLE_TRANSPORT_ERRORS)


def is_retryable_callback_error(exc: BaseException) -> bool:
    """Return whether an idempotent callback failure should remain pending."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(
        exc,
        (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError),
    )


async def post_json_with_retries(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY_S,
    owner_url: Optional[str] = None,
    on_owner_failure: Optional[Callable[[], None]] = None,
    is_retryable: Optional[Callable[[BaseException], bool]] = None,
    should_attempt: Optional[Callable[[], bool]] = None,
) -> None:
    """POST ``payload`` to ``url`` with bounded exponential backoff.

    By default, retries 5xx responses and the historical narrow transport
    exception set:

    - ``httpx.ConnectError``
    - ``httpx.TimeoutException`` (connect, read, write, or pool timeout)
    - ``httpx.RemoteProtocolError`` (server closed the connection mid-response)
    - 5xx ``httpx.HTTPStatusError`` responses

    Callers with idempotent delivery semantics may pass a broader
    ``is_retryable`` classifier. ``None`` selects the default policy.

    When ``should_attempt`` is provided, it is checked immediately before the
    owner attempt and each service attempt. Returning ``False`` ends delivery
    without issuing another request.

    Does not retry on 4xx, which reflect a caller bug or stale endpoint rather
    than a transient network condition. Raises the final exception on exhaustion.

    ``owner_url`` — when given, a single owner pre-attempt runs *before* the
    service loop and does not consume the service retry budget (so the trusted
    ``url`` always gets its full ``max_attempts``; total tries are
    ``max_attempts + 1``). The owner is best-effort: any failure — a transport
    error OR any non-2xx response (4xx or 5xx) — marks the replica unusable, is
    logged, and falls through to the service loop. The owner is never retried
    and an owner response never raises. ``owner_url`` must already carry the
    request path; the caller is responsible for validating it against ``url``.
    When ``on_owner_failure`` is provided, it runs exactly once after such a
    failed owner pre-attempt and before the trusted service loop begins.

    When the trusted ``url`` returns 503 with a delta-seconds ``Retry-After``,
    that attempt waits ``min(hint, RETRY_AFTER_MAX_WAIT_S)`` instead of the
    exponential backoff.
    """
    if should_attempt is not None and not should_attempt():
        return

    if owner_url is not None:
        if await _owner_delivered(client, owner_url, url, payload):
            return
        if on_owner_failure is not None:
            on_owner_failure()

    last_exc: Optional[BaseException] = None
    retryable_error = is_retryable or _is_default_retryable_http_error
    for attempt in range(max_attempts):
        if should_attempt is not None and not should_attempt():
            return
        retry_after_s: Optional[float] = None
        try:
            response = await _post_without_reading_response_body(client, url, payload)
            response.raise_for_status()
            return
        except httpx.HTTPError as e:
            if not retryable_error(e):
                raise
            last_exc = e
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 503:
                retry_after_s = parse_retry_after_seconds(e.response.headers.get("Retry-After"))
        if attempt < max_attempts - 1:
            if retry_after_s is not None:
                await asyncio.sleep(min(retry_after_s, RETRY_AFTER_MAX_WAIT_S))
            else:
                await asyncio.sleep(base_delay * (2**attempt))
    if last_exc is None:
        raise RuntimeError(
            f"post_json_with_retries exhausted {max_attempts} attempts but recorded no exception"
        )
    raise last_exc
