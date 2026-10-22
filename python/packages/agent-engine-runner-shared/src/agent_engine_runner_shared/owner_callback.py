"""Owner-callback pre-attempt shared by every sync owner-preferred OE POST.

Policy (mirrored by ``owner_callback.ts`` on the TypeScript side and by the
async ``_owner_delivered`` in ``server/http_retry.py`` — keep all three in
sync): the replica-specific owner URL gets exactly one best-effort POST before
the trusted service URL. Any owner failure — a transport error, a redirect
(never followed: the owner URL is only validated as a headless replica of the
service origin, so a 3xx could forward the body elsewhere; httpx does not
follow redirects unless asked), or any other non-2xx — falls back to the
service path. The owner is never retried and this helper never raises.
Response bodies are never parsed, inspected, or logged (externally
influenced); they are discarded with a bounded drain solely so the keep-alive
connection can be returned to the pool.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import httpx

__all__ = [
    "MAX_DISCARD_BYTES",
    "MAX_DISCARD_SECONDS",
    "discard_budget_seconds",
    "discard_response_body",
    "owner_delivered",
    "post_without_reading_response_body",
]


# Upper bound on bytes discarded per response. OE acks on these paths are a
# few bytes of JSON; anything larger is abandoned and that one connection is
# closed instead of pooled, so an oversized owner body costs nothing further.
MAX_DISCARD_BYTES = 64 * 1024

# Upper bound on wall-clock time spent discarding a body. The byte cap alone does
# not bound time: a peer that trickles one byte per read keeps every individual
# read under httpx's per-read timeout, so the drain could run for a very long
# time before the byte cap is reached. The drain exists only to return a
# keep-alive connection to the pool, so once the budget elapses that one
# connection is closed instead of pooled.
MAX_DISCARD_SECONDS = 5.0


def discard_budget_seconds(timeout: Any = httpx.USE_CLIENT_DEFAULT) -> float:
    """Wall-clock budget for draining, clamped to the caller's per-request timeout.

    The byte cap bounds volume; this bounds time. The drain exists only to return
    a keep-alive connection to the pool, so it must not outlast the caller's own
    timeout intent: a best-effort callback configured for a couple of seconds must
    not block for the full ``MAX_DISCARD_SECONDS`` just because a peer trickled
    bytes under httpx's per-read timeout. An unknown timeout form (the
    ``httpx.USE_CLIENT_DEFAULT`` sentinel) keeps the full budget.
    """
    if isinstance(timeout, bool):  # bool is an int subclass; never a duration
        seconds: Optional[float] = None
    elif isinstance(timeout, (int, float)):
        seconds = float(timeout)
    elif isinstance(timeout, httpx.Timeout):
        seconds = timeout.read
    else:
        seconds = None
    if seconds is None:
        return MAX_DISCARD_SECONDS
    return max(0.0, min(MAX_DISCARD_SECONDS, seconds))


def discard_response_body(
    response: httpx.Response, *, timeout: Any = httpx.USE_CLIENT_DEFAULT
) -> None:
    """Drain up to MAX_DISCARD_BYTES within the time budget so the connection can be reused.

    httpcore only returns an HTTP/1.1 connection to the pool once the body has
    been read to the end; an unread body closes the socket. The bytes are never
    decoded, stored, or logged: OE/owner bodies are externally influenced and
    stay opaque to the runner. Both caps are needed — the byte cap bounds volume,
    the time cap bounds a slow-drip peer — and either one abandons the drain.
    The time cap is additionally clamped to the caller's ``timeout`` (see
    :func:`discard_budget_seconds`).
    """
    deadline = time.monotonic() + discard_budget_seconds(timeout)
    remaining = MAX_DISCARD_BYTES
    for chunk in response.iter_raw():
        remaining -= len(chunk)
        if remaining < 0 or time.monotonic() >= deadline:
            break


def post_without_reading_response_body(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    *,
    timeout: Any = httpx.USE_CLIENT_DEFAULT,
) -> httpx.Response:
    """POST without exposing the body, preserving the received status on close errors.

    ``timeout`` is applied per request so callers can share one pooled client
    across callback profiles. The sentinel default keeps the client's timeout;
    passing ``None`` would disable timeouts entirely.
    """
    response: httpx.Response | None = None
    try:
        with client.stream(
            "POST", url, json=payload, follow_redirects=False, timeout=timeout
        ) as response:
            discard_response_body(response, timeout=timeout)
    except Exception:
        if response is None:
            raise
        # Headers already arrived, so delivery status is authoritative. A
        # teardown failure must not cause a duplicate callback attempt.
    return response


def owner_delivered(
    client: httpx.Client,
    owner_url: str,
    payload: dict,
    log: Callable[[str], None],
    *,
    timeout: Any = httpx.USE_CLIENT_DEFAULT,
) -> bool:
    """Best-effort single POST to the replica-specific owner URL.

    Returns True only on a 2xx response (delivered; the caller is done).
    Any failure sends the caller to the trusted service path.
    """
    try:
        response = post_without_reading_response_body(client, owner_url, payload, timeout=timeout)
        if response.is_success:
            return True
        failure = f"HTTP {response.status_code}"
    except Exception as exc:  # noqa: BLE001 - owner is best-effort; any failure falls back
        failure = str(exc) or type(exc).__name__
    log(f"owner URL {owner_url} unusable ({failure}); falling back to service")
    return False
