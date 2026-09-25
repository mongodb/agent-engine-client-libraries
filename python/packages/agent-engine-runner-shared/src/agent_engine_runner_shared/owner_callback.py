"""Owner-callback pre-attempt shared by every sync owner-preferred OE POST.

Policy (mirrored by ``owner_callback.ts`` on the TypeScript side and by the
async ``_owner_delivered`` in ``server/http_retry.py`` — keep all three in
sync): the replica-specific owner URL gets exactly one best-effort POST before
the trusted service URL. Any owner failure — a transport error, a redirect
(never followed: the owner URL is only validated as a headless replica of the
service origin, so a 3xx could forward the body elsewhere; httpx does not
follow redirects unless asked), or any other non-2xx — falls back to the
service path. The owner is never retried and this helper never raises.
Response bodies are never read or logged (externally influenced).
"""

from __future__ import annotations

from typing import Any, Callable

import httpx

__all__ = ["owner_delivered", "post_without_reading_response_body"]


def post_without_reading_response_body(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
) -> httpx.Response:
    """POST without consuming the body, preserving the received status on close errors."""
    response: httpx.Response | None = None
    try:
        with client.stream("POST", url, json=payload, follow_redirects=False) as response:
            pass
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
) -> bool:
    """Best-effort single POST to the replica-specific owner URL.

    Returns True only on a 2xx response (delivered; the caller is done).
    Any failure sends the caller to the trusted service path.
    """
    try:
        response = post_without_reading_response_body(client, owner_url, payload)
        if response.is_success:
            return True
        failure = f"HTTP {response.status_code}"
    except Exception as exc:  # noqa: BLE001 - owner is best-effort; any failure falls back
        failure = str(exc) or type(exc).__name__
    log(f"owner URL {owner_url} unusable ({failure}); falling back to service")
    return False
