"""Shared HTTP transport base for the in-package memory runtimes.

Owns the mechanics every runtime shares: the bounded retry loop, the typed
error mapping, the ``memories`` extraction, the httpx client construction, and
the context-manager lifecycle. The composing runtime supplies only its
per-operation request bodies and (via the ``headers`` argument) its auth
header. This module is private and exports nothing into the package root.
"""

from __future__ import annotations

import random
import time
from types import TracebackType
from typing import Any, cast

import httpx

from agent_engine_sdk_memory.errors import (
    MemoryAuthError,
    MemoryBadRequestError,
    MemoryConnectionError,
    MemoryNotProvisionedError,
    MemoryServerError,
)

_MAX_ATTEMPTS = 3  # initial + 2 retries
_BASE_DELAY = 0.2  # seconds
_RETRYABLE_STATUS = frozenset({502, 503, 504})

_NOT_PROVISIONED_CODES = {
    "AGENT_NOT_DEPLOYED",
    "NO_WORKSPACE_ENDPOINT",
    "PROJECT_RUNTIME_FAILED",
}


def raise_for_status(resp: httpx.Response) -> None:
    """Map a non-success response to a typed ``MemoryAPIError``."""
    if resp.is_success:
        return

    try:
        parsed = resp.json()
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        body = cast("dict[str, Any]", parsed)
        code = body.get("code") or None
        # "detail" is FastAPI's convention and is what the memory server
        # returns; without it a structured message surfaces as raw JSON. Only
        # strings qualify — FastAPI validation errors put a list there.
        detail = body.get("detail")
        message = (
            body.get("error")
            or body.get("message")
            or (detail if isinstance(detail, str) else None)
            or resp.text
        )
        response_text = None
    else:
        code = None
        message = resp.text
        response_text = resp.text

    status = resp.status_code
    if status in (401, 403):
        raise MemoryAuthError(
            message, status=status, code=code, response_text=response_text
        )
    if code in _NOT_PROVISIONED_CODES:
        raise MemoryNotProvisionedError(
            message, status=status, code=code, response_text=response_text
        )
    if 400 <= status < 500:
        raise MemoryBadRequestError(
            message, status=status, code=code, response_text=response_text
        )
    raise MemoryServerError(
        message, status=status, code=code, response_text=response_text
    )


class _HttpTransport:
    """Shared HTTP mechanics for the in-package memory runtimes.

    Callers build per-operation request bodies and pass their auth header
    (if any) via the ``headers`` argument; everything else — retries, error
    mapping, response extraction, and lifecycle — lives here.
    """

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers=headers or {},
            transport=transport,
        )

    def _sleep_backoff(self, attempt: int) -> None:
        time.sleep(_BASE_DELAY * (2**attempt) + random.uniform(0, _BASE_DELAY))

    def _request_with_retry(
        self, method: str, path: str, *, retry: bool = True, **kwargs: Any
    ) -> httpx.Response:
        """Issue a request, optionally retrying transport errors and 502/503/504.

        ``kwargs`` are forwarded unchanged to the httpx client on every attempt,
        so a write whose body already carries ``idempotency_key`` resends the
        same key on retry while read calls (which never set one) carry none on
        any attempt. ``retry=False`` makes a single attempt with no retries — for
        non-idempotent writes (CRUD creates) that carry no idempotency key, where
        a retry after the server committed could duplicate the record.
        """
        max_attempts = _MAX_ATTEMPTS if retry else 1
        for attempt in range(max_attempts):
            try:
                resp = self._client.request(method, path, **kwargs)
            except httpx.TransportError as exc:
                if attempt + 1 >= max_attempts:
                    raise MemoryConnectionError(str(exc), status=None) from exc
                self._sleep_backoff(attempt)
                continue
            if resp.status_code in _RETRYABLE_STATUS and attempt + 1 < max_attempts:
                self._sleep_backoff(attempt)
                continue
            return resp
        # Unreachable: the loop always returns or raises within max_attempts.
        raise MemoryConnectionError("retry loop exhausted", status=None)

    def _post(
        self, path: str, body: dict[str, Any], *, retry: bool = True
    ) -> httpx.Response:
        resp = self._request_with_retry("POST", path, retry=retry, json=body)
        raise_for_status(resp)
        return resp

    def _get(self, path: str, params: dict[str, Any]) -> httpx.Response:
        """GET with the same retry loop and typed-error mapping as ``_post``."""
        resp = self._request_with_retry("GET", path, params=params)
        raise_for_status(resp)
        return resp

    def _get_optional(self, path: str, params: dict[str, Any]) -> httpx.Response | None:
        """GET that returns ``None`` on 404 instead of raising.

        Used by CRUD look-ups where a missing entry is a normal outcome
        (get_semantic, get_taxonomic, get_procedural).
        """
        resp = self._request_with_retry("GET", path, params=params)
        if resp.status_code == 404:
            return None
        raise_for_status(resp)
        return resp

    def _extract_list(
        self, resp: httpx.Response, key: str, what: str
    ) -> list[dict[str, Any]]:
        """Extract the list under ``key`` from a 200 response.

        A non-JSON body, a non-object payload, or a non-list value is surfaced
        as a typed ``MemoryServerError`` rather than a raw
        ``AttributeError``/``TypeError``; an absent or null value is treated as
        no results. ``what`` names the response in error messages (e.g.
        ``"search"`` or ``"taxonomic"``).
        """
        try:
            payload = resp.json()
        except ValueError as exc:
            raise MemoryServerError(
                f"the memory server returned a non-JSON {what} response",
                status=resp.status_code,
                response_text=resp.text,
            ) from exc
        if not isinstance(payload, dict):
            raise MemoryServerError(
                f"unexpected {what} response shape from the memory server",
                status=resp.status_code,
                response_text=resp.text,
            )
        value = cast("dict[str, Any]", payload).get(key)
        if value is None:
            return []
        if not isinstance(value, list):
            raise MemoryServerError(
                f"unexpected {what} response shape from the memory server",
                status=resp.status_code,
                response_text=resp.text,
            )
        return cast("list[dict[str, Any]]", value)

    def _json_object(self, resp: httpx.Response, what: str) -> dict[str, Any]:
        """Parse a 2xx body as a JSON object, mapping bad shapes to typed errors.

        A non-JSON body or a non-object payload on an otherwise successful
        response is surfaced as ``MemoryServerError`` rather than a raw
        ``ValueError``/``ValidationError`` escaping to the caller, keeping the
        success path inside the SDK's typed-error contract. ``what`` names the
        operation in the error message (e.g. ``"create semantic"``).
        """
        try:
            payload = resp.json()
        except ValueError as exc:
            raise MemoryServerError(
                f"the memory server returned a non-JSON {what} response",
                status=resp.status_code,
                response_text=resp.text,
            ) from exc
        if not isinstance(payload, dict):
            raise MemoryServerError(
                f"unexpected {what} response shape from the memory server",
                status=resp.status_code,
                response_text=resp.text,
            )
        return cast("dict[str, Any]", payload)

    def _memories(self, resp: httpx.Response) -> list[dict[str, Any]]:
        """Extract the ``memories`` list from a 200 search/discover response."""
        return self._extract_list(resp, "memories", "search")

    def _entries(self, resp: httpx.Response) -> list[dict[str, Any]]:
        """Extract the ``entries`` list from a 200 taxonomic-list response."""
        return self._extract_list(resp, "entries", "taxonomic")

    # Package-internal delegation surface used by _HttpMemoryRuntime and
    # _EmptyTenancyCrudClient (composition). Aliases without the leading
    # underscore satisfy pyright's reportPrivateUsage rule when delegates
    # reach across class boundaries.
    post = _post
    get = _get
    get_optional = _get_optional
    json_object = _json_object
    memories = _memories
    entries = _entries

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> _HttpTransport:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["_HttpTransport"]
