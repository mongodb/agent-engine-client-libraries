"""Validation of a request-supplied replica-specific OE owner callback URL.

Mirrors TypeScript's ``agent-engine-runner-shared/src/server/owner_url.ts``. Both runtimes
are parallel implementations of the same server architecture, so this file and
its TypeScript twin must be changed together.

Dark-ship owner-callback fallback: the OE may stamp a ``*_owner_url``
onto the execute / tool request naming the specific OE replica that owns the
execution. The runner delivers stream chunks, terminal callbacks and tool
results straight to that replica and falls back to the trusted service URL when
that owner-specific callback attempt is unusable.

That owner URL is request-supplied, which reopens the callback-hijack / SSRF
surface that :func:`agent_engine_runner_shared.server.oe_url.resolve_oe_url` exists to close
(read that module's docstring). Unlike the OE base URL — where the runner's own
deploy-time ``OE_URL`` is simply preferred — the owner URL has no trusted
counterpart to fall back to, so it is accepted only when it is provably the
headless-service address of one replica *behind the already-trusted service
URL*: identical scheme, identical port (explicit or scheme default), and a host
of exactly ``<one-dns-label>.<service-label>-headless.<rest>`` when the service
host is ``<service-label>.<rest>``. Anything else is logged and discarded —
never an error — and the caller keeps using the trusted service URL.
"""

from __future__ import annotations

import logging
import re
from typing import Optional
from urllib.parse import SplitResult, urlsplit

logger = logging.getLogger(__name__)

__all__ = ["resolve_owner_url"]

_DEFAULT_PORTS = {"http": 80, "https": 443}
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
# A single DNS label: alphanumerics plus interior hyphens. A dashed pod IP such
# as ``10-1-2-3`` — the label the OE actually stamps — qualifies.
_DNS_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


def _effective_port(parts: SplitResult) -> Optional[int]:
    """Port for ``parts`` — explicit if present, else the scheme default."""
    if parts.port is not None:
        return parts.port
    return _DEFAULT_PORTS.get(parts.scheme)


def _describe_url_for_log(parts: SplitResult) -> str:
    """Stable, non-secret URL summary safe for warning logs."""
    host = parts.hostname or ""
    if parts.port is None:
        return repr(f"{parts.scheme}://{host}")
    return repr(f"{parts.scheme}://{host}:{parts.port}")


def _canonical_origin(parts: SplitResult) -> str:
    """Canonical bare origin, omitting scheme-default ports."""
    host = parts.hostname or ""
    port = _effective_port(parts)
    default_port = _DEFAULT_PORTS.get(parts.scheme)
    if port is None or port == default_port:
        return f"{parts.scheme}://{host}"
    return f"{parts.scheme}://{host}:{port}"


def resolve_owner_url(owner_url: Optional[str], service_url: str) -> Optional[str]:
    """Return canonical owner origin when it is a valid replica of ``service_url``, else None.

    ``None`` means "no usable owner URL"; the caller sends to the trusted
    ``service_url`` instead. A malformed or forged owner URL is logged at
    WARNING and discarded rather than raised — a rejected value must degrade to
    the safe default, never fail the callback.
    """
    if not owner_url:
        return None
    if _ASCII_CONTROL.search(owner_url):
        logger.warning(
            "Discarding owner URL (contains control characters); using configured service URL"
        )
        return None

    try:
        owner = urlsplit(owner_url)
        service = urlsplit(service_url)
    except ValueError:
        logger.warning("Discarding unparseable owner URL; using configured service URL")
        return None

    service_summary = _describe_url_for_log(service)

    def reject(reason: str) -> None:
        logger.warning("Discarding owner URL (%s); using service URL %s", reason, service_summary)

    # The OE stamps a bare origin. Credentials, a path, a query, or a fragment
    # are never expected and would ride along when a callback path is appended,
    # so treat their presence as a forged value.
    if (
        owner.username
        or owner.password
        or owner.path not in ("", "/")
        or owner.query
        or owner.fragment
    ):
        reject("not a bare origin")
        return None

    if owner.scheme != service.scheme:
        reject("scheme mismatch")
        return None

    try:
        owner_port = _effective_port(owner)
        service_port = _effective_port(service)
    except ValueError:
        reject("invalid port")
        return None
    if owner_port is None or owner_port != service_port:
        reject("port mismatch")
        return None

    owner_host = owner.hostname
    service_host = service.hostname
    if not owner_host or not service_host:
        reject("missing host")
        return None

    service_label, service_dot, service_rest = service_host.partition(".")
    if not service_dot or not service_label or not service_rest:
        reject("service host is not <label>.<rest>")
        return None

    # partition() on the first dot guarantees exactly one leading label, so the
    # exact-match on the remainder below enforces the full shape.
    owner_label, owner_dot, owner_rest = owner_host.partition(".")
    if not owner_dot or not _DNS_LABEL.match(owner_label):
        reject("owner host is not <label>.<rest>")
        return None
    if owner_rest != f"{service_label}-headless.{service_rest}":
        reject("owner host is not a headless replica of the service host")
        return None

    return _canonical_origin(owner)
