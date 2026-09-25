"""
CORS origin allowlist resolution for Runner SDK servers.

Mirrors TypeScript's ``agent-engine-runner-shared/src/server/cors.ts``. Both runtimes
are parallel implementations of the same server architecture, so this file
and its TypeScript twin must be changed together.

The default is deny-all rather than ``*``. Runner servers register
credential-bearing routes (``/execute``, ``/invoke_llm``, AER session reads),
so a wildcard default meant any web page a developer visited could drive tool
execution and read stored conversation history once ``agentengine dev`` published
the ports locally.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

__all__ = ["CorsPolicy", "resolve_cors_policy"]


@dataclass(frozen=True)
class CorsPolicy:
    """Resolved CORS settings for ``CORSMiddleware``.

    ``allow_origins`` is empty for the deny-all default.
    ``allow_credentials`` is never True alongside a wildcard origin — that
    combination lets any site issue credentialed requests and read the
    responses.
    """

    allow_origins: List[str]
    allow_credentials: bool


_DEFAULT_PORTS = {"http": 80, "https": 443}

#: A DNS name, an IPv4 literal, or an IPv6 literal (unbracketed by ``urlsplit``).
_HOST_RE = re.compile(r"^(?:[a-z0-9.-]+|[0-9a-f:.]+)$")


def _normalize_origin(origin: str) -> Optional[str]:
    """Normalize ``scheme://host[:port]`` into the form browsers put in the
    ``Origin`` header — lowercased scheme and host, default port dropped — so
    an operator-written ``HTTPS://A.com:443`` still matches ``https://a.com``.

    Returns ``None`` for anything an ``Origin`` header never carries (a path,
    userinfo, a query, a non-http(s) scheme, or wildcards inside the host), so
    a malformed entry is reported at startup rather than silently never
    matching. Kept behaviourally identical to the TypeScript twin's
    ``normalizeOrigin``.
    """
    # The literal ``null`` origin sandboxed iframes and ``file://`` documents send.
    if origin == "null":
        return "null"
    parts = urlsplit(origin)
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS:
        return None
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        return None
    if parts.username or parts.password:
        return None
    try:
        host, port = parts.hostname, parts.port
    except ValueError:
        # A non-numeric or out-of-range port.
        return None
    if not host or not _HOST_RE.match(host):
        return None
    if ":" in host:
        host = f"[{host}]"
    if port is None or port == _DEFAULT_PORTS[scheme]:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def resolve_cors_policy(raw: Optional[str]) -> CorsPolicy:
    """Resolve the CORS policy from a raw ``CORS_ALLOWED_ORIGINS`` value.

    Entries are trimmed before validation: an operator who writes
    ``a.com, b.com`` reasonably expects both to be honoured, and comparing
    the untrimmed ``" b.com"`` meant a configured origin silently never
    matched — an operator could believe they had restricted origins when
    they had not.
    """
    entries = [entry.strip() for entry in (raw or "").split(",")]
    entries = [entry for entry in entries if entry]

    if not entries:
        # Distinguished from the misconfiguration warnings below: an unset
        # value used to mean "allow every origin", so a deployment that never
        # set it is exactly the one whose cross-origin access this removes.
        logger.warning(
            "CORS_ALLOWED_ORIGINS is unset — denying all cross-origin requests. "
            "Set it to a comma-separated list of origins to allow any."
        )
        return CorsPolicy(allow_origins=[], allow_credentials=False)

    if "*" in entries:
        # Explicit opt-in only, and never with credentials. Kept as an
        # escape hatch for local experimentation; production origins must
        # be listed.
        logger.warning(
            "CORS_ALLOWED_ORIGINS contains '*' — allowing all origins WITHOUT "
            "credentials. List explicit origins to enable credentialed requests."
        )
        return CorsPolicy(allow_origins=["*"], allow_credentials=False)

    valid: List[str] = []
    for entry in entries:
        normalized = _normalize_origin(entry)
        if normalized is not None:
            if normalized not in valid:
                valid.append(normalized)
        else:
            logger.warning(
                "Ignoring malformed CORS origin %r — expected scheme://host[:port] "
                "with no trailing path.",
                entry,
            )

    if not valid:
        return CorsPolicy(allow_origins=[], allow_credentials=False)
    return CorsPolicy(allow_origins=valid, allow_credentials=True)
