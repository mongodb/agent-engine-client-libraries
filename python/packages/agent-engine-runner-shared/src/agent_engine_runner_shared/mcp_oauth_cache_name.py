"""Stable MCP OAuth cache-name derivation."""

from __future__ import annotations

import hashlib
import re

_READABLE_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_UNSAFE_NAME_CHAR_RE = re.compile(r"[^A-Za-z0-9_-]+")


def mcp_oauth_cache_name(server_name: str, server_url: str) -> str:
    """Return a collision-free cache basename scoped to alias and endpoint.

    ``agent.yaml`` imposes no charset on aliases, so uniqueness comes from
    hashing the raw alias whenever the readable form would lose information,
    and from the endpoint hash that stops the same alias sharing credentials
    across different MCP servers.
    """

    return f"{_cache_base(server_name)}-{_sha256(server_url)}"


def _cache_base(name: str) -> str:
    if _READABLE_NAME_RE.fullmatch(name):
        return name
    safe = _UNSAFE_NAME_CHAR_RE.sub("-", name.strip()).strip("-_")
    if len(safe) > 40:
        safe = safe[:40].strip("-_")
    raw_hash = _sha256(name)[:16]
    return f"{safe}-{raw_hash}" if safe else raw_hash


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
