"""Materialize deployed MCP OAuth secrets into the file-backed token cache."""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from agent_engine_runner_shared.mcp_oauth_cache_lock import mcp_oauth_cache_file_lock
from agent_engine_runner_shared.mcp_oauth_cache_name import mcp_oauth_cache_name

MCP_OAUTH_DIR_ENV = "AGENTIC_MCP_OAUTH_DIR"
MCP_OAUTH_SECRET_ENV_PREFIX = "AGENTIC_MCP_OAUTH_B64_"
PLATFORM_MCP_OAUTH_CACHE_DIR = Path("/tmp/agentic/mcp-oauth")
_SAFE_ENV_CACHE_NAME_RE = re.compile(r"^[A-Z0-9_]+$")

__all__ = ["materialize_mcp_oauth_secret_cache"]


def materialize_mcp_oauth_secret_cache(
    *,
    default_cache_dir: Path = PLATFORM_MCP_OAUTH_CACHE_DIR,
) -> int:
    """Establish the writable runtime cache and decode MCP OAuth secrets into it."""

    cache_dir = Path(os.environ.get(MCP_OAUTH_DIR_ENV, default_cache_dir))
    os.environ.setdefault(MCP_OAUTH_DIR_ENV, str(cache_dir))
    cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    secret_env = _mcp_oauth_secret_env()
    if not secret_env:
        return 0

    materialized = 0
    for env_name, encoded_cache in secret_env:
        raw_cache, payload = _decode_cache_secret(env_name, encoded_cache)
        cache_name = _cache_name(env_name, payload)
        _write_cache_file(cache_dir, cache_name, raw_cache)
        os.environ.pop(env_name, None)
        materialized += 1

    return materialized


def _mcp_oauth_secret_env() -> list[tuple[str, str]]:
    return sorted(
        (name, value)
        for name, value in os.environ.items()
        if name.startswith(MCP_OAUTH_SECRET_ENV_PREFIX)
    )


def _decode_cache_secret(env_name: str, encoded_cache: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw_cache = base64.b64decode(encoded_cache, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(
            f"invalid MCP OAuth secret {env_name}: value is not valid base64"
        ) from exc

    try:
        payload = json.loads(raw_cache.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"invalid MCP OAuth secret {env_name}: decoded value is not valid JSON"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"invalid MCP OAuth secret {env_name}: decoded value must be a JSON object"
        )
    return raw_cache, payload


def _cache_name(env_name: str, payload: dict[str, Any]) -> str:
    # The runtime looks the cache up by the configured server URL's hash, so a
    # payload without server_url could never be found — fail loudly instead of
    # materializing an unreadable file.
    server_url = payload.get("server_url")
    if not isinstance(server_url, str) or not server_url.strip():
        raise RuntimeError(f"invalid MCP OAuth secret {env_name}: missing server_url")

    server_name = payload.get("server_name")
    if isinstance(server_name, str) and server_name.strip():
        return mcp_oauth_cache_name(server_name, server_url)

    env_cache_name = env_name.removeprefix(MCP_OAUTH_SECRET_ENV_PREFIX)
    if not env_cache_name or not _SAFE_ENV_CACHE_NAME_RE.fullmatch(env_cache_name):
        raise RuntimeError(f"invalid MCP OAuth secret {env_name}: invalid cache name")
    return mcp_oauth_cache_name(env_cache_name.lower(), server_url)


def _write_cache_file(cache_dir: Path, cache_name: str, raw_cache: bytes) -> None:
    cache_file = cache_dir / f"{cache_name}.json"
    with mcp_oauth_cache_file_lock(cache_file):
        fd, tmp_name = tempfile.mkstemp(prefix=f".tmp-{cache_name}-", dir=cache_dir)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw_cache)
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, cache_file)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
