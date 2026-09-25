"""Advisory file locks coordinating all MCP OAuth cache writers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

CACHE_LOCK_TIMEOUT_SECONDS = 2.0


def mcp_oauth_cache_lock_path(cache_file: Path) -> Path:
    return Path(f"{cache_file}.lock")


@contextmanager
def mcp_oauth_cache_file_lock(cache_file: Path) -> Iterator[None]:
    lock = FileLock(
        mcp_oauth_cache_lock_path(cache_file),
        timeout=CACHE_LOCK_TIMEOUT_SECONDS,
    )
    try:
        with lock:
            yield
    except Timeout as err:
        raise RuntimeError(f"failed to acquire MCP OAuth cache lock for {cache_file!s}") from err
