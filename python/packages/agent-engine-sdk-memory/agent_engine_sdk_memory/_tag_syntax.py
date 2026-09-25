"""Syntax-only client-side checks for custom-type names and tag payloads.

Mirrors the shape rules of the memory-server's tag canonicalization with
byte-identical messages, so a fail-fast client error reads the same as a
server rejection. Deliberately NOT mirrored (server-owned): the int64 range
guard, dotted+nested mixed-form detection, operator-expression detection,
key-count caps, and declaration membership — those values reach the platform
intact, so the server's own rejection is the one the caller sees.

Non-finite floats are the exception: JSON cannot represent them, so the value
never survives the wire for the server to judge. Left unchecked, Python raises
an opaque encoder error naming no tag, and JavaScript silently serializes NaN
to null. Rejecting here is what makes the server's rule reachable at all.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping
from typing import Any, cast

from agent_engine_sdk_memory.errors import MemoryClientError

# Mirror of pkg/memoryconfig/reserved_names.json (parity-tested in-repo).
RESERVED_TYPE_NAMES: frozenset[str] = frozenset(
    {
        "semantic",
        "episodic",
        "taxonomic",
        "procedural",
        "snapshot",
        "entity",
        "preferences",
        "turn",
        "stm",
        "short_term",
    }
)

_MAX_TAG_KEY_DEPTH = 2

# Mirror of the name_pattern in pkg/memoryconfig/reserved_names.json
# (parity-tested in-repo). A name that cannot be declared can never exist, so
# checking it here turns an unroutable request into a precise local error.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def validate_memory_type(memory_type: Any) -> None:
    """Reject empty, reserved, and malformed type names before any HTTP call."""
    if not isinstance(memory_type, str) or not memory_type.strip():
        raise MemoryClientError("memory_type must be a non-empty string")
    if memory_type in RESERVED_TYPE_NAMES:
        raise MemoryClientError(
            f"'{memory_type}' is a built-in memory type and is not"
            " accepted by this operation"
        )
    if not _NAME_RE.match(memory_type):
        raise MemoryClientError(
            f"custom type name {memory_type!r} must match {_NAME_RE.pattern}"
        )


def _check_scalar(path: str, value: Any) -> None:
    if isinstance(value, str):
        if not value.strip():
            raise MemoryClientError(f"tag '{path}' must have a non-empty string value")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise MemoryClientError(f"tag '{path}' must be a finite number")
    if not isinstance(value, (bool, int, float)):
        raise MemoryClientError(
            f"tag '{path}' must be a non-empty string, number, or boolean"
        )


def _check_path(path: str) -> None:
    segments = path.split(".")
    if any(not segment for segment in segments):
        raise MemoryClientError(f"tag '{path}' must have a non-empty path")
    if len(segments) > _MAX_TAG_KEY_DEPTH:
        raise MemoryClientError(f"tag '{path}': nesting is at most one level deep")


def _flatten(tags: Mapping[str, Any]) -> Iterator[tuple[str, Any]]:
    """Yield (dotted path, value) pairs from either accepted wire form.

    Group-level rules are applied here, so the per-leaf checks that follow run
    over one uniform sequence rather than once per wire form.
    """
    for raw_key, value in tags.items():
        key = str(raw_key)
        if not isinstance(value, Mapping):
            yield key, value
            continue
        if "." in key:
            raise MemoryClientError(f"tag '{key}': nesting is at most one level deep")
        if not value:
            raise MemoryClientError(f"tag group '{key}' must contain at least one key")
        for subkey, subvalue in cast(Mapping[Any, Any], value).items():
            if (
                not isinstance(subkey, str)
                or "." in subkey
                or isinstance(subvalue, Mapping)
            ):
                raise MemoryClientError(
                    f"tag '{key}': nesting is at most one level deep"
                )
            yield f"{key}.{subkey}", subvalue


def validate_tag_syntax(tags: Mapping[str, Any]) -> None:
    """Validate key syntax and scalar value types for either wire form."""
    for path, value in _flatten(tags):
        _check_path(path)
        _check_scalar(path, value)
