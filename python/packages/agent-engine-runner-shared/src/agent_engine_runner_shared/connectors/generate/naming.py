"""Deterministic tool naming for compiled operations."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from agent_engine_runner_shared.connectors.generate.spec import GenerateError, _quote


def _operation_raw_name(operation: dict[str, Any], method: str, path: str) -> str:
    """The identity a tool name is derived from: the operationId, or a
    deterministic method/path fallback for specs without operationIds."""
    raw = operation.get("operationId")
    if raw is None:
        return method.lower() + "_" + re.sub(r"[^A-Za-z0-9]+", "_", path.strip("/")).strip("_")
    if not isinstance(raw, str) or not raw:
        raise GenerateError(
            f"operation at {method} {_quote(path)}: operationId must be a non-empty string"
        )
    return raw


def _assign_tool_names(connector: str, raw_names: list[str], identities: list[str]) -> list[str]:
    """Deterministic MCP-safe tool names whose collision resolution is
    independent of document order: every member of a normalized-name collision
    group gets a digest suffix derived from its own full operation identity
    (including method and path, since legal ``$ref`` composition reuses
    operationIds), so reordering a spec never swaps which operation owns which
    persisted name."""
    bases = [re.sub(r"[^A-Za-z0-9_-]", "_", f"{connector}_{raw}")[:64] for raw in raw_names]
    collisions = {base for base in bases if bases.count(base) > 1}
    names: list[str] = []
    for base, identity in zip(bases, identities):
        if base in collisions:
            digest = hashlib.sha256(identity.encode()).hexdigest()[:8]
            base = f"{base[:55]}-{digest}"
        if not base.strip("_-"):
            base = "operation"
        names.append(base)
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise GenerateError(
            f"operations produce colliding tool names: {duplicates}; "
            "operationIds must be unique within a spec"
        )
    return names
