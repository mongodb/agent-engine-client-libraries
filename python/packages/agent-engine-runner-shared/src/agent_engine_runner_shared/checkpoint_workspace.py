"""Checkpoint ``thread_id`` workspace scope — shared by write and read paths."""

from __future__ import annotations

import os

from agent_engine_runner_shared.db_naming import project_scoping_required


def resolve_checkpoint_workspace_id(wire_workspace_id: str | None = None) -> str | None:
    """Resolve the workspace scope used for LangGraph checkpoint thread_ids.

    The pod's ``APP_ID`` (set by ECP at deploy time) is authoritative. Managed
    runtimes require it and fail closed when it is missing. Intentionally
    unscoped local and integration-test runtimes may instead use the wire
    ``workspace_id`` from the most recent ``/execute`` request so writes and
    reads resolve to the same composite key. When neither is set in those
    runtimes, checkpoints remain unscoped (bare ``session_id``).
    """
    app_id = os.environ.get("APP_ID")
    if app_id:
        return app_id
    if project_scoping_required():
        raise RuntimeError("checkpoint workspace scope is required but APP_ID is not set")
    if wire_workspace_id:
        return wire_workspace_id
    return None
