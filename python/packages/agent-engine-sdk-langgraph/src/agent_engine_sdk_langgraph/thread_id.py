"""Workspace-scoped LangGraph ``thread_id`` composition.

LangGraph's ``MongoDBSaver`` keys checkpoints by ``(thread_id, checkpoint_ns,
checkpoint_id)`` and exposes no tenant field. Within one project (one MongoDB
database), two agents that happen to reuse the same ``session_id`` would read
and write the same checkpoint documents, silently corrupting each other's
state. We scope the key by suffixing the ``workspace_id`` so the composite is
unique per agent.

By default callers supply ``session_id`` and the platform composes the
``thread_id`` on write, then strips it back to ``session_id`` on read.
Agents that need a different formula (for example Holly-style
``{session_id}__{actor_id}``) register ``@app.resolve_thread_id``; that
hook's return value is used verbatim and these helpers are not applied on
the write path. Atlas Agent Engine session-history queries still look up derived keys
only. The OE rejects user-supplied ``session_id`` values containing ``":"``
(HTTP 400), so the default suffix boundary is always unambiguous and
``session_id_from_thread_id`` can split on the last ``":"`` without risk of
clipping the caller's own id.

Both the write path (``agent.py``, default derivation) and the read path
(``query.py``) import these functions so the composition can never drift
between them.
"""

from __future__ import annotations

from collections.abc import Callable

from agent_engine_sdk import RequestContext

# Fallback thread_id when no session_id is supplied. Mirrors the historical
# "default" used by the agent runner before workspace scoping existed.
_DEFAULT_SESSION_ID = "default"


def scoped_thread_id(session_id: str | None, workspace_id: str | None) -> str:
    """Compose the workspace-scoped ``thread_id`` for a checkpoint key.

    Returns ``f"{session_id}:{workspace_id}"`` when ``workspace_id`` is set,
    otherwise the bare ``session_id`` (so an agent running without a workspace
    — e.g. local dev — still gets a usable, if unscoped, key).
    """
    sid = session_id or _DEFAULT_SESSION_ID
    if not workspace_id:
        return sid
    return f"{sid}:{workspace_id}"


def checkpoint_thread_id(
    ctx: RequestContext,
    resolve_thread_id: Callable[[RequestContext], str] | None = None,
) -> str:
    """LangGraph ``thread_id`` for a platform request — the same key invoke uses."""
    if resolve_thread_id is not None:
        thread_id = resolve_thread_id(ctx)
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError(
                "@app.resolve_thread_id must return a non-empty string; "
                f"got {thread_id!r}"
            )
        return thread_id
    return scoped_thread_id(ctx.session_id, ctx.workspace_id)


def thread_ids_for_query(session_id: str, workspace_id: str | None) -> list[str]:
    """Return the ``thread_id`` keys that may hold checkpoints for a session.

    When a workspace scope is known, only the composite
    ``f"{session_id}:{workspace_id}"`` key is queried — never the bare
    ``session_id``. Bare keys are shared by every workspace on the same
    store, so including them would let one tenant read or poison another's
    conversation history. Legacy unscoped checkpoints are intentionally not
    queried.
    """
    if not workspace_id:
        return [session_id]
    return [scoped_thread_id(session_id, workspace_id)]


def thread_ids_for_sessions_query(
    session_ids: list[str], workspace_id: str | None
) -> list[str]:
    """Flatten :func:`thread_ids_for_query` across many sessions, deduped."""
    seen: set[str] = set()
    result: list[str] = []
    for session_id in session_ids:
        for thread_id in thread_ids_for_query(session_id, workspace_id):
            if thread_id not in seen:
                seen.add(thread_id)
                result.append(thread_id)
    return result


def session_id_from_thread_id(thread_id: str, workspace_id: str | None) -> str:
    """Recover the plain ``session_id`` from a workspace-scoped ``thread_id``.

    Inverse of :func:`scoped_thread_id`. Strips a trailing
    ``f":{workspace_id}"`` when present; returns ``thread_id`` unchanged when
    ``workspace_id`` is empty or the suffix is absent (e.g. legacy checkpoints
    written before scoping existed).
    """
    if not workspace_id:
        return thread_id
    suffix = f":{workspace_id}"
    if thread_id.endswith(suffix):
        return thread_id[: -len(suffix)]
    return thread_id
