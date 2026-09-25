"""Per-project database name resolution with historical-name compatibility.

Mirrors the orchestration-engine Go implementation
(``internal/domains/orchestration-engine/dbresolve.go``) and the memory server
copy (``mongomem_core/db_naming.py``). The codebases are independent, so the
algorithm is intentionally duplicated rather than shared. Keep them in sync.
See ``docs/decisions/008-per-project-database-isolation.md``.

The runner/AER applies this to the agent *store* database only; memory writes
go through the memory-server proxy, which scopes its own database.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Protocol

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


def resolve_project_scoped_db(
    base: str,
    project_id: str,
    existing: Iterable[str],
    legacy_bases: Iterable[str] = (),
) -> str:
    """Derive the effective database name for ``base`` scoped to ``project_id``.

    1. ``project_id == ""`` → return ``base`` unchanged.
    2. ``base`` already ends with ``_{project_id}`` → return as-is (idempotent).
    3. ``scoped = base + "_" + project_id``:
       - ``scoped`` exists on the cluster → use it.
       - else an existing scoped ``legacy_bases`` candidate → use it.
       - else an existing unscoped ``legacy_bases`` candidate → use it.
       - else → use ``scoped`` (fresh deployment).

    Unscoped fallback is limited to known platform defaults during private
    preview. The current base and arbitrary names are never auto-adopted.
    """
    if not project_id:
        return base
    suffix = "_" + project_id
    if base.endswith(suffix):
        return base
    scoped = base + suffix
    names = set(existing)
    legacy_names = tuple(legacy_bases)
    if scoped in names:
        return scoped
    for legacy in legacy_names:
        legacy_scoped = legacy + suffix
        if legacy_scoped in names:
            return legacy_scoped
    for legacy in legacy_names:
        if legacy in names:
            return legacy
    return scoped


def project_scoping_required() -> bool:
    """Whether an empty PROJECT_ID must fail closed (REQUIRE_PROJECT_SCOPED_DB).

    ECP stamps this flag on managed AER pods (where PROJECT_ID is always injected),
    so an empty PROJECT_ID there fails closed rather than silently writing to the
    unscoped store. Local CLI dev leaves the flag unset and uses the unscoped name.
    """
    return os.environ.get("REQUIRE_PROJECT_SCOPED_DB", "").strip().lower() in _TRUTHY


class _ListsDatabaseNames(Protocol):
    def list_database_names(self) -> list[str]: ...


def resolve_effective_db(
    client: _ListsDatabaseNames,
    base: str,
    project_id: str,
    *,
    required: bool | None = None,
    label: str = "database",
    legacy_bases: Iterable[str] = (),
) -> str:
    """Resolve ``base`` against the live cluster via ``list_database_names``.

    Empty ``project_id`` raises when scoping is required (``required``, defaulting
    to :func:`project_scoping_required`), else warns and returns ``base``. A
    listing failure raises so a transient error cannot create a competing
    current-name database beside an existing legacy one. ``legacy_bases`` are
    previous defaults whose project-scoped forms, then bare forms, are adopted
    before a fresh scoped database is created during private preview.
    """
    if required is None:
        required = project_scoping_required()

    if not project_id:
        if required:
            raise RuntimeError(
                f"PROJECT_ID is empty but per-project DB isolation is required "
                f"(REQUIRE_PROJECT_SCOPED_DB); refusing to use the unscoped {label} {base!r}"
            )
        logger.warning(
            "PROJECT_ID is empty; per-project DB isolation disabled, using unscoped %s %r",
            label,
            base,
        )
        return base

    try:
        existing = client.list_database_names()
    except Exception as exc:  # noqa: BLE001 - normalize driver failures at the boundary
        raise RuntimeError(
            f"database discovery failed for project {project_id!r}; refusing to select a database"
        ) from exc

    resolved = resolve_project_scoped_db(base, project_id, existing, legacy_bases)
    if resolved != base + "_" + project_id and not base.endswith("_" + project_id):
        logger.info("using historical %s %r for project %r", label, resolved, project_id)
    return resolved
