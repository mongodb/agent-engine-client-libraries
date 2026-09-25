"""Single source of truth for the consolidated data-plane database name.

All platform-owned stores (execution logs, checkpoints, memory, traces)
live in one database.  Override via the ``MDB_AGENTIC_STORE_DB`` environment
variable; the default is ``mdb_store``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_engine_runner_shared.db_naming import _ListsDatabaseNames

_DEFAULT_DB_NAME = "mdb_store"
_LEGACY_DEFAULT_DB_NAMES = ("mdb_agentic_store",)
_ENV_VAR = "MDB_AGENTIC_STORE_DB"

# Memoized per-project-resolved store DB names, keyed by base name. Resolution
# requires a live MongoClient (to list databases) which is not available when
# get_store_db_name runs during construction, so it is deferred to the first
# consumer that has a client and cached process-wide — one list_database_names
# round trip per base, shared by traces, the checkpointer, and ADK sessions
# (which use the same base in normal deployments).
_resolved_store_db: dict[str, str] = {}


def get_store_db_name() -> str:
    """Return the base (unscoped) consolidated data-plane database name.

    Reads ``MDB_AGENTIC_STORE_DB`` from the environment on every call so that
    late configuration (e.g. ``load_dotenv()`` after import) is respected. For
    the per-project-scoped name use :func:`resolve_store_db_name`.
    """
    value = os.environ.get(_ENV_VAR, "").strip()
    return value if value else _DEFAULT_DB_NAME


def resolve_store_db_name(
    client: "_ListsDatabaseNames",
    base: str | None = None,
) -> str:
    """Return the per-project-scoped store DB name, resolved once and cached.

    Applies the same resolution the OE uses (``agent_engine_runner_shared.db_naming``) against
    the live cluster, so the AER/SDK writers converge on the same database the OE
    reads. An explicit ``base`` or ``MDB_AGENTIC_STORE_DB`` value is returned
    exactly and skips discovery. Otherwise the platform default is resolved and
    memoized so only the first caller pays the round trip.
    """
    if base:
        return base
    configured = os.environ.get(_ENV_VAR, "").strip()
    if configured:
        return configured
    base = _DEFAULT_DB_NAME
    if base not in _resolved_store_db:
        from agent_engine_runner_shared.db_naming import resolve_effective_db

        _resolved_store_db[base] = resolve_effective_db(
            client,
            base,
            os.environ.get("PROJECT_ID", "").strip(),
            label="agent store database",
            legacy_bases=_LEGACY_DEFAULT_DB_NAMES,
        )
    return _resolved_store_db[base]


def reset_store_db_cache() -> None:
    """Clear the memoized resolved store DB names (test seam)."""
    _resolved_store_db.clear()
