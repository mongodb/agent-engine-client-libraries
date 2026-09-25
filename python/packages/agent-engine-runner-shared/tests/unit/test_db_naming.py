"""Tests for per-project store-DB resolution (agent_engine_runner_shared.db_naming + db_config).

Mirrors the memory-server and OE test suites — kept in sync deliberately.
The runner/AER resolves the same algorithm as the OE so writers and
readers converge on the same database.
"""

from __future__ import annotations

import pytest

from agent_engine_runner_shared import db_config
from agent_engine_runner_shared.db_naming import (
    project_scoping_required,
    resolve_effective_db,
    resolve_project_scoped_db,
)

PROJ = "proj1"


@pytest.mark.parametrize(
    "base, project_id, existing, expected",
    [
        (
            "mdb_agentic_store",
            "",
            ["mdb_agentic_store", "mdb_agentic_store_proj1"],
            "mdb_agentic_store",
        ),
        ("custom_store", "", [], "custom_store"),
        ("mdb_agentic_store_proj1", PROJ, [], "mdb_agentic_store_proj1"),
        ("mdb_agentic_store", PROJ, ["mdb_agentic_store_proj1"], "mdb_agentic_store_proj1"),
        (
            "mdb_agentic_store",
            PROJ,
            ["mdb_agentic_store", "admin"],
            "mdb_agentic_store_proj1",
        ),
        ("mdb_agentic_store", PROJ, ["admin"], "mdb_agentic_store_proj1"),
        (
            "mdb_agentic_store",
            PROJ,
            ["mdb_agentic_store", "mdb_agentic_store_proj1"],
            "mdb_agentic_store_proj1",
        ),
        ("mdb_agentic_store", PROJ, ["mdb_agentic_store_other"], "mdb_agentic_store_proj1"),
        ("mdb_agentic_store_proj1_extra", PROJ, [], "mdb_agentic_store_proj1_extra_proj1"),
    ],
)
def test_resolve_project_scoped_db(base, project_id, existing, expected):
    assert resolve_project_scoped_db(base, project_id, existing) == expected


class _FakeClient:
    def __init__(self, names):
        self._names = names
        self.calls = 0

    def list_database_names(self):
        self.calls += 1
        return self._names


class _FailingClient:
    def list_database_names(self):
        raise RuntimeError("cluster unreachable")


def test_resolve_effective_db_uses_scoped_when_present():
    assert (
        resolve_effective_db(_FakeClient(["mdb_agentic_store_proj1"]), "mdb_agentic_store", PROJ)
        == "mdb_agentic_store_proj1"
    )


def test_resolve_effective_db_adopts_previous_unscoped_default():
    assert (
        resolve_effective_db(
            _FakeClient(["mdb_agentic_store"]),
            "mdb_store",
            PROJ,
            legacy_bases=["mdb_agentic_store"],
        )
        == "mdb_agentic_store"
    )


def test_resolve_effective_db_adopts_previous_scoped_default():
    project_id = "0123456789abcdef01234567"
    assert (
        resolve_effective_db(
            _FakeClient(["mdb_agentic_store", f"mdb_agentic_store_{project_id}"]),
            "mdb_store",
            project_id,
            legacy_bases=["mdb_agentic_store"],
        )
        == f"mdb_agentic_store_{project_id}"
    )


def test_resolve_effective_db_fails_closed_on_list_failure():
    with pytest.raises(RuntimeError, match="database discovery failed"):
        resolve_effective_db(_FailingClient(), "mdb_agentic_store", PROJ)


def test_resolve_effective_db_empty_project_not_required(monkeypatch):
    monkeypatch.delenv("REQUIRE_PROJECT_SCOPED_DB", raising=False)
    assert resolve_effective_db(_FakeClient([]), "mdb_agentic_store", "") == "mdb_agentic_store"


def test_resolve_effective_db_empty_project_required_raises():
    with pytest.raises(RuntimeError, match="PROJECT_ID is empty"):
        resolve_effective_db(_FakeClient([]), "mdb_agentic_store", "", required=True)


def test_project_scoping_required_reads_env(monkeypatch):
    monkeypatch.delenv("REQUIRE_PROJECT_SCOPED_DB", raising=False)
    assert project_scoping_required() is False
    monkeypatch.setenv("REQUIRE_PROJECT_SCOPED_DB", "yes")
    assert project_scoping_required() is True


def test_default_scoped_store_name_fits_atlas_flex_limit():
    project_id = "0123456789abcdef01234567"
    assert db_config.get_store_db_name() == "mdb_store"
    assert len(f"{db_config.get_store_db_name()}_{project_id}".encode()) <= 38


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    monkeypatch.delenv("MDB_AGENTIC_STORE_DB", raising=False)
    monkeypatch.delenv("REQUIRE_PROJECT_SCOPED_DB", raising=False)
    db_config.reset_store_db_cache()
    yield
    db_config.reset_store_db_cache()


def test_resolve_store_db_name_memoizes(monkeypatch):
    monkeypatch.setenv("PROJECT_ID", "proj1")
    client = _FakeClient(["mdb_agentic_store_proj1"])
    assert db_config.resolve_store_db_name(client) == "mdb_agentic_store_proj1"
    # Second call returns the cached value without another round trip.
    assert db_config.resolve_store_db_name(client) == "mdb_agentic_store_proj1"
    assert client.calls == 1


def test_resolve_store_db_name_uses_env_base(monkeypatch):
    monkeypatch.setenv("PROJECT_ID", "proj1")
    monkeypatch.setenv("MDB_AGENTIC_STORE_DB", "custom_store")
    client = _FakeClient([])
    assert db_config.resolve_store_db_name(client) == "custom_store"
    assert client.calls == 0


def test_explicit_override_bypasses_cached_default(monkeypatch):
    monkeypatch.setenv("PROJECT_ID", "proj1")
    client = _FakeClient(["mdb_agentic_store_proj1"])
    assert db_config.resolve_store_db_name(client) == "mdb_agentic_store_proj1"

    monkeypatch.setenv("MDB_AGENTIC_STORE_DB", "mdb_store")
    assert db_config.resolve_store_db_name(client) == "mdb_store"
    assert client.calls == 1


def test_programmatic_override_is_exact(monkeypatch):
    monkeypatch.setenv("PROJECT_ID", "proj1")
    client = _FakeClient(["mdb_agentic_store_proj1"])
    assert db_config.resolve_store_db_name(client, base="custom_store") == "custom_store"
    assert client.calls == 0


def test_empty_programmatic_override_uses_managed_default(monkeypatch):
    monkeypatch.setenv("PROJECT_ID", "proj1")
    client = _FakeClient(["mdb_agentic_store_proj1"])
    assert db_config.resolve_store_db_name(client, base="") == "mdb_agentic_store_proj1"
    assert client.calls == 1
