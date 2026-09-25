"""
/health route tests: the tracing degraded/attached/disabled mapping onto the
HTTP response. Tracing state is forced through tracing_setup's
module flags so the real tracing_status() and route handler produce the
response; assertions hit the route via FastAPI's TestClient — no internal
flag inspection, so a broken health-response mapping fails here even when
unit tests pass.
"""

import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import agent_engine_runner_shared.tracing.setup as tracing_setup
from agent_engine_runner_shared.server.base import BaseServer


class _HealthTestServer(BaseServer):
    @property
    def mode_name(self) -> str:
        return "aer"

    def register_routes(self, app) -> None:
        pass

    def get_health_details(self):
        return {"tools_registered": 2}


@pytest.fixture
def reset_tracing_flags(monkeypatch):
    """Isolate tracing_setup's exporter-state flags across tests."""
    monkeypatch.setattr(tracing_setup, "_mongodb_exporter_attached", False)
    monkeypatch.setattr(tracing_setup, "_mongodb_degraded", False)
    return monkeypatch


def _health_client() -> TestClient:
    server = _HealthTestServer(SimpleNamespace(app_name="test-app"))
    return TestClient(server.create_app())


def test_health_reports_degraded_when_trace_store_unreachable(reset_tracing_flags):
    reset_tracing_flags.setattr(tracing_setup, "_mongodb_degraded", True)

    resp = _health_client().get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["details"]["tracing"] == {"database_exporter": "degraded"}
    # Mode-specific details survive alongside the nested tracing detail.
    assert body["details"]["tools_registered"] == 2
    assert body["component"] == "test-app"
    assert body["mode"] == "aer"
    assert body["version"] == "1.0.0"


def test_health_reports_healthy_when_trace_store_attached(reset_tracing_flags):
    reset_tracing_flags.setattr(tracing_setup, "_mongodb_exporter_attached", True)

    resp = _health_client().get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["details"]["tracing"] == {"database_exporter": "attached"}
    assert body["details"]["tools_registered"] == 2


def test_health_reports_tracing_disabled_when_no_store_configured(reset_tracing_flags):
    resp = _health_client().get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["details"]["tracing"] == {"database_exporter": "disabled"}
    assert body["details"]["tools_registered"] == 2


def test_health_ok_when_tracing_extra_not_installed(monkeypatch, reset_tracing_flags):
    """A base install without the optional tracing extra must not 500 on /health.

    None in sys.modules makes the handler's `from agent_engine_runner_shared.tracing import
    tracing_status` raise ImportError — the same failure a pod without the
    tracing extra hits. The route must fall back to "disabled" and stay 200,
    mirroring the ImportError tolerance in TenantRuntime and the AER path.
    """
    monkeypatch.setitem(sys.modules, "agent_engine_runner_shared.tracing", None)

    resp = _health_client().get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["details"]["tracing"] == {"database_exporter": "disabled"}
    assert body["details"]["tools_registered"] == 2
