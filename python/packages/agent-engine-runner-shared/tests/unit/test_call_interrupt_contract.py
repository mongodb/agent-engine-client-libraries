"""Pin the Python per-call interrupt receiver to the shared contract fixture.

Mirrors TypeScript's ``tests/unit/server_call_interrupt_contract.test.ts``:
both suites execute the same vectors from
``client-libraries/test-fixtures/interrupt-call/contract.json``, so the two
receivers cannot drift on wire behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from agent_engine_runner_shared.server.tool import ToolServer

FIXTURE_PATH = (
    Path(__file__).resolve().parents[6] / "test-fixtures" / "interrupt-call" / "contract.json"
)

with FIXTURE_PATH.open() as fh:
    _CONTRACT = json.load(fh)


def _run_case(monkeypatch: pytest.MonkeyPatch, case: dict) -> None:
    monkeypatch.setenv("RUNNER_AUTH_TOKEN", "s3cret")
    monkeypatch.setenv("APP_ID", "ws-1")

    mock_runtime = Mock()
    mock_runtime._tools = {}
    mock_runtime._tool_definitions = {}
    server = ToolServer(mock_runtime)

    request = case["request"]
    setup = case.get("setup", "none")
    if setup in ("ended", "active"):
        handle = server.drain_registry.begin_work(
            request["execution_id"], step_number=request["step_number"]
        )
    if setup == "ended":
        server.drain_registry.end_work(handle)

    client = TestClient(server.create_app())
    resp = client.post("/interrupt/call", json=request, headers={"Authorization": "Bearer s3cret"})

    expect = case["expect"]
    assert resp.status_code == expect["status"], resp.text
    if "outcome" in expect:
        assert resp.json()["outcome"] == expect["outcome"]


@pytest.mark.parametrize("case", _CONTRACT["cases"], ids=[c["name"] for c in _CONTRACT["cases"]])
def test_call_interrupt_contract(monkeypatch: pytest.MonkeyPatch, case: dict) -> None:
    _run_case(monkeypatch, case)
