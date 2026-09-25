"""Pin the Python drain receiver to the shared contract fixture.

Mirrors TypeScript's ``tests/unit/server_drain_contract.test.ts``: both suites
execute the same vectors from
``client-libraries/test-fixtures/drain/contract.json``, so the two receivers
cannot drift on wire behavior.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from agent_engine_runner_shared.server.tool import ToolServer

FIXTURE_PATH = Path(__file__).resolve().parents[6] / "test-fixtures" / "drain" / "contract.json"

with FIXTURE_PATH.open() as fh:
    _CONTRACT = json.load(fh)


def _run_case(monkeypatch: pytest.MonkeyPatch, case: dict) -> None:
    monkeypatch.setenv("RUNNER_AUTH_TOKEN", "s3cret")
    monkeypatch.setenv("APP_ID", "ws-1")

    mock_runtime = Mock()
    mock_runtime._tools = {}
    mock_runtime._tool_definitions = {}
    server = ToolServer(mock_runtime)

    execution_id = case["request"]["execution_id"]
    setup = case.get("setup", "none")
    if setup in ("ended", "active"):
        handle = server.drain_registry.begin_work(execution_id)
    if setup == "ended":
        server.drain_registry.end_work(handle)

    request = dict(case["request"])
    if "deadline_offset_ms" in request:
        request["deadline_at_ms"] = int(time.time() * 1000) + request.pop("deadline_offset_ms")

    client = TestClient(server.create_app())
    resp = client.post("/drain", json=request, headers={"Authorization": "Bearer s3cret"})

    expect = case["expect"]
    assert resp.status_code == expect["status"], resp.text
    if "outcome" in expect:
        assert resp.json()["outcome"] == expect["outcome"]
    if "reason_code" in expect:
        assert resp.json().get("reason_code") == expect["reason_code"]


@pytest.mark.parametrize("case", _CONTRACT["cases"], ids=[c["name"] for c in _CONTRACT["cases"]])
def test_drain_contract(monkeypatch: pytest.MonkeyPatch, case: dict) -> None:
    _run_case(monkeypatch, case)
