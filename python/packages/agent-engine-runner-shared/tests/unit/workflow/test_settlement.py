from __future__ import annotations

from typing import Any

import pytest

from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import StepActivityEntry
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import WorkflowIdentity
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.generated.workflow.v1.state_pb2 import StateSnapshot
from agent_engine_runner_shared.workflow import SettledExecutionError, attempt_context_scope
from agent_engine_runner_shared.workflow.settlement import settle_execution


def _attempt() -> AttemptContext:
    return AttemptContext(
        attempt_id="attempt-1",
        fencing_token=7,
        workflow_identity=WorkflowIdentity(
            session_id="session-1",
            execution_id="execution-1",
        ),
    )


@pytest.mark.asyncio
async def test_settlement_finalizes_before_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []

    class Client:
        def __init__(self, oe_url: str) -> None:
            assert oe_url == "http://oe"

        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            pass

        async def finalize_step(self, command: Any) -> list[Any]:
            calls.append(("step", command))
            return []

        async def complete_execution(self, command: Any) -> None:
            calls.append(("complete", command))

    monkeypatch.setattr(
        "agent_engine_runner_shared.workflow.settlement.get_current_oe_url", lambda: "http://oe"
    )
    monkeypatch.setattr(
        "agent_engine_runner_shared.workflow.settlement.AsyncWorkflowClient", Client
    )
    attempt = _attempt()
    state = StateSnapshot()

    with attempt_context_scope(attempt):
        await settle_execution(attempt, state)

    assert [kind for kind, _ in calls] == ["step", "complete"]
    assert calls[0][1].state == state
    assert calls[1][1].state == state


@pytest.mark.asyncio
async def test_settlement_requires_callback_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent_engine_runner_shared.workflow.settlement.get_current_oe_url", lambda: None
    )

    with pytest.raises(SettledExecutionError, match="OE callback URL"):
        await settle_execution(_attempt(), StateSnapshot())


@pytest.mark.asyncio
async def test_settlement_rejects_suspension_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = False

    class Client:
        def __init__(self, oe_url: str) -> None:
            assert oe_url == "http://oe"

        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            pass

        async def finalize_step(self, _command: Any) -> list[StepActivityEntry]:
            return [StepActivityEntry()]

        async def complete_execution(self, _command: Any) -> None:
            nonlocal completed
            completed = True

    monkeypatch.setattr(
        "agent_engine_runner_shared.workflow.settlement.get_current_oe_url", lambda: "http://oe"
    )
    monkeypatch.setattr(
        "agent_engine_runner_shared.workflow.settlement.AsyncWorkflowClient", Client
    )

    attempt = _attempt()
    with (
        attempt_context_scope(attempt),
        pytest.raises(SettledExecutionError, match="suspension entries"),
    ):
        await settle_execution(attempt, StateSnapshot())

    assert completed is False
