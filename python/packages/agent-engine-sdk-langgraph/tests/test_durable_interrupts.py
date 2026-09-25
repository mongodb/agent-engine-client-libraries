"""Durable session handling for native LangGraph interrupts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.errors import GraphInterrupt
from langgraph.types import Command, Interrupt

from agent_engine_sdk_langgraph import durable_session as module
from agent_engine_sdk_langgraph.checkpoint_branch import pending_interrupts_from_state
from agent_engine_sdk_langgraph.durable_session import settle_interrupts
from agent_engine_runner_shared.context import current_oe_url
from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ACTIVITY_KIND_TOOL,
    ACTIVITY_OUTCOME_KIND_COMPLETED,
    ACTIVITY_OUTCOME_KIND_SUSPENDED,
    ActivityOutcome,
    StepActivityEntry,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import WorkflowIdentity
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.workflow import attempt_context_scope
from agent_engine_runner_shared.workflow.activity import build_activity_command
from agent_engine_runner_shared.workflow.context import (
    record_interrupted_activity,
    record_observed_activity,
)
from agent_engine_runner_shared.workflow.protojson import proto_value_to_json


def _attempt() -> AttemptContext:
    return AttemptContext(
        attempt_id="attempt-1",
        fencing_token=1,
        workflow_identity=WorkflowIdentity(
            session_id="session-1",
            execution_id="execution-1",
        ),
    )


def _interrupt(value: Any, native_id: str) -> Interrupt:
    return Interrupt(value=value, id=native_id)


class _Client:
    def __init__(self, *, resolved: bool = False) -> None:
        self.resolved = resolved
        self.commands: list[Any] = []

    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        pass

    async def finalize_step(self, command: Any) -> list[StepActivityEntry]:
        self.commands.append(command)
        entries: list[StepActivityEntry] = []
        suspensions = sorted(
            command.suspensions,
            key=lambda item: item.position.activity_ordinal,
        )
        for suspension in suspensions:
            outcome = ActivityOutcome(
                activity_id=f"activity-{suspension.position.activity_ordinal}",
                outcome_kind=(
                    ACTIVITY_OUTCOME_KIND_COMPLETED
                    if self.resolved
                    else ACTIVITY_OUTCOME_KIND_SUSPENDED
                ),
            )
            if self.resolved:
                outcome.result.string_value = (
                    "answer-a"
                    if suspension.position.activity_ordinal == 1
                    else "answer-b"
                )
            else:
                outcome.suspension.CopyFrom(suspension.suspension)
            entries.append(
                StepActivityEntry(position=suspension.position, outcome=outcome)
            )
        return entries


async def _settle(
    monkeypatch: pytest.MonkeyPatch,
    client: _Client,
) -> Command | list[dict[str, Any]]:
    monkeypatch.setattr(module, "AsyncWorkflowClient", lambda _: client)
    url_token = current_oe_url.set("http://oe")
    try:
        attempt = _attempt()
        with attempt_context_scope(attempt):
            interrupts = [
                _interrupt({"branch": "b"}, "native-b"),
                _interrupt({"branch": "a"}, "native-a"),
            ]
            for ordinal, interrupt in ((2, interrupts[0]), (1, interrupts[1])):
                command = build_activity_command(
                    attempt=attempt,
                    kind=ACTIVITY_KIND_TOOL,
                    name=f"review-{ordinal}",
                    activity_ordinal=ordinal,
                    semantic_input={"arguments": {"branch": interrupt.value["branch"]}},
                )
                record_observed_activity(command.position)
                record_interrupted_activity(command, GraphInterrupt([interrupt]))
            return await settle_interrupts(
                attempt=attempt,
                interrupts=interrupts,
            )
    finally:
        current_oe_url.reset(url_token)


@pytest.mark.anyio
async def test_new_interrupts_finalize_one_atomic_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client()

    result = await _settle(monkeypatch, client)

    assert result == [
        {"id": "activity-1", "value": {"branch": "a"}},
        {"id": "activity-2", "value": {"branch": "b"}},
    ]
    command = client.commands[0]
    assert len(command.suspensions) == 2
    assert all(
        entry.activity_kind == ACTIVITY_KIND_TOOL for entry in command.suspensions
    )
    assert [
        proto_value_to_json(entry.semantic_input) for entry in command.suspensions
    ] == [
        {"arguments": {"branch": "b"}},
        {"arguments": {"branch": "a"}},
    ]
    assert {
        position.activity_ordinal for position in command.observed_activity_positions
    } == {1, 2}


@pytest.mark.anyio
async def test_recorded_answers_resume_fresh_native_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = await _settle(monkeypatch, _Client(resolved=True))

    assert isinstance(result, Command)
    assert result.resume == {"native-a": "answer-a", "native-b": "answer-b"}


@pytest.mark.anyio
async def test_interrupt_rejects_inexact_integer_before_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client()
    monkeypatch.setattr(module, "AsyncWorkflowClient", lambda _: client)
    url_token = current_oe_url.set("http://oe")
    try:
        attempt = _attempt()
        interrupt = _interrupt({"nested": [2**53]}, "native-1")
        with attempt_context_scope(attempt):
            command = build_activity_command(
                attempt=attempt,
                kind=ACTIVITY_KIND_TOOL,
                name="review",
                activity_ordinal=1,
                semantic_input={"arguments": {}},
            )
            record_observed_activity(command.position)
            record_interrupted_activity(command, GraphInterrupt([interrupt]))
            with pytest.raises(TypeError, match="JSON-safe"):
                await settle_interrupts(attempt=attempt, interrupts=[interrupt])
    finally:
        current_oe_url.reset(url_token)

    assert client.commands == []


def test_checkpoint_interrupts_deduplicate_parent_projection() -> None:
    child_interrupt = Interrupt(value={"approve": True}, id="native-1")
    parent_interrupt = Interrupt(value={"approve": True}, id="native-1")
    child_task = SimpleNamespace(
        path=("__pregel_pull", "review"),
        name="review",
        interrupts=(child_interrupt,),
        state=None,
    )
    child = SimpleNamespace(
        config={"configurable": {"checkpoint_ns": "subgraph:task-1"}},
        tasks=(child_task,),
    )
    root = SimpleNamespace(
        config={"configurable": {"checkpoint_ns": ""}},
        tasks=(
            SimpleNamespace(
                path=("__pregel_pull", "subgraph"),
                name="subgraph",
                interrupts=(parent_interrupt,),
                state=child,
            ),
        ),
    )

    pending = pending_interrupts_from_state(root)

    assert pending == [child_interrupt]
