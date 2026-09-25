"""Behavioral tests for durable activity Memory synchronization."""

from __future__ import annotations

import json

import pytest

from agent_engine_runner_shared.generated.workflow.v1.activity_pb2 import (
    ActivityContext,
    ActivityMemoryCommand,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import (
    TenantScope,
    WorkflowIdentity,
)
from agent_engine_runner_shared.workflow.memory import DurableMemoryState


def _context() -> ActivityContext:
    return ActivityContext(
        workflow_identity=WorkflowIdentity(
            tenant_scope=TenantScope(
                org_id="org-1",
                project_id="project-1",
                workspace_id="workspace-1",
            ),
            session_id="session-1",
            execution_id="execution-1",
        ),
        activity_id="activity-1",
        attempt_id="attempt-1",
        fencing_token=7,
    )


class _Client:
    def __init__(self) -> None:
        self.commands: list[ActivityMemoryCommand] = []
        self.error: Exception | None = None

    def ensure_memory_written(self, command: ActivityMemoryCommand) -> None:
        self.commands.append(command)
        if self.error is not None:
            raise self.error


def test_llm_sync_writes_pending_input_once_and_each_activity_result() -> None:
    memory = DurableMemoryState("hello")
    client = _Client()

    memory.synchronize_llm(client, _context(), {"content": "hi"}, user_id="user-1")
    memory.synchronize_llm(client, _context(), {"content": "later"}, user_id="user-1")

    first_payloads = [json.loads(write.payload_json) for write in client.commands[0].memory_writes]
    assert [(payload["role"], payload["content"]) for payload in first_payloads] == [
        ("user", "hello"),
        ("assistant", "hi"),
    ]
    assert [write.id for write in client.commands[1].memory_writes] == [
        "workflow:execution-1:activity-1:assistant:0"
    ]
    assert json.loads(client.commands[1].memory_writes[0].payload_json)["content"] == "later"


def test_failed_delivery_retains_pending_input_for_retry() -> None:
    memory = DurableMemoryState("hello")
    client = _Client()
    client.error = RuntimeError("Memory unavailable")

    with pytest.raises(RuntimeError, match="Memory unavailable"):
        memory.synchronize_llm(client, _context(), {"content": "first"}, user_id="user-1")

    client.error = None
    memory.synchronize_llm(client, _context(), {"content": "retry"}, user_id="user-1")

    assert client.commands[0].memory_writes[0].id == "workflow:execution-1:input"
    assert client.commands[1].memory_writes[0].id == "workflow:execution-1:input"


def test_incompatible_llm_result_is_skipped_without_blocking_acknowledgement() -> None:
    memory = DurableMemoryState("hello")
    client = _Client()

    memory.synchronize_llm(
        client,
        _context(),
        {"content": "", "tool_calls": [{"args": ["unsupported"]}]},
        user_id="user-1",
    )

    assert [write.id for write in client.commands[0].memory_writes] == [
        "workflow:execution-1:input"
    ]


def test_empty_tool_result_is_skipped_without_blocking_acknowledgement() -> None:
    memory = DurableMemoryState()
    client = _Client()

    memory.synchronize_tool(
        client,
        _context(),
        None,
        user_id="user-1",
        tool_call_id="call-1",
        tool_name="lookup",
    )

    assert list(client.commands[0].memory_writes) == []


def test_interrupted_tool_marker_is_not_published_as_content() -> None:
    from agent_engine_runner_shared.secure_wrapper import CALL_INTERRUPTED_ARTIFACT_KEY

    memory = DurableMemoryState("approve the claim")
    client = _Client()

    memory.synchronize_tool(
        client,
        _context(),
        {CALL_INTERRUPTED_ARTIFACT_KEY: True},
        user_id="user-1",
        tool_call_id="call-1",
        tool_name="approve_claim",
    )

    payloads = [json.loads(write.payload_json) for write in client.commands[0].memory_writes]
    assert [(payload["role"], payload["content"]) for payload in payloads] == [
        ("user", "approve the claim")
    ]


def test_interrupted_shaped_tool_data_remains_content() -> None:
    memory = DurableMemoryState()
    client = _Client()

    memory.synchronize_tool(
        client,
        _context(),
        {"interrupted": True},
        user_id="user-1",
        tool_call_id="call-1",
        tool_name="lookup",
    )

    (write,) = client.commands[0].memory_writes
    assert json.loads(write.payload_json)["content"] == "{'interrupted': True}"
