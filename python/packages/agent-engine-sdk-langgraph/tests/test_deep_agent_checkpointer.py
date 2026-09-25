"""Deep Agent's private durable checkpointer policy."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Send

from agent_engine_sdk_langgraph.deep_agent_checkpointer import (
    checkpointer_for_deep_agent,
)
from agent_engine_sdk_langgraph.platform_checkpointer import (
    PlatformCheckpointer,
    UnsupportedDurableGraphError,
)
from agent_engine_runner_shared.generated.workflow.v1.common_pb2 import WorkflowIdentity
from agent_engine_runner_shared.generated.workflow.v1.runtime_pb2 import AttemptContext
from agent_engine_runner_shared.workflow import attempt_context_scope


def _attempt() -> AttemptContext:
    return AttemptContext(
        attempt_id="attempt-1",
        fencing_token=1,
        owner_id="aer-1",
        workflow_identity=WorkflowIdentity(
            session_id="session-1",
            execution_id="execution-1",
        ),
    )


def test_only_deep_agent_checkpointer_accepts_send_writes() -> None:
    platform = PlatformCheckpointer(native=InMemorySaver())
    deep_agent = checkpointer_for_deep_agent(platform)
    writes = [("__pregel_tasks", [Send("tools", [{"id": "call-1"}])])]

    assert isinstance(deep_agent, PlatformCheckpointer)
    assert deep_agent.native is platform.native
    assert deep_agent._scratch is platform._scratch

    with attempt_context_scope(_attempt()):
        config = platform.put(
            {},
            empty_checkpoint(),
            {"source": "input", "step": -1},
            {},
        )
        with pytest.raises(UnsupportedDurableGraphError, match="Send"):
            platform.put_writes(config, writes, "task-1")
        deep_agent.put_writes(config, writes, "task-1")


def test_only_deep_agent_checkpointer_accepts_send_checkpoint() -> None:
    platform = PlatformCheckpointer(native=InMemorySaver())
    deep_agent = checkpointer_for_deep_agent(platform)
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {
        "__pregel_tasks": [Send("tools", [{"id": "call-1"}])]
    }

    with attempt_context_scope(_attempt()):
        with pytest.raises(UnsupportedDurableGraphError, match="Send"):
            platform.put({}, checkpoint, {"source": "input", "step": -1}, {})
        deep_agent.put({}, checkpoint, {"source": "input", "step": -1}, {})


@pytest.mark.anyio
async def test_deep_agent_checkpointer_accepts_send_writes_async() -> None:
    platform = PlatformCheckpointer(native=InMemorySaver())
    deep_agent = checkpointer_for_deep_agent(platform)
    writes = [("__pregel_tasks", Send("tools", [{"id": "call-1"}]))]

    with attempt_context_scope(_attempt()):
        config = await platform.aput(
            {},
            empty_checkpoint(),
            {"source": "input", "step": -1},
            {},
        )
        with pytest.raises(UnsupportedDurableGraphError, match="Send"):
            await platform.aput_writes(config, writes, "task-1")
        await deep_agent.aput_writes(config, writes, "task-1")


def test_non_platform_checkpointer_is_unchanged() -> None:
    checkpointer = InMemorySaver()

    assert checkpointer_for_deep_agent(checkpointer) is checkpointer
    assert checkpointer_for_deep_agent(None) is None
