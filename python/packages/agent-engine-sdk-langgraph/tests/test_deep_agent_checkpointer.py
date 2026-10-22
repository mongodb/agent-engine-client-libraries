"""Deep Agent's durable checkpointer view."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from agent_engine_sdk_langgraph.deep_agent_checkpointer import (
    DeepAgentPlatformCheckpointer,
    checkpointer_for_deep_agent,
)
from agent_engine_sdk_langgraph.platform_checkpointer import PlatformCheckpointer


def test_deep_agent_view_shares_platform_state() -> None:
    platform = PlatformCheckpointer(native=InMemorySaver())
    deep_agent = checkpointer_for_deep_agent(platform)

    assert isinstance(deep_agent, DeepAgentPlatformCheckpointer)
    assert deep_agent.native is platform.native
    assert deep_agent._scratch is platform._scratch


def test_non_platform_checkpointer_is_unchanged() -> None:
    checkpointer = InMemorySaver()

    assert checkpointer_for_deep_agent(checkpointer) is checkpointer
    assert checkpointer_for_deep_agent(None) is None
