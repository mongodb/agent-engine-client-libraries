"""Checkpointer view for graphs created by Deep Agent.

LangChain's agent factory routes pending ToolCalls through LangGraph ``Send``
packets. Deep Agent's recorded activity paths predate Send task segments, so
the runtime recognizes this view and keeps those tasks on their parent path.
"""

from __future__ import annotations

from typing import Any

from agent_engine_sdk_langgraph.platform_checkpointer import PlatformCheckpointer

__all__ = ["DeepAgentPlatformCheckpointer", "checkpointer_for_deep_agent"]


class DeepAgentPlatformCheckpointer(PlatformCheckpointer):
    """Share platform state while marking the graph as Deep Agent-owned."""

    def __init__(self, platform: PlatformCheckpointer) -> None:
        super().__init__(native=platform.native, scratch=platform._scratch)


def checkpointer_for_deep_agent(checkpointer: Any) -> Any:
    """Return the Deep Agent view for a platform checkpointer."""
    if isinstance(checkpointer, PlatformCheckpointer):
        return DeepAgentPlatformCheckpointer(checkpointer)
    return checkpointer
