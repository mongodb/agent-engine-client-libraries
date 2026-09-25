"""Checkpointer policy for graphs created by Deep Agent.

LangChain's agent factory routes pending ToolCalls through LangGraph ``Send``
packets. Atlas Agent Engine permits that adapter-owned routing without enabling ``Send``
for application-authored graphs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agent_engine_sdk_langgraph.platform_checkpointer import PlatformCheckpointer

__all__ = ["checkpointer_for_deep_agent"]


class _DeepAgentPlatformCheckpointer(PlatformCheckpointer):
    """Share platform state while accepting Deep Agent's internal routing."""

    def __init__(self, platform: PlatformCheckpointer) -> None:
        super().__init__(native=platform.native, scratch=platform._scratch)

    def _reject_unsupported_send(self, writes: Sequence[tuple[str, Any]]) -> None:
        # The adapter owns this graph's topology. Do not inspect or expose
        # Deep Agent's framework-private Send packet shape as a public contract.
        pass


def checkpointer_for_deep_agent(checkpointer: Any) -> Any:
    """Return the internal-routing view for a platform checkpointer."""
    if isinstance(checkpointer, PlatformCheckpointer):
        return _DeepAgentPlatformCheckpointer(checkpointer)
    return checkpointer
