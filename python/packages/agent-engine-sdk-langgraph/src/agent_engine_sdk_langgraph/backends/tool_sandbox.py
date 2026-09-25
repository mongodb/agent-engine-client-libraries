"""AgentEngineToolSandboxBackend — canonical name for AgentEngineToolPodBackend.

This module exists so agent authors can use the stable sandbox terminology
without depending on the legacy AER/ToolPod names. The implementation is
identical; both names refer to the same class.

.. code-block:: python

    from agent_engine_sdk_langgraph.backends.tool_sandbox import AgentEngineToolSandboxBackend
"""

from __future__ import annotations

from agent_engine_sdk_langgraph.backends.toolpod import AgentEngineToolPodBackend

AgentEngineToolSandboxBackend = AgentEngineToolPodBackend
"""Canonical name for AgentEngineToolPodBackend.

``AgentEngineToolSandboxBackend is AgentEngineToolPodBackend`` — both names
resolve to the same class object. Use the canonical name in new code.
"""

__all__ = ["AgentEngineToolSandboxBackend"]
