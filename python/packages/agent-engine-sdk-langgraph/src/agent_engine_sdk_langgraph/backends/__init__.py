"""Backend implementations for Atlas Agent Engine deep agent integration.

``AgentEngineToolPodBackend`` (canonical name: ``AgentEngineToolSandboxBackend``)
is intentionally NOT imported eagerly from this package. It lives in
``.toolpod`` (or ``.tool_sandbox`` for the canonical name) which imports
from ``deepagents``, and ``deepagents`` is an optional dependency — any
code path that doesn't use deep agents must be able to ``import
agent_engine_sdk_langgraph.backends`` without pulling ``deepagents`` in.

Import it explicitly when you need it:

    from agent_engine_sdk_langgraph.backends.tool_sandbox import AgentEngineToolSandboxBackend

``AgentEngineToolPodBackend`` remains available as a deprecated alias::

    from agent_engine_sdk_langgraph.backends.toolpod import AgentEngineToolPodBackend
"""

__all__: list[str] = []
