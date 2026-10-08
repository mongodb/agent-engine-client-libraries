"""LangChain SDK for MongoDB Atlas Agent Engine.

``AgentEngineToolSandboxBackend`` (deprecated: ``AgentEngineToolPodBackend``) and
``create_agent_engine_deep_agent`` are intentionally NOT re-exported at the
package root. They require the optional ``deepagents`` extra and importing
them eagerly from ``__init__.py`` would break any downstream package that
depends on ``agent-engine-sdk-langgraph`` without installing that extra.
Import them directly when needed::

    from agent_engine_sdk_langgraph.deep_agent import create_agent_engine_deep_agent
    from agent_engine_sdk_langgraph.backends.tool_sandbox import AgentEngineToolSandboxBackend

Or, for the common case, use ``App.deep_agent()`` which lazy-imports both.
"""

from .agent import LangGraphBaseAgent
from .durable_errors import UnsupportedDurableGraphError
from .output_parser import LangGraphOutputParser, RawStreamItem
from .platform_checkpointer import PlatformCheckpointer
from .runtime import App
from agent_engine_sdk_memory import Memory
from agent_engine_runner_shared import SessionFinishStatus

# Memory is re-exported from agent-engine-sdk-memory for a stable import path.
# ``agent_engine_sdk_langgraph.Memory is agent_engine_sdk_memory.Memory`` (same object).
# For app-bound ambient identity, use App.memory; constructing
# Memory(service_account_token=...) / Memory(base_url=...) is the HTTP/direct path.

__all__ = [
    "App",
    "LangGraphBaseAgent",
    "LangGraphOutputParser",
    "Memory",
    "PlatformCheckpointer",
    "RawStreamItem",
    "SessionFinishStatus",
    "UnsupportedDurableGraphError",
]
