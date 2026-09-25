"""Tenant-registered LangGraph adapter hooks.

Kept out of ``agent.py`` so ``runtime.py`` can type-check against these
callables without importing the adapter (which imports the runtime).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.types import Command
from agent_engine_sdk import AgentInput, RequestContext

# A value LangGraph can ingest directly as the first arg to ``ainvoke``/
# ``astream``: a graph state dict or a ``Command``.
GraphInput = dict[str, Any] | Command
# Hook that maps a fresh invocation to graph input. Registered via
# ``@app.prepare_agent_input``; replaces the default message-wrapping when set.
PrepareAgentInput = Callable[[AgentInput, RequestContext], GraphInput]
# Hook that maps RequestContext to the LangGraph checkpoint thread_id.
# Registered via ``@app.resolve_thread_id``; replaces workspace-scoped
# session derivation when set. Return value is used verbatim.
ResolveThreadId = Callable[[RequestContext], str]

__all__ = [
    "GraphInput",
    "PrepareAgentInput",
    "ResolveThreadId",
]
