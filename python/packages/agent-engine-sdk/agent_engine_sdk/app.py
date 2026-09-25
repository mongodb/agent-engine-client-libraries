"""Base application class for framework-specific SDKs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent_engine_sdk.models import ToolDefinition


class BaseApp(ABC):
    """Abstract base class for framework-specific SDK integrations.

    Each framework SDK (LangGraph, CrewAI, etc.) subclasses ``BaseApp``
    and implements the abstract methods using framework-native constructs.

    The platform runtime sets ``tool_wrapper`` and ``llm_wrapper`` on the
    instance before calling ``entrypoint()`` so that tools and LLMs are
    routed through the secure execution layer.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.tool_wrapper: Any = None
        self.llm_wrapper: Any = None

    @abstractmethod
    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Return all registered tool definitions.

        Used by the OE to obtain tool execution information.
        """
        ...

    @abstractmethod
    def tools(self) -> Any:
        """Return a list of wrapped, framework-specific tools."""
        ...

    @abstractmethod
    def tool(self, *args: Any, **kwargs: Any) -> Any:
        """Decorator that registers a tool on this app."""
        ...

    @abstractmethod
    def entrypoint(self, fn: Any) -> Any:
        """Decorator that registers the agent builder function.

        Called by runtimes to obtain a ``BaseAgent``.
        """
        ...
