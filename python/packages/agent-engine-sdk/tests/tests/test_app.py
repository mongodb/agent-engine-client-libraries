"""Tests for agent_engine_sdk.app."""

from typing import Any

import pytest
from agent_engine_sdk.app import BaseApp
from agent_engine_sdk.models import ToolDefinition


class _ConcreteApp(BaseApp):
    """Minimal concrete implementation for testing."""

    def __init__(self) -> None:
        super().__init__(name="TestApp")
        self._tool_defs: list[ToolDefinition] = []

    def get_tool_definitions(self) -> list[ToolDefinition]:
        return self._tool_defs

    def tools(self) -> Any:
        return []

    def tool(self, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func

        return decorator

    def entrypoint(self, fn: Any) -> Any:
        return fn


class TestBaseApp:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            BaseApp(name="fail")  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self) -> None:
        app = _ConcreteApp()
        assert app.name == "TestApp"

    def test_wrapper_attributes_default_to_none(self) -> None:
        app = _ConcreteApp()
        assert app.tool_wrapper is None
        assert app.llm_wrapper is None

    def test_wrapper_injection(self) -> None:
        app = _ConcreteApp()
        app.tool_wrapper = "mock_tool_wrapper"
        app.llm_wrapper = "mock_llm_wrapper"
        assert app.tool_wrapper == "mock_tool_wrapper"
        assert app.llm_wrapper == "mock_llm_wrapper"

    def test_get_tool_definitions(self) -> None:
        app = _ConcreteApp()
        assert app.get_tool_definitions() == []
