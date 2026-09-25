"""Tests for TenantRuntime.register_tool() — raw tool registration."""

from agent_engine_runner_shared import TenantRuntime


class TestRegisterTool:
    """Tests for register_tool storing raw functions and metadata."""

    def test_stores_raw_function(self):
        """register_tool stores the function in _tools by name."""
        runtime = TenantRuntime(app_name="Test")

        def my_func(x: str) -> str:
            return x

        runtime.register_tool(
            name="my_func",
            func=my_func,
            metadata={"is_local": True},
        )

        assert runtime._tools["my_func"] is my_func

    def test_stores_metadata(self):
        """register_tool stores the metadata dict in _tool_definitions."""
        runtime = TenantRuntime(app_name="Test")

        def my_func(x: str) -> str:
            return x

        metadata = {
            "name": "my_func",
            "is_local": True,
            "network": ["api.openai.com"],
            "timeout_seconds": 60,
        }
        runtime.register_tool(name="my_func", func=my_func, metadata=metadata)

        assert runtime._tool_definitions["my_func"] == metadata

    def test_does_not_create_lc_tools(self):
        """register_tool does not store anything in _lc_tools."""
        runtime = TenantRuntime(app_name="Test")

        def my_func(x: str) -> str:
            return x

        runtime.register_tool(name="my_func", func=my_func, metadata={})

        assert not hasattr(runtime, "_lc_tools") or "my_func" not in getattr(
            runtime, "_lc_tools", {}
        )
