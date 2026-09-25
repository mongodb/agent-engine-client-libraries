"""Integration tests for the TenantRuntime.

Note: tool registration via @app.tool() now lives in App (agent-engine-sdk-langgraph).
TenantRuntime exposes register_tool() for raw function registration.
"""

from agent_engine_runner_shared import TenantRuntime


def test_runtime_creation():
    """Test basic runtime creation."""
    app = TenantRuntime(
        app_name="Test Agent",
        app_version="1.0.0",
    )

    assert app.app_name == "Test Agent"
    assert app.app_version == "1.0.0"


def test_runtime_register_tool():
    """Test register_tool() stores raw function."""
    runtime = TenantRuntime()

    def my_tool(x: int) -> int:
        """Multiply by 2."""
        return x * 2

    runtime.register_tool("my_tool", my_tool, {})
    assert "my_tool" in runtime._tools
    assert runtime._tools["my_tool"] is my_tool
    assert my_tool(5) == 10


def test_runtime_register_tool_with_metadata():
    """Test register_tool() with metadata."""
    runtime = TenantRuntime()

    def secure_operation(data: str) -> str:
        """Convert to uppercase."""
        return data.upper()

    runtime.register_tool("secure_operation", secure_operation, {"network": ["api.example.com"]})
    assert "secure_operation" in runtime._tools
    assert runtime._tool_definitions["secure_operation"]["network"] == ["api.example.com"]
    assert secure_operation("hello") == "HELLO"


def test_runtime_register_multiple_tools():
    """Test registering multiple tools."""
    runtime = TenantRuntime()

    def tool_one(x: int) -> int:
        """Add 1."""
        return x + 1

    def tool_two(x: int) -> int:
        """Multiply by 2."""
        return x * 2

    def tool_three(x: int) -> int:
        """Subtract 1."""
        return x - 1

    runtime.register_tool("tool_one", tool_one, {})
    runtime.register_tool("tool_two", tool_two, {})
    runtime.register_tool("tool_three", tool_three, {"timeout": 60})

    assert len(runtime._tools) == 3
    assert "tool_one" in runtime._tools
    assert "tool_two" in runtime._tools
    assert "tool_three" in runtime._tools

    assert tool_one(5) == 6
    assert tool_two(5) == 10
    assert tool_three(5) == 4
