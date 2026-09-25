"""Tests for synchronous tool registration via TenantRuntime.

Note: The @app.tool() decorator now lives in App (agent-engine-sdk-langgraph).
TenantRuntime exposes register_tool() for raw function registration.
"""

from agent_engine_runner_shared import TenantRuntime


def test_sync_tool_register():
    """Test registering a sync function via register_tool()."""
    runtime = TenantRuntime()

    def add(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    runtime.register_tool("add", add, {})
    assert "add" in runtime._tools
    result = add(2, 3)
    assert result == 5


def test_sync_tool_register_with_metadata():
    """Test registering a sync function with tool metadata."""
    runtime = TenantRuntime()

    def summarize(text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()

    runtime.register_tool("summarize", summarize, metadata={"timeout_seconds": 120})
    assert "summarize" in runtime._tools
    result = summarize("hello")
    assert result == "HELLO"


def test_sync_tool_register_with_redact_metadata():
    """Test register_tool() with redact_fields metadata."""
    runtime = TenantRuntime()

    def login(username: str, password: str) -> bool:
        """Authenticate user."""
        return username == "admin" and password == "secret"

    runtime.register_tool("login", login, metadata={"redact_fields": ["password"]})
    assert "login" in runtime._tools
    assert login("admin", "secret") is True
