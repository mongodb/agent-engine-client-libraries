"""Tests for asynchronous tool registration via TenantRuntime.

Note: The @app.tool() decorator now lives in App (agent-engine-sdk-langgraph).
TenantRuntime exposes register_tool() for raw function registration.
"""

import pytest

from agent_engine_runner_shared import TenantRuntime


@pytest.mark.asyncio
async def test_async_tool_register():
    """Test registering an async function via register_tool()."""
    runtime = TenantRuntime()

    async def async_add(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    runtime.register_tool("async_add", async_add, {})
    assert "async_add" in runtime._tools
    result = await async_add(2, 3)
    assert result == 5


@pytest.mark.asyncio
async def test_async_tool_register_with_metadata():
    """Test registering an async function with tool metadata."""
    runtime = TenantRuntime()

    async def async_summarize(text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()

    runtime.register_tool("async_summarize", async_summarize, metadata={"timeout_seconds": 120})
    assert "async_summarize" in runtime._tools
    result = await async_summarize("hello")
    assert result == "HELLO"
