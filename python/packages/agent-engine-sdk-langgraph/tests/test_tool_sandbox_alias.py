"""Test that AgentEngineToolSandboxBackend aliases AgentEngineToolPodBackend."""


class TestToolSandboxAlias:
    """Canonical name resolves to the same implementation object."""

    def test_alias_is_same_class(self):
        from agent_engine_sdk_langgraph.backends.toolpod import (
            AgentEngineToolPodBackend,
        )
        from agent_engine_sdk_langgraph.backends.tool_sandbox import (
            AgentEngineToolSandboxBackend,
        )

        assert AgentEngineToolSandboxBackend is AgentEngineToolPodBackend

    def test_alias_id_property(self):
        from agent_engine_sdk_langgraph.backends.tool_sandbox import (
            AgentEngineToolSandboxBackend,
        )

        backend = AgentEngineToolSandboxBackend()
        assert backend.id == "agent-engine-toolpod"

    def test_legacy_still_works(self):
        """The legacy import path must remain functional."""
        from agent_engine_sdk_langgraph.backends.toolpod import (
            AgentEngineToolPodBackend,
        )

        backend = AgentEngineToolPodBackend()
        assert backend.id == "agent-engine-toolpod"

    def test_canonical_module_export_list(self):
        """tool_sandbox module exports only AgentEngineToolSandboxBackend."""
        import agent_engine_sdk_langgraph.backends.tool_sandbox as ts

        assert hasattr(ts, "AgentEngineToolSandboxBackend")
        assert "AgentEngineToolSandboxBackend" in ts.__all__

    def test_backend_imports_without_deepagents(self):
        """The backends package must import without deepagents installed."""
        # Verify that importing backends (not the submodule) doesn't fail.
        import agent_engine_sdk_langgraph.backends

        assert hasattr(agent_engine_sdk_langgraph.backends, "__all__")
