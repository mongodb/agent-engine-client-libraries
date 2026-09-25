"""Tests for the App runtime."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_engine_runner_shared import RuntimeMode


def _make_app(mode: RuntimeMode = RuntimeMode.AER) -> Any:
    """Create an App with a mocked TenantRuntime."""
    with patch("agent_engine_sdk_adk.runtime.TenantRuntime") as MockRuntime:
        mock_rt = MagicMock()
        mock_rt.mode = mode
        MockRuntime.return_value = mock_rt
        from agent_engine_sdk_adk.runtime import App

        app = App(app_name="test-agent")
        app._runtime = mock_rt
        return app


class TestAppDeprecation:
    def test_deprecation_warning_on_org_id(self) -> None:
        """App(org_id=...) emits DeprecationWarning; the value is not forwarded."""
        with patch("agent_engine_sdk_adk.runtime.TenantRuntime") as MockRuntime:
            mock_rt = MagicMock()
            MockRuntime.return_value = mock_rt
            from agent_engine_sdk_adk.runtime import App

            with pytest.warns(DeprecationWarning, match="org_id"):
                App(app_name="test-agent", org_id="some_org")

        assert MockRuntime.call_args.kwargs.get("org_id") is None


class TestAppRunner:
    def test_get_agent_uses_the_apps_stable_adk_runner(self) -> None:
        from agent_engine_sdk_adk.runner import DurableADKRunner

        app = _make_app()

        @app.entrypoint
        def build_agent() -> MagicMock:
            return MagicMock()

        with patch.object(app, "_register_hooks"):
            first = app.get_agent()
            second = app.get_agent()

        assert isinstance(app.runner, DurableADKRunner)
        assert first.runner is app.runner
        assert second.runner is app.runner


class TestAppLlm:
    def test_llm_returns_secure_llm_in_aer_mode(self) -> None:
        from google.adk.models.google_llm import Gemini

        from agent_engine_sdk_adk.secure_llm import SecureLlm

        app = _make_app(mode=RuntimeMode.AER)

        with patch("agent_engine_sdk_adk.runtime.register_llm"):
            result = app.llm(Gemini(model="gemini-2.5-flash"))

        assert isinstance(result, SecureLlm)
        assert result.model == "gemini-2.5-flash"

    def test_llm_returns_raw_in_tool_mode(self) -> None:
        from google.adk.models.google_llm import Gemini

        from agent_engine_sdk_adk.secure_llm import SecureLlm

        app = _make_app(mode=RuntimeMode.TOOL)

        with patch("agent_engine_sdk_adk.runtime.register_llm"):
            result = app.llm(Gemini(model="gemini-2.5-flash"))

        assert not isinstance(result, SecureLlm)


class TestAppHooks:
    def test_register_hooks_registers_durable_adapter_llm_and_instrumentation(
        self,
    ) -> None:
        from agent_engine_sdk_adk.llm_adapter import ADKLLMAdapter
        from agent_engine_sdk_adk.runtime import App

        workflow_path = "agent_engine_runner_shared.hooks.register_workflow_adapter"
        factory_path = "agent_engine_runner_shared.hooks.register_llm_adapter_factory"
        instrumentor_path = "agent_engine_runner_shared.hooks.register_instrumentor"
        with (
            patch(workflow_path) as mock_workflow,
            patch(factory_path) as mock_llm_factory,
            patch(instrumentor_path) as mock_instrumentor,
            patch(
                "agent_engine_sdk_adk.runtime._adapter_version", return_value="2.3.4"
            ),
        ):
            App._register_hooks()
            mock_workflow.assert_called_once_with("google-adk", "2.3.4")
            mock_llm_factory.assert_called_once_with(ADKLLMAdapter)
            mock_instrumentor.assert_called_once()


class TestAppTools:
    def test_aer_tools_are_secure_wrapped_callables(self) -> None:
        app = _make_app(mode=RuntimeMode.AER)

        @app.tool()
        def my_tool(x: str) -> str:
            """A regular tool."""
            return x

        mock_secure = patch(
            "agent_engine_sdk_adk.runtime.create_secure_tool_function",
            side_effect=lambda **kw: kw["original_tool"],
        )
        with mock_secure:
            tools = app.tools()

        assert len(tools) == 1
        assert callable(tools[0])
        assert tools[0].__name__ == "my_tool"

    def test_tools_forwards_redact_fields_from_metadata(self) -> None:
        app = _make_app(mode=RuntimeMode.AER)

        registered_metadata: dict[str, dict] = {}
        app._runtime.register_tool.side_effect = lambda name, func, metadata: (
            registered_metadata.__setitem__(name, metadata)
        )
        app._runtime.get_tool_metadata.side_effect = lambda name: (
            registered_metadata.get(name, {})
        )

        @app.tool(redact_fields=["card_number"])
        def charge_customer(card_number: str) -> str:
            """Charge a customer."""
            return "ok"

        with patch(
            "agent_engine_sdk_adk.runtime.create_secure_tool_function",
            side_effect=lambda **kw: kw["original_tool"],
        ) as mock_secure:
            app.tools()

        mock_secure.assert_called_once()
        assert mock_secure.call_args.kwargs["redact_fields"] == ["card_number"]

    def test_in_pod_request_confirmation_fails_closed(self) -> None:
        app = _make_app(mode=RuntimeMode.TOOL)

        @app.tool()
        def remote_tool(x: str, tool_context: Any = None) -> str:
            """A remote tool."""
            assert tool_context is not None
            tool_context.request_confirmation(hint="approve?")
            return x

        tools = app.tools()

        class _Context:
            def request_confirmation(self, hint: str) -> None:
                raise AssertionError("native request_confirmation must not run")

        with pytest.raises(RuntimeError, match="in-pod"):
            tools[0]("value", tool_context=_Context())
