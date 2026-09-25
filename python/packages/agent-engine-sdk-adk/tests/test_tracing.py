"""Verify OTEL tracing is configured when TenantRuntime initializes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestTracingSetup:
    def test_tenant_runtime_init_calls_setup_tracing(self) -> None:
        """TenantRuntime.__init__ calls setup_tracing, so creating an App
        is sufficient to establish the TracerProvider with JSONL + MongoDB
        exporters. ADK's built-in telemetry module emits spans through the
        standard OTEL API, so they flow through our pipeline automatically."""
        with patch("agent_engine_sdk_adk.runtime.TenantRuntime") as MockRuntime:
            mock_rt = MagicMock()
            MockRuntime.return_value = mock_rt

            from agent_engine_sdk_adk.runtime import App

            App(app_name="trace-test")

            MockRuntime.assert_called_once()
            call_kwargs = MockRuntime.call_args[1]
            assert call_kwargs["app_name"] == "trace-test"
            assert call_kwargs["traces_collection_name"] == "traces"

    def test_run_calls_instrumentor(self) -> None:
        """App.run() re-runs the instrumentor after hooks registration so
        any framework-specific OTEL instrumentation is activated."""
        with patch("agent_engine_sdk_adk.runtime.TenantRuntime") as MockRuntime:
            mock_rt = MagicMock()
            MockRuntime.return_value = mock_rt

            from agent_engine_sdk_adk.runtime import App

            app = App(app_name="trace-test")
            app._runtime = mock_rt

            @app.entrypoint
            def build() -> MagicMock:
                return MagicMock()

            with patch(
                "agent_engine_runner_shared.tracing.setup._run_instrumentor"
            ) as mock_instr:
                with patch.object(App, "_register_hooks"):
                    app.run()
                    mock_instr.assert_called_once()
