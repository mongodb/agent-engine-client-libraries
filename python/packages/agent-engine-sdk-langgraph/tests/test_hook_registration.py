"""Tests that App.run() registers all framework hooks with agent-engine-runner-shared."""

from unittest.mock import Mock, patch

import pytest

from agent_engine_runner_shared import hooks


@pytest.fixture(autouse=True)
def _reset_hooks():
    yield
    hooks.reset_hooks()


class TestAppRunRegistersHooks:
    def _make_app(self):
        from agent_engine_sdk_langgraph.runtime import App

        app = App(app_name="test-agent")

        @app.entrypoint
        def build_graph():
            return Mock()

        return app

    @patch("agent_engine_runner_shared.runtime.TenantRuntime.register_and_run")
    def test_registers_suspend_handler(self, mock_run):
        app = self._make_app()
        app.run()
        handler = hooks.get_suspend_handler()
        assert handler is not None

    @patch("agent_engine_runner_shared.runtime.TenantRuntime.register_and_run")
    def test_registers_durable_activity_suspend_handler(self, mock_run):
        app = self._make_app()
        app.run()
        handler = hooks.get_durable_activity_suspend_handler()
        assert handler is not None

    @patch("agent_engine_runner_shared.runtime.TenantRuntime.register_and_run")
    def test_registers_llm_adapter_factory(self, mock_run):
        app = self._make_app()
        app.run()
        factory = hooks.get_llm_adapter_factory()
        assert factory is not None

    @patch("agent_engine_runner_shared.runtime.TenantRuntime.register_and_run")
    def test_registers_instrumentor(self, mock_run):
        app = self._make_app()
        app.run()
        instrumentor = hooks.get_instrumentor()
        assert instrumentor is not None

    @patch("agent_engine_runner_shared.runtime.TenantRuntime.register_and_run")
    def test_all_hooks_registered_before_server_starts(self, mock_run):
        """All hooks must be registered before register_and_run is called."""
        registered = {}

        def capture_state(*args, **kwargs):
            registered["suspend"] = hooks.get_suspend_handler()
            registered["durable_activity_suspend"] = (
                hooks.get_durable_activity_suspend_handler()
            )
            registered["adapter_factory"] = hooks.get_llm_adapter_factory()
            registered["instrumentor"] = hooks.get_instrumentor()

        mock_run.side_effect = capture_state
        app = self._make_app()
        app.run()

        assert registered["suspend"] is not None
        assert registered["durable_activity_suspend"] is not None
        assert registered["adapter_factory"] is not None
        assert registered["instrumentor"] is not None

    @patch("agent_engine_runner_shared.runtime.TenantRuntime.register_and_run")
    @patch("agent_engine_runner_shared.tracing.setup._run_instrumentor")
    def test_instrumentor_runs_after_registration(self, mock_run_inst, mock_run):
        """_run_instrumentor must be called after hooks are registered,
        not only during TenantRuntime.__init__ (when hooks are still None)."""
        call_order = []

        def track_register(*args, **kwargs):
            call_order.append("register_and_run")

        def track_instrumentor():
            # Verify the hook is registered by the time this is called
            instrumentor = hooks.get_instrumentor()
            call_order.append(
                f"run_instrumentor:hook={'set' if instrumentor else 'none'}"
            )

        mock_run.side_effect = track_register
        mock_run_inst.side_effect = track_instrumentor

        app = self._make_app()
        app.run()

        # _run_instrumentor should be called at least once AFTER hooks are registered
        # (it may also be called during __init__ when hooks are None — that's fine,
        # as long as it's called again after registration)
        post_registration_calls = [
            c for c in call_order if c == "run_instrumentor:hook=set"
        ]
        assert len(post_registration_calls) >= 1, (
            f"_run_instrumentor was never called with hooks registered. "
            f"Call order: {call_order}"
        )
