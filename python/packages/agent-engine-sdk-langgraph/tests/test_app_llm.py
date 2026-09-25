"""Tests for App.llm() — LLM wrapping and creation."""

from unittest.mock import MagicMock, patch

import pytest

from agent_engine_sdk_langgraph import App
from agent_engine_runner_shared import RuntimeMode


@pytest.fixture(autouse=True)
def _reset_hooks():
    from agent_engine_runner_shared import hooks

    hooks.reset_hooks()
    # app.llm() is only callable inside the entrypoint's dynamic extent; these
    # tests exercise App.llm() directly (not via get_agent()), so simulate
    # being inside the entrypoint for the duration of each test.
    with hooks.entrypoint_scope():
        yield
    hooks.reset_hooks()


class TestAppLlmAERMode:
    """Tests for App.llm() in AER mode."""

    @patch("agent_engine_sdk_langgraph.runtime.SecureWrappedLLM")
    @patch("agent_engine_sdk_langgraph.runtime.get_current_wrapper")
    def test_unnamed_llm_uses_default_sentinel(self, mock_get_wrapper, mock_secure_cls):
        """Unnamed app.llm() registers under '__default__' and forwards that id."""
        app = App(app_name="Test")
        app._runtime.mode = RuntimeMode.AER
        mock_llm = MagicMock()
        mock_secure_cls.return_value = MagicMock()

        app.llm(mock_llm)

        mock_secure_cls.assert_called_once_with(
            llm=mock_llm, get_wrapper=mock_get_wrapper, llm_id="__default__"
        )

    @patch("agent_engine_sdk_langgraph.runtime.SecureWrappedLLM")
    @patch("agent_engine_sdk_langgraph.runtime.get_current_wrapper")
    def test_named_llm_passes_id_to_secure_wrapped_llm(
        self, mock_get_wrapper, mock_secure_cls
    ):
        """A named call forwards llm_id to SecureWrappedLLM."""
        app = App(app_name="Test")
        app._runtime.mode = RuntimeMode.AER
        mock_llm = MagicMock()
        mock_secure_cls.return_value = MagicMock()

        app.llm(mock_llm, llm_id="primary")

        mock_secure_cls.assert_called_once_with(
            llm=mock_llm, get_wrapper=mock_get_wrapper, llm_id="primary"
        )


class TestAppLlmRegistration:
    """Tests for App.llm() registration into the named-LLM registry."""

    def _make_tool_app(self):
        app = App(app_name="Test")
        app._runtime.mode = RuntimeMode.TOOL
        return app

    def test_unnamed_llm_registered_under_default_sentinel(self):
        from agent_engine_runner_shared.hooks import get_named_llm

        app = self._make_tool_app()
        mock_llm = MagicMock()

        app.llm(mock_llm)

        assert get_named_llm("__default__") is mock_llm

    def test_named_llm_registered_in_registry(self):
        from agent_engine_runner_shared.hooks import get_named_llm

        app = self._make_tool_app()
        llm_a, llm_b = MagicMock(), MagicMock()

        app.llm(llm_a, llm_id="model-a")
        app.llm(llm_b, llm_id="model-b")

        assert get_named_llm("model-a") is llm_a
        assert get_named_llm("model-b") is llm_b

    def test_second_unnamed_llm_raises(self):
        """Two unnamed app.llm() calls collide on the '__default__' id."""
        app = self._make_tool_app()
        app.llm(MagicMock())

        with pytest.raises(ValueError, match="already registered"):
            app.llm(MagicMock())

    def test_duplicate_named_id_raises(self):
        app = self._make_tool_app()
        app.llm(MagicMock(), llm_id="primary")

        with pytest.raises(ValueError, match="already registered"):
            app.llm(MagicMock(), llm_id="primary")

    def test_tool_mode_returns_llm_unwrapped(self):
        app = self._make_tool_app()
        mock_llm = MagicMock()
        result = app.llm(mock_llm, llm_id="primary")
        assert result is mock_llm

    def test_unnamed_then_named_succeeds(self):
        """An unnamed call followed by a named call is allowed."""
        from agent_engine_runner_shared.hooks import get_named_llm

        app = self._make_tool_app()
        unnamed_llm = MagicMock()
        named_llm = MagicMock()

        app.llm(unnamed_llm)
        app.llm(named_llm, llm_id="primary")

        assert get_named_llm("__default__") is unnamed_llm
        assert get_named_llm("primary") is named_llm
