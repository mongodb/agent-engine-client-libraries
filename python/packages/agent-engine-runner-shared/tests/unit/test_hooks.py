"""Tests for agent_engine_runner_shared.hooks — framework hook registry."""

import pytest

import agent_engine_runner_shared.hooks as hooks


@pytest.fixture(autouse=True)
def _reset_hooks():
    """Reset all hooks to None between tests."""
    yield
    hooks.reset_hooks()


def _stub_suspend(payload):
    return {"decision": "approved"}


def _stub_adapter(llm, tools=None):
    return "fake-adapter"


def _stub_instrumentor():
    pass


class TestSuspendHandler:
    def test_get_raises_when_unregistered(self):
        assert hooks.get_suspend_handler() is None

    def test_register_and_get(self):
        hooks.register_suspend_handler(_stub_suspend)
        assert hooks.get_suspend_handler() is _stub_suspend


class TestLLMAdapterFactory:
    def test_get_raises_when_unregistered(self):
        with pytest.raises(RuntimeError, match="No LLM adapter factory registered"):
            hooks.get_llm_adapter_factory()

    def test_register_and_get(self):
        hooks.register_llm_adapter_factory(_stub_adapter)
        assert hooks.get_llm_adapter_factory() is _stub_adapter


class TestInstrumentor:
    def test_get_returns_none_when_unregistered(self):
        assert hooks.get_instrumentor() is None

    def test_register_and_get(self):
        hooks.register_instrumentor(_stub_instrumentor)
        assert hooks.get_instrumentor() is _stub_instrumentor


class TestResetHooks:
    def test_reset_clears_all(self):
        hooks.register_suspend_handler(_stub_suspend)
        hooks.register_llm_adapter_factory(_stub_adapter)
        hooks.register_instrumentor(_stub_instrumentor)
        with hooks.entrypoint_scope():
            hooks.register_llm("primary", object())

        hooks.reset_hooks()

        assert hooks.get_suspend_handler() is None
        with pytest.raises(RuntimeError):
            hooks.get_llm_adapter_factory()
        assert hooks.get_instrumentor() is None
        assert hooks.has_named_llms() is False


class TestLLMRegistry:
    def test_get_raises_when_not_registered(self):
        with pytest.raises(KeyError, match="llm_id 'missing' not registered"):
            hooks.get_named_llm("missing")

    def test_register_and_get(self):
        fake_llm = object()
        with hooks.entrypoint_scope():
            hooks.register_llm("primary", fake_llm)
        assert hooks.get_named_llm("primary") is fake_llm

    def test_has_named_llms_false_when_empty(self):
        assert hooks.has_named_llms() is False

    def test_has_named_llms_true_after_register(self):
        with hooks.entrypoint_scope():
            hooks.register_llm("primary", object())
        assert hooks.has_named_llms() is True

    def test_register_multiple_ids(self):
        llm_a, llm_b = object(), object()
        with hooks.entrypoint_scope():
            hooks.register_llm("a", llm_a)
            hooks.register_llm("b", llm_b)
        assert hooks.get_named_llm("a") is llm_a
        assert hooks.get_named_llm("b") is llm_b

    def test_register_duplicate_id_raises(self):
        with hooks.entrypoint_scope():
            hooks.register_llm("primary", object())
            with pytest.raises(ValueError, match="already registered"):
                hooks.register_llm("primary", object())

    def test_reset_clears_registry(self):
        with hooks.entrypoint_scope():
            hooks.register_llm("x", object())
        hooks.reset_hooks()
        assert hooks.has_named_llms() is False
        with pytest.raises(KeyError):
            hooks.get_named_llm("x")


class TestEntrypointScopeEnforcement:
    def test_register_llm_outside_entrypoint_raises(self):
        with pytest.raises(RuntimeError, match="must be called inside"):
            hooks.register_llm("primary", object())

    def test_register_llm_inside_entrypoint_scope_succeeds(self):
        with hooks.entrypoint_scope():
            hooks.register_llm("primary", object())
        assert hooks.get_named_llm("primary") is not None

    def test_entrypoint_scope_restores_previous_state_on_exit(self):
        with hooks.entrypoint_scope():
            with hooks.entrypoint_scope():
                hooks.register_llm("nested", object())
            # Still inside the outer scope: should not raise.
            hooks.register_llm("outer", object())
        with pytest.raises(RuntimeError, match="must be called inside"):
            hooks.register_llm("after", object())

    def test_entrypoint_scope_restores_state_on_exception(self):
        with pytest.raises(ValueError):
            with hooks.entrypoint_scope():
                raise ValueError("boom")
        with pytest.raises(RuntimeError, match="must be called inside"):
            hooks.register_llm("primary", object())
