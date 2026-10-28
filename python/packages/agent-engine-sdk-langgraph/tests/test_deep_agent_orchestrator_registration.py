"""``App.deep_agent()`` and an already-registered orchestrator model.

An agent that opts into per-execution builds has to register every ``llm_id``
before it reads invocation context, because the Tool Pod's context-free
discovery build only retains what was registered before the builder stopped.
``deep_agent()`` registers the orchestrator itself under ``__default__``, so an
author who has already registered it must not hit a duplicate id.

Registering is not the whole contract. The model still has to reach the graph
wrapped, or its calls go straight to the provider and bypass OE approvals,
auditing, and guardrails. ``app.llm()`` registers the model it is given and
returns a *different* object, so an author who keeps its own raw instance and
passes that on is a supported shape and is covered here.

These drive the public ``App.deep_agent()`` and inspect the model it hands to
the deep-agent factory, rather than calling the private resolver directly.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_engine_runner_shared import RuntimeMode
from agent_engine_runner_shared.hooks import (
    entrypoint_scope,
    reset_hooks,
    snapshot_llm_registry,
)
from agent_engine_sdk_langgraph import App


class _FakeSecureWrappedLLM:
    """Stand-in for ``SecureWrappedLLM`` that records what it was built from.

    Substituting the class keeps ``isinstance`` meaningful inside the resolver:
    a fake instance is recognised as already wrapped, a raw model is not.
    """

    def __init__(self, *, llm: Any, get_wrapper: Any, llm_id: str) -> None:
        self.llm = llm
        self.get_wrapper = get_wrapper
        self.llm_id = llm_id


@pytest.fixture(autouse=True)
def _fake_wrapper(monkeypatch):
    monkeypatch.setattr(
        "agent_engine_sdk_langgraph.runtime.SecureWrappedLLM",
        _FakeSecureWrappedLLM,
    )


@pytest.fixture(autouse=True)
def _reset_hooks():
    # The registry is process-global, so a second test would otherwise hit
    # "llm_id already registered".
    reset_hooks()
    yield
    reset_hooks()


def _app(mode: RuntimeMode) -> App:
    app = App(app_name="Test Agent")
    mock_runtime = MagicMock()
    mock_runtime.mode = mode
    mock_runtime.agent_config.path = None
    mock_runtime.agent_config.feature_enabled.return_value = True
    app._runtime = mock_runtime
    return app


def _register_through_app_llm(app: App, raw: Any) -> Any:
    """Register ``raw`` the way an author would, returning ``App.llm``'s result."""
    with entrypoint_scope():
        return app.llm(raw)


def _secure_llm_forwarded_to_factory(app: App, llm: Any) -> Any:
    """Call ``App.deep_agent(llm)`` and return the model it forwarded."""
    create = MagicMock(return_value=MagicMock())
    deep_agent_mod = MagicMock()
    deep_agent_mod.create_agent_engine_deep_agent = create
    toolpod_mod = MagicMock()
    toolpod_mod.AgentEngineToolPodBackend = MagicMock(return_value=MagicMock())

    with (
        patch.dict(
            sys.modules,
            {
                "agent_engine_sdk_langgraph.deep_agent": deep_agent_mod,
                "agent_engine_sdk_langgraph.backends.toolpod": toolpod_mod,
            },
        ),
        entrypoint_scope(),
    ):
        app.deep_agent(llm, backend=MagicMock(), checkpointer=None)

    return create.call_args.kwargs["secure_llm"]


def test_registers_the_orchestrator_when_the_author_has_not() -> None:
    app = _app(RuntimeMode.AER)
    raw = MagicMock()

    secure = _secure_llm_forwarded_to_factory(app, raw)

    assert isinstance(secure, _FakeSecureWrappedLLM)
    assert secure.llm is raw
    assert secure.llm_id == "__default__"
    assert snapshot_llm_registry()["__default__"] is raw


def test_wraps_a_raw_model_the_author_already_registered() -> None:
    app = _app(RuntimeMode.AER)
    raw = MagicMock()
    _register_through_app_llm(app, raw)

    secure = _secure_llm_forwarded_to_factory(app, raw)

    assert isinstance(secure, _FakeSecureWrappedLLM)
    assert secure.llm is raw
    assert secure.llm_id == "__default__"
    # No second registration: the registry still holds the original model.
    assert snapshot_llm_registry()["__default__"] is raw


def test_reuses_an_already_wrapped_model() -> None:
    app = _app(RuntimeMode.AER)
    raw = MagicMock()
    wrapped = _register_through_app_llm(app, raw)

    secure = _secure_llm_forwarded_to_factory(app, wrapped)

    assert secure is wrapped
    assert snapshot_llm_registry()["__default__"] is raw


def test_tool_mode_hands_back_the_raw_model() -> None:
    app = _app(RuntimeMode.TOOL)
    raw = MagicMock()
    _register_through_app_llm(app, raw)

    secure = _secure_llm_forwarded_to_factory(app, raw)

    assert secure is raw
    assert snapshot_llm_registry()["__default__"] is raw
