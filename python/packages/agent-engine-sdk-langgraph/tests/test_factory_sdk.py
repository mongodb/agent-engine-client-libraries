"""Tests for the deep-agent factory and ``App.deep_agent()`` integration.

Covers:
- ``SecureWrappedLLM`` is passed as the model to ``create_deep_agent``.
- The compiled graph is wrapped in ``LangGraphBaseAgent`` via ``get_agent()``.
- ``AgentEngineToolPodBackend`` is passed as the backend to ``create_deep_agent``.
- ``SubAgent`` specs with string models are rejected with ``RuntimeError``.
"""

from typing import Any, TypedDict, cast
from unittest.mock import MagicMock, patch

import pytest
from deepagents import AsyncSubAgent, CompiledSubAgent, SubAgent
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_engine_sdk_langgraph.deep_agent import create_agent_engine_deep_agent
from agent_engine_sdk_langgraph.durable_deep_agent import DurableDeepAgentMiddleware
from agent_engine_sdk_langgraph.platform_checkpointer import PlatformCheckpointer
from agent_engine_sdk_langgraph.stopped_tool_call_middleware import (
    StoppedToolCallMiddleware,
)
from agent_engine_sdk_langgraph.runtime import App


def _specs(*specs: dict[str, Any]) -> list[SubAgent | CompiledSubAgent | AsyncSubAgent]:
    """Cast plain test dicts to the SubAgent union type pyright expects."""
    return cast(list[SubAgent | CompiledSubAgent | AsyncSubAgent], list(specs))


class _State(TypedDict, total=False):
    value: str


def _checkpointed_graph() -> CompiledStateGraph:
    builder = StateGraph(_State)
    builder.add_node("work", lambda state: state)
    builder.add_edge(START, "work")
    builder.add_edge("work", END)
    return builder.compile(checkpointer=InMemorySaver())


# Subagent-tree validation has its own focused test file
# (``test_subagents.py``) that drives ``validate_subagent_tree`` directly
# against fixture specs. This file keeps the integration paths through
# ``create_agent_engine_deep_agent`` and ``App.deep_agent``.

# ---------------------------------------------------------------------------
# create_agent_engine_deep_agent
# ---------------------------------------------------------------------------


class TestCreateAgentEngineDeepAgent:
    """Tests for create_agent_engine_deep_agent()."""

    @patch("agent_engine_sdk_langgraph.deep_agent.create_deep_agent")
    def test_calls_create_deep_agent_with_correct_args(self, mock_create):
        """Verify create_deep_agent is called with correct arguments."""
        secure_llm = MagicMock(spec=BaseChatModel)
        backend = MagicMock()
        tools = [MagicMock()]
        subagents = _specs(
            {
                "name": "helper",
                "description": "A helper",
                "system_prompt": "Help.",
                "model": MagicMock(spec=BaseChatModel),
            }
        )
        system_prompt = "You are an agent."
        middleware = ()

        create_agent_engine_deep_agent(
            secure_llm=secure_llm,
            backend=backend,
            tools=tools,
            subagents=subagents,
            system_prompt=system_prompt,
            middleware=middleware,
        )

        mock_create.assert_called_once()
        _, kwargs = mock_create.call_args
        assert kwargs["model"] is secure_llm
        assert kwargs["backend"] is backend
        assert kwargs["tools"] is tools
        assert kwargs["subagents"] == subagents
        assert kwargs["system_prompt"] == system_prompt
        assert kwargs["checkpointer"] is None
        assert kwargs["store"] is None
        assert kwargs["skills"] is None
        assert len(kwargs["middleware"]) == 2
        assert isinstance(kwargs["middleware"][0], StoppedToolCallMiddleware)
        assert isinstance(kwargs["middleware"][1], DurableDeepAgentMiddleware)

    @patch("agent_engine_sdk_langgraph.deep_agent.create_deep_agent")
    def test_passes_checkpointer_store_skills(self, mock_create, tmp_path):
        """Verify checkpointer, store, and skills are forwarded.

        ``skills=`` entries are pre-validated before being forwarded, so
        this test writes a well-formed SKILL.md to ``tmp_path`` and asserts
        the validated path is forwarded unchanged (valid skill pass-through).

        Passed straight through to ``create_deep_agent`` (not built into a
        middleware instance here) so deepagents' own ``skills=`` wiring for
        the auto-injected general-purpose subagent keeps working; tracing is
        added separately by patching ``SkillsMiddleware`` at import time
        (see ``skills_tracing.py``), which this mock bypasses entirely.
        """
        secure_llm = MagicMock(spec=BaseChatModel)
        backend = MagicMock()
        mock_checkpointer = MagicMock()
        mock_store = MagicMock()

        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: test skill\n---\n\n# body\n",
            encoding="utf-8",
        )
        mock_skills = [str(skill_dir)]

        create_agent_engine_deep_agent(
            secure_llm=secure_llm,
            backend=backend,
            checkpointer=mock_checkpointer,
            store=mock_store,
            skills=mock_skills,
        )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["checkpointer"] is mock_checkpointer
        assert call_kwargs["store"] is mock_store
        assert call_kwargs["skills"] == mock_skills

    @patch("agent_engine_sdk_langgraph.deep_agent.create_deep_agent")
    def test_scopes_platform_checkpointer_to_deep_agent(self, mock_create) -> None:
        platform = PlatformCheckpointer(native=InMemorySaver())

        create_agent_engine_deep_agent(
            secure_llm=MagicMock(spec=BaseChatModel),
            backend=MagicMock(),
            checkpointer=platform,
        )

        forwarded = mock_create.call_args.kwargs["checkpointer"]
        assert isinstance(forwarded, PlatformCheckpointer)
        assert forwarded is not platform
        assert forwarded.native is platform.native
        assert forwarded._scratch is platform._scratch

    @patch("agent_engine_sdk_langgraph.deep_agent.create_deep_agent")
    def test_backend_is_required_param(self, mock_create):
        """Calling create_agent_engine_deep_agent without backend raises TypeError."""
        with pytest.raises(TypeError, match="backend"):
            create_agent_engine_deep_agent(  # type: ignore[call-arg]
                secure_llm=MagicMock(spec=BaseChatModel),
            )
        mock_create.assert_not_called()

    @patch("agent_engine_sdk_langgraph.deep_agent.create_deep_agent")
    def test_explicit_backend_passed_through(self, mock_create):
        """When a backend is provided, it is forwarded to create_deep_agent."""
        custom_backend = MagicMock()

        create_agent_engine_deep_agent(
            secure_llm=MagicMock(spec=BaseChatModel),
            backend=custom_backend,
        )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["backend"] is custom_backend

    @patch("agent_engine_sdk_langgraph.deep_agent.create_deep_agent")
    def test_returns_create_deep_agent_result(self, mock_create):
        """Return value is whatever create_deep_agent returns."""
        mock_graph = MagicMock()
        mock_create.return_value = mock_graph

        result = create_agent_engine_deep_agent(
            secure_llm=MagicMock(spec=BaseChatModel),
            backend=MagicMock(),
        )

        assert result is mock_graph

    @patch("agent_engine_sdk_langgraph.deep_agent.create_deep_agent")
    def test_validates_subagents_before_calling_create(self, mock_create):
        """RuntimeError raised for string model AND create_deep_agent not called."""
        bad_subagents = _specs(
            {
                "name": "bad",
                "description": "Bad agent",
                "system_prompt": "Bad.",
                "model": "openai:gpt-4o",
            }
        )

        with pytest.raises(RuntimeError, match="bypasses OE routing"):
            create_agent_engine_deep_agent(
                secure_llm=MagicMock(spec=BaseChatModel),
                backend=MagicMock(),
                subagents=bad_subagents,
            )

        mock_create.assert_not_called()

    @patch("agent_engine_sdk_langgraph.deep_agent.create_deep_agent")
    def test_configures_durable_guard_for_compiled_subagent_checkpointer(
        self, mock_create
    ) -> None:
        runnable = _checkpointed_graph().with_config({"tags": ["wrapped"]})

        create_agent_engine_deep_agent(
            secure_llm=MagicMock(spec=BaseChatModel),
            backend=MagicMock(),
            subagents=_specs(
                {
                    "name": "isolated",
                    "description": "Uses its own checkpointer",
                    "runnable": runnable,
                }
            ),
        )

        durable_middleware = mock_create.call_args.kwargs["middleware"][1]
        assert isinstance(durable_middleware, DurableDeepAgentMiddleware)
        assert durable_middleware.unsupported_subagent_names == frozenset({"isolated"})


# ---------------------------------------------------------------------------
# App.deep_agent()
# ---------------------------------------------------------------------------


def _enable_deep_agent_flag(app: App) -> None:
    """Flip ``features.deep_agent`` on the App's parsed agent_config.

    The fail-fast guard in App.deep_agent() reads the flag via
    ``agent_config.feature_enabled("deep_agent", default=False)`` which
    returns ``self.features.deep_agent`` when non-None. Setting the field
    directly is cleaner than swapping the method and works with pydantic's
    validate-assignment semantics on ``AgentFeatureConfig`` (extra="ignore").
    """
    app._runtime.agent_config.features.deep_agent = True


class TestAppDeepAgentFeatureFlag:
    """The ``features.deep_agent`` opt-in guards the factory."""

    def test_raises_when_flag_missing(self):
        """Without ``features.deep_agent: true`` the factory fails fast
        with a message that names the exact fix in ``agent.yaml``, rather
        than deferring to an opaque 'unknown tool' error at the first
        filesystem_* call."""
        app = App(app_name="Test Agent")
        # The default RuntimeAgentConfig has deep_agent=None → feature_enabled
        # returns False. No patching needed; this is the real code path.
        mock_llm = MagicMock(spec=BaseChatModel)
        with pytest.raises(RuntimeError, match="features.deep_agent"):
            app.deep_agent(mock_llm)

    def test_passes_when_flag_enabled(self):
        """When the flag is on, construction proceeds to the factory."""
        app = App(app_name="Test Agent")
        _enable_deep_agent_flag(app)
        mock_llm = MagicMock(spec=BaseChatModel)
        app.llm = MagicMock(return_value=MagicMock(spec=BaseChatModel))
        with patch(
            "agent_engine_sdk_langgraph.deep_agent.create_agent_engine_deep_agent"
        ) as mock_factory:
            app.deep_agent(mock_llm)
            mock_factory.assert_called_once()


class TestAppDeepAgent:
    """Tests for App.deep_agent() method."""

    @pytest.fixture(autouse=True)
    def _enable_deep_agent(self):
        """Every test in this class exercises the factory-wiring path, which
        is gated by ``features.deep_agent``. Patch
        ``AgentFeatureConfig.__init__`` defaults so every ``App`` constructed
        inside the class sees ``deep_agent=True`` without needing an on-disk
        ``agent.yaml``. ``TestAppDeepAgentFeatureFlag`` above is the one
        class that deliberately bypasses this to exercise the gate itself."""
        from agent_engine_runner_shared.agent_config import AgentFeatureConfig

        original_init = AgentFeatureConfig.__init__

        def patched_init(self, **kwargs):
            kwargs.setdefault("deep_agent", True)
            original_init(self, **kwargs)

        with patch.object(AgentFeatureConfig, "__init__", patched_init):
            yield

    @patch("agent_engine_sdk_langgraph.deep_agent.create_agent_engine_deep_agent")
    def test_creates_secure_wrapped_llm(self, mock_factory):
        """App.deep_agent() calls self.llm(llm) to wrap the model."""
        app = App(app_name="Test Agent")
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_secure = MagicMock(spec=BaseChatModel)
        app.llm = MagicMock(return_value=mock_secure)

        app.deep_agent(mock_llm)

        app.llm.assert_called_once_with(mock_llm)
        # Verify secure_llm was passed to factory
        call_kwargs = mock_factory.call_args
        assert call_kwargs.kwargs["secure_llm"] is mock_secure

    @patch("agent_engine_sdk_langgraph.backends.toolpod.AgentEngineToolPodBackend")
    @patch("agent_engine_sdk_langgraph.deep_agent.create_agent_engine_deep_agent")
    def test_defaults_backend_to_agent_engine_toolpod(
        self, mock_factory, mock_backend_cls
    ):
        """App.deep_agent() creates AgentEngineToolPodBackend when no backend provided."""
        mock_backend_instance = MagicMock()
        mock_backend_cls.return_value = mock_backend_instance
        app = App(app_name="Test Agent")
        mock_llm = MagicMock(spec=BaseChatModel)
        app.llm = MagicMock(return_value=MagicMock(spec=BaseChatModel))

        app.deep_agent(mock_llm)

        mock_backend_cls.assert_called_once()
        call_kwargs = mock_factory.call_args.kwargs
        assert call_kwargs["backend"] is mock_backend_instance

    @patch("agent_engine_sdk_langgraph.deep_agent.create_agent_engine_deep_agent")
    def test_custom_backend_passed_through(self, mock_factory):
        """Custom backend overrides the default AgentEngineToolPodBackend."""
        app = App(app_name="Test Agent")
        mock_llm = MagicMock(spec=BaseChatModel)
        app.llm = MagicMock(return_value=MagicMock(spec=BaseChatModel))
        custom_backend = MagicMock()

        app.deep_agent(mock_llm, backend=custom_backend)

        call_kwargs = mock_factory.call_args.kwargs
        assert call_kwargs["backend"] is custom_backend

    @patch("agent_engine_sdk_langgraph.deep_agent.create_agent_engine_deep_agent")
    def test_checkpointer_defaults_to_app_checkpointer(self, mock_factory):
        """Checkpointer defaults to app.checkpointer() when not provided."""
        app = App(app_name="Test Agent")
        mock_llm = MagicMock(spec=BaseChatModel)
        app.llm = MagicMock(return_value=MagicMock(spec=BaseChatModel))
        mock_cp = MagicMock()
        app.checkpointer = MagicMock(return_value=mock_cp)

        app.deep_agent(mock_llm)

        app.checkpointer.assert_called_once()
        call_kwargs = mock_factory.call_args.kwargs
        assert call_kwargs["checkpointer"] is mock_cp

    @patch("agent_engine_sdk_langgraph.deep_agent.create_agent_engine_deep_agent")
    def test_explicit_none_checkpointer_disables_default(self, mock_factory):
        """Passing checkpointer=None disables the default app.checkpointer()."""
        app = App(app_name="Test Agent")
        mock_llm = MagicMock(spec=BaseChatModel)
        app.llm = MagicMock(return_value=MagicMock(spec=BaseChatModel))
        app.checkpointer = MagicMock(return_value=MagicMock())

        app.deep_agent(mock_llm, checkpointer=None)

        app.checkpointer.assert_not_called()
        call_kwargs = mock_factory.call_args.kwargs
        assert call_kwargs["checkpointer"] is None

    @patch("agent_engine_sdk_langgraph.deep_agent.create_agent_engine_deep_agent")
    def test_custom_checkpointer_passed_through(self, mock_factory):
        """Custom checkpointer is forwarded instead of the default."""
        app = App(app_name="Test Agent")
        mock_llm = MagicMock(spec=BaseChatModel)
        app.llm = MagicMock(return_value=MagicMock(spec=BaseChatModel))
        custom_cp = MagicMock()

        app.deep_agent(mock_llm, checkpointer=custom_cp)

        call_kwargs = mock_factory.call_args.kwargs
        assert call_kwargs["checkpointer"] is custom_cp

    @patch("agent_engine_sdk_langgraph.deep_agent.create_agent_engine_deep_agent")
    def test_passes_all_kwargs_through(self, mock_factory):
        """All kwargs are forwarded to create_agent_engine_deep_agent."""
        app = App(app_name="Test Agent")
        mock_llm = MagicMock(spec=BaseChatModel)
        app.llm = MagicMock(return_value=MagicMock(spec=BaseChatModel))

        tools = [MagicMock()]
        subagents = _specs({"name": "a", "description": "a", "system_prompt": "a"})
        system_prompt = "Custom prompt"
        middleware = (MagicMock(),)
        mock_store = MagicMock()
        mock_skills = ["/path/to/skill"]

        app.deep_agent(
            mock_llm,
            tools=tools,
            subagents=subagents,
            system_prompt=system_prompt,
            middleware=middleware,
            store=mock_store,
            skills=mock_skills,
        )

        call_kwargs = mock_factory.call_args.kwargs
        assert call_kwargs["tools"] is tools
        assert call_kwargs["subagents"] is subagents
        assert call_kwargs["system_prompt"] == system_prompt
        assert call_kwargs["middleware"] is middleware
        assert call_kwargs["store"] is mock_store
        assert call_kwargs["skills"] is mock_skills

    @patch("agent_engine_sdk_langgraph.deep_agent.create_agent_engine_deep_agent")
    def test_returns_compiled_state_graph(self, mock_factory):
        """Return value is the CompiledStateGraph from create_agent_engine_deep_agent."""
        app = App(app_name="Test Agent")
        mock_llm = MagicMock(spec=BaseChatModel)
        app.llm = MagicMock(return_value=MagicMock(spec=BaseChatModel))
        mock_graph = MagicMock()
        mock_factory.return_value = mock_graph

        result = app.deep_agent(mock_llm)

        assert result is mock_graph

    def test_validates_subagent_string_model_raises(self):
        """String model in SubAgent spec raises RuntimeError."""
        app = App(app_name="Test Agent")
        mock_llm = MagicMock(spec=BaseChatModel)
        app.llm = MagicMock(return_value=MagicMock(spec=BaseChatModel))

        bad_subagents = _specs(
            {
                "name": "bad",
                "description": "bad",
                "system_prompt": "bad",
                "model": "openai:gpt-4o",
            }
        )

        with pytest.raises(RuntimeError, match="bypasses OE routing"):
            app.deep_agent(mock_llm, subagents=bad_subagents)
