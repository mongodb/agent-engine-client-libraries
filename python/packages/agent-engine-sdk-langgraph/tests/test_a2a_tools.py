"""Tests for A2A LangChain tool wrappers and App.a2a_tools()."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

from agent_engine_sdk_langgraph.a2a_tools import build_a2a_tools


@dataclass(frozen=True)
class FakeSkill:
    name: str = "tell-joke"
    description: str = "Tell a joke"


@dataclass(frozen=True)
class FakeAgent:
    agent_id: str = "ws-jokes"
    name: str = "Jokes"
    description: str = "Tells jokes"
    skills: list = field(default_factory=lambda: [FakeSkill()])
    capabilities: list = field(default_factory=list)


@dataclass(frozen=True)
class FakeResponse:
    status: str = "completed"
    result: str | None = "a joke"
    error: str | None = None
    execution_id: str = "exec-1"


def _make_runtime(a2a_value=None):
    runtime = MagicMock()
    type(runtime).a2a = PropertyMock(return_value=a2a_value)
    return runtime


class TestBuildA2ATools:
    def test_returns_two_tools(self):
        tools = build_a2a_tools(_make_runtime())
        assert len(tools) == 2
        assert tools[0].name == "discover_available_agents"
        assert tools[1].name == "invoke_a2a_agent"

    def test_tools_have_descriptions(self):
        tools = build_a2a_tools(_make_runtime())
        assert "Discover" in tools[0].description
        assert "Invoke" in tools[1].description


class TestDiscoverTool:
    def test_returns_agents(self):
        mock_a2a = MagicMock()
        mock_a2a.discover_agents.return_value = [FakeAgent()]
        tools = build_a2a_tools(_make_runtime(mock_a2a))

        result = json.loads(tools[0].invoke({}))
        assert len(result["agents"]) == 1
        assert result["agents"][0]["agent_id"] == "ws-jokes"
        assert result["agents"][0]["skills"][0]["name"] == "tell-joke"

    def test_filters_out_self(self, monkeypatch):
        monkeypatch.setenv("APP_ID", "ws-self")
        mock_a2a = MagicMock()
        mock_a2a.discover_agents.return_value = [
            FakeAgent(agent_id="ws-self"),
            FakeAgent(agent_id="ws-other", name="Other"),
        ]
        tools = build_a2a_tools(_make_runtime(mock_a2a))

        result = json.loads(tools[0].invoke({}))
        assert len(result["agents"]) == 1
        assert result["agents"][0]["agent_id"] == "ws-other"

    def test_returns_error_when_a2a_none(self):
        tools = build_a2a_tools(_make_runtime(None))
        result = json.loads(tools[0].invoke({}))
        assert "error" in result
        assert "A2A_JWT_SECRET" in result["error"]

    def test_returns_error_on_exception(self):
        mock_a2a = MagicMock()
        mock_a2a.discover_agents.side_effect = RuntimeError("connection refused")
        tools = build_a2a_tools(_make_runtime(mock_a2a))

        result = json.loads(tools[0].invoke({}))
        assert "error" in result
        assert "connection refused" in result["error"]

    def test_returns_empty_message_when_no_agents(self):
        mock_a2a = MagicMock()
        mock_a2a.discover_agents.return_value = []
        tools = build_a2a_tools(_make_runtime(mock_a2a))

        result = json.loads(tools[0].invoke({}))
        assert result["agents"] == []
        assert "message" in result
        assert "A2A_JWT_SECRET" in result["message"]
        assert "a2a.enabled" in result["message"]


class TestInvokeTool:
    def test_returns_response(self):
        mock_a2a = MagicMock()
        mock_a2a.invoke_agent.return_value = FakeResponse()
        tools = build_a2a_tools(_make_runtime(mock_a2a))

        invoke_args = {"agent_id": "ws-target", "message": "hello"}
        result = json.loads(tools[1].invoke(invoke_args))
        assert result["status"] == "completed"
        assert result["result"] == "a joke"
        assert result["execution_id"] == "exec-1"

    def test_returns_error_when_a2a_none(self):
        tools = build_a2a_tools(_make_runtime(None))
        invoke_args = {"agent_id": "ws-target", "message": "hello"}
        result = json.loads(tools[1].invoke(invoke_args))
        assert "error" in result

    def test_returns_error_on_exception(self):
        mock_a2a = MagicMock()
        mock_a2a.invoke_agent.side_effect = RuntimeError("timeout")
        tools = build_a2a_tools(_make_runtime(mock_a2a))

        invoke_args = {"agent_id": "ws-target", "message": "hello"}
        result = json.loads(tools[1].invoke(invoke_args))
        assert "error" in result
        assert "timeout" in result["error"]

    def test_passes_correct_args(self):
        mock_a2a = MagicMock()
        mock_a2a.invoke_agent.return_value = FakeResponse()
        tools = build_a2a_tools(_make_runtime(mock_a2a))

        tools[1].invoke({"agent_id": "ws-target", "message": "do something"})
        mock_a2a.invoke_agent.assert_called_once_with(
            agent_id="ws-target", message="do something", custom_headers=None
        )


class TestAppA2ATools:
    def test_returns_empty_when_a2a_disabled(self, monkeypatch, tmp_path: Path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.yaml").write_text(
            "name: test-agent\nentrypoint: test.main:app\n"
        )
        from agent_engine_sdk_langgraph import App

        app = App(app_name="test-agent")
        assert app.a2a_tools() == []

    def test_returns_tools_when_a2a_enabled(self, monkeypatch, tmp_path: Path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent.yaml").write_text(
            "name: test-agent\nentrypoint: test.main:app\n"
            "a2a:\n  enabled: true\n  skills:\n"
            "    - name: test\n      description: A test skill\n"
        )
        from agent_engine_sdk_langgraph import App

        app = App(app_name="test-agent")
        tools = app.a2a_tools()
        assert len(tools) == 2
        assert tools[0].name == "discover_available_agents"
        assert tools[1].name == "invoke_a2a_agent"


class TestInvokeToolCustomHeaders:
    def test_passes_custom_headers_as_json(self):
        mock_a2a = MagicMock()
        mock_a2a.invoke_agent.return_value = FakeResponse()
        tools = build_a2a_tools(_make_runtime(mock_a2a))

        tools[1].invoke(
            {
                "agent_id": "ws-target",
                "message": "hello",
                "custom_headers": '{"oauth-token": "abc"}',
            }
        )
        mock_a2a.invoke_agent.assert_called_once_with(
            agent_id="ws-target",
            message="hello",
            custom_headers={"oauth-token": "abc"},
        )

    def test_ignores_empty_custom_headers(self):
        mock_a2a = MagicMock()
        mock_a2a.invoke_agent.return_value = FakeResponse()
        tools = build_a2a_tools(_make_runtime(mock_a2a))

        tools[1].invoke(
            {
                "agent_id": "ws-target",
                "message": "hello",
                "custom_headers": "",
            }
        )
        mock_a2a.invoke_agent.assert_called_once_with(
            agent_id="ws-target",
            message="hello",
            custom_headers=None,
        )

    def test_returns_error_on_invalid_json(self):
        mock_a2a = MagicMock()
        tools = build_a2a_tools(_make_runtime(mock_a2a))

        result = json.loads(
            tools[1].invoke(
                {
                    "agent_id": "ws-target",
                    "message": "hello",
                    "custom_headers": "not-valid-json",
                }
            )
        )
        assert "error" in result
        assert "JSON" in result["error"]
        mock_a2a.invoke_agent.assert_not_called()
