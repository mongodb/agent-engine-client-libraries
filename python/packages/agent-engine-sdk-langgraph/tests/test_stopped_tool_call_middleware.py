"""Tests for StoppedToolCallMiddleware.

Locks down each branch (all-stopped, partial, none, no-batch) of
`before_model` so behavior stays deterministic.
"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from agent_engine_sdk_langgraph.stopped_tool_call_middleware import (
    ALL_INTERRUPTED_MESSAGE,
    StoppedToolCallMiddleware,
    _latest_tool_batch,
)
from agent_engine_runner_shared.secure_wrapper import CALL_INTERRUPTED_ARTIFACT_KEY

_NO_RUNTIME = cast(Runtime[Any], None)

INTERRUPTED_ARTIFACT = {CALL_INTERRUPTED_ARTIFACT_KEY: True}


def _ai_with_tool_calls(*call_ids: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "t", "args": {}, "id": call_id} for call_id in call_ids],
    )


class TestLatestToolBatch:
    def test_empty_messages_returns_empty_batch(self) -> None:
        assert _latest_tool_batch([]) == []

    def test_plain_assistant_reply_has_no_batch(self) -> None:
        messages = [HumanMessage(content="hi"), AIMessage(content="hello there")]
        assert _latest_tool_batch(messages) == []

    def test_returns_tool_messages_in_original_order(self) -> None:
        tool_1 = ToolMessage(content="a", tool_call_id="1")
        tool_2 = ToolMessage(content="b", tool_call_id="2")
        messages = [
            HumanMessage(content="hi"),
            _ai_with_tool_calls("1", "2"),
            tool_1,
            tool_2,
        ]
        assert _latest_tool_batch(messages) == [tool_1, tool_2]

    def test_only_considers_the_most_recent_batch(self) -> None:
        """An older tool batch earlier in history must not leak into the
        current decision — only what follows the newest AIMessage counts."""
        stale_tool = ToolMessage(content="stale", tool_call_id="0")
        current_tool = ToolMessage(content="current", tool_call_id="1")
        messages = [
            _ai_with_tool_calls("0"),
            stale_tool,
            AIMessage(content="following up"),
            _ai_with_tool_calls("1"),
            current_tool,
        ]
        assert _latest_tool_batch(messages) == [current_tool]


class TestStoppedToolCallMiddleware:
    def setup_method(self) -> None:
        self.middleware = StoppedToolCallMiddleware()

    def _run(self, messages: list) -> dict | None:
        return self.middleware.before_model({"messages": messages}, _NO_RUNTIME)

    def test_no_tool_batch_is_a_noop(self) -> None:
        messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
        assert self._run(messages) is None

    def test_all_interrupted_ends_the_turn_deterministically(self) -> None:
        messages = [
            HumanMessage(content="hi"),
            _ai_with_tool_calls("1", "2"),
            ToolMessage(
                content="stopped", artifact=INTERRUPTED_ARTIFACT, tool_call_id="1"
            ),
            ToolMessage(
                content="stopped", artifact=INTERRUPTED_ARTIFACT, tool_call_id="2"
            ),
        ]

        result = self._run(messages)

        assert result is not None
        assert result["jump_to"] == "end"
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == ALL_INTERRUPTED_MESSAGE

    def test_single_interrupted_call_ends_the_turn(self) -> None:
        """A lone tool call, if interrupted, is trivially "all interrupted" —
        same deterministic outcome as the multi-call case."""
        messages = [
            HumanMessage(content="hi"),
            _ai_with_tool_calls("1"),
            ToolMessage(
                content="stopped", artifact=INTERRUPTED_ARTIFACT, tool_call_id="1"
            ),
        ]

        result = self._run(messages)

        assert result is not None
        assert result["jump_to"] == "end"

    def test_partial_interruption_defers_to_the_model(self) -> None:
        """Real results exist alongside the interrupted one — let the model
        react normally instead of short-circuiting."""
        messages = [
            HumanMessage(content="hi"),
            _ai_with_tool_calls("1", "2"),
            ToolMessage(content="real result", tool_call_id="1"),
            ToolMessage(
                content="stopped", artifact=INTERRUPTED_ARTIFACT, tool_call_id="2"
            ),
        ]

        assert self._run(messages) is None

    def test_no_interruption_is_a_noop(self) -> None:
        messages = [
            HumanMessage(content="hi"),
            _ai_with_tool_calls("1"),
            ToolMessage(content="real result", tool_call_id="1"),
        ]

        assert self._run(messages) is None

    def test_non_dict_artifact_is_not_mistaken_for_interrupted(self) -> None:
        """A tool's own artifact (any shape other than our reserved key) must
        never be misread as an interrupt marker."""
        messages = [
            HumanMessage(content="hi"),
            _ai_with_tool_calls("1"),
            ToolMessage(
                content="real result",
                artifact=["some", "real", "artifact"],
                tool_call_id="1",
            ),
        ]

        assert self._run(messages) is None

    def test_declares_end_as_a_valid_jump_target(self) -> None:
        """LangChain requires can_jump_to metadata on any hook that returns
        jump_to, or the graph rejects the jump at compile/run time."""
        assert "end" in getattr(self.middleware.before_model, "__can_jump_to__", [])
