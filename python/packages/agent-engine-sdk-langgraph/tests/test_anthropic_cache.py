"""Tests for Anthropic prompt-cache breakpoint placement.

The LangChain agent middleware that would normally add these markers is gated
on the provider client type, and the platform's wrapped model never matches it.
A marker carried in a message block also does not survive the agent-pod to
tool-pod conversion, so the tool-pod adapter places the system marker itself
and hands the tail breakpoint to the provider call option. The tail must go
through the option because a tool message, the usual tail of an agent loop, can
only be marked on its ``tool_result`` block, and Anthropic rejects a breakpoint
nested inside one.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableBinding

from agent_engine_sdk_langgraph.anthropic_cache import (
    _attach_breakpoint_to_last_block,
    _mark_anthropic_system_prompt,
    _take_anthropic_cache_control,
    matches,
)


class TestAnthropicCache:
    """Unit tests for the rules that live in the provider module."""

    def _anthropic(self) -> Any:
        langchain_anthropic = pytest.importorskip("langchain_anthropic")
        return langchain_anthropic.ChatAnthropic(
            model_name="claude-sonnet-5", api_key="test-key"
        )

    def test_detects_anthropic_client(self):
        assert matches(self._anthropic()) is True

    def test_ignores_other_providers(self):
        assert matches(MagicMock()) is False
        assert matches(MagicMock()) is False

    def test_detects_anthropic_client_through_runnable_binding(self):
        model = self._anthropic()
        binding = MagicMock(spec=RunnableBinding)
        binding.bound = model
        assert matches(binding) is True

    def test_takes_option_and_returns_default(self):
        kwargs: dict[str, Any] = {}
        assert _take_anthropic_cache_control(self._anthropic(), kwargs) == {
            "type": "ephemeral"
        }
        assert kwargs == {}

    def test_caller_value_wins(self):
        kwargs: dict[str, Any] = {"cache_control": {"type": "ephemeral", "ttl": "1h"}}
        assert _take_anthropic_cache_control(self._anthropic(), kwargs) == {
            "type": "ephemeral",
            "ttl": "1h",
        }
        assert "cache_control" not in kwargs

    def test_true_selects_default(self):
        kwargs: dict[str, Any] = {"cache_control": True}
        assert _take_anthropic_cache_control(self._anthropic(), kwargs) == {
            "type": "ephemeral"
        }

    def test_falsy_value_disables_placement(self):
        kwargs: dict[str, Any] = {"cache_control": False}
        assert _take_anthropic_cache_control(self._anthropic(), kwargs) is None
        assert "cache_control" not in kwargs

    def test_other_provider_keeps_option(self):
        kwargs: dict[str, Any] = {"cache_control": {"type": "ephemeral"}}
        assert _take_anthropic_cache_control(MagicMock(), kwargs) is None
        assert kwargs == {"cache_control": {"type": "ephemeral"}}

    def test_marks_system_prompt_only(self):
        tagged = _mark_anthropic_system_prompt(
            [
                SystemMessage(content="you are helpful"),
                HumanMessage(content="hi"),
            ],
            {"type": "ephemeral"},
        )

        assert tagged[0].content == [
            {
                "type": "text",
                "text": "you are helpful",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        assert tagged[1].content == "hi"

    def test_does_not_tag_a_non_system_first_message(self):
        tagged = _mark_anthropic_system_prompt(
            [HumanMessage(content="hi")], {"type": "ephemeral"}
        )
        assert tagged[0].content == "hi"

    def test_skips_thinking_blocks_and_empty_text(self):
        assert (
            _attach_breakpoint_to_last_block(
                [{"type": "text", "text": ""}, {"type": "thinking", "thinking": "x"}],
                {"type": "ephemeral"},
            )
            is None
        )

    def test_existing_marker_wins(self):
        """Guards the case where a marker does reach a block."""
        blocks = [
            {"type": "text", "text": "hi"},
            {"type": "text", "text": "tail", "cache_control": {"type": "ephemeral"}},
        ]
        assert (
            _attach_breakpoint_to_last_block(blocks, {"type": "ephemeral", "ttl": "1h"})
            is None
        )

    def test_direct_client_payload_carries_both_breakpoints(self):
        """The markers reach the wire in the form the direct API accepts.

        This inspects the provider payload because the ticket is about caching
        instructions that never arrived, and a unit assertion on our own kwargs
        cannot show that they did.
        """
        model = self._anthropic()
        payload = model._get_request_payload(
            _mark_anthropic_system_prompt(
                [SystemMessage(content="you are helpful"), HumanMessage(content="hi")],
                {"type": "ephemeral"},
            ),
            cache_control={"type": "ephemeral"},
        )

        assert payload["system"] == [
            {
                "type": "text",
                "text": "you are helpful",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        assert payload["cache_control"] == {"type": "ephemeral"}

    def test_gateway_client_marks_the_tool_result_block(self):
        """A gateway transport expands the option onto the tool_result block.

        It must not nest the breakpoint inside the block's content, which
        Anthropic rejects, and it must not leak the top-level field that only
        the direct API accepts.
        """
        langchain_anthropic = pytest.importorskip("langchain_anthropic")

        class GatewayChatAnthropic(langchain_anthropic.ChatAnthropic):
            @property
            def _llm_type(self) -> str:
                return "anthropic-chat-gateway"

        model = GatewayChatAnthropic(model_name="claude-sonnet-5", api_key="test-key")
        payload = model._get_request_payload(
            _mark_anthropic_system_prompt(
                [
                    SystemMessage(content="you are helpful"),
                    HumanMessage(content="hi"),
                    ToolMessage(content="tool output", tool_call_id="call_1"),
                ],
                {"type": "ephemeral"},
            ),
            cache_control={"type": "ephemeral"},
        )

        assert "cache_control" not in payload
        tool_result = payload["messages"][-1]["content"][-1]
        assert tool_result["type"] == "tool_result"
        assert tool_result["cache_control"] == {"type": "ephemeral"}
