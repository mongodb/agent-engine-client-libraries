"""Tests for extract_turn_messages user-turn handling.

The user message is graph *input* and frequently never appears in
result_messages (LangGraph agents that only emit AIMessages). These tests lock
in that the user turn is recorded from the original message string, is never
duplicated when the agent does echo it, and is correctly suppressed on resume
legs so a suspend/resume cycle does not double-write messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from agent_engine_runner_shared.memory import extract_turn_messages


@dataclass
class _Msg:
    """Minimal stand-in for an sdk-core Message object."""

    role: str
    content: Any = ""
    tool_calls: Optional[List[dict]] = None
    tool_call_id: str = ""
    name: str = ""
    is_error: bool = False


def _roles(turn_messages: List[dict]) -> List[str]:
    return [m["role"] for m in turn_messages]


def test_user_turn_recorded_when_absent_from_result_messages():
    """Class-B agent: nodes emit only AIMessages, so the HumanMessage never
    surfaces in result_messages. The user turn must still be recorded."""
    result_messages = [
        _Msg(role="assistant", content="It's 62F and sunny in NYC"),
    ]

    out = extract_turn_messages(
        message="what's the weather in NYC",
        result_messages=result_messages,
        user_id="u1",
    )

    assert _roles(out) == ["user", "assistant"]
    assert out[0]["content"] == "what's the weather in NYC"
    assert out[0]["user_id"] == "u1"
    assert out[1]["content"] == "It's 62F and sunny in NYC"


def test_user_turn_with_tool_messages():
    """Full tool-calling turn with no user message in result_messages."""
    result_messages = [
        _Msg(
            role="assistant",
            content="",
            tool_calls=[{"id": "c1", "name": "get_weather", "args": {"city": "NYC"}}],
        ),
        _Msg(role="tool", content="62F sunny", tool_call_id="c1", name="get_weather"),
        _Msg(role="assistant", content="It's 62F and sunny in NYC"),
    ]

    out = extract_turn_messages(
        message="weather in NYC?",
        result_messages=result_messages,
        user_id="u1",
    )

    assert _roles(out) == ["user", "assistant", "tool", "assistant"]
    assert out[0]["content"] == "weather in NYC?"


def test_user_turn_not_duplicated_when_echoed():
    """Class-A agent: the user message IS present in result_messages. It must be
    recorded exactly once, not twice."""
    result_messages = [
        _Msg(role="user", content="hello"),
        _Msg(role="assistant", content="hi there"),
    ]

    out = extract_turn_messages(
        message="hello",
        result_messages=result_messages,
        user_id="u1",
    )

    assert _roles(out) == ["user", "assistant"]
    assert sum(1 for m in out if m["role"] == "user") == 1
    assert out[1]["content"] == "hi there"


def test_prior_history_excluded():
    """When result_messages carries earlier turns, only the current turn (from
    the matched user message onward) is written, plus the prepended user turn is
    deduped against the matched copy."""
    result_messages = [
        _Msg(role="user", content="first question"),
        _Msg(role="assistant", content="first answer"),
        _Msg(role="user", content="second question"),
        _Msg(role="assistant", content="second answer"),
    ]

    out = extract_turn_messages(
        message="second question",
        result_messages=result_messages,
        user_id="u1",
    )

    # Only the second turn is written, with exactly one user message.
    assert _roles(out) == ["user", "assistant"]
    assert out[0]["content"] == "second question"
    assert out[1]["content"] == "second answer"


def test_repeated_prompt_uses_latest_matching_turn():
    """Repeated prompt text selects the current turn, not older history."""
    result_messages = [
        _Msg(role="user", content="yes"),
        _Msg(role="assistant", content="old answer"),
        _Msg(role="user", content="yes"),
        _Msg(role="assistant", content="new answer"),
    ]

    out = extract_turn_messages(
        message="yes",
        result_messages=result_messages,
        user_id="u1",
    )

    assert [(message["role"], message["content"]) for message in out] == [
        ("user", "yes"),
        ("assistant", "new answer"),
    ]


def test_tool_error_status_is_preserved():
    result_messages = [
        _Msg(
            role="tool",
            content="lookup failed",
            tool_call_id="c1",
            name="lookup",
            is_error=True,
        )
    ]

    out = extract_turn_messages(
        message="find it",
        result_messages=result_messages,
        user_id="u1",
    )

    assert out[1]["is_error"] is True


def test_include_user_turn_false_suppresses_user():
    """On a resume leg (include_user_turn=False), no user turn is written even
    though a message string is supplied — only the assistant/tool tail."""
    result_messages = [
        _Msg(role="assistant", content="resumed answer"),
    ]

    out = extract_turn_messages(
        message="original prompt",
        result_messages=result_messages,
        user_id="u1",
        include_user_turn=False,
    )

    assert _roles(out) == ["assistant"]
    assert all(m["role"] != "user" for m in out)


def test_include_user_turn_false_suppresses_user_found_in_result_messages():
    """Robustness: even if result_messages contains a user message (e.g. replayed
    history from a future adapter or a values-mode stream), include_user_turn=False
    must suppress it — the flag governs all user turns, not just the prepended one."""
    result_messages = [
        _Msg(role="user", content="a stray/history user message"),
        _Msg(role="assistant", content="resumed answer"),
    ]

    out = extract_turn_messages(
        message="",  # resume leg: no new user prompt
        result_messages=result_messages,
        user_id="u1",
        include_user_turn=False,
    )

    assert _roles(out) == ["assistant"]
    assert all(m["role"] != "user" for m in out)


def test_multimodal_echoed_user_message_deduped():
    """When the echoed user message has multimodal (list) content, dedup must
    still match the plain-string message so the user turn is written once, not
    twice. Comparison is on normalized content."""
    result_messages = [
        _Msg(role="user", content=[{"type": "text", "text": "hello"}]),
        _Msg(role="assistant", content="hi there"),
    ]

    out = extract_turn_messages(
        message="hello",
        result_messages=result_messages,
        user_id="u1",
    )

    assert _roles(out) == ["user", "assistant"]
    assert sum(1 for m in out if m["role"] == "user") == 1
    assert out[0]["content"] == "hello"


def test_empty_message_writes_no_user_turn():
    """An empty message (tool-only / resume edge) must not produce a blank user turn."""
    result_messages = [
        _Msg(role="assistant", content="answer"),
    ]

    out = extract_turn_messages(
        message="",
        result_messages=result_messages,
        user_id="u1",
    )

    assert _roles(out) == ["assistant"]


def test_no_duplicate_messages_across_suspend_resume():
    """Simulates a HITL suspend/resume cycle and asserts the combined STM writes
    contain each message exactly once with no duplication across the boundary.

    - Suspend leg: fresh invoke (include_user_turn=True). result_messages are the
      pre-suspend node outputs. Writes: user prompt + pre-suspend assistant/tool.
    - Resume leg: include_user_turn=False (no new user prompt). LangGraph does NOT
      replay pre-suspend node outputs into the post-resume stream, so the resume's
      result_messages are disjoint post-resume outputs.
    """
    # Pre-suspend: user asked, assistant called a tool, then graph interrupted.
    presuspend_messages = [
        _Msg(role="assistant", content="", tool_calls=[{"id": "c1", "name": "lookup", "args": {}}]),
        _Msg(role="tool", content="needs human approval", tool_call_id="c1", name="lookup"),
    ]
    suspend_out = extract_turn_messages(
        message="do the risky thing",
        result_messages=presuspend_messages,
        user_id="u1",
        include_user_turn=True,
    )

    # Post-resume: only the new assistant output produced after the human approved.
    postresume_messages = [
        _Msg(role="assistant", content="done, approved and executed"),
    ]
    resume_out = extract_turn_messages(
        message="",  # resume carries no new user prompt
        result_messages=postresume_messages,
        user_id="u1",
        include_user_turn=False,
    )

    combined = suspend_out + resume_out
    contents = [(m["role"], m.get("content")) for m in combined]

    # Exactly one user turn (from the suspend leg), no duplicate assistant/tool.
    assert contents == [
        ("user", "do the risky thing"),
        ("assistant", ""),
        ("tool", "needs human approval"),
        ("assistant", "done, approved and executed"),
    ]
    assert sum(1 for m in combined if m["role"] == "user") == 1
