"""Tests for caller-payload context functions.

Verify get/set/clear semantics and that the payload from one execution
context does not leak into another.
"""

from unittest.mock import MagicMock

from agent_engine_runner_shared.context import (
    clear_execution_context,
    get_current_payload,
    set_execution_context,
)

# Helpers to satisfy required positional args
_EXEC_ID = "test-exec-001"
_WRAPPER = MagicMock()
_OE_URL = "http://localhost:8000"


def test_set_execution_context_with_payload_returns_it():
    """set_execution_context with payload makes get_current_payload return it."""
    payload = {"message": "Hello", "extra": {"screen": "policy-list"}}
    tokens = set_execution_context(_EXEC_ID, _WRAPPER, _OE_URL, payload=payload)

    assert get_current_payload() == payload
    clear_execution_context(tokens)


def test_set_execution_context_without_payload_returns_empty_dict():
    """set_execution_context called without payload makes get_current_payload return {}."""
    tokens = set_execution_context(_EXEC_ID, _WRAPPER, _OE_URL)

    assert get_current_payload() == {}
    clear_execution_context(tokens)


def test_clear_execution_context_resets_payload_to_empty():
    """clear_execution_context resets the payload so get_current_payload returns {}."""
    tokens = set_execution_context(_EXEC_ID, _WRAPPER, _OE_URL, payload={"extra": "v"})
    assert get_current_payload() == {"extra": "v"}

    clear_execution_context(tokens)

    assert get_current_payload() == {}


def test_sequential_contexts_do_not_leak_payload():
    """A payload set in one context is not visible after a new context is established."""
    tokens1 = set_execution_context(_EXEC_ID, _WRAPPER, _OE_URL, payload={"extra": "first"})
    assert get_current_payload() == {"extra": "first"}
    clear_execution_context(tokens1)

    tokens2 = set_execution_context(_EXEC_ID, _WRAPPER, _OE_URL)
    result = get_current_payload()

    assert result == {}, f"Payload from the first context leaked into the second: {result}"
    clear_execution_context(tokens2)


def test_get_current_payload_returns_empty_dict_before_any_context_set():
    """get_current_payload returns {} when no execution context has been established."""
    assert get_current_payload() == {}
