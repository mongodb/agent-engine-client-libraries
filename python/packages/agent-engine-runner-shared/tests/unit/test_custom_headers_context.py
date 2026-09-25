"""Tests for custom_headers context functions.

Tests verify isolation, get/set/clear semantics, and that headers from one
execution context do not leak into another.
"""

from unittest.mock import MagicMock

from agent_engine_runner_shared.context import (
    clear_execution_context,
    get_current_custom_headers,
    set_execution_context,
)

# Helpers to satisfy required positional args
_EXEC_ID = "test-exec-001"
_WRAPPER = MagicMock()
_OE_URL = "http://localhost:8000"


def test_set_execution_context_with_custom_headers_returns_them():
    """set_execution_context with custom_headers makes get_current_custom_headers return those headers."""
    headers = {"authorization": "Bearer token-abc", "x-tenant-id": "tenant-1"}
    tokens = set_execution_context(_EXEC_ID, _WRAPPER, _OE_URL, custom_headers=headers)

    result = get_current_custom_headers()

    assert result == headers
    clear_execution_context(tokens)


def test_set_execution_context_without_custom_headers_returns_empty_dict():
    """set_execution_context called without custom_headers makes get_current_custom_headers return {}."""
    tokens = set_execution_context(_EXEC_ID, _WRAPPER, _OE_URL)

    result = get_current_custom_headers()

    assert result == {}
    clear_execution_context(tokens)


def test_clear_execution_context_resets_custom_headers_to_empty():
    """clear_execution_context resets custom headers so get_current_custom_headers returns {}."""
    tokens = set_execution_context(
        _EXEC_ID, _WRAPPER, _OE_URL, custom_headers={"x-some-header": "value"}
    )
    # Precondition: headers are set
    assert get_current_custom_headers() == {"x-some-header": "value"}

    clear_execution_context(tokens)

    assert get_current_custom_headers() == {}


def test_sequential_contexts_do_not_leak_headers():
    """Headers set in one execution context are not visible after a new context is established."""
    # First context sets headers
    first_headers = {"authorization": "Bearer first-secret", "x-request-id": "req-1"}
    tokens1 = set_execution_context(_EXEC_ID, _WRAPPER, _OE_URL, custom_headers=first_headers)
    assert get_current_custom_headers() == first_headers
    clear_execution_context(tokens1)

    # Second context is established without headers
    tokens2 = set_execution_context(_EXEC_ID, _WRAPPER, _OE_URL)

    result = get_current_custom_headers()
    assert result == {}, f"Headers from the first context leaked into the second context: {result}"
    clear_execution_context(tokens2)


def test_sequential_contexts_with_different_headers_do_not_cross_contaminate():
    """Two sequential contexts each with different headers see only their own headers."""
    first_headers = {"x-tenant-id": "tenant-A"}
    second_headers = {"x-tenant-id": "tenant-B", "x-request-id": "req-2"}

    tokens1 = set_execution_context(_EXEC_ID, _WRAPPER, _OE_URL, custom_headers=first_headers)
    assert get_current_custom_headers() == first_headers
    clear_execution_context(tokens1)

    tokens2 = set_execution_context(_EXEC_ID, _WRAPPER, _OE_URL, custom_headers=second_headers)
    result = get_current_custom_headers()

    assert result == second_headers
    assert result.get("x-tenant-id") == "tenant-B", (
        "Second context should see its own tenant-id, not the first context's"
    )
    clear_execution_context(tokens2)


def test_get_current_custom_headers_returns_empty_dict_before_any_context_set():
    """get_current_custom_headers returns {} when no execution context has been established."""
    result = get_current_custom_headers()

    assert result == {}
