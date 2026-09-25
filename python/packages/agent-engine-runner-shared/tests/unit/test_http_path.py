"""Shared URL path-segment quoting tests."""

from __future__ import annotations

import pytest

from agent_engine_runner_shared.http_path import quote_path_segment


def test_quote_path_segment_encodes_specials() -> None:
    assert quote_path_segment("session 1") == "session%201"


def test_quote_path_segment_rejects_traversal() -> None:
    for bad in ("", ".", "..", "a/b", "a\\b", "%2e%2e", "%252e%252e", "a?b", "a#b"):
        with pytest.raises(RuntimeError, match="path segment is invalid"):
            quote_path_segment(bad)


def test_quote_path_segment_rejects_malformed_percent() -> None:
    with pytest.raises(RuntimeError, match="path segment is invalid"):
        quote_path_segment("100%zz")
