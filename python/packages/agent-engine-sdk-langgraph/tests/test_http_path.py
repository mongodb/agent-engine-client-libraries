"""Tests for outbound URL path-segment encoding."""

from __future__ import annotations

import pytest

from agent_engine_sdk_langgraph.http_path import quote_path_segment


class TestQuotePathSegment:
    def test_encodes_a_normal_identifier(self) -> None:
        assert quote_path_segment("session-1") == "session-1"
        assert quote_path_segment("a b") == "a%20b"

    @pytest.mark.parametrize(
        "value",
        [
            "",
            ".",
            "..",
            "a/b",
            "a\\b",
            "a?b",
            "a#b",
            "%2e",
            "%2E",
            "%2e%2e",
            "%2E%2E",
            "%2f",
            "%2F",
            "%5c",
            "%252e%252e",
            "%252f",
            "foo%2fbar",
            "%",
            "%2",
            "%zz",
        ],
    )
    def test_rejects_raw_encoded_and_double_encoded_traversal(self, value: str) -> None:
        with pytest.raises(RuntimeError, match="path segment is invalid"):
            quote_path_segment(value)
