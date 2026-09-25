"""Tests for thinking-token stripping and filtering utilities."""

from agent_engine_runner_shared.utils import filter_thinking_tokens, strip_thinking


class TestFilterThinkingTokens:
    """Unit tests for the filter_thinking_tokens helper."""

    def test_plain_text_passes_through(self):
        streamable, buf, inside = filter_thinking_tokens("hello", "", False)
        assert streamable == "hello"
        assert buf == ""
        assert inside is False

    def test_full_think_block_suppressed(self):
        streamable, buf, inside = filter_thinking_tokens(
            "<think>reasoning</think>visible", "", False
        )
        assert streamable == "visible"
        assert buf == ""
        assert inside is False

    def test_open_think_tag_buffers(self):
        streamable, buf, inside = filter_thinking_tokens("<think>start", "", False)
        assert streamable == ""
        assert inside is True

    def test_close_think_tag_resumes(self):
        streamable, buf, inside = filter_thinking_tokens("more</think>after", "buffered", True)
        assert streamable == "after"
        assert inside is False

    def test_multi_chunk_think_block(self):
        buf = ""
        inside = False
        streamed = []

        for token in ["He", "<think>", "reason", "</think>", "llo"]:
            s, buf, inside = filter_thinking_tokens(token, buf, inside)
            if s:
                streamed.append(s)

        assert "".join(streamed) == "Hello"
        assert inside is False


class TestStripThinking:
    """Unit tests for the strip_thinking helper."""

    def test_removes_think_block(self):
        assert strip_thinking("<think>foo</think>bar") == "bar"

    def test_removes_unclosed_think(self):
        assert strip_thinking("before<think>trailing") == "before"

    def test_empty_string(self):
        assert strip_thinking("") == ""

    def test_no_think_tags(self):
        assert strip_thinking("plain text") == "plain text"

    def test_multiline_think_block(self):
        text = "<think>\nline1\nline2\n</think>result"
        assert strip_thinking(text) == "result"
