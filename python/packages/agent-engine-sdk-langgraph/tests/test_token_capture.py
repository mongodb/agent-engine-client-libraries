"""Tests for token usage extraction in SecureWrappedLLM."""

from langchain_core.messages import AIMessage

from agent_engine_runner_shared.secure_wrapper import extract_usage


class TestExtractUsage:
    """Tests for extract_usage helper function."""

    def test_openai_style_usage(self):
        """OpenAI-style response_metadata with usage dict."""
        msg = AIMessage(
            content="Hello",
            response_metadata={
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
                "model_name": "gpt-4o",
            },
        )
        usage = extract_usage(msg, fallback_model="fallback")
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert usage["model"] == "gpt-4o"

    def test_anthropic_style_usage(self):
        """Anthropic-style with input_tokens/output_tokens keys."""
        msg = AIMessage(
            content="Hello",
            response_metadata={
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
            },
        )
        usage = extract_usage(msg, fallback_model="claude-3-5-sonnet")
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150  # computed from prompt + completion
        assert usage["model"] == "claude-3-5-sonnet"

    def test_token_usage_key_fallback(self):
        """Falls back to token_usage dict when usage is missing."""
        msg = AIMessage(
            content="Hello",
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                },
            },
        )
        usage = extract_usage(msg)
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50

    def test_empty_usage_falls_through_to_token_usage(self):
        """Empty usage must not skip a valid token_usage sibling."""
        msg = AIMessage(
            content="Hello",
            response_metadata={
                "usage": {},
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                },
            },
        )
        usage = extract_usage(msg)
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50

    def test_malformed_usage_falls_through_to_token_usage(self):
        """Unusable usage must not skip a valid token_usage sibling."""
        msg = AIMessage(
            content="Hello",
            response_metadata={
                "usage": {"foo": 1},
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                },
            },
        )
        usage = extract_usage(msg)
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50

    def test_missing_response_metadata_attr(self):
        """AIMessage without response_metadata attribute returns all None."""
        msg = AIMessage(content="Hello")
        # Remove response_metadata if it exists
        if hasattr(msg, "response_metadata"):
            msg.response_metadata = None  # type: ignore[assignment]  # intentionally testing None
        usage = extract_usage(msg, fallback_model="test-model")
        assert usage["prompt_tokens"] is None
        assert usage["completion_tokens"] is None
        assert usage["total_tokens"] is None
        assert usage["model"] == "test-model"

    def test_empty_usage_dict(self):
        """Empty usage dict returns None for all tokens."""
        msg = AIMessage(
            content="Hello",
            response_metadata={"usage": {}},
        )
        usage = extract_usage(msg)
        assert usage["prompt_tokens"] is None
        assert usage["completion_tokens"] is None
        assert usage["total_tokens"] is None

    def test_non_aimessage_response(self):
        """Non-AIMessage (e.g. plain string) should handle gracefully."""
        usage = extract_usage("plain string response", fallback_model="test")
        assert usage["prompt_tokens"] is None
        assert usage["completion_tokens"] is None
        assert usage["model"] == "test"

    def test_model_name_from_response_metadata(self):
        """Model name should come from response_metadata when available."""
        msg = AIMessage(
            content="Hello",
            response_metadata={
                "model_name": "gpt-4o-from-metadata",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )
        usage = extract_usage(msg, fallback_model="fallback-model")
        assert usage["model"] == "gpt-4o-from-metadata"

    def test_model_name_fallback(self):
        """Model name falls back to wrapper attribute when not in metadata."""
        msg = AIMessage(
            content="Hello",
            response_metadata={
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )
        usage = extract_usage(msg, fallback_model="claude-3")
        assert usage["model"] == "claude-3"

    def test_model_key_in_response_metadata(self):
        """Some providers use 'model' instead of 'model_name'."""
        msg = AIMessage(
            content="Hello",
            response_metadata={
                "model": "gpt-4o-via-model-key",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )
        usage = extract_usage(msg, fallback_model="fallback")
        assert usage["model"] == "gpt-4o-via-model-key"

    def test_total_tokens_computed_when_missing(self):
        """total_tokens should be computed if not provided but prompt+completion are."""
        msg = AIMessage(
            content="Hello",
            response_metadata={
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    # no total_tokens
                },
            },
        )
        usage = extract_usage(msg)
        assert usage["total_tokens"] == 150

    def test_usage_metadata_fallback(self):
        """Falls back to usage_metadata when response_metadata has no usage."""
        msg = AIMessage(content="Hello", response_metadata={})
        # Simulate usage_metadata attribute (used by Cerebras streaming)
        object.__setattr__(
            msg,
            "usage_metadata",
            {
                "prompt_tokens": 200,
                "completion_tokens": 80,
                "total_tokens": 280,
            },
        )
        usage = extract_usage(msg, fallback_model="qwen-3-32b")
        assert usage["prompt_tokens"] == 200
        assert usage["completion_tokens"] == 80
        assert usage["total_tokens"] == 280
        assert usage["model"] == "qwen-3-32b"

    def test_usage_metadata_with_input_output_keys(self):
        """usage_metadata with input_tokens/output_tokens keys should work."""
        msg = AIMessage(content="Hello", response_metadata={})
        object.__setattr__(
            msg,
            "usage_metadata",
            {
                "input_tokens": 150,
                "output_tokens": 60,
            },
        )
        usage = extract_usage(msg)
        assert usage["prompt_tokens"] == 150
        assert usage["completion_tokens"] == 60
        assert usage["total_tokens"] == 210  # computed

    def test_response_metadata_takes_precedence_over_usage_metadata(self):
        """response_metadata.usage should be preferred over usage_metadata."""
        msg = AIMessage(
            content="Hello",
            response_metadata={
                "usage": {"prompt_tokens": 100, "completion_tokens": 50}
            },
        )
        object.__setattr__(
            msg,
            "usage_metadata",
            {
                "prompt_tokens": 999,
                "completion_tokens": 999,
            },
        )
        usage = extract_usage(msg)
        assert usage["prompt_tokens"] == 100  # from response_metadata, not 999
        assert usage["completion_tokens"] == 50

    def test_anthropic_usage_object_in_response_metadata(self):
        from types import SimpleNamespace

        msg = AIMessage(
            content="Hello",
            response_metadata={
                "usage": SimpleNamespace(input_tokens=1048, output_tokens=1222),
            },
        )
        usage = extract_usage(msg, fallback_model="claude-sonnet-4-6")
        assert usage["prompt_tokens"] == 1048
        assert usage["completion_tokens"] == 1222
        assert usage["total_tokens"] == 2270
