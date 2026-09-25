"""Tests for token usage extraction (extract_usage).

Uses a lightweight mock instead of AIMessage since agent-engine-runner-shared has no
langchain-core dependency.
"""

from collections.abc import Mapping
from types import SimpleNamespace

from agent_engine_runner_shared.secure_wrapper import extract_usage


def _msg(response_metadata=None, **extra_attrs):
    """Build a lightweight mock that quacks like an LLM response."""
    obj = SimpleNamespace(response_metadata=response_metadata)
    for k, v in extra_attrs.items():
        setattr(obj, k, v)
    return obj


class TestExtractUsage:
    """Tests for extract_usage helper function."""

    def test_openai_style_usage(self):
        """OpenAI-style response_metadata with usage dict."""
        msg = _msg(
            response_metadata={
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
                "model_name": "gpt-4o",
            }
        )
        usage = extract_usage(msg, fallback_model="fallback")
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert usage["model"] == "gpt-4o"

    def test_anthropic_style_usage(self):
        """Anthropic-style with input_tokens/output_tokens keys."""
        msg = _msg(
            response_metadata={
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
            }
        )
        usage = extract_usage(msg, fallback_model="claude-3-5-sonnet")
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert usage["model"] == "claude-3-5-sonnet"

    def test_token_usage_key_fallback(self):
        """Falls back to token_usage dict when usage is missing."""
        msg = _msg(
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                },
            }
        )
        usage = extract_usage(msg)
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50

    def test_empty_usage_falls_through_to_token_usage(self):
        """Empty usage must not skip a valid token_usage sibling."""
        msg = _msg(
            response_metadata={
                "usage": {},
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                },
            }
        )
        usage = extract_usage(msg)
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50

    def test_malformed_usage_falls_through_to_token_usage(self):
        """Unusable usage must not skip a valid token_usage sibling."""
        msg = _msg(
            response_metadata={
                "usage": {"foo": 1},
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                },
            }
        )
        usage = extract_usage(msg)
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50

    def test_missing_response_metadata(self):
        """Response with None response_metadata returns all None."""
        msg = _msg(response_metadata=None)
        usage = extract_usage(msg, fallback_model="test-model")
        assert usage["prompt_tokens"] is None
        assert usage["completion_tokens"] is None
        assert usage["total_tokens"] is None
        assert usage["model"] == "test-model"

    def test_empty_usage_dict(self):
        """Empty usage dict returns None for all tokens."""
        msg = _msg(response_metadata={"usage": {}})
        usage = extract_usage(msg)
        assert usage["prompt_tokens"] is None
        assert usage["completion_tokens"] is None
        assert usage["total_tokens"] is None

    def test_non_object_response(self):
        """Non-object (e.g. plain string) should handle gracefully."""
        usage = extract_usage("plain string response", fallback_model="test")
        assert usage["prompt_tokens"] is None
        assert usage["completion_tokens"] is None
        assert usage["model"] == "test"

    def test_model_name_from_response_metadata(self):
        """Model name should come from response_metadata when available."""
        msg = _msg(
            response_metadata={
                "model_name": "gpt-4o-from-metadata",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )
        usage = extract_usage(msg, fallback_model="fallback-model")
        assert usage["model"] == "gpt-4o-from-metadata"

    def test_model_name_fallback(self):
        """Model name falls back to fallback_model when not in metadata."""
        msg = _msg(
            response_metadata={
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )
        usage = extract_usage(msg, fallback_model="claude-3")
        assert usage["model"] == "claude-3"

    def test_model_key_in_response_metadata(self):
        """Some providers use 'model' instead of 'model_name'."""
        msg = _msg(
            response_metadata={
                "model": "gpt-4o-via-model-key",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )
        usage = extract_usage(msg, fallback_model="fallback")
        assert usage["model"] == "gpt-4o-via-model-key"

    def test_total_tokens_computed_when_missing(self):
        """total_tokens should be computed if not provided but prompt+completion are."""
        msg = _msg(
            response_metadata={
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                },
            }
        )
        usage = extract_usage(msg)
        assert usage["total_tokens"] == 150

    def test_usage_metadata_fallback(self):
        """Falls back to usage_metadata when response_metadata has no usage."""
        msg = _msg(
            response_metadata={},
            usage_metadata={
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
        msg = _msg(
            response_metadata={},
            usage_metadata={
                "input_tokens": 150,
                "output_tokens": 60,
            },
        )
        usage = extract_usage(msg)
        assert usage["prompt_tokens"] == 150
        assert usage["completion_tokens"] == 60
        assert usage["total_tokens"] == 210

    def test_response_metadata_takes_precedence_over_usage_metadata(self):
        """response_metadata.usage should be preferred over usage_metadata."""
        msg = _msg(
            response_metadata={"usage": {"prompt_tokens": 100, "completion_tokens": 50}},
            usage_metadata={"prompt_tokens": 999, "completion_tokens": 999},
        )
        usage = extract_usage(msg)
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50


class TokenMapping(Mapping):
    """Mapping that is not a dict — Anthropic/LangChain sometimes expose this."""

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


class TestExtractUsageProviderObjects:
    """coerce non-dict usage blobs instead of dropping them."""

    def test_anthropic_usage_object_in_response_metadata(self):
        msg = _msg(
            response_metadata={
                "usage": SimpleNamespace(input_tokens=1048, output_tokens=1222),
            }
        )
        usage = extract_usage(msg, fallback_model="claude-sonnet-4-6")
        assert usage["prompt_tokens"] == 1048
        assert usage["completion_tokens"] == 1222
        assert usage["total_tokens"] == 2270
        assert usage["model"] == "claude-sonnet-4-6"

    def test_usage_metadata_non_dict_mapping(self):
        msg = _msg(
            response_metadata={},
            usage_metadata=TokenMapping({"input_tokens": 10, "output_tokens": 5}),
        )
        usage = extract_usage(msg)
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15

    def test_top_level_usage_attribute_when_metadata_empty(self):
        msg = _msg(
            response_metadata={},
            usage=SimpleNamespace(input_tokens=7, output_tokens=3),
        )
        usage = extract_usage(msg)
        assert usage["prompt_tokens"] == 7
        assert usage["completion_tokens"] == 3
        assert usage["total_tokens"] == 10

    def test_malformed_usage_string_returns_none_without_raising(self):
        msg = _msg(response_metadata={"usage": "n/a"})
        usage = extract_usage(msg, fallback_model="test")
        assert usage["prompt_tokens"] is None
        assert usage["completion_tokens"] is None
        assert usage["total_tokens"] is None
        assert usage["model"] == "test"

    def test_model_dump_usage_object(self):
        class DumpUsage:
            def model_dump(self):
                return {"input_tokens": 11, "output_tokens": 4}

        msg = _msg(response_metadata={"usage": DumpUsage()})
        usage = extract_usage(msg)
        assert usage["prompt_tokens"] == 11
        assert usage["completion_tokens"] == 4
        assert usage["total_tokens"] == 15
