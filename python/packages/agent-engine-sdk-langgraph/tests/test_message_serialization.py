"""Tests for LangChain message serialization helpers (dict ↔ AIMessage round-trips)."""

from langchain_core.messages import AIMessage

from agent_engine_sdk_langgraph.messages import (
    dict_to_ai_message,
    dict_to_ai_message_chunk,
    message_to_dict,
)


class TestDictToAiMessage:
    """Tests for dict_to_ai_message conversion."""

    def test_preserves_id(self):
        data = {"content": "hello", "tool_calls": [], "id": "run-abc-123"}
        msg = dict_to_ai_message(data)
        assert msg.id == "run-abc-123"

    def test_preserves_response_metadata(self):
        metadata = {"model": "gpt-4", "usage": {"prompt_tokens": 10}}
        data = {"content": "hello", "tool_calls": [], "response_metadata": metadata}
        msg = dict_to_ai_message(data)
        assert msg.response_metadata == metadata

    def test_preserves_name(self):
        data = {"content": "hello", "tool_calls": [], "name": "assistant"}
        msg = dict_to_ai_message(data)
        assert msg.name == "assistant"

    def test_preserves_usage_metadata(self):
        usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        data = {"content": "hello", "tool_calls": [], "usage_metadata": usage}
        msg = dict_to_ai_message(data)
        assert msg.usage_metadata is not None
        assert msg.usage_metadata["input_tokens"] == 10
        assert msg.usage_metadata["output_tokens"] == 5

    def test_preserves_additional_kwargs(self):
        data = {
            "content": "hello",
            "tool_calls": [],
            "additional_kwargs": {"refusal": None},
        }
        msg = dict_to_ai_message(data)
        assert msg.additional_kwargs == {"refusal": None}

    def test_without_optional_fields(self):
        data = {"content": "hello", "tool_calls": []}
        msg = dict_to_ai_message(data)
        assert msg.content == "hello"

    def test_content_and_tool_calls(self):
        data = {
            "content": "I'll look that up.",
            "tool_calls": [
                {
                    "name": "search",
                    "args": {"q": "test"},
                    "id": "tc1",
                    "type": "tool_call",
                }
            ],
        }
        msg = dict_to_ai_message(data)
        assert msg.content == "I'll look that up."
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "search"


class TestDictToAiMessageChunk:
    """Tests for dict_to_ai_message_chunk conversion."""

    def test_preserves_all_fields(self):
        data = {
            "content": "tok",
            "tool_calls": [],
            "id": "run-xyz-456",
            "response_metadata": {"finish_reason": "stop"},
            "name": "assistant",
            "usage_metadata": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            },
        }
        chunk = dict_to_ai_message_chunk(data)
        assert chunk.id == "run-xyz-456"
        assert chunk.response_metadata == {"finish_reason": "stop"}
        assert chunk.name == "assistant"
        assert chunk.usage_metadata is not None
        assert chunk.usage_metadata["input_tokens"] == 1


class TestMessageToDict:
    """Tests for message_to_dict conversion."""

    def test_preserves_all_fields(self):
        msg = AIMessage(
            content="hello",
            id="run-abc-123",
            name="assistant",
            response_metadata={"model": "gpt-4"},
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            additional_kwargs={"refusal": None},
        )
        result = message_to_dict(msg)
        assert result["id"] == "run-abc-123"
        assert result["name"] == "assistant"
        assert result["response_metadata"] == {"model": "gpt-4"}
        assert result["usage_metadata"] == msg.usage_metadata
        assert result["additional_kwargs"] == {"refusal": None}

    def test_includes_framework_field(self):
        msg = AIMessage(content="hello")
        result = message_to_dict(msg)
        assert result["framework"] == "langchain"


class TestRoundTrip:
    """Full round-trip serialization: AIMessage → dict → AIMessage."""

    def test_round_trip_ai_message(self):
        original = AIMessage(
            content="hello world",
            id="run-abc-123",
            name="assistant",
            response_metadata={"model": "gpt-4", "finish_reason": "stop"},
            tool_calls=[{"name": "foo", "args": {}, "id": "tc1", "type": "tool_call"}],
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            additional_kwargs={"refusal": None},
        )
        serialized = message_to_dict(original)
        restored = dict_to_ai_message(serialized)
        assert restored.content == original.content
        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.response_metadata == original.response_metadata
        assert restored.tool_calls == original.tool_calls
        assert restored.usage_metadata == original.usage_metadata
        assert restored.additional_kwargs == original.additional_kwargs

    def test_round_trip_ai_message_chunk(self):
        original = AIMessage(
            content="hello",
            id="run-abc-123",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        serialized = message_to_dict(original)
        restored = dict_to_ai_message_chunk(serialized)
        assert restored.content == original.content
        assert restored.id == original.id
        assert restored.response_metadata == original.response_metadata
        assert restored.usage_metadata == original.usage_metadata


class TestDeserializeMessagesPreservation:
    """Tests for _deserialize_messages preserving typed sdk-core Message fields."""

    def test_preserves_all_message_fields(self):
        from agent_engine_sdk import LLMToolCall, Message

        from agent_engine_runner_shared.server.tool import _deserialize_messages

        messages = [
            Message(
                role="assistant",
                content="hello",
                name="assistant",
                tool_calls=[LLMToolCall(id="tc1", name="search", args={"q": "test"})],
            ),
            Message(
                role="tool",
                content="result",
                tool_call_id="tc1",
                name="search",
            ),
            Message(role="user", content="hi"),
        ]
        result = _deserialize_messages(messages)
        assert isinstance(result[0], Message)
        assert result[0].role == "assistant"
        assert result[0].name == "assistant"
        assert result[0].tool_calls == [
            {"id": "tc1", "name": "search", "args": {"q": "test"}}
        ]
        assert result[1].role == "tool"
        assert result[1].tool_call_id == "tc1"
        assert result[1].name == "search"
        assert result[2].role == "user"
        assert result[2].content == "hi"
