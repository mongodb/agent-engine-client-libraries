"""Tests for LangChain message conversion layer."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from agent_engine_sdk import (
    DocumentBlock,
    ImageBlock,
    LLMResponse,
    LLMToolCall,
    Message,
    TextBlock,
)

from agent_engine_sdk_langgraph.messages import (
    _serialize_tool_calls,
    lc_messages_to_platform,
    lc_to_platform_message,
    lc_to_workflow_message,
    llm_response_to_chat_result,
    platform_messages_to_lc,
    platform_to_lc_message,
    workflow_to_lc_message,
)


class TestLcToPlatformMessage:
    """Tests for lc_to_platform_message conversion."""

    def test_human_message(self):
        """HumanMessage converts to role='user'."""
        lc_msg = HumanMessage(content="Hello")
        platform_msg = lc_to_platform_message(lc_msg)

        assert platform_msg.role == "user"
        assert platform_msg.content == "Hello"
        assert platform_msg.tool_calls is None
        assert platform_msg.tool_call_id is None

    def test_ai_message(self):
        """AIMessage converts to role='assistant'."""
        lc_msg = AIMessage(content="Hi there")
        platform_msg = lc_to_platform_message(lc_msg)

        assert platform_msg.role == "assistant"
        assert platform_msg.content == "Hi there"
        assert platform_msg.tool_calls is None

    def test_system_message(self):
        """SystemMessage converts to role='system'."""
        lc_msg = SystemMessage(content="You are a helpful assistant.")
        platform_msg = lc_to_platform_message(lc_msg)

        assert platform_msg.role == "system"
        assert platform_msg.content == "You are a helpful assistant."

    def test_tool_message(self):
        """ToolMessage converts to role='tool' with tool_call_id."""
        lc_msg = ToolMessage(content="Result: 42", tool_call_id="tc_123")
        platform_msg = lc_to_platform_message(lc_msg)

        assert platform_msg.role == "tool"
        assert platform_msg.content == "Result: 42"
        assert platform_msg.tool_call_id == "tc_123"
        assert platform_msg.is_error is False

    def test_failed_tool_message(self):
        lc_msg = ToolMessage(
            content="lookup failed", tool_call_id="tc_123", status="error"
        )

        platform_msg = lc_to_platform_message(lc_msg)

        assert platform_msg.is_error is True

    def test_tool_message_preserves_name(self):
        """ToolMessage.name is preserved during conversion."""
        lc_msg = ToolMessage(content="72°F", tool_call_id="tc_1", name="get_weather")
        platform_msg = lc_to_platform_message(lc_msg)

        assert platform_msg.name == "get_weather"

    def test_ai_message_with_tool_calls(self):
        """AIMessage with tool_calls preserves them as dicts."""
        lc_msg = AIMessage(
            content="Let me search for that.",
            tool_calls=[
                {
                    "name": "search",
                    "args": {"query": "test"},
                    "id": "tc_1",
                    "type": "tool_call",
                }
            ],
        )
        platform_msg = lc_to_platform_message(lc_msg)

        assert platform_msg.role == "assistant"
        assert platform_msg.tool_calls is not None
        assert len(platform_msg.tool_calls) == 1
        assert platform_msg.tool_calls[0]["name"] == "search"
        assert platform_msg.tool_calls[0]["args"] == {"query": "test"}
        assert platform_msg.tool_calls[0]["id"] == "tc_1"

    def test_empty_content(self):
        """Empty content is preserved."""
        lc_msg = HumanMessage(content="")
        platform_msg = lc_to_platform_message(lc_msg)

        assert platform_msg.content == ""

    def test_multimodal_text_content(self):
        """Multimodal content with text blocks preserves structure as ContentBlocks."""
        lc_msg = HumanMessage(
            content=[
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/img.png"},
                },
            ]
        )
        platform_msg = lc_to_platform_message(lc_msg)

        # Content should be a list of ContentBlocks
        assert isinstance(platform_msg.content, list)
        assert len(platform_msg.content) == 2

        # First block is TextBlock
        assert isinstance(platform_msg.content[0], TextBlock)
        assert platform_msg.content[0].text == "What is in this image?"

        # Second block is ImageBlock
        assert isinstance(platform_msg.content[1], ImageBlock)
        assert platform_msg.content[1].url == "https://example.com/img.png"

    def test_multimodal_only_non_text_blocks(self):
        """Multimodal content with only non-text blocks preserves as ImageBlock."""
        lc_msg = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/img.png"},
                },
            ]
        )
        platform_msg = lc_to_platform_message(lc_msg)

        # Content should be a list with one ImageBlock
        assert isinstance(platform_msg.content, list)
        assert len(platform_msg.content) == 1
        assert isinstance(platform_msg.content[0], ImageBlock)
        assert platform_msg.content[0].url == "https://example.com/img.png"

    def test_image_url_as_string_shorthand(self):
        """LangChain shorthand format with image_url as plain string is handled."""
        lc_msg = HumanMessage(
            content=[
                {"type": "image_url", "image_url": "https://example.com/img.png"},
            ]
        )
        platform_msg = lc_to_platform_message(lc_msg)

        # Should correctly extract URL from string format
        assert isinstance(platform_msg.content, list)
        assert len(platform_msg.content) == 1
        assert isinstance(platform_msg.content[0], ImageBlock)
        assert platform_msg.content[0].url == "https://example.com/img.png"

    def test_image_without_url_skipped(self):
        """Image block without url is skipped with warning."""
        lc_msg = HumanMessage(
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "image", "file_id": "file_abc123"},  # No url, will be skipped
            ]
        )
        platform_msg = lc_to_platform_message(lc_msg)

        # Only the text block should remain (image skipped)
        assert platform_msg.content == "Hello"

    def test_multimodal_single_text_block_simplifies_to_string(self):
        """Multimodal content with single text block simplifies to string."""
        lc_msg = HumanMessage(content=[{"type": "text", "text": "Just text"}])
        platform_msg = lc_to_platform_message(lc_msg)

        # Should be simplified to string
        assert isinstance(platform_msg.content, str)
        assert platform_msg.content == "Just text"

    def test_multimodal_document_block(self):
        """Document content blocks are preserved."""
        lc_msg = HumanMessage(
            content=[
                {"type": "text", "text": "Here is a document:"},
                {
                    "type": "document",
                    "url": "https://example.com/doc.pdf",
                    "mime_type": "application/pdf",
                    "filename": "doc.pdf",
                },
            ]
        )
        platform_msg = lc_to_platform_message(lc_msg)

        assert isinstance(platform_msg.content, list)
        assert len(platform_msg.content) == 2
        assert isinstance(platform_msg.content[1], DocumentBlock)
        assert platform_msg.content[1].url == "https://example.com/doc.pdf"
        assert platform_msg.content[1].mime_type == "application/pdf"
        assert platform_msg.content[1].filename == "doc.pdf"

    def test_serialize_tool_calls_passthrough(self):
        """Dict tool calls are passed through unchanged."""
        tc: ToolCall = {
            "name": "search",
            "args": {"q": "test"},
            "id": "tc_1",
            "type": "tool_call",
        }
        result = _serialize_tool_calls([tc])

        assert result == [tc]

    def test_serialize_tool_calls_empty_list(self):
        """Empty list returns empty list."""
        assert _serialize_tool_calls([]) == []

    def test_serialize_tool_calls_warns_on_missing_keys(self, caplog):
        """Tool call dicts missing required keys emit a warning."""
        bad_input: Any = [{"name": "search"}]
        result = _serialize_tool_calls(bad_input)

        assert result == [{"name": "search"}]
        assert "missing keys" in caplog.text

    def test_serialize_tool_calls_rejects_non_dict(self):
        """Non-dict tool calls raise TypeError."""
        bad_input: Any = [object()]
        with pytest.raises(TypeError, match="Expected dict tool call"):
            _serialize_tool_calls(bad_input)


def test_workflow_tool_message_preserves_supplied_id() -> None:
    """Durable snapshot conversion preserves application message identity."""
    original = ToolMessage(
        content="result", tool_call_id="call-1", id="application-message"
    )

    restored = workflow_to_lc_message(lc_to_workflow_message(original))

    assert restored.id == "application-message"


class TestPlatformToLcMessage:
    """Tests for platform_to_lc_message conversion."""

    def test_user_to_human_message(self):
        """Platform role='user' converts to HumanMessage."""
        platform_msg = Message(role="user", content="Hello")
        lc_msg = platform_to_lc_message(platform_msg)

        assert isinstance(lc_msg, HumanMessage)
        assert lc_msg.content == "Hello"

    def test_assistant_to_ai_message(self):
        """Platform role='assistant' converts to AIMessage."""
        platform_msg = Message(role="assistant", content="Hi there")
        lc_msg = platform_to_lc_message(platform_msg)

        assert isinstance(lc_msg, AIMessage)
        assert lc_msg.content == "Hi there"

    def test_system_to_system_message(self):
        """Platform role='system' converts to SystemMessage."""
        platform_msg = Message(role="system", content="You are helpful.")
        lc_msg = platform_to_lc_message(platform_msg)

        assert isinstance(lc_msg, SystemMessage)
        assert lc_msg.content == "You are helpful."

    def test_tool_to_tool_message(self):
        """Platform role='tool' converts to ToolMessage with tool_call_id."""
        platform_msg = Message(role="tool", content="42", tool_call_id="tc_123")
        lc_msg = platform_to_lc_message(platform_msg)

        assert isinstance(lc_msg, ToolMessage)
        assert lc_msg.content == "42"
        assert lc_msg.tool_call_id == "tc_123"

    def test_assistant_with_tool_calls(self):
        """Platform assistant with tool_calls converts to AIMessage with tool_calls."""
        expected_tool_call = {
            "name": "search",
            "args": {"q": "test"},
            "id": "tc_1",
            "type": "tool_call",
        }
        tool_call = LLMToolCall.model_validate(expected_tool_call)
        platform_msg = Message(
            role="assistant",
            content="Searching...",
            tool_calls=[tool_call],
        )
        lc_msg = platform_to_lc_message(platform_msg)

        assert isinstance(lc_msg, AIMessage)
        assert lc_msg.tool_calls == [expected_tool_call]

    def test_unknown_role_defaults_to_human(self):
        """Unknown role defaults to HumanMessage via the conversion mapping."""
        # Message.role is now Literal["user","assistant","tool","system"],
        # so we can't construct an invalid Message directly.
        # Instead, test the conversion mapping's fallback by simulating
        # a message with an unrecognized role using model_construct
        # (bypasses Pydantic validation, as would happen with external data).
        platform_msg = Message.model_construct(role="unknown", content="test")
        lc_msg = platform_to_lc_message(platform_msg)

        assert isinstance(lc_msg, HumanMessage)

    def test_tool_message_missing_tool_call_id_skipped(self, caplog):
        """Platform tool message without tool_call_id is skipped with warning."""
        platform_msg = Message(role="tool", content="result", tool_call_id=None)

        result = platform_to_lc_message(platform_msg)

        # Returns None when tool_call_id is missing
        assert result is None
        assert "missing tool_call_id" in caplog.text

    def test_tool_message_empty_tool_call_id_skipped(self, caplog):
        """Platform tool message with empty tool_call_id is skipped with warning."""
        platform_msg = Message(role="tool", content="result", tool_call_id="")

        result = platform_to_lc_message(platform_msg)

        # Returns None when tool_call_id is empty
        assert result is None
        assert "missing tool_call_id" in caplog.text

    def test_multimodal_content_to_lc(self):
        """Platform message with ContentBlocks converts to LangChain multimodal."""
        platform_msg = Message(
            role="user",
            content=[
                TextBlock(text="What's in this image?"),
                ImageBlock(url="https://example.com/img.png"),
            ],
        )
        lc_msg = platform_to_lc_message(platform_msg)

        assert isinstance(lc_msg, HumanMessage)
        assert isinstance(lc_msg.content, list)
        assert len(lc_msg.content) == 2
        assert lc_msg.content[0] == {"type": "text", "text": "What's in this image?"}
        image_block = lc_msg.content[1]
        assert isinstance(image_block, dict)
        assert image_block["type"] == "image"
        assert image_block["url"] == "https://example.com/img.png"

    def test_document_content_to_lc(self):
        """Platform message with DocumentBlock converts to LangChain format."""
        platform_msg = Message(
            role="user",
            content=[
                TextBlock(text="Check this doc:"),
                DocumentBlock(
                    url="https://example.com/doc.pdf",
                    mime_type="application/pdf",
                    filename="doc.pdf",
                ),
            ],
        )
        lc_msg = platform_to_lc_message(platform_msg)

        assert lc_msg is not None
        assert isinstance(lc_msg.content, list)
        doc_block = lc_msg.content[1]
        assert isinstance(doc_block, dict)
        assert doc_block["type"] == "document"
        assert doc_block["url"] == "https://example.com/doc.pdf"
        assert doc_block["mime_type"] == "application/pdf"
        assert doc_block["filename"] == "doc.pdf"


class TestLLMResponseToChatResult:
    def test_restores_usage_from_metadata(self):
        response = LLMResponse(
            content="Hello",
            metadata={"usage": {"input_tokens": 3, "output_tokens": 2}},
        )

        result = llm_response_to_chat_result(response)

        assert result.generations[0].message.response_metadata == {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
        }

    def test_restores_message_metadata(self):
        response = LLMResponse(
            content="Hello",
            id="run-1",
            name="assistant",
            additional_kwargs={"refusal": None},
            response_metadata={"finish_reason": "stop"},
        )

        result = llm_response_to_chat_result(response)
        message = result.generations[0].message

        assert message.id == "run-1"
        assert message.name == "assistant"
        assert message.additional_kwargs == {"refusal": None}
        assert message.response_metadata == {"finish_reason": "stop"}


class TestRoundTrip:
    """Round-trip conversion tests: LC -> Platform -> LC."""

    def test_human_message_round_trip(self):
        """HumanMessage survives round-trip."""
        original = HumanMessage(content="Hello world")
        platform = lc_to_platform_message(original)
        restored = platform_to_lc_message(platform)

        assert isinstance(restored, HumanMessage)
        assert restored.content == original.content

    def test_ai_message_round_trip(self):
        """AIMessage survives round-trip."""
        original = AIMessage(
            content="Hello",
            id="run-abc-123",
            name="assistant",
            response_metadata={"model": "gpt-4", "finish_reason": "stop"},
            additional_kwargs={"refusal": None},
        )
        platform = lc_to_platform_message(original)
        restored = platform_to_lc_message(platform)

        assert isinstance(restored, AIMessage)
        assert restored.content == original.content
        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.response_metadata == original.response_metadata
        assert restored.additional_kwargs == original.additional_kwargs

    def test_ai_message_with_tool_calls_round_trip(self):
        """AIMessage with tool_calls survives round-trip."""
        tool_call = {
            "name": "foo",
            "args": {"x": 1},
            "id": "tc_1",
            "type": "tool_call",
        }
        original = AIMessage(content="Calling tool", tool_calls=[tool_call])
        platform = lc_to_platform_message(original)
        restored = platform_to_lc_message(platform)

        assert isinstance(restored, AIMessage)
        assert restored.content == original.content
        assert restored.tool_calls == original.tool_calls

    def test_tool_message_round_trip(self):
        """ToolMessage survives round-trip."""
        original = ToolMessage(content="result", tool_call_id="tc_123")
        platform = lc_to_platform_message(original)
        restored = platform_to_lc_message(platform)

        assert isinstance(restored, ToolMessage)
        assert restored.content == original.content
        assert restored.tool_call_id == original.tool_call_id

    def test_tool_message_name_round_trip(self):
        """ToolMessage.name survives round-trip conversion."""
        original = ToolMessage(content="72°F", tool_call_id="tc_1", name="get_weather")
        platform = lc_to_platform_message(original)
        restored = platform_to_lc_message(platform)

        assert isinstance(restored, ToolMessage)
        assert restored.name == "get_weather"
        assert restored.tool_call_id == "tc_1"
        assert restored.content == "72°F"

    def test_tool_message_error_status_round_trip(self):
        original = ToolMessage(
            content="lookup failed", tool_call_id="tc_1", status="error"
        )
        platform = lc_to_platform_message(original)
        restored = platform_to_lc_message(platform)

        assert isinstance(restored, ToolMessage)
        assert restored.status == "error"

    def test_multimodal_message_round_trip(self):
        """Multimodal HumanMessage survives round-trip."""
        original = HumanMessage(
            content=[
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/img.png"},
                },
            ]
        )
        platform = lc_to_platform_message(original)
        restored = platform_to_lc_message(platform)

        assert isinstance(restored, HumanMessage)
        assert isinstance(restored.content, list)
        assert len(restored.content) == 2
        # Text content preserved
        text_block = restored.content[0]
        assert isinstance(text_block, dict)
        assert text_block["type"] == "text"
        assert text_block["text"] == "What is in this image?"
        # Image URL preserved
        image_block = restored.content[1]
        assert isinstance(image_block, dict)
        assert image_block["type"] == "image"
        assert image_block["url"] == "https://example.com/img.png"


class TestBatchConversion:
    """Tests for batch message conversion."""

    def test_lc_messages_to_platform(self):
        """Batch conversion from LC to platform."""
        lc_messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi!"),
        ]
        platform_messages = lc_messages_to_platform(lc_messages)

        assert len(platform_messages) == 3
        assert platform_messages[0].role == "system"
        assert platform_messages[1].role == "user"
        assert platform_messages[2].role == "assistant"

    def test_platform_messages_to_lc(self):
        """Batch conversion from platform to LC."""
        platform_messages = [
            Message(role="system", content="Be helpful."),
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!"),
        ]
        lc_messages = platform_messages_to_lc(platform_messages)

        assert len(lc_messages) == 3
        assert isinstance(lc_messages[0], SystemMessage)
        assert isinstance(lc_messages[1], HumanMessage)
        assert isinstance(lc_messages[2], AIMessage)

    def test_empty_list(self):
        """Empty list conversions work."""
        assert lc_messages_to_platform([]) == []
        assert platform_messages_to_lc([]) == []

    def test_platform_messages_to_lc_skips_invalid_tool_messages(self, caplog):
        """Batch conversion skips tool messages with missing tool_call_id."""
        platform_messages = [
            Message(role="user", content="Hi"),
            Message(role="tool", content="result", tool_call_id=None),  # skipped
            Message(role="assistant", content="Done"),
        ]
        lc_messages = platform_messages_to_lc(platform_messages)

        # Only 2 messages - the invalid tool message is skipped
        assert len(lc_messages) == 2
        assert isinstance(lc_messages[0], HumanMessage)
        assert isinstance(lc_messages[1], AIMessage)
        assert "missing tool_call_id" in caplog.text

    def test_batch_round_trip(self):
        """Batch round-trip preserves message types and content."""
        original = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="User input"),
            AIMessage(content="Assistant response"),
            ToolMessage(content="Tool result", tool_call_id="tc_1"),
        ]
        platform = lc_messages_to_platform(original)
        restored = platform_messages_to_lc(platform)

        assert len(restored) == len(original)
        for orig, rest in zip(original, restored):
            assert type(orig) is type(rest)
            assert orig.content == rest.content
