"""Tests for agent_engine_sdk.models."""

import pytest
from agent_engine_sdk.models import (
    AgentInput,
    AgentOutput,
    DocumentBlock,
    ImageBlock,
    LLMResponse,
    LLMStreamChunk,
    LLMTokenUsage,
    Message,
    MessageArtifact,
    MessageArtifactMetadata,
    OutputParser,
    RequestContext,
    StreamEvent,
    TextBlock,
    ToolCallChunk,
    ToolDefinition,
    collect_message_artifact_metadata,
)
from pydantic import ValidationError


class TestToolDefinition:
    def test_construction(self) -> None:
        td = ToolDefinition(
            name="search", description="Search", args_schema={}, callable=lambda: None
        )
        assert td.name == "search"

    def test_defaults(self) -> None:
        td = ToolDefinition(name="t", description="d", args_schema={}, callable=None)
        assert td.remote is True
        assert td.network == []
        assert td.timeout_seconds == 30
        assert td.redact_fields == []

    def test_round_trip(self) -> None:
        td = ToolDefinition(
            name="t", description="d", args_schema={"type": "object"}, callable=None
        )
        data = td.model_dump()
        restored = ToolDefinition.model_validate(data)
        assert restored.name == td.name
        assert restored.args_schema == td.args_schema

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            ToolDefinition()  # type: ignore[call-arg]


class TestMessage:
    def test_construction(self) -> None:
        m = Message(role="user", content="hello")
        assert m.role == "user"
        assert m.tool_calls is None
        assert m.tool_call_id is None
        assert m.is_error is None

    def test_round_trip(self) -> None:
        m = Message(
            role="tool",
            content="failed",
            tool_calls=[{"id": "1"}],
            is_error=True,
        )
        data = m.model_dump()
        restored = Message.model_validate(data)
        assert restored.tool_calls == [{"id": "1"}]
        assert restored.is_error is True

    def test_invalid_dict_equality_returns_false(self) -> None:
        m = Message(role="user", content="hello")
        assert (m == {"content": "missing role"}) is False

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            Message()  # type: ignore[call-arg]


class TestMessageArtifactMetadata:
    def test_to_message_metadata_exports_message_artifact_contract(self) -> None:
        metadata = MessageArtifactMetadata(
            artifacts=[
                MessageArtifact(
                    id="chart-1",
                    kind="chart",
                    title="Loss frequency",
                    data=[{"cohort": "1 pedal", "loss_frequency": 173.2}],
                )
            ]
        )

        assert metadata.to_message_metadata() == {
            "artifacts": [
                {
                    "id": "chart-1",
                    "kind": "chart",
                    "title": "Loss frequency",
                    "data": [{"cohort": "1 pedal", "loss_frequency": 173.2}],
                }
            ],
        }

    def test_collects_valid_artifacts_from_message_additional_kwargs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        metadata = collect_message_artifact_metadata(
            {
                "artifacts": [
                    {"id": "query-1", "kind": "mongodb_query", "title": "Request 1"},
                    {"id": "chart-1", "kind": "chart", "title": "Loss frequency"},
                    {"id": "missing-kind"},
                ],
            },
        )

        assert metadata is not None
        assert metadata.to_message_metadata() == {
            "artifacts": [
                {"id": "query-1", "kind": "mongodb_query", "title": "Request 1"},
                {"id": "chart-1", "kind": "chart", "title": "Loss frequency"},
            ],
        }
        assert "Skipping malformed message artifact metadata" in caplog.text


class TestLLMResponse:
    def test_construction(self) -> None:
        r = LLMResponse(content="response")
        assert r.content == "response"
        assert r.tool_calls is None
        assert r.metadata == {}

    def test_round_trip(self) -> None:
        r = LLMResponse(content="r", metadata={"k": "v"})
        data = r.model_dump()
        restored = LLMResponse.model_validate(data)
        assert restored.metadata == {"k": "v"}

    def test_usage_does_not_auto_populate_metadata(self) -> None:
        r = LLMResponse(
            content="r", usage=LLMTokenUsage(input_tokens=1, output_tokens=2)
        )
        assert r.usage is not None
        assert r.metadata == {}

    def test_usage_validates_from_metadata(self) -> None:
        r = LLMResponse(
            content="r",
            metadata={"usage": {"input_tokens": 4, "output_tokens": 5}},
        )
        assert r.usage is not None
        assert r.usage.total_tokens == 9

    def test_message_metadata_round_trip(self) -> None:
        r = LLMResponse(
            content="r",
            id="run-1",
            name="assistant",
            additional_kwargs={"refusal": None},
            response_metadata={"finish_reason": "stop"},
        )
        restored = LLMResponse.model_validate(r.model_dump())
        assert restored.id == "run-1"
        assert restored.name == "assistant"
        assert restored.additional_kwargs == {"refusal": None}
        assert restored.response_metadata == {"finish_reason": "stop"}

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            LLMResponse()  # type: ignore[call-arg]


class TestAgentInput:
    def test_construction(self) -> None:
        ai = AgentInput(payload={"message": "hi"})
        assert ai.payload == {"message": "hi"}

    def test_round_trip(self) -> None:
        ai = AgentInput(payload={"message": "hi", "k": "v"})
        data = ai.model_dump()
        restored = AgentInput.model_validate(data)
        assert restored.payload == {"message": "hi", "k": "v"}

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            AgentInput()  # type: ignore[call-arg]


class TestAgentOutput:
    def test_construction(self) -> None:
        ao = AgentOutput(response={"response": "ok"})
        assert ao.response == {"response": "ok"}

    def test_round_trip(self) -> None:
        ao = AgentOutput(response={"response": "ok", "k": "v"})
        data = ao.model_dump()
        restored = AgentOutput.model_validate(data)
        assert restored.response == {"response": "ok", "k": "v"}

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            AgentOutput()  # type: ignore[call-arg]


class TestStreamEvent:
    def test_construction(self) -> None:
        e = StreamEvent(data={"content": "hello"})
        assert e.data == {"content": "hello"}
        assert e.event is None

    def test_with_event(self) -> None:
        e = StreamEvent(data={"content": "hello"}, event="token")
        assert e.event == "token"

    def test_round_trip(self) -> None:
        e = StreamEvent(data={"content": "hello", "k": "v"}, event="token")
        data = e.model_dump()
        restored = StreamEvent.model_validate(data)
        assert restored.data == {"content": "hello", "k": "v"}

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            StreamEvent()  # type: ignore[call-arg]

    def test_rejects_control_chars_in_event(self) -> None:
        # Parity with @mongodb-js/agent-engine-sdk: CR/LF/NUL in the event name
        # could forge SSE frames, so they are rejected at the schema boundary.
        for bad in ("token\ndata: {}", "token\r\nevent: fake", "token\x00"):
            with pytest.raises(ValidationError):
                StreamEvent(data={"content": "hi"}, event=bad)

    def test_custom_event_defaults_none(self) -> None:
        assert StreamEvent(data={"content": "hello"}).custom_event is None

    def test_custom_event_round_trip(self) -> None:
        e = StreamEvent(data={}, event="loading", custom_event={"event": "x"})
        restored = StreamEvent.model_validate(e.model_dump())
        assert restored.custom_event == {"event": "x"}


class TestOutputParser:
    def test_missing_abstract_method_cannot_instantiate(self) -> None:
        class MissingOnError(OutputParser):
            async def parse(self, item, ctx):  # type: ignore[no-untyped-def]
                yield item

        with pytest.raises(TypeError):
            MissingOnError()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self) -> None:
        class Complete(OutputParser):
            async def parse(self, item, ctx):  # type: ignore[no-untyped-def]
                yield item

            async def on_stream_error(self, ctx, error):  # type: ignore[no-untyped-def]
                return None

        assert Complete().stream_modes == ()


class TestTextBlock:
    def test_construction(self) -> None:
        tb = TextBlock(text="Hello world")
        assert tb.type == "text"
        assert tb.text == "Hello world"

    def test_default_type(self) -> None:
        tb = TextBlock(text="test")
        assert tb.type == "text"

    def test_round_trip(self) -> None:
        tb = TextBlock(text="test content")
        data = tb.model_dump()
        restored = TextBlock.model_validate(data)
        assert restored.text == "test content"
        assert restored.type == "text"

    def test_missing_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            TextBlock()  # type: ignore[call-arg]


class TestImageBlock:
    def test_construction_with_url(self) -> None:
        ib = ImageBlock(url="https://example.com/img.png")
        assert ib.type == "image"
        assert ib.url == "https://example.com/img.png"
        assert ib.mime_type is None

    def test_construction_with_mime_type(self) -> None:
        ib = ImageBlock(url="https://example.com/img.png", mime_type="image/png")
        assert ib.url == "https://example.com/img.png"
        assert ib.mime_type == "image/png"

    def test_data_url_for_base64(self) -> None:
        """Data URLs can be used for inline base64 content."""
        data_url = "data:image/png;base64,aGVsbG8="
        ib = ImageBlock(url=data_url)
        assert ib.url == data_url

    def test_validation_requires_url(self) -> None:
        with pytest.raises(ValidationError):
            ImageBlock()  # type: ignore[call-arg]

    def test_round_trip(self) -> None:
        ib = ImageBlock(url="https://example.com/img.png", mime_type="image/jpeg")
        data = ib.model_dump()
        restored = ImageBlock.model_validate(data)
        assert restored.url == "https://example.com/img.png"
        assert restored.mime_type == "image/jpeg"


class TestDocumentBlock:
    def test_construction_with_url(self) -> None:
        db = DocumentBlock(url="https://example.com/doc.pdf")
        assert db.type == "document"
        assert db.url == "https://example.com/doc.pdf"
        assert db.mime_type is None
        assert db.filename is None

    def test_construction_with_all_fields(self) -> None:
        db = DocumentBlock(
            url="https://example.com/doc.pdf",
            mime_type="application/pdf",
            filename="document.pdf",
        )
        assert db.mime_type == "application/pdf"
        assert db.filename == "document.pdf"

    def test_data_url_for_base64(self) -> None:
        """Data URLs can be used for inline base64 content."""
        data_url = "data:application/pdf;base64,dGVzdA=="
        db = DocumentBlock(url=data_url)
        assert db.url == data_url

    def test_validation_requires_url(self) -> None:
        with pytest.raises(ValidationError):
            DocumentBlock()  # type: ignore[call-arg]

    def test_round_trip(self) -> None:
        db = DocumentBlock(
            url="https://example.com/doc.pdf",
            mime_type="application/pdf",
            filename="doc.pdf",
        )
        data = db.model_dump()
        restored = DocumentBlock.model_validate(data)
        assert restored.url == "https://example.com/doc.pdf"
        assert restored.filename == "doc.pdf"


class TestToolCallChunk:
    def test_construction_all_none(self) -> None:
        """All fields are optional - empty chunk is valid."""
        chunk = ToolCallChunk()
        assert chunk.id is None
        assert chunk.name is None
        assert chunk.args is None

    def test_construction_with_values(self) -> None:
        chunk = ToolCallChunk(id="call_123", name="search", args='{"query":')
        assert chunk.id == "call_123"
        assert chunk.name == "search"
        assert chunk.args == '{"query":'

    def test_round_trip(self) -> None:
        chunk = ToolCallChunk(
            id="call_123", name="search", args='{"q": "test"}', type="tool_call"
        )
        data = chunk.model_dump()
        restored = ToolCallChunk.model_validate(data)
        assert restored.id == "call_123"
        assert restored.args == '{"q": "test"}'
        assert restored.type == "tool_call"

    def test_serialization_emits_args(self) -> None:
        """Wire format uses 'args' not 'arguments'."""
        chunk = ToolCallChunk(
            id="call_1",
            name="search",
            args='{"q": "test"}',
            index=0,
        )
        data = chunk.model_dump(exclude_none=True)
        assert "args" in data, f"Expected 'args' key in serialized output, got: {data}"
        assert "arguments" not in data
        assert data["args"] == '{"q": "test"}'


class TestLLMStreamChunk:
    def test_construction_empty(self) -> None:
        """Empty chunk is valid - all fields optional."""
        chunk = LLMStreamChunk()
        assert chunk.content is None
        assert chunk.tool_calls is None
        assert chunk.usage is None

    def test_content_chunk(self) -> None:
        """Content-only chunk (most common during streaming)."""
        chunk = LLMStreamChunk(content="Hello")
        assert chunk.content == "Hello"
        assert chunk.tool_calls is None

    def test_tool_call_chunk(self) -> None:
        """Chunk with partial tool calls."""
        tool_chunk = ToolCallChunk(id="call_1", name="search")
        chunk = LLMStreamChunk(tool_calls=[tool_chunk])
        assert chunk.tool_calls is not None
        assert len(chunk.tool_calls) == 1
        assert chunk.tool_calls[0].name == "search"

    def test_usage_chunk(self) -> None:
        """Final chunk with usage metadata."""
        chunk = LLMStreamChunk(
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )
        assert chunk.usage is not None
        assert chunk.usage["total_tokens"] == 30

    def test_message_metadata_chunk(self) -> None:
        chunk = LLMStreamChunk(
            id="run-1",
            name="assistant",
            additional_kwargs={"refusal": None},
            response_metadata={"finish_reason": "stop"},
        )
        assert chunk.id == "run-1"
        assert chunk.name == "assistant"
        assert chunk.additional_kwargs == {"refusal": None}
        assert chunk.response_metadata == {"finish_reason": "stop"}

    def test_round_trip(self) -> None:
        chunk = LLMStreamChunk(
            content="Hi",
            tool_calls=[ToolCallChunk(id="c1", name="fn")],
            usage={"total": 5},
        )
        data = chunk.model_dump()
        restored = LLMStreamChunk.model_validate(data)
        assert restored.content == "Hi"
        assert restored.tool_calls is not None
        assert restored.tool_calls[0].name == "fn"
        assert restored.usage == {"total": 5}


class TestMessageMultimodal:
    def test_string_content(self) -> None:
        """String content works (backward compatible)."""
        m = Message(role="user", content="Hello")
        assert m.content == "Hello"

    def test_content_blocks(self) -> None:
        """List of ContentBlocks works."""
        m = Message(
            role="user",
            content=[
                TextBlock(text="What's in this image?"),
                ImageBlock(url="https://example.com/img.png"),
            ],
        )
        assert isinstance(m.content, list)
        assert len(m.content) == 2
        assert isinstance(m.content[0], TextBlock)
        assert isinstance(m.content[1], ImageBlock)

    def test_mixed_content_round_trip(self) -> None:
        """Message with ContentBlocks survives round-trip."""
        m = Message(
            role="user",
            content=[
                TextBlock(text="Check this:"),
                DocumentBlock(url="https://example.com/doc.pdf", filename="doc.pdf"),
            ],
        )
        data = m.model_dump()
        restored = Message.model_validate(data)

        assert isinstance(restored.content, list)
        assert len(restored.content) == 2
        # Pydantic correctly deserializes to ContentBlock types
        assert isinstance(restored.content[0], TextBlock)
        assert restored.content[0].text == "Check this:"
        assert isinstance(restored.content[1], DocumentBlock)
        assert restored.content[1].url == "https://example.com/doc.pdf"

    def test_empty_content_list(self) -> None:
        """Empty list is valid content."""
        m = Message(role="user", content=[])
        assert m.content == []

    def test_single_text_block(self) -> None:
        """Single TextBlock in list is valid."""
        m = Message(role="user", content=[TextBlock(text="Just text")])
        assert len(m.content) == 1
        assert m.content[0].text == "Just text"


class TestRequestContext:
    def test_defaults_all_none(self):
        ctx = RequestContext()
        assert ctx.execution_id is None
        assert ctx.session_id is None
        assert ctx.user_id is None
        assert ctx.request_headers is None

    def test_accepts_all_fields(self):
        ctx = RequestContext(
            execution_id="inv-abc",
            session_id="sess-1",
            user_id="user-1",
            request_headers={"x-custom": "value"},
        )
        assert ctx.execution_id == "inv-abc"
        assert ctx.session_id == "sess-1"
        assert ctx.user_id == "user-1"
        assert ctx.request_headers == {"x-custom": "value"}
