"""Unit tests for LLM routing through tool executor pods."""

import json
import logging
from collections.abc import Mapping
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_engine_sdk.models import (
    LLMInvocationOptions,
    LLMStreamChunk,
    LLMTokenUsage,
    LLMToolCall,
    Message,
    ToolCallChunk,
)

from agent_engine_runner_shared import hooks
from agent_engine_runner_shared.models import (
    InvokeLLMRequestArguments,
    LLMPodInvokeRequest,
    LLMPodInvokeResponse,
    LLMPodStreamEvent,
    LLMResult,
    accumulate_stream_usage,
    add_token_usage,
    coerce_token_usage,
    merge_token_usage,
)
from agent_engine_runner_shared.server.tool import _deserialize_messages

# =============================================================================
# Model Serialization Tests
# =============================================================================


class TestLLMPodInvokeRequestSerialization:
    """Tests for LLMPodInvokeRequest serialization/deserialization."""

    def test_basic_serialization(self):
        """Request serializes to dict and back."""
        request = LLMPodInvokeRequest(
            execution_id="exec-123",
            arguments=InvokeLLMRequestArguments(
                model="gpt-4o-mini",
                llm_id="primary",
                messages=[{"role": "user", "content": "Hello"}],
            ),
        )
        data = request.model_dump()
        restored = LLMPodInvokeRequest(**data)
        assert restored.execution_id == "exec-123"
        assert restored.arguments.model == "gpt-4o-mini"
        assert restored.arguments.messages == [Message(role="user", content="Hello")]
        assert restored.arguments.llm_id == "primary"
        assert restored.arguments.stop_sequences is None
        assert restored.arguments.tools is None
        assert restored.arguments.options is None

    def test_full_serialization(self):
        """Request with all fields serializes correctly."""
        request = LLMPodInvokeRequest(
            execution_id="exec-456",
            arguments=InvokeLLMRequestArguments(
                model="gemini-2.5-flash",
                llm_id="primary",
                messages=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "What is 2+2?"},
                ],
                stop_sequences=["\n", "END"],
                tools=[{"name": "calculator", "parameters": {"type": "object"}}],
                options=LLMInvocationOptions(max_tokens=100),
            ),
        )
        data = request.model_dump()
        restored = LLMPodInvokeRequest(**data)
        assert restored.arguments.stop_sequences == ["\n", "END"]
        assert restored.arguments.tools == [
            {"name": "calculator", "parameters": {"type": "object"}}
        ]
        assert restored.arguments.options == LLMInvocationOptions(max_tokens=100)

    def test_json_roundtrip(self):
        """Request survives JSON serialization roundtrip."""
        request = LLMPodInvokeRequest(
            execution_id="exec-789",
            arguments=InvokeLLMRequestArguments(
                model="gpt-4o",
                llm_id="primary",
                messages=[{"role": "user", "content": "Hi"}],
            ),
        )
        json_str = request.model_dump_json()
        restored = LLMPodInvokeRequest.model_validate_json(json_str)
        assert restored == request

    def test_flat_payload_is_lifted_into_typed_arguments(self):
        """Older flat /invoke_llm bodies still validate into the typed envelope."""
        restored = LLMPodInvokeRequest.model_validate(
            {
                "execution_id": "exec-flat",
                "model": "gpt-4o",
                "llm_id": "primary",
                "messages": [{"role": "user", "content": "Hello"}],
                "stop": ["END"],
                "kwargs": {"max_tokens": 32},
            }
        )

        assert restored.execution_id == "exec-flat"
        assert restored.arguments.model == "gpt-4o"
        assert restored.arguments.messages == [Message(role="user", content="Hello")]
        assert restored.arguments.stop_sequences == ["END"]
        assert restored.arguments.options == LLMInvocationOptions(max_tokens=32)

    def test_flat_payload_step_number_stays_out_of_arguments(self):
        """step_number is routing metadata: preserved top-level on a flat
        body, never poured into the LLM arguments bag."""
        restored = LLMPodInvokeRequest.model_validate(
            {
                "execution_id": "exec-flat",
                "platform_trace_id": "0123456789abcdef0123456789abcdef",
                "step_number": 3,
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )

        assert restored.platform_trace_id == "0123456789abcdef0123456789abcdef"
        assert restored.step_number == 3
        assert "step_number" not in restored.arguments.model_dump()

    def test_llm_id_round_trips(self):
        """llm_id field survives JSON serialization."""
        request = LLMPodInvokeRequest(
            execution_id="exec-1",
            arguments=InvokeLLMRequestArguments(
                model="gpt-4o",
                llm_id="primary",
                messages=[{"role": "user", "content": "Hi"}],
            ),
        )
        restored = LLMPodInvokeRequest.model_validate_json(request.model_dump_json())
        assert restored.arguments.llm_id == "primary"

    def test_llm_id_defaults_to_sentinel(self):
        """Missing llm_id defaults to '__default__' for backward compat."""
        args = InvokeLLMRequestArguments(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert args.llm_id == "__default__"
        # Verify the default survives JSON round-trip (the actual wire-compat scenario)
        restored = InvokeLLMRequestArguments.model_validate_json(args.model_dump_json())
        assert restored.llm_id == "__default__"


class TestLLMPodInvokeResponseSerialization:
    """Tests for LLMPodInvokeResponse serialization."""

    def test_success_response(self):
        """Success response serializes correctly."""
        response = LLMPodInvokeResponse(
            status="success",
            result={"content": "Hello!", "tool_calls": []},
            pod_name="pod-abc",
            duration_ms=150.5,
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        data = response.model_dump()
        assert data["status"] == "success"
        assert data["result"]["content"] == "Hello!"
        assert data["pod_name"] == "pod-abc"
        assert data["usage"]["total_tokens"] == 15

    def test_error_response(self):
        """Error response serializes correctly."""
        response = LLMPodInvokeResponse(
            status="error",
            error="Rate limit exceeded",
            pod_name="pod-xyz",
            duration_ms=50.0,
        )
        data = response.model_dump()
        assert data["status"] == "error"
        assert data["error"] == "Rate limit exceeded"
        assert data["result"] is None


class TestLLMPodStreamEventSerialization:
    """Tests for typed tool-pod SSE events."""

    def test_chunk_event_json_roundtrip(self):
        event = LLMPodStreamEvent(
            content="Hel",
            tool_call_chunks=[ToolCallChunk(id="call-1", name="lookup", args='{"q"')],
            tool_calls=[],
        )

        restored = LLMPodStreamEvent.model_validate_json(event.model_dump_json())

        assert restored.content == "Hel"
        assert restored.tool_call_chunks is not None
        assert restored.tool_call_chunks[0].name == "lookup"
        assert restored.tool_calls == []

    def test_done_event_json_roundtrip(self):
        event = LLMPodStreamEvent(
            done=True,
            pod_name="tool-pod-1",
            duration_ms=23.5,
            usage={"input_tokens": 10, "output_tokens": 4},
        )

        restored = LLMPodStreamEvent.model_validate_json(event.model_dump_json())

        assert restored.done is True
        assert restored.pod_name == "tool-pod-1"
        assert restored.duration_ms == 23.5
        assert restored.usage == {
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
        }

    def test_interrupted_event_json_roundtrip(self):
        event = LLMPodStreamEvent(interrupted=True, pod_name="tool-pod-1", duration_ms=12.0)

        restored = LLMPodStreamEvent.model_validate_json(event.model_dump_json())

        assert restored.interrupted is True
        assert restored.done is None
        assert restored.error is None
        assert restored.pod_name == "tool-pod-1"

    def test_message_metadata_event_json_roundtrip(self):
        event = LLMPodStreamEvent(
            content="final",
            id="run-1",
            name="assistant",
            additional_kwargs={"refusal": None},
            response_metadata={"finish_reason": "stop"},
        )

        restored = LLMPodStreamEvent.model_validate_json(event.model_dump_json())

        assert restored.id == "run-1"
        assert restored.name == "assistant"
        assert restored.additional_kwargs == {"refusal": None}
        assert restored.response_metadata == {"finish_reason": "stop"}


class TestSecureLLMProxyStreamCollection:
    """Tests for reconstructing final responses from stream chunks."""

    def test_absent_tool_call_index_uses_active_call(self):
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        tool_calls = SecureLLMProxy._convert_stream_chunks_to_tool_calls(
            [
                ToolCallChunk(id="call-1", name="lookup", args='{"q":'),
                ToolCallChunk(args='"weather"}'),
            ]
        )

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call-1"
        assert tool_calls[0].args == {"q": "weather"}

    def test_response_from_stream_chunks_preserves_message_metadata(self):
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        response = SecureLLMProxy.response_from_stream_chunks(
            [
                LLMStreamChunk(content="Hello"),
                LLMStreamChunk(
                    id="run-1",
                    name="assistant",
                    additional_kwargs={"refusal": None},
                    response_metadata={"finish_reason": "stop"},
                ),
            ]
        )

        assert response.content == "Hello"
        assert response.id == "run-1"
        assert response.name == "assistant"
        assert response.additional_kwargs == {"refusal": None}
        assert response.response_metadata == {"finish_reason": "stop"}

    def test_synthetic_tool_call_encoding_preserves_idless_calls_and_type(self):
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        chunks = SecureLLMProxy._convert_tool_calls_to_stream_chunks(
            [
                LLMToolCall(name="lookup", args={"q": "one"}, type="tool_call"),
                LLMToolCall(name="search", args={"q": "two"}, type="function"),
            ]
        )

        assert chunks is not None
        assert [chunk.index for chunk in chunks] == [0, 1]
        assert [chunk.type for chunk in chunks] == ["tool_call", "function"]

        response = SecureLLMProxy.response_from_stream_chunks([LLMStreamChunk(tool_calls=chunks)])

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].name == "lookup"
        assert response.tool_calls[0].args == {"q": "one"}
        assert response.tool_calls[0].type == "tool_call"
        assert response.tool_calls[1].name == "search"
        assert response.tool_calls[1].args == {"q": "two"}
        assert response.tool_calls[1].type == "function"

    def test_conflicting_tool_call_id_indices_log_warning(self, caplog):
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        caplog.set_level(logging.WARNING, logger="agent_engine_runner_shared.secure_llm_proxy")

        tool_calls = SecureLLMProxy._convert_stream_chunks_to_tool_calls(
            [
                ToolCallChunk(id="call-1", name="lookup", args='{"q": "one"}', index=0),
                ToolCallChunk(id="call-1", name="search", args='{"q": "two"}', index=1),
            ]
        )

        assert tool_calls is not None
        assert "Tool call id call-1 changed stream index from 0 to 1" in caplog.text


class TestLLMResultUsageExtraction:
    def test_extract_usage_from_response_metadata_usage(self):
        response = MagicMock()
        response.usage_metadata = None
        response.response_metadata = {"usage": {"prompt_tokens": 3, "completion_tokens": 2}}
        response.metadata = {}

        usage = LLMResult.extract_usage(response)

        assert usage is not None
        assert usage.prompt_tokens == 3
        assert usage.completion_tokens == 2
        assert usage.total_tokens == 5

    def test_extract_usage_from_response_metadata_token_usage(self):
        response = MagicMock()
        response.usage_metadata = None
        response.response_metadata = {"token_usage": {"input_tokens": 4, "output_tokens": 6}}
        response.metadata = {}

        usage = LLMResult.extract_usage(response)

        assert usage is not None
        assert usage.input_tokens == 4
        assert usage.output_tokens == 6
        assert usage.total_tokens == 10

    def test_extract_usage_from_anthropic_usage_object(self):
        response = MagicMock()
        response.usage_metadata = None
        response.response_metadata = {
            "usage": SimpleNamespace(input_tokens=1048, output_tokens=1222)
        }
        response.metadata = {}
        response.usage = None

        usage = LLMResult.extract_usage(response)

        assert usage is not None
        assert usage.input_tokens == 1048
        assert usage.output_tokens == 1222
        assert usage.total_tokens == 2270

    def test_extract_usage_from_non_dict_mapping(self):
        class TokenMapping(Mapping):
            def __init__(self, data):
                self._data = data

            def __getitem__(self, key):
                return self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        response = MagicMock()
        response.usage_metadata = TokenMapping({"input_tokens": 10, "output_tokens": 5})
        response.response_metadata = {}
        response.metadata = {}
        response.usage = None

        usage = LLMResult.extract_usage(response)

        assert usage is not None
        assert usage.input_tokens == 10
        assert usage.output_tokens == 5

    def test_extract_usage_from_usage_attribute(self):
        response = MagicMock()
        response.usage_metadata = None
        response.response_metadata = {}
        response.metadata = {}
        response.usage = SimpleNamespace(input_tokens=7, output_tokens=3)

        usage = LLMResult.extract_usage(response)

        assert usage is not None
        assert usage.input_tokens == 7
        assert usage.output_tokens == 3

    def test_malformed_usage_returns_none_without_raising(self):
        response = MagicMock()
        response.usage_metadata = "n/a"
        response.response_metadata = {"usage": "n/a"}
        response.metadata = {"usage": "n/a"}
        response.usage = "n/a"

        assert LLMResult.extract_usage(response) is None
        assert coerce_token_usage("n/a") is None
        assert coerce_token_usage({"input_tokens": float("nan"), "output_tokens": 2}) is not None
        assert (
            coerce_token_usage({"input_tokens": float("nan"), "output_tokens": 2}).output_tokens
            == 2
        )

    def test_from_response_coerces_usage_object_in_metadata(self):
        class Response:
            content = "ok"
            tool_calls = []
            usage_metadata = None
            response_metadata = {
                "usage": SimpleNamespace(input_tokens=1048, output_tokens=1222),
                "model_name": "claude-sonnet-4-6",
            }
            additional_kwargs = None
            metadata = {}
            id = "msg-1"
            name = None
            usage = None

        result = LLMResult.from_response(Response())
        assert result.content == "ok"
        assert result.usage is not None
        assert result.usage.input_tokens == 1048
        assert result.usage.output_tokens == 1222
        assert result.response_metadata is not None
        assert result.response_metadata["usage"]["input_tokens"] == 1048
        assert result.response_metadata["model_name"] == "claude-sonnet-4-6"

    def test_from_response_drops_nonfinite_and_raising_usage(self):
        class RaisingUsage:
            @property
            def input_tokens(self):
                raise RuntimeError("nope")

            output_tokens = 4

        class NanUsage:
            input_tokens = float("nan")
            output_tokens = float("inf")

        class Response:
            content = "still works"
            tool_calls = []
            usage_metadata = None
            response_metadata = {"usage": RaisingUsage(), "token_usage": NanUsage()}
            additional_kwargs = {"blob": object()}
            metadata = {"usage": {"input_tokens": 3, "secret": object()}}
            id = None
            name = None
            usage = None

        result = LLMResult.from_response(Response())
        assert result.content == "still works"
        assert "secret" not in result.metadata.get("usage", {})


class TestAddTokenUsage:
    def test_adds_zero_filled_langchain_deltas(self):
        first = LLMTokenUsage(input_tokens=18, output_tokens=1, total_tokens=19)
        second = LLMTokenUsage(input_tokens=0, output_tokens=4, total_tokens=4)
        added = add_token_usage(first, second)
        assert added is not None
        assert added.input_tokens == 18
        assert added.output_tokens == 5
        assert added.total_tokens == 23


class TestAccumulateStreamUsage:
    def test_takes_last_cumulative_snapshot(self):
        first = LLMTokenUsage(input_tokens=10, output_tokens=1, total_tokens=11)
        second = LLMTokenUsage(input_tokens=10, output_tokens=2, total_tokens=12)
        accumulated = accumulate_stream_usage(first, second)
        assert accumulated is not None
        assert accumulated.input_tokens == 10
        assert accumulated.output_tokens == 2
        assert accumulated.total_tokens == 12

    def test_adds_zero_filled_langchain_deltas(self):
        first = LLMTokenUsage(input_tokens=18, output_tokens=1, total_tokens=19)
        second = LLMTokenUsage(input_tokens=0, output_tokens=4, total_tokens=4)
        accumulated = accumulate_stream_usage(first, second)
        assert accumulated is not None
        assert accumulated.input_tokens == 18
        assert accumulated.output_tokens == 5
        assert accumulated.total_tokens == 23

    def test_merges_split_input_and_output_chunks(self):
        first = LLMTokenUsage(input_tokens=1048)
        second = LLMTokenUsage(output_tokens=1222)
        accumulated = accumulate_stream_usage(first, second)
        assert accumulated is not None
        assert accumulated.input_tokens == 1048
        assert accumulated.output_tokens == 1222
        assert accumulated.total_tokens == 2270


class TestMergeTokenUsage:
    def test_merges_split_input_and_output_chunks(self):
        first = LLMTokenUsage(input_tokens=1048)
        second = LLMTokenUsage(output_tokens=1222)
        merged = merge_token_usage(first, second)
        assert merged is not None
        assert merged.input_tokens == 1048
        assert merged.output_tokens == 1222
        assert merged.total_tokens == 2270

    def test_incoming_none_keeps_existing(self):
        first = LLMTokenUsage(input_tokens=10, output_tokens=2)
        assert merge_token_usage(first, None) is first

    def test_existing_none_takes_incoming(self):
        second = LLMTokenUsage(input_tokens=3, output_tokens=4)
        assert merge_token_usage(None, second) is second

    def test_preserves_explicit_total_tokens(self):
        first = LLMTokenUsage(input_tokens=1, output_tokens=1, total_tokens=99)
        merged = merge_token_usage(first, first)
        assert merged is not None
        assert merged.total_tokens == 99
        assert merged.input_tokens == 1
        assert merged.output_tokens == 1

    def test_zero_filled_incoming_does_not_add(self):
        """Snapshot merge last-wins zeros; adapters must add LangChain deltas."""
        first = LLMTokenUsage(input_tokens=18, output_tokens=1, total_tokens=19)
        second = LLMTokenUsage(input_tokens=0, output_tokens=4, total_tokens=4)
        merged = merge_token_usage(first, second)
        assert merged is not None
        assert merged.input_tokens == 0
        assert merged.output_tokens == 4


# =============================================================================
# Message Deserialization Tests
# =============================================================================


class TestMessageDeserialization:
    """Tests for _deserialize_messages helper.

    _deserialize_messages returns sdk-core Message objects
    instead of LangChain BaseMessage objects.
    """

    def test_human_message(self):
        """Deserializes human message."""
        from agent_engine_sdk import Message

        messages = _deserialize_messages([Message(role="user", content="Hello")])
        assert len(messages) == 1
        assert isinstance(messages[0], Message)
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"

    def test_ai_message(self):
        """Deserializes AI message."""
        from agent_engine_sdk import Message

        messages = _deserialize_messages([Message(role="assistant", content="Hi there")])
        assert len(messages) == 1
        assert isinstance(messages[0], Message)
        assert messages[0].role == "assistant"
        assert messages[0].content == "Hi there"

    def test_system_message(self):
        """Deserializes system message."""
        messages = _deserialize_messages([Message(role="system", content="Be helpful.")])
        assert len(messages) == 1
        assert messages[0].role == "system"

    def test_tool_message(self):
        """Deserializes tool message with tool_call_id."""
        messages = _deserialize_messages(
            [Message(role="tool", content="result: 42", tool_call_id="call-123")]
        )
        assert len(messages) == 1
        assert messages[0].role == "tool"
        assert messages[0].tool_call_id == "call-123"

    def test_tool_message_with_name(self):
        """Deserializes tool message preserving the name field."""
        messages = _deserialize_messages(
            [
                Message(
                    role="tool",
                    content="result: 42",
                    tool_call_id="call-123",
                    name="calculator",
                )
            ]
        )
        assert len(messages) == 1
        assert messages[0].role == "tool"
        assert messages[0].tool_call_id == "call-123"
        assert messages[0].name == "calculator"

    def test_ai_message_with_tool_calls(self):
        """Deserializes AI message with tool_calls."""
        tool_calls = [{"name": "calc", "args": {"x": 1}, "id": "tc-1", "type": "tool_call"}]
        messages = _deserialize_messages(
            [Message(role="assistant", content="", tool_calls=tool_calls)]
        )
        assert len(messages) == 1
        assert messages[0].role == "assistant"
        assert messages[0].tool_calls == tool_calls

    def test_multiple_messages(self):
        """Deserializes a conversation with multiple message types."""
        messages = _deserialize_messages(
            [
                Message(role="system", content="You are helpful."),
                Message(role="user", content="What is 2+2?"),
                Message(role="assistant", content="4"),
            ]
        )
        assert len(messages) == 3
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"

    def test_returns_validated_message_objects(self):
        """Validated messages are returned without relying on dict compatibility."""
        original = Message(role="user", content="test")
        messages = _deserialize_messages([original])
        assert messages == [original]


# =============================================================================
# Streaming Retry Tests (tool pod _handle_invoke_llm_stream)
# =============================================================================


def _parse_sse_events(raw_events: list[str]) -> list[dict]:
    """Parse SSE 'data: {...}' strings into dicts."""
    results = []
    for event in raw_events:
        if event.startswith("data: "):
            results.append(json.loads(event[len("data: ") :].strip()))
    return results


def _make_tool_server():
    """Create a ToolServer with a mocked runtime for testing."""
    import asyncio

    from agent_engine_runner_shared.server.tool import ToolServer

    mock_runtime = MagicMock()
    server = ToolServer.__new__(ToolServer)
    server.runtime = mock_runtime
    # __new__ skips __init__ (BaseServer setup needs a real runtime), so seed
    # the gate the LLM handlers acquire, the restriction-mode flag
    # and the lazy entrypoint-load state (_ensure_llm_registry_loaded());
    # mark it already-loaded so these LLM-routing tests don't also need a
    # working _graph_builder.
    server._execute_gate = asyncio.Lock()
    server._restriction_disabled = False
    server._llm_registry_lock = asyncio.Lock()
    server._llm_registry_loaded = True
    return server


class TestStreamRetryDuringIteration:
    """Tests that _handle_invoke_llm_stream retries errors raised during iteration.

    _create_llm_for_pod returns a BaseLLM adapter whose .astream()
    yields LLMStreamChunk objects. These tests mock at the adapter level.
    """

    def _make_stream_chunk(self, content="", tool_calls=None, usage=None):
        """Create an LLMStreamChunk for testing."""
        from agent_engine_sdk.models import LLMStreamChunk

        return LLMStreamChunk(content=content, tool_calls=tool_calls, usage=usage)

    @pytest.mark.asyncio
    async def test_stream_retries_on_429_during_iteration(self):
        """First iteration raises 429; retry succeeds and yields chunks + summary."""
        server = _make_tool_server()
        mock_adapter = MagicMock()
        chunk = self._make_stream_chunk(content="Hello")

        call_count = 0

        async def fake_astream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("429 Too Many Requests")
            yield chunk

        mock_adapter.astream = fake_astream

        with (
            patch.object(server, "_create_llm_for_pod", return_value=mock_adapter),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            request = LLMPodInvokeRequest(
                execution_id="exec-1",
                arguments=InvokeLLMRequestArguments(
                    model="test-model",
                    llm_id="primary",
                    messages=[{"role": "user", "content": "Hi"}],
                ),
            )
            events = []
            async for event in server._handle_invoke_llm_stream(request):
                events.append(event)

        parsed = _parse_sse_events(events)
        assert any(e.get("content") == "Hello" for e in parsed)
        assert any(e.get("done") is True for e in parsed)
        assert not any("error" in e for e in parsed)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_stream_no_retry_after_chunks_yielded(self):
        """Error after first chunk is yielded — no retry, error SSE emitted."""
        server = _make_tool_server()
        mock_adapter = MagicMock()
        chunk = self._make_stream_chunk(content="partial")

        async def fake_astream(*args, **kwargs):
            yield chunk
            raise Exception("429 Too Many Requests")

        mock_adapter.astream = fake_astream

        with (
            patch.object(server, "_create_llm_for_pod", return_value=mock_adapter),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            request = LLMPodInvokeRequest(
                execution_id="exec-1",
                arguments=InvokeLLMRequestArguments(
                    model="test-model",
                    llm_id="primary",
                    messages=[{"role": "user", "content": "Hi"}],
                ),
            )
            events = []
            async for event in server._handle_invoke_llm_stream(request):
                events.append(event)

        parsed = _parse_sse_events(events)
        assert any(e.get("content") == "partial" for e in parsed)
        assert any("error" in e for e in parsed)
        assert not any(e.get("done") is True for e in parsed)

    @pytest.mark.asyncio
    async def test_stream_chunk_with_content_and_usage_keeps_both(self):
        """A chunk carrying content and usage emits the content event and final usage."""
        from agent_engine_sdk.models import LLMStreamChunk

        from agent_engine_runner_shared.context import get_current_trace_id

        server = _make_tool_server()
        mock_adapter = MagicMock()
        seen: list[str | None] = []

        async def fake_astream(*args, **kwargs):
            seen.append(get_current_trace_id())
            yield LLMStreamChunk(
                content="Hello",
                usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            )

        mock_adapter.astream = fake_astream

        with patch.object(server, "_create_llm_for_pod", return_value=mock_adapter):
            request = LLMPodInvokeRequest(
                execution_id="exec-1",
                platform_trace_id="0123456789abcdef0123456789abcdef",
                arguments=InvokeLLMRequestArguments(
                    model="test-model",
                    llm_id="primary",
                    messages=[{"role": "user", "content": "Hi"}],
                ),
            )
            events = []
            async for event in server._handle_invoke_llm_stream(request):
                events.append(event)

        parsed = _parse_sse_events(events)
        assert any(e.get("content") == "Hello" for e in parsed)
        done_events = [e for e in parsed if e.get("done") is True]
        assert len(done_events) == 1
        assert done_events[0]["usage"]["total_tokens"] == 5
        assert seen == [request.platform_trace_id]
        assert get_current_trace_id() is None

    @pytest.mark.asyncio
    async def test_stream_merges_split_input_and_output_usage(self):
        """Anthropic-style split: input on an early chunk, output on a later one."""
        from agent_engine_sdk.models import LLMStreamChunk

        server = _make_tool_server()
        mock_adapter = MagicMock()

        async def fake_astream(*args, **kwargs):
            yield LLMStreamChunk(content="Hello", usage={"input_tokens": 1048})
            yield LLMStreamChunk(usage={"output_tokens": 1222})

        mock_adapter.astream = fake_astream

        with patch.object(server, "_create_llm_for_pod", return_value=mock_adapter):
            request = LLMPodInvokeRequest(
                execution_id="exec-1",
                arguments=InvokeLLMRequestArguments(
                    model="test-model",
                    llm_id="primary",
                    messages=[{"role": "user", "content": "Hi"}],
                ),
            )
            events = []
            async for event in server._handle_invoke_llm_stream(request):
                events.append(event)

        parsed = _parse_sse_events(events)
        done_events = [e for e in parsed if e.get("done") is True]
        assert len(done_events) == 1
        usage = done_events[0]["usage"]
        assert usage["input_tokens"] == 1048
        assert usage["output_tokens"] == 1222
        assert usage["total_tokens"] == 2270

    @pytest.mark.asyncio
    async def test_stream_preserves_explicit_total_on_done_event(self):
        from agent_engine_sdk.models import LLMStreamChunk

        server = _make_tool_server()
        mock_adapter = MagicMock()

        async def fake_astream(*args, **kwargs):
            yield LLMStreamChunk(
                content="Hello",
                usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 99},
            )

        mock_adapter.astream = fake_astream

        with patch.object(server, "_create_llm_for_pod", return_value=mock_adapter):
            request = LLMPodInvokeRequest(
                execution_id="exec-1",
                arguments=InvokeLLMRequestArguments(
                    model="test-model",
                    llm_id="primary",
                    messages=[{"role": "user", "content": "Hi"}],
                ),
            )
            events = []
            async for event in server._handle_invoke_llm_stream(request):
                events.append(event)

        parsed = _parse_sse_events(events)
        done_events = [e for e in parsed if e.get("done") is True]
        assert len(done_events) == 1
        assert done_events[0]["usage"]["input_tokens"] == 1
        assert done_events[0]["usage"]["output_tokens"] == 1
        assert done_events[0]["usage"]["total_tokens"] == 99

    @pytest.mark.asyncio
    async def test_stream_done_usage_keeps_adapter_cumulative_snapshot(self):
        from agent_engine_sdk.models import LLMStreamChunk

        server = _make_tool_server()
        mock_adapter = MagicMock()

        async def fake_astream(*args, **kwargs):
            yield LLMStreamChunk(
                content="Hel",
                usage={"input_tokens": 18, "output_tokens": 1, "total_tokens": 19},
            )
            yield LLMStreamChunk(
                content="lo",
                usage={"input_tokens": 18, "output_tokens": 5, "total_tokens": 23},
            )

        mock_adapter.astream = fake_astream

        with patch.object(server, "_create_llm_for_pod", return_value=mock_adapter):
            request = LLMPodInvokeRequest(
                execution_id="exec-1",
                arguments=InvokeLLMRequestArguments(
                    model="test-model",
                    llm_id="primary",
                    messages=[{"role": "user", "content": "Hi"}],
                ),
            )
            events = []
            async for event in server._handle_invoke_llm_stream(request):
                events.append(event)

        parsed = _parse_sse_events(events)
        done_events = [e for e in parsed if e.get("done") is True]
        assert done_events[0]["usage"]["input_tokens"] == 18
        assert done_events[0]["usage"]["output_tokens"] == 5
        assert done_events[0]["usage"]["total_tokens"] == 23

    @pytest.mark.asyncio
    async def test_stream_exhausts_retries(self):
        """Every attempt raises 429 — error SSE after all retries exhausted."""
        server = _make_tool_server()
        mock_adapter = MagicMock()
        call_count = 0

        async def fake_astream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("429 Too Many Requests")
            # Make this an async generator
            yield  # pragma: no cover

        mock_adapter.astream = fake_astream

        with (
            patch.object(server, "_create_llm_for_pod", return_value=mock_adapter),
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch("agent_engine_runner_shared.utils.LLM_MAX_RETRIES", 3),
        ):
            request = LLMPodInvokeRequest(
                execution_id="exec-1",
                arguments=InvokeLLMRequestArguments(
                    model="test-model",
                    llm_id="primary",
                    messages=[{"role": "user", "content": "Hi"}],
                ),
            )
            events = []
            async for event in server._handle_invoke_llm_stream(request):
                events.append(event)

        parsed = _parse_sse_events(events)
        assert len(parsed) == 1
        assert "error" in parsed[0]
        assert "429" in parsed[0]["error"]
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_stream_normalizes_langchain_style_chunk_objects(self):
        """Raw LangChain-like chunks are normalized into sdk-core SSE payloads."""
        server = _make_tool_server()
        mock_adapter = MagicMock()

        class RawChunk:
            def __init__(self):
                self.content = "Hello"
                self.tool_call_chunks = [
                    {
                        "id": "call-1",
                        "name": "lookup",
                        "args": '{"q":"weather"}',
                        "type": "tool_call_chunk",
                        "index": 0,
                    }
                ]
                self.id = "msg-1"
                self.name = "assistant"
                self.response_metadata = {"finish_reason": "tool_calls"}
                self.additional_kwargs = {"provider": "test"}
                self.usage = None

        async def fake_astream(*args, **kwargs):
            yield RawChunk()

        mock_adapter.astream = fake_astream

        with patch.object(server, "_create_llm_for_pod", return_value=mock_adapter):
            request = LLMPodInvokeRequest(
                execution_id="exec-1",
                arguments=InvokeLLMRequestArguments(
                    model="test-model",
                    llm_id="primary",
                    messages=[{"role": "user", "content": "Hi"}],
                ),
            )
            events = []
            async for event in server._handle_invoke_llm_stream(request):
                events.append(event)

        parsed = _parse_sse_events(events)
        assert parsed[0]["content"] == "Hello"
        assert parsed[0]["tool_call_chunks"] == [
            {
                "id": "call-1",
                "name": "lookup",
                "args": '{"q":"weather"}',
                "type": "tool_call_chunk",
                "index": 0,
            }
        ]
        assert parsed[0]["id"] == "msg-1"
        assert parsed[0]["name"] == "assistant"
        assert parsed[0]["response_metadata"] == {"finish_reason": "tool_calls"}
        assert parsed[0]["additional_kwargs"] == {"provider": "test"}
        assert parsed[1]["done"] is True

    @pytest.mark.asyncio
    async def test_stream_normalizes_langchain_tool_call_chunk_dict_args(self):
        """Structured tool_call_chunk args are JSON-encoded before SSE emission."""
        server = _make_tool_server()
        mock_adapter = MagicMock()

        class RawChunk:
            def __init__(self):
                self.content = ""
                self.tool_call_chunks = [
                    {
                        "id": "call-1",
                        "name": "lookup",
                        "args": {"q": "weather"},
                        "type": "tool_call_chunk",
                        "index": 0,
                    }
                ]
                self.id = None
                self.name = None
                self.response_metadata = None
                self.additional_kwargs = None
                self.usage = None

        async def fake_astream(*args, **kwargs):
            yield RawChunk()

        mock_adapter.astream = fake_astream

        with patch.object(server, "_create_llm_for_pod", return_value=mock_adapter):
            request = LLMPodInvokeRequest(
                execution_id="exec-1",
                arguments=InvokeLLMRequestArguments(
                    model="test-model",
                    llm_id="primary",
                    messages=[{"role": "user", "content": "Hi"}],
                ),
            )
            events = []
            async for event in server._handle_invoke_llm_stream(request):
                events.append(event)

        parsed = _parse_sse_events(events)
        assert parsed[0]["tool_call_chunks"] == [
            {
                "id": "call-1",
                "name": "lookup",
                "args": json.dumps({"q": "weather"}),
                "type": "tool_call_chunk",
                "index": 0,
            }
        ]


class TestStreamLLMChunks:
    """Direct tests for tool-pod retry behavior in _stream_llm_chunks."""

    def _make_request(self):
        return LLMPodInvokeRequest(
            execution_id="exec-1",
            arguments=InvokeLLMRequestArguments(
                model="test-model",
                llm_id="primary",
                messages=[{"role": "user", "content": "Hi"}],
            ),
        )

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self):
        server = _make_tool_server()
        mock_adapter = MagicMock()
        call_count = 0

        async def fake_astream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("429 Too Many Requests")
            yield LLMStreamChunk(content="retry success")

        mock_adapter.astream = fake_astream

        with (
            patch.object(server, "_create_llm_for_pod", return_value=mock_adapter),
            patch("agent_engine_runner_shared.server.tool.is_retryable_error", return_value=True),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            chunks = [chunk async for chunk in server._stream_llm_chunks(self._make_request())]

        assert [chunk.content for chunk in chunks] == ["retry success"]
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_raises_last_error(self):
        server = _make_tool_server()
        mock_adapter = MagicMock()
        call_count = 0

        async def fake_astream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("429 Too Many Requests")
            yield  # pragma: no cover

        mock_adapter.astream = fake_astream

        with (
            patch.object(server, "_create_llm_for_pod", return_value=mock_adapter),
            patch("agent_engine_runner_shared.server.tool.is_retryable_error", return_value=True),
            patch("agent_engine_runner_shared.server.tool.LLM_MAX_RETRIES", 3),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(Exception, match="429 Too Many Requests"):
                _ = [chunk async for chunk in server._stream_llm_chunks(self._make_request())]

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_after_first_chunk_yielded(self):
        server = _make_tool_server()
        mock_adapter = MagicMock()
        call_count = 0

        async def fake_astream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            yield LLMStreamChunk(content="partial")
            raise Exception("429 Too Many Requests")

        mock_adapter.astream = fake_astream

        chunks: list[LLMStreamChunk] = []
        with (
            patch.object(server, "_create_llm_for_pod", return_value=mock_adapter),
            patch("agent_engine_runner_shared.server.tool.is_retryable_error", return_value=True),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(Exception, match="429 Too Many Requests"):
                async for chunk in server._stream_llm_chunks(self._make_request()):
                    chunks.append(chunk)

        assert [chunk.content for chunk in chunks] == ["partial"]
        assert call_count == 1


class TestHandleInvokeLLM:
    """Tests for _handle_invoke_llm (non-streaming invoke path)."""

    def _make_request(self, **overrides):
        defaults = {
            "model": "gpt-4o",
            "llm_id": "primary",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": None,
            "stop_sequences": None,
            "options": None,
        }
        execution_id = overrides.pop("execution_id", "exec-test-1")
        platform_trace_id = overrides.pop("platform_trace_id", None)
        defaults.update(overrides)
        return LLMPodInvokeRequest(
            execution_id=execution_id,
            platform_trace_id=platform_trace_id,
            arguments=InvokeLLMRequestArguments(**defaults),
        )

    @pytest.mark.asyncio
    async def test_success_returns_content_and_usage(self):
        """Successful invoke returns status=success with content, tool_calls, usage."""
        from agent_engine_sdk.models import LLMStreamChunk, ToolCallChunk

        from agent_engine_runner_shared.context import get_current_trace_id

        server = _make_tool_server()
        mock_adapter = AsyncMock()
        seen: list[str | None] = []

        async def fake_astream(*args, **kwargs):
            seen.append(get_current_trace_id())
            yield LLMStreamChunk(content="Hello!")
            yield LLMStreamChunk(
                tool_calls=[ToolCallChunk(id="tc1", name="search", args='{"q": "test"}', index=0)]
            )
            yield LLMStreamChunk(usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

        mock_adapter.astream = fake_astream

        request = self._make_request(platform_trace_id="0123456789abcdef0123456789abcdef")
        with patch.object(server, "_create_llm_for_pod", return_value=mock_adapter):
            response = await server._handle_invoke_llm(request)

        assert response.status == "success"
        assert response.result["content"] == "Hello!"
        assert len(response.result["tool_calls"]) == 1
        assert response.result["tool_calls"][0]["name"] == "search"
        assert response.usage["total_tokens"] == 15
        assert response.pod_name is not None
        assert response.duration_ms > 0
        assert seen == [request.platform_trace_id]
        assert get_current_trace_id() is None

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """Retries on rate-limit error and succeeds on second attempt."""
        from agent_engine_sdk.models import LLMStreamChunk

        server = _make_tool_server()
        mock_adapter = AsyncMock()

        rate_limit_err = Exception("Rate limit exceeded")
        rate_limit_err.status_code = 429  # type: ignore[attr-defined]

        call_count = 0

        async def fake_astream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise rate_limit_err
            yield LLMStreamChunk(content="Retry success")

        mock_adapter.astream = fake_astream

        with (
            patch.object(server, "_create_llm_for_pod", return_value=mock_adapter),
            patch("agent_engine_runner_shared.server.tool.is_retryable_error", return_value=True),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            response = await server._handle_invoke_llm(self._make_request())

        assert response.status == "success"
        assert response.result["content"] == "Retry success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        """Returns error when all retries are exhausted."""
        server = _make_tool_server()
        mock_adapter = AsyncMock()

        rate_limit_err = Exception("Rate limit exceeded")
        rate_limit_err.status_code = 429  # type: ignore[attr-defined]

        async def fake_astream(*args, **kwargs):
            raise rate_limit_err
            yield  # pragma: no cover

        mock_adapter.astream = fake_astream

        with (
            patch.object(server, "_create_llm_for_pod", return_value=mock_adapter),
            patch("agent_engine_runner_shared.server.tool.is_retryable_error", return_value=True),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            response = await server._handle_invoke_llm(self._make_request())

        assert response.status == "error"
        assert "Rate limit" in response.error
        assert response.pod_name is not None
        assert response.duration_ms > 0


class TestNamedLLMRouting:
    """Tests for tool pod routing by llm_id."""

    @pytest.fixture(autouse=True)
    def _setup_hooks(self):
        hooks.register_llm_adapter_factory(
            lambda llm, tools=None, tool_choice=None: MagicMock(_llm=llm)
        )
        yield
        hooks.reset_hooks()

    def test_named_llm_used_when_llm_id_set(self):
        """The named registry is the only routing path on the tool pod."""
        llm_primary = MagicMock(name="primary_llm")
        with hooks.entrypoint_scope():
            hooks.register_llm("primary", llm_primary)

        server = _make_tool_server()
        result = server._create_llm_for_pod("primary")

        assert result._llm is llm_primary

    def test_missing_llm_id_raises_key_error(self):
        """Requesting an unregistered llm_id raises KeyError immediately."""
        server = _make_tool_server()
        with pytest.raises(KeyError, match="llm_id 'missing'"):
            server._create_llm_for_pod("missing")

    @pytest.mark.asyncio
    async def test_stream_passes_llm_id_from_request(self):
        """_stream_llm_chunks passes llm_id from request args to _create_llm_for_pod."""
        server = _make_tool_server()
        captured = {}

        mock_adapter = MagicMock()

        async def fake_astream(*args, **kwargs):
            yield LLMStreamChunk(content="hello")

        mock_adapter.astream = fake_astream

        def spy(llm_id, tools=None, tool_choice=None):
            captured["llm_id"] = llm_id
            return mock_adapter

        server._create_llm_for_pod = spy

        request = LLMPodInvokeRequest(
            execution_id="exec-1",
            arguments=InvokeLLMRequestArguments(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                llm_id="secondary",
            ),
        )
        _ = [chunk async for chunk in server._stream_llm_chunks(request)]
        assert captured["llm_id"] == "secondary"
