"""Tests for LangChainLLMAdapter."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableBinding
from agent_engine_sdk import Message
from agent_engine_sdk.models import LLMResponse, LLMStreamChunk, LLMToolSchema

from agent_engine_sdk_langgraph.llm_adapter import (
    LangChainLLMAdapter,
    _supports_stream_usage,
)


class _CapturingModel:
    """Stand-in for a chat model that records the kwargs passed to astream.

    Subclasses set ``model_fields`` to mimic whether a provider declares the
    ``stream_usage`` option — OpenAI-compatible / Anthropic do; Gemini does not.
    """

    model_fields: dict = {}

    def __init__(self) -> None:
        self.captured_kwargs: dict | None = None

    async def astream(self, messages, **kwargs):
        self.captured_kwargs = kwargs
        chunk = MagicMock()
        chunk.content = "hi"
        chunk.tool_call_chunks = []
        chunk.tool_calls = []
        chunk.usage_metadata = None
        yield chunk


class _SupportsStreamUsage(_CapturingModel):
    model_fields = {"stream_usage": object()}


class _NoStreamUsage(_CapturingModel):
    model_fields = {"temperature": object()}


class _SupportsStreamUsageWithTools(_SupportsStreamUsage):
    """Like _SupportsStreamUsage but bind_tools returns a RunnableBinding around self.

    Mirrors production: LangChainLLMAdapter(llm, tools=tools) stores
    self._llm = llm.bind_tools(...), which is a RunnableBinding.
    """

    def bind_tools(self, _tools):
        binding = MagicMock(spec=RunnableBinding)
        binding.bound = self
        binding.astream = self.astream
        return binding


class _ResponsesApi(_SupportsStreamUsage):
    use_responses_api = True

    def bind_tools(self, _tools):
        binding = MagicMock(spec=RunnableBinding)
        binding.bound = self
        binding.astream = self.astream
        return binding


class _ImplicitResponsesApi(_SupportsStreamUsage):
    """LangChain routes these via _use_responses_api even when the flag is unset."""

    def _use_responses_api(self, _payload):
        return True


class _ResponsesProbeRaises(_SupportsStreamUsage):
    def _use_responses_api(self, _payload):
        raise RuntimeError("probe failed")


async def _drain(adapter, **astream_kwargs):
    async for _ in adapter.astream(
        [Message(role="user", content="hi")], **astream_kwargs
    ):
        pass


class TestStreamUsage:
    """Tests for automatic stream_usage injection."""

    def test_supports_stream_usage_true_when_field_declared(self):
        assert _supports_stream_usage(_SupportsStreamUsage()) is True

    def test_supports_stream_usage_false_when_field_absent(self):
        assert _supports_stream_usage(_NoStreamUsage()) is False

    def test_supports_stream_usage_false_for_unknown_object(self):
        # A bare mock has no real model_fields dict — must not error or inject.
        assert _supports_stream_usage(MagicMock()) is False

    @pytest.mark.anyio
    async def test_injects_stream_usage_for_supported_model(self):
        """astream enables stream_usage for providers that declare it."""
        model = _SupportsStreamUsage()
        await _drain(LangChainLLMAdapter(model))
        assert model.captured_kwargs is not None
        assert model.captured_kwargs.get("stream_usage") is True

    @pytest.mark.anyio
    async def test_does_not_inject_for_unsupported_model(self):
        """astream leaves providers that emit usage natively untouched."""
        model = _NoStreamUsage()
        await _drain(LangChainLLMAdapter(model))
        assert model.captured_kwargs is not None
        assert "stream_usage" not in model.captured_kwargs

    @pytest.mark.anyio
    async def test_caller_provided_stream_usage_is_preserved(self):
        """An explicit stream_usage from the caller is not overridden."""
        model = _SupportsStreamUsage()
        await _drain(LangChainLLMAdapter(model), stream_usage=False)
        assert model.captured_kwargs is not None
        assert model.captured_kwargs.get("stream_usage") is False

    @pytest.mark.anyio
    async def test_injects_stream_usage_through_runnable_binding(self):
        """stream_usage is injected when self._llm is a RunnableBinding (tools bound).

        Production always calls LangChainLLMAdapter(llm, tools=tools), which
        stores self._llm = llm.bind_tools(...) — a RunnableBinding. The
        _supports_stream_usage helper must unwrap .bound to reach model_fields;
        this test exercises that path end-to-end.
        """
        model = _SupportsStreamUsageWithTools()
        tool = MagicMock()
        tool.to_langchain_dict.return_value = {"name": "t", "description": "d"}
        adapter = LangChainLLMAdapter(model, tools=[tool])

        assert isinstance(adapter._llm, RunnableBinding)

        await _drain(adapter)
        assert model.captured_kwargs is not None
        assert model.captured_kwargs.get("stream_usage") is True

    def test_supports_stream_usage_false_for_responses_api(self):
        assert _supports_stream_usage(_ResponsesApi()) is False
        assert _supports_stream_usage(_ImplicitResponsesApi()) is False

    def test_supports_stream_usage_true_when_responses_probe_raises(self):
        assert _supports_stream_usage(_ResponsesProbeRaises()) is True

    @pytest.mark.anyio
    async def test_does_not_inject_for_responses_api(self):
        model = _ResponsesApi()
        await _drain(LangChainLLMAdapter(model))
        assert model.captured_kwargs is not None
        assert "stream_usage" not in model.captured_kwargs

    @pytest.mark.anyio
    async def test_does_not_inject_for_implicit_responses_api(self):
        model = _ImplicitResponsesApi()
        await _drain(LangChainLLMAdapter(model))
        assert model.captured_kwargs is not None
        assert "stream_usage" not in model.captured_kwargs

    @pytest.mark.anyio
    async def test_does_not_inject_for_responses_api_through_runnable_binding(self):
        model = _ResponsesApi()
        tool = MagicMock()
        tool.to_langchain_dict.return_value = {"name": "t", "description": "d"}
        adapter = LangChainLLMAdapter(model, tools=[tool])
        assert isinstance(adapter._llm, RunnableBinding)
        await _drain(adapter)
        assert model.captured_kwargs is not None
        assert "stream_usage" not in model.captured_kwargs


class TestLangChainLLMAdapterInit:
    """Tests for adapter initialization."""

    def test_binds_tools_when_provided(self):
        """Calls bind_tools on the LLM when tools are given."""
        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound
        tools = [LLMToolSchema(name="search", description="search", parameters={})]

        adapter = LangChainLLMAdapter(mock_llm, tools=tools)

        mock_llm.bind_tools.assert_called_once_with(
            [{"name": "search", "description": "search", "parameters": {}}]
        )
        assert adapter._llm is mock_bound

    def test_no_bind_tools_when_none(self):
        """Does not call bind_tools when tools is None."""
        mock_llm = MagicMock()

        adapter = LangChainLLMAdapter(mock_llm)

        mock_llm.bind_tools.assert_not_called()
        assert adapter._llm is mock_llm

    def test_forwards_tool_choice_to_bind_tools(self):
        """tool_choice is passed to bind_tools so LangChain forces the chosen
        tool (e.g. with_structured_output's schema)."""
        mock_llm = MagicMock()
        tools = [LLMToolSchema(name="Brief", description="brief", parameters={})]

        LangChainLLMAdapter(mock_llm, tools=tools, tool_choice="Brief")

        mock_llm.bind_tools.assert_called_once_with(
            [{"name": "Brief", "description": "brief", "parameters": {}}],
            tool_choice="Brief",
        )

    def test_tool_choice_ignored_without_tools(self):
        """tool_choice with no tools cannot bind anything, so bind_tools is not
        called (a forced choice is meaningless without a tool list)."""
        mock_llm = MagicMock()

        adapter = LangChainLLMAdapter(mock_llm, tool_choice="Brief")

        mock_llm.bind_tools.assert_not_called()
        assert adapter._llm is mock_llm

    def test_forwards_tool_choice_false_to_bind_tools(self):
        """tool_choice=False is a documented LangChain value that disables
        forced tool use. It must be forwarded, not dropped by a truthiness
        guard."""
        mock_llm = MagicMock()
        tools = [LLMToolSchema(name="Brief", description="brief", parameters={})]

        LangChainLLMAdapter(mock_llm, tools=tools, tool_choice=False)

        mock_llm.bind_tools.assert_called_once_with(
            [{"name": "Brief", "description": "brief", "parameters": {}}],
            tool_choice=False,
        )


class TestInvoke:
    """Tests for synchronous invoke."""

    def test_converts_messages_and_returns_llm_response(self):
        """Converts Messages to LC, calls invoke, returns LLMResponse."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Hello!"
        mock_response.tool_calls = []
        mock_response.usage_metadata = None
        mock_llm.invoke.return_value = mock_response

        adapter = LangChainLLMAdapter(mock_llm)
        result = adapter.invoke([Message(role="user", content="hi")])

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"
        mock_llm.invoke.assert_called_once()


class TestAinvoke:
    """Tests for async ainvoke."""

    @pytest.mark.anyio
    async def test_async_invoke_returns_llm_response(self):
        """Async ainvoke returns LLMResponse."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Async hello!"
        mock_response.tool_calls = []
        mock_response.usage_metadata = {
            "input_tokens": 5,
            "output_tokens": 10,
            "total_tokens": 15,
        }
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        adapter = LangChainLLMAdapter(mock_llm)
        result = await adapter.ainvoke([Message(role="user", content="hi")])

        assert isinstance(result, LLMResponse)
        assert result.content == "Async hello!"
        assert result.usage is not None
        assert result.usage.input_tokens == 5

    @pytest.mark.anyio
    async def test_ainvoke_with_tool_calls(self):
        """ainvoke preserves tool_calls in response."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.tool_calls = [
            {"id": "tc1", "name": "search", "args": {"q": "test"}},
        ]
        mock_response.usage_metadata = None
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        adapter = LangChainLLMAdapter(mock_llm)
        msgs = [Message(role="user", content="search for test")]
        result = await adapter.ainvoke(msgs)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search"


class TestAstream:
    """Tests for async streaming."""

    @pytest.mark.anyio
    async def test_yields_llm_stream_chunks(self):
        """astream yields LLMStreamChunk objects."""
        mock_llm = MagicMock()

        chunk1 = MagicMock()
        chunk1.content = "Hello"
        chunk1.tool_call_chunks = []
        chunk1.tool_calls = []
        chunk1.usage_metadata = None

        chunk2 = MagicMock()
        chunk2.content = " world"
        chunk2.tool_call_chunks = []
        chunk2.tool_calls = []
        chunk2.usage_metadata = None

        async def mock_astream(*args, **kwargs):
            yield chunk1
            yield chunk2

        mock_llm.astream = mock_astream

        adapter = LangChainLLMAdapter(mock_llm)
        chunks = []
        async for chunk in adapter.astream([Message(role="user", content="hi")]):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert all(isinstance(c, LLMStreamChunk) for c in chunks)
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"

    @pytest.mark.anyio
    async def test_stream_with_tool_call_chunks(self):
        """astream converts tool_call_chunks to ToolCallChunk."""
        mock_llm = MagicMock()

        chunk = MagicMock()
        chunk.content = ""
        chunk.tool_call_chunks = [
            {"id": "tc1", "name": "search", "args": '{"q":', "index": 0},
        ]
        chunk.tool_calls = []
        chunk.usage_metadata = None

        async def mock_astream(*args, **kwargs):
            yield chunk

        mock_llm.astream = mock_astream

        adapter = LangChainLLMAdapter(mock_llm)
        chunks = []
        async for c in adapter.astream([Message(role="user", content="hi")]):
            chunks.append(c)

        assert len(chunks) == 1
        assert chunks[0].tool_calls is not None
        assert chunks[0].tool_calls[0].name == "search"
        assert chunks[0].tool_calls[0].index == 0

    @pytest.mark.anyio
    async def test_stream_with_tool_call_chunk_dict_args(self):
        """astream JSON-encodes structured tool_call_chunk args."""
        mock_llm = MagicMock()

        chunk = MagicMock()
        chunk.content = ""
        chunk.tool_call_chunks = [
            {"id": "tc1", "name": "search", "args": {"q": "test"}, "index": 0},
        ]
        chunk.tool_calls = []
        chunk.usage_metadata = None

        async def mock_astream(*args, **kwargs):
            yield chunk

        mock_llm.astream = mock_astream

        adapter = LangChainLLMAdapter(mock_llm)
        chunks = []
        async for c in adapter.astream([Message(role="user", content="hi")]):
            chunks.append(c)

        assert len(chunks) == 1
        assert chunks[0].tool_calls is not None
        assert chunks[0].tool_calls[0].args == json.dumps({"q": "test"})

    @pytest.mark.anyio
    async def test_stream_with_usage(self):
        """astream extracts usage from final chunk."""
        mock_llm = MagicMock()

        chunk = MagicMock()
        chunk.content = ""
        chunk.tool_call_chunks = []
        chunk.tool_calls = []
        chunk.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        }

        async def mock_astream(*args, **kwargs):
            yield chunk

        mock_llm.astream = mock_astream

        adapter = LangChainLLMAdapter(mock_llm)
        chunks = []
        async for c in adapter.astream([Message(role="user", content="hi")]):
            chunks.append(c)

        assert chunks[0].usage is not None
        assert chunks[0].usage["input_tokens"] == 10

    @pytest.mark.anyio
    async def test_stream_partial_usage_on_separate_chunks(self):
        """Later chunks carry a running cumulative snapshot of usage."""
        mock_llm = MagicMock()

        early = MagicMock()
        early.content = "Hel"
        early.tool_call_chunks = []
        early.tool_calls = []
        early.usage_metadata = {"input_tokens": 1048}

        late = MagicMock()
        late.content = "lo"
        late.tool_call_chunks = []
        late.tool_calls = []
        late.usage_metadata = {"output_tokens": 1222}

        async def mock_astream(*args, **kwargs):
            yield early
            yield late

        mock_llm.astream = mock_astream

        adapter = LangChainLLMAdapter(mock_llm)
        chunks = []
        async for c in adapter.astream([Message(role="user", content="hi")]):
            chunks.append(c)

        assert len(chunks) == 2
        assert chunks[0].usage is not None
        assert chunks[0].usage["input_tokens"] == 1048
        assert chunks[1].usage is not None
        assert chunks[1].usage["input_tokens"] == 1048
        assert chunks[1].usage["output_tokens"] == 1222
        assert chunks[1].usage["total_tokens"] == 2270

    @pytest.mark.anyio
    async def test_stream_accumulates_zero_filled_langchain_deltas(self):
        """Gemini-style additive UsageMetadata: later chunks zero-fill input."""
        mock_llm = MagicMock()

        early = MagicMock()
        early.content = "Hel"
        early.tool_call_chunks = []
        early.tool_calls = []
        early.usage_metadata = {
            "input_tokens": 18,
            "output_tokens": 1,
            "total_tokens": 19,
        }

        late = MagicMock()
        late.content = "lo"
        late.tool_call_chunks = []
        late.tool_calls = []
        late.usage_metadata = {
            "input_tokens": 0,
            "output_tokens": 4,
            "total_tokens": 4,
        }

        async def mock_astream(*args, **kwargs):
            yield early
            yield late

        mock_llm.astream = mock_astream

        adapter = LangChainLLMAdapter(mock_llm)
        chunks = []
        async for c in adapter.astream([Message(role="user", content="hi")]):
            chunks.append(c)

        assert chunks[1].usage is not None
        assert chunks[1].usage["input_tokens"] == 18
        assert chunks[1].usage["output_tokens"] == 5
        assert chunks[1].usage["total_tokens"] == 23

    @pytest.mark.anyio
    async def test_stream_takes_cumulative_usage_snapshots(self):
        """Providers that repeat growing totals must not be added."""
        mock_llm = MagicMock()

        early = MagicMock()
        early.content = "Hel"
        early.tool_call_chunks = []
        early.tool_calls = []
        early.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 1,
            "total_tokens": 11,
        }

        late = MagicMock()
        late.content = "lo"
        late.tool_call_chunks = []
        late.tool_calls = []
        late.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
        }

        async def mock_astream(*args, **kwargs):
            yield early
            yield late

        mock_llm.astream = mock_astream

        adapter = LangChainLLMAdapter(mock_llm)
        chunks = []
        async for c in adapter.astream([Message(role="user", content="hi")]):
            chunks.append(c)

        assert chunks[1].usage is not None
        assert chunks[1].usage["input_tokens"] == 10
        assert chunks[1].usage["output_tokens"] == 2
        assert chunks[1].usage["total_tokens"] == 12

    @pytest.mark.anyio
    async def test_invoke_accepts_usage_object_in_response_metadata(self):
        from types import SimpleNamespace

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Hello!"
        mock_response.tool_calls = []
        mock_response.usage_metadata = None
        mock_response.response_metadata = {
            "usage": SimpleNamespace(input_tokens=1048, output_tokens=1222),
        }
        mock_response.additional_kwargs = {}
        mock_response.metadata = {}
        mock_response.usage = None
        mock_llm.invoke.return_value = mock_response

        adapter = LangChainLLMAdapter(mock_llm)
        result = adapter.invoke([Message(role="user", content="hi")])

        assert result.content == "Hello!"
        assert result.usage is not None
        assert result.usage.input_tokens == 1048
        assert result.usage.output_tokens == 1222

    @pytest.mark.anyio
    async def test_stream_accepts_usage_object_in_response_metadata(self):
        from types import SimpleNamespace

        mock_llm = MagicMock()
        chunk = MagicMock()
        chunk.content = "Hello"
        chunk.tool_call_chunks = []
        chunk.tool_calls = []
        chunk.usage_metadata = None
        chunk.response_metadata = {
            "usage": SimpleNamespace(input_tokens=1048, output_tokens=1222),
        }
        chunk.additional_kwargs = {}
        chunk.usage = None

        async def mock_astream(*args, **kwargs):
            yield chunk

        mock_llm.astream = mock_astream

        adapter = LangChainLLMAdapter(mock_llm)
        chunks = []
        async for c in adapter.astream([Message(role="user", content="hi")]):
            chunks.append(c)

        assert chunks[0].content == "Hello"
        assert chunks[0].usage is not None
        assert chunks[0].usage["input_tokens"] == 1048
        assert chunks[0].response_metadata is not None
        assert chunks[0].response_metadata["usage"]["input_tokens"] == 1048

    @pytest.mark.anyio
    async def test_stream_preserves_message_metadata(self):
        """astream preserves LangChain message metadata fields."""
        mock_llm = MagicMock()

        chunk = AIMessageChunk(
            content="Done",
            id="run-1",
            name="assistant",
            additional_kwargs={"refusal": None},
            response_metadata={"finish_reason": "stop"},
        )

        async def mock_astream(*args, **kwargs):
            yield chunk

        mock_llm.astream = mock_astream

        adapter = LangChainLLMAdapter(mock_llm)
        chunks = []
        async for c in adapter.astream([Message(role="user", content="hi")]):
            chunks.append(c)

        assert len(chunks) == 1
        assert chunks[0].id == "run-1"
        assert chunks[0].name == "assistant"
        assert chunks[0].additional_kwargs == {"refusal": None}
        assert chunks[0].response_metadata == {"finish_reason": "stop"}

    @pytest.mark.anyio
    async def test_stream_with_list_content_blocks(self):
        """astream normalizes list-of-dicts content to string."""
        mock_llm = MagicMock()

        chunk = MagicMock()
        chunk.content = [
            {"type": "text", "text": "Hello from "},
            {"type": "text", "text": "content blocks"},
        ]
        chunk.tool_call_chunks = []
        chunk.tool_calls = []
        chunk.usage_metadata = None

        async def mock_astream(*args, **kwargs):
            yield chunk

        mock_llm.astream = mock_astream

        adapter = LangChainLLMAdapter(mock_llm)
        chunks = []
        async for c in adapter.astream([Message(role="user", content="hi")]):
            chunks.append(c)

        assert len(chunks) == 1
        assert isinstance(chunks[0].content, str)
        assert chunks[0].content == "Hello from content blocks"

    @pytest.mark.anyio
    async def test_stream_with_single_text_content_block(self):
        """astream normalizes a single-element list content block to string."""
        mock_llm = MagicMock()

        chunk = MagicMock()
        chunk.content = [{"type": "text", "text": "single block"}]
        chunk.tool_call_chunks = []
        chunk.tool_calls = []
        chunk.usage_metadata = None

        async def mock_astream(*args, **kwargs):
            yield chunk

        mock_llm.astream = mock_astream

        adapter = LangChainLLMAdapter(mock_llm)
        chunks = []
        async for c in adapter.astream([Message(role="user", content="hi")]):
            chunks.append(c)

        assert len(chunks) == 1
        assert isinstance(chunks[0].content, str)
        assert chunks[0].content == "single block"

    @pytest.mark.anyio
    async def test_ainvoke_falls_back_to_sync_invoke(self):
        """ainvoke falls back to sync invoke when ainvoke is not a coroutine."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "sync fallback"
        mock_response.tool_calls = []
        mock_response.usage_metadata = None
        # ainvoke exists but is NOT a coroutine function (e.g., plain method)
        mock_llm.ainvoke = lambda *a, **kw: mock_response
        mock_llm.invoke.return_value = mock_response

        adapter = LangChainLLMAdapter(mock_llm)
        result = await adapter.ainvoke([Message(role="user", content="hi")])

        assert isinstance(result, LLMResponse)
        assert result.content == "sync fallback"
        mock_llm.invoke.assert_called_once()

    @pytest.mark.anyio
    async def test_astream_falls_back_to_sync_stream(self):
        """astream falls back to sync stream when astream is not available."""
        mock_llm = MagicMock(spec=["invoke", "stream"])

        chunk1 = MagicMock()
        chunk1.content = "sync"
        chunk1.tool_call_chunks = []
        chunk1.tool_calls = []
        chunk1.usage_metadata = None

        chunk2 = MagicMock()
        chunk2.content = " stream"
        chunk2.tool_call_chunks = []
        chunk2.tool_calls = []
        chunk2.usage_metadata = None

        mock_llm.stream.return_value = iter([chunk1, chunk2])

        adapter = LangChainLLMAdapter(mock_llm)
        chunks = []
        async for c in adapter.astream([Message(role="user", content="hi")]):
            chunks.append(c)

        assert len(chunks) == 2
        assert chunks[0].content == "sync"
        assert chunks[1].content == " stream"
        mock_llm.stream.assert_called_once()

    @pytest.mark.anyio
    async def test_astream_handles_awaitable_return(self):
        """astream handles awaitable astream() return."""
        mock_llm = MagicMock()

        chunk = MagicMock()
        chunk.content = "awaitable"
        chunk.tool_call_chunks = []
        chunk.tool_calls = []
        chunk.usage_metadata = None

        async def async_iter():
            yield chunk

        # astream returns a coroutine that resolves to an async iterator
        async def mock_astream(*args, **kwargs):
            return async_iter()

        mock_llm.astream = mock_astream

        adapter = LangChainLLMAdapter(mock_llm)
        chunks = []
        async for c in adapter.astream([Message(role="user", content="hi")]):
            chunks.append(c)

        assert len(chunks) == 1
        assert chunks[0].content == "awaitable"
