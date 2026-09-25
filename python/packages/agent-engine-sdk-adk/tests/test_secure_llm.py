"""Tests for SecureLlm — ADK BaseLlm that routes through SecureLLMProxy."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from agent_engine_sdk.models import LLMResponse, LLMStreamChunk, LLMTokenUsage


def _make_wrapper() -> MagicMock:
    from agent_engine_runner_shared.secure_wrapper import OperationalStepAllocator

    wrapper = MagicMock()
    wrapper.oe_url = "http://oe:8080"
    wrapper.execution_id = "exec-123"
    wrapper.operational_steps = OperationalStepAllocator()
    wrapper.step_counter = 0
    wrapper.durable_memory = None

    def next_operational_step() -> int:
        wrapper.step_counter = wrapper.operational_steps.next()
        return wrapper.step_counter

    def observe_operational_step(n: int) -> None:
        wrapper.operational_steps.observe_at_least(n)
        wrapper.step_counter = wrapper.operational_steps.current()

    wrapper.next_operational_step.side_effect = next_operational_step
    wrapper.observe_operational_step.side_effect = observe_operational_step
    return wrapper


class TestSecureLlmNonStreaming:
    @pytest.mark.asyncio
    async def test_non_streaming_yields_single_response(self) -> None:
        from google.adk.models import LlmRequest
        from google.genai import types

        from agent_engine_sdk_adk.secure_llm import SecureLlm

        wrapper = _make_wrapper()
        llm = SecureLlm(
            model="gemini-2.5-flash",
            get_wrapper=lambda: wrapper,
            llm_id="default",
        )

        mock_chunks = [
            LLMStreamChunk(content="Hello world"),
        ]

        request = LlmRequest(
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="hi")],
                )
            ],
            config=types.GenerateContentConfig(),
        )

        mock_response = LLMResponse(content="Hello world")

        with patch("agent_engine_sdk_adk.secure_llm.SecureLLMProxy") as MockProxy:
            proxy_instance = MagicMock()
            proxy_instance.stream.return_value = iter(mock_chunks)
            proxy_instance.response_from_stream_chunks.return_value = mock_response
            proxy_instance.last_duration_ms = 100.0
            proxy_instance.last_from_cache = False
            proxy_instance.last_latest_step_number = None
            MockProxy.return_value = proxy_instance

            responses: list[Any] = []
            async for resp in llm.generate_content_async(request, stream=False):
                responses.append(resp)

        assert len(responses) == 1
        assert responses[0].partial is False
        assert responses[0].content is not None
        assert any(p.text == "Hello world" for p in responses[0].content.parts)
        assert MockProxy.call_args.kwargs["durable_memory"] is None


class TestSecureLlmStreaming:
    @pytest.mark.asyncio
    async def test_streaming_yields_partial_then_final(self) -> None:
        from google.adk.models import LlmRequest
        from google.genai import types

        from agent_engine_sdk_adk.secure_llm import SecureLlm

        wrapper = _make_wrapper()
        llm = SecureLlm(
            model="gemini-2.5-flash",
            get_wrapper=lambda: wrapper,
            llm_id="default",
        )

        mock_chunks = [
            LLMStreamChunk(content="Hello"),
            LLMStreamChunk(content=" world"),
        ]
        mock_response = LLMResponse(content="Hello world")

        request = LlmRequest(
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="hi")],
                )
            ],
            config=types.GenerateContentConfig(),
        )

        with patch("agent_engine_sdk_adk.secure_llm.SecureLLMProxy") as MockProxy:
            proxy_instance = MagicMock()
            proxy_instance.stream.return_value = iter(mock_chunks)
            proxy_instance.response_from_stream_chunks.return_value = mock_response
            proxy_instance.last_duration_ms = 100.0
            proxy_instance.last_from_cache = False
            proxy_instance.last_latest_step_number = None
            MockProxy.return_value = proxy_instance

            responses: list[Any] = []
            async for resp in llm.generate_content_async(request, stream=True):
                responses.append(resp)

        partials = [r for r in responses if r.partial]
        finals = [r for r in responses if not r.partial]
        assert len(partials) >= 1
        assert len(finals) == 1
        assert finals[0].content is not None
        assert any(p.text == "Hello world" for p in finals[0].content.parts)


class TestEnrichSpanWithTokenCounts:
    def test_sets_prompt_and_completion_on_recording_span(self) -> None:
        from agent_engine_sdk_adk.secure_llm import _enrich_span_with_token_counts

        response = LLMResponse(
            content="hi",
            usage=LLMTokenUsage(prompt_tokens=100, completion_tokens=20),
        )
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        with patch("agent_engine_sdk_adk.secure_llm.trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            _enrich_span_with_token_counts(response)

        mock_span.set_attribute.assert_any_call("llm.token_count.prompt", 100)
        mock_span.set_attribute.assert_any_call("llm.token_count.completion", 20)

    def test_uses_input_output_tokens_fallback(self) -> None:
        from agent_engine_sdk_adk.secure_llm import _enrich_span_with_token_counts

        response = LLMResponse(
            content="hi",
            usage=LLMTokenUsage(input_tokens=50, output_tokens=10),
        )
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        with patch("agent_engine_sdk_adk.secure_llm.trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            _enrich_span_with_token_counts(response)

        mock_span.set_attribute.assert_any_call("llm.token_count.prompt", 50)
        mock_span.set_attribute.assert_any_call("llm.token_count.completion", 10)

    def test_skips_when_no_usage(self) -> None:
        from agent_engine_sdk_adk.secure_llm import _enrich_span_with_token_counts

        response = LLMResponse(content="hi")
        mock_span = MagicMock()
        with patch("agent_engine_sdk_adk.secure_llm.trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            _enrich_span_with_token_counts(response)

        mock_span.set_attribute.assert_not_called()

    def test_skips_when_span_not_recording(self) -> None:
        from agent_engine_sdk_adk.secure_llm import _enrich_span_with_token_counts

        response = LLMResponse(
            content="hi",
            usage=LLMTokenUsage(prompt_tokens=100, completion_tokens=20),
        )
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False
        with patch("agent_engine_sdk_adk.secure_llm.trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            _enrich_span_with_token_counts(response)

        mock_span.set_attribute.assert_not_called()


class TestSecureLlmErrors:
    @pytest.mark.asyncio
    async def test_raises_when_no_wrapper(self) -> None:
        from google.adk.models import LlmRequest
        from google.genai import types

        from agent_engine_sdk_adk.secure_llm import SecureLlm

        llm = SecureLlm(
            model="gemini-2.5-flash",
            get_wrapper=lambda: None,
            llm_id="default",
        )
        request = LlmRequest(
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="hi")],
                )
            ],
            config=types.GenerateContentConfig(),
        )

        with pytest.raises(RuntimeError, match="not initialized"):
            async for _ in llm.generate_content_async(request):
                pass

    @pytest.mark.asyncio
    async def test_propagates_policy_denied(self) -> None:
        from google.adk.models import LlmRequest
        from google.genai import types

        from agent_engine_runner_shared.secure_wrapper import PolicyDeniedException
        from agent_engine_sdk_adk.secure_llm import SecureLlm

        wrapper = _make_wrapper()
        llm = SecureLlm(
            model="gemini-2.5-flash",
            get_wrapper=lambda: wrapper,
            llm_id="default",
        )
        request = LlmRequest(
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="hi")],
                )
            ],
            config=types.GenerateContentConfig(),
        )

        with patch("agent_engine_sdk_adk.secure_llm.SecureLLMProxy") as MockProxy:
            proxy_instance = MagicMock()
            proxy_instance.stream.side_effect = PolicyDeniedException("blocked")
            MockProxy.return_value = proxy_instance

            with pytest.raises(PolicyDeniedException):
                async for _ in llm.generate_content_async(request):
                    pass
