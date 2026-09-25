"""Integration tests for SecureWrappedLLM with the real SecureLLMProxy."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatResult

from agent_engine_sdk_langgraph.secure_llm import SecureWrappedLLM
from agent_engine_runner_shared.models import LLMPodStreamEvent, ToolExecuteResponse
from agent_engine_runner_shared.secure_wrapper import LLMInvocationError


class _FakeSSEEvent:
    def __init__(self, data: str):
        self.data = data


class _FakeEventSource:
    response = MagicMock()

    def __init__(self, events: list[_FakeSSEEvent]):
        self._events = events

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_sse(self):
        yield from self._events


def _event_source_from_fake_model(
    llm: FakeListChatModel,
    *,
    prompt: str,
    duration_ms: float = 11.0,
) -> _FakeEventSource:
    events: list[_FakeSSEEvent] = []
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        event = LLMPodStreamEvent(
            content=chunk.content if isinstance(chunk.content, str) else None,
            id=getattr(chunk, "id", None),
            name=getattr(chunk, "name", None),
            response_metadata=getattr(chunk, "response_metadata", None) or None,
            additional_kwargs=getattr(chunk, "additional_kwargs", None) or None,
        )
        events.append(_FakeSSEEvent(event.model_dump_json(exclude_none=True)))
    events.append(
        _FakeSSEEvent(
            LLMPodStreamEvent(
                done=True,
                pod_name="tool-pod-1",
                duration_ms=duration_ms,
            ).model_dump_json(exclude_none=True)
        )
    )
    return _FakeEventSource(events)


def _make_wrapped(
    llm: Any | None = None,
    *,
    model_name: str | None = None,
) -> tuple[Any, MagicMock, SecureWrappedLLM]:
    mock_llm = llm or MagicMock()
    if llm is None:
        mock_llm.model = "gpt-4o"
        mock_llm.temperature = 0.0

    mock_wrapper = MagicMock()
    mock_wrapper.oe_url = "http://localhost:8080"
    mock_wrapper.execution_id = "exec-123"
    from agent_engine_runner_shared.secure_wrapper import OperationalStepAllocator

    mock_wrapper.operational_steps = OperationalStepAllocator()
    mock_wrapper.step_counter = 0

    def next_operational_step() -> int:
        mock_wrapper.step_counter = mock_wrapper.operational_steps.next()
        return mock_wrapper.step_counter

    def observe_operational_step(n: int) -> None:
        mock_wrapper.operational_steps.observe_at_least(n)
        mock_wrapper.step_counter = mock_wrapper.operational_steps.current()

    mock_wrapper.next_operational_step.side_effect = next_operational_step
    mock_wrapper.observe_operational_step.side_effect = observe_operational_step

    wrapped = SecureWrappedLLM(
        llm=mock_llm,
        get_wrapper=MagicMock(return_value=mock_wrapper),
        llm_id="primary",
        model_name=model_name,
    )
    return mock_llm, mock_wrapper, wrapped


class TestGenerateIntegration:
    @patch("agent_engine_sdk_langgraph.secure_llm.Metrics")
    @patch("agent_engine_runner_shared.secure_llm_proxy.connect_sse")
    @patch("agent_engine_runner_shared.secure_llm_proxy.request_oe_approval")
    def test_generate_with_fake_list_chat_model_oe_stream_relay(
        self, mock_approval, mock_connect_sse, mock_metrics
    ) -> None:
        llm = FakeListChatModel(responses=["Hello world"])
        mock_connect_sse.return_value = _event_source_from_fake_model(llm, prompt="Hi")
        mock_approval.return_value = ToolExecuteResponse(
            proceed=True,
            route_to="http://oe:8000/tool/stream/exec-123/1",
            latest_step_number=4,
        )

        _, mock_wrapper, wrapped = _make_wrapped(llm, model_name="fake-list-chat-model")

        result = wrapped._generate([HumanMessage(content="Hi")])

        assert isinstance(result, ChatResult)
        assert result.generations[0].message.content == "Hello world"
        assert mock_wrapper.step_counter == 4
        mock_connect_sse.assert_called_once()
        mock_metrics.record_latency.assert_called_once_with(
            "llm_call", 11.0, model_name="fake-list-chat-model"
        )
        mock_metrics.record_error.assert_not_called()

    @patch("agent_engine_runner_shared.secure_llm_proxy.request_oe_approval")
    def test_generate_returns_chat_result_from_oe_owned_result(
        self, mock_approval
    ) -> None:
        mock_approval.return_value = ToolExecuteResponse(
            proceed=True,
            status="success",
            result={
                "content": "Hello!",
                "tool_calls": [
                    {"id": "call-1", "name": "lookup", "arguments": {"q": "hi"}}
                ],
                "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
                "id": "run-1",
                "name": "assistant",
                "additional_kwargs": {"refusal": None},
                "response_metadata": {"finish_reason": "stop"},
            },
            duration_ms=12.0,
            latest_step_number=4,
            pod_name="tool-pod-1",
        )

        mock_llm, mock_wrapper, wrapped = _make_wrapped()

        result = wrapped._generate(
            [HumanMessage(content="Hi")], stop=["END"], max_tokens=32
        )

        assert isinstance(result, ChatResult)
        assert result.generations[0].message.content == "Hello!"
        assert result.generations[0].message.response_metadata == {
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
            "finish_reason": "stop",
        }
        assert result.generations[0].message.id == "run-1"
        assert result.generations[0].message.name == "assistant"
        assert result.generations[0].message.additional_kwargs == {"refusal": None}
        assert mock_wrapper.step_counter == 4
        mock_llm.invoke.assert_not_called()

        call_kwargs = mock_approval.call_args.kwargs
        assert call_kwargs["arguments"]["stop"] == ["END"]
        assert call_kwargs["arguments"]["options"] == {"max_tokens": 32}

    @patch("agent_engine_runner_shared.secure_llm_proxy.request_oe_approval")
    def test_generate_returns_cached_result_without_calling_llm(
        self, mock_approval
    ) -> None:
        mock_approval.return_value = ToolExecuteResponse(
            proceed=True,
            status="success",
            result={"content": "Cached!", "tool_calls": None, "metadata": {}},
            from_cache=True,
            duration_ms=5.0,
        )

        mock_llm, _, wrapped = _make_wrapped()

        result = wrapped._generate([HumanMessage(content="Hi")])

        assert result.generations[0].message.content == "Cached!"
        mock_llm.invoke.assert_not_called()

    @patch("agent_engine_runner_shared.secure_llm_proxy.request_oe_approval")
    def test_generate_raises_oe_owned_error(self, mock_approval) -> None:
        mock_approval.return_value = ToolExecuteResponse(
            proceed=True,
            status="error",
            error="llm pod failed",
            duration_ms=8.0,
        )

        _, _, wrapped = _make_wrapped()

        with pytest.raises(LLMInvocationError, match="llm pod failed"):
            wrapped._generate([HumanMessage(content="Hi")])


class TestStreamIntegration:
    @patch("agent_engine_runner_shared.secure_llm_proxy.request_oe_approval")
    def test_stream_synthesizes_chunks_from_oe_owned_result(
        self, mock_approval
    ) -> None:
        mock_approval.return_value = ToolExecuteResponse(
            proceed=True,
            status="success",
            result={
                "content": "Hello",
                "tool_calls": [
                    {"id": "call-1", "name": "lookup", "arguments": {"q": "weather"}}
                ],
                "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
            },
            duration_ms=10.0,
        )

        mock_llm, _, wrapped = _make_wrapped()

        chunks = list(wrapped._stream([HumanMessage(content="Hi")]))

        assert len(chunks) == 2
        assert chunks[0].message.content == "Hello"
        assert chunks[0].message.tool_call_chunks[0]["name"] == "lookup"  # type: ignore[union-attr]
        usage_message = chunks[1].message
        assert isinstance(usage_message, AIMessageChunk)
        assert usage_message.usage_metadata == {
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
        }
        mock_llm.stream.assert_not_called()

    @patch("agent_engine_runner_shared.secure_llm_proxy.request_oe_approval")
    def test_stream_raises_oe_owned_error(self, mock_approval) -> None:
        mock_approval.return_value = ToolExecuteResponse(
            proceed=True,
            status="error",
            error="llm pod failed",
        )

        _, _, wrapped = _make_wrapped()

        with pytest.raises(LLMInvocationError, match="llm pod failed"):
            list(wrapped._stream([HumanMessage(content="Hi")]))
