"""Unit tests for SecureLLMProxy's OE-owned invoke_llm contract."""

from __future__ import annotations

import json
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import httpx
import pytest
from agent_engine_sdk.models import LLMInvocationOptions, Message

from agent_engine_runner_shared.models import ToolExecuteResponse


class TestSecureLLMProxyConstruction:
    def test_construction(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        assert proxy.oe_url == "http://localhost:8080"
        assert proxy.execution_id == "exec-123"
        assert proxy.step_counter == 0

    def test_strips_trailing_slash(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080/", execution_id="exec-123", llm_id="primary"
        )
        assert proxy.oe_url == "http://localhost:8080"


class TestSecureLLMProxyInvoke:
    def test_invoke_requests_full_oe_execution_payload(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080",
            execution_id="exec-123",
            llm_id="primary",
            model_name="gpt-4o",
            bound_tools=[{"name": "lookup", "parameters": {"type": "object"}}],
        )

        messages = [Message(role="user", content="Hello")]

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                status="success",
                result={"content": "Hi", "tool_calls": [], "metadata": {}},
                duration_ms=12.0,
                latest_step_number=4,
                pod_name="tool-pod-1",
            )

            result = proxy.invoke(
                messages,
                stop=["END"],
                options=LLMInvocationOptions(max_tokens=64),
            )

        assert result.content == "Hi"
        assert proxy.step_counter == 4
        assert proxy.last_duration_ms == 12.0
        assert proxy.last_from_cache is False
        assert proxy.last_pod_name == "tool-pod-1"

        mock_approval.assert_called_once()
        call_kwargs = mock_approval.call_args.kwargs
        assert call_kwargs["tool_name"] == "invoke_llm"
        assert call_kwargs["arguments"]["model"] == "gpt-4o"
        assert call_kwargs["arguments"]["llm_id"] == "primary"
        assert call_kwargs["arguments"]["stop"] == ["END"]
        assert call_kwargs["arguments"]["tools"] == [
            {"name": "lookup", "parameters": {"type": "object"}}
        ]
        assert call_kwargs["arguments"]["options"] == {"max_tokens": 64}
        assert call_kwargs["arguments"]["stream"] is True
        assert call_kwargs["arguments"]["messages"] == [
            {
                "role": "user",
                "content": "Hello",
            }
        ]
        assert "custom_headers" not in call_kwargs
        assert "custom_headers" not in call_kwargs["arguments"]
        assert call_kwargs["timeout"].read == pytest.approx(300.0)

    def test_bound_tool_choice_rides_on_invoke_request(self) -> None:
        """A forced tool choice (e.g. with_structured_output) must travel to OE
        in the invoke_llm arguments so the tool pod can force the call.

        Regression guard: tool_choice was previously dropped, so the
        structured-output tool was offered but never required.
        """
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080",
            execution_id="exec-123",
            llm_id="primary",
            model_name="gpt-4o",
            bound_tools=[{"name": "Brief", "parameters": {"type": "object"}}],
            bound_tool_choice="Brief",
        )

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                status="success",
                result={"content": "Hi", "tool_calls": [], "metadata": {}},
                duration_ms=1.0,
            )
            proxy.invoke([Message(role="user", content="Hello")])

        arguments = mock_approval.call_args.kwargs["arguments"]
        assert arguments["tool_choice"] == "Brief"

    def test_invoke_returns_cached_result(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                status="success",
                result={"content": "Cached", "tool_calls": None, "metadata": {}},
                from_cache=True,
                duration_ms=5.0,
            )

            result = proxy.invoke(messages)

        assert result.content == "Cached"
        assert proxy.last_from_cache is True
        assert proxy.last_duration_ms == 5.0

    def test_invoke_accepts_legacy_cached_result_shape(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                cached_result={"content": "Cached", "tool_calls": None, "metadata": {}},
                from_cache=True,
                duration_ms=7.0,
            )

            result = proxy.invoke(messages)

        assert result.content == "Cached"
        assert proxy.last_from_cache is True

    def test_invoke_raises_on_policy_denied(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
        from agent_engine_runner_shared.secure_wrapper import PolicyDeniedException

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=False,
                reason="Rate limited",
            )

            with pytest.raises(PolicyDeniedException, match="Rate limited"):
                proxy.invoke(messages)

    def test_invoke_policy_denied_carries_guardrail_meta(self) -> None:
        from agent_engine_runner_shared.models import GuardrailMeta
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
        from agent_engine_runner_shared.secure_wrapper import PolicyDeniedException

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]
        policy_id = "6641abc123"

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=False,
                reason="blocked by policy",
                guardrail_meta=GuardrailMeta(
                    guardrail_id=policy_id,
                    guardrail_category="output_validation",
                ),
            )

            with pytest.raises(PolicyDeniedException) as exc_info:
                proxy.invoke(messages)

        assert exc_info.value.guardrail_meta.guardrail_id == policy_id
        assert exc_info.value.guardrail_meta.guardrail_category == "output_validation"

    def test_invoke_terminal_execution_is_not_policy_denied(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
        from agent_engine_runner_shared.secure_wrapper import (
            PolicyDeniedException,
            TerminalExecutionError,
            ToolExecutionError,
        )

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=False,
                reason="execution already error",
            )

            with pytest.raises(TerminalExecutionError, match="already ended in error") as exc_info:
                proxy.invoke(messages)

        assert "Policy denied" not in str(exc_info.value)
        assert isinstance(exc_info.value, ToolExecutionError)
        assert not isinstance(exc_info.value, PolicyDeniedException)

    def test_invoke_returns_substitute_on_guardrail_block(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="__default__"
        )
        messages = [Message(role="user", content="Hello")]
        substitute = "This response was blocked by a content policy."

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=False,
                status="blocked",
                result=substitute,
                reason="1 guardrail policy triggered a block decision",
            )

            result = proxy.invoke(messages)

        assert result.content == substitute

    def test_stream_yields_substitute_on_guardrail_block(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="__default__"
        )
        messages = [Message(role="user", content="Hello")]
        substitute = "This response was blocked by a content policy."

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=False,
                status="blocked",
                result=substitute,
                reason="1 guardrail policy triggered a block decision",
            )

            chunks = list(proxy.stream(messages))

        assert len(chunks) == 1
        assert chunks[0].content == substitute

    def test_stream_raises_on_non_string_block_result(self) -> None:
        """Non-string result on a blocked response is a contract violation and
        must surface as a PolicyDeniedException rather than be coerced to a
        chunk via str(), which would leak a Python repr to the agent."""
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
        from agent_engine_runner_shared.secure_wrapper import PolicyDeniedException

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="__default__"
        )
        messages = [Message(role="user", content="Hello")]

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=False,
                status="blocked",
                result={"content": "should not be reachable"},
                reason="Unexpected result shape",
            )

            with pytest.raises(PolicyDeniedException, match="Unexpected result shape"):
                list(proxy.stream(messages))

    def test_invoke_raises_on_oe_owned_error(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
        from agent_engine_runner_shared.secure_wrapper import LLMInvocationError

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                status="error",
                error="tool pod failed",
                duration_ms=9.0,
            )

            with pytest.raises(LLMInvocationError, match="tool pod failed") as exc_info:
                proxy.invoke(messages)
            assert exc_info.value.source == "llm"

    def test_invoke_retries_oe_retryable_error_then_succeeds(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.side_effect = [
                ToolExecuteResponse(
                    proceed=True,
                    status="error",
                    error="tool pod is no longer reserved for this session; retry request",
                    retryable=True,
                ),
                ToolExecuteResponse(
                    proceed=True,
                    status="success",
                    result={"content": "Hi", "tool_calls": []},
                ),
            ]

            result = proxy.invoke(messages)

        assert result.content == "Hi"
        assert mock_approval.call_count == 2
        assert (
            mock_approval.call_args_list[0].kwargs["step"]
            == mock_approval.call_args_list[1].kwargs["step"]
        )

    def test_invoke_yields_interrupted_marker_on_inline_interrupted_status(self) -> None:
        """A non-streaming/replayed interrupted step must yield the marker, not raise."""
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            # OE returns interrupted with no result (inline direct path or durable
            # replay). Previously this fell through to "Unexpected OE invoke_llm
            # status" and raised.
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                status="interrupted",
                duration_ms=5.0,
            )

            chunks = list(proxy.stream(messages))

        assert chunks[-1].response_metadata == {"interrupted": True}

    def test_invoke_normalizes_usage_into_metadata(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                status="success",
                result={
                    "content": "Hi",
                    "tool_calls": [{"id": "call-1", "name": "lookup", "arguments": {"q": "hi"}}],
                    "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
                },
            )

            result = proxy.invoke(messages)

        assert result.content == "Hi"
        assert result.tool_calls == [{"id": "call-1", "name": "lookup", "args": {"q": "hi"}}]
        assert result.metadata == {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}

    def test_llm_id_included_in_request(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080",
            execution_id="exec-1",
            llm_id="primary",
            model_name="gpt-4o",
        )

        messages = [Message(role="user", content="Hi")]

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                status="success",
                result={"content": "Hi", "tool_calls": [], "metadata": {}},
                duration_ms=10.0,
            )
            proxy.invoke(messages)

        call_kwargs = mock_approval.call_args.kwargs
        assert call_kwargs["arguments"]["llm_id"] == "primary"


class TestSecureLLMProxyStream:
    @pytest.fixture(autouse=True)
    def _no_stream_retry_sleep(self) -> Iterator[None]:
        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.sleep_oe_stream_retry"
        ) as mock_sleep:
            self._sleep_oe_stream_retry = mock_sleep
            yield

    def test_stream_yields_synthetic_chunks_from_final_result(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                status="success",
                result={
                    "content": "Hello!",
                    "tool_calls": [
                        {"id": "call-1", "name": "lookup", "arguments": {"q": "weather"}}
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
                },
                duration_ms=15.0,
            )

            chunks = list(proxy.stream(messages))

        assert len(chunks) == 2
        assert chunks[0].content == "Hello!"
        assert chunks[0].tool_calls is not None
        assert chunks[0].tool_calls[0].name == "lookup"
        assert chunks[0].tool_calls[0].args == json.dumps({"q": "weather"})
        assert chunks[1].usage == {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
        assert proxy.last_duration_ms == 15.0

    def test_stream_handles_usage_only_result(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                status="success",
                result={"content": "", "tool_calls": [], "usage": {"total_tokens": 12}},
            )

            chunks = list(proxy.stream(messages))

        assert len(chunks) == 1
        assert chunks[0].usage == {"total_tokens": 12}

    def test_stream_raises_on_oe_owned_error(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
        from agent_engine_runner_shared.secure_wrapper import LLMInvocationError

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                status="error",
                error="llm pod failed",
            )

            with pytest.raises(LLMInvocationError, match="llm pod failed") as exc_info:
                list(proxy.stream(messages))
            assert exc_info.value.source == "llm"

    def test_stream_forwards_stop_and_kwargs_to_oe_request(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                status="success",
                result={"content": "Hello!", "tool_calls": [], "metadata": {}},
            )

            list(
                proxy.stream(
                    messages,
                    stop=["DONE"],
                    options=LLMInvocationOptions(max_tokens=32),
                )
            )

        call_kwargs = mock_approval.call_args.kwargs
        assert call_kwargs["arguments"]["stop"] == ["DONE"]
        assert call_kwargs["arguments"]["options"] == {"max_tokens": 32}
        assert call_kwargs["arguments"]["stream"] is True
        assert call_kwargs["timeout"].read == pytest.approx(300.0)

    def test_stream_connects_to_routed_oe_sse_endpoint(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        class FakeSSEEvent:
            def __init__(self, data: str):
                self.data = data

        class FakeEventSource:
            response = MagicMock()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def iter_sse(self):
                yield FakeSSEEvent(json.dumps({"content": "Hel", "tool_call_chunks": []}))
                yield FakeSSEEvent(json.dumps({"content": "lo", "tool_call_chunks": []}))
                yield FakeSSEEvent(
                    json.dumps(
                        {
                            "done": True,
                            "pod_name": "tool-pod-1",
                            "duration_ms": 23.0,
                            "usage": {
                                "input_tokens": 10,
                                "output_tokens": 4,
                                "total_tokens": 14,
                            },
                        }
                    )
                )

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080",
            execution_id="exec-123",
            llm_id="primary",
            model_name="gpt-4o",
        )
        messages = [Message(role="user", content="Hello")]

        with (
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
            ) as mock_approval,
            patch("agent_engine_runner_shared.secure_llm_proxy.connect_sse") as mock_connect_sse,
        ):
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                route_to="http://oe:8000/tool/stream/exec-123/7",
            )
            mock_connect_sse.return_value = FakeEventSource()

            chunks = list(proxy.stream(messages, step=7))

        assert [chunk.content for chunk in chunks[:2]] == ["Hel", "lo"]
        assert chunks[2].usage == {
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
        }
        assert proxy.last_duration_ms == 23.0
        assert proxy.last_pod_name == "tool-pod-1"

        connect_kwargs = mock_connect_sse.call_args.kwargs
        assert "json" not in connect_kwargs
        assert mock_connect_sse.call_args.args[2] == "http://oe:8000/tool/stream/exec-123/7"

    def test_stream_raises_on_oe_sse_connection_error(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
        from agent_engine_runner_shared.secure_wrapper import LLMInvocationError

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]
        request = httpx.Request("POST", "http://oe:8000/tool/stream/exec-123/7")

        with (
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
            ) as mock_approval,
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.connect_sse",
                side_effect=httpx.ReadError("relay timeout", request=request),
            ) as mock_connect_sse,
        ):
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                route_to="http://oe:8000/tool/stream/exec-123/7",
            )

            with pytest.raises(LLMInvocationError, match="relay timeout") as exc_info:
                list(proxy.stream(messages, step=7))
            assert exc_info.value.source is None

        assert mock_connect_sse.call_count == 3
        assert self._sleep_oe_stream_retry.call_count == 2
        assert self._sleep_oe_stream_retry.call_args_list[0].args[0] == 30.0

    def test_stream_raises_on_truncated_oe_sse(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
        from agent_engine_runner_shared.secure_wrapper import LLMInvocationError

        class FakeSSEEvent:
            def __init__(self, data: str):
                self.data = data

        class FakeEventSource:
            response = MagicMock()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def iter_sse(self):
                yield FakeSSEEvent(json.dumps({"content": "partial"}))

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with (
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
            ) as mock_approval,
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.connect_sse",
                return_value=FakeEventSource(),
            ) as mock_connect_sse,
        ):
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                route_to="http://oe:8000/tool/stream/exec-123/7",
            )

            with pytest.raises(
                LLMInvocationError,
                match="truncated response",
            ) as exc_info:
                list(proxy.stream(messages, step=7))
            assert exc_info.value.source is None

        assert mock_connect_sse.call_count == 1
        assert self._sleep_oe_stream_retry.call_count == 0

    def test_stream_retries_truncated_sse_with_no_output(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
        from agent_engine_runner_shared.secure_wrapper import LLMInvocationError
        from agent_engine_runner_shared.utils import OE_DISPATCH_TAKEOVER_RETRY_DELAY_S

        class FakeSSEEvent:
            def __init__(self, data: str):
                self.data = data

        class FakeEventSource:
            response = MagicMock()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def iter_sse(self):
                return iter(())

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with (
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
            ) as mock_approval,
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.connect_sse",
                return_value=FakeEventSource(),
            ) as mock_connect_sse,
        ):
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                route_to="http://oe:8000/tool/stream/exec-123/7",
            )

            with pytest.raises(
                LLMInvocationError,
                match="truncated response",
            ) as exc_info:
                list(proxy.stream(messages, step=7))
            assert exc_info.value.source is None

        assert mock_connect_sse.call_count == 3
        assert self._sleep_oe_stream_retry.call_count == 2
        assert self._sleep_oe_stream_retry.call_args_list[0].args[0] == (
            OE_DISPATCH_TAKEOVER_RETRY_DELAY_S
        )

    def test_stream_stamps_provider_source_on_in_band_sse_error(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
        from agent_engine_runner_shared.secure_wrapper import LLMInvocationError

        class FakeSSEEvent:
            def __init__(self, data: str):
                self.data = data

        class FakeEventSource:
            response = MagicMock()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def iter_sse(self):
                yield FakeSSEEvent(json.dumps({"error": "Resource not found"}))

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with (
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
            ) as mock_approval,
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.connect_sse",
                return_value=FakeEventSource(),
            ),
        ):
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                route_to="http://oe:8000/tool/stream/exec-123/7",
            )

            with pytest.raises(LLMInvocationError, match="Resource not found") as exc_info:
                list(proxy.stream(messages, step=7))
            assert exc_info.value.source == "llm"

    def test_stream_reconnects_on_retryable_sse_error(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        class FakeSSEEvent:
            def __init__(self, data: str):
                self.data = data

        class FakeEventSource:
            def __init__(self, events: list[dict]):
                self._events = events
                self.response = MagicMock()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def iter_sse(self):
                for event in self._events:
                    yield FakeSSEEvent(json.dumps(event))

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]
        stream_url = "http://oe:8000/tool/stream/exec-123/7"

        with (
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
            ) as mock_approval,
            patch("agent_engine_runner_shared.secure_llm_proxy.connect_sse") as mock_connect_sse,
        ):
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                route_to=stream_url,
            )
            mock_connect_sse.side_effect = [
                FakeEventSource(
                    [
                        {
                            "error": "tool pod is no longer reserved for this session; retry request",
                            "retryable": True,
                        }
                    ]
                ),
                FakeEventSource(
                    [
                        {"content": "Hi", "tool_call_chunks": []},
                        {"done": True, "pod_name": "tool-pod-1", "duration_ms": 11.0},
                    ]
                ),
            ]

            chunks = list(proxy.stream(messages, step=7))

        assert [chunk.content for chunk in chunks] == ["Hi"]
        assert mock_connect_sse.call_count == 2
        assert mock_connect_sse.call_args_list[0].args[2] == stream_url
        assert mock_connect_sse.call_args_list[1].args[2] == stream_url

    def test_stream_retries_transport_disconnect_then_takeover(self) -> None:
        """Owner death drops the SSE socket without ReleaseDispatch. The client
        waits the heartbeat TTL, then reconnects on the same URL and recovers.
        """
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
        from agent_engine_runner_shared.utils import OE_DISPATCH_TAKEOVER_RETRY_DELAY_S

        class FakeSSEEvent:
            def __init__(self, data: str):
                self.data = data

        class FakeEventSource:
            def __init__(self, events: list[dict]):
                self._events = events
                self.response = MagicMock()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def iter_sse(self):
                for event in self._events:
                    yield FakeSSEEvent(json.dumps(event))

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]
        stream_url = "http://oe:8000/tool/stream/exec-123/7"
        request = httpx.Request("POST", stream_url)

        with (
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
            ) as mock_approval,
            patch("agent_engine_runner_shared.secure_llm_proxy.connect_sse") as mock_connect_sse,
        ):
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                route_to=stream_url,
            )
            mock_connect_sse.side_effect = [
                httpx.ReadError("connection reset", request=request),
                FakeEventSource(
                    [
                        {
                            "error": "invoke_llm relay already active for this step",
                            "retryable": True,
                            "retry_after_ms": 0,
                        }
                    ]
                ),
                FakeEventSource(
                    [
                        {"content": "recovered", "tool_call_chunks": []},
                        {"done": True, "pod_name": "tool-pod-1", "duration_ms": 11.0},
                    ]
                ),
            ]

            chunks = list(proxy.stream(messages, step=7))

        assert [chunk.content for chunk in chunks if chunk.content] == ["recovered"]
        assert mock_connect_sse.call_count == 3
        assert mock_connect_sse.call_args_list[0].args[2] == stream_url
        assert mock_connect_sse.call_args_list[2].args[2] == stream_url
        assert self._sleep_oe_stream_retry.call_args_list[0].args[0] == (
            OE_DISPATCH_TAKEOVER_RETRY_DELAY_S
        )
        assert self._sleep_oe_stream_retry.call_args_list[1].args[0] == 0.0

    def test_stream_does_not_retry_transport_disconnect_after_content(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
        from agent_engine_runner_shared.secure_wrapper import LLMInvocationError

        class FakeSSEEvent:
            def __init__(self, data: str):
                self.data = data

        class DisconnectAfterContent:
            def __init__(self, request: httpx.Request):
                self._request = request
                self.response = MagicMock()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def iter_sse(self):
                yield FakeSSEEvent(json.dumps({"content": "Hel", "tool_call_chunks": []}))
                raise httpx.ReadError("connection reset", request=self._request)

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]
        stream_url = "http://oe:8000/tool/stream/exec-123/7"
        request = httpx.Request("POST", stream_url)

        with (
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
            ) as mock_approval,
            patch("agent_engine_runner_shared.secure_llm_proxy.connect_sse") as mock_connect_sse,
        ):
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                route_to=stream_url,
            )
            mock_connect_sse.side_effect = [
                DisconnectAfterContent(request),
                DisconnectAfterContent(request),
            ]

            chunks: list = []
            with pytest.raises(LLMInvocationError, match="connection reset"):
                for chunk in proxy.stream(messages, step=7):
                    chunks.append(chunk)

        assert [chunk.content for chunk in chunks] == ["Hel"]
        assert mock_connect_sse.call_count == 1
        assert self._sleep_oe_stream_retry.call_count == 0

    def test_stream_does_not_retry_transport_disconnect_after_tool_call(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
        from agent_engine_runner_shared.secure_wrapper import LLMInvocationError

        class FakeSSEEvent:
            def __init__(self, data: str):
                self.data = data

        class DisconnectAfterToolCall:
            def __init__(self, request: httpx.Request):
                self._request = request
                self.response = MagicMock()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def iter_sse(self):
                yield FakeSSEEvent(
                    json.dumps(
                        {
                            "tool_call_chunks": [
                                {
                                    "id": "call-1",
                                    "name": "search",
                                    "args": '{"q":',
                                    "index": 0,
                                }
                            ]
                        }
                    )
                )
                raise httpx.ReadError("connection reset", request=self._request)

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]
        stream_url = "http://oe:8000/tool/stream/exec-123/7"
        request = httpx.Request("POST", stream_url)

        with (
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
            ) as mock_approval,
            patch("agent_engine_runner_shared.secure_llm_proxy.connect_sse") as mock_connect_sse,
        ):
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                route_to=stream_url,
            )
            mock_connect_sse.return_value = DisconnectAfterToolCall(request)

            chunks: list = []
            with pytest.raises(LLMInvocationError, match="connection reset"):
                for chunk in proxy.stream(messages, step=7):
                    chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].tool_calls is not None
        assert chunks[0].tool_calls[0].name == "search"
        assert mock_connect_sse.call_count == 1
        assert self._sleep_oe_stream_retry.call_count == 0

    def test_stream_does_not_retry_consumer_close(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        class FakeSSEEvent:
            def __init__(self, data: str):
                self.data = data

        class FakeEventSource:
            response = MagicMock()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def iter_sse(self):
                yield FakeSSEEvent(json.dumps({"content": "Hel", "tool_call_chunks": []}))
                yield FakeSSEEvent(json.dumps({"content": "lo", "tool_call_chunks": []}))
                yield FakeSSEEvent(
                    json.dumps({"done": True, "pod_name": "tool-pod-1", "duration_ms": 11.0})
                )

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with (
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
            ) as mock_approval,
            patch("agent_engine_runner_shared.secure_llm_proxy.connect_sse") as mock_connect_sse,
        ):
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                route_to="http://oe:8000/tool/stream/exec-123/7",
            )
            mock_connect_sse.return_value = FakeEventSource()

            gen = proxy.stream(messages, step=7)
            assert next(gen).content == "Hel"
            gen.close()

        assert mock_connect_sse.call_count == 1
        self._sleep_oe_stream_retry.assert_not_called()

    def test_stream_yields_interrupted_marker_on_oe_interrupt(self) -> None:
        """An OE-signalled interrupt ends the stream cleanly with a marker, not an
        error, so the agent graph can continue."""
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        class FakeSSEEvent:
            def __init__(self, data: str):
                self.data = data

        class FakeEventSource:
            response = MagicMock()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def iter_sse(self):
                yield FakeSSEEvent(json.dumps({"content": "Par", "tool_call_chunks": []}))
                # OE aborts the in-flight call with a bare sentinel, then the stream
                # ends. The relay emits only {"interrupted": true} — no pod/duration.
                yield FakeSSEEvent(json.dumps({"interrupted": True}))

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with (
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
            ) as mock_approval,
            patch(
                "agent_engine_runner_shared.secure_llm_proxy.connect_sse",
                return_value=FakeEventSource(),
            ),
        ):
            mock_approval.return_value = ToolExecuteResponse(
                proceed=True,
                route_to="http://oe:8000/tool/stream/exec-123/7",
            )

            # Must NOT raise (unlike the truncated-SSE case): interrupt is a clean stop.
            chunks = list(proxy.stream(messages, step=7))

        assert chunks[0].content == "Par"
        assert chunks[-1].response_metadata == {"interrupted": True}


class TestSerializeBoundTools:
    def test_tools_with_schema(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        mock_tool = MagicMock()
        mock_schema = MagicMock()
        mock_schema.schema.return_value = {"name": "my_tool", "type": "object"}
        mock_tool.tool_call_schema = mock_schema
        proxy.bound_tools = [mock_tool]

        assert proxy._serialize_bound_tools() == [{"name": "my_tool", "type": "object"}]

    def test_tools_with_json_schema_dict(self) -> None:
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-123", llm_id="primary"
        )
        mock_tool = MagicMock()
        mock_tool.name = "github__search_issues"
        mock_tool.description = "Search issues"
        mock_tool.tool_call_schema = {
            "description": "Search issues",
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        proxy.bound_tools = [mock_tool]

        assert proxy._serialize_bound_tools() == [
            {
                "name": "github__search_issues",
                "description": "Search issues",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]


class TestSecureLLMProxyGuardrailRequireReview:
    """Tests for the guardrail require_review suspension/resume flow."""

    @pytest.fixture(autouse=True)
    def _reset_suspend_handler(self) -> Iterator[None]:
        from agent_engine_runner_shared.hooks import register_suspend_handler

        register_suspend_handler(None)
        yield
        register_suspend_handler(None)

    @staticmethod
    def _make_require_review_response() -> ToolExecuteResponse:
        return ToolExecuteResponse(
            proceed=False,
            status="require_review",
            result=None,
            reason="policy violated",
        )

    def test_request_oe_execution_allows_require_review_to_propagate(self) -> None:
        """_request_oe_execution must NOT raise PolicyDeniedException for require_review."""
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-rr", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=False,
                status="require_review",
                result=None,
                reason="1 guardrail policy requires human review",
            )
            # Patch _handle_require_review so stream() doesn't try to call interrupt()
            with patch.object(proxy, "_handle_require_review", return_value=iter([])):
                list(proxy.stream(messages))

        mock_approval.assert_called_once()

    def test_stream_calls_interrupt_on_require_review(self) -> None:
        """stream() must call interrupt() when OE returns require_review."""
        from agent_engine_runner_shared.hooks import register_suspend_handler
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-rr", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        interrupt_calls: list = []

        class SimulatedInterrupt(Exception):
            pass

        def fake_interrupt(payload: dict) -> None:
            interrupt_calls.append(payload)
            raise SimulatedInterrupt("suspended by LangGraph")

        register_suspend_handler(fake_interrupt)

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=False,
                status="require_review",
                result=None,
                reason="Guardrail requires review",
            )

            with pytest.raises(SimulatedInterrupt):
                list(proxy.stream(messages))

        assert len(interrupt_calls) == 1
        payload = interrupt_calls[0]
        assert payload["suspend_reason"] == "guardrail_require_review"
        assert payload["allowed_decisions"] == ["approve", "deny"]

    def test_stream_suspend_payload_includes_guardrail_meta(self) -> None:
        """suspend_payload must include guardrail_id and guardrail_category when OE returns them."""
        from agent_engine_runner_shared.hooks import register_suspend_handler
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-rr-meta", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        interrupt_calls: list = []

        class SimulatedInterrupt(Exception):
            pass

        def fake_interrupt(payload: dict) -> None:
            interrupt_calls.append(payload)
            raise SimulatedInterrupt("suspended")

        register_suspend_handler(fake_interrupt)

        policy_id = "6641abc123"
        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            from agent_engine_runner_shared.models import GuardrailMeta

            mock_approval.return_value = ToolExecuteResponse(
                proceed=False,
                status="require_review",
                result=None,
                reason="Guardrail requires review",
                guardrail_meta=GuardrailMeta(
                    guardrail_id=policy_id,
                    guardrail_category="output_validation",
                ),
            )

            with pytest.raises(SimulatedInterrupt):
                list(proxy.stream(messages))

        assert len(interrupt_calls) == 1
        payload = interrupt_calls[0]
        assert payload["guardrail_meta"]["guardrail_id"] == policy_id
        assert payload["guardrail_meta"]["guardrail_category"] == "output_validation"

    def test_stream_approve_yields_pending_llm_content(self) -> None:
        """On resume with approve, stream() yields the original LLM content chunks."""
        from agent_engine_runner_shared.hooks import register_suspend_handler
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-rr", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        pending_content = {"content": "original LLM output", "tool_calls": None, "metadata": {}}

        def fake_interrupt(payload: dict) -> dict:
            return {
                "guardrail_review": {
                    "decision": "approve",
                    "pending_llm_content": pending_content,
                }
            }

        register_suspend_handler(fake_interrupt)

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=False,
                status="require_review",
                result=None,
                reason="Guardrail requires review",
            )

            chunks = list(proxy.stream(messages))

        assert len(chunks) >= 1
        assert chunks[0].content == "original LLM output"

    def test_stream_deny_raises_policy_denied(self) -> None:
        """On resume with deny, stream() raises PolicyDeniedException."""
        from agent_engine_runner_shared.hooks import register_suspend_handler
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
        from agent_engine_runner_shared.secure_wrapper import PolicyDeniedException

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-rr", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        def fake_interrupt(payload: dict) -> dict:
            return {"guardrail_review": {"decision": "deny"}}

        register_suspend_handler(fake_interrupt)

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=False,
                status="require_review",
                result=None,
                reason="Guardrail requires review",
            )

            with pytest.raises(PolicyDeniedException, match="denied by reviewer"):
                list(proxy.stream(messages))

    def test_stream_no_suspend_handler_raises_policy_denied(self) -> None:
        """stream() raises PolicyDeniedException when no suspend handler is registered."""
        from agent_engine_runner_shared.hooks import register_suspend_handler
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
        from agent_engine_runner_shared.secure_wrapper import PolicyDeniedException

        register_suspend_handler(None)

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-rr", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=False,
                status="require_review",
                result=None,
                reason="Guardrail requires review",
            )

            with pytest.raises(PolicyDeniedException, match="no suspend handler registered"):
                list(proxy.stream(messages))

    def test_stream_approve_missing_pending_content_raises_llm_error(self) -> None:
        """Approve with no pending_llm_content in resume data raises LLMInvocationError."""
        from agent_engine_runner_shared.hooks import register_suspend_handler
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
        from agent_engine_runner_shared.secure_wrapper import LLMInvocationError

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-rr", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        def fake_interrupt(payload: dict) -> dict:
            return {"guardrail_review": {"decision": "approve"}}

        register_suspend_handler(fake_interrupt)

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=False,
                status="require_review",
                result=None,
                reason="Guardrail requires review",
            )

            with pytest.raises(LLMInvocationError, match="pending LLM content"):
                list(proxy.stream(messages))

    def test_stream_non_dict_human_decision_raises(self) -> None:
        """Non-dict return from interrupt() raises LLMInvocationError."""
        from agent_engine_runner_shared.hooks import register_suspend_handler
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
        from agent_engine_runner_shared.secure_wrapper import LLMInvocationError

        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080", execution_id="exec-rr", llm_id="primary"
        )
        messages = [Message(role="user", content="Hello")]

        def fake_interrupt(payload: dict) -> str:
            return "not a dict"

        register_suspend_handler(fake_interrupt)

        with patch(
            "agent_engine_runner_shared.secure_llm_proxy.request_oe_approval"
        ) as mock_approval:
            mock_approval.return_value = ToolExecuteResponse(
                proceed=False,
                status="require_review",
                result=None,
                reason="Guardrail requires review",
            )

            with pytest.raises(LLMInvocationError, match="expected dict from interrupt"):
                list(proxy.stream(messages))

    def test_missing_guardrail_review_key_raises_llm_error(self) -> None:
        """When interrupt() returns a dict without 'guardrail_review', raise LLMInvocationError."""
        from agent_engine_runner_shared.hooks import register_suspend_handler
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
        from agent_engine_runner_shared.secure_wrapper import LLMInvocationError

        def fake_interrupt(payload: dict) -> dict:
            # Wrong key — simulates OE/Python version skew
            return {"human_review": {"decision": "approve"}}

        register_suspend_handler(fake_interrupt)
        proxy = SecureLLMProxy(
            oe_url="http://oe",
            execution_id="exec-123",
            llm_id="primary",
        )
        mock_response = self._make_require_review_response()

        with pytest.raises(LLMInvocationError, match="guardrail_review"):
            list(proxy._handle_require_review(mock_response))
