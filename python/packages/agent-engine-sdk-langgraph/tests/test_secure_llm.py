"""Tests for SecureWrappedLLM in agent-engine-sdk-langgraph."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatResult
from agent_engine_sdk.models import (
    LLMInvocationOptions,
    LLMResponse,
    LLMStreamChunk,
    ToolCallChunk,
)


class TestSecureWrappedLLMConstruction:
    def test_construction(self) -> None:
        from agent_engine_sdk_langgraph.secure_llm import SecureWrappedLLM

        mock_llm = MagicMock()
        mock_llm.model = "gpt-4o"
        wrapped = SecureWrappedLLM(
            llm=mock_llm, get_wrapper=MagicMock(return_value=None), llm_id="primary"
        )

        assert wrapped._llm is mock_llm
        assert wrapped.model_name == "gpt-4o"

    def test_model_name_from_model_name_attr(self) -> None:
        from agent_engine_sdk_langgraph.secure_llm import SecureWrappedLLM

        mock_llm = MagicMock(spec=[])
        mock_llm.model_name = "gemini-2.5-flash"
        wrapped = SecureWrappedLLM(
            llm=mock_llm, get_wrapper=MagicMock(return_value=None), llm_id="primary"
        )

        assert wrapped.model_name == "gemini-2.5-flash"


class TestSecureWrappedLLMIdentifyingParams:
    def test_identifying_params_includes_llm_id(self) -> None:
        """_identifying_params must include llm_id to prevent cache collisions."""
        from agent_engine_sdk_langgraph.secure_llm import SecureWrappedLLM

        mock_llm = MagicMock()
        mock_llm.model = "gpt-4o"
        wrapped = SecureWrappedLLM(
            llm=mock_llm, get_wrapper=MagicMock(return_value=None), llm_id="primary"
        )

        params = wrapped._identifying_params
        assert params["llm_id"] == "primary"

    def test_different_llm_ids_produce_different_params(self) -> None:
        """Different llm_ids produce different identifying params."""
        from agent_engine_sdk_langgraph.secure_llm import SecureWrappedLLM

        mock_llm = MagicMock()
        mock_llm.model = "gpt-4o"
        primary = SecureWrappedLLM(
            llm=mock_llm, get_wrapper=MagicMock(return_value=None), llm_id="primary"
        )
        secondary = SecureWrappedLLM(
            llm=mock_llm, get_wrapper=MagicMock(return_value=None), llm_id="secondary"
        )

        assert primary._identifying_params != secondary._identifying_params


class TestSecureWrappedLLMBindTools:
    def test_bind_tools_delegates_when_supported(self) -> None:
        from agent_engine_sdk_langgraph.secure_llm import SecureWrappedLLM

        mock_inner_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_inner_llm.bind_tools.return_value = mock_bound_llm
        mock_inner_llm.model = "gpt-4o"

        wrapped = SecureWrappedLLM(
            llm=mock_inner_llm,
            get_wrapper=MagicMock(return_value=None),
            llm_id="primary",
        )

        tools = [{"name": "test_tool"}]
        result = wrapped.bind_tools(tools, tool_choice="auto")

        mock_inner_llm.bind_tools.assert_called_once_with(tools, tool_choice="auto")
        assert result._llm is mock_bound_llm
        assert result._bound_tools == tools

    def test_bind_tools_preserves_llm_id(self) -> None:
        from agent_engine_sdk_langgraph.secure_llm import SecureWrappedLLM

        mock_inner_llm = MagicMock()
        mock_inner_llm.model = "gpt-4o"
        mock_inner_llm.bind_tools.return_value = MagicMock()

        wrapped = SecureWrappedLLM(
            llm=mock_inner_llm,
            get_wrapper=MagicMock(return_value=None),
            llm_id="primary",
        )
        result = wrapped.bind_tools([{"name": "search"}])
        assert result._llm_id == "primary"

    def test_bind_tools_falls_back_when_not_supported(self) -> None:
        from agent_engine_sdk_langgraph.secure_llm import SecureWrappedLLM

        mock_inner_llm = MagicMock(spec=[])
        mock_inner_llm.model = "custom-llm"
        wrapped = SecureWrappedLLM(
            llm=mock_inner_llm,
            get_wrapper=MagicMock(return_value=None),
            llm_id="primary",
        )

        tools = [{"name": "test_tool"}]
        result = wrapped.bind_tools(tools)

        assert result._llm is mock_inner_llm
        assert result._bound_tools == tools

    def test_bind_tools_converts_bare_pydantic_schema_for_proxy(self) -> None:
        """A raw Pydantic class (as bound by with_structured_output) must be
        stored in a form SecureLLMProxy can serialize, not dropped.

        Regression guard: SecureLLMProxy._serialize_bound_tools skips a
        bare ``ModelMetaclass`` ("Skipping unrecognized tool type"), so the
        structured-output tool never reached the model and the parsed result
        was always None. agent-engine-runner-shared is framework-neutral, so the LangChain
        conversion must happen here in bind_tools.
        """
        from pydantic import BaseModel

        from agent_engine_sdk_langgraph.secure_llm import SecureWrappedLLM
        from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy

        class Brief(BaseModel):
            """An account brief."""

            sections: list[str]

        mock_inner_llm = MagicMock()
        mock_inner_llm.model = "gpt-4o"
        mock_inner_llm.bind_tools.return_value = MagicMock(spec=[])

        wrapped = SecureWrappedLLM(
            llm=mock_inner_llm,
            get_wrapper=MagicMock(return_value=None),
            llm_id="primary",
        )
        result = wrapped.bind_tools([Brief], tool_choice="Brief")

        # The real proxy must serialize the stored tool rather than drop it.
        proxy = SecureLLMProxy(
            oe_url="http://localhost:8080",
            execution_id="exec-123",
            bound_tools=list(result._bound_tools),
        )
        serialized = proxy._serialize_bound_tools()
        assert serialized is not None
        assert len(serialized) == 1
        langchain_dict = serialized[0].to_langchain_dict()
        function = langchain_dict.get("function")
        assert isinstance(function, dict)
        assert function["name"] == "Brief"

    def test_bind_tools_preserves_basetool_and_dict_unchanged(self) -> None:
        """Tools the proxy already handles natively (LangChain BaseTools and
        plain dicts) must pass through bind_tools unchanged, so existing
        serialization behavior is untouched."""
        from langchain_core.tools import tool

        from agent_engine_sdk_langgraph.secure_llm import SecureWrappedLLM

        @tool
        def search(query: str) -> str:
            """Search."""
            return query

        mock_inner_llm = MagicMock()
        mock_inner_llm.model = "gpt-4o"
        mock_inner_llm.bind_tools.return_value = MagicMock(spec=[])

        wrapped = SecureWrappedLLM(
            llm=mock_inner_llm,
            get_wrapper=MagicMock(return_value=None),
            llm_id="primary",
        )
        dict_tool = {"name": "lookup", "parameters": {"type": "object"}}
        result = wrapped.bind_tools([search, dict_tool])

        assert result._bound_tools[0] is search
        assert result._bound_tools[1] is dict_tool

    def test_normalize_bound_tool_falls_back_on_unconvertible_type(self) -> None:
        """A tool type ``convert_to_openai_tool`` cannot interpret must be passed
        through unchanged with a warning, not raise at bind_tools time. This
        preserves the pre-AP-2933 silent-skip safety net in SecureLLMProxy."""
        from agent_engine_sdk_langgraph.secure_llm import _normalize_bound_tool

        class Unconvertible:
            """No tool_call_schema; convert_to_openai_tool cannot handle it."""

        exotic = Unconvertible()

        assert _normalize_bound_tool(exotic) is exotic

    def test_bind_tools_captures_tool_choice(self) -> None:
        """tool_choice passed to bind_tools must be captured so it can be
        forwarded to the tool pod's bind_tools call. Without it the
        structured-output tool is offered but never forced."""
        from agent_engine_sdk_langgraph.secure_llm import SecureWrappedLLM

        mock_inner_llm = MagicMock()
        mock_inner_llm.model = "gpt-4o"
        mock_inner_llm.bind_tools.return_value = MagicMock(spec=[])

        wrapped = SecureWrappedLLM(
            llm=mock_inner_llm,
            get_wrapper=MagicMock(return_value=None),
            llm_id="primary",
        )
        result = wrapped.bind_tools([{"name": "Brief"}], tool_choice="Brief")
        assert result._bound_tool_choice == "Brief"

    def test_bind_tools_tool_choice_defaults_none(self) -> None:
        from agent_engine_sdk_langgraph.secure_llm import SecureWrappedLLM

        mock_inner_llm = MagicMock()
        mock_inner_llm.model = "gpt-4o"
        mock_inner_llm.bind_tools.return_value = MagicMock(spec=[])

        wrapped = SecureWrappedLLM(
            llm=mock_inner_llm,
            get_wrapper=MagicMock(return_value=None),
            llm_id="primary",
        )
        result = wrapped.bind_tools([{"name": "search"}])
        assert result._bound_tool_choice is None


def _make_wrapper_and_wrapped():
    from agent_engine_sdk_langgraph.secure_llm import SecureWrappedLLM
    from agent_engine_runner_shared.secure_wrapper import OperationalStepAllocator

    mock_llm = MagicMock()
    mock_llm.model = "gpt-4o"

    mock_wrapper = MagicMock()
    mock_wrapper.oe_url = "http://localhost:8080"
    mock_wrapper.execution_id = "exec-123"
    mock_wrapper.operational_steps = OperationalStepAllocator()
    mock_wrapper.step_counter = 0
    mock_wrapper.durable_memory = None

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
    )
    return mock_llm, mock_wrapper, wrapped


@pytest.mark.parametrize("transport", ["generate", "stream"])
def test_deep_agent_message_ids_are_stamped_before_proxy(transport: str) -> None:
    from agent_engine_sdk_langgraph.durable_deep_agent import _deep_agent_task_scope

    _, _, wrapped = _make_wrapper_and_wrapped()
    messages = [
        HumanMessage(content="research", id="random-input"),
        ToolMessage(
            content="updated",
            tool_call_id="write-todos-a",
            id="random-tool-result",
        ),
    ]

    with patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy") as MockProxy:
        mock_proxy = MockProxy.return_value
        mock_proxy.stream.return_value = iter([LLMStreamChunk(content="Done")])
        mock_proxy.response_from_stream_chunks.return_value = LLMResponse(
            content="Done"
        )
        mock_proxy.last_duration_ms = 0.0
        mock_proxy.last_from_cache = False
        mock_proxy.last_latest_step_number = None

        with _deep_agent_task_scope("task-a"):
            if transport == "generate":
                wrapped._generate(messages)
            else:
                list(wrapped._stream(messages))

    expected_ids = [
        "durable-deep-agent-input:task-a:0",
        "durable-deep-agent-tool-result:task-a:write-todos-a",
    ]
    assert [message.id for message in messages] == expected_ids
    sent_messages = mock_proxy.stream.call_args.kwargs["messages"]
    assert [message.id for message in sent_messages] == expected_ids


class TestSecureWrappedLLMGenerate:
    def test_generate_raises_without_wrapper(self) -> None:
        from agent_engine_sdk_langgraph.secure_llm import SecureWrappedLLM

        mock_llm = MagicMock()
        mock_llm.model = "gpt-4o"
        wrapped = SecureWrappedLLM(
            llm=mock_llm, get_wrapper=MagicMock(return_value=None), llm_id="primary"
        )

        with pytest.raises(RuntimeError, match="SecureToolWrapper is not initialized"):
            wrapped._generate([HumanMessage(content="Hello")])

    @patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy")
    def test_generate_routes_through_proxy_with_stop_and_kwargs(
        self, MockProxy
    ) -> None:
        _, mock_wrapper, wrapped = _make_wrapper_and_wrapped()
        mock_proxy = MockProxy.return_value
        mock_proxy.stream.return_value = iter([LLMStreamChunk(content="Hi")])
        mock_proxy.response_from_stream_chunks.return_value = LLMResponse(content="Hi")
        mock_proxy.last_duration_ms = 12.0
        mock_proxy.last_from_cache = False
        mock_proxy.last_latest_step_number = 4

        result = wrapped._generate(
            [HumanMessage(content="Hello")], stop=["END"], max_tokens=32
        )

        assert isinstance(result, ChatResult)
        assert result.generations[0].message.content == "Hi"
        MockProxy.assert_called_once_with(
            oe_url="http://localhost:8080",
            execution_id="exec-123",
            llm_id="primary",
            model_name="gpt-4o",
            bound_tools=None,
            bound_tool_choice=None,
            operational_steps=mock_wrapper.operational_steps,
            durable_memory=None,
            guardrail_review_protocol=True,
        )
        call_kwargs = mock_proxy.stream.call_args.kwargs
        assert call_kwargs["step"] == 1
        assert call_kwargs["stop"] == ["END"]
        assert call_kwargs["options"] == LLMInvocationOptions(max_tokens=32)
        assert mock_wrapper.step_counter == 4

    @patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy")
    def test_generate_forwards_bound_tool_choice_to_proxy(self, MockProxy) -> None:
        """tool_choice captured by bind_tools must reach SecureLLMProxy so it can
        be forwarded to the tool pod and force the structured-output tool call."""
        _, _, wrapped = _make_wrapper_and_wrapped()
        bound = wrapped.bind_tools([{"name": "Brief"}], tool_choice="Brief")

        mock_proxy = MockProxy.return_value
        mock_proxy.stream.return_value = iter([LLMStreamChunk(content="Hi")])
        mock_proxy.response_from_stream_chunks.return_value = LLMResponse(content="Hi")
        mock_proxy.last_duration_ms = 0.0
        mock_proxy.last_from_cache = False
        mock_proxy.last_latest_step_number = None

        bound._generate([HumanMessage(content="Hello")])

        assert MockProxy.call_args.kwargs["bound_tool_choice"] == "Brief"

    @patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy")
    def test_generate_repairs_missing_tool_messages_before_proxy(
        self, MockProxy
    ) -> None:
        _, _, wrapped = _make_wrapper_and_wrapped()
        mock_proxy = MockProxy.return_value
        mock_proxy.stream.return_value = iter([LLMStreamChunk(content="Recovered")])
        mock_proxy.response_from_stream_chunks.return_value = LLMResponse(
            content="Recovered"
        )
        mock_proxy.last_duration_ms = 0.0
        mock_proxy.last_from_cache = False
        mock_proxy.last_latest_step_number = None

        wrapped._generate(
            [
                HumanMessage(content="Check Sentry"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_sentry",
                            "name": "sentry__search_issues",
                            "args": {"query": "project:agentengine-cli"},
                        }
                    ],
                ),
                HumanMessage(content="Now use Glean"),
            ]
        )

        sent_messages = mock_proxy.stream.call_args.kwargs["messages"]
        assert [message.role for message in sent_messages] == [
            "user",
            "assistant",
            "tool",
            "user",
        ]
        assert sent_messages[2].tool_call_id == "call_sentry"
        assert sent_messages[2].name == "sentry__search_issues"
        assert (
            "failed before the platform recorded a result" in sent_messages[2].content
        )

    @patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy")
    def test_generate_preserves_existing_tool_messages(self, MockProxy) -> None:
        _, _, wrapped = _make_wrapper_and_wrapped()
        mock_proxy = MockProxy.return_value
        mock_proxy.stream.return_value = iter([LLMStreamChunk(content="Done")])
        mock_proxy.response_from_stream_chunks.return_value = LLMResponse(
            content="Done"
        )
        mock_proxy.last_duration_ms = 0.0
        mock_proxy.last_from_cache = False
        mock_proxy.last_latest_step_number = None

        wrapped._generate(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_search",
                            "name": "search",
                            "args": {"query": "hello"},
                        }
                    ],
                ),
                ToolMessage(
                    content="existing result",
                    tool_call_id="call_search",
                    name="search",
                ),
                HumanMessage(content="continue"),
            ]
        )

        sent_messages = mock_proxy.stream.call_args.kwargs["messages"]
        assert len(sent_messages) == 3
        assert sent_messages[1].content == "existing result"

    @patch("agent_engine_sdk_langgraph.secure_llm.log_cached_result")
    @patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy")
    def test_generate_logs_cached_results_from_proxy(
        self, MockProxy, mock_log_cached
    ) -> None:
        _, _, wrapped = _make_wrapper_and_wrapped()
        mock_proxy = MockProxy.return_value
        mock_proxy.stream.return_value = iter([LLMStreamChunk(content="Cached")])
        mock_proxy.response_from_stream_chunks.return_value = LLMResponse(
            content="Cached"
        )
        mock_proxy.last_duration_ms = 0.0
        mock_proxy.last_from_cache = True
        mock_proxy.last_latest_step_number = None

        wrapped._generate([HumanMessage(content="Hello")])

        mock_log_cached.assert_called_once_with("invoke_llm", 1, prefix="LLM")

    @patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy")
    def test_generate_raises_on_policy_denial(self, MockProxy) -> None:
        from agent_engine_runner_shared.secure_wrapper import PolicyDeniedException

        _, _, wrapped = _make_wrapper_and_wrapped()
        MockProxy.return_value.stream.side_effect = PolicyDeniedException(
            "Rate limited"
        )

        with pytest.raises(PolicyDeniedException, match="Rate limited"):
            wrapped._generate([HumanMessage(content="Hello")])

    @patch("agent_engine_sdk_langgraph.secure_llm.Metrics")
    @patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy")
    def test_generate_records_latency_metric(self, MockProxy, MockMetrics) -> None:
        _, _, wrapped = _make_wrapper_and_wrapped()
        mock_proxy = MockProxy.return_value
        mock_proxy.stream.return_value = iter([LLMStreamChunk(content="Hi")])
        mock_proxy.response_from_stream_chunks.return_value = LLMResponse(content="Hi")
        mock_proxy.last_duration_ms = 150.0
        mock_proxy.last_from_cache = False
        mock_proxy.last_latest_step_number = None

        wrapped._generate([HumanMessage(content="Hello")])

        MockMetrics.record_latency.assert_called_once_with(
            "llm_call", 150.0, model_name="gpt-4o"
        )

    @patch("agent_engine_sdk_langgraph.secure_llm.Metrics")
    @patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy")
    def test_generate_records_error_metric_on_llm_error(
        self, MockProxy, MockMetrics
    ) -> None:
        from agent_engine_runner_shared.secure_wrapper import LLMInvocationError

        _, _, wrapped = _make_wrapper_and_wrapped()
        mock_proxy = MockProxy.return_value
        mock_proxy.stream.side_effect = LLMInvocationError("boom")
        mock_proxy.last_duration_ms = 25.0

        with pytest.raises(LLMInvocationError, match="boom"):
            wrapped._generate([HumanMessage(content="Hello")])

        MockMetrics.record_latency.assert_called_once_with(
            "llm_call", 25.0, model_name="gpt-4o"
        )
        MockMetrics.record_error.assert_called_once_with(
            "llm_call", model_name="gpt-4o"
        )


class TestSecureWrappedLLMStream:
    @patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy")
    def test_stream_forwards_bound_tool_choice_to_proxy(self, MockProxy) -> None:
        """tool_choice must also reach the proxy on the streaming path."""
        _, _, wrapped = _make_wrapper_and_wrapped()
        bound = wrapped.bind_tools([{"name": "Brief"}], tool_choice="Brief")

        mock_proxy = MockProxy.return_value
        mock_proxy.last_duration_ms = 0.0
        mock_proxy.last_from_cache = False
        mock_proxy.last_latest_step_number = None
        mock_proxy.stream.return_value = iter([LLMStreamChunk(content="Hi")])

        list(bound._stream([HumanMessage(content="Hello")]))

        assert MockProxy.call_args.kwargs["bound_tool_choice"] == "Brief"

    def test_stream_raises_without_wrapper(self) -> None:
        from agent_engine_sdk_langgraph.secure_llm import SecureWrappedLLM

        mock_llm = MagicMock()
        mock_llm.model = "gpt-4o"
        wrapped = SecureWrappedLLM(
            llm=mock_llm, get_wrapper=MagicMock(return_value=None), llm_id="primary"
        )

        with pytest.raises(RuntimeError, match="SecureToolWrapper is not initialized"):
            list(wrapped._stream([HumanMessage(content="Hello")]))

    @patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy")
    def test_stream_yields_converted_chunks(self, MockProxy) -> None:
        _, mock_wrapper, wrapped = _make_wrapper_and_wrapped()
        mock_proxy = MockProxy.return_value
        mock_proxy.last_duration_ms = 100.0
        mock_proxy.last_from_cache = False
        mock_proxy.last_latest_step_number = 5
        mock_proxy.stream.return_value = iter(
            [
                LLMStreamChunk(content="Hello"),
                LLMStreamChunk(
                    tool_calls=[ToolCallChunk(id="call_1", name="search", index=0)]
                ),
            ]
        )

        chunks = list(
            wrapped._stream([HumanMessage(content="Hi")], stop=["DONE"], max_tokens=64)
        )

        assert len(chunks) == 2
        assert chunks[0].message.content == "Hello"
        tc = chunks[1].message.tool_call_chunks[0]  # type: ignore[union-attr]
        assert tc["name"] == "search"
        call_kwargs = mock_proxy.stream.call_args.kwargs
        assert call_kwargs["step"] == 1
        assert call_kwargs["stop"] == ["DONE"]
        assert call_kwargs["options"] == LLMInvocationOptions(max_tokens=64)
        assert mock_wrapper.step_counter == 5

    @patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy")
    def test_stream_repairs_missing_tool_messages_before_proxy(self, MockProxy) -> None:
        _, _, wrapped = _make_wrapper_and_wrapped()
        mock_proxy = MockProxy.return_value
        mock_proxy.last_duration_ms = 0.0
        mock_proxy.last_from_cache = False
        mock_proxy.last_latest_step_number = None
        mock_proxy.stream.return_value = iter([LLMStreamChunk(content="Recovered")])

        list(
            wrapped._stream(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "call_tool",
                                "name": "remote_tool",
                                "args": {"query": "hello"},
                            }
                        ],
                    ),
                    HumanMessage(content="continue"),
                ]
            )
        )

        sent_messages = mock_proxy.stream.call_args.kwargs["messages"]
        assert [message.role for message in sent_messages] == [
            "assistant",
            "tool",
            "user",
        ]
        assert sent_messages[1].tool_call_id == "call_tool"

    @patch("agent_engine_sdk_langgraph.secure_llm.log_cached_result")
    @patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy")
    def test_stream_logs_cached_results_from_proxy(
        self, MockProxy, mock_log_cached
    ) -> None:
        _, _, wrapped = _make_wrapper_and_wrapped()
        mock_proxy = MockProxy.return_value
        mock_proxy.last_duration_ms = 0.0
        mock_proxy.last_from_cache = True
        mock_proxy.last_latest_step_number = None
        mock_proxy.stream.return_value = iter([LLMStreamChunk(content="Cached")])

        list(wrapped._stream([HumanMessage(content="Hi")]))

        mock_log_cached.assert_called_once_with("invoke_llm", 1, prefix="LLM")

    @patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy")
    def test_stream_fallback_to_generate_when_no_stream(self, MockProxy) -> None:
        mock_llm, mock_wrapper, wrapped = _make_wrapper_and_wrapped()
        del mock_llm.stream
        mock_proxy = MockProxy.return_value
        mock_proxy.stream.return_value = iter([LLMStreamChunk(content="Fallback")])
        mock_proxy.response_from_stream_chunks.return_value = LLMResponse(
            content="Fallback"
        )
        mock_proxy.last_duration_ms = 0.0
        mock_proxy.last_from_cache = False
        mock_proxy.last_latest_step_number = None

        chunks = list(wrapped._stream([HumanMessage(content="Hello")]))

        assert len(chunks) == 1
        assert chunks[0].message.content == "Fallback"
        assert mock_wrapper.step_counter == 1
        mock_proxy.stream.assert_called_once()

    @patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy")
    def test_stream_raises_on_policy_denial(self, MockProxy) -> None:
        from agent_engine_runner_shared.secure_wrapper import PolicyDeniedException

        _, _, wrapped = _make_wrapper_and_wrapped()
        MockProxy.return_value.stream.side_effect = PolicyDeniedException("Denied")

        with pytest.raises(PolicyDeniedException, match="Denied"):
            list(wrapped._stream([HumanMessage(content="Hello")]))

    @patch("agent_engine_sdk_langgraph.secure_llm.Metrics")
    @patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy")
    def test_stream_records_error_metric_on_llm_error(
        self, MockProxy, MockMetrics
    ) -> None:
        from agent_engine_runner_shared.secure_wrapper import LLMInvocationError

        _, _, wrapped = _make_wrapper_and_wrapped()
        mock_proxy = MockProxy.return_value
        mock_proxy.stream.side_effect = LLMInvocationError("boom")
        mock_proxy.last_duration_ms = 30.0

        with pytest.raises(LLMInvocationError, match="boom"):
            list(wrapped._stream([HumanMessage(content="Hello")]))

        MockMetrics.record_latency.assert_called_once_with(
            "llm_call", 30.0, model_name="gpt-4o"
        )
        MockMetrics.record_error.assert_called_once_with(
            "llm_call", model_name="gpt-4o"
        )


class TestEnrichSpanWithTokenCounts:
    def test_sets_prompt_completion_total_on_recording_span(self) -> None:
        from agent_engine_sdk.models import LLMTokenUsage

        from agent_engine_sdk_langgraph.secure_llm import _enrich_span_with_token_counts

        usage = LLMTokenUsage(prompt_tokens=100, completion_tokens=40, total_tokens=140)
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        with patch("agent_engine_sdk_langgraph.secure_llm.trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            _enrich_span_with_token_counts(usage)

        mock_span.set_attribute.assert_any_call("llm.token_count.prompt", 100)
        mock_span.set_attribute.assert_any_call("llm.token_count.completion", 40)
        mock_span.set_attribute.assert_any_call("llm.token_count.total", 140)

    def test_uses_input_output_tokens_as_fallback(self) -> None:
        from agent_engine_sdk.models import LLMTokenUsage

        from agent_engine_sdk_langgraph.secure_llm import _enrich_span_with_token_counts

        usage = LLMTokenUsage(input_tokens=50, output_tokens=10)
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        with patch("agent_engine_sdk_langgraph.secure_llm.trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            _enrich_span_with_token_counts(usage)

        mock_span.set_attribute.assert_any_call("llm.token_count.prompt", 50)
        mock_span.set_attribute.assert_any_call("llm.token_count.completion", 10)

    def test_sets_gemini_cache_read_and_reasoning_from_nested_details(self) -> None:
        """Mirrors LangChain's real usage_metadata shape: cache_read lives under
        input_token_details and reasoning under output_token_details."""
        from agent_engine_sdk.models import LLMTokenUsage

        from agent_engine_sdk_langgraph.secure_llm import _enrich_span_with_token_counts

        usage = LLMTokenUsage.model_validate(
            {
                "input_tokens": 200,
                "output_tokens": 80,
                "total_tokens": 280,
                "input_token_details": {"cache_read": 30},
                "output_token_details": {"reasoning": 15},
            }
        )
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        with patch("agent_engine_sdk_langgraph.secure_llm.trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            _enrich_span_with_token_counts(usage)

        mock_span.set_attribute.assert_any_call(
            "llm.token_count.prompt_details.cache_read", 30
        )
        mock_span.set_attribute.assert_any_call(
            "llm.token_count.completion_details.reasoning", 15
        )

    def test_sets_cache_read_and_reasoning_from_flat_fields(self) -> None:
        """Tolerate an already-flattened usage shape as a fallback."""
        from agent_engine_sdk.models import LLMTokenUsage

        from agent_engine_sdk_langgraph.secure_llm import _enrich_span_with_token_counts

        usage = LLMTokenUsage.model_validate(
            {
                "input_tokens": 200,
                "output_tokens": 80,
                "total_tokens": 280,
                "cache_read": 30,
                "reasoning": 15,
            }
        )
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        with patch("agent_engine_sdk_langgraph.secure_llm.trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            _enrich_span_with_token_counts(usage)

        mock_span.set_attribute.assert_any_call(
            "llm.token_count.prompt_details.cache_read", 30
        )
        mock_span.set_attribute.assert_any_call(
            "llm.token_count.completion_details.reasoning", 15
        )

    def test_skips_when_usage_is_none(self) -> None:
        from agent_engine_sdk_langgraph.secure_llm import _enrich_span_with_token_counts

        mock_span = MagicMock()
        with patch("agent_engine_sdk_langgraph.secure_llm.trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            _enrich_span_with_token_counts(None)

        mock_span.set_attribute.assert_not_called()

    def test_skips_when_span_not_recording(self) -> None:
        from agent_engine_sdk.models import LLMTokenUsage

        from agent_engine_sdk_langgraph.secure_llm import _enrich_span_with_token_counts

        usage = LLMTokenUsage(prompt_tokens=100, completion_tokens=20)
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False
        with patch("agent_engine_sdk_langgraph.secure_llm.trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            _enrich_span_with_token_counts(usage)

        mock_span.set_attribute.assert_not_called()

    def test_does_not_raise_on_non_numeric_extra(self) -> None:
        """A non-numeric extra (extras are untyped) must not fail the LLM call."""
        from agent_engine_sdk.models import LLMTokenUsage

        from agent_engine_sdk_langgraph.secure_llm import _enrich_span_with_token_counts

        usage = LLMTokenUsage.model_validate(
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cache_read": "not-a-number",
            }
        )
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        with patch("agent_engine_sdk_langgraph.secure_llm.trace") as mock_trace:
            mock_trace.get_current_span.return_value = mock_span
            # Must not raise despite the bad cache_read value.
            _enrich_span_with_token_counts(usage)

        # The numeric attributes set before the bad value are still recorded.
        mock_span.set_attribute.assert_any_call("llm.token_count.prompt", 100)
        mock_span.set_attribute.assert_any_call("llm.token_count.completion", 20)


class TestSecureWrappedLLMSpanEnrichment:
    @patch("agent_engine_sdk_langgraph.secure_llm._enrich_span_with_token_counts")
    @patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy")
    def test_generate_enriches_span_with_response_usage(
        self, MockProxy, mock_enrich
    ) -> None:
        from agent_engine_sdk.models import (
            LLMResponse,
            LLMStreamChunk,
            LLMTokenUsage,
        )

        _, _, wrapped = _make_wrapper_and_wrapped()
        mock_proxy = MockProxy.return_value
        usage = LLMTokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        mock_proxy.stream.return_value = iter([LLMStreamChunk(content="Hi")])
        mock_proxy.response_from_stream_chunks.return_value = LLMResponse(
            content="Hi", usage=usage
        )
        mock_proxy.last_duration_ms = 0.0
        mock_proxy.last_from_cache = False
        mock_proxy.last_latest_step_number = None

        wrapped._generate([HumanMessage(content="Hello")])

        mock_enrich.assert_called_once_with(usage)

    @patch("agent_engine_sdk_langgraph.secure_llm._enrich_span_with_token_counts")
    @patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy")
    def test_generate_enriches_span_with_none_when_no_usage(
        self, MockProxy, mock_enrich
    ) -> None:
        from agent_engine_sdk.models import LLMResponse, LLMStreamChunk

        _, _, wrapped = _make_wrapper_and_wrapped()
        mock_proxy = MockProxy.return_value
        mock_proxy.stream.return_value = iter([LLMStreamChunk(content="Hi")])
        mock_proxy.response_from_stream_chunks.return_value = LLMResponse(content="Hi")
        mock_proxy.last_duration_ms = 0.0
        mock_proxy.last_from_cache = False
        mock_proxy.last_latest_step_number = None

        wrapped._generate([HumanMessage(content="Hello")])

        mock_enrich.assert_called_once_with(None)

    @patch("agent_engine_sdk_langgraph.secure_llm._enrich_span_with_token_counts")
    @patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy")
    def test_stream_enriches_span_with_usage_chunk(
        self, MockProxy, mock_enrich
    ) -> None:
        from agent_engine_sdk.models import LLMStreamChunk, LLMTokenUsage

        _, _, wrapped = _make_wrapper_and_wrapped()
        usage = LLMTokenUsage(prompt_tokens=60, completion_tokens=20, total_tokens=80)
        mock_proxy = MockProxy.return_value
        mock_proxy.last_duration_ms = 0.0
        mock_proxy.last_from_cache = False
        mock_proxy.last_latest_step_number = None
        mock_proxy.stream.return_value = iter(
            [
                LLMStreamChunk(content="Hello"),
                LLMStreamChunk(usage=usage),
            ]
        )

        list(wrapped._stream([HumanMessage(content="Hi")]))

        mock_enrich.assert_called_once_with(usage)

    @patch("agent_engine_sdk_langgraph.secure_llm._enrich_span_with_token_counts")
    @patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy")
    def test_stream_enriches_span_with_none_when_no_usage_chunk(
        self, MockProxy, mock_enrich
    ) -> None:
        from agent_engine_sdk.models import LLMStreamChunk

        _, _, wrapped = _make_wrapper_and_wrapped()
        mock_proxy = MockProxy.return_value
        mock_proxy.last_duration_ms = 0.0
        mock_proxy.last_from_cache = False
        mock_proxy.last_latest_step_number = None
        mock_proxy.stream.return_value = iter([LLMStreamChunk(content="Hello")])

        list(wrapped._stream([HumanMessage(content="Hi")]))

        mock_enrich.assert_called_once_with(None)

    @patch("agent_engine_sdk_langgraph.secure_llm._enrich_span_with_token_counts")
    @patch("agent_engine_sdk_langgraph.secure_llm.SecureLLMProxy")
    def test_stream_enriches_span_on_error_after_usage_captured(
        self, MockProxy, mock_enrich
    ) -> None:
        """Usage captured before a mid-stream failure is still enriched (finally)."""
        from agent_engine_sdk.models import LLMStreamChunk, LLMTokenUsage

        from agent_engine_runner_shared.secure_wrapper import LLMInvocationError

        _, _, wrapped = _make_wrapper_and_wrapped()
        usage = LLMTokenUsage(prompt_tokens=60, completion_tokens=20, total_tokens=80)
        mock_proxy = MockProxy.return_value
        mock_proxy.last_duration_ms = 0.0
        mock_proxy.last_from_cache = False
        mock_proxy.last_latest_step_number = None

        def _stream_then_fail():
            yield LLMStreamChunk(content="Hello")
            yield LLMStreamChunk(usage=usage)
            raise LLMInvocationError("boom")

        mock_proxy.stream.return_value = _stream_then_fail()

        with pytest.raises(LLMInvocationError):
            list(wrapped._stream([HumanMessage(content="Hi")]))

        mock_enrich.assert_called_once_with(usage)
