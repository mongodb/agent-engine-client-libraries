"""
SecureWrappedLLM - LangChain BaseChatModel adapter for secure LLM calls.

This is a thin adapter that provides the LangChain BaseChatModel interface
while delegating invoke_llm orchestration to SecureLLMProxy in agent-engine-runner-shared.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Any, cast

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from agent_engine_sdk.models import (
    LLMInvocationOptions,
    LLMTokenUsage,
    LLMToolSchema,
)
from opentelemetry import trace

from agent_engine_runner_shared.span_kinds import (
    OPENINFERENCE_SPAN_KIND,
    OpenInferenceSpanKind,
)
from agent_engine_runner_shared.span_names import (
    MODEL_REQUEST_PREPARE,
    MODEL_RESPONSE_PROCESS,
)
from agent_engine_runner_shared.tracing import get_tracer

from agent_engine_sdk_langgraph.messages import (
    lc_messages_to_platform,
    llm_response_to_chat_result,
    llm_stream_chunk_to_generation_chunk,
)
from agent_engine_sdk_langgraph.durable_deep_agent import (
    _stamp_deep_agent_message_ids,
)
from agent_engine_runner_shared.metrics import Metrics
from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
from agent_engine_runner_shared.secure_wrapper import (
    LLMInvocationError,
    PolicyDeniedException,
)
from agent_engine_runner_shared.utils import (
    log_cached_result,
    log_llm_messages,
    log_llm_response,
    log_policy_blocked,
    log_separator,
)

# Re-export for backward compatibility (tests import from here)
__all__ = ["SecureWrappedLLM", "PolicyDeniedException", "LLMInvocationError"]

_logger = logging.getLogger(__name__)
_MISSING_TOOL_RESULT_TEMPLATE = (
    "Tool execution failed before the platform recorded a result for {tool_name}. "
    "Treat this as a failed tool call and continue with the user request."
)


def _repair_missing_tool_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Insert missing ToolMessages after assistant tool calls before LLM invocation.

    A tool node failure can leave a checkpoint with an assistant ``tool_calls``
    message but no matching ``ToolMessage``. Providers reject that history on
    the next LLM call, so repair the prompt sent to OE while preserving the
    original checkpoint state.
    """

    repaired: list[BaseMessage] = []
    inserted = 0
    index = 0

    while index < len(messages):
        message = messages[index]
        repaired.append(message)
        index += 1

        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue

        expected: list[tuple[str, str | None]] = []
        for tool_call in message.tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = tool_call.get("id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                continue
            tool_name = tool_call.get("name")
            expected.append(
                (tool_call_id, tool_name if isinstance(tool_name, str) else None)
            )

        if not expected:
            continue

        seen: set[str] = set()
        while index < len(messages) and isinstance(messages[index], ToolMessage):
            tool_message = cast(ToolMessage, messages[index])
            if tool_message.tool_call_id:
                seen.add(tool_message.tool_call_id)
            repaired.append(tool_message)
            index += 1

        for tool_call_id, tool_name in expected:
            if tool_call_id in seen:
                continue
            display_name = tool_name or "the requested tool"
            repaired.append(
                ToolMessage(
                    content=_MISSING_TOOL_RESULT_TEMPLATE.format(
                        tool_name=display_name
                    ),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    status="error",
                )
            )
            inserted += 1

    if inserted:
        _logger.warning(
            "Inserted %d missing tool message(s) before LLM invocation", inserted
        )
        return repaired
    return messages


def _enrich_span_with_token_counts(usage: LLMTokenUsage | None) -> None:
    """Set llm.token_count.* attributes on the current OTel span.

    For Gemini the inner ChatGoogleGenerativeAI span runs in the tool pod under
    a separate trace, so there is no automatic parent-child link that would let
    the OpenInference instrumentor propagate token counts to the SecureWrappedLLM
    span.  We therefore hoist the usage data returned by OE directly onto the
    span while it is still open.
    """
    if usage is None:
        return
    # Telemetry enrichment must never fail the LLM call. The model_extra fields
    # below are untyped, so a non-numeric value would otherwise raise here.
    try:
        span = trace.get_current_span()
        if not span.is_recording():
            return
        prompt = (
            usage.prompt_tokens
            if usage.prompt_tokens is not None
            else usage.input_tokens
        )
        completion = (
            usage.completion_tokens
            if usage.completion_tokens is not None
            else usage.output_tokens
        )
        if prompt is not None:
            span.set_attribute("llm.token_count.prompt", prompt)
        if completion is not None:
            span.set_attribute("llm.token_count.completion", completion)
        if usage.total_tokens is not None:
            span.set_attribute("llm.token_count.total", usage.total_tokens)
        # LangChain's standard usage_metadata nests these one level down
        # (input_token_details.cache_read, output_token_details.reasoning);
        # tolerate an already-flattened shape as a fallback.
        extra = usage.model_extra or {}
        input_details = extra.get("input_token_details") or {}
        output_details = extra.get("output_token_details") or {}
        cache_read = input_details.get("cache_read")
        if cache_read is None:
            cache_read = extra.get("cache_read")
        if cache_read is not None:
            span.set_attribute(
                "llm.token_count.prompt_details.cache_read", int(cache_read)
            )
        reasoning = output_details.get("reasoning")
        if reasoning is None:
            reasoning = extra.get("reasoning")
        if reasoning is not None:
            span.set_attribute(
                "llm.token_count.completion_details.reasoning", int(reasoning)
            )
    except Exception:  # noqa: BLE001 - enrichment is best-effort observability
        _logger.warning("Failed to enrich span with token counts", exc_info=True)


def _normalize_bound_tool(tool: Any) -> Any:
    """Return a bound tool in a form ``SecureLLMProxy`` can serialize.

    ``SecureLLMProxy._serialize_bound_tools`` (agent-engine-runner-shared, framework-neutral)
    handles objects exposing ``tool_call_schema`` (LangChain BaseTools) and
    already-serialized ``dict`` / ``LLMToolSchema`` entries, but drops anything
    else. The notable casualty is the bare Pydantic class that
    ``with_structured_output`` binds, which the proxy logs as "Skipping
    unrecognized tool type" and never sends to the model (AP-2933).

    Convert those leftovers to the canonical OpenAI tool dict here, in the
    LangChain-aware layer, so agent-engine-runner-shared keeps no LangChain dependency.
    Tools the proxy already understands are returned untouched so their
    existing serialization is unchanged.

    ``convert_to_openai_tool`` raises on types it cannot interpret (e.g. an
    exotic ``RunnableBinding``). Such tools were previously skipped with a
    warning by ``SecureLLMProxy``, so fall back to passing them through
    unchanged rather than turning a silent skip into a hard crash at
    ``bind_tools`` time. This mirrors the TypeScript ``normalizeBoundTool``.
    """
    if hasattr(tool, "tool_call_schema") or isinstance(tool, (dict, LLMToolSchema)):
        return tool
    try:
        return convert_to_openai_tool(tool)
    except (ValueError, TypeError):
        _logger.warning(
            "_normalize_bound_tool: cannot convert %s to an OpenAI tool dict; "
            "passing through unchanged (SecureLLMProxy may skip it)",
            type(tool).__name__,
        )
        return tool


class SecureWrappedLLM(BaseChatModel):
    """LangChain LLM wrapper that routes all calls through OE.

    This wraps a real LLM (Gemini, OpenAI, etc.) and delegates OE
    orchestration to SecureLLMProxy. The adapter handles:
    - LangChain ↔ sdk-core message conversion
    - Translating OE-owned invoke_llm results back into LangChain types
    - Logging and metrics (LangChain-specific concerns)

    The SecureToolWrapper is obtained via a callable at invocation time,
    allowing the graph to be built before execution context exists.
    """

    model_name: str = "secure-wrapped-llm"
    _llm: Any = None
    _get_wrapper: Any = None
    _bound_tools: Sequence[Any] = []
    _bound_tool_choice: Any = None

    def __init__(
        self,
        llm: Any,
        get_wrapper: Callable,
        llm_id: str,
        model_name: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._llm = llm
        self._get_wrapper = get_wrapper
        self._bound_tools = []
        self._bound_tool_choice = None
        self._llm_id: str = llm_id
        self.model_name = (
            model_name
            or getattr(llm, "model", None)
            or getattr(llm, "model_name", "unknown")
        )

    @property
    def _llm_type(self) -> str:
        return "secure-wrapped-llm"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "llm_id": self._llm_id}

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> SecureWrappedLLM:
        """Bind tools to the LLM.

        ``tool_choice`` is captured so it can be forwarded to the tool pod's
        ``bind_tools`` call: invocations route through SecureLLMProxy and never
        touch the inner LangChain binding, so any binding kwargs left only on
        ``self._llm`` are otherwise lost (AP-2933).
        """
        new_llm = SecureWrappedLLM(
            llm=self._llm.bind_tools(tools, **kwargs)
            if hasattr(self._llm, "bind_tools")
            else self._llm,
            get_wrapper=self._get_wrapper,
            llm_id=self._llm_id,
            model_name=self.model_name,
        )
        new_llm._bound_tools = [_normalize_bound_tool(tool) for tool in tools]
        new_llm._bound_tool_choice = kwargs.get("tool_choice")
        return new_llm

    def _get_wrapper_or_raise(self) -> Any:
        """Get the wrapper, raising an error if not available."""
        wrapper = self._get_wrapper()
        if wrapper is None:
            raise RuntimeError(
                "SecureToolWrapper is not initialized. "
                "This means the LLM is being invoked outside of an execution context. "
                "Ensure the graph is invoked via AER's /execute endpoint."
            )
        return wrapper

    @staticmethod
    def _build_invocation_options(
        kwargs: dict[str, Any],
    ) -> LLMInvocationOptions | None:
        """Convert LangChain's loose kwargs into a typed invocation-options model."""
        if not kwargs:
            return None
        return LLMInvocationOptions.model_validate(kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using the wrapped LLM."""
        wrapper = self._get_wrapper_or_raise()
        tracer = get_tracer("agent-engine-sdk-langgraph.secure_llm")

        with tracer.start_as_current_span(
            MODEL_REQUEST_PREPARE,
            attributes={OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKind.CHAIN.value},
        ):
            step = wrapper.next_operational_step()

            _stamp_deep_agent_message_ids(messages)
            messages = _repair_missing_tool_messages(messages)

            log_separator()
            log_llm_messages(messages, prefix="LLM")

            # Convert LangChain messages to sdk-core format
            platform_messages = lc_messages_to_platform(messages)
            options = self._build_invocation_options(kwargs)

            # Create per-call proxy and delegate
            proxy = SecureLLMProxy(
                oe_url=wrapper.oe_url,
                execution_id=wrapper.execution_id,
                llm_id=self._llm_id,
                model_name=self.model_name,
                bound_tools=list(self._bound_tools) if self._bound_tools else None,
                bound_tool_choice=self._bound_tool_choice,
                operational_steps=wrapper.operational_steps,
                durable_memory=wrapper.durable_memory,
            )

        # proxy.stream() (the provider round trip via OE) is left
        # uninstrumented: LangChainInstrumentor already wraps this whole
        # method in its own LLM-kind span, so the gap between
        # request.prepare and response.process is the provider call.
        try:
            stream_chunks = list(
                proxy.stream(
                    messages=platform_messages,
                    step=step,
                    stop=stop,
                    options=options,
                )
            )
            response = proxy.response_from_stream_chunks(stream_chunks)
        except PolicyDeniedException as e:
            log_policy_blocked("invoke_llm", step, str(e), prefix="LLM")
            raise
        except LLMInvocationError:
            Metrics.record_latency(
                "llm_call",
                proxy.last_duration_ms,
                model_name=self.model_name or "unknown",
            )
            Metrics.record_error("llm_call", model_name=self.model_name or "unknown")
            raise

        with tracer.start_as_current_span(
            MODEL_RESPONSE_PROCESS,
            attributes={OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKind.CHAIN.value},
        ):
            if proxy.last_latest_step_number is not None:
                wrapper.observe_operational_step(proxy.last_latest_step_number)

            if proxy.last_from_cache:
                log_cached_result("invoke_llm", step, prefix="LLM")

            _enrich_span_with_token_counts(response.usage)

            # Convert response to LangChain format
            result = llm_response_to_chat_result(response)

            # Log and record metrics
            ai_message = result.generations[0].message
            log_llm_response(ai_message, step, prefix="LLM")
            duration_ms = proxy.last_duration_ms
            Metrics.record_latency(
                "llm_call", duration_ms, model_name=self.model_name or "unknown"
            )

        return result

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream a response using the wrapped LLM."""
        wrapper = self._get_wrapper_or_raise()

        # Check if underlying LLM supports streaming BEFORE touching step counter
        # If not, delegate entirely to _generate which handles its own step/OE flow
        if not (hasattr(self._llm, "stream") and callable(self._llm.stream)):
            result = self._generate(messages, stop, run_manager, **kwargs)
            msg = result.generations[0].message
            msg_data = msg.model_dump()
            msg_data["type"] = "AIMessageChunk"
            yield ChatGenerationChunk(message=AIMessageChunk.model_validate(msg_data))
            return

        tracer = get_tracer("agent-engine-sdk-langgraph.secure_llm")

        with tracer.start_as_current_span(
            MODEL_REQUEST_PREPARE,
            attributes={OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKind.CHAIN.value},
        ):
            step = wrapper.next_operational_step()

            _stamp_deep_agent_message_ids(messages)
            messages = _repair_missing_tool_messages(messages)

            log_separator()
            log_llm_messages(messages, prefix="LLM")

            # Convert LangChain messages to sdk-core format
            platform_messages = lc_messages_to_platform(messages)
            options = self._build_invocation_options(kwargs)

            # Create per-call proxy and delegate
            proxy = SecureLLMProxy(
                oe_url=wrapper.oe_url,
                execution_id=wrapper.execution_id,
                llm_id=self._llm_id,
                model_name=self.model_name,
                bound_tools=list(self._bound_tools) if self._bound_tools else None,
                bound_tool_choice=self._bound_tool_choice,
                operational_steps=wrapper.operational_steps,
                durable_memory=wrapper.durable_memory,
            )

        # Gap between request.prepare and response.process is the provider
        # round trip, same reasoning as _generate above.
        last_usage: LLMTokenUsage | None = None
        # Spans the whole consumption loop, not just the first chunk: a
        # streaming response isn't "processed" until the stream ends.
        response_process_span = tracer.start_span(
            MODEL_RESPONSE_PROCESS,
            attributes={OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKind.CHAIN.value},
        )
        try:
            for sdk_chunk in proxy.stream(
                messages=platform_messages,
                step=step,
                stop=stop,
                options=options,
            ):
                if sdk_chunk.usage is not None:
                    last_usage = sdk_chunk.usage
                yield llm_stream_chunk_to_generation_chunk(sdk_chunk)
        except PolicyDeniedException as e:
            log_policy_blocked("invoke_llm", step, str(e), prefix="LLM")
            raise
        except LLMInvocationError:
            Metrics.record_latency(
                "llm_call",
                proxy.last_duration_ms,
                model_name=self.model_name or "unknown",
            )
            Metrics.record_error("llm_call", model_name=self.model_name or "unknown")
            raise
        finally:
            # Enrich on both success and error paths: any usage captured before a
            # mid-stream failure would otherwise be discarded.
            _enrich_span_with_token_counts(last_usage)
            response_process_span.end()

        if proxy.last_latest_step_number is not None:
            wrapper.observe_operational_step(proxy.last_latest_step_number)

        if proxy.last_from_cache:
            log_cached_result("invoke_llm", step, prefix="LLM")

        # Record metrics
        Metrics.record_latency(
            "llm_call", proxy.last_duration_ms, model_name=self.model_name or "unknown"
        )
