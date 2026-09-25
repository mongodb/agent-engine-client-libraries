"""SecureLlm — ADK BaseLlm that routes all LLM calls through the OE."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any

from agent_engine_sdk.models import LLMResponse as PlatformLLMResponse
from agent_engine_sdk.models import Message
from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from opentelemetry import trace

from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
from agent_engine_sdk_adk.messages import (
    content_to_platform_messages,
    parse_tool_args,
    tools_dict_to_schemas,
)
from agent_engine_sdk_adk.route import has_stable_operation_identity

_logger = logging.getLogger(__name__)


class SecureLlm(BaseLlm):
    """ADK BaseLlm that routes generate_content_async through SecureLLMProxy.

    In AER mode, every LLM call is intercepted and sent to the Orchestration
    Engine for approval, audit logging, and routing to the Tool Pod where the
    actual LLM API call happens.
    """

    _get_wrapper: Callable[[], Any]
    _llm_id: str

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        model: str,
        get_wrapper: Callable[[], Any],
        llm_id: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._get_wrapper = get_wrapper
        self._llm_id = llm_id

    def _get_wrapper_or_raise(self) -> Any:
        wrapper = self._get_wrapper()
        if wrapper is None:
            raise RuntimeError(
                "SecureToolWrapper is not initialized. "
                "This means the LLM is being invoked outside of an execution context. "
                "Ensure the agent is invoked via AER's /execute endpoint."
            )
        return wrapper

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        wrapper = self._get_wrapper_or_raise()
        step = wrapper.next_operational_step()

        platform_messages = [
            m for c in llm_request.contents for m in content_to_platform_messages(c)
        ]

        if llm_request.config and llm_request.config.system_instruction:  # type: ignore[reportUnknownMemberType]
            si: Any = llm_request.config.system_instruction  # type: ignore[reportUnknownMemberType]
            si_text = ""
            if isinstance(si, str):
                si_text = si
            elif isinstance(si, types.Content) and si.parts:
                si_text = "".join(p.text for p in si.parts if p.text)
            if si_text:
                platform_messages.insert(0, Message(role="system", content=si_text))

        tool_schemas = None
        if llm_request.tools_dict:
            tool_schemas = tools_dict_to_schemas(llm_request.tools_dict)

        proxy = SecureLLMProxy(
            oe_url=wrapper.oe_url,
            execution_id=wrapper.execution_id,
            llm_id=self._llm_id,
            model_name=self.model,
            bound_tools=tool_schemas,
            operational_steps=wrapper.operational_steps,
            stable_operation_identity=has_stable_operation_identity(),
            durable_memory=wrapper.durable_memory,
        )

        chunks = await asyncio.to_thread(
            lambda: list(proxy.stream(messages=platform_messages, step=step))
        )

        if proxy.last_latest_step_number is not None:
            wrapper.observe_operational_step(proxy.last_latest_step_number)

        if stream:
            for chunk in chunks:
                if chunk.content:
                    partial_content = types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=chunk.content)],
                    )
                    yield LlmResponse(content=partial_content, partial=True)

        response = proxy.response_from_stream_chunks(chunks)
        _enrich_span_with_token_counts(response)
        final_content = _llm_response_to_content(response)
        yield LlmResponse(content=final_content, partial=False)


def _enrich_span_with_token_counts(response: PlatformLLMResponse) -> None:
    """Set ``llm.token_count.*`` on the current OTEL span."""
    usage = response.usage
    if usage is None:
        return
    span = trace.get_current_span()
    if not span.is_recording():
        return
    prompt = usage.prompt_tokens or usage.input_tokens
    completion = usage.completion_tokens or usage.output_tokens
    if prompt is not None:
        span.set_attribute("llm.token_count.prompt", prompt)
    if completion is not None:
        span.set_attribute("llm.token_count.completion", completion)


def _llm_response_to_content(response: Any) -> types.Content | None:
    """Convert a platform LLMResponse to ADK Content."""
    parts: list[types.Part] = []
    if response.content:
        parts.append(types.Part.from_text(text=response.content))
    if response.tool_calls:
        for tc in response.tool_calls:
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        name=tc.name or "unknown",
                        id=tc.id or None,
                        args=parse_tool_args(tc.args),
                    )
                )
            )
    return types.Content(role="model", parts=parts) if parts else None
