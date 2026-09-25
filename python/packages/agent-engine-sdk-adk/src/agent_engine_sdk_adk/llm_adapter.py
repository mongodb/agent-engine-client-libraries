"""ADKLLMAdapter — wraps ADK BaseLlm as platform BaseLLM for Tool Pod use."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from agent_engine_sdk.interfaces import BaseLLM
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

if TYPE_CHECKING:
    from google.adk.models import BaseLlm
from agent_engine_sdk.models import (
    LLMResponse,
    LLMStreamChunk,
    LLMTokenUsage,
    LLMToolSchema,
    Message,
    ToolCallChunk,
)

from agent_engine_sdk_adk.messages import (
    content_to_platform_messages,
    platform_message_to_content,
)


class ADKLLMAdapter(BaseLLM):
    """Wraps an ADK BaseLlm to implement the platform BaseLLM protocol.

    Used by the Tool Pod to invoke LLMs through the framework-neutral interface
    while the underlying provider is an ADK BaseLlm.
    """

    def __init__(
        self,
        llm: BaseLlm,
        tools: list[LLMToolSchema] | None = None,
        tool_choice: Any = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        # Accepted for parity with the shared LLM-adapter factory contract.
        # ADK has no equivalent forced-tool-selection knob, so the value is
        # intentionally ignored here.
        del tool_choice

    def _build_request(self, messages: list[Message], **kwargs: object) -> LlmRequest:
        contents = [platform_message_to_content(m) for m in messages]
        config_kwargs: dict[str, Any] = {}

        if self._tools:
            declarations: list[types.FunctionDeclaration] = []
            for schema in self._tools:
                if schema.name:
                    params: dict[str, Any] | None = None
                    if isinstance(schema.parameters, dict):
                        params = dict(schema.parameters)
                    declarations.append(
                        types.FunctionDeclaration(
                            name=schema.name,
                            description=schema.description or "",
                            parameters_json_schema=params,
                        )
                    )
            if declarations:
                config_kwargs["tools"] = [
                    types.Tool(function_declarations=declarations)
                ]

        stop = kwargs.get("stop")
        if isinstance(stop, list):
            config_kwargs["stop_sequences"] = stop

        # The caller supplies ``thinking_budget`` (the SDK does not read env
        # vars); when present it tunes Gemini's reasoning budget.
        thinking_budget = kwargs.get("thinking_budget")
        if isinstance(thinking_budget, int):
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget,
            )

        config = types.GenerateContentConfig(**config_kwargs)
        # The model name must be set on the request: the Tool Pod calls
        # generate_content_async directly with a hand-built LlmRequest (no ADK
        # flow populates it). Fail fast when it's missing rather than letting
        # the provider be hit with model=None and return an opaque 404.
        model = getattr(self._llm, "model", None)
        if not model:
            raise ValueError(
                "ADK LLM has no 'model' name; cannot build an LlmRequest. "
                "Ensure the configured BaseLlm exposes a non-empty 'model'."
            )
        return LlmRequest(
            model=model,
            contents=contents,
            config=config,
        )

    def invoke(self, messages: list[Message], **kwargs: object) -> LLMResponse:
        try:
            asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run, self.ainvoke(messages, **kwargs)
                ).result()
        except RuntimeError:
            return asyncio.run(self.ainvoke(messages, **kwargs))

    async def ainvoke(self, messages: list[Message], **kwargs: object) -> LLMResponse:
        request = self._build_request(messages, **kwargs)

        final_response: LlmResponse | None = None
        async for response in self._llm.generate_content_async(request, stream=False):
            final_response = response

        if final_response is None or final_response.content is None:
            return LLMResponse(content="")

        usage: LLMTokenUsage | None = None
        if final_response.usage_metadata is not None:
            usage_metadata = final_response.usage_metadata
            usage = LLMTokenUsage(
                prompt_tokens=getattr(usage_metadata, "prompt_token_count", None),
                completion_tokens=getattr(
                    usage_metadata, "candidates_token_count", None
                ),
                total_tokens=getattr(usage_metadata, "total_token_count", None),
            )

        # An LLM response Content carries text and/or function calls (no tool
        # responses), so take the single non-tool message it produces.
        platform_messages = content_to_platform_messages(final_response.content)
        non_tool_message = next(
            (message for message in platform_messages if message.role != "tool"), None
        )
        if non_tool_message is None:
            return LLMResponse(content="", usage=usage)
        return LLMResponse(
            content=non_tool_message.content
            if isinstance(non_tool_message.content, str)
            else "",
            tool_calls=non_tool_message.tool_calls,
            usage=usage,
        )

    async def astream(
        self, messages: list[Message], **kwargs: object
    ) -> AsyncIterator[LLMStreamChunk]:
        # ADK's progressive streaming (stream=True) emits each SSE chunk with
        # the *cumulative* content, not a delta. Feeding those into the AER's
        # response_from_stream_chunks accumulator duplicates text and — worse —
        # concatenates the full function-call args from every chunk into
        # invalid JSON, which parses to empty args. So invoke stream=False,
        # accumulate the full response here, and emit one combined chunk with
        # each tool call at a distinct index (Gemini emits the complete
        # function-call args under stream=False).
        request = self._build_request(messages, **kwargs)

        final_text = ""
        tool_chunks: list[ToolCallChunk] = []
        usage: LLMTokenUsage | None = None
        async for response in self._llm.generate_content_async(request, stream=False):
            if response.usage_metadata is not None:
                usage_metadata = response.usage_metadata
                usage = LLMTokenUsage(
                    prompt_tokens=getattr(usage_metadata, "prompt_token_count", None),
                    completion_tokens=getattr(
                        usage_metadata, "candidates_token_count", None
                    ),
                    total_tokens=getattr(usage_metadata, "total_token_count", None),
                )
            if not (response.content and response.content.parts):
                continue
            for part in response.content.parts:
                if part.text:
                    final_text += part.text
                elif part.function_call is not None:
                    function_call = part.function_call
                    tool_chunks.append(
                        ToolCallChunk(
                            id=function_call.id or None,
                            name=function_call.name or "",
                            args=json.dumps(dict(function_call.args))
                            if function_call.args
                            else "{}",
                            index=len(tool_chunks),
                        )
                    )

        yield LLMStreamChunk(
            content=final_text or None,
            tool_calls=tool_chunks or None,
        )
        if usage is not None:
            yield LLMStreamChunk(usage=usage)
