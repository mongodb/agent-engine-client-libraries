"""Lightweight adapter wrapping a LangChain BaseChatModel as BaseLLM.

Used by the tool pod (server/tool.py) to invoke LLMs through the
framework-neutral BaseLLM protocol while the underlying provider is
still a LangChain ChatModel.

After AP-220 (T6), runtime._create_llm will return BaseLLM directly,
and this adapter will only be used internally by sdklanggraph.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, cast

from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.runnables import RunnableBinding
from agent_engine_sdk.interfaces import BaseLLM
from agent_engine_sdk.models import (
    LLMResponse,
    LLMStreamChunk,
    LLMTokenUsage,
    LLMToolSchema,
    Message,
    ToolCallChunk,
)
from pydantic import JsonValue

from agent_engine_sdk_langgraph.messages import platform_messages_to_lc
from agent_engine_runner_shared.async_utils import iterate_in_thread
from agent_engine_runner_shared.models import (
    LLMResult,
    accumulate_stream_usage,
    json_safe_metadata,
)
from agent_engine_runner_shared.utils import normalize_content, normalize_tool_call_args


class LangChainLLMAdapter(BaseLLM):
    """Wraps a LangChain BaseChatModel to implement the BaseLLM protocol.

    Accepts sdk-core ``Message`` objects, converts to LangChain messages
    for the underlying LLM, and converts responses back to sdk-core types.
    """

    def __init__(
        self,
        llm: Any,
        tools: list[LLMToolSchema] | None = None,
        tool_choice: Any | None = None,
    ):
        serialized_tools = (
            [tool.to_langchain_dict() for tool in tools] if tools else None
        )
        if serialized_tools:
            # tool_choice (e.g. a function name from with_structured_output) is
            # forwarded into bind_tools, where LangChain translates it against
            # the bound tools to force the call. A choice without tools cannot
            # bind to anything, so it is ignored (AP-2933). Guard on `is not
            # None` rather than truthiness: tool_choice=False is a documented
            # LangChain value that disables forced tool use and must survive.
            bind_kwargs = (
                {"tool_choice": tool_choice} if tool_choice is not None else {}
            )
            self._llm = llm.bind_tools(serialized_tools, **bind_kwargs)
        else:
            self._llm = llm

    def invoke(self, messages: list[Message], **kwargs: object) -> LLMResponse:
        """Synchronous LLM invocation."""
        lc_messages = platform_messages_to_lc(messages)
        response = self._llm.invoke(lc_messages, **kwargs)
        return _response_to_llm_response(response)

    async def ainvoke(self, messages: list[Message], **kwargs: object) -> LLMResponse:
        """Async LLM invocation."""
        lc_messages = platform_messages_to_lc(messages)
        if asyncio.iscoroutinefunction(getattr(self._llm, "ainvoke", None)):
            response = await self._llm.ainvoke(lc_messages, **kwargs)
        else:
            response = self._llm.invoke(lc_messages, **kwargs)
        return _response_to_llm_response(response)

    async def astream(
        self, messages: list[Message], **kwargs: object
    ) -> AsyncIterator[LLMStreamChunk]:
        """Async streaming LLM invocation."""
        import inspect

        lc_messages = platform_messages_to_lc(messages)

        if hasattr(self._llm, "astream"):
            # Chat Completions omits usage unless asked. Responses.create
            # rejects stream_usage, so a caller-supplied value still wins
            # and we never invent the kwarg for that path.
            if "stream_usage" not in kwargs and _supports_stream_usage(self._llm):
                kwargs["stream_usage"] = True
            stream = self._llm.astream(lc_messages, **kwargs)
            if hasattr(stream, "__aiter__"):
                running: LLMTokenUsage | None = None
                async for chunk in stream:
                    stream_chunk, running = _with_running_usage(chunk, running)
                    yield stream_chunk
                return
            # If astream returned a coroutine, await it to get the async iterator
            if inspect.isawaitable(stream):
                stream = await stream
                running = None
                async for chunk in stream:
                    stream_chunk, running = _with_running_usage(chunk, running)
                    yield stream_chunk
                return

        # Fallback to sync stream without blocking the async event loop.
        iterator = iter(self._llm.stream(lc_messages, **kwargs))
        running = None
        async for chunk in iterate_in_thread(iterator):
            stream_chunk, running = _with_running_usage(
                cast(BaseMessageChunk, chunk), running
            )
            yield stream_chunk


def _unwrap_chat_model(llm: Any) -> Any:
    return llm.bound if isinstance(llm, RunnableBinding) else llm


def _uses_responses_api(llm: Any) -> bool:
    """True when LangChain will call ``Responses.create``, which rejects ``stream_usage``."""
    model = _unwrap_chat_model(llm)
    if getattr(model, "use_responses_api", None) is True:
        return True
    probe = getattr(model, "_use_responses_api", None)
    extra = getattr(model, "model_kwargs", None)
    if not callable(probe):
        return False
    try:
        return bool(probe(extra if isinstance(extra, dict) else {}))
    except Exception:
        return False


def _supports_stream_usage(llm: Any) -> bool:
    """True when the model declares ``stream_usage`` and is not on the Responses API."""
    model = _unwrap_chat_model(llm)
    fields = getattr(type(model), "model_fields", None)
    return (
        isinstance(fields, dict)
        and "stream_usage" in fields
        and not _uses_responses_api(model)
    )


def _response_to_llm_response(response: BaseMessage) -> LLMResponse:
    """Convert a LangChain response to sdk-core LLMResponse."""
    return LLMResult.from_response(response).to_response()


def _optional_str_attr(message: BaseMessage, attr: str) -> str | None:
    value = getattr(message, attr, None)
    return value if isinstance(value, str) else None


def _with_running_usage(
    chunk: BaseMessageChunk, running: LLMTokenUsage | None
) -> tuple[LLMStreamChunk, LLMTokenUsage | None]:
    stream_chunk = _chunk_to_llm_stream_chunk(chunk)
    running = accumulate_stream_usage(running, stream_chunk.usage)
    if running is not None:
        stream_chunk.usage = running
    return stream_chunk, running


def _optional_json_dict_attr(
    message: BaseMessage, attr: str
) -> dict[str, JsonValue] | None:
    try:
        value = getattr(message, attr, None)
    except Exception:
        return None
    return json_safe_metadata(value)


def _chunk_to_llm_stream_chunk(chunk: BaseMessageChunk) -> LLMStreamChunk:
    """Convert a LangChain stream chunk to sdk-core LLMStreamChunk."""
    raw_content = chunk.content
    content = normalize_content(raw_content)

    # Convert tool_call_chunks
    tool_calls: list[ToolCallChunk] | None = None
    raw_chunks = getattr(chunk, "tool_call_chunks", None)
    if raw_chunks:
        tool_calls = []
        for tc in raw_chunks:
            if isinstance(tc, dict):
                tool_calls.append(
                    ToolCallChunk(
                        id=tc.get("id"),
                        name=tc.get("name"),
                        args=normalize_tool_call_args(tc.get("args")),
                        type=tc.get("type"),
                        index=tc.get("index"),
                    )
                )
            elif hasattr(tc, "model_dump"):
                d = tc.model_dump()
                tool_calls.append(
                    ToolCallChunk(
                        id=d.get("id"),
                        name=d.get("name"),
                        args=normalize_tool_call_args(d.get("args")),
                        type=d.get("type"),
                        index=d.get("index"),
                    )
                )

    # If no tool_call_chunks, check merged tool_calls
    if not tool_calls:
        raw_calls = getattr(chunk, "tool_calls", None)
        if raw_calls:
            tool_calls = []
            for tc in raw_calls:
                if isinstance(tc, dict):
                    tool_calls.append(
                        ToolCallChunk(
                            id=tc.get("id"),
                            name=tc.get("name"),
                            args=normalize_tool_call_args(tc.get("args")),
                            type=tc.get("type"),
                        )
                    )

    usage = LLMResult.extract_usage(chunk)

    return LLMStreamChunk(
        content=content or None,
        tool_calls=tool_calls or None,
        usage=usage,
        id=_optional_str_attr(chunk, "id"),
        name=_optional_str_attr(chunk, "name"),
        response_metadata=_optional_json_dict_attr(chunk, "response_metadata"),
        additional_kwargs=_optional_json_dict_attr(chunk, "additional_kwargs"),
    )
