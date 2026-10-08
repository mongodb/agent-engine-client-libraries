"""Tool Pod side of an OE LLM activity: run the registered native model."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from agent_engine_sdk import (
    LLMResponse,
    LLMStreamChunk,
    LLMTokenUsage,
    LLMToolSchema,
    Message,
    ToolCallChunk,
)
from agents import FunctionTool, Model, ModelSettings, ModelTracing, OpenAIProvider
from agents.items import TResponseInputItem
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseTextDeltaEvent,
)

from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
from agent_engine_sdk_openai_agents.errors import UnsupportedDurableOpenAIAgentsError
from agent_engine_sdk_openai_agents.items import validate_items
from agent_engine_sdk_openai_agents.wire import (
    TransportedOutputSchema,
    native_settings,
    reasoning_options,
)

__all__ = ["OpenAIAgentsLLM", "RegisteredModel"]


# Request fields a registration may not set through extra_body or extra_args.
# The first group has the provider continue a conversation it stores, which a
# replay cannot reproduce. The rest are what the adapter builds for each call
# from the Runner's request: overriding one would run a hosted tool, change the
# conversation, or change the output contract outside OE's record.
_PROVIDER_CONTINUATION = frozenset(
    {"previous_response_id", "conversation", "conversation_id", "prompt"}
)
# Request fields the typed ModelSettings reasoning and verbosity fields own.
# Set as raw extras they would skip the check of what the model API supports
# and override the per-call value.
_TYPED_SETTINGS = frozenset({"reasoning", "reasoning_effort", "verbosity"})
_ADAPTER_OWNED = frozenset(
    {
        "model",
        "input",
        "messages",
        "instructions",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "stream",
        "text",
        "response_format",
        "include",
        "background",
    }
)


@dataclass(frozen=True)
class RegisteredModel:
    """The native model and provider settings an app registers for its agent.

    A name resolves through the SDK's default OpenAI provider, which reads the
    Tool Pod's OpenAI environment. Pass a configured ``agents.Model`` to use a
    custom client instead.
    """

    model: str | Model
    settings: ModelSettings = field(default_factory=ModelSettings)

    def validate(self) -> None:
        """Reject a registration the Tool Pod cannot honor.

        Reasoning effort and verbosity apply here as registered; other
        reasoning settings are rejected as they are per call.

        Called when a turn starts and before a model call, not at
        registration: the AER builds the agent outside the error handling that
        reports a turn's failure to the caller.
        """
        if isinstance(self.model, str) and not self.model.strip():
            raise ValueError("model name must not be empty")
        reasoning_options(self.settings)
        # Only the SDK Runner reads retry settings, and the Tool Pod calls the
        # model directly, so a registered retry policy would silently do nothing.
        if self.settings.retry is not None:
            raise UnsupportedDurableOpenAIAgentsError(
                "ModelSettings.retry has no effect on a registered model; the "
                "Tool Pod already retries transient provider failures"
            )
        # The Runner's loop is also what enforces ModelSettings.timeout, so a
        # registered one would never bound the call.
        if self.settings.timeout is not None:
            raise UnsupportedDurableOpenAIAgentsError(
                "ModelSettings.timeout has no effect on a registered model; set "
                "it on the agent, where it applies per call"
            )
        # The Runner chooses tools per call. A registered choice would apply to
        # every call that names none, forcing the same tool again and again.
        if self.settings.tool_choice is not None:
            raise UnsupportedDurableOpenAIAgentsError(
                "ModelSettings.tool_choice cannot be registered on a model; set "
                "it on the agent, where it applies per call"
            )
        # OE owns the conversation. These request fields would have the provider
        # continue one it stores instead, which a replay cannot reproduce.
        for name in ("extra_body", "extra_args"):
            extra = getattr(self.settings, name)
            hosted = (
                sorted(_PROVIDER_CONTINUATION & set(cast(Mapping[str, object], extra)))
                if isinstance(extra, Mapping)
                else []
            )
            if hosted:
                raise UnsupportedDurableOpenAIAgentsError(
                    f"ModelSettings.{name} sets {', '.join(hosted)}: "
                    "provider-hosted continuation would bypass OE-owned history"
                )
            owned = (
                sorted(_ADAPTER_OWNED & set(cast(Mapping[str, object], extra)))
                if isinstance(extra, Mapping)
                else []
            )
            if owned:
                raise UnsupportedDurableOpenAIAgentsError(
                    f"ModelSettings.{name} sets {', '.join(owned)}: the adapter "
                    "builds these request fields for each call from the Runner's "
                    "request"
                )
            typed = (
                sorted(_TYPED_SETTINGS & set(cast(Mapping[str, object], extra)))
                if isinstance(extra, Mapping)
                else []
            )
            if typed:
                raise UnsupportedDurableOpenAIAgentsError(
                    f"ModelSettings.{name} sets {', '.join(typed)}: use "
                    "ModelSettings.reasoning and ModelSettings.verbosity instead"
                )


class OpenAIAgentsLLM:
    """Platform ``BaseLLM`` over a registered native OpenAI Agents model."""

    def __init__(
        self,
        llm: RegisteredModel,
        tools: list[LLMToolSchema] | None = None,
        tool_choice: object = None,
    ) -> None:
        llm.validate()
        self._model = (
            OpenAIProvider().get_model(llm.model)
            if isinstance(llm.model, str)
            else llm.model
        )
        self._settings = llm.settings
        self._tools = [_function_tool(schema) for schema in tools or []]
        self._tool_choice = tool_choice

    def invoke(self, messages: list[Message], **kwargs: object) -> LLMResponse:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(messages, **kwargs))
        # asyncio.run cannot nest inside a running loop; give it its own thread.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.ainvoke(messages, **kwargs)).result()

    async def ainvoke(self, messages: list[Message], **kwargs: object) -> LLMResponse:
        chunks = [chunk async for chunk in self.astream(messages, **kwargs)]
        return SecureLLMProxy.response_from_stream_chunks(chunks)

    async def astream(
        self, messages: list[Message], **kwargs: object
    ) -> AsyncIterator[LLMStreamChunk]:
        if kwargs.pop("stop", None) is not None:
            raise UnsupportedDurableOpenAIAgentsError(
                "stop sequences are not supported"
            )
        response_format = kwargs.pop("response_format", None)
        instructions, items = _native_input(messages)
        streamed_text = False
        completed = False
        async for event in self._model.stream_response(
            instructions,
            items,
            native_settings(self._settings, kwargs, self._tool_choice),
            list(self._tools),
            None
            if response_format is None
            else TransportedOutputSchema(response_format),
            [],
            ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        ):
            if isinstance(event, ResponseTextDeltaEvent) and event.delta:
                streamed_text = True
                yield LLMStreamChunk(content=event.delta)
            elif isinstance(event, ResponseCompletedEvent):
                if completed:
                    raise UnsupportedDurableOpenAIAgentsError(
                        "the model stream completed more than once"
                    )
                completed = True
                yield _completed_chunk(event, include_text=not streamed_text)
        # A stream that ends without completing would otherwise be recorded by
        # OE as a successful, truncated answer.
        if not completed:
            raise UnsupportedDurableOpenAIAgentsError(
                "the model stream ended without a completed response"
            )


def _native_input(
    messages: list[Message],
) -> tuple[str | None, list[TResponseInputItem]]:
    instructions: list[str] = []
    items: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message.content, str):
            raise UnsupportedDurableOpenAIAgentsError(
                "durable OpenAI Agents model calls carry text content only"
            )
        if message.role == "system":
            instructions.append(message.content)
        elif message.role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            )
        else:
            if message.content or not message.tool_calls:
                items.append({"role": message.role, "content": message.content})
            for call in message.tool_calls or []:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": json.dumps(
                            call.args if call.args is not None else {},
                            separators=(",", ":"),
                        ),
                    }
                )
    # The AER sent these through the same item contract; re-check before
    # they reach the provider.
    validate_items(items)
    return "\n\n".join(instructions) or None, cast(list[TResponseInputItem], items)


def _function_tool(schema: LLMToolSchema) -> FunctionTool:
    if not schema.name or not isinstance(schema.parameters, dict):
        raise UnsupportedDurableOpenAIAgentsError(
            "platform tool schemas require a name and JSON parameters"
        )

    async def not_executable(_context: object, _arguments: str) -> str:
        raise RuntimeError("tools advertised to a Tool Pod model never run here")

    return FunctionTool(
        name=schema.name,
        description=schema.description or "",
        params_json_schema=cast(dict[str, Any], schema.parameters),
        on_invoke_tool=not_executable,
        strict_json_schema=schema.strict is not False,
    )


def _completed_chunk(
    event: ResponseCompletedEvent, *, include_text: bool
) -> LLMStreamChunk:
    response = event.response
    text: list[str] = []
    calls: list[ToolCallChunk] = []
    for item in response.output:
        if isinstance(item, ResponseOutputMessage):
            for part in item.content:
                if not isinstance(part, ResponseOutputText):
                    raise UnsupportedDurableOpenAIAgentsError(
                        f"unsupported model output content {part.type!r}"
                    )
                text.append(part.text)
        elif isinstance(item, ResponseFunctionToolCall):
            calls.append(
                ToolCallChunk(
                    id=item.call_id,
                    name=item.name,
                    args=item.arguments,
                    type="function",
                    index=len(calls),
                )
            )
        elif not isinstance(item, ResponseReasoningItem):
            # Reasoning items are provider metadata; anything else is an effect
            # the durable runtime does not own.
            raise UnsupportedDurableOpenAIAgentsError(
                f"unsupported model output item {item.type!r}"
            )
    usage = response.usage
    return LLMStreamChunk(
        content="".join(text) if include_text and text else None,
        tool_calls=calls or None,
        id=response.id,
        usage=None
        if usage is None
        else LLMTokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            # Some providers complete a response without token details.
            reasoning_tokens=getattr(
                usage.output_tokens_details, "reasoning_tokens", None
            ),
        ),
    )
