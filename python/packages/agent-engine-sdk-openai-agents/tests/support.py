"""Test doubles shared by the adapter's model and turn tests.

They stand in for the two things a unit test cannot reach: OE, which carries a
model activity from the AER to the Tool Pod, and the provider model the Tool
Pod runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import pytest
from agent_engine_sdk import LLMInvocationOptions, LLMStreamChunk, Message
from agents import Model, ModelSettings, ModelTracing
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextDeltaEvent,
    ResponseUsage,
)

from agent_engine_runner_shared.hooks import get_llm_adapter_factory, get_named_llm
from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
from agent_engine_runner_shared.workflow.context import current_operation_path
from agent_engine_sdk_openai_agents import secure_model
from agent_engine_sdk_openai_agents.llm_adapter import OpenAIAgentsLLM, RegisteredModel


class FakeWrapper:
    oe_url = "http://oe"
    execution_id = "exec-1"
    durable_memory = None

    def __init__(self) -> None:
        self.operational_steps = object()
        self.step = 0

    def next_operational_step(self) -> int:
        self.step += 1
        return self.step

    def observe_operational_step(self, step: int) -> None:
        self.step = max(self.step, step)


class ScriptedNativeModel(Model):
    """A Tool Pod native model that answers with scripted output items."""

    def __init__(
        self,
        *turns: list[ResponseOutputItem],
        deltas: bool = True,
        completions: int = 1,
        usage: ResponseUsage | None = None,
        respond: Callable[[Any], list[ResponseOutputItem]] | None = None,
    ) -> None:
        self.turns = list(turns)
        # Answers from the request instead of the script, for models several
        # runs call at once in no fixed order.
        self.respond = respond
        self.deltas = deltas
        self.usage = usage
        # How many response.completed events to send; 1 is a well-formed stream.
        self.completions = completions
        self.calls: list[dict[str, Any]] = []

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the Tool Pod adapter streams")

    async def stream_response(  # type: ignore[override]
        self,
        system_instructions: str | None,
        input: Any,
        model_settings: ModelSettings,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: ModelTracing,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        self.calls.append(
            {
                "instructions": system_instructions,
                "input": input,
                "settings": model_settings,
                "tools": [tool.name for tool in tools],
                "output_schema": output_schema,
            }
        )
        output = self.respond(input) if self.respond else self.turns.pop(0)
        for item in output:
            if self.deltas and isinstance(item, ResponseOutputMessage):
                for part in item.content:
                    assert isinstance(part, ResponseOutputText)
                    yield ResponseTextDeltaEvent(
                        type="response.output_text.delta",
                        item_id=item.id,
                        output_index=0,
                        content_index=0,
                        delta=part.text,
                        logprobs=[],
                        sequence_number=0,
                    )
        for _ in range(self.completions):
            yield ResponseCompletedEvent(
                type="response.completed",
                response=Response(
                    id=f"resp-{len(self.calls)}",
                    created_at=0,
                    model="gpt-test",
                    object="response",
                    output=output,
                    tool_choice="auto",
                    tools=[],
                    parallel_tool_calls=False,
                    usage=self.usage,
                ),
                sequence_number=1,
            )


def text_item(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="native-msg",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def function_call_item(
    call_id: str, arguments: str = '{"order_id":"A1"}', name: str = "lookup_order"
) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"native-{call_id}",
        call_id=call_id,
        type="function_call",
        name=name,
        arguments=arguments,
    )


def fake_proxy_class(
    answer: Callable[[Any, dict[str, Any]], Iterator[LLMStreamChunk]],
    *,
    preallocate_tools: bool = False,
) -> type:
    """A stand-in ``SecureLLMProxy`` that answers each model call with ``answer``.

    ``preallocate_tools`` mirrors the real proxy's fold, which assigns each tool
    call's activity ordinal in the model's order before any tool runs; it needs
    a bound attempt.
    """

    class FakeProxy:
        instances: list[FakeProxy] = []
        response_from_stream_chunks = staticmethod(
            SecureLLMProxy.response_from_stream_chunks
        )

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.last_latest_step_number = None
            self.calls: list[dict[str, Any]] = []
            FakeProxy.instances.append(self)

        def stream(
            self, *, messages: list[Message], step: int, options: LLMInvocationOptions
        ) -> Iterator[LLMStreamChunk]:
            call = {"messages": messages, "step": step, "options": options}
            self.calls.append(call)
            chunks = list(answer(self, call))
            # The real proxy resolves the operation path when it admits the
            # activity, which may be after another run's call has finished.
            call["path"] = [
                (segment.name, segment.ordinal)
                for segment in current_operation_path().segments
            ]
            if preallocate_tools:
                fold = SecureLLMProxy._response_payload_from_chunks  # pyright: ignore[reportPrivateUsage]
                SecureLLMProxy._preallocate_tool_calls_from_result(fold(chunks))  # pyright: ignore[reportPrivateUsage]
            yield from chunks

    return FakeProxy


def route_to_tool_pod(
    monkeypatch: pytest.MonkeyPatch,
    registered: RegisteredModel | None = None,
    *,
    preallocate_tools: bool = False,
) -> type:
    """Stand in for OE: run each AER model activity on the Tool Pod adapter.

    Without ``registered``, resolve the model the way the Tool Pod does: from
    the LLM registry the app's entrypoint filled, through the registered
    adapter factory.
    """

    def answer(proxy: Any, call: dict[str, Any]) -> Iterator[LLMStreamChunk]:
        kwargs: dict[str, Any] = {
            "tools": proxy.kwargs["bound_tools"],
            "tool_choice": proxy.kwargs["bound_tool_choice"],
        }
        llm = (
            OpenAIAgentsLLM(registered, **kwargs)
            if registered is not None
            else get_llm_adapter_factory()(
                get_named_llm(proxy.kwargs["llm_id"]), **kwargs
            )
        )

        async def collect() -> list[LLMStreamChunk]:
            kwargs = call["options"].to_model_kwargs()
            return [chunk async for chunk in llm.astream(call["messages"], **kwargs)]

        yield from asyncio.run(collect())

    proxy_class = fake_proxy_class(answer, preallocate_tools=preallocate_tools)
    monkeypatch.setattr(secure_model, "SecureLLMProxy", proxy_class)
    return proxy_class
