from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
from agent_engine_sdk import (
    LLMStreamChunk,
    Message,
    ToolCallChunk,
)
from agents import (
    Agent,
    Model,
    ModelSettings,
    ModelTracing,
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
    RunConfig,
    Runner,
    function_tool,
)
from agents.models.default_models import get_default_model_settings
from openai import APIStatusError, AsyncOpenAI
from openai.types.responses import (
    ResponseFunctionWebSearch,
    ResponseReasoningItem,
    ResponseTextDeltaEvent,
    ResponseUsage,
)
from openai.types.shared import Reasoning
from pydantic import BaseModel

from agent_engine_runner_shared.secure_llm_proxy import SecureLLMProxy
from agent_engine_sdk_openai_agents import (
    UnsupportedDurableOpenAIAgentsError,
    secure_model,
)
from agent_engine_sdk_openai_agents.llm_adapter import OpenAIAgentsLLM, RegisteredModel
from agent_engine_sdk_openai_agents.secure_model import SecureModel
from tests.support import (
    FakeWrapper,
    ScriptedNativeModel,
    fake_proxy_class,
    function_call_item,
    route_to_tool_pod,
    text_item,
)


class Answer(BaseModel):
    status: str


@function_tool
def lookup_order(order_id: str) -> str:
    return f"{order_id} shipped"


async def test_runner_turn_round_trips_through_the_tool_pod_model(
    monkeypatch: pytest.MonkeyPatch, wrapper: FakeWrapper
) -> None:
    native = ScriptedNativeModel(
        [function_call_item("call-1")], [text_item('{"status":"shipped"}')]
    )
    proxy_class = route_to_tool_pod(
        monkeypatch,
        RegisteredModel(
            native,
            ModelSettings(
                temperature=0.2,
                top_p=0.9,
                reasoning=Reasoning(effort="low"),
                verbosity="medium",
            ),
        ),
    )
    agent = Agent(
        name="support",
        instructions="Answer order questions.",
        model=SecureModel(llm_id="__default__", model_name="gpt-test"),
        model_settings=ModelSettings(
            max_tokens=64,
            top_p=0.5,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            timeout=30.0,
            tool_choice="lookup_order",
            parallel_tool_calls=True,
            reasoning=Reasoning(effort="high"),
            verbosity="low",
        ),
        tools=[lookup_order],
        output_type=Answer,
    )

    result = Runner.run_streamed(
        agent, "Where is A1?", run_config=RunConfig(tracing_disabled=True)
    )
    deltas = [
        event.data.delta
        async for event in result.stream_events()
        if event.type == "raw_response_event"
        and isinstance(event.data, ResponseTextDeltaEvent)
    ]

    assert result.final_output == Answer(status="shipped")
    assert deltas == ['{"status":"shipped"}']
    first, second = native.calls
    assert first["instructions"] == "Answer order questions."
    assert first["input"] == [{"role": "user", "content": "Where is A1?"}]
    assert first["tools"] == ["lookup_order"]
    assert first["output_schema"].name() == "Answer"
    # Per-call settings win over the registration, which keeps the rest.
    assert (
        first["settings"].max_tokens,
        first["settings"].top_p,
        first["settings"].frequency_penalty,
        first["settings"].presence_penalty,
        first["settings"].timeout,
        first["settings"].tool_choice,
        first["settings"].temperature,
        first["settings"].parallel_tool_calls,
        first["settings"].reasoning,
        first["settings"].verbosity,
    ) == (
        64,
        0.5,
        0.1,
        0.2,
        30.0,
        "lookup_order",
        0.2,
        True,
        Reasoning(effort="high"),
        "low",
    )
    assert second["input"][1:] == [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "lookup_order",
            "arguments": '{"order_id":"A1"}',
        },
        {"type": "function_call_output", "call_id": "call-1", "output": "A1 shipped"},
    ]
    assert [proxy.calls[0]["step"] for proxy in proxy_class.instances] == [1, 2]
    call_item = result.to_input_list()[1]
    assert (call_item["call_id"], call_item["id"]) == ("call-1", "fc_call-1")


async def _stream(model: SecureModel, **overrides: Any) -> list[Any]:
    request: dict[str, Any] = {
        "system_instructions": None,
        "input": "hi",
        "model_settings": ModelSettings(parallel_tool_calls=False),
        "tools": [],
        "output_schema": None,
        "handoffs": [],
        "tracing": ModelTracing.DISABLED,
        "previous_response_id": None,
        "conversation_id": None,
        "prompt": None,
    }
    request.update(overrides)
    return [event async for event in model.stream_response(**request)]


def _fixed_answer(*chunks: LLMStreamChunk) -> Callable[..., Iterator[LLMStreamChunk]]:
    return lambda _proxy, _call: iter(chunks)


async def test_replayed_activity_reproduces_the_same_runner_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A replacement attempt replays the recorded response at the same
    # operational step; committed state compares the resulting items.
    recorded = _fixed_answer(
        LLMStreamChunk(content="Checking."),
        LLMStreamChunk(
            tool_calls=[
                ToolCallChunk(id="call-1", name="lookup_order", args="{}", index=0)
            ]
        ),
    )
    monkeypatch.setattr(secure_model, "SecureLLMProxy", fake_proxy_class(recorded))
    outputs = []
    for _attempt in range(2):
        monkeypatch.setattr(secure_model, "get_current_wrapper", FakeWrapper)
        events = await _stream(SecureModel(llm_id="x", model_name="m"))
        outputs.append([item.model_dump() for item in events[-1].response.output])

    assert outputs[0] == outputs[1]
    assert [item["id"] for item in outputs[0]] == ["msg_exec-1_1", "fc_call-1"]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"model_settings": ModelSettings(temperature=0)}, r"temperature.*app\.llm"),
        ({"model_settings": ModelSettings(retry={"max_retries": 2})}, "retries"),
        (
            {"model_settings": ModelSettings(reasoning=Reasoning(summary="auto"))},
            r"reasoning\.summary",
        ),
        (
            {
                "model_settings": ModelSettings(
                    reasoning=Reasoning(effort="high", mode="pro")
                )
            },
            r"reasoning\.mode",
        ),
        ({"previous_response_id": "resp-0"}, "continuation"),
        ({"conversation_id": "conv-0"}, "continuation"),
        ({"tracing": ModelTracing.ENABLED}, "tracing"),
        ({"tools": [object()]}, "outside platform tool execution"),
    ],
)
async def test_calls_that_would_bypass_oe_fail_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    wrapper: FakeWrapper,
    overrides: dict[str, Any],
    match: str,
) -> None:
    proxy_class = fake_proxy_class(_fixed_answer(LLMStreamChunk(content="x")))
    monkeypatch.setattr(secure_model, "SecureLLMProxy", proxy_class)

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match=match):
        await _stream(SecureModel(llm_id="x", model_name="m"), **overrides)
    assert proxy_class.instances == []


async def test_model_call_outside_an_aer_execution_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(secure_model, "get_current_wrapper", lambda: None)

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="AER execution"):
        await _stream(SecureModel(llm_id="x", model_name="m"))


@pytest.mark.parametrize("streamed", [False, True])
async def test_provider_token_usage_reaches_the_runner(
    monkeypatch: pytest.MonkeyPatch, wrapper: FakeWrapper, streamed: bool
) -> None:
    usage = ResponseUsage.model_validate(
        {
            "input_tokens": 12,
            "output_tokens": 3,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 2},
        }
    )
    route_to_tool_pod(
        monkeypatch,
        RegisteredModel(ScriptedNativeModel([text_item("Hi.")], usage=usage)),
    )
    agent = Agent(
        name="support", model=SecureModel(llm_id="__default__", model_name="m")
    )
    run_config = RunConfig(
        tracing_disabled=True,
        model_settings=ModelSettings(parallel_tool_calls=False),
    )

    if streamed:
        result = Runner.run_streamed(agent, "hi", run_config=run_config)
        async for _event in result.stream_events():
            pass
    else:
        result = await Runner.run(agent, "hi", run_config=run_config)

    runner_usage = result.context_wrapper.usage
    assert (
        runner_usage.input_tokens,
        runner_usage.output_tokens,
        runner_usage.total_tokens,
        runner_usage.output_tokens_details.reasoning_tokens,
    ) == (12, 3, 15, 2)


async def test_usage_without_token_details_still_completes(
    monkeypatch: pytest.MonkeyPatch, wrapper: FakeWrapper
) -> None:
    # Some providers complete a response with aggregate usage only.
    usage = ResponseUsage.model_construct(
        input_tokens=12, output_tokens=3, total_tokens=15, output_tokens_details=None
    )
    route_to_tool_pod(
        monkeypatch,
        RegisteredModel(ScriptedNativeModel([text_item("Hi.")], usage=usage)),
    )
    agent = Agent(
        name="support", model=SecureModel(llm_id="__default__", model_name="m")
    )

    result = await Runner.run(agent, "hi", run_config=RunConfig(tracing_disabled=True))

    runner_usage = result.context_wrapper.usage
    assert result.final_output == "Hi."
    assert (
        runner_usage.total_tokens,
        runner_usage.output_tokens_details.reasoning_tokens,
    ) == (15, 0)


async def test_settings_equal_to_the_sdk_defaults_survive_when_assigned(
    monkeypatch: pytest.MonkeyPatch, wrapper: FakeWrapper
) -> None:
    # The SDK clears Agent(model_settings=...) that equals its own defaults,
    # because it cannot tell them from a defaulted field. Assigning the same
    # settings after construction keeps them, and they still win per call.
    # The SDK's defaults depend on its default model, which the environment
    # may name.
    monkeypatch.delenv("OPENAI_DEFAULT_MODEL", raising=False)
    sdk_defaults = get_default_model_settings()
    assert sdk_defaults.reasoning is not None and sdk_defaults.verbosity == "low"
    native = ScriptedNativeModel([text_item("Hi.")])
    route_to_tool_pod(
        monkeypatch,
        RegisteredModel(
            native,
            ModelSettings(reasoning=Reasoning(effort="high"), verbosity="high"),
        ),
    )
    agent = Agent(
        name="support", model=SecureModel(llm_id="__default__", model_name="m")
    )
    agent.model_settings = sdk_defaults

    await Runner.run(agent, "hi", run_config=RunConfig(tracing_disabled=True))

    applied = native.calls[0]["settings"]
    assert (applied.reasoning, applied.verbosity) == (
        Reasoning(effort=sdk_defaults.reasoning.effort),
        "low",
    )


async def test_parallel_function_calls_reach_the_runner_in_model_order(
    monkeypatch: pytest.MonkeyPatch, wrapper: FakeWrapper
) -> None:
    calls = [
        ToolCallChunk(id=f"call-{n}", name="lookup_order", args=f'{{"n":{n}}}', index=n)
        for n in range(2)
    ]
    monkeypatch.setattr(
        secure_model,
        "SecureLLMProxy",
        fake_proxy_class(_fixed_answer(LLMStreamChunk(tool_calls=calls))),
    )

    events = await _stream(SecureModel(llm_id="x", model_name="m"))

    output = events[-1].response.output
    assert [(item.call_id, item.arguments) for item in output] == [
        ("call-0", '{"n":0}'),
        ("call-1", '{"n":1}'),
    ]


async def _tool_pod_chunks(
    native: Model,
    messages: list[Message],
    registered_settings: ModelSettings | None = None,
    **options: object,
) -> list[LLMStreamChunk]:
    llm = OpenAIAgentsLLM(
        RegisteredModel(native, registered_settings or ModelSettings())
    )
    return [chunk async for chunk in llm.astream(messages, **options)]


async def test_tool_pod_ignores_reasoning_and_keeps_completed_text_without_deltas() -> (
    None
):
    reasoning = ResponseReasoningItem(id="r1", type="reasoning", summary=[])
    native = ScriptedNativeModel([reasoning, text_item("done")], deltas=False)

    chunks = await _tool_pod_chunks(native, [Message(role="user", content="hi")])

    assert SecureLLMProxy.response_from_stream_chunks(chunks).content == "done"


@pytest.mark.parametrize(
    ("messages", "options", "match"),
    [
        (
            [Message(role="user", content="hi")],
            {"reasoning_effort": "extreme"},
            "reasoning_effort must be",
        ),
        (
            [Message(role="user", content="hi")],
            {"verbosity": "loud"},
            "verbosity must be",
        ),
        ([Message(role="user", content="hi")], {"top_k": 5}, "top_k"),
        ([Message(role="user", content="hi")], {"stop": ["x"]}, "stop"),
        (
            [Message(role="user", content="hi")],
            {
                "response_format": {
                    "type": "json_object",
                    "json_schema": {"name": "A", "schema": {}, "strict": True},
                }
            },
            "json_schema format",
        ),
        (
            [Message(role="user", content=[{"type": "text", "text": "hi"}])],
            {},
            "text content",
        ),
    ],
)
async def test_tool_pod_rejects_requests_outside_the_wire_contract(
    messages: list[Message], options: dict[str, object], match: str
) -> None:
    native = ScriptedNativeModel([text_item("unused")])

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match=match):
        await _tool_pod_chunks(native, messages, **options)
    assert native.calls == []


async def test_tool_pod_rejects_hosted_tool_output() -> None:
    hosted = ResponseFunctionWebSearch.model_validate(
        {
            "type": "web_search_call",
            "id": "ws1",
            "status": "completed",
            "action": {"type": "search", "query": "q"},
        }
    )
    native = ScriptedNativeModel([hosted])

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match="web_search_call"):
        await _tool_pod_chunks(native, [Message(role="user", content="hi")])


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        (ModelSettings(reasoning=Reasoning(summary="auto")), r"reasoning\.summary"),
        # Only the Runner reads retry; the Tool Pod would silently ignore it.
        (ModelSettings(retry={"max_retries": 2}), "retry has no effect"),
        (ModelSettings(timeout=30), "timeout has no effect"),
        # It would force the same tool on every call that names none.
        (ModelSettings(tool_choice="lookup_order"), "tool_choice cannot be registered"),
        # The provider would continue a conversation it stores.
        (
            ModelSettings(extra_body={"previous_response_id": "resp-0"}),
            r"extra_body sets previous_response_id",
        ),
        (
            ModelSettings(extra_args={"conversation": "conv-0", "prompt": {}}),
            r"extra_args sets conversation, prompt",
        ),
        # The adapter builds these request fields from the Runner's request.
        (
            ModelSettings(extra_body={"tools": [{"type": "web_search"}]}),
            r"extra_body sets tools",
        ),
        (
            ModelSettings(extra_args={"input": "x", "stream": False}),
            r"extra_args sets input, stream",
        ),
        # Raw extras would skip the reasoning checks and override each call.
        (
            ModelSettings(extra_body={"reasoning": {"effort": "high"}}),
            r"extra_body sets reasoning: use ModelSettings.reasoning",
        ),
        (
            ModelSettings(extra_args={"reasoning_effort": "high", "verbosity": "low"}),
            r"extra_args sets reasoning_effort, verbosity",
        ),
        # The Responses API carries verbosity under text, which the adapter builds.
        (
            ModelSettings(extra_body={"text": {"verbosity": "high"}}),
            r"extra_body sets text",
        ),
    ],
)
def test_tool_pod_rejects_registered_settings_it_cannot_honor(
    settings: ModelSettings, match: str
) -> None:
    native = ScriptedNativeModel([text_item("unused")])

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match=match):
        OpenAIAgentsLLM(RegisteredModel(native, settings))
    assert native.calls == []


def test_registered_extra_request_fields_that_continue_nothing_are_accepted() -> None:
    settings = ModelSettings(
        extra_body={"service_tier": "flex"}, extra_args={"user": "u-1"}, store=False
    )

    RegisteredModel("gpt-test", settings).validate()


@pytest.mark.parametrize(
    ("completions", "match"),
    [(0, "without a completed response"), (2, "more than once")],
)
async def test_tool_pod_fails_a_stream_without_exactly_one_completion(
    completions: int, match: str
) -> None:
    # A truncated or ambiguous stream must fail the activity instead of being
    # recorded as the model's answer.
    native = ScriptedNativeModel([text_item("The order has")], completions=completions)

    with pytest.raises(UnsupportedDurableOpenAIAgentsError, match=match):
        await _tool_pod_chunks(native, [Message(role="user", content="hi")])


async def test_tool_pod_invoke_works_inside_a_running_event_loop() -> None:
    llm = OpenAIAgentsLLM(RegisteredModel(ScriptedNativeModel([text_item("done")])))

    response = llm.invoke([Message(role="user", content="hi")])

    assert response.content == "done"


@pytest.mark.parametrize("per_call_effort", [None, "high"])
@pytest.mark.parametrize("api", ["responses", "chat"])
async def test_reasoning_settings_reach_the_provider_request(
    api: str, per_call_effort: str | None
) -> None:
    requests: list[dict[str, Any]] = []

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        # The body is the evidence; refusing the call ends it there.
        return httpx.Response(400, json={"error": {"message": "stop here"}})

    client = AsyncOpenAI(
        api_key="test",
        base_url="https://provider.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)),
    )
    native: Model = (
        OpenAIResponsesModel(model="m", openai_client=client)
        if api == "responses"
        else OpenAIChatCompletionsModel(model="m", openai_client=client)
    )
    registered = ModelSettings(reasoning=Reasoning(effort="low"), verbosity="medium")

    # A per-call effort wins; without one the registered effort applies. The
    # registered verbosity applies either way.
    per_call = {} if per_call_effort is None else {"reasoning_effort": per_call_effort}
    with pytest.raises(APIStatusError):
        await _tool_pod_chunks(
            native,
            [Message(role="user", content="hi")],
            registered_settings=registered,
            **per_call,
        )

    effort = per_call_effort or "low"
    (body,) = requests
    if api == "responses":
        assert body["reasoning"] == {"effort": effort}
        assert body["text"]["verbosity"] == "medium"
    else:
        assert (body["reasoning_effort"], body["verbosity"]) == (effort, "medium")
