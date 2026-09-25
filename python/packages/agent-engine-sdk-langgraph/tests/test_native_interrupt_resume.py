"""Real-LangGraph coverage for native interrupt suspend and resume."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from agent_engine_sdk import AgentInput, RequestContext, StreamEvent
from pydantic import BaseModel, JsonValue

from agent_engine_sdk_langgraph import LangGraphOutputParser, RawStreamItem
from agent_engine_sdk_langgraph.agent import LangGraphBaseAgent
from agent_engine_sdk_langgraph.runtime import App
from agent_engine_runner_shared import RuntimeMode, hooks
from agent_engine_runner_shared.context import (
    clear_execution_context,
    set_execution_context,
)
from agent_engine_runner_shared.models import ToolExecuteResponse
from agent_engine_runner_shared.secure_wrapper import SecureToolWrapper


class _NativeInterruptParser(LangGraphOutputParser):
    stream_modes = ("messages",)

    async def parse(
        self, item: RawStreamItem, ctx: RequestContext
    ) -> AsyncIterator[BaseModel | JsonValue]:
        if item.stream_mode != "messages":
            return
        message, _ = item.payload
        if not isinstance(message, (AIMessage, AIMessageChunk)):
            return
        content = str(message.content)
        if content.startswith(("before", "after")):
            yield {"kind": "parser", "text": content}

    async def on_stream_error(
        self, ctx: RequestContext, error: BaseException
    ) -> BaseModel | JsonValue | None:
        return None


async def _suspend(
    agent: LangGraphBaseAgent, session_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    events = [
        event
        async for event in agent.stream(
            RequestContext(session_id=session_id),
            AgentInput(payload={"message": "start"}),
        )
    ]
    assert [event.event for event in events] == ["suspend"]
    data = events[0].data
    assert isinstance(data, dict)
    metadata = data["metadata"]
    assert isinstance(metadata, dict)
    return data, metadata


async def _resume(
    agent: LangGraphBaseAgent,
    session_id: str,
    metadata: dict[str, Any],
    resume_data: Any,
) -> list[StreamEvent]:
    return [
        event
        async for event in agent.stream(
            RequestContext(
                session_id=session_id,
                resume=True,
                resume_data=resume_data,
                metadata=metadata,
            ),
            AgentInput(payload={"message": ""}),
        )
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "payload",
    [
        "approve?",
        42,
        True,
        None,
        ["a", 2, False, None],
        {"nested": {"items": [1, 1.5], "enabled": True}},
    ],
)
async def test_native_interrupt_accepts_arbitrary_json_payload(payload: Any) -> None:
    received: list[Any] = []

    def gate(state: MessagesState) -> dict[str, Any]:
        answer = interrupt(payload)
        received.append(answer)
        return {"messages": [AIMessage(content="resumed")]}

    builder = StateGraph(MessagesState)
    builder.add_node("gate", gate)
    builder.add_edge(START, "gate")
    builder.add_edge("gate", END)
    agent = LangGraphBaseAgent(builder.compile(checkpointer=InMemorySaver()))

    data, metadata = await _suspend(agent, f"payload-{type(payload).__name__}")
    assert data["interrupts"][0]["value"] == payload
    interrupt_id = data["interrupts"][0]["id"]

    events = await _resume(
        agent,
        f"payload-{type(payload).__name__}",
        metadata,
        {interrupt_id: {"accepted": True}},
    )

    assert [event.event for event in events] == ["result"]
    assert received == [{"accepted": True}]


@pytest.mark.anyio
async def test_native_interrupt_resumes_single_interrupt_with_plain_message() -> None:
    received: list[Any] = []

    def gate(state: MessagesState) -> dict[str, Any]:
        received.append(interrupt("question"))
        return {"messages": [AIMessage(content="resumed")]}

    builder = StateGraph(MessagesState)
    builder.add_node("gate", gate)
    builder.add_edge(START, "gate")
    builder.add_edge("gate", END)
    agent = LangGraphBaseAgent(builder.compile(checkpointer=InMemorySaver()))

    _, metadata = await _suspend(agent, "plain-message")
    events = await _resume(agent, "plain-message", metadata, "approved")

    assert [event.event for event in events] == ["result"]
    assert received == ["approved"]


@pytest.mark.anyio
async def test_legacy_human_review_dict_is_one_interrupt_answer() -> None:
    received: list[Any] = []

    def gate(state: MessagesState) -> dict[str, Any]:
        received.append(interrupt("legacy review"))
        return {"messages": [AIMessage(content="resumed")]}

    builder = StateGraph(MessagesState)
    builder.add_node("gate", gate)
    builder.add_edge(START, "gate")
    builder.add_edge("gate", END)
    agent = LangGraphBaseAgent(builder.compile(checkpointer=InMemorySaver()))
    human_review = {
        "human_review": {"decision": "approve", "reviewer_notes": "looks good"}
    }

    _, metadata = await _suspend(agent, "legacy-human-review")
    events = await _resume(agent, "legacy-human-review", metadata, human_review)

    assert [event.event for event in events] == ["result"]
    assert received == [human_review]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "answer",
    [
        "text",
        42,
        1.5,
        True,
        None,
        ["a", 2, False, None],
        {"nested": {"items": [1, 1.5], "enabled": True, "empty": None}},
    ],
)
async def test_native_interrupt_preserves_structured_resume_types(answer: Any) -> None:
    received: list[Any] = []

    def gate(state: MessagesState) -> dict[str, Any]:
        received.append(interrupt({"question": "value?"}))
        return {"messages": [AIMessage(content="resumed")]}

    builder = StateGraph(MessagesState)
    builder.add_node("gate", gate)
    builder.add_edge(START, "gate")
    builder.add_edge("gate", END)
    agent = LangGraphBaseAgent(builder.compile(checkpointer=InMemorySaver()))
    session_id = f"answer-{type(answer).__name__}-{answer!s}"

    data, metadata = await _suspend(agent, session_id)
    interrupt_id = data["interrupts"][0]["id"]
    events = await _resume(agent, session_id, metadata, {interrupt_id: answer})

    assert [event.event for event in events] == ["result"]
    assert received == [answer]
    assert type(received[0]) is type(answer)


@pytest.mark.anyio
async def test_tool_result_suspend_resumes_with_decision_as_tool_result() -> None:
    app = App(app_name="Native interrupt tool test")
    app._runtime.mode = RuntimeMode.AER
    app._register_hooks()

    @app.tool(is_local=False)
    def review_claim(claim_id: str) -> str:
        """Request a human decision for a claim."""
        return app.suspend(
            reason="claim_review",
            context={"claim_id": claim_id, "allowed_decisions": ["approve", "deny"]},
        )

    suspend_wire = review_claim("claim-1")
    wrapped_tool = app.get_tools()[0]
    wrapper = SecureToolWrapper(oe_url="http://oe:8000", execution_id="exec-tool")
    received: list[Any] = []

    def tool_node(state: MessagesState) -> dict[str, Any]:
        result = wrapped_tool.func(claim_id="claim-1", tool_call_id="call-1")
        content = result[0] if isinstance(result, tuple) else result
        received.append(json.loads(content))
        return {"messages": [AIMessage(content="tool resumed")]}

    builder = StateGraph(MessagesState)
    builder.add_node("tool", tool_node)
    builder.add_edge(START, "tool")
    builder.add_edge("tool", END)
    agent = LangGraphBaseAgent(builder.compile(checkpointer=InMemorySaver()))
    # HITL suspend is gated on OE's status channel, so the mock
    # must report status="suspend" for the wrapper to fire the interrupt.
    approval = ToolExecuteResponse(
        proceed=True,
        status="suspend",
        result=suspend_wire,
        duration_ms=1,
    )
    tokens = set_execution_context(
        execution_id="exec-tool",
        wrapper=wrapper,
        oe_url="http://oe:8000",
    )
    try:
        with patch(
            "agent_engine_runner_shared.secure_wrapper.request_oe_approval",
            return_value=approval,
        ) as request_approval:
            data, metadata = await _suspend(agent, "tool-result-suspend")
            expected_payload = {
                "suspend_reason": "claim_review",
                "suspend_context": {
                    "claim_id": "claim-1",
                    "allowed_decisions": ["approve", "deny"],
                },
            }
            assert data["interrupts"][0]["value"] == expected_payload
            interrupt_id = data["interrupts"][0]["id"]
            decision = {"decision": "approve", "reviewer_notes": "verified"}

            events = await _resume(
                agent,
                "tool-result-suspend",
                metadata,
                {interrupt_id: decision},
            )
    finally:
        clear_execution_context(tokens)
        hooks.reset_hooks()

    assert [event.event for event in events] == ["result"]
    assert received == [decision]
    assert request_approval.call_count == 2


@pytest.mark.anyio
async def test_local_tool_native_interrupt_runs_in_graph_context() -> None:
    app = App(app_name="In-process native interrupt tool test")
    app._runtime.mode = RuntimeMode.AER
    app._register_hooks()

    @app.tool(is_local=True)
    def review_claim(claim_id: str) -> Any:
        """Request a native framework decision for a claim."""
        return interrupt({"claim_id": claim_id, "question": "approve?"})

    wrapped_tool = app.get_tools()[0]
    wrapper = SecureToolWrapper(oe_url="http://oe:8000", execution_id="exec-tool")
    received: list[Any] = []

    def tool_node(state: MessagesState) -> dict[str, Any]:
        result = wrapped_tool.func(claim_id="claim-1", tool_call_id="call-1")
        received.append(result[0] if isinstance(result, tuple) else result)
        return {"messages": [AIMessage(content="tool resumed")]}

    builder = StateGraph(MessagesState)
    builder.add_node("tool", tool_node)
    builder.add_edge(START, "tool")
    builder.add_edge("tool", END)
    agent = LangGraphBaseAgent(builder.compile(checkpointer=InMemorySaver()))
    approval = ToolExecuteResponse(
        proceed=True,
        route_to="callback",
        latest_step_number=1,
    )
    tokens = set_execution_context(
        execution_id="exec-tool",
        wrapper=wrapper,
        oe_url="http://oe:8000",
    )
    try:
        with patch(
            "agent_engine_runner_shared.secure_wrapper.request_oe_approval",
            return_value=approval,
        ) as request_approval:
            with patch(
                "agent_engine_runner_shared.secure_wrapper.report_oe_result"
            ) as report_result:
                data, metadata = await _suspend(agent, "local-tool-native-interrupt")
                pending = data["interrupts"][0]
                assert pending["value"] == {
                    "claim_id": "claim-1",
                    "question": "approve?",
                }
                decision = {"decision": "approve"}

                events = await _resume(
                    agent,
                    "local-tool-native-interrupt",
                    metadata,
                    {pending["id"]: decision},
                )
    finally:
        clear_execution_context(tokens)
        hooks.reset_hooks()

    assert [event.event for event in events] == ["result"]
    assert received == [decision]
    assert request_approval.call_count == 2
    assert all(
        call.kwargs["is_local"] is True for call in request_approval.call_args_list
    )
    assert report_result.call_count == 2
    assert [call.kwargs["status"] for call in report_result.call_args_list] == [
        "interrupted",
        "success",
    ]
    assert report_result.call_args.kwargs["result"] == decision


@pytest.mark.anyio
async def test_custom_parser_events_continue_across_suspend_and_resume() -> None:
    def before(state: MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content="before suspend")]}

    def gate(state: MessagesState) -> dict[str, Any]:
        decision = interrupt({"question": "continue?"})
        return {"messages": [AIMessage(content=f"after resume:{decision}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("before", before)
    builder.add_node("gate", gate)
    builder.add_edge(START, "before")
    builder.add_edge("before", "gate")
    builder.add_edge("gate", END)
    agent = LangGraphBaseAgent(
        builder.compile(checkpointer=InMemorySaver()),
        output_parser=_NativeInterruptParser,
        use_custom_parser=True,
    )

    fresh_events = [
        event
        async for event in agent.stream(
            RequestContext(session_id="custom-parser"),
            AgentInput(payload={"message": "start"}),
        )
    ]
    assert [event.custom_event for event in fresh_events if event.custom_event] == [
        {"kind": "parser", "text": "before suspend"}
    ]
    assert fresh_events[-1].event == "suspend"
    suspend_data = fresh_events[-1].data
    assert isinstance(suspend_data, dict)
    metadata = suspend_data["metadata"]
    assert isinstance(metadata, dict)
    suspend_interrupts = suspend_data["interrupts"]
    assert isinstance(suspend_interrupts, list)
    first_interrupt = suspend_interrupts[0]
    assert isinstance(first_interrupt, dict)
    interrupt_id = first_interrupt["id"]
    assert isinstance(interrupt_id, str)

    resumed_events = await _resume(
        agent,
        "custom-parser",
        metadata,
        {interrupt_id: "approve"},
    )

    assert [event.custom_event for event in resumed_events if event.custom_event] == [
        {"kind": "parser", "text": "after resume:approve"}
    ]
    assert resumed_events[-1].event == "result"


@pytest.mark.anyio
async def test_native_interrupt_resumes_parallel_branches_in_one_map() -> None:
    received: dict[str, Any] = {}

    def branch(name: str) -> Any:
        def node(state: MessagesState) -> dict[str, Any]:
            received[name] = interrupt({"branch": name})
            return {"messages": [AIMessage(content=f"resumed:{name}")]}

        return node

    builder = StateGraph(MessagesState)
    for name in ("alpha", "beta", "gamma"):
        builder.add_node(name, branch(name))
        builder.add_edge(START, name)
        builder.add_edge(name, END)
    agent = LangGraphBaseAgent(builder.compile(checkpointer=InMemorySaver()))

    data, metadata = await _suspend(agent, "parallel")
    interrupts = data["interrupts"]
    assert {item["value"]["branch"] for item in interrupts} == {
        "alpha",
        "beta",
        "gamma",
    }
    resume_map = {
        item["id"]: {
            "branch": item["value"]["branch"],
            "position": index,
            "approved": index % 2 == 0,
        }
        for index, item in enumerate(interrupts)
    }

    events = await _resume(agent, "parallel", metadata, resume_map)

    assert [event.event for event in events] == ["result"]
    assert received == {answer["branch"]: answer for answer in resume_map.values()}


@pytest.mark.anyio
async def test_stream_native_interrupt_in_compiled_child_resumes_once() -> None:
    received: list[Any] = []

    @tool
    def human_review(claim_id: str) -> str:
        """Request a human decision for a claim."""
        decision = interrupt({"claim_id": claim_id, "question": "approve?"})
        received.append(decision)
        return str(decision)

    def request_review(state: MessagesState) -> dict[str, Any]:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "human_review",
                            "args": {"claim_id": "claim-1"},
                            "id": "review-claim-1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    child = StateGraph(MessagesState)
    child.add_node("tools", ToolNode([human_review]))
    child.add_edge(START, "tools")
    child.add_edge("tools", END)

    parent = StateGraph(MessagesState)
    parent.add_node("request_review", request_review)
    parent.add_node("review", child.compile())
    parent.add_edge(START, "request_review")
    parent.add_edge("request_review", "review")
    parent.add_edge("review", END)
    agent = LangGraphBaseAgent(parent.compile(checkpointer=InMemorySaver()))
    session_id = "compiled-child"

    suspend_data, metadata = await _suspend(agent, session_id)

    pending = suspend_data["interrupts"]
    assert len(pending) == 1
    assert pending[0]["id"]
    assert pending[0]["value"] == {
        "claim_id": "claim-1",
        "question": "approve?",
    }
    decision = {"decision": "approve"}
    completed_events = await _resume(
        agent,
        session_id,
        metadata,
        {pending[0]["id"]: decision},
    )

    assert [event.event for event in completed_events] == ["result"]
    assert received == [decision]
