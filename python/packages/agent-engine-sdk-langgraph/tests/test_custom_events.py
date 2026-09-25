"""Tests for the LangGraph AER custom-event transport."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, TypedDict
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from agent_engine_sdk import AgentInput, RequestContext
from pydantic import BaseModel, JsonValue

from agent_engine_sdk_langgraph import LangGraphOutputParser, RawStreamItem
from agent_engine_sdk_langgraph.agent import LangGraphBaseAgent
from agent_engine_sdk_langgraph.output_parser import (
    DisabledCustomEventTransport,
    LangGraphCustomEventTransport,
)
from agent_engine_runner_shared.custom_events import (
    clear_custom_event_transport,
    custom_event_transport,
    emit_custom_event,
    emit_custom_event_sync,
    set_custom_event_transport,
)
from agent_engine_runner_shared.server.chunk_types import CUSTOM_EVENT


class _SyncEmitState(TypedDict):
    messages: Annotated[list, add_messages]


class _InertMessagesParser(LangGraphOutputParser):
    """Registered parser that does not claim ``custom`` (platform fallback)."""

    stream_modes = ("messages",)

    async def parse(
        self, item: RawStreamItem, ctx: RequestContext
    ) -> AsyncIterator[BaseModel | JsonValue]:
        return
        yield  # pragma: no cover — async generator marker

    async def on_stream_error(
        self, ctx: RequestContext, error: BaseException
    ) -> BaseModel | JsonValue | None:
        return None


def test_langgraph_transport_writes_opaque_payload_from_sync_node() -> None:
    writer = MagicMock()
    with patch(
        "agent_engine_sdk_langgraph.output_parser.get_stream_writer",
        return_value=writer,
    ):
        transport = LangGraphCustomEventTransport()
        transport.emit_sync({"event": "step", "data": "sync node"})

    writer.assert_called_once_with({"event": "step", "data": "sync node"})


@pytest.mark.anyio
async def test_langgraph_transport_writes_opaque_payload_via_stream_writer() -> None:
    writer = MagicMock()
    with patch(
        "agent_engine_sdk_langgraph.output_parser.get_stream_writer",
        return_value=writer,
    ):
        transport = LangGraphCustomEventTransport()
        await transport.emit({"event": "step", "data": "Fetching account information"})

    writer.assert_called_once_with(
        {"event": "step", "data": "Fetching account information"}
    )


@pytest.mark.anyio
async def test_langgraph_transport_raises_when_writer_unavailable() -> None:
    with patch(
        "agent_engine_sdk_langgraph.output_parser.get_stream_writer",
        side_effect=RuntimeError("Called get_config outside of a runnable context"),
    ):
        transport = LangGraphCustomEventTransport()
        with pytest.raises(RuntimeError, match="stream writer is unavailable"):
            await transport.emit({"event": "step", "data": "x"})


@pytest.mark.anyio
async def test_disabled_transport_fails_closed() -> None:
    transport = DisabledCustomEventTransport()
    with pytest.raises(RuntimeError, match="only available during AER streaming"):
        await transport.emit({"event": "step", "data": "x"})
    with pytest.raises(RuntimeError, match="only available during AER streaming"):
        transport.emit_sync({"event": "step", "data": "x"})


@pytest.mark.anyio
async def test_async_context_installs_transport_for_emit_custom_event() -> None:
    writer = MagicMock()
    with patch(
        "agent_engine_sdk_langgraph.output_parser.get_stream_writer",
        return_value=writer,
    ):
        with custom_event_transport(LangGraphCustomEventTransport()):
            await emit_custom_event({"event": "message", "data": "hello"})

    writer.assert_called_once_with({"event": "message", "data": "hello"})


def test_sync_transport_dispatches_to_sync_node() -> None:
    writer = MagicMock()
    with patch(
        "agent_engine_sdk_langgraph.output_parser.get_stream_writer",
        return_value=writer,
    ):
        transport = LangGraphCustomEventTransport()
        token = set_custom_event_transport(transport)
        try:
            emit_custom_event_sync({"event": "message", "data": "hello"})
        finally:
            clear_custom_event_transport(token)

    writer.assert_called_once_with({"event": "message", "data": "hello"})


@pytest.mark.anyio
async def test_real_sync_langgraph_node_emits_via_agent_stream() -> None:
    """Sync LangGraph nodes deliver opaque payloads through Atlas Agent Engine stream().

    Client-visible delivery requires ``use_custom_parser`` (and Atlas Agent Engine only
    then subscribes to LangGraph ``custom``).
    """

    def sync_node(state: _SyncEmitState) -> dict:
        emit_custom_event_sync({"event": "step", "data": "from sync node"})
        return {"messages": [AIMessage(content="done")]}

    graph = StateGraph(_SyncEmitState)
    graph.add_node("n", sync_node)
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    agent = LangGraphBaseAgent(
        graph.compile(),
        output_parser=_InertMessagesParser,
        use_custom_parser=True,
    )

    events = [
        e
        async for e in agent.stream(
            RequestContext(), AgentInput(payload={"message": "hi"})
        )
    ]

    customs = [e for e in events if e.custom_event is not None]
    assert [e.custom_event for e in customs] == [
        {"event": "step", "data": "from sync node"}
    ]
    assert all(e.event == CUSTOM_EVENT for e in customs)
    assert events[-1].event == "result"
    assert isinstance(events[-1].data, dict)
    assert events[-1].data["response"] == "done"
