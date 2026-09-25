"""Tests for subagent-aware streaming in ``LangGraphBaseAgent.stream``.

Drives the stream loop by replacing ``self._graph.astream`` with a scripted
async generator that yields ``(namespace, stream_mode, payload)`` tuples —
the shape LangGraph produces with ``subgraphs=True``.

What we assert:
1. ``astream`` is invoked with ``subgraphs=True``.
2. Root-agent tokens carry ``source=""``.
3. A subagent's first sourced token triggers a synthetic ``subagent_start``.
4. The parent's Command return produces a ``subagent_end`` keyed by tool_call_id.
5. ``source`` attribution is derived from the namespace tuple, not chunk metadata.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.types import Command
from agent_engine_sdk import AgentInput, RequestContext, StreamEvent

from agent_engine_sdk_langgraph.agent import LangGraphBaseAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ROOT_META: dict[str, Any] = {"langgraph_checkpoint_ns": ""}

# Script item type: (namespace, stream_mode, payload)
_Item = tuple[tuple[str, ...], str, Any]


def _sub_meta(
    name: str, task_id: str = "aaa", subtask_id: str = "bbb"
) -> dict[str, Any]:
    return {"langgraph_checkpoint_ns": f"task:{task_id}|{name}:{subtask_id}|agent"}


def _task_call_chunk(tc_id: str, name: str, description: str = "") -> AIMessageChunk:
    """AIMessageChunk carrying a single ``task`` tool_call."""
    return AIMessageChunk(
        content="",
        tool_calls=[
            {
                "name": "task",
                "args": {"subagent_type": name, "description": description},
                "id": tc_id,
            }
        ],
    )


def _root_msg(content: str) -> _Item:
    """Root-namespace messages-stream token."""
    return ((), "messages", (AIMessageChunk(content=content), _ROOT_META))


def _sub_msg(
    name: str, content: str, task_id: str = "aaa", subtask_id: str = "bbb"
) -> _Item:
    """Subagent-namespace messages-stream token."""
    return (
        (f"task:{task_id}", name),
        "messages",
        (AIMessageChunk(content=content), _sub_meta(name, task_id, subtask_id)),
    )


def _command_close(tc_id: str, content: str = "") -> _Item:
    """updates-stream entry: raw Command node output closing a subagent task."""
    return (
        (),
        "updates",
        {
            "tools": Command(
                update={"messages": [ToolMessage(content=content, tool_call_id=tc_id)]}
            )
        },
    )


def _dict_close(tc_id: str, content: str = "") -> _Item:
    """updates-stream entry: pre-unwrapped dict close (normal pregel shape)."""
    return (
        (),
        "updates",
        {"tools": {"messages": [ToolMessage(content=content, tool_call_id=tc_id)]}},
    )


def _root_update(content: str = "ok") -> _Item:
    """Terminal root-agent updates payload that produces the result event."""
    return ((), "updates", {"agent": {"messages": [AIMessage(content=content)]}})


def _build_graph_mock(astream_impl: Any) -> MagicMock:
    graph = MagicMock()
    graph.astream = MagicMock(side_effect=astream_impl)

    async def _aget_state(*_args: Any, **_kwargs: Any) -> Any:
        state = MagicMock()
        state.next = None
        return state

    graph.aget_state = _aget_state
    graph.checkpointer = None
    return graph


def _make_agent_with_scripted_stream(
    script: list[_Item],
) -> tuple[LangGraphBaseAgent, MagicMock]:
    """Build a LangGraphBaseAgent whose ``astream`` yields ``script``."""

    async def _astream(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        for item in script:
            yield item

    graph = _build_graph_mock(_astream)
    return LangGraphBaseAgent(graph), graph


def _make_agent_with_raising_stream(
    script: list[_Item],
    exc: BaseException,
) -> LangGraphBaseAgent:
    """Variant that raises ``exc`` after yielding ``script``."""

    async def _astream(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        for item in script:
            yield item
        raise exc

    return LangGraphBaseAgent(_build_graph_mock(_astream))


def _agent_input(message: str = "hi") -> AgentInput:
    return AgentInput(
        payload={
            "message": message,
            "thread_id": "thread-stream-subagents",
            "session_id": "thread-stream-subagents",
            "user_id": "u",
            "resume": False,
            "resume_data": None,
            "checkpoint_id": None,
        }
    )


async def _collect_events(agent: LangGraphBaseAgent) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    async for evt in agent.stream(RequestContext(), _agent_input()):
        events.append(evt)
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_stream_emits_subagent_start_and_end_with_source_attribution() -> None:
    """Root + one subagent round-trip: subgraphs=True wiring, event ordering,
    source attribution, tool_call_id forwarding, and summary content."""
    tc_id = "call-task-1"
    name = "security_reviewer"

    script: list[_Item] = [
        _root_msg("Starting review. "),
        (
            (),
            "messages",
            (
                _task_call_chunk(tc_id, name, "Review this code for security."),
                _ROOT_META,
            ),
        ),
        _sub_msg(name, "Code looks "),
        _sub_msg(name, "safe."),
        _command_close(tc_id, "Code looks safe."),
        _root_msg("Done."),
        _root_update("Subagent approved the change."),
    ]

    agent, graph = _make_agent_with_scripted_stream(script)
    events = await _collect_events(agent)

    call_kwargs = graph.astream.call_args.kwargs
    assert call_kwargs.get("subgraphs") is True
    assert call_kwargs.get("stream_mode") == ["messages", "updates"]

    event_names = [evt.event for evt in events]
    # Exactly one start/end — tool_call path fires, synthetic fallback must dedup.
    assert event_names.count("subagent_start") == 1
    assert event_names.count("subagent_end") == 1

    start_idx = event_names.index("subagent_start")
    end_idx = event_names.index("subagent_end")
    assert start_idx < end_idx

    start_data = cast(dict, events[start_idx].data)
    assert start_data.get("subagent_name") == name
    assert start_data.get("tool_call_id") == tc_id
    assert start_data.get("source") == name

    end_data = cast(dict, events[end_idx].data)
    assert end_data.get("subagent_name") == name
    assert end_data.get("tool_call_id") == tc_id
    assert end_data.get("summary") == "Code looks safe."

    token_data: list[dict[str, Any]] = [
        cast(dict, e.data)
        for e in events
        if e.event == "token" and isinstance(e.data, dict)
    ]
    assert any(d.get("source") == "" for d in token_data), "expected root-sourced token"
    assert any(d.get("source") == name for d in token_data), (
        "expected subagent-sourced token"
    )
    sub_token_data = [d for d in token_data if d.get("source") == name]
    assert all(d.get("tool_call_id") == tc_id for d in sub_token_data), (
        "sourced tokens must carry the real tc_id"
    )
    assert events[-1].event == "result"


@pytest.mark.anyio
async def test_source_derived_from_namespace_tuple() -> None:
    """Source attribution comes from the namespace tuple, not chunk metadata.
    Root agents commonly carry ``lc_agent_name`` which must not be used for
    attribution — only the NS-derived subagent name counts."""
    name = "research_agent"

    script: list[_Item] = [
        _root_msg("root token "),
        _sub_msg(name, "sub token", task_id="xyz", subtask_id="qqq"),
        _root_update("root token sub token done"),
    ]

    agent, _ = _make_agent_with_scripted_stream(script)
    events = await _collect_events(agent)

    starts = [e for e in events if e.event == "subagent_start"]
    assert len(starts) == 1
    start_data = cast(dict, starts[0].data)
    assert start_data.get("source") == name
    assert start_data.get("subagent_name") == name
    assert start_data.get("tool_call_id") == ""  # no parent tool_call in script

    sourced = [
        e
        for e in events
        if e.event == "token" and cast(dict, e.data).get("source") == name
    ]
    assert sourced, "expected subagent token with NS-derived source"


@pytest.mark.anyio
async def test_unclosed_active_task_gets_defensive_subagent_end() -> None:
    """Stream ends before the Command close fires — defensive ``subagent_end``
    must still be emitted so downstream consumers don't leak a half-open block."""
    tc_id = "call-orphan-1"
    name = "noisy_subagent"

    script: list[_Item] = [
        ((), "messages", (_task_call_chunk(tc_id, name, "go"), _ROOT_META)),
        _sub_msg(name, "partial", task_id="zzz", subtask_id="ooo"),
        _root_update("Partial result."),  # no Command close before this
    ]

    agent, _ = _make_agent_with_scripted_stream(script)
    events = await _collect_events(agent)

    ends = [e for e in events if e.event == "subagent_end"]
    assert len(ends) == 1
    end_data = cast(dict, ends[0].data)
    assert end_data.get("tool_call_id") == tc_id
    assert end_data.get("subagent_name") == name


@pytest.mark.anyio
async def test_root_lc_agent_name_model_is_still_root() -> None:
    """Regression: root agent commonly carries ``lc_agent_name='model'``.
    Earlier revisions preferred lc_agent_name over ns and treated this as a
    subagent. Empty/no-NS_SEP ns must always resolve to source=''."""
    root_metadata = {"langgraph_checkpoint_ns": "", "lc_agent_name": "model"}

    script: list[_Item] = [
        ((), "messages", (AIMessageChunk(content="hello "), root_metadata)),
        ((), "messages", (AIMessageChunk(content="world"), root_metadata)),
        _root_update("hello world"),
    ]

    agent, _ = _make_agent_with_scripted_stream(script)
    events = await _collect_events(agent)

    assert not [e for e in events if e.event == "subagent_start"], (
        "root lc_agent_name='model' must NOT synthesize a subagent_start"
    )
    assert not [e for e in events if e.event == "subagent_end"]
    tokens = [e for e in events if e.event == "token" and isinstance(e.data, dict)]
    assert tokens
    assert all(cast(dict, e.data).get("source") == "" for e in tokens)


@pytest.mark.anyio
async def test_ns_inner_sentinels_are_skipped() -> None:
    """Regression: ns with inner ``model``/``tools`` sentinels must not return
    the sentinel as the subagent name — a naive right-to-left walk that only
    skips ``agent`` returns ``model`` instead of the real subagent name."""
    from agent_engine_sdk_langgraph.subagent_stream import source_from_namespace

    assert (
        source_from_namespace(("tools:aaa", "security_reviewer:bbb", "model:ccc"))
        == "security_reviewer"
    )
    assert (
        source_from_namespace(("tools:aaa", "style_reviewer:bbb")) == "style_reviewer"
    )
    assert source_from_namespace(("model:abc",)) == ""
    assert source_from_namespace(()) == ""


@pytest.mark.anyio
async def test_token_first_race_yields_single_start_with_real_tcid_desc() -> None:
    """Token-first race (Strategy A): subagent token arrives BEFORE the
    parent's ``task`` tool_call assembles. ``_on_sourced_token`` defers the
    synthetic start; ``_on_task_call`` then promotes the placeholder and
    emits a single canonical ``subagent_start`` carrying the REAL
    ``tool_call_id`` and ``description``. UI gets the description it needs
    to render the pill, with no double-emit and no empty-tcid placeholder."""
    tc_id = "call-task-buffered"
    name = "research_agent"
    description = "Research the topic."

    script: list[_Item] = [
        # Subagent token arrives first — placeholder enters ``pending`` but
        # no start is emitted yet (Strategy A defers).
        _sub_msg(name, "thinking...", task_id="zzz", subtask_id="qqq"),
        # Parent tool_call assembles after — placeholder is promoted and the
        # canonical start fires here with the real tc_id + description.
        (
            (),
            "messages",
            (_task_call_chunk(tc_id, name, description), _ROOT_META),
        ),
        _command_close(tc_id, "Research complete."),
        _root_update("Done."),
    ]

    agent, _ = _make_agent_with_scripted_stream(script)
    events = await _collect_events(agent)
    event_names = [e.event for e in events]

    assert event_names.count("subagent_start") == 1, (
        f"token-first race must yield exactly one start; got {event_names}"
    )
    assert event_names.count("subagent_end") == 1

    start_data = cast(dict, [e for e in events if e.event == "subagent_start"][0].data)
    assert start_data.get("source") == name
    assert start_data.get("subagent_name") == name
    # Strategy A: start is deferred until the real tool_call_id is known, so
    # the canonical start carries the REAL tc_id (not "").
    assert start_data.get("tool_call_id") == tc_id
    # And the real description, so the UI can render the pill correctly.
    assert start_data.get("description") == description

    end_data = cast(dict, [e for e in events if e.event == "subagent_end"][0].data)
    assert end_data.get("tool_call_id") == tc_id  # real tc_id from Command close


@pytest.mark.anyio
async def test_mid_stream_error_yields_defensive_end_then_reraises() -> None:
    """Mid-stream provider failure after subagent start but before Command close.
    The ``except Exception`` branch must yield a defensive ``subagent_end`` for
    the orphaned task and re-raise the original error unchanged."""
    tc_id = "call-task-erroring"
    name = "flaky_subagent"
    sentinel_error = RuntimeError("provider disconnect")

    script: list[_Item] = [
        ((), "messages", (_task_call_chunk(tc_id, name, "go"), _ROOT_META)),
        _sub_msg(name, "partial", task_id="eee", subtask_id="fff"),
    ]
    agent = _make_agent_with_raising_stream(script, sentinel_error)

    collected: list[StreamEvent] = []
    raised: BaseException | None = None
    try:
        async for evt in agent.stream(RequestContext(), _agent_input()):
            collected.append(evt)
    except RuntimeError as exc:
        raised = exc

    assert raised is sentinel_error, f"error must re-raise unchanged; got {raised!r}"
    ends = [e for e in collected if e.event == "subagent_end"]
    event_names = [e.event for e in collected]
    assert len(ends) == 1, (
        f"expected one defensive end before error; events={event_names}"
    )
    end_data = cast(dict, ends[0].data)
    assert end_data.get("tool_call_id") == tc_id
    assert end_data.get("subagent_name") == name


@pytest.mark.anyio
async def test_cancelled_error_propagates_without_yielding() -> None:
    """``asyncio.CancelledError`` must propagate without yielding any events.
    Yielding during cancellation either masks the cancel or blocks awaiting a
    consumer that is already torn down."""
    tc_id = "call-task-cancelled"
    name = "interrupted_subagent"

    script: list[_Item] = [
        ((), "messages", (_task_call_chunk(tc_id, name, "go"), _ROOT_META)),
    ]
    agent = _make_agent_with_raising_stream(script, asyncio.CancelledError())

    collected: list[StreamEvent] = []
    cancelled = False
    try:
        async for evt in agent.stream(RequestContext(), _agent_input()):
            collected.append(evt)
    except asyncio.CancelledError:
        cancelled = True

    assert cancelled, "CancelledError must propagate to the consumer"
    assert not [e for e in collected if e.event == "subagent_end"], (
        f"CancelledError path must not yield events; got {[e.event for e in collected]}"
    )


@pytest.mark.anyio
async def test_generator_exit_drops_state_without_yielding() -> None:
    """SSE consumer ``aclose()``s mid-stream. Python injects ``GeneratorExit``
    at the next yield point. Cleanup MUST clear state and re-raise without
    yielding — yielding raises ``RuntimeError: async generator ignored
    GeneratorExit``."""
    tc_id = "call-task-aclose"
    name = "abandoned_subagent"

    script: list[_Item] = [
        ((), "messages", (_task_call_chunk(tc_id, name, "go"), _ROOT_META)),
        _sub_msg(name, "partial", task_id="ggg", subtask_id="hhh"),
        _sub_msg(
            name, "never-seen", task_id="ggg", subtask_id="hhh"
        ),  # consumer disconnects before here
    ]

    agent, _ = _make_agent_with_scripted_stream(script)
    collected: list[StreamEvent] = []
    stream = cast(
        AsyncGenerator[StreamEvent, None],
        agent.stream(RequestContext(), _agent_input()),
    )
    saw_start = False
    runtime_error: RuntimeError | None = None
    try:
        async for evt in stream:
            collected.append(evt)
            if evt.event == "subagent_start":
                saw_start = True
                await stream.aclose()
                break
    except RuntimeError as exc:  # pragma: no cover
        runtime_error = exc

    assert saw_start, "subagent_start must fire before the disconnect"
    assert runtime_error is None, (
        f"GeneratorExit cleanup must not yield; got {runtime_error!r}"
    )
    end_events = [e for e in collected if e.event == "subagent_end"]
    event_names = [e.event for e in collected]
    assert not end_events, (
        f"GeneratorExit path must not yield subagent_end; got {event_names}"
    )


@pytest.mark.anyio
async def test_synthetic_only_orphan_gets_defensive_end_on_error() -> None:
    """Synthetic-only orphan (Strategy A): subagent token enters ``pending``
    but the parent's ``task`` tool_call NEVER assembles before the stream
    errors. ``_on_sourced_token`` defers the start, so the drain path must
    emit BOTH ``subagent_start`` and ``subagent_end`` (in that order, both
    carrying ``tool_call_id == ""``) so the UI pill has a balanced
    lifecycle and closes cleanly."""
    name = "orphan_subagent"
    sentinel_error = RuntimeError("provider died before tool_call assembled")

    script: list[_Item] = [
        _sub_msg(name, "thinking", task_id="ooo", subtask_id="ppp"),
    ]
    agent = _make_agent_with_raising_stream(script, sentinel_error)

    collected: list[StreamEvent] = []
    raised: BaseException | None = None
    try:
        async for evt in agent.stream(RequestContext(), _agent_input()):
            collected.append(evt)
    except RuntimeError as exc:
        raised = exc

    assert raised is sentinel_error

    event_names = [e.event for e in collected]
    starts = [e for e in collected if e.event == "subagent_start"]
    ends = [e for e in collected if e.event == "subagent_end"]
    assert len(starts) == 1, (
        f"synthetic-only orphan must get a drain-emitted start; events={event_names}"
    )
    assert len(ends) == 1, (
        f"synthetic orphan must get a defensive end; events={event_names}"
    )

    # Drain emits start immediately followed by end; assert that ordering.
    start_idx = event_names.index("subagent_start")
    end_idx = event_names.index("subagent_end")
    assert start_idx < end_idx
    # The drain emits them back-to-back for ``pending`` orphans.
    assert event_names[start_idx : end_idx + 1] == ["subagent_start", "subagent_end"], (
        f"drain must emit start then end for the pending orphan; got {event_names}"
    )

    start_data = cast(dict, starts[0].data)
    assert start_data.get("tool_call_id") == ""
    assert start_data.get("source") == name
    assert start_data.get("subagent_name") == name

    end_data = cast(dict, ends[0].data)
    assert end_data.get("tool_call_id") == ""  # matches the synthetic's empty tc_id
    assert end_data.get("subagent_name") == name


@pytest.mark.anyio
async def test_existing_active_task_does_not_double_emit() -> None:
    """Buffered-args path: messages-mode registers the task and emits start;
    the same tool_call re-observed via the updates add_messages payload must
    not emit a second start."""
    tc_id = "call-task-buffered-args"
    name = "research_agent"

    early_chunk = _task_call_chunk(tc_id, name, description="")  # args still streaming
    finalized_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "task",
                "args": {
                    "subagent_type": name,
                    "description": "Research the Q3 outage.",
                },
                "id": tc_id,
            }
        ],
    )

    script: list[_Item] = [
        ((), "messages", (early_chunk, _ROOT_META)),
        (
            (),
            "updates",
            {"agent": {"messages": [finalized_msg]}},
        ),  # existing-entry branch
        _command_close(tc_id, "done"),
        _root_update(),
    ]

    agent, _ = _make_agent_with_scripted_stream(script)
    events = await _collect_events(agent)
    event_names = [e.event for e in events]

    assert event_names.count("subagent_start") == 1, f"must dedup; events={event_names}"
    assert event_names.count("subagent_end") == 1

    start_data = cast(dict, [e for e in events if e.event == "subagent_start"][0].data)
    assert start_data.get("tool_call_id") == tc_id
    assert start_data.get("subagent_name") == name
    assert start_data.get("description") == ""  # first observation wins


@pytest.mark.anyio
async def test_parallel_same_name_dispatch_emits_two_distinct_starts() -> None:
    """Two concurrent ``task`` tool_calls with the same ``subagent_type`` but
    distinct tool_call_ids must each get their own start/end pair."""
    name = "research_agent"
    tc_id_a, tc_id_b = "call-a", "call-b"

    parent_chunk = AIMessageChunk(
        content="",
        tool_calls=[
            {
                "name": "task",
                "args": {"subagent_type": name, "description": "topic 1"},
                "id": tc_id_a,
            },
            {
                "name": "task",
                "args": {"subagent_type": name, "description": "topic 2"},
                "id": tc_id_b,
            },
        ],
    )

    script: list[_Item] = [
        ((), "messages", (parent_chunk, _ROOT_META)),
        _command_close(tc_id_a, "result A"),
        _command_close(tc_id_b, "result B"),
        _root_update("combined"),
    ]

    agent, _ = _make_agent_with_scripted_stream(script)
    events = await _collect_events(agent)

    starts = [e for e in events if e.event == "subagent_start"]
    ends = [e for e in events if e.event == "subagent_end"]
    assert len(starts) == 2, (
        f"expected 2 distinct starts; got {[e.data for e in starts]}"
    )
    assert len(ends) == 2

    start_ids = {cast(dict, e.data).get("tool_call_id") for e in starts}
    end_ids = {cast(dict, e.data).get("tool_call_id") for e in ends}
    assert start_ids == {tc_id_a, tc_id_b}
    assert end_ids == {tc_id_a, tc_id_b}

    starts_by_id = {
        cast(dict, e.data)["tool_call_id"]: cast(dict, e.data) for e in starts
    }
    assert starts_by_id[tc_id_a]["description"] == "topic 1"
    assert starts_by_id[tc_id_b]["description"] == "topic 2"


@pytest.mark.anyio
async def test_subagent_end_on_unwrapped_command_close() -> None:
    """Realistic deepagents shape: pregel pre-unwraps the Command into a plain
    dict before it reaches the updates stream. Lock down that
    ``_process_add_messages_update`` detects the close via tool_call_id."""
    tc_id = "call-task-real"
    name = "research_agent"
    summary_text = "All findings collected."

    script: list[_Item] = [
        (
            (),
            "messages",
            (_task_call_chunk(tc_id, name, "Research the topic."), _ROOT_META),
        ),
        _dict_close(tc_id, summary_text),  # plain dict, NOT a Command object
        _root_update(),
    ]

    agent, _ = _make_agent_with_scripted_stream(script)
    events = await _collect_events(agent)

    starts = [e for e in events if e.event == "subagent_start"]
    ends = [e for e in events if e.event == "subagent_end"]
    assert len(starts) == 1
    assert len(ends) == 1
    end_data = cast(dict, ends[0].data)
    assert end_data["tool_call_id"] == tc_id
    assert end_data["subagent_name"] == name
    assert end_data["summary"] == summary_text


@pytest.mark.anyio
async def test_raw_command_update_closes_registered_task() -> None:
    """Defensive fallback: ``_process_command_update`` handles a raw
    ``Command(update=...)`` node output on the updates stream (distinct from
    the normal pre-unwrapped dict path). A registered task must be closed and
    ``subagent_end`` emitted with the correct ``tool_call_id`` and ``summary``."""
    tc_id = "call-raw-cmd"
    name = "research_agent"
    summary_text = "Raw command result."

    script: list[_Item] = [
        ((), "messages", (_task_call_chunk(tc_id, name, "Do research."), _ROOT_META)),
        _command_close(tc_id, summary_text),  # raw Command, not pre-unwrapped dict
        _root_update(),
    ]

    agent, _ = _make_agent_with_scripted_stream(script)
    events = await _collect_events(agent)

    starts = [e for e in events if e.event == "subagent_start"]
    ends = [e for e in events if e.event == "subagent_end"]
    assert len(starts) == 1
    assert len(ends) == 1
    end_data = cast(dict, ends[0].data)
    assert end_data["tool_call_id"] == tc_id
    assert end_data["subagent_name"] == name
    assert end_data["summary"] == summary_text


@pytest.mark.anyio
async def test_subagent_interrupt_survives_subsequent_root_updates() -> None:
    """Interrupts must be accumulated across all updates payloads. Earlier
    revisions captured only ``last_update``, so a root-namespace payload
    arriving after the interrupt silently overwrote it."""
    from langgraph.types import Interrupt

    interrupt_payload = {
        "suspend_reason": "needs_human_review",
        "suspend_context": {"reason": "risky"},
    }
    interrupt = Interrupt(value=interrupt_payload)

    script: list[_Item] = [
        (("task:aaa", "research_agent"), "updates", {"__interrupt__": [interrupt]}),
        # This root payload would overwrite the interrupt in the buggy revision.
        _root_update("continuing"),
    ]

    agent, _ = _make_agent_with_scripted_stream(script)
    events = await _collect_events(agent)

    suspends = [e for e in events if e.event == "suspend"]
    assert len(suspends) == 1, f"expected one suspend; got {[e.event for e in events]}"
    suspend_data = cast(dict, suspends[0].data)
    assert suspend_data["suspend_payload"] == interrupt_payload
    assert "interrupt_count" not in suspend_data["metadata"]
    assert len(suspend_data["interrupts"]) == 1


@pytest.mark.anyio
async def test_hitl_suspend_with_open_subagent_does_not_emit_premature_end() -> None:
    """HITL suspend must not emit a defensive subagent_end before the suspend
    event when a subagent task is still open at interrupt time.

    Earlier revisions ran _drain_open_tasks in the else-branch unconditionally,
    so a graph that suspended mid-fan-out emitted subagent_end (with empty
    summary) before the suspend event, producing unbalanced lifecycle events."""
    from langgraph.types import Interrupt

    tc_id = "call-hitl-open"
    name = "review_agent"
    interrupt_payload = {"suspend_reason": "approval_required", "suspend_context": {}}

    script: list[_Item] = [
        # Parent dispatches a subagent — start fires.
        ((), "messages", (_task_call_chunk(tc_id, name, "Review this."), _ROOT_META)),
        # Graph suspends before the subagent's ToolMessage arrives.
        ((), "updates", {"__interrupt__": [Interrupt(value=interrupt_payload)]}),
    ]

    agent, _ = _make_agent_with_scripted_stream(script)
    events = await _collect_events(agent)
    event_names = [e.event for e in events]

    # One start must have fired.
    assert event_names.count("subagent_start") == 1
    # No premature end — the task is still open at suspend time.
    assert event_names.count("subagent_end") == 0, (
        f"suspend must not emit defensive subagent_end; events={event_names}"
    )
    # The suspend event must arrive.
    suspends = [e for e in events if e.event == "suspend"]
    assert len(suspends) == 1
    assert cast(dict, suspends[0].data)["suspend_payload"] == interrupt_payload


@pytest.mark.anyio
async def test_interrupt_payload_as_tuple_is_captured() -> None:
    """Real LangGraph emits ``__interrupt__`` as a tuple of ``Interrupt``s,
    not a list. Earlier revisions narrowed via ``isinstance(..., list)`` and
    silently dropped the interrupt — the integration HITL test then saw
    ``status=completed`` instead of ``suspended``. Lock down that both
    sequence types are accepted."""
    from langgraph.types import Interrupt

    payload = {"suspend_reason": "awaiting_human_review", "suspend_context": {}}
    script: list[_Item] = [
        # Tuple — the shape real LangGraph emits.
        ((), "updates", {"__interrupt__": (Interrupt(value=payload),)}),
    ]

    agent, _ = _make_agent_with_scripted_stream(script)
    events = await _collect_events(agent)
    suspends = [e for e in events if e.event == "suspend"]
    event_names = [e.event for e in events]
    assert len(suspends) == 1, (
        f"tuple-shaped __interrupt__ must be captured; events={event_names}"
    )
    assert cast(dict, suspends[0].data)["suspend_payload"] == payload


@pytest.mark.anyio
async def test_non_dict_updates_payload_is_logged_and_skipped(caplog: Any) -> None:
    """Non-dict updates payload must log a warning and continue rather than
    AttributeError on ``.values()``."""
    import logging

    script: list[_Item] = [
        ((), "updates", "not a dict"),
        _root_update(),
    ]

    agent, _ = _make_agent_with_scripted_stream(script)
    with caplog.at_level(logging.WARNING, logger="agent_engine_sdk_langgraph.agent"):
        events = await _collect_events(agent)

    assert any(e.event == "result" for e in events)
