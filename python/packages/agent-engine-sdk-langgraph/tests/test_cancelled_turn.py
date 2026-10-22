"""Cancelled-turn checkpoint commit: the drain-teardown write that keeps a
stopped turn visible in playground history without making it resumable."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from agent_engine_sdk import AgentInput, RequestContext
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from agent_engine_sdk_langgraph.agent import (
    _TURN_MESSAGE_ID_METADATA_KEY,
    LangGraphBaseAgent,
    _stash_turn_message_id,
)
from agent_engine_sdk_langgraph.cancelled_turn import (
    CANCELLED_TURN_MESSAGE_ID_METADATA_KEY,
    CANCELLED_TURN_MESSAGE_IDS_METADATA_KEY,
    CANCELLED_TURN_METADATA_KEY,
    commit_cancelled_turn,
)
from agent_engine_sdk_langgraph.platform_checkpointer import PlatformCheckpointer


class _RecordingSaver:
    """Duck-typed native saver: serves one canned tip tuple, records puts.

    After a put, the tip IS the last-put checkpoint — a real saver reflects
    its own writes, and the commit hook reads the tip back for the coordinate.
    """

    def __init__(self, latest: Any = None) -> None:
        self._latest = latest
        self.puts: list[tuple[dict, dict, dict, dict]] = []
        self.write_calls: list[tuple] = []

    async def aget_tuple(self, config: dict) -> Any:
        if self.puts:
            last_config, last_checkpoint, last_metadata, _ = self.puts[-1]
            return SimpleNamespace(
                checkpoint=last_checkpoint,
                metadata=last_metadata,
                config=last_config,
            )
        return self._latest

    async def aput(
        self, config: dict, checkpoint: dict, metadata: dict, new_versions: dict
    ) -> dict:
        self.puts.append((config, checkpoint, metadata, new_versions))
        return config

    async def aput_writes(self, *args: Any) -> None:
        self.write_calls.append(args)

    def get_next_version(self, current: Any, channel: Any) -> int:
        return 1 if current is None else current + 1


def _tuple(
    *,
    checkpoint_id: str,
    messages: list[Any],
    channel_versions: dict[str, Any],
    versions_seen: dict[str, dict[str, Any]],
    step: int,
    pending_sends: list[Any] | None = None,
) -> Any:
    return SimpleNamespace(
        checkpoint={
            "v": 1,
            "id": checkpoint_id,
            "ts": "2026-09-28T00:00:00+00:00",
            "channel_values": {"messages": messages},
            "channel_versions": channel_versions,
            "versions_seen": versions_seen,
            "pending_sends": pending_sends if pending_sends is not None else ["send-1"],
        },
        metadata={"source": "loop", "step": step, "parents": {}},
        config={
            "configurable": {
                "thread_id": "t",
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint_id,
            }
        },
    )


def _kwargs(message: Any) -> dict[str, Any]:
    return getattr(message, "additional_kwargs", None) or {}


@pytest.mark.anyio
async def test_identical_consecutive_prompts_cancel_before_input_persistence() -> None:
    # Turn 1 "try again" committed; turn 2 "try again" cancelled before its
    # input checkpoint landed. The captured id is absent from the stored
    # messages, so the commit injects a NEW opener message under that id and
    # leaves the completed first turn unmarked — a content comparison would
    # wrongly pick h1 here.
    messages = [
        HumanMessage(content="try again", id="h1"),
        AIMessage(content="done", id="a1"),
    ]
    latest = _tuple(
        checkpoint_id="c2",
        messages=messages,
        channel_versions={"messages": 1},
        versions_seen={"node_a": {"messages": 1}},
        step=1,
    )
    saver = _RecordingSaver(latest=latest)

    message_id = await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="try again",
        turn_message_id="input:turn-2",
        node_names={"node_a"},
    )

    assert message_id == "input:turn-2"
    _, checkpoint, metadata, _ = saver.puts[0]
    stored = checkpoint["channel_values"]["messages"]
    assert [m.id for m in stored] == ["h1", "a1", "input:turn-2"]
    assert _kwargs(stored[0]) == {}  # completed turn 1 must not be marked
    assert checkpoint["channel_versions"]["messages"] == 2  # bumped for the append
    assert metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY] == "input:turn-2"


@pytest.mark.anyio
async def test_identical_consecutive_prompts_after_input_persistence() -> None:
    # Both turns sent "try again" and turn 2's input DID land. The captured
    # id picks h2 — h1 stays unmarked and no duplicate is appended.
    messages = [
        HumanMessage(content="try again", id="h1"),
        AIMessage(content="ok", id="a1"),
        HumanMessage(content="try again", id="h2"),
    ]
    latest = _tuple(
        checkpoint_id="c9",
        messages=messages,
        channel_versions={"messages": 2},
        versions_seen={"node_a": {"messages": 1}},
        step=4,
    )
    saver = _RecordingSaver(latest=latest)

    message_id = await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="try again",
        turn_message_id="h2",
        node_names={"node_a"},
    )

    assert message_id == "h2"
    config, checkpoint, metadata, _ = saver.puts[0]
    assert config["configurable"]["checkpoint_id"] == "c9"  # parent chain
    assert checkpoint["id"] != "c9"
    stored = checkpoint["channel_values"]["messages"]
    assert [m.id for m in stored] == ["h1", "a1", "h2"]
    # Channel versions unchanged — nothing ran — but every node now reads as
    # caught up, so the tip presents no pending work.
    assert checkpoint["channel_versions"]["messages"] == 2
    assert checkpoint["versions_seen"]["node_a"]["messages"] == 2
    assert checkpoint["pending_sends"] == []
    assert metadata["step"] == 5
    assert metadata["source"] == "loop"
    assert metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY] == "h2"  # h1 unmarked
    # The canned tip's own state must not be mutated through the copy.
    assert latest.checkpoint["versions_seen"]["node_a"]["messages"] == 1


@pytest.mark.anyio
async def test_matches_turn_by_captured_id_not_content() -> None:
    # A prepare_input hook rewrote the message content: the payload text must
    # not be consulted — the captured id alone picks the turn opener.
    messages = [
        HumanMessage(content="original phrasing", id="h1"),
        AIMessage(content="first reply", id="a1"),
        HumanMessage(content="rewritten by the hook", id="h2"),
    ]
    latest = _tuple(
        checkpoint_id="c9",
        messages=messages,
        channel_versions={"messages": 2},
        versions_seen={"node_a": {"messages": 2}},
        step=4,
    )
    saver = _RecordingSaver(latest=latest)

    message_id = await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="original phrasing",
        turn_message_id="h2",
        node_names={"node_a"},
    )

    assert message_id == "h2"
    stored = saver.puts[0][1]["channel_values"]["messages"]
    assert [m.id for m in stored] == ["h1", "a1", "h2"]  # no append
    assert saver.puts[0][2][CANCELLED_TURN_MESSAGE_ID_METADATA_KEY] == "h2"


@pytest.mark.anyio
async def test_no_captured_id_appends_a_fresh_opener_for_an_identical_prompt() -> None:
    # turn_message_id=None means the turn's input was never built (a cancel
    # during preparation). An identical earlier prompt must NOT be treated as
    # this turn's opener: append a fresh message and record that id, leaving
    # the earlier turn untouched.
    messages = [
        HumanMessage(content="stop this", id="h1"),
        AIMessage(content="older reply", id="a1"),
    ]
    latest = _tuple(
        checkpoint_id="c9",
        messages=messages,
        channel_versions={"messages": 2},
        versions_seen={"node_a": {"messages": 2}},
        step=4,
    )
    saver = _RecordingSaver(latest=latest)

    message_id = await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="stop this",
        turn_message_id=None,
        node_names={"node_a"},
    )

    assert message_id is not None and message_id.startswith("msg-cancelled-")
    stored = saver.puts[0][1]["channel_values"]["messages"]
    assert [m.id for m in stored] == ["h1", "a1", message_id]
    assert saver.puts[0][2][CANCELLED_TURN_MESSAGE_ID_METADATA_KEY] == message_id


@pytest.mark.anyio
async def test_no_captured_id_injects_with_a_fresh_id() -> None:
    # No captured id (cancel beat the input checkpoint): inject with a fresh
    # marker id.
    messages = [
        HumanMessage(content="older turn", id="h1"),
        AIMessage(content="older reply", id="a1"),
    ]
    latest = _tuple(
        checkpoint_id="c2",
        messages=messages,
        channel_versions={"messages": 1},
        versions_seen={"node_a": {"messages": 1}},
        step=1,
    )
    saver = _RecordingSaver(latest=latest)

    message_id = await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="never landed",
        turn_message_id=None,
        node_names={"node_a"},
    )

    assert message_id is not None and message_id.startswith("msg-cancelled-")
    _, checkpoint, metadata, _ = saver.puts[0]
    stored = checkpoint["channel_values"]["messages"]
    assert [m.id for m in stored][:2] == ["h1", "a1"]
    assert isinstance(stored[2], HumanMessage)
    assert stored[2].content == "never landed"
    assert stored[2].id == message_id
    assert checkpoint["channel_versions"]["messages"] == 2  # bumped for the append
    assert metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY] == message_id


@pytest.mark.anyio
async def test_commit_does_not_mutate_the_shared_source_message() -> None:
    # The opener is recorded in the checkpoint's metadata, not on the message;
    # the deserialized source message object is shared read state and must
    # stay untouched.
    source_opener = HumanMessage(content="stop this", id="h2")
    latest = _tuple(
        checkpoint_id="c9",
        messages=[
            HumanMessage(content="earlier", id="h1"),
            AIMessage(content="done earlier", id="a1"),
            source_opener,
        ],
        channel_versions={"messages": 2},
        versions_seen={"node_a": {"messages": 2}},
        step=4,
    )
    saver = _RecordingSaver(latest=latest)

    await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="stop this",
        turn_message_id="h2",
        node_names={"node_a"},
    )

    written = saver.puts[0][1]["channel_values"]["messages"][2]
    assert _kwargs(written) == {}
    assert source_opener.additional_kwargs == {}
    assert saver.puts[0][2][CANCELLED_TURN_MESSAGE_ID_METADATA_KEY] == "h2"


@pytest.mark.anyio
async def test_never_run_node_is_terminalized() -> None:
    # node_b never ran in this thread, so it has no versions_seen entry. Left
    # absent it would read as pending; the commit must catch it up over every
    # channel.
    latest = _tuple(
        checkpoint_id="c9",
        messages=[HumanMessage(content="stop this", id="h2")],
        channel_versions={"messages": 2},
        versions_seen={"node_a": {"messages": 1}},
        step=4,
    )
    saver = _RecordingSaver(latest=latest)

    await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="stop this",
        turn_message_id="h2",
        node_names={"node_a", "node_b"},
    )

    written_seen = saver.puts[0][1]["versions_seen"]
    assert written_seen["node_a"]["messages"] == 2
    assert written_seen["node_b"] == {"messages": 2}
    assert latest.checkpoint["versions_seen"] == {"node_a": {"messages": 1}}


@pytest.mark.anyio
async def test_empty_thread_commits_user_message_as_first_checkpoint() -> None:
    saver = _RecordingSaver(latest=None)

    message_id = await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="hi there",
        turn_message_id="input:turn-1",
        node_names={"node_a"},
    )

    assert message_id == "input:turn-1"
    assert len(saver.puts) == 1
    config, checkpoint, metadata, _ = saver.puts[0]
    assert config["configurable"] == {
        "thread_id": "t",
        "checkpoint_ns": "",
        "checkpoint_id": None,
    }
    messages = checkpoint["channel_values"]["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "hi there"
    assert messages[0].id == "input:turn-1"
    assert checkpoint["channel_versions"]["messages"] == 1
    assert checkpoint["versions_seen"] == {"node_a": {"messages": 1}}
    assert checkpoint["pending_sends"] == []
    assert metadata[CANCELLED_TURN_METADATA_KEY] is True
    assert metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY] == "input:turn-1"
    assert metadata["step"] == 0


@pytest.mark.anyio
async def test_empty_message_terminalizes_without_a_turn_opener() -> None:
    # Resume-cancel shape: no fresh user message. The marker tip still lands
    # (terminalized, so the cancelled super-step never looks resumable) but
    # no message is injected and no turn message id is recorded.
    latest = _tuple(
        checkpoint_id="c9",
        messages=[
            HumanMessage(content="original question", id="h1"),
            AIMessage(content="partial", id="a1"),
        ],
        channel_versions={"messages": 1},
        versions_seen={"node_a": {"messages": 0}},
        step=4,
    )
    saver = _RecordingSaver(latest=latest)

    message_id = await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="",
        turn_message_id=None,
        node_names={"node_a", "node_b"},
    )

    assert message_id is None
    _, checkpoint, metadata, _ = saver.puts[0]
    stored = checkpoint["channel_values"]["messages"]
    assert [m.id for m in stored] == ["h1", "a1"]
    assert all(_kwargs(m) == {} for m in stored)
    assert checkpoint["versions_seen"]["node_a"]["messages"] == 1
    assert checkpoint["versions_seen"]["node_b"] == {"messages": 1}
    assert metadata[CANCELLED_TURN_METADATA_KEY] is True
    assert CANCELLED_TURN_MESSAGE_ID_METADATA_KEY not in metadata


@pytest.mark.anyio
async def test_commit_never_attaches_pending_writes() -> None:
    saver = _RecordingSaver(latest=None)
    await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="hi",
        turn_message_id=None,
        node_names=set(),
    )
    assert saver.write_calls == []


@pytest.mark.anyio
async def test_real_two_node_graph_tip_is_terminal_and_never_resumes() -> None:
    """Real-graph proof: cancel with a never-run downstream node, then the
    synthetic tip must report no pending work and never schedule the
    cancelled node on resume; a follow-up turn starts fresh."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph, calls = _two_node_graph(saver)

    config: RunnableConfig = {"configurable": {"thread_id": "t"}}
    await graph.ainvoke(
        {"messages": [HumanMessage(content="go", id="input:1")]}, config
    )
    assert calls == ["a"]  # stopped before node_b ever ran

    resolved = await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="go",
        turn_message_id="input:1",
        node_names=graph.get_graph().nodes.keys(),
    )
    assert resolved == "input:1"

    tip = await saver.aget_tuple(config)
    assert tip is not None
    marker_metadata = cast("dict[str, Any]", tip.metadata)
    assert marker_metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY] == "input:1"

    state = await graph.aget_state(config)
    assert state.next == ()
    assert state.tasks == ()

    # Resume-without-input must not schedule the cancelled node_b.
    await graph.ainvoke(None, config)
    assert calls == ["a"]

    # A follow-up turn starts fresh: node_a runs once for it. (interrupt_before
    # is compile-time, so this invoke also parks before node_b — resuming THAT
    # real pending interrupt is what runs node_b, in contrast to the synthetic
    # tip above, where resume-without-input was a no-op.)
    await graph.ainvoke(
        {"messages": [HumanMessage(content="again", id="input:2")]}, config
    )
    assert calls == ["a", "a"]
    await graph.ainvoke(None, config)
    assert calls == ["a", "a", "b"]
    final_state = await graph.aget_state(config)
    assert final_state.next == ()


@pytest.mark.anyio
async def test_stop_during_a_dispatched_send_tool_never_leaves_pending_work() -> None:
    """Real-graph proof for Send routing (the deep agents dispatch tool calls
    as Send packets): the checkpoint committed while the dispatched tool
    executes carries the queued packet, and the marker must not present it as
    pending — fork preflight accepts the tip and the cancelled payload never
    runs, on the source or on a copy."""
    import asyncio

    from langgraph.checkpoint.memory import InMemorySaver

    from agent_engine_sdk_langgraph.checkpoint_branch import (
        completed_root_checkpoint,
        copy_checkpoint,
    )

    saver = InMemorySaver()
    graph, payloads, release = _send_graph(saver)

    config: RunnableConfig = {"configurable": {"thread_id": "t"}}
    run = asyncio.create_task(
        graph.ainvoke({"messages": [HumanMessage(content="go", id="input:1")]}, config)
    )
    while not payloads:  # dispatched tool executing — the cancel window
        await asyncio.sleep(0)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    # Precondition: the latest committed checkpoint queues the Send.
    latest = await saver.aget_tuple(cast(Any, config))
    assert latest is not None
    queued = (latest.checkpoint.get("channel_values") or {}).get("__pregel_tasks")
    assert queued

    resolved = await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="go",
        turn_message_id="input:1",
        node_names=graph.get_graph().nodes.keys(),
    )
    assert resolved == "input:1"

    # The marker is the source tip and presents no pending work: fork preflight
    # accepts it.
    state = await graph.aget_state(config)
    assert state.next == ()
    assert state.tasks == ()
    marker_source = await completed_root_checkpoint(
        graph, thread_id="t", history_id=None
    )

    # Resume-without-input runs none of the cancelled work (the worker logged
    # its payload once, when the Send dispatched before the cancel)...
    dispatched = len(payloads)
    await graph.ainvoke(None, config)
    assert len(payloads) == dispatched

    # ...and neither does a copy, through a second copy from the first.
    await copy_checkpoint(
        graph, {"configurable": {"thread_id": "t-fork"}}, marker_source
    )
    copied = await graph.aget_state({"configurable": {"thread_id": "t-fork"}})
    assert copied.next == ()
    assert not copied.tasks
    copied_messages = copied.values["messages"]
    assert [m.content for m in copied_messages] == ["go", "dispatch"]
    copied_tip = await saver.aget_tuple({"configurable": {"thread_id": "t-fork"}})
    assert copied_tip is not None
    copied_tip_metadata = cast("dict[str, Any]", copied_tip.metadata)
    assert copied_tip_metadata[CANCELLED_TURN_METADATA_KEY] is True
    # The source opener id rides onto the copy so the reader still classifies
    # the forked opener as stopped.
    assert (
        copied_tip_metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY]
        == copied_messages[0].id
    )

    second_source = await completed_root_checkpoint(
        graph, thread_id="t-fork", history_id=None
    )
    await copy_checkpoint(
        graph, {"configurable": {"thread_id": "t-fork-2"}}, second_source
    )
    copied2 = await graph.aget_state({"configurable": {"thread_id": "t-fork-2"}})
    assert copied2.next == ()
    assert not copied2.tasks
    copied2_messages = copied2.values["messages"]
    assert [m.content for m in copied2_messages] == ["go", "dispatch"]
    copied2_tip = await saver.aget_tuple({"configurable": {"thread_id": "t-fork-2"}})
    assert copied2_tip is not None
    copied2_tip_metadata = cast("dict[str, Any]", copied2_tip.metadata)
    assert copied2_tip_metadata[CANCELLED_TURN_METADATA_KEY] is True
    assert (
        copied2_tip_metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY]
        == copied2_messages[0].id
    )

    await graph.ainvoke(None, {"configurable": {"thread_id": "t-fork"}})
    await graph.ainvoke(None, {"configurable": {"thread_id": "t-fork-2"}})
    assert len(payloads) == dispatched

    # A fresh turn on the copied stopped tip starts fresh, dispatching anew.
    release.set()
    await graph.ainvoke(
        {"messages": [HumanMessage(content="again", id="input:2")]},
        {"configurable": {"thread_id": "t-fork"}},
    )
    assert len(payloads) == dispatched + 1


@pytest.mark.anyio
async def test_second_identical_prompt_cancelled_during_preparation() -> None:
    """Real-saver proof: turn 2 repeats turn 1's prompt and is cancelled during
    preparation (its input was never built, so no id was captured). The commit
    must append a separate stopped opener — the earlier opener and its
    completed steps stay untouched."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph, calls = _two_node_graph(saver)
    config: RunnableConfig = {"configurable": {"thread_id": "t"}}
    await graph.ainvoke(
        {"messages": [HumanMessage(content="go", id="input:1")]}, config
    )
    await graph.ainvoke(None, config)  # past the compile-time interrupt
    assert calls == ["a", "b"]  # turn 1 completed fully

    resolved = await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="go",
        turn_message_id=None,  # prep-phase cancel: input never built
        node_names=graph.get_graph().nodes.keys(),
    )
    assert resolved is not None and resolved.startswith("msg-cancelled-")

    state = await graph.aget_state(config)
    assert state.next == ()
    assert state.tasks == ()
    messages = state.values["messages"]
    assert [m.content for m in messages] == ["go", "a done", "b done", "go"]
    assert messages[0].id == "input:1"  # turn 1's opener stays unmarked
    assert messages[3].id == resolved
    tip = await saver.aget_tuple(config)
    assert tip is not None
    tip_metadata = cast("dict[str, Any]", tip.metadata)
    assert tip_metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY] == resolved


def _two_node_graph(saver: Any) -> Any:
    """A → B with a compile-time interrupt before B, and the call log."""
    from langgraph.graph import END, START, MessagesState, StateGraph

    calls: list[str] = []

    def node_a(state: MessagesState) -> dict[str, Any]:
        calls.append("a")
        return {"messages": [AIMessage(content="a done", id=f"a-{len(calls)}")]}

    def node_b(state: MessagesState) -> dict[str, Any]:
        calls.append("b")
        return {"messages": [AIMessage(content="b done", id="b-1")]}

    builder = StateGraph(MessagesState)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)
    return builder.compile(checkpointer=saver, interrupt_before=["node_b"]), calls


def _send_graph(saver: Any) -> Any:
    """Router dispatching its tool call as a Send — the deep agents' routing."""
    import asyncio

    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.types import Send

    payloads: list[Any] = []
    release = asyncio.Event()

    async def worker(state: Any) -> dict[str, Any]:
        payloads.append(state)
        await release.wait()
        return {"messages": [AIMessage(content="tool done", id="tool-1")]}

    builder = StateGraph(MessagesState)
    builder.add_node(
        "router", lambda state: {"messages": [AIMessage(content="dispatch")]}
    )
    builder.add_node("worker", worker)
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router", lambda state: [Send("worker", "tool payload")], ["worker"]
    )
    builder.add_edge("worker", END)
    return builder.compile(checkpointer=saver), payloads, release


@pytest.mark.anyio
async def test_fork_reproduces_a_stopped_tip_and_later_history() -> None:
    """Forking from a cancelled-turn marker tip — and from later history built
    on one — reproduces the authoritative state, marker tip included."""
    from langgraph.checkpoint.memory import InMemorySaver

    from agent_engine_sdk_langgraph.checkpoint_branch import (
        completed_root_checkpoint,
        copy_checkpoint,
    )

    saver = InMemorySaver()
    graph, _ = _two_node_graph(saver)
    config: RunnableConfig = {"configurable": {"thread_id": "t"}}
    await graph.ainvoke(
        {"messages": [HumanMessage(content="go", id="input:1")]}, config
    )

    await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="go",
        turn_message_id="input:1",
        node_names=graph.get_graph().nodes.keys(),
    )

    # Fork directly from the marker tip.
    source = await completed_root_checkpoint(graph, thread_id="t", history_id=None)
    await copy_checkpoint(graph, {"configurable": {"thread_id": "t-fork"}}, source)
    copied = await graph.aget_state({"configurable": {"thread_id": "t-fork"}})
    assert copied.next == ()
    assert not copied.tasks
    source_messages = (await graph.aget_state(config)).values["messages"]
    copied_messages = copied.values["messages"]
    assert [m.content for m in copied_messages] == [m.content for m in source_messages]
    copied_tip = await saver.aget_tuple({"configurable": {"thread_id": "t-fork"}})
    assert copied_tip is not None
    copied_tip_metadata = cast("dict[str, Any]", copied_tip.metadata)
    assert copied_tip_metadata[CANCELLED_TURN_METADATA_KEY] is True
    # The source opener id rides onto the copy so the reader still classifies
    # the forked opener as stopped.
    assert (
        copied_tip_metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY]
        == copied_messages[0].id
    )

    # A follow-up turn on the source, then fork again: the full later history
    # survives.
    await graph.ainvoke(
        {"messages": [HumanMessage(content="again", id="input:2")]}, config
    )
    await graph.ainvoke(None, config)  # past the compile-time interrupt
    source = await completed_root_checkpoint(graph, thread_id="t", history_id=None)
    await copy_checkpoint(graph, {"configurable": {"thread_id": "t-fork-2"}}, source)
    copied2 = await graph.aget_state({"configurable": {"thread_id": "t-fork-2"}})
    copied2_messages = copied2.values["messages"]
    assert [m.content for m in copied2_messages] == [
        "go",
        "a done",
        "again",
        "a done",
        "b done",
    ]
    # Replay rebuilds checkpoints and drops the per-turn marker metadata, so the
    # fork re-records the source's cancelled openers on the copied tip — the
    # earlier stopped opener keeps its id and stays classified on the fork.
    copied2_tip = await saver.aget_tuple({"configurable": {"thread_id": "t-fork-2"}})
    assert copied2_tip is not None
    copied2_tip_metadata = cast("dict[str, Any]", copied2_tip.metadata)
    assert copied2_tip_metadata[CANCELLED_TURN_METADATA_KEY] is True
    assert copied2_messages[0].id == "input:1"
    assert copied2_tip_metadata[CANCELLED_TURN_MESSAGE_IDS_METADATA_KEY] == ["input:1"]


@pytest.mark.anyio
async def test_fork_from_a_first_turn_cancelled_before_any_checkpoint() -> None:
    """The marker tip is the thread's ONLY checkpoint — plain replay reports
    "no checkpoint history"; the synthesized superstep makes it forkable."""
    from langgraph.checkpoint.memory import InMemorySaver

    from agent_engine_sdk_langgraph.checkpoint_branch import (
        completed_root_checkpoint,
        copy_checkpoint,
    )

    saver = InMemorySaver()
    graph, _ = _two_node_graph(saver)

    await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="never ran",
        turn_message_id="input:1",
        node_names=graph.get_graph().nodes.keys(),
    )

    source = await completed_root_checkpoint(graph, thread_id="t", history_id=None)
    await copy_checkpoint(graph, {"configurable": {"thread_id": "t-fork"}}, source)
    copied = await graph.aget_state({"configurable": {"thread_id": "t-fork"}})
    copied_messages = copied.values["messages"]
    assert [m.content for m in copied_messages] == ["never ran"]
    copied_tip = await saver.aget_tuple({"configurable": {"thread_id": "t-fork"}})
    assert copied_tip is not None
    copied_tip_metadata = cast("dict[str, Any]", copied_tip.metadata)
    assert copied_tip_metadata[CANCELLED_TURN_METADATA_KEY] is True
    # The source opener id rides onto the copy so the reader still classifies
    # the forked opener as stopped.
    assert (
        copied_tip_metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY]
        == copied_messages[0].id
    )


@pytest.mark.anyio
async def test_a_forked_stopped_tip_is_itself_forkable() -> None:
    """Two consecutive forks of an early-cancelled first turn: the first copy's
    terminalized tip carries the marker, so the second fork replays it as a
    marker too — stopped opener preserved, nothing left pending."""
    from langgraph.checkpoint.memory import InMemorySaver

    from agent_engine_sdk_langgraph.checkpoint_branch import (
        completed_root_checkpoint,
        copy_checkpoint,
    )

    saver = InMemorySaver()
    graph, _ = _two_node_graph(saver)

    await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="never ran",
        turn_message_id="input:1",
        node_names=graph.get_graph().nodes.keys(),
    )

    source = await completed_root_checkpoint(graph, thread_id="t", history_id=None)
    await copy_checkpoint(graph, {"configurable": {"thread_id": "t-fork"}}, source)
    tip = await saver.aget_tuple({"configurable": {"thread_id": "t-fork"}})
    assert tip is not None
    tip_metadata = cast("dict[str, Any]", tip.metadata)
    assert tip_metadata[CANCELLED_TURN_METADATA_KEY] is True
    assert tip_metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY] == "input:1"

    second = await completed_root_checkpoint(graph, thread_id="t-fork", history_id=None)
    await copy_checkpoint(graph, {"configurable": {"thread_id": "t-fork-2"}}, second)
    copied = await graph.aget_state({"configurable": {"thread_id": "t-fork-2"}})
    assert copied.next == ()
    assert not copied.tasks
    copied_messages = copied.values["messages"]
    assert [m.content for m in copied_messages] == ["never ran"]
    copied_tip = await saver.aget_tuple({"configurable": {"thread_id": "t-fork-2"}})
    assert copied_tip is not None
    copied_tip_metadata = cast("dict[str, Any]", copied_tip.metadata)
    assert copied_tip_metadata[CANCELLED_TURN_METADATA_KEY] is True
    # The source opener id rides onto the copy so the reader still classifies
    # the forked opener as stopped.
    assert (
        copied_tip_metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY]
        == copied_messages[0].id
    )


def test_stash_records_last_human_message_id() -> None:
    ctx = RequestContext(session_id="s", workspace_id="w", metadata={})
    _stash_turn_message_id(
        ctx,
        {
            "messages": [
                HumanMessage(content="x", id="h1"),
                AIMessage(content="y", id="a1"),
            ]
        },
    )
    stashed = ctx.metadata or {}
    assert stashed[_TURN_MESSAGE_ID_METADATA_KEY] == "h1"


def test_stash_assigns_an_id_to_an_id_less_hook_built_opener() -> None:
    # A prepare_input hook is not required to set ids: the opener gets one at
    # build time (LangGraph would only mint it inside the run, too late to
    # capture), and that assigned id is what the teardown commit looks up.
    ctx = RequestContext(session_id="s", workspace_id="w", metadata={})
    opener = HumanMessage(content="rewritten by the hook")
    graph_input: dict[str, Any] = {"messages": [opener]}

    _stash_turn_message_id(ctx, graph_input)

    assert opener.id is not None and opener.id.startswith("input:")
    stashed = ctx.metadata or {}
    assert stashed[_TURN_MESSAGE_ID_METADATA_KEY] == opener.id


def test_stash_ignores_resume_shapes_and_non_list_messages() -> None:
    ctx = RequestContext(session_id="s", workspace_id="w", metadata={})
    _stash_turn_message_id(ctx, Command(resume="yes"))  # not a dict: no-op
    _stash_turn_message_id(ctx, {"messages": "nope"})  # not a list: no-op
    assert _TURN_MESSAGE_ID_METADATA_KEY not in (ctx.metadata or {})


@pytest.mark.anyio
async def test_resume_commit_recovers_the_persisted_opener_from_the_loaded_checkpoint() -> (
    None
):
    # A resume captures no id: the commit recovers the opener from the
    # checkpoint it already loaded and records it — never appends a duplicate.
    opener = HumanMessage(content="the prompt", id="input:orig-1")
    saver = _RecordingSaver(
        latest=_tuple(
            checkpoint_id="cp1",
            messages=[opener, AIMessage(content="working", id="a1")],
            channel_versions={"messages": 1},
            versions_seen={"node_a": {"messages": 1}},
            step=1,
        )
    )

    resolved = await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="the prompt",
        turn_message_id=None,
        node_names={"node_a"},
        resume=True,
    )

    assert resolved == "input:orig-1"
    written = saver.puts[0][1]["channel_values"]["messages"]
    assert [message.id for message in written] == ["input:orig-1", "a1"]
    assert saver.puts[0][2][CANCELLED_TURN_MESSAGE_ID_METADATA_KEY] == "input:orig-1"


@pytest.mark.anyio
async def test_native_resume_cancel_records_the_persisted_opener() -> None:
    """Real-graph: a turn parks at a HITL interrupt, resumes, then is cancelled.
    The commit recovers the opener the first execution persisted — exactly one
    opener recorded, no duplicate prompt."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph, calls = _two_node_graph(saver)
    config: RunnableConfig = {"configurable": {"thread_id": "t"}}
    await graph.ainvoke(
        {"messages": [HumanMessage(content="the prompt", id="input:orig-1")]}, config
    )
    assert calls == ["a"]  # parked before node_b

    await graph.ainvoke(None, config)  # resume through the interrupt
    assert calls == ["a", "b"]

    resolved = await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="the prompt",
        turn_message_id=None,  # a resume captures no id
        node_names=graph.get_graph().nodes.keys(),
        resume=True,
    )
    assert resolved == "input:orig-1"

    state = await graph.aget_state(config)
    humans = [m for m in state.values["messages"] if m.type == "human"]
    assert [m.content for m in humans] == ["the prompt"]
    tip = await saver.aget_tuple(config)
    assert tip is not None
    tip_metadata = cast("dict[str, Any]", tip.metadata)
    assert tip_metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY] == "input:orig-1"


@pytest.mark.anyio
async def test_native_resume_cancel_during_preparation_records_the_persisted_opener() -> (
    None
):
    """A Stop that lands before the resumed step runs still recovers the
    persisted opener — the commit reads it from the checkpoint it already
    loaded, so no interruptible lookup can miss it."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph, calls = _two_node_graph(saver)
    config: RunnableConfig = {"configurable": {"thread_id": "t"}}
    await graph.ainvoke(
        {"messages": [HumanMessage(content="the prompt", id="input:orig-1")]}, config
    )
    assert calls == ["a"]

    # No resume step runs: the cancel beats it.
    resolved = await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="the prompt",
        turn_message_id=None,
        node_names=graph.get_graph().nodes.keys(),
        resume=True,
    )
    assert resolved == "input:orig-1"

    state = await graph.aget_state(config)
    humans = [m for m in state.values["messages"] if m.type == "human"]
    assert [m.content for m in humans] == ["the prompt"]
    tip = await saver.aget_tuple(config)
    assert tip is not None
    tip_metadata = cast("dict[str, Any]", tip.metadata)
    assert tip_metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY] == "input:orig-1"


@pytest.mark.anyio
async def test_native_resume_cancel_while_the_continuation_is_gated() -> None:
    """Real HITL: the turn suspends, resumes through the maintained
    ``Command(resume=...)`` contract, commits an intermediate step, and is
    cancelled while the next resumed step is still running. The commit recovers
    the persisted opener and retains the completed step, in order."""
    import asyncio

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.types import interrupt

    started = asyncio.Event()
    release = asyncio.Event()

    def gate(state: Any) -> dict[str, Any]:
        interrupt("question")
        return {}

    def step_one(state: Any) -> dict[str, Any]:
        return {"messages": [AIMessage(content="step one", id="a-1")]}

    async def gated(state: Any) -> dict[str, Any]:
        started.set()
        await release.wait()
        return {"messages": [AIMessage(content="step two", id="a-2")]}

    builder = StateGraph(MessagesState)
    builder.add_node("gate", gate)
    builder.add_node("step_one", step_one)
    builder.add_node("gated", gated)
    builder.add_edge(START, "gate")
    builder.add_edge("gate", "step_one")
    builder.add_edge("step_one", "gated")
    builder.add_edge("gated", END)
    saver = InMemorySaver()
    graph = builder.compile(checkpointer=saver)
    config: RunnableConfig = {"configurable": {"thread_id": "t"}}

    # Turn 1 suspends at the real HITL interrupt.
    await graph.ainvoke(
        {"messages": [HumanMessage(content="the prompt", id="input:orig-1")]}, config
    )

    # Resume through the maintained continuation contract; the gated step blocks.
    run = asyncio.create_task(graph.ainvoke(Command(resume="approved"), config))
    await started.wait()

    # Stop lands while the continuation runs.
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    resolved = await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="the prompt",
        turn_message_id=None,
        node_names=graph.get_graph().nodes.keys(),
        resume=True,
    )
    assert resolved == "input:orig-1"

    state = await graph.aget_state(config)
    messages = state.values["messages"]
    assert [m.content for m in messages] == ["the prompt", "step one"]
    humans = [m for m in messages if m.type == "human"]
    assert len(humans) == 1
    assert humans[0].id == "input:orig-1"
    tip = await saver.aget_tuple(config)
    assert tip is not None
    tip_metadata = cast("dict[str, Any]", tip.metadata)
    assert tip_metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY] == "input:orig-1"

    # A fresh turn preserves the stopped opener and the completed step.
    release.set()
    await graph.ainvoke(
        {"messages": [HumanMessage(content="next", id="input:2")]}, config
    )
    state = await graph.aget_state(config)
    messages = state.values["messages"]
    assert messages[0].id == "input:orig-1"
    assert [m.content for m in messages if m.type == "human"] == ["the prompt", "next"]


@pytest.mark.anyio
async def test_resume_cancel_opener_survives_a_later_fresh_turn() -> None:
    """After a resume cancel, a fresh turn appends its own opener; the stopped
    one keeps its id."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph, _ = _two_node_graph(saver)
    config: RunnableConfig = {"configurable": {"thread_id": "t"}}
    await graph.ainvoke(
        {"messages": [HumanMessage(content="the prompt", id="input:orig-1")]}, config
    )
    await graph.ainvoke(None, config)
    await commit_cancelled_turn(
        saver,
        thread_id="t",
        user_message="the prompt",
        turn_message_id=None,
        node_names=graph.get_graph().nodes.keys(),
        resume=True,
    )

    await graph.ainvoke(
        {"messages": [HumanMessage(content="next", id="input:2")]}, config
    )
    await graph.ainvoke(None, config)

    state = await graph.aget_state(config)
    messages = state.values["messages"]
    assert messages[0].id == "input:orig-1"
    humans = [m for m in messages if m.type == "human"]
    assert [m.content for m in humans] == ["the prompt", "next"]


class TestAgentHook:
    """LangGraphBaseAgent.commit_cancelled_turn — the duck-typed hook
    runner-shared calls from the drain teardown path."""

    def _agent(self, saver: Any, prepare_input: Any = None) -> LangGraphBaseAgent:
        graph = MagicMock()
        graph.checkpointer = (
            PlatformCheckpointer(native=cast(Any, saver)) if saver is not None else None
        )
        graph.get_graph.return_value.nodes = {"node_a": object()}
        return LangGraphBaseAgent(graph, prepare_input=prepare_input)

    @pytest.mark.anyio
    async def test_commits_via_the_native_saver(self) -> None:
        saver = _RecordingSaver(latest=None)
        agent = self._agent(saver)
        ctx = RequestContext(session_id="sess-1", workspace_id="ws-1")

        committed = await agent.commit_cancelled_turn(
            ctx, AgentInput(payload={"message": "stop it"})
        )

        assert committed is not None
        assert len(saver.puts) == 1
        config = saver.puts[0][0]
        assert config["configurable"]["thread_id"] == "sess-1:ws-1"
        # No stash on ctx (input never built): the commit appends a fresh
        # opener under a minted id.
        message = saver.puts[0][1]["channel_values"]["messages"][0]
        assert message.id.startswith("msg-cancelled-")
        assert saver.puts[0][2][CANCELLED_TURN_MESSAGE_ID_METADATA_KEY] == message.id

    @pytest.mark.anyio
    async def test_hook_consumes_the_stashed_turn_message_id(self) -> None:
        saver = _RecordingSaver(latest=None)
        agent = self._agent(saver)
        ctx = RequestContext(
            session_id="sess-1",
            workspace_id="ws-1",
            metadata={_TURN_MESSAGE_ID_METADATA_KEY: "input:captured-1"},
        )

        committed = await agent.commit_cancelled_turn(
            ctx, AgentInput(payload={"message": "stop it"})
        )

        assert committed is not None
        message = saver.puts[0][1]["channel_values"]["messages"][0]
        assert message.id == "input:captured-1"
        # The stash is single-use: consumed by the hook.
        assert _TURN_MESSAGE_ID_METADATA_KEY not in (ctx.metadata or {})

    @pytest.mark.anyio
    async def test_resume_flag_recovers_the_persisted_opener(self) -> None:
        # The hook forwards ctx.resume so the commit recovers the opener a
        # native HITL resume already persisted, instead of appending a copy.
        opener = HumanMessage(content="the prompt", id="input:orig-1")
        saver = _RecordingSaver(
            latest=_tuple(
                checkpoint_id="cp1",
                messages=[opener],
                channel_versions={"messages": 1},
                versions_seen={"node_a": {"messages": 1}},
                step=1,
            )
        )
        agent = self._agent(saver)
        ctx = RequestContext(
            session_id="sess-1", workspace_id="ws-1", resume=True, resume_data={}
        )

        await agent.commit_cancelled_turn(
            ctx, AgentInput(payload={"message": "the prompt"})
        )

        written = saver.puts[0][1]["channel_values"]["messages"]
        assert [message.id for message in written] == ["input:orig-1"]
        assert saver.puts[0][2][CANCELLED_TURN_MESSAGE_ID_METADATA_KEY] == (
            "input:orig-1"
        )

    @pytest.mark.anyio
    async def test_stateless_graph_is_a_noop(self) -> None:
        agent = self._agent(None)
        ctx = RequestContext(session_id="sess-1", workspace_id="ws-1")
        assert (
            await agent.commit_cancelled_turn(ctx, AgentInput(payload={"message": "m"}))
            is None
        )

    @pytest.mark.anyio
    async def test_durable_session_is_a_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        saver = _RecordingSaver(latest=None)
        agent = self._agent(saver)
        monkeypatch.setattr(
            LangGraphBaseAgent,
            "_execution_session",
            lambda self: SimpleNamespace(is_durable=True),
        )
        ctx = RequestContext(session_id="sess-1", workspace_id="ws-1")
        assert (
            await agent.commit_cancelled_turn(ctx, AgentInput(payload={"message": "m"}))
            is None
        )
        assert saver.puts == []

    @pytest.mark.anyio
    async def test_empty_message_is_a_noop(self) -> None:
        saver = _RecordingSaver(latest=None)
        agent = self._agent(saver)
        ctx = RequestContext(session_id="sess-1", workspace_id="ws-1")
        assert (
            await agent.commit_cancelled_turn(ctx, AgentInput(payload={"message": ""}))
            is None
        )
        assert saver.puts == []

    @pytest.mark.anyio
    async def test_prepare_input_rewrite_keeps_one_recorded_opener(self) -> None:
        # A prepare_input hook rewrites content and drops ids. The run assigns
        # and captures the opener's id at build time, so the cancel commit
        # records the hook's own persisted message — never a duplicate
        # raw-prompt append.
        def prepare(input: AgentInput, ctx: RequestContext) -> Any:
            payload = input.payload if isinstance(input.payload, dict) else {}
            return {
                "messages": [
                    HumanMessage(content=f"wrapped: {payload.get('message', '')}")
                ]
            }

        saver = _RecordingSaver(latest=None)
        agent = self._agent(saver, prepare_input=prepare)
        ctx = RequestContext(session_id="sess-1", workspace_id="ws-1", metadata={})
        agent_input = AgentInput(payload={"message": "original"})

        session = agent._execution_session()
        graph_input, _ = await agent._prepare_run(ctx, agent_input, session)
        opener = graph_input["messages"][0]
        assert opener.content == "wrapped: original"
        assert opener.id.startswith("input:")

        # The input checkpoint then lands with the assigned id (the real
        # graph run persists it); the cancel commit must find it by identity.
        saver._latest = _tuple(
            checkpoint_id="c1",
            messages=[opener],
            channel_versions={"messages": 1},
            versions_seen={"node_a": {"messages": 1}},
            step=0,
        )

        committed = await agent.commit_cancelled_turn(ctx, agent_input)

        assert committed is not None
        stored = saver.puts[0][1]["channel_values"]["messages"]
        assert len(stored) == 1  # no duplicate raw-prompt message appended
        assert stored[0].id == opener.id
        assert stored[0].content == "wrapped: original"
        assert saver.puts[0][2][CANCELLED_TURN_MESSAGE_ID_METADATA_KEY] == opener.id


class TestHookMessageLikeOpeners:
    """Hooks may return supported message-likes (tuples, role dicts) instead of
    BaseMessages: the input-build boundary converts them, so the id the graph
    persists is the id the teardown commit captured — one opener recorded
    before the completed step, never a duplicate raw-prompt append."""

    @staticmethod
    def _agent(saver: Any, prepare_input: Any = None) -> LangGraphBaseAgent:
        from langgraph.graph import END, START, MessagesState, StateGraph

        def reply(state: MessagesState) -> dict[str, Any]:
            return {"messages": [AIMessage(content="reply done", id="a-1")]}

        builder = StateGraph(MessagesState)
        builder.add_node("reply", reply)
        builder.add_edge(START, "reply")
        builder.add_edge("reply", END)
        return LangGraphBaseAgent(
            builder.compile(checkpointer=saver), prepare_input=prepare_input
        )

    @staticmethod
    async def _tip_messages(saver: Any) -> list[Any]:
        tip = await saver.aget_tuple({"configurable": {"thread_id": "sess-1:ws-1"}})
        return list(tip.checkpoint["channel_values"]["messages"])

    @staticmethod
    async def _tip_metadata(saver: Any) -> dict[str, Any]:
        tip = await saver.aget_tuple({"configurable": {"thread_id": "sess-1:ws-1"}})
        return cast("dict[str, Any]", tip.metadata)

    @pytest.mark.anyio
    async def test_tuple_opener_is_captured_and_recorded(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        saver = InMemorySaver()
        agent = self._agent(
            saver,
            prepare_input=lambda input, ctx: {"messages": [("human", "stop this")]},
        )
        ctx = RequestContext(session_id="sess-1", workspace_id="ws-1", metadata={})
        agent_input = AgentInput(payload={"message": "stop this"})

        await agent.invoke(ctx, agent_input)
        assert await agent.commit_cancelled_turn(ctx, agent_input) is not None

        messages = await self._tip_messages(saver)
        assert [m.content for m in messages] == ["stop this", "reply done"]
        assert messages[0].id.startswith("input:")
        assert (await self._tip_metadata(saver))[
            CANCELLED_TURN_MESSAGE_ID_METADATA_KEY
        ] == messages[0].id

    @pytest.mark.anyio
    async def test_role_dict_opener_keeps_a_given_id(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        saver = InMemorySaver()
        agent = self._agent(
            saver,
            prepare_input=lambda input, ctx: {
                "messages": [{"role": "user", "content": "stop this", "id": "hook-1"}]
            },
        )
        ctx = RequestContext(session_id="sess-1", workspace_id="ws-1", metadata={})
        agent_input = AgentInput(payload={"message": "stop this"})

        await agent.invoke(ctx, agent_input)
        assert await agent.commit_cancelled_turn(ctx, agent_input) is not None

        messages = await self._tip_messages(saver)
        assert [m.content for m in messages] == ["stop this", "reply done"]
        assert messages[0].id == "hook-1"
        assert (await self._tip_metadata(saver))[
            CANCELLED_TURN_MESSAGE_ID_METADATA_KEY
        ] == "hook-1"

    @pytest.mark.anyio
    async def test_role_dict_opener_without_id_gets_the_captured_id(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        saver = InMemorySaver()
        agent = self._agent(
            saver,
            prepare_input=lambda input, ctx: {
                "messages": [{"role": "user", "content": "stop this"}]
            },
        )
        ctx = RequestContext(session_id="sess-1", workspace_id="ws-1", metadata={})
        agent_input = AgentInput(payload={"message": "stop this"})

        await agent.invoke(ctx, agent_input)
        assert await agent.commit_cancelled_turn(ctx, agent_input) is not None

        messages = await self._tip_messages(saver)
        assert len(messages) == 2  # no duplicate opener
        assert messages[0].id.startswith("input:")
        assert (await self._tip_metadata(saver))[
            CANCELLED_TURN_MESSAGE_ID_METADATA_KEY
        ] == messages[0].id

    @pytest.mark.anyio
    async def test_overwrite_opener_replaces_history_and_is_recorded(self) -> None:
        """Overwrite survives the build-time conversion: the hook's list
        replaces prior history wholesale, and the new opener is still the one
        the cancel commit records."""
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import Overwrite

        saver = InMemorySaver()
        ctx = RequestContext(session_id="sess-1", workspace_id="ws-1", metadata={})
        await self._agent(saver).invoke(ctx, AgentInput(payload={"message": "first"}))

        agent = self._agent(
            saver,
            prepare_input=lambda input, ctx: {
                "messages": Overwrite([HumanMessage(content="fresh slate")])
            },
        )
        hooked_input = AgentInput(payload={"message": "ignored by the hook"})
        await agent.invoke(ctx, hooked_input)
        assert await agent.commit_cancelled_turn(ctx, hooked_input) is not None

        messages = await self._tip_messages(saver)
        assert [m.content for m in messages] == ["fresh slate", "reply done"]
        assert messages[0].id.startswith("input:")
        assert (await self._tip_metadata(saver))[
            CANCELLED_TURN_MESSAGE_ID_METADATA_KEY
        ] == messages[0].id

    @pytest.mark.anyio
    async def test_serialized_overwrite_opener_is_accepted_and_recorded(self) -> None:
        """A hook may return the serialized __overwrite__ wire shape: same
        replacement semantics through the build-boundary conversion — prior
        history replaced, one recorded opener, no rejection, no duplicate."""
        from langgraph.checkpoint.memory import InMemorySaver

        saver = InMemorySaver()
        ctx = RequestContext(session_id="sess-1", workspace_id="ws-1", metadata={})
        await self._agent(saver).invoke(ctx, AgentInput(payload={"message": "first"}))

        agent = self._agent(
            saver,
            prepare_input=lambda input, ctx: {
                "messages": {"__overwrite__": [("human", "fresh slate")]}
            },
        )
        hooked_input = AgentInput(payload={"message": "ignored by the hook"})
        await agent.invoke(ctx, hooked_input)
        assert await agent.commit_cancelled_turn(ctx, hooked_input) is not None

        messages = await self._tip_messages(saver)
        assert [m.content for m in messages] == ["fresh slate", "reply done"]
        assert messages[0].id.startswith("input:")
        assert (await self._tip_metadata(saver))[
            CANCELLED_TURN_MESSAGE_ID_METADATA_KEY
        ] == messages[0].id
