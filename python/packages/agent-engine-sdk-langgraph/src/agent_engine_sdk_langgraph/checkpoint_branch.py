"""LangGraph checkpoint coordinates, state inspection, and branch copies.

Durable OE attempts inspect pending checkpoint tasks but do not use checkpoint
history as their replay source.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterable
from typing import Any, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, empty_checkpoint
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Interrupt, Overwrite, StateSnapshot, StateUpdate
from pydantic import BaseModel, ConfigDict, Field

from agent_engine_sdk_langgraph.cancelled_turn import (
    CANCELLED_TURN_MESSAGE_ID_METADATA_KEY,
    CANCELLED_TURN_MESSAGE_IDS_METADATA_KEY,
    CANCELLED_TURN_METADATA_KEY,
    cancelled_opener_ids_from_metadata,
    clear_queued_tasks,
    terminalize_versions_seen,
)
from agent_engine_sdk_langgraph.suspend import canonical_interrupt_value

_logger = logging.getLogger(__name__)


def pending_interrupts_from_state(state: Any) -> list[Interrupt]:
    """Read pending interrupts from a root snapshot and its subgraphs."""
    pending: list[Interrupt] = []
    values_by_id: dict[str, str] = {}

    def visit(snapshot: Any) -> None:
        for task in getattr(snapshot, "tasks", ()):
            task_state = getattr(task, "state", None)
            if getattr(task_state, "tasks", None) is not None:
                # Parent snapshots can repeat a subgraph interrupt, so visit
                # children first and keep each native id only once.
                visit(task_state)
            for interrupt in getattr(task, "interrupts", ()):
                serialized_value = canonical_interrupt_value(
                    interrupt.id, interrupt.value
                )
                previous_value = values_by_id.get(interrupt.id)
                if previous_value is None:
                    values_by_id[interrupt.id] = serialized_value
                    pending.append(interrupt)
                elif previous_value != serialized_value:
                    raise RuntimeError(
                        "LangGraph returned conflicting values for interrupt "
                        f'"{interrupt.id}"'
                    )

    visit(state)
    return pending


class LangGraphCheckpoint(BaseModel):
    """Exact checkpoint coordinate understood only by this adapter."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)


class _LangGraphMetadata(BaseModel):
    """LangGraph-owned values carried through framework-neutral metadata."""

    model_config = ConfigDict(extra="ignore")

    langgraph_branch_point: LangGraphCheckpoint | None = None
    langgraph_checkpoint: LangGraphCheckpoint | None = None


def branch_point_from_metadata(
    metadata: dict[str, Any] | None,
) -> LangGraphCheckpoint | None:
    if metadata is None:
        return None
    return _LangGraphMetadata.model_validate(metadata).langgraph_branch_point


def checkpoint_metadata(
    checkpoint: LangGraphCheckpoint | None,
) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    return {"langgraph_checkpoint": checkpoint.model_dump()}


def normalize_checkpoint_id(value: Any) -> str | None:
    """Normalize checkpoint ids from payload values.

    Fresh executions often carry ``checkpoint_id=None`` through the payload.
    Treat that as absent instead of the literal string ``"None"``, otherwise
    we suppress latest-checkpoint lookup and start a new checkpoint root.
    """
    if value is None:
        return None

    checkpoint_id = str(value).strip()
    if not checkpoint_id or checkpoint_id.lower() == "none":
        return None

    return checkpoint_id


async def latest_checkpoint_id(graph: Any, thread_id: str) -> str | None:
    """Best-effort latest native checkpoint id for ``thread_id``.

    Lookup failure must not turn a successful or suspended execution into an
    error. Without the coordinate, a later request to branch from that
    execution fails explicitly instead.
    """
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None or not hasattr(checkpointer, "aget_tuple"):
        return None

    try:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        if checkpoint_tuple and hasattr(checkpoint_tuple, "checkpoint"):
            checkpoint = checkpoint_tuple.checkpoint
            if isinstance(checkpoint, dict):
                return checkpoint.get("id")
    except Exception:
        _logger.exception(
            "Failed to get checkpoint ID for thread %s — degrading to None "
            "without changing the agent execution outcome",
            thread_id,
        )
    return None


async def langgraph_checkpoint_for_thread(
    graph: Any, thread_id: str
) -> LangGraphCheckpoint | None:
    checkpoint_id = await latest_checkpoint_id(graph, thread_id)
    if checkpoint_id is None:
        return None
    return LangGraphCheckpoint(thread_id=thread_id, checkpoint_id=checkpoint_id)


def _root_graph_checkpoint(config: RunnableConfig) -> LangGraphCheckpoint:
    """Read a root-graph coordinate, or fail closed on a subgraph namespace.

    Branch copy replays root supersteps onto an empty thread. A non-empty
    ``checkpoint_ns`` is a nested graph; those checkpoints are not a valid
    branch point and must not be treated as one.
    """
    configurable = config.get("configurable") or {}
    if configurable.get("checkpoint_ns", ""):
        raise RuntimeError("LangGraph checkpoint branching supports root graphs only")
    return LangGraphCheckpoint(
        thread_id=cast(str, configurable["thread_id"]),
        checkpoint_id=cast(str, configurable["checkpoint_id"]),
    )


async def _checkpoint_history(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    config: RunnableConfig,
) -> AsyncIterator[StateSnapshot]:
    while True:
        snapshot = await graph.aget_state(config)
        _root_graph_checkpoint(snapshot.config)
        yield snapshot
        if snapshot.parent_config is None:
            return
        config = snapshot.parent_config


def _state_updates(snapshot: StateSnapshot) -> list[StateUpdate]:
    updates: list[StateUpdate] = []
    for task in snapshot.tasks:
        if task.error is not None:
            raise RuntimeError(
                "LangGraph checkpoint branching does not support failed history"
            )
        if task.state is not None:
            raise RuntimeError(
                "LangGraph checkpoint branching does not support persistent "
                "subgraph history"
            )
        if task.interrupts and task.result is None:
            continue
        if task.result is not None and not isinstance(task.result, dict):
            raise RuntimeError("LangGraph checkpoint task result is not a state update")
        updates.append(StateUpdate(task.result, task.name, task.id))
    return updates


def _state_without_message_ids(value: Any) -> Any:
    """Graph values comparable after a branch copy.

    ``abulk_update_state`` rebuilds messages and assigns new LangChain ids.
    Equality on raw ``values`` would fail on those ids even when the copied
    conversation matches the source, so ids are stripped before compare.
    """
    if isinstance(value, BaseMessage):
        return _state_without_message_ids(value.model_dump(exclude={"id"}))
    if isinstance(value, dict):
        return {
            key: _state_without_message_ids(nested) for key, nested in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_state_without_message_ids(nested) for nested in value]
    return value


async def completed_root_checkpoint(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    *,
    thread_id: str,
    history_id: str | None,
) -> LangGraphCheckpoint:
    """Load a completed root checkpoint.

    Fork validates this before CreateBranch so a missing or incomplete
    history point does not create a branch session.
    """
    configurable: dict[str, Any] = {"thread_id": thread_id, "checkpoint_ns": ""}
    if history_id:
        configurable["checkpoint_id"] = history_id
    snapshot = await graph.aget_state({"configurable": configurable})
    found = snapshot.config.get("configurable") or {}
    checkpoint_id = found.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise RuntimeError("session has no completed history to fork from")
    if found.get("checkpoint_ns"):
        raise RuntimeError("LangGraph checkpoint branching supports root graphs only")
    if history_id and checkpoint_id != history_id:
        raise RuntimeError("LangGraph branch point was not found")
    if snapshot.next or snapshot.tasks:
        raise RuntimeError("history point is not a completed root checkpoint")
    return LangGraphCheckpoint(thread_id=thread_id, checkpoint_id=checkpoint_id)


def _cancelled_tip_updates(snapshot: Any, writer_node: str) -> list[StateUpdate]:
    """Materialize a cancelled-turn marker tip as one synthetic superstep.

    The marker checkpoint deliberately carries no checkpoint_writes rows, so
    task-derived replay would lose its mutations (the rescued user message and
    the stopped stamp). Overwrite replaces the messages channel wholesale with
    the tip's authoritative list — the dest replay re-mints message ids, so
    id-keyed reducer replacement cannot carry the stamp.
    """
    messages = list((snapshot.values or {}).get("messages") or [])
    if not messages:
        return []
    return [
        StateUpdate({"messages": Overwrite(messages)}, writer_node, "cancelled-turn")
    ]


def _replay_writer_node(graph: Any) -> str:
    """Node a synthetic superstep's writes are attributed to."""
    names = [
        str(node) for node in graph.get_graph().nodes if not str(node).startswith("__")
    ]
    return sorted(names)[0] if names else "__start__"


async def checkpoint_replay_plan(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    source: LangGraphCheckpoint,
) -> tuple[list[list[StateUpdate]], list[str]]:
    """Build the copyable superstep plan, or raise before any dest write.

    Returns the supersteps and every cancelled-opener id the source history
    records. Replay rebuilds checkpoints and drops per-turn marker metadata, so
    the caller re-records this set on the copy — otherwise a forked stopped
    turn would lose its badge.
    """
    source_config: RunnableConfig = {
        "configurable": {
            "thread_id": source.thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": source.checkpoint_id,
        }
    }
    selected = await graph.aget_state(source_config)
    if _root_graph_checkpoint(selected.config) != source:
        raise RuntimeError("LangGraph branch point was not found")
    if selected.next or selected.tasks:
        raise RuntimeError("LangGraph branch point must be a completed root checkpoint")
    supersteps: list[list[StateUpdate]] = []
    cancelled_openers: list[str] = []
    for snapshot in reversed(
        [item async for item in _checkpoint_history(graph, source_config)]
    ):
        snapshot_metadata = getattr(snapshot, "metadata", None)
        if isinstance(snapshot_metadata, dict) and snapshot_metadata.get(
            CANCELLED_TURN_METADATA_KEY
        ):
            cancelled_openers.extend(
                sorted(cancelled_opener_ids_from_metadata(snapshot_metadata))
            )
            updates = _cancelled_tip_updates(snapshot, _replay_writer_node(graph))
        else:
            updates = _state_updates(snapshot)
        if updates:
            supersteps.append(updates)
    if not supersteps:
        raise RuntimeError("LangGraph branch point has no checkpoint history")
    return supersteps, cancelled_openers


async def _terminalize_copied_marker_tip(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    thread_id: str,
    *,
    cancelled_message_id: Any = None,
    cancelled_openers: Iterable[str] = (),
) -> None:
    """Close the copied tip and record the thread's cancelled openers on it.

    Replaying a marker tip applies its message state wholesale (no task
    writes), so nodes that never ran in the source have no ``versions_seen``
    coverage on the dest and would read as pending — a follow-up on the fork
    would resume the cancelled super-step, and the copy's own source-equality
    check rejects a non-terminal tip. The written tip keeps the marker: it is
    itself an authoritative-values, writes-less checkpoint and must replay as
    one when the fork becomes a fork source.

    ``cancelled_message_id`` is the source tip's opener id and
    ``cancelled_openers`` every opener the source history recorded. Replay drops
    the per-turn marker metadata, so the copy re-records them here — the reader
    derives the outcome from this metadata, not from the message.
    """
    checkpointer = graph.checkpointer
    if checkpointer is None or isinstance(checkpointer, bool):
        return
    # Any: the saver is a duck type here (native unwrap or the checkpointer
    # itself); the marker metadata rides keys beyond CheckpointMetadata.
    saver: Any = getattr(checkpointer, "native", None) or checkpointer
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    latest = await saver.aget_tuple(config)
    if latest is None:
        return
    channel_versions = dict(latest.checkpoint.get("channel_versions") or {})
    versions_seen = {
        node: dict(seen)
        for node, seen in (latest.checkpoint.get("versions_seen") or {}).items()
    }
    terminalize_versions_seen(
        versions_seen, channel_versions, graph.get_graph().nodes.keys()
    )
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = clear_queued_tasks(
        dict(latest.checkpoint.get("channel_values") or {})
    )
    checkpoint["channel_versions"] = channel_versions
    checkpoint["versions_seen"] = versions_seen
    metadata: dict[str, Any] = {
        "source": "loop",
        "step": (latest.metadata or {}).get("step", -1) + 1,
        "parents": {},
        CANCELLED_TURN_METADATA_KEY: True,
    }
    if isinstance(cancelled_message_id, str) and cancelled_message_id:
        metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY] = cancelled_message_id
    carried = sorted({opener for opener in cancelled_openers if opener})
    if carried:
        metadata[CANCELLED_TURN_MESSAGE_IDS_METADATA_KEY] = carried
    await saver.aput(
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": latest.checkpoint["id"],
            }
        },
        checkpoint,
        metadata,
        channel_versions,
    )


async def copy_checkpoint(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    target_config: RunnableConfig,
    source: LangGraphCheckpoint,
    *,
    plan: tuple[list[list[StateUpdate]], list[str]] | None = None,
) -> None:
    """Rebuild ``source`` in the empty thread named by ``target_config``."""
    checkpointer = graph.checkpointer
    if not isinstance(checkpointer, BaseCheckpointSaver):
        raise RuntimeError("LangGraph checkpoint branching requires a checkpointer")

    target_configurable = dict(target_config.get("configurable") or {})
    target_thread_id = cast(str, target_configurable["thread_id"])
    async for _ in checkpointer.alist(
        {"configurable": {"thread_id": target_thread_id}}, limit=1
    ):
        raise RuntimeError("LangGraph branch destination thread is not empty")

    supersteps, cancelled_openers = (
        plan if plan is not None else await checkpoint_replay_plan(graph, source)
    )
    source_config: RunnableConfig = {
        "configurable": {
            "thread_id": source.thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": source.checkpoint_id,
        }
    }
    selected = await graph.aget_state(source_config)
    target_configurable["checkpoint_ns"] = ""
    target_configurable.pop("checkpoint_id", None)
    destination_config: RunnableConfig = {
        **target_config,
        "configurable": target_configurable,
    }
    copied_config = await graph.abulk_update_state(destination_config, supersteps)
    selected_metadata = getattr(selected, "metadata", None)
    marker_tip = isinstance(selected_metadata, dict) and bool(
        selected_metadata.get(CANCELLED_TURN_METADATA_KEY)
    )
    # Re-record the source's cancelled openers on the copy whenever the history
    # held any — not only when the tip itself is a marker — so a fork of later
    # history keeps every earlier stopped turn classified.
    if marker_tip or cancelled_openers:
        await _terminalize_copied_marker_tip(
            graph,
            target_thread_id,
            cancelled_message_id=(
                selected_metadata.get(CANCELLED_TURN_MESSAGE_ID_METADATA_KEY)
                if isinstance(selected_metadata, dict)
                else None
            ),
            cancelled_openers=cancelled_openers,
        )
        copied = await graph.aget_state(
            {"configurable": {"thread_id": target_thread_id, "checkpoint_ns": ""}}
        )
    else:
        copied = await graph.aget_state(copied_config)
    if (
        copied.next
        or copied.tasks
        or _state_without_message_ids(copied.values)
        != _state_without_message_ids(selected.values)
    ):
        raise RuntimeError("Copied LangGraph branch point does not match its source")


async def destination_thread_is_empty(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    config: RunnableConfig,
) -> bool:
    """True when ``config``'s thread has no checkpoints yet."""
    checkpointer = graph.checkpointer
    if not isinstance(checkpointer, BaseCheckpointSaver):
        raise RuntimeError("LangGraph checkpoint branching requires a checkpointer")
    thread_id = cast(str, (config.get("configurable") or {})["thread_id"])
    async for _ in checkpointer.alist(
        {"configurable": {"thread_id": thread_id}}, limit=1
    ):
        return False
    return True


async def initialize_checkpoint_branch(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    config: RunnableConfig,
    source: LangGraphCheckpoint,
) -> None:
    """Copy ``source`` onto an empty dest thread.

    In-app fork already materializes dest. Atlas Agent Engine dest invoke may still send
    the source coordinate; if dest already has checkpoints, leave it.
    """
    if not await destination_thread_is_empty(graph, config):
        return
    await copy_checkpoint(graph, config, source)
