"""Commit a terminal checkpoint for a cancelled turn so history can serve it.

Nothing checkpoints on cancel: LangGraph commits only at super-step
boundaries, and the drain tears the loop down mid-super-step, so the turn
vanishes from playground history. This writes one synthetic root-namespace
checkpoint from the last committed state (never the in-flight super-step —
its channel values are mid-mutation), records the turn-opening message id in
the checkpoint metadata so the history reader can classify it as stopped, and
terminal-izes ``versions_seen`` for every graph node so the tip presents as
completed: the cancelled super-step must never look resumable, and fork
validation rejects snapshots with pending tasks.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import empty_checkpoint

_logger = logging.getLogger(__name__)

CANCELLED_TURN_METADATA_KEY = "agent_engine_cancelled_turn"
CANCELLED_TURN_MESSAGE_ID_METADATA_KEY = "agent_engine_cancelled_turn_message_id"
# A fork carries the source thread's whole set of cancelled-opener ids onto its
# copied tip under this plural key: replay rebuilds checkpoints and drops the
# per-turn marker metadata, so the fork must re-record the set or a forked
# stopped turn would lose its badge.
CANCELLED_TURN_MESSAGE_IDS_METADATA_KEY = "agent_engine_cancelled_turn_message_ids"

# Framework-owned channel carrying queued Send packets (the deep agents'
# dispatched tool calls). Rebuilt from channel_values on read; absence reads
# as an empty queue.
TASKS_CHANNEL_KEY = "__pregel_tasks"


async def commit_cancelled_turn(
    saver: Any,
    *,
    thread_id: str,
    user_message: str,
    turn_message_id: str | None,
    node_names: Collection[str],
    resume: bool = False,
) -> str | None:
    """Write the cancelled-turn marker checkpoint; return the turn's user-message id.

    ``saver`` is the native Mongo-backed saver (unwrap ``PlatformCheckpointer``
    at the call site). ``turn_message_id`` is the id captured when the turn's
    graph input was built — identity, not prompt text, so repeated or
    hook-rewritten prompts can't misidentify the turn. ``None`` means no
    opener id was captured at build: a preparation-phase cancel, a resume
    turn, or a hook return outside the supported message-likes (those are
    converted and captured at build). A ``resume`` turn recovers its already
    persisted opener from the loaded checkpoint; otherwise the stopped message
    is appended under a freshly minted id — never matched against an older
    prompt's content. No ``checkpoint_writes`` rows are attached — the next
    invoke's cancelled-predecessor fence stays a no-op on this tip, and the
    half-dead super-step's orphaned writes are never joined to it.
    """
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    latest = await saver.aget_tuple(config)

    channel_values: dict[str, Any] = {}
    channel_versions: dict[str, Any] = {}
    versions_seen: dict[str, dict[str, Any]] = {}
    parent_id: str | None = None
    prior_step = -1
    if latest is not None:
        channel_values = dict(latest.checkpoint.get("channel_values") or {})
        channel_versions = dict(latest.checkpoint.get("channel_versions") or {})
        versions_seen = {
            node: dict(seen)
            for node, seen in (latest.checkpoint.get("versions_seen") or {}).items()
        }
        parent_id = latest.checkpoint["id"]
        prior_step = (latest.metadata or {}).get("step", -1)
        clear_queued_tasks(channel_values)

    messages = list(channel_values.get("messages") or [])
    resolved_id = _locate_turn_message(messages, turn_message_id)
    if resolved_id is None and resume:
        # A resume opens no new user message: the opener is the last human
        # message already committed to the thread. Recover it here, from the
        # checkpoint the commit already loaded, so a Stop that beats the
        # preamble's capture still stamps the persisted opener instead of
        # appending a duplicate. Gated on resume — a fresh-turn preparation
        # cancel must append, never adopt an older identical prompt.
        resolved_id = _last_human_message_id(messages)
    if resolved_id is None and user_message:
        # The cancel beat even the input checkpoint: inject the user message
        # under its captured id or the turn cannot exist in history at all.
        resolved_id = turn_message_id or f"msg-cancelled-{uuid4()}"
        messages.append(HumanMessage(content=user_message, id=resolved_id))
        channel_versions["messages"] = saver.get_next_version(
            channel_versions.get("messages"), None
        )
    channel_values["messages"] = messages

    # Every graph node reads as caught up — including nodes that never ran and
    # therefore have no versions_seen entry. A node absent there presents as
    # pending, and fork validation rejects a tip with pending work.
    terminalize_versions_seen(versions_seen, channel_versions, node_names)

    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = channel_values
    checkpoint["channel_versions"] = channel_versions
    checkpoint["versions_seen"] = versions_seen

    metadata: dict[str, Any] = {
        "source": "loop",
        "step": prior_step + 1,
        "parents": {},
        CANCELLED_TURN_METADATA_KEY: True,
    }
    if resolved_id is not None:
        metadata[CANCELLED_TURN_MESSAGE_ID_METADATA_KEY] = resolved_id
    await saver.aput(
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": parent_id,
            }
        },
        checkpoint,
        metadata,
        channel_versions,
    )
    return resolved_id


def clear_queued_tasks(channel_values: dict[str, Any]) -> dict[str, Any]:
    """Drop the framework's queued Send packets from copied channel values.

    The last committed checkpoint can carry Sends dispatched to a still-running
    tool (the deep agents route tool calls through ``__pregel_tasks``), and
    PUSH work is not consumed by terminalizing ``versions_seen`` — a marker
    built on it would present pending tasks and be rejected by fork
    validation, and a resume would run the cancelled work. Dropping the
    channel reads back as an empty queue.
    """
    channel_values.pop(TASKS_CHANNEL_KEY, None)
    return channel_values


def terminalize_versions_seen(
    versions_seen: dict[str, dict[str, Any]],
    channel_versions: dict[str, Any],
    node_names: Collection[str],
) -> dict[str, dict[str, Any]]:
    """Mark every node as having seen every current channel version.

    A node absent from (or behind in) ``versions_seen`` presents as pending:
    resume would schedule it and fork validation rejects a tip with pending
    work. Shared by the cancelled-turn commit and the fork copy of a
    marker-tipped thread, which replays values without task writes.
    """
    for node in set(node_names) | versions_seen.keys():
        node_seen = versions_seen.setdefault(node, {})
        for channel, version in channel_versions.items():
            node_seen[channel] = version
    return versions_seen


def _locate_turn_message(
    messages: list[Any], turn_message_id: str | None
) -> str | None:
    """Resolve the turn-opening message by the id captured at input build.

    No content fallback: an identical earlier prompt would make content
    matching stamp the previous turn's opener, so a missing captured id is
    always answered by appending, never by matching.
    """
    if turn_message_id is None:
        return None
    for message in messages:
        if getattr(message, "id", None) == turn_message_id:
            return turn_message_id
    return None


def _last_human_message_id(messages: list[Any]) -> str | None:
    """Id of the last human message, or None when it has none.

    Used only to recover a native HITL resume's already-persisted opener (see
    ``commit_cancelled_turn``); a fresh turn never reaches it, so it can never
    adopt an older identical prompt.
    """
    for message in reversed(messages):
        if getattr(message, "type", None) != "human":
            continue
        message_id = getattr(message, "id", None)
        return message_id if isinstance(message_id, str) and message_id else None
    return None


def cancelled_opener_ids_from_metadata(metadata: Any) -> set[str]:
    """Opener ids a cancelled-turn checkpoint's metadata names.

    A marker checkpoint carries its own turn's singular opener id; a fork
    carries the source thread's whole set on its copied tip as a plural list.
    Never read from a message kwarg an agent could author, so both the reader
    and the fork resolve the same trusted set.
    """
    if not isinstance(metadata, dict) or not metadata.get(CANCELLED_TURN_METADATA_KEY):
        return set()
    ids: set[str] = set()
    message_id = metadata.get(CANCELLED_TURN_MESSAGE_ID_METADATA_KEY)
    if isinstance(message_id, str) and message_id:
        ids.add(message_id)
    carried = metadata.get(CANCELLED_TURN_MESSAGE_IDS_METADATA_KEY)
    if isinstance(carried, list):
        ids.update(
            candidate
            for candidate in carried
            if isinstance(candidate, str) and candidate
        )
    return ids
