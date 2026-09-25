"""LangGraph-backed implementation of :class:`AERQueryPlugin`.

Reads from the same MongoDB collections that LangGraph's ``MongoDBSaver``
writes to (``checkpoints``, ``checkpoint_writes``). Deserialization is
delegated to the saver's ``serde`` so this code does not need to know how
LangGraph encodes state on disk.

Checkpoints are keyed by a workspace-scoped ``thread_id`` —
``f"{session_id}:{workspace_id}"`` (see :mod:`agent_engine_sdk_langgraph.thread_id`)
— so two agents in the same project DB cannot collide on a shared
``session_id``. The plugin is constructed with its own ``workspace_id`` (the
agent's ``APP_ID``); it composes that scope onto the plain ``session_id`` it
receives before querying, and strips it back off before returning. Callers
therefore only ever see the plain ``session_id``.

Reads never query the bare unscoped key when a workspace scope is known:
bare keys are shared across every workspace on the store.
An empty scope is legitimate only on explicitly unscoped runtimes (local
dev / tests, no ``APP_ID``), whose writes are bare-keyed to match. Managed
runtimes carry ``REQUIRE_PROJECT_SCOPED_DB`` and reject a missing ``APP_ID``.

This composition is the same one the write path applies in
:mod:`agent_engine_sdk_langgraph.agent`, so reads and writes resolve to identical
keys. The OE proxy still pre-filters the forwarded ``session_id`` list by
``workspace_id`` via the ``executions`` collection — defense in depth on top
of this scoping. See ``agent_engine_runner_shared.server.query`` for the full trust-model
discussion.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.mongodb import MongoDBSaver
from agent_engine_sdk.models import (
    LLMToolCall,
    SessionMessage,
    SessionMessagesResponse,
    SessionsSummaryResponse,
    SessionSummary,
)
from pydantic import JsonValue, TypeAdapter, ValidationError

from .thread_id import (
    session_id_from_thread_id,
    thread_ids_for_query,
    thread_ids_for_sessions_query,
)

logger = logging.getLogger(__name__)
_JSON_VALUE_ADAPTER = TypeAdapter(JsonValue)

_PREVIEW_MAX_CHARS = 80
_PREVIEW_WRITES_PER_SESSION = 5

# Cap on the bytes logged from a deserialization exception. langgraph's
# serde errors can carry chunks of the raw on-disk payload (and therefore
# user content) — truncate so a Splunk-visible warning never doubles as a
# data-leak vector.
_DESERIALIZATION_ERROR_MAX_CHARS = 200

# Cap the per-session checkpoint scan in ``get_messages_for_session``. LangGraph
# writes one checkpoint per graph step, so an unbounded scan is O(turns) in
# both DB I/O and deserialization cost. Mirrors the OE's historical cap on
# the equivalent Go read path. Sessions with more than this many checkpoints
# return the messages present in the most recent ``_MAX_CHECKPOINTS_PER_SESSION``
# checkpoints; older turns are not surfaced. Raise this if real sessions
# routinely exceed it.
_MAX_CHECKPOINTS_PER_SESSION = 500

# Server-side and client-side time budgets on the read paths. Without
# these, a slow / hung MongoDB pins the executor thread serving the
# request indefinitely; with them, the request fails fast and the thread
# is returned to the pool. The two numbers exist in a sandwich: maxTimeMS
# is enforced inside the database, the asyncio timeout is enforced in the
# AER's event loop. The asyncio bound is set higher to allow maxTimeMS
# to surface its own error first when both fire.
_AGGREGATION_TIMEOUT_MS = 15_000
_MESSAGE_READ_TIMEOUT_SECONDS = 20.0

# Cap on how many per-session checkpoint scans the summaries path runs at
# once. A sessions-list request can name hundreds of sessions; without a bound
# each one would open its own checkpoint cursor simultaneously.
_MESSAGE_COUNT_CONCURRENCY = 10

# LangChain BaseMessage.type → platform Message role used on the wire.
# Matches the live invoke/stream vocabulary in sdk-core (user, assistant, tool,
# system). "function" is an older LangChain message type collapsed onto "tool"
# so callers do not need to distinguish.
_ROLE_MAP = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}


class LangGraphQueryPlugin:
    """AERQueryPlugin backed by LangGraph's MongoDBSaver."""

    def __init__(
        self,
        checkpointer: MongoDBSaver,
        workspace_id: str = "",
        workspace_id_resolver: Callable[[], str | None] | None = None,
    ) -> None:
        self._checkpointer = checkpointer
        self._checkpoints_collection = checkpointer.checkpoint_collection
        self._writes_collection = checkpointer.writes_collection
        self._serde = checkpointer.serde
        # Static fallback for tests; production passes workspace_id_resolver
        # from TenantRuntime.get_checkpoint_workspace_id so reads track the
        # same APP_ID / wire fallback as the write path.
        self._workspace_id = workspace_id or ""
        self._workspace_id_resolver = workspace_id_resolver

    def _effective_workspace_id(self) -> str:
        """Resolve the workspace scope for a read.

        ``""`` is a legitimate scope — an explicitly unscoped runtime (local
        dev / tests, no ``APP_ID``) whose checkpoints are bare-keyed by
        construction, matching the write path. The runner rejects this state
        when managed-runtime scoping is required. A resolver returning ``None``
        signals a genuinely unresolvable scope (a custom resolver that cannot
        make that call) and fails closed rather than silently reading the
        shared store.
        """
        if self._workspace_id_resolver is not None:
            resolved = self._workspace_id_resolver()
            if resolved is None:
                raise RuntimeError(
                    "checkpoint workspace scope unavailable; refusing to serve "
                    "session queries without a workspace"
                )
            return resolved
        return self._workspace_id

    async def get_summaries_for_sessions(
        self, session_ids: list[str]
    ) -> SessionsSummaryResponse:
        if not session_ids:
            return SessionsSummaryResponse(sessions=[])
        loop = asyncio.get_running_loop()
        summaries = await loop.run_in_executor(
            None, self._aggregate_session_summaries, list(session_ids)
        )
        # ``message_count`` is filled from the same authoritative message list
        # the messages endpoint returns, so the two always agree (AP-3064).
        # It cannot come from the aggregation: the saver stores checkpoints as
        # serde-encoded blobs, opaque to aggregation operators.
        counts = await self._collect_message_counts(
            [summary.session_id for summary in summaries]
        )
        for summary in summaries:
            summary.message_count = counts.get(summary.session_id, 0)
        return SessionsSummaryResponse(sessions=summaries)

    async def get_messages_for_session(
        self, session_id: str
    ) -> SessionMessagesResponse:
        authoritative = await self._load_authoritative_messages(session_id)
        if authoritative is None:
            return SessionMessagesResponse(messages=[])

        authoritative_messages, first_appearance_by_index = authoritative
        return SessionMessagesResponse(
            messages=[
                _to_session_message(
                    message=message,
                    session_id=session_id,
                    message_index=message_index,
                    timestamp=first_appearance_by_index.get(message_index, ""),
                )
                for message_index, message in enumerate(authoritative_messages)
            ]
        )

    async def _load_authoritative_messages(
        self, session_id: str
    ) -> tuple[list[Any], dict[int, str]] | None:
        """Resolve a session's authoritative message list.

        The ``messages`` channel of the most recent checkpoint that carries a
        usable one, plus each message index's first-appearance timestamp.
        Returns ``None`` when no checkpoint has a usable ``messages`` channel.

        Both the messages endpoint and the sessions-list ``message_count``
        derive from this so the two always agree (AP-3064) — ``message_count``
        reports the number of messages, not the number of checkpoints.
        """
        # Walk checkpoints via MongoDBSaver.alist() — deserialization, cursor
        # lifecycle, and timestamp surfacing (``checkpoint["ts"]``) are all
        # owned by LangGraph. A corrupt checkpoint here propagates to the
        # caller rather than being silently skipped: that's the right call for
        # state-store corruption.
        thread_ids = thread_ids_for_query(session_id, self._effective_workspace_id())
        checkpoints_newest_first = await asyncio.wait_for(
            self._collect_checkpoints_for_thread_ids(thread_ids),
            timeout=_MESSAGE_READ_TIMEOUT_SECONDS,
        )

        # Keep only checkpoints whose ``messages`` channel is a usable list;
        # everything else is skipped. The most recent such checkpoint is the
        # authoritative message list.
        checkpoints_with_messages_newest_first = [
            (checkpoint_tuple, messages)
            for checkpoint_tuple in checkpoints_newest_first
            if (messages := _extract_messages(checkpoint_tuple)) is not None
        ]

        if not checkpoints_with_messages_newest_first:
            return None

        _, authoritative_messages = checkpoints_with_messages_newest_first[0]
        return authoritative_messages, _attribute_first_appearance_timestamps(
            checkpoints_with_messages_newest_first
        )

    async def _collect_message_counts(self, session_ids: list[str]) -> dict[str, int]:
        """Count messages per session from the authoritative message list.

        Unlike the single-session messages endpoint — where a corrupt or
        unreadable checkpoint rightly surfaces as a 500 — a failed read here
        degrades that one session's count to 0 and leaves the rest of the list
        intact. One unreadable session must not take down a summaries request
        naming hundreds of others.

        Scans run in bounded batches: a sessions-list request can name
        hundreds of sessions, and letting every one open its own checkpoint
        cursor at once would put hundreds of concurrent scans on the DB.
        """
        counts: dict[str, int] = {}
        for offset in range(0, len(session_ids), _MESSAGE_COUNT_CONCURRENCY):
            batch = session_ids[offset : offset + _MESSAGE_COUNT_CONCURRENCY]
            results = await asyncio.gather(
                *(self._load_authoritative_messages(sid) for sid in batch),
                return_exceptions=True,
            )
            for session_id, result in zip(batch, results, strict=True):
                if isinstance(result, BaseException):
                    logger.warning(
                        "message count unavailable for session %s; reporting 0: %s",
                        session_id,
                        result,
                    )
                    counts[session_id] = 0
                    continue
                counts[session_id] = len(result[0]) if result is not None else 0
        return counts

    async def _collect_checkpoints_for_thread_ids(
        self, thread_ids: list[str]
    ) -> list[Any]:
        """Materialize a bounded checkpoint window across one or more thread keys.

        Legacy checkpoints may still be keyed by the bare ``session_id`` while
        new writes use the workspace-scoped composite; both are queried and
        merged newest-first so in-flight conversations stay readable across
        deploy. Pulled into its own coroutine so the caller can wrap it in
        ``asyncio.wait_for``; without that bound, a hung MongoDB would keep an
        executor thread parked indefinitely.
        """
        checkpoints_newest_first: list[Any] = []
        for thread_id in thread_ids:
            thread_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
            checkpoints_newest_first.extend(
                [
                    checkpoint_tuple
                    async for checkpoint_tuple in self._checkpointer.alist(
                        thread_config, limit=_MAX_CHECKPOINTS_PER_SESSION
                    )
                ]
            )

        checkpoints_newest_first.sort(
            key=_checkpoint_timestamp,
            reverse=True,
        )
        return checkpoints_newest_first[:_MAX_CHECKPOINTS_PER_SESSION]

    # ------------------------------------------------------------------
    # Session summaries
    # ------------------------------------------------------------------

    def _aggregate_session_summaries(
        self, session_ids: list[str]
    ) -> list[SessionSummary]:
        workspace_id = self._effective_workspace_id()
        thread_ids = thread_ids_for_sessions_query(session_ids, workspace_id)
        aggregate_doc_by_session_id: dict[str, dict[str, Any]] = {}
        for aggregate_doc in self._checkpoints_collection.aggregate(
            _session_summaries_pipeline(thread_ids),
            maxTimeMS=_AGGREGATION_TIMEOUT_MS,
        ):
            session_id = session_id_from_thread_id(
                str(aggregate_doc.get("_id", "")), workspace_id
            )
            if not session_id:
                continue
            aggregate_doc_by_session_id[session_id] = _merge_session_summary_aggregate(
                aggregate_doc_by_session_id.get(session_id),
                aggregate_doc,
            )

        preview_by_session_id = self._collect_previews(
            list(aggregate_doc_by_session_id.keys())
        )

        return [
            SessionSummary(
                session_id=session_id,
                last_activity=_iso_or_empty(aggregate_doc.get("latest_ts")),
                created_at=_iso_or_empty(aggregate_doc.get("first_ts")),
                # Filled in by the caller from the authoritative message list.
                message_count=0,
                first_message_preview=preview_by_session_id.get(session_id, ""),
            )
            for session_id, aggregate_doc in aggregate_doc_by_session_id.items()
        ]

    def _collect_previews(self, session_ids: list[str]) -> dict[str, str]:
        if not session_ids:
            return {}

        workspace_id = self._effective_workspace_id()
        thread_ids = thread_ids_for_sessions_query(session_ids, workspace_id)
        writes_by_session_id: dict[str, list[dict[str, Any]]] = {}
        for aggregate_doc in self._writes_collection.aggregate(
            _message_previews_pipeline(thread_ids),
            maxTimeMS=_AGGREGATION_TIMEOUT_MS,
        ):
            session_id = session_id_from_thread_id(
                str(aggregate_doc.get("_id", "")), workspace_id
            )
            if not session_id:
                continue
            writes_by_session_id.setdefault(session_id, []).extend(
                aggregate_doc.get("writes") or []
            )

        return {
            session_id: preview
            for session_id, candidate_writes in writes_by_session_id.items()
            if (
                preview := self._first_non_empty_human_preview(
                    candidate_writes, session_id
                )
            )
        }

    def _first_non_empty_human_preview(
        self, candidate_writes: list[dict[str, Any]], session_id: str
    ) -> str:
        """Return the first non-empty human-message preview across ``writes``.

        ``writes`` is the per-session $push result from the preview pipeline,
        ordered oldest-first. Returns ``""`` if no write decodes to a human
        message with text content.
        """
        candidate_previews = (
            self._decode_human_preview(write_record, session_id)
            for write_record in candidate_writes
        )
        return next(
            (preview for preview in candidate_previews if preview),
            "",
        )

    def _decode_human_preview(
        self, write_record: dict[str, Any], session_id: str
    ) -> str:
        """Decode a single ``checkpoint_writes`` document and return the
        first human-authored text content it contains, truncated.

        Returns ``""`` for non-human writes, malformed writes, and any
        deserialization failures (logged at WARNING for observability).
        """
        serde_type_tag = write_record.get("type")
        serialized_value = write_record.get("value")
        if not serde_type_tag or serialized_value is None:
            return ""

        try:
            deserialized = self._serde.loads_typed((serde_type_tag, serialized_value))
        except Exception as exc:  # noqa: BLE001 — corrupt single write shouldn't fail the list
            # Surface at warning level so schema drift / data corruption is
            # observable in Splunk — a debug-level swallow would silently
            # serve empty previews to every caller. Bound the message
            # length so deserialization errors that echo raw payload bytes
            # (or user content) don't bloat logs or leak data.
            logger.warning(
                "preview deserialization failed for session %s: %s: %s",
                session_id,
                type(exc).__name__,
                str(exc)[:_DESERIALIZATION_ERROR_MAX_CHARS],
            )
            return ""

        decoded_messages = (
            deserialized if isinstance(deserialized, list) else [deserialized]
        )
        human_contents = (
            _human_content(decoded_message) for decoded_message in decoded_messages
        )
        return next(
            (content[:_PREVIEW_MAX_CHARS] for content in human_contents if content),
            "",
        )


# ----------------------------------------------------------------------
# Module-level helpers — kept stateless and pure so they're easy to read
# and easy to test in isolation.
# ----------------------------------------------------------------------


def _merge_session_summary_aggregate(
    existing: dict[str, Any] | None, new_doc: dict[str, Any]
) -> dict[str, Any]:
    """Collapse duplicate summary rows for the same session."""
    if existing is None:
        return dict(new_doc)

    latest_ts = existing.get("latest_ts")
    new_latest_ts = new_doc.get("latest_ts")
    if latest_ts is not None and new_latest_ts is not None:
        latest_ts = max(latest_ts, new_latest_ts)
    else:
        latest_ts = latest_ts or new_latest_ts

    first_ts = existing.get("first_ts")
    new_first_ts = new_doc.get("first_ts")
    if first_ts is not None and new_first_ts is not None:
        first_ts = min(first_ts, new_first_ts)
    else:
        first_ts = first_ts or new_first_ts

    return {
        "_id": existing.get("_id", new_doc.get("_id")),
        "latest_ts": latest_ts,
        "first_ts": first_ts,
    }


def _session_summaries_pipeline(thread_ids: list[str]) -> list[dict[str, Any]]:
    """Aggregation pipeline that reduces per-session checkpoint rows to
    one summary row per ``thread_id`` (first/last activity). The message count
    is derived separately from the authoritative message list (see
    ``_collect_message_counts``) so it matches the messages endpoint."""
    return [
        {"$match": {"thread_id": {"$in": thread_ids}}},
        {"$addFields": {"_ts": {"$toDate": "$_id"}}},
        {
            "$group": {
                "_id": "$thread_id",
                "latest_ts": {"$max": "$_ts"},
                "first_ts": {"$min": "$_ts"},
            }
        },
        {"$sort": {"latest_ts": -1}},
    ]


def _message_previews_pipeline(thread_ids: list[str]) -> list[dict[str, Any]]:
    """Aggregation pipeline that pulls the oldest few ``messages``-channel
    writes per session so we can extract a human-message preview.

    ``$firstN`` caps the per-session write count at the accumulator — the
    full sorted stream is never materialized in memory. The preceding
    ``$sort: {_id: 1}`` makes "first N" mean "oldest N".
    """
    return [
        {
            "$match": {
                "thread_id": {"$in": thread_ids},
                "channel": "messages",
            }
        },
        {"$sort": {"_id": 1}},
        {
            "$group": {
                "_id": "$thread_id",
                "writes": {
                    "$firstN": {
                        "input": {"type": "$type", "value": "$value"},
                        "n": _PREVIEW_WRITES_PER_SESSION,
                    }
                },
            }
        },
    ]


def _extract_messages(checkpoint_tuple: Any) -> list[Any] | None:
    """Return the ``messages`` channel from a ``CheckpointTuple`` if it's
    a list, otherwise ``None``.

    A ``None`` return signals "skip this checkpoint" — the channel is
    either absent, malformed, or the checkpoint itself isn't a dict.
    """
    checkpoint = getattr(checkpoint_tuple, "checkpoint", None)
    if not isinstance(checkpoint, dict):
        return None
    channel_values = checkpoint.get("channel_values", {})
    if not isinstance(channel_values, dict):
        return None
    messages = channel_values.get("messages")
    return messages if isinstance(messages, list) else None


def _checkpoint_timestamp(checkpoint_tuple: Any) -> str:
    """Read ``checkpoint["ts"]`` — LangGraph's own ISO-8601 timestamp
    field — falling back to the empty string if it's missing."""
    checkpoint = getattr(checkpoint_tuple, "checkpoint", None)
    if not isinstance(checkpoint, dict):
        return ""
    return str(checkpoint.get("ts") or "")


def _attribute_first_appearance_timestamps(
    checkpoints_with_messages_newest_first: list[tuple[Any, list[Any]]],
) -> dict[int, str]:
    """For each message index in the conversation history, return the
    timestamp of the *earliest* checkpoint where that index existed.

    The caller passes pairs in newest-first order (the natural ``alist()``
    direction); this function walks them oldest-first so that
    ``setdefault`` records each index's first appearance.
    """
    first_appearance_by_index: dict[int, str] = {}
    for checkpoint_tuple, messages_at_this_step in reversed(
        checkpoints_with_messages_newest_first
    ):
        timestamp = _checkpoint_timestamp(checkpoint_tuple)
        for message_index in range(len(messages_at_this_step)):
            first_appearance_by_index.setdefault(message_index, timestamp)
    return first_appearance_by_index


def _to_session_message(
    *,
    message: Any,
    session_id: str,
    message_index: int,
    timestamp: str,
) -> SessionMessage:
    raw_message_type = getattr(message, "type", "") or ""
    role = _ROLE_MAP.get(raw_message_type, raw_message_type)
    name = getattr(message, "name", "") or ""
    message_id = getattr(message, "id", "") or f"msg-{session_id}-{message_index}"
    additional_kwargs = getattr(message, "additional_kwargs", None)
    return SessionMessage(
        id=message_id,
        role=role,
        content=_message_text(message),
        timestamp=timestamp,
        session_id=session_id,
        name=name,
        tool_calls=_tool_calls(message),
        # ToolMessage.tool_call_id is the id of the assistant tool call this
        # message answers; absent on every other message type.
        tool_call_id=getattr(message, "tool_call_id", None) or None,
        additional_kwargs=_json_safe_metadata(additional_kwargs),
    )


def _tool_calls(message: Any) -> list[LLMToolCall] | None:
    """Surface an AI message's normalized tool calls as typed records.

    LangChain stores them on ``AIMessage.tool_calls`` as dicts carrying
    ``id``/``name``/``args``/``type``; the ``id`` is the stable join key the
    matching tool result echoes back in its ``tool_call_id``. Returns
    ``None`` for messages without tool calls so the field is omitted on the
    wire.
    """
    raw_tool_calls = getattr(message, "tool_calls", None)
    if not isinstance(raw_tool_calls, list):
        return None
    tool_calls = [
        LLMToolCall(
            id=tool_call.get("id"),
            name=tool_call.get("name"),
            args=tool_call.get("args"),
            type=tool_call.get("type"),
        )
        for tool_call in raw_tool_calls
        if isinstance(tool_call, dict)
    ]
    return tool_calls or None


def _json_safe_metadata(value: Any) -> dict[str, JsonValue] | None:
    if not isinstance(value, dict):
        return None

    metadata: dict[str, JsonValue] = {}
    for key, raw_field in value.items():
        if not isinstance(key, str):
            continue
        try:
            metadata[key] = cast(
                JsonValue,
                _JSON_VALUE_ADAPTER.validate_python(raw_field),
            )
        except ValidationError:
            logger.warning("Skipping non-JSON message metadata field: %s", key)

    return metadata or None


def _message_text(message: Any) -> str:
    """Return a flat-string view of a LangChain message's content.

    Uses ``BaseMessage.text`` when available — it collapses str/list
    content to a string and extracts only ``type: "text"`` blocks.
    Falls back to ``str(content)`` for stand-ins that don't implement
    the LangChain message protocol (e.g. test doubles).
    """
    text_accessor = getattr(message, "text", None)
    if text_accessor is None:
        return str(getattr(message, "content", "") or "")
    return str(text_accessor)


def _human_content(decoded_message: Any) -> str:
    if getattr(decoded_message, "type", None) != "human":
        return ""
    return _message_text(decoded_message)


def _iso_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return ""
