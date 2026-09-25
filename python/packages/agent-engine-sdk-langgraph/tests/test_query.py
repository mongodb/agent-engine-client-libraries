"""Tests for :class:`LangGraphQueryPlugin`.

The plugin is exercised against a stubbed ``MongoDBSaver`` that exposes:

* ``checkpoint_collection`` — supports ``aggregate(...)`` returning
  pre-canned docs and ``find(...)`` returning a cursor sorted by
  ``checkpoint_id``.
* ``writes_collection`` — supports ``aggregate(...)`` for the preview
  pipeline.
* ``serde`` — a stub whose ``loads_typed`` returns whatever Python value
  was paired with the marker the test wrote into the fake document. This
  lets us pretend each ``(type, value)`` pair decodes to a known
  LangChain message list without depending on JsonPlusSerializer
  internals.

The goal here is to verify the plugin's contract end-to-end at the
Python layer — pipeline shape, role mapping, timestamp attribution,
empty-input handling — not to exercise the MongoDB / LangGraph
serialization stack itself.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from agent_engine_sdk.models import SessionsSummaryResponse

from agent_engine_sdk_langgraph.query import LangGraphQueryPlugin

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubSerde:
    """Serde stand-in: stores ``(type, value)`` → Python object mappings.

    Tests pre-register what each fake serialized blob should decode to,
    so the plugin code under test can call ``serde.loads_typed`` exactly
    the way it would against a real ``JsonPlusSerializer`` instance.

    Unregistered ``(type, value)`` pairs raise ``KeyError`` from
    ``loads_typed`` — this models the "corrupt doc" case the plugin's
    broad ``except Exception`` is meant to defend against.
    """

    def __init__(self) -> None:
        self._table: dict[tuple[str, Any], Any] = {}

    def register(self, type_str: str, value: Any, decoded: Any) -> None:
        self._table[(type_str, value)] = decoded

    def loads_typed(self, pair: tuple[str, Any]) -> Any:
        return self._table[pair]


class _StubAggregateCursor:
    """Iterable backed by a fixed list of docs."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = list(docs)

    def __iter__(self):
        return iter(self._docs)


class _StubCollection:
    """In-memory collection that records the pipelines it was called with."""

    def __init__(self) -> None:
        self.aggregate_calls: list[dict[str, Any]] = []
        self._aggregate_result: list[dict[str, Any]] = []

    def set_aggregate_result(self, docs: list[dict[str, Any]]) -> None:
        self._aggregate_result = docs

    def aggregate(
        self, pipeline: list[dict[str, Any]], **kwargs: Any
    ) -> _StubAggregateCursor:
        # Record both the pipeline and any kwargs so tests can assert on
        # operation-level settings like ``maxTimeMS``.
        self.aggregate_calls.append({"pipeline": pipeline, **kwargs})
        return _StubAggregateCursor(self._aggregate_result)


class _CheckpointTuple:
    """Minimal stand-in for langgraph.checkpoint.base.CheckpointTuple.

    Carries only the ``checkpoint`` field — that's the only one the plugin
    looks at. The real CheckpointTuple is a NamedTuple with more fields
    (config, metadata, parent_config, pending_writes); none of them matter
    for our read path.
    """

    def __init__(self, checkpoint: dict[str, Any]) -> None:
        self.checkpoint = checkpoint


class _StubSaver:
    """MongoDBSaver stand-in.

    Surfaces the two collection handles the plugin uses for aggregations
    (``checkpoint_collection``, ``writes_collection``), the saver's serde
    (used by the preview aggregation's per-write decode), and ``alist`` —
    the async checkpoint iterator the message-read path now goes through.
    """

    def __init__(self) -> None:
        self.checkpoint_collection = _StubCollection()
        self.writes_collection = _StubCollection()
        self.serde = _StubSerde()
        self.alist_calls: list[dict[str, Any]] = []
        self._alist_result: list[_CheckpointTuple] = []
        self._alist_results_by_thread_id: dict[str, list[_CheckpointTuple]] = {}

    def set_alist_result(self, checkpoints: list[dict[str, Any]]) -> None:
        """Pre-canned response for the next alist() call.

        MongoDBSaver.alist returns checkpoints in newest-first order; tests
        should follow that convention when populating the result.
        """
        self._alist_result = [_CheckpointTuple(ck) for ck in checkpoints]
        self._alist_results_by_thread_id = {}

    def set_alist_results_by_thread_id(
        self, results_by_thread_id: dict[str, list[dict[str, Any]]]
    ) -> None:
        """Per-thread pre-canned responses for dual-read tests."""
        self._alist_result = []
        self._alist_results_by_thread_id = {
            thread_id: [_CheckpointTuple(ck) for ck in checkpoints]
            for thread_id, checkpoints in results_by_thread_id.items()
        }

    async def alist(self, config: dict[str, Any], **kwargs: Any):
        self.alist_calls.append({"config": config, **kwargs})
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id in self._alist_results_by_thread_id:
            checkpoints = self._alist_results_by_thread_id[thread_id]
        else:
            checkpoints = self._alist_result
        for ct in checkpoints:
            yield ct


def _iso(seconds_after_epoch: int) -> str:
    """ISO-8601 string at a known instant — used as the ``ts`` field on
    test checkpoints, matching what MongoDBSaver returns from alist()."""
    return datetime(
        2026, 5, 22, 9, 0, seconds_after_epoch, tzinfo=timezone.utc
    ).isoformat()


# ---------------------------------------------------------------------------
# get_summaries_for_sessions
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_summaries_for_sessions_empty_input_short_circuits() -> None:
    saver = _StubSaver()
    plugin = LangGraphQueryPlugin(saver)  # type: ignore[arg-type]

    result = await plugin.get_summaries_for_sessions([])

    assert result == SessionsSummaryResponse(sessions=[])
    assert saver.checkpoint_collection.aggregate_calls == []
    assert saver.writes_collection.aggregate_calls == []


@pytest.mark.anyio
async def test_get_summaries_for_sessions_returns_per_session_aggregation() -> None:
    saver = _StubSaver()
    latest = datetime(2026, 5, 22, 10, 0, 0, tzinfo=timezone.utc)
    first = datetime(2026, 5, 22, 9, 0, 0, tzinfo=timezone.utc)
    saver.checkpoint_collection.set_aggregate_result(
        [
            {
                "_id": "session-a",
                "latest_ts": latest,
                "first_ts": first,
            }
        ]
    )
    # Preview pipeline returns one write per session; the plugin should
    # decode it and pull the first human message's content.
    saver.writes_collection.set_aggregate_result(
        [
            {
                "_id": "session-a",
                "writes": [{"type": "msgpack", "value": b"opaque"}],
            }
        ]
    )
    saver.serde.register("msgpack", b"opaque", [HumanMessage(content="hello world")])
    # message_count comes from the authoritative message list, not the
    # checkpoint rows: two messages spread over three checkpoints count as 2.
    saver.set_alist_result(
        [
            {
                "ts": _iso(2),
                "channel_values": {
                    "messages": [
                        HumanMessage(content="hello world", id="h1"),
                        AIMessage(content="hi", id="a1"),
                    ]
                },
            },
            {
                "ts": _iso(1),
                "channel_values": {
                    "messages": [HumanMessage(content="hello world", id="h1")]
                },
            },
            {"ts": _iso(0), "channel_values": {}},
        ]
    )

    plugin = LangGraphQueryPlugin(saver)  # type: ignore[arg-type]
    result = await plugin.get_summaries_for_sessions(["session-a"])

    assert len(result.sessions) == 1
    info = result.sessions[0]
    assert info.session_id == "session-a"
    assert info.last_activity == latest.isoformat()
    assert info.created_at == first.isoformat()
    assert info.message_count == 2
    assert info.first_message_preview == "hello world"


@pytest.mark.anyio
async def test_get_summaries_for_sessions_pipeline_filters_to_requested_ids() -> None:
    saver = _StubSaver()
    saver.checkpoint_collection.set_aggregate_result([])
    plugin = LangGraphQueryPlugin(saver)  # type: ignore[arg-type]

    await plugin.get_summaries_for_sessions(["s1", "s2"])

    assert len(saver.checkpoint_collection.aggregate_calls) == 1
    call = saver.checkpoint_collection.aggregate_calls[0]
    match_stage = call["pipeline"][0]
    assert match_stage == {"$match": {"thread_id": {"$in": ["s1", "s2"]}}}
    # Aggregation must carry a server-side timeout so a hung Mongo cannot
    # pin the AER's executor thread indefinitely.
    assert call["maxTimeMS"] > 0


@pytest.mark.anyio
async def test_preview_pipeline_scoped_to_messages_channel() -> None:
    saver = _StubSaver()
    saver.checkpoint_collection.set_aggregate_result(
        [
            {
                "_id": "s1",
                "latest_ts": datetime(2026, 5, 22, 10, 0, 0, tzinfo=timezone.utc),
                "first_ts": datetime(2026, 5, 22, 9, 0, 0, tzinfo=timezone.utc),
            }
        ]
    )
    saver.writes_collection.set_aggregate_result([])
    plugin = LangGraphQueryPlugin(saver)  # type: ignore[arg-type]

    await plugin.get_summaries_for_sessions(["s1"])

    assert len(saver.writes_collection.aggregate_calls) == 1
    call = saver.writes_collection.aggregate_calls[0]
    match_stage = call["pipeline"][0]
    assert match_stage["$match"]["channel"] == "messages"
    assert match_stage["$match"]["thread_id"] == {"$in": ["s1"]}
    assert call["maxTimeMS"] > 0


@pytest.mark.anyio
async def test_get_summaries_for_sessions_skips_non_human_preview_writes() -> None:
    """Preview should be drawn from the first human message, not an AI reply."""
    saver = _StubSaver()
    saver.checkpoint_collection.set_aggregate_result(
        [
            {
                "_id": "s1",
                "latest_ts": datetime(2026, 5, 22, 10, 0, 0, tzinfo=timezone.utc),
                "first_ts": datetime(2026, 5, 22, 9, 0, 0, tzinfo=timezone.utc),
            }
        ]
    )
    saver.writes_collection.set_aggregate_result(
        [
            {
                "_id": "s1",
                "writes": [
                    {"type": "msgpack", "value": b"ai-only"},
                    {"type": "msgpack", "value": b"first-human"},
                ],
            }
        ]
    )
    saver.serde.register("msgpack", b"ai-only", [AIMessage(content="hi back")])
    saver.serde.register("msgpack", b"first-human", [HumanMessage(content="hey")])

    plugin = LangGraphQueryPlugin(saver)  # type: ignore[arg-type]
    result = await plugin.get_summaries_for_sessions(["s1"])

    assert result.sessions[0].first_message_preview == "hey"


@pytest.mark.anyio
async def test_skips_corrupt_preview_write_and_continues() -> None:
    """A corrupt preview write must not stop the plugin from trying the next write.

    Models the case where MongoDBSaver's serde format has drifted and a
    write blob no longer deserializes. The plugin should log and skip,
    not propagate the error to the caller.
    """
    saver = _StubSaver()
    saver.checkpoint_collection.set_aggregate_result(
        [
            {
                "_id": "s1",
                "latest_ts": datetime(2026, 5, 22, 10, 0, 0, tzinfo=timezone.utc),
                "first_ts": datetime(2026, 5, 22, 9, 0, 0, tzinfo=timezone.utc),
            }
        ]
    )
    saver.writes_collection.set_aggregate_result(
        [
            {
                "_id": "s1",
                "writes": [
                    # First write is deliberately not registered with the
                    # stub serde — loads_typed will raise KeyError.
                    {"type": "msgpack", "value": b"corrupt-blob"},
                    {"type": "msgpack", "value": b"valid-human"},
                ],
            }
        ]
    )
    saver.serde.register(
        "msgpack", b"valid-human", [HumanMessage(content="hello after corrupt")]
    )

    plugin = LangGraphQueryPlugin(saver)  # type: ignore[arg-type]
    result = await plugin.get_summaries_for_sessions(["s1"])

    assert result.sessions[0].first_message_preview == "hello after corrupt"


@pytest.mark.anyio
async def test_get_summaries_for_sessions_preview_truncated_to_eighty_chars() -> None:
    saver = _StubSaver()
    saver.checkpoint_collection.set_aggregate_result(
        [
            {
                "_id": "s1",
                "latest_ts": datetime(2026, 5, 22, 10, 0, 0, tzinfo=timezone.utc),
                "first_ts": datetime(2026, 5, 22, 9, 0, 0, tzinfo=timezone.utc),
            }
        ]
    )
    long_content = "x" * 200
    saver.writes_collection.set_aggregate_result(
        [
            {
                "_id": "s1",
                "writes": [{"type": "msgpack", "value": b"long"}],
            }
        ]
    )
    saver.serde.register("msgpack", b"long", [HumanMessage(content=long_content)])

    plugin = LangGraphQueryPlugin(saver)  # type: ignore[arg-type]
    result = await plugin.get_summaries_for_sessions(["s1"])

    assert result.sessions[0].first_message_preview == "x" * 80


# ---------------------------------------------------------------------------
# get_messages_for_session
# ---------------------------------------------------------------------------
#
# The message-read path goes through MongoDBSaver.alist(), which yields
# already-deserialized CheckpointTuple objects in newest-first order. Each
# CheckpointTuple carries a ``checkpoint`` dict; the plugin reads ``ts``
# (LangGraph's own ISO timestamp field) and ``channel_values.messages``
# from it. Tests populate ``set_alist_result`` with the same shape and
# rely on the stub to yield in the order given.


@pytest.mark.anyio
async def test_get_messages_for_session_returns_empty_when_no_checkpoints() -> None:
    saver = _StubSaver()
    saver.set_alist_result([])
    plugin = LangGraphQueryPlugin(saver)  # type: ignore[arg-type]

    result = await plugin.get_messages_for_session("missing-session")

    assert result.messages == []


@pytest.mark.anyio
async def test_get_messages_for_session_maps_langchain_message_types_to_roles() -> None:
    """Every entry in ``_ROLE_MAP`` is exercised, plus the pass-through
    fallback for an unmapped ``BaseMessage.type``."""

    class _UnknownTypeMessage:
        """Stand-in for a non-LangChain message — verifies the role
        fallback returns the raw type string unchanged."""

        type = "mystery"
        content = "?"
        id = "u1"
        name = ""

    saver = _StubSaver()
    saver.set_alist_result(
        [
            {
                "ts": _iso(0),
                "channel_values": {
                    "messages": [
                        SystemMessage(content="sys", id="s1"),
                        HumanMessage(content="hi", id="h1"),
                        AIMessage(content="hello", id="a1"),
                        ToolMessage(
                            content="42", tool_call_id="tc", name="calc", id="t1"
                        ),
                        # Legacy LangChain type: "function" must collapse to "tool"
                        # for callers that don't distinguish.
                        FunctionMessage(content="legacy", name="legacy_fn", id="f1"),
                        _UnknownTypeMessage(),
                    ]
                },
            }
        ]
    )

    plugin = LangGraphQueryPlugin(saver)  # type: ignore[arg-type]
    result = await plugin.get_messages_for_session("s1")

    assert [m.role for m in result.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",  # function → tool collapse
        "mystery",  # unmapped type passes through unchanged
    ]
    assert [m.id for m in result.messages] == ["s1", "h1", "a1", "t1", "f1", "u1"]
    assert result.messages[3].name == "calc"
    assert result.messages[4].name == "legacy_fn"


@pytest.mark.anyio
async def test_get_messages_for_session_synthesizes_id_when_missing() -> None:
    saver = _StubSaver()
    saver.set_alist_result(
        [
            {
                "ts": _iso(0),
                "channel_values": {"messages": [HumanMessage(content="hi")]},
            }
        ]
    )

    plugin = LangGraphQueryPlugin(saver)  # type: ignore[arg-type]
    result = await plugin.get_messages_for_session("s1")

    assert result.messages[0].id == "msg-s1-0"


@pytest.mark.anyio
async def test_get_messages_for_session_preserves_message_additional_kwargs() -> None:
    saver = _StubSaver()
    artifact_metadata = {
        "artifacts": [
            {
                "id": "artifact-chart-loss-frequency",
                "kind": "chart",
                "title": "Loss frequency by pedal cohort",
            }
        ],
        "unsafe": object(),
    }
    saver.set_alist_result(
        [
            {
                "ts": _iso(0),
                "channel_values": {
                    "messages": [
                        AIMessage(
                            content="Here is the analysis.",
                            id="a1",
                            additional_kwargs=artifact_metadata,
                        )
                    ]
                },
            }
        ]
    )

    plugin = LangGraphQueryPlugin(saver)  # type: ignore[arg-type]
    result = await plugin.get_messages_for_session("s1")

    assert result.messages[0].additional_kwargs == {
        "artifacts": [
            {
                "id": "artifact-chart-loss-frequency",
                "kind": "chart",
                "title": "Loss frequency by pedal cohort",
            }
        ],
    }


@pytest.mark.anyio
async def test_messages_attributes_first_appearance_timestamp_per_message() -> None:
    """Each message's timestamp should be the ``ts`` of the checkpoint
    where it first appeared."""
    saver = _StubSaver()
    ts_old = _iso(0)
    ts_new = _iso(30)
    # alist yields newest-first.
    saver.set_alist_result(
        [
            {
                "ts": ts_new,
                "channel_values": {
                    "messages": [
                        HumanMessage(content="hi", id="h1"),
                        AIMessage(content="hello", id="a1"),
                    ]
                },
            },
            {
                "ts": ts_old,
                "channel_values": {"messages": [HumanMessage(content="hi", id="h1")]},
            },
        ]
    )

    plugin = LangGraphQueryPlugin(saver)  # type: ignore[arg-type]
    result = await plugin.get_messages_for_session("s1")

    assert result.messages[0].timestamp == ts_old
    assert result.messages[1].timestamp == ts_new


@pytest.mark.anyio
async def test_get_messages_for_session_query_scoped_to_session_id() -> None:
    """alist() must be called with the thread_id-keyed config and the cap limit."""
    from agent_engine_sdk_langgraph.query import _MAX_CHECKPOINTS_PER_SESSION

    saver = _StubSaver()
    saver.set_alist_result([])
    plugin = LangGraphQueryPlugin(saver)  # type: ignore[arg-type]

    await plugin.get_messages_for_session("session-x")

    assert len(saver.alist_calls) == 1
    call = saver.alist_calls[0]
    assert call["config"] == {"configurable": {"thread_id": "session-x"}}
    assert call["limit"] == _MAX_CHECKPOINTS_PER_SESSION


@pytest.mark.anyio
async def test_get_messages_for_session_stringifies_multimodal_content() -> None:
    """Multimodal content (list of blocks) collapses to text via
    BaseMessage.text — only ``type: "text"`` blocks contribute."""
    saver = _StubSaver()
    saver.set_alist_result(
        [
            {
                "ts": _iso(0),
                "channel_values": {
                    "messages": [
                        HumanMessage(
                            content=[
                                {"type": "text", "text": "first "},
                                {"type": "image", "url": "http://example/img"},
                                {"type": "text", "text": "and second"},
                            ],
                            id="h1",
                        )
                    ]
                },
            }
        ]
    )

    plugin = LangGraphQueryPlugin(saver)  # type: ignore[arg-type]
    result = await plugin.get_messages_for_session("s1")

    assert result.messages[0].content == "first and second"


@pytest.mark.anyio
async def test_messages_skip_checkpoint_with_non_list_messages() -> None:
    """When channel_values.messages is missing or not a list, the
    checkpoint is silently skipped rather than crashing.

    alist() handles deserialization errors itself; this test is only about
    type-shape resilience of the messages channel after a successful
    deserialization (a graph that doesn't use the ``messages`` channel at
    all should not crash the plugin)."""
    saver = _StubSaver()
    ts_good = _iso(30)
    # Newest-first: the good checkpoint comes first because we want to
    # observe that the plugin returns its message list rather than being
    # confused by the malformed ones beneath it.
    saver.set_alist_result(
        [
            {
                "ts": ts_good,
                "channel_values": {"messages": [HumanMessage(content="ok", id="h1")]},
            },
            {"ts": _iso(20), "channel_values": {"messages": None}},
            {"ts": _iso(10), "channel_values": {"messages": {"not": "a list"}}},
            {"ts": _iso(0), "channel_values": {}},  # messages key absent
        ]
    )

    plugin = LangGraphQueryPlugin(saver)  # type: ignore[arg-type]
    result = await plugin.get_messages_for_session("s1")

    assert len(result.messages) == 1
    assert result.messages[0].id == "h1"
    assert result.messages[0].timestamp == ts_good


@pytest.mark.anyio
async def test_get_messages_for_session_surfaces_tool_call_join_key() -> None:
    """An AI message's tool_calls (with ids) and each ToolMessage's
    tool_call_id are surfaced so a consumer can join a call to its result —
    including two calls to the same tool, which are distinguishable purely
    by id."""
    saver = _StubSaver()
    saver.set_alist_result(
        [
            {
                "ts": _iso(0),
                "channel_values": {
                    "messages": [
                        AIMessage(
                            content="",
                            id="a1",
                            tool_calls=[
                                {
                                    "id": "call_1",
                                    "name": "search",
                                    "args": {"q": "tokyo"},
                                    "type": "tool_call",
                                },
                                {
                                    "id": "call_2",
                                    "name": "search",
                                    "args": {"q": "kyoto"},
                                    "type": "tool_call",
                                },
                            ],
                        ),
                        ToolMessage(
                            content="result-1",
                            tool_call_id="call_1",
                            name="search",
                            id="t1",
                        ),
                        ToolMessage(
                            content="result-2",
                            tool_call_id="call_2",
                            name="search",
                            id="t2",
                        ),
                    ]
                },
            }
        ]
    )

    plugin = LangGraphQueryPlugin(saver)  # type: ignore[arg-type]
    result = await plugin.get_messages_for_session("s1")

    ai_message = result.messages[0]
    assert ai_message.tool_calls is not None
    assert [tc.id for tc in ai_message.tool_calls] == ["call_1", "call_2"]
    assert [tc.name for tc in ai_message.tool_calls] == ["search", "search"]
    assert ai_message.tool_calls[0].args == {"q": "tokyo"}
    assert ai_message.tool_call_id is None

    assert result.messages[1].tool_call_id == "call_1"
    assert result.messages[1].tool_calls is None
    assert result.messages[2].tool_call_id == "call_2"


@pytest.mark.anyio
async def test_get_messages_for_session_omits_tool_fields_for_plain_messages() -> None:
    """Messages with no tool calls carry neither field, so existing clients
    that ignore them see an unchanged shape."""
    saver = _StubSaver()
    saver.set_alist_result(
        [
            {
                "ts": _iso(0),
                "channel_values": {
                    "messages": [
                        HumanMessage(content="hi", id="h1"),
                        AIMessage(content="hello", id="a1"),
                    ]
                },
            }
        ]
    )

    plugin = LangGraphQueryPlugin(saver)  # type: ignore[arg-type]
    result = await plugin.get_messages_for_session("s1")

    assert all(m.tool_calls is None for m in result.messages)
    assert all(m.tool_call_id is None for m in result.messages)


# ---------------------------------------------------------------------------
# Workspace-scoped reads
#
# When the plugin is constructed with a workspace_id, reads must query the
# composite ``{session_id}:{workspace_id}`` thread_id (matching the write
# path) and strip the workspace back off before returning, so callers only
# ever see the plain session_id.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_messages_read_uses_workspace_scoped_thread_id() -> None:
    saver = _StubSaver()
    saver.set_alist_result([])
    plugin = LangGraphQueryPlugin(saver, workspace_id="ws-1")  # type: ignore[arg-type]

    await plugin.get_messages_for_session("session-x")

    # Only the scoped key is queried — the bare key is shared by every
    # workspace on the store.
    assert len(saver.alist_calls) == 1
    assert saver.alist_calls[0]["config"] == {
        "configurable": {"thread_id": "session-x:ws-1"}
    }


@pytest.mark.anyio
async def test_summaries_query_uses_composite_and_returns_plain_session_id() -> None:
    saver = _StubSaver()
    latest = datetime(2026, 5, 22, 10, 0, 0, tzinfo=timezone.utc)
    first = datetime(2026, 5, 22, 9, 0, 0, tzinfo=timezone.utc)
    # The aggregation groups by thread_id, so ``_id`` comes back as the
    # composite key; the plugin must strip ``:ws-1`` before returning.
    saver.checkpoint_collection.set_aggregate_result(
        [
            {
                "_id": "session-a:ws-1",
                "latest_ts": latest,
                "first_ts": first,
            }
        ]
    )
    saver.writes_collection.set_aggregate_result([])
    plugin = LangGraphQueryPlugin(saver, workspace_id="ws-1")  # type: ignore[arg-type]

    result = await plugin.get_summaries_for_sessions(["session-a"])

    # The $match must target only the composite key.
    match_stage = saver.checkpoint_collection.aggregate_calls[0]["pipeline"][0]
    assert match_stage == {"$match": {"thread_id": {"$in": ["session-a:ws-1"]}}}
    # ...but the caller sees only the plain session_id.
    assert len(result.sessions) == 1
    assert result.sessions[0].session_id == "session-a"


@pytest.mark.anyio
async def test_preview_pipeline_uses_composite_thread_id() -> None:
    saver = _StubSaver()
    saver.checkpoint_collection.set_aggregate_result(
        [
            {
                "_id": "session-a:ws-1",
                "latest_ts": datetime(2026, 5, 22, 10, 0, 0, tzinfo=timezone.utc),
                "first_ts": datetime(2026, 5, 22, 9, 0, 0, tzinfo=timezone.utc),
            }
        ]
    )
    saver.writes_collection.set_aggregate_result([])
    plugin = LangGraphQueryPlugin(saver, workspace_id="ws-1")  # type: ignore[arg-type]

    await plugin.get_summaries_for_sessions(["session-a"])

    match_stage = saver.writes_collection.aggregate_calls[0]["pipeline"][0]
    assert match_stage["$match"]["thread_id"] == {"$in": ["session-a:ws-1"]}


@pytest.mark.anyio
async def test_messages_read_does_not_fall_back_to_unscoped_thread_id() -> None:
    """Checkpoints under the bare (unscoped) session key are NOT served once a
    workspace scope exists — that key is shared across tenants."""
    saver = _StubSaver()
    ts = _iso(0)
    saver.set_alist_results_by_thread_id(
        {
            "session-x": [
                {
                    "ts": ts,
                    "channel_values": {
                        "messages": [HumanMessage(content="legacy", id="h1")]
                    },
                }
            ],
            "session-x:ws-1": [],
        }
    )
    plugin = LangGraphQueryPlugin(saver, workspace_id="ws-1")  # type: ignore[arg-type]

    result = await plugin.get_messages_for_session("session-x")

    assert result.messages == []
    assert [c["config"] for c in saver.alist_calls] == [
        {"configurable": {"thread_id": "session-x:ws-1"}}
    ]


@pytest.mark.anyio
async def test_summaries_merge_legacy_and_scoped_rows() -> None:
    saver = _StubSaver()
    legacy_latest = datetime(2026, 5, 22, 9, 30, 0, tzinfo=timezone.utc)
    legacy_first = datetime(2026, 5, 22, 9, 0, 0, tzinfo=timezone.utc)
    scoped_latest = datetime(2026, 5, 22, 10, 0, 0, tzinfo=timezone.utc)
    scoped_first = datetime(2026, 5, 22, 9, 45, 0, tzinfo=timezone.utc)
    saver.checkpoint_collection.set_aggregate_result(
        [
            {
                "_id": "session-a",
                "latest_ts": legacy_latest,
                "first_ts": legacy_first,
            },
            {
                "_id": "session-a:ws-1",
                "latest_ts": scoped_latest,
                "first_ts": scoped_first,
            },
        ]
    )
    saver.writes_collection.set_aggregate_result([])
    saver.set_alist_results_by_thread_id(
        {
            "session-a": [
                {
                    "ts": _iso(0),
                    "channel_values": {
                        "messages": [HumanMessage(content="legacy", id="h1")]
                    },
                }
            ],
            "session-a:ws-1": [
                {
                    "ts": _iso(2),
                    "channel_values": {
                        "messages": [
                            HumanMessage(content="scoped", id="h2"),
                            AIMessage(content="reply", id="a1"),
                        ]
                    },
                }
            ],
        }
    )
    plugin = LangGraphQueryPlugin(saver, workspace_id="ws-1")  # type: ignore[arg-type]

    result = await plugin.get_summaries_for_sessions(["session-a"])

    assert len(result.sessions) == 1
    summary = result.sessions[0]
    assert summary.session_id == "session-a"
    assert summary.last_activity == scoped_latest.isoformat()
    assert summary.created_at == legacy_first.isoformat()
    # Newest checkpoint across both thread keys wins, and its message list —
    # not the checkpoint total — is the count.
    assert summary.message_count == 2


@pytest.mark.anyio
async def test_summary_message_count_matches_messages_endpoint() -> None:
    """A tool-calling turn writes several checkpoints but yields four messages."""
    saver = _StubSaver()
    messages = [
        HumanMessage(content="what is 2+2?", id="h1"),
        AIMessage(
            content="",
            id="a1",
            tool_calls=[{"name": "calc", "args": {"x": 2}, "id": "tc1"}],
        ),
        ToolMessage(content="4", id="t1", tool_call_id="tc1"),
        AIMessage(content="4", id="a2"),
    ]
    saver.checkpoint_collection.set_aggregate_result(
        [
            {
                "_id": "session-a",
                "latest_ts": datetime(2026, 5, 22, 10, 0, 0, tzinfo=timezone.utc),
                "first_ts": datetime(2026, 5, 22, 9, 0, 0, tzinfo=timezone.utc),
            }
        ]
    )
    saver.writes_collection.set_aggregate_result([])
    # Six checkpoints (LangGraph writes one per graph super-step) carrying a
    # message list that grows to four — newest first.
    prefix_lengths = [4, 4, 3, 2, 1, 1]
    saver.set_alist_result(
        [
            {
                "ts": _iso(len(prefix_lengths) - index),
                "channel_values": {"messages": messages[:prefix_length]},
            }
            for index, prefix_length in enumerate(prefix_lengths)
        ]
    )
    plugin = LangGraphQueryPlugin(saver)  # type: ignore[arg-type]

    summaries = await plugin.get_summaries_for_sessions(["session-a"])
    endpoint_messages = await plugin.get_messages_for_session("session-a")

    assert summaries.sessions[0].message_count == len(endpoint_messages.messages) == 4


@pytest.mark.anyio
async def test_summary_message_count_degrades_to_zero_on_unreadable_session() -> None:
    """One unreadable session must not fail the whole summaries request."""
    saver = _StubSaver()
    latest = datetime(2026, 5, 22, 10, 0, 0, tzinfo=timezone.utc)
    saver.checkpoint_collection.set_aggregate_result(
        [
            {"_id": "bad-session", "latest_ts": latest, "first_ts": latest},
            {"_id": "good-session", "latest_ts": latest, "first_ts": latest},
        ]
    )
    saver.writes_collection.set_aggregate_result([])
    saver.set_alist_results_by_thread_id(
        {
            "good-session": [
                {
                    "ts": _iso(0),
                    "channel_values": {
                        "messages": [HumanMessage(content="hi", id="h1")]
                    },
                }
            ]
        }
    )
    original_alist = saver.alist

    def failing_alist(config: dict[str, Any], **kwargs: Any):
        if config.get("configurable", {}).get("thread_id") == "bad-session":
            raise ValueError("corrupt checkpoint")
        return original_alist(config, **kwargs)

    saver.alist = failing_alist  # type: ignore[method-assign]
    plugin = LangGraphQueryPlugin(saver)  # type: ignore[arg-type]

    result = await plugin.get_summaries_for_sessions(["bad-session", "good-session"])

    counts = {s.session_id: s.message_count for s in result.sessions}
    assert counts == {"bad-session": 0, "good-session": 1}


@pytest.mark.anyio
async def test_query_plugin_uses_workspace_id_resolver() -> None:
    saver = _StubSaver()
    saver.set_alist_result([])
    resolver_calls = 0

    def resolver() -> str:
        nonlocal resolver_calls
        resolver_calls += 1
        return "ws-dynamic"

    plugin = LangGraphQueryPlugin(
        saver,  # type: ignore[arg-type]
        workspace_id_resolver=resolver,
    )

    await plugin.get_messages_for_session("session-x")

    assert resolver_calls == 1
    assert saver.alist_calls[0]["config"] == {
        "configurable": {"thread_id": "session-x:ws-dynamic"}
    }


@pytest.mark.anyio
async def test_query_plugin_fails_closed_when_resolver_has_no_workspace() -> None:
    """A resolver returning None signals a genuinely unresolvable scope (a
    custom resolver that cannot decide) and fails closed."""
    saver = _StubSaver()
    saver.set_alist_result([])

    plugin = LangGraphQueryPlugin(
        saver,  # type: ignore[arg-type]
        workspace_id_resolver=lambda: None,
    )

    with pytest.raises(RuntimeError, match="workspace scope unavailable"):
        await plugin.get_messages_for_session("session-x")

    assert saver.alist_calls == []


@pytest.mark.anyio
async def test_query_plugin_allows_explicitly_unscoped_runtime() -> None:
    """An empty scope from the runtime resolver is legitimate: no APP_ID means
    local dev / tests, where writes are bare-keyed and reads must match — the
    normal `agentengine dev up` history path."""
    saver = _StubSaver()
    saver.set_alist_result([])

    plugin = LangGraphQueryPlugin(
        saver,  # type: ignore[arg-type]
        workspace_id_resolver=lambda: "",
    )

    result = await plugin.get_messages_for_session("session-x")

    assert result.messages == []
    assert [c["config"] for c in saver.alist_calls] == [
        {"configurable": {"thread_id": "session-x"}}
    ]


def test_history_human_role_matches_live_message_vocabulary() -> None:
    """Session history and live invoke must emit the same role for HumanMessage."""
    from agent_engine_sdk_langgraph.messages import lc_to_platform_message
    from agent_engine_sdk_langgraph.query import _ROLE_MAP

    assert _ROLE_MAP["human"] == lc_to_platform_message(HumanMessage(content="x")).role
