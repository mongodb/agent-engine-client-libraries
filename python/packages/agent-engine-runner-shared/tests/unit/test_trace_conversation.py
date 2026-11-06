"""Input-message trimming: only the messages a call added are stored."""

from __future__ import annotations

from unittest.mock import MagicMock

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode, TraceFlags

from agent_engine_runner_shared.tracing.conversation import (
    ATTR_INPUT_MESSAGES_OFFSET,
    ATTR_INPUT_MESSAGES_RECORDED_BY,
    ATTR_INPUT_MESSAGES_STREAM,
    ATTR_INPUT_MESSAGES_TOTAL,
    ATTR_RUNTIME_MODE,
    RECORDED_BY_CALLER,
    ConversationDeltaTracker,
    caller_records_conversation,
    mark_recorded_by_caller,
)
from agent_engine_runner_shared.tracing.exporters import MongoDBSpanExporter

TRACE_ID = "12345678123456781234567812345678"


def _messages(*contents: str) -> dict[str, object]:
    attributes: dict[str, object] = {}
    for index, content in enumerate(contents):
        attributes[f"llm.input_messages.{index}.message.role"] = "user"
        attributes[f"llm.input_messages.{index}.message.content"] = content
    return attributes


def _stored_contents(attributes: dict[str, object]) -> list[str]:
    contents = [
        (key, value)
        for key, value in attributes.items()
        if key.startswith("llm.input_messages.") and key.endswith(".message.content")
    ]
    return [str(value) for _, value in sorted(contents)]


def test_first_call_keeps_the_whole_conversation():
    tracker = ConversationDeltaTracker()

    trimmed = tracker.trim(_messages("hi"), trace_id=TRACE_ID, stream_id="aer")

    assert _stored_contents(trimmed) == ["hi"]
    assert trimmed[ATTR_INPUT_MESSAGES_OFFSET] == 0
    assert trimmed[ATTR_INPUT_MESSAGES_TOTAL] == 1
    assert trimmed[ATTR_INPUT_MESSAGES_STREAM] == "aer"


def test_later_call_keeps_only_the_new_messages():
    tracker = ConversationDeltaTracker()
    tracker.trim(_messages("hi"), trace_id=TRACE_ID, stream_id="aer")

    trimmed = tracker.trim(_messages("hi", "there", "again"), trace_id=TRACE_ID, stream_id="aer")

    assert _stored_contents(trimmed) == ["there", "again"]
    assert trimmed[ATTR_INPUT_MESSAGES_OFFSET] == 1
    assert trimmed[ATTR_INPUT_MESSAGES_TOTAL] == 3
    # Renumbered so the reader sees a list starting at message zero.
    assert trimmed["llm.input_messages.0.message.content"] == "there"


def test_repeated_call_stores_no_messages():
    tracker = ConversationDeltaTracker()
    tracker.trim(_messages("hi", "there"), trace_id=TRACE_ID, stream_id="aer")

    trimmed = tracker.trim(_messages("hi", "there"), trace_id=TRACE_ID, stream_id="aer")

    assert _stored_contents(trimmed) == []
    assert trimmed[ATTR_INPUT_MESSAGES_OFFSET] == 2
    assert trimmed[ATTR_INPUT_MESSAGES_TOTAL] == 2


def test_rewritten_history_stores_the_full_list():
    tracker = ConversationDeltaTracker()
    tracker.trim(_messages("hi", "there"), trace_id=TRACE_ID, stream_id="aer")

    trimmed = tracker.trim(_messages("hi", "different"), trace_id=TRACE_ID, stream_id="aer")

    assert _stored_contents(trimmed) == ["hi", "different"]
    assert trimmed[ATTR_INPUT_MESSAGES_OFFSET] == 0


def test_shorter_history_stores_the_full_list_and_resets():
    tracker = ConversationDeltaTracker()
    tracker.trim(_messages("hi", "there"), trace_id=TRACE_ID, stream_id="aer")

    trimmed = tracker.trim(_messages("hi"), trace_id=TRACE_ID, stream_id="aer")

    assert _stored_contents(trimmed) == ["hi"]
    assert trimmed[ATTR_INPUT_MESSAGES_OFFSET] == 0


def test_streams_do_not_share_progress():
    tracker = ConversationDeltaTracker()
    tracker.trim(_messages("hi"), trace_id=TRACE_ID, stream_id="aer")

    tool_pod = tracker.trim(_messages("hi"), trace_id=TRACE_ID, stream_id="tool")

    assert _stored_contents(tool_pod) == ["hi"]
    assert tool_pod[ATTR_INPUT_MESSAGES_OFFSET] == 0


def test_traces_do_not_share_progress():
    tracker = ConversationDeltaTracker()
    tracker.trim(_messages("hi"), trace_id=TRACE_ID, stream_id="aer")

    other_trace = tracker.trim(_messages("hi", "there"), trace_id="f" * 32, stream_id="aer")

    assert _stored_contents(other_trace) == ["hi", "there"]
    assert other_trace[ATTR_INPUT_MESSAGES_OFFSET] == 0


def test_spans_without_messages_are_untouched():
    tracker = ConversationDeltaTracker()
    attributes = {"tool.parameters": "{}"}

    assert tracker.trim(attributes, trace_id=TRACE_ID, stream_id="aer") == attributes


def test_the_serialized_prompt_is_dropped_alongside_the_messages():
    """OpenInference also writes the whole prompt to input.value."""
    tracker = ConversationDeltaTracker()
    attributes = {
        **_messages("hi", "there"),
        "input.value": '{"messages": [["user", "hi"], ["user", "there"]]}',
        "input.mime_type": "application/json",
    }

    trimmed = tracker.trim(attributes, trace_id=TRACE_ID, stream_id="aer")

    assert "input.value" not in trimmed
    assert "input.mime_type" not in trimmed
    assert _stored_contents(trimmed) == ["hi", "there"]


def test_copy_is_independent_of_the_original():
    tracker = ConversationDeltaTracker()
    tracker.trim(_messages("hi"), trace_id=TRACE_ID, stream_id="aer")

    staged = tracker.copy()
    staged.trim(_messages("hi", "there"), trace_id=TRACE_ID, stream_id="aer")

    # The staged write is invisible to the original, which has only stored
    # "hi", so the same prompt still reports one message of progress.
    after = tracker.trim(_messages("hi", "there"), trace_id=TRACE_ID, stream_id="aer")
    assert after[ATTR_INPUT_MESSAGES_OFFSET] == 1


def test_tracker_bounds_the_number_of_conversations():
    tracker = ConversationDeltaTracker(max_conversations=2)
    for index in range(3):
        tracker.trim(_messages("hi"), trace_id=f"{index:032x}", stream_id="aer")

    resent = tracker.trim(_messages("hi"), trace_id="0" * 32, stream_id="aer")
    assert resent[ATTR_INPUT_MESSAGES_OFFSET] == 0


def _span(attributes: dict[str, object]) -> MagicMock:
    context = SpanContext(
        trace_id=int(TRACE_ID, 16),
        span_id=0x1234567812345678,
        is_remote=False,
        trace_flags=TraceFlags.SAMPLED,
    )
    span = MagicMock(spec=ReadableSpan)
    span.get_span_context.return_value = context
    span.parent = None
    span.name = "ChatOpenAI"
    span.kind = SpanKind.INTERNAL
    span.start_time = 1000
    span.end_time = 2000
    span.status = Status(StatusCode.OK)
    span.attributes = attributes
    span.resource.attributes = {
        "service.name": "my-agent",
        ATTR_RUNTIME_MODE: "aer",
    }
    span.events = []
    return span


def test_mongodb_exporter_trims_each_written_span():
    collection = MagicMock()
    exporter = MongoDBSpanExporter(collection)

    exporter.export([_span(_messages("hi"))])
    exporter.export([_span(_messages("hi", "there"))])

    first, second = collection.insert_many.call_args_list
    first_attrs = first[0][0][0]["attributes"]
    second_attrs = second[0][0][0]["attributes"]
    assert _stored_contents(first_attrs) == ["hi"]
    assert _stored_contents(second_attrs) == ["there"]
    assert second_attrs[ATTR_INPUT_MESSAGES_OFFSET] == 1


def test_mongodb_exporter_does_not_advance_progress_when_the_write_fails():
    """A dropped batch must not make later spans omit messages it never stored."""
    collection = MagicMock()
    collection.insert_many.side_effect = [Exception("Connection failed"), None]
    exporter = MongoDBSpanExporter(collection)

    assert exporter.export([_span(_messages("hi"))]).name == "FAILURE"
    assert exporter.export([_span(_messages("hi", "again"))]).name == "SUCCESS"

    retried_attrs = collection.insert_many.call_args[0][0][0]["attributes"]
    # The failed span never landed, so the retry has to carry the whole prompt.
    assert _stored_contents(retried_attrs) == ["hi", "again"]
    assert retried_attrs[ATTR_INPUT_MESSAGES_OFFSET] == 0


def _caller_recorded(attributes: dict[str, object]) -> dict[str, object]:
    return {**attributes, ATTR_INPUT_MESSAGES_RECORDED_BY: RECORDED_BY_CALLER}


def test_mongodb_exporter_drops_a_prompt_the_caller_records():
    """The agent's span already stores it; the tool-pod copy is redundant."""
    collection = MagicMock()
    exporter = MongoDBSpanExporter(collection)
    attributes = _caller_recorded(
        {
            **_messages("hi", "there"),
            "input.value": '{"messages": [["user", "hi"]]}',
            "input.mime_type": "application/json",
        }
    )

    exporter.export([_span(attributes)])

    stored = collection.insert_many.call_args[0][0][0]["attributes"]
    assert _stored_contents(stored) == []
    assert "input.value" not in stored
    assert "input.mime_type" not in stored
    # The marker outlives the prompt: it is how a reader finds the stored copy.
    assert stored[ATTR_INPUT_MESSAGES_RECORDED_BY] == RECORDED_BY_CALLER
    # No offset is written, so a reader does not replay this span as a slice.
    assert ATTR_INPUT_MESSAGES_OFFSET not in stored


def test_a_dropped_prompt_does_not_advance_the_tracked_conversation():
    """A later tool-pod span that records its own prompt starts from the whole list."""
    collection = MagicMock()
    exporter = MongoDBSpanExporter(collection)

    exporter.export([_span(_caller_recorded(_messages("hi", "there")))])
    exporter.export([_span(_messages("asked a provider directly"))])

    direct = collection.insert_many.call_args[0][0][0]["attributes"]
    assert _stored_contents(direct) == ["asked a provider directly"]
    assert direct[ATTR_INPUT_MESSAGES_OFFSET] == 0


def test_mark_recorded_by_caller_marks_only_inside_the_context():
    outside = MagicMock()
    mark_recorded_by_caller(outside)
    outside.set_attribute.assert_not_called()

    inside = MagicMock()
    with caller_records_conversation():
        mark_recorded_by_caller(inside)

    inside.set_attribute.assert_called_once_with(
        ATTR_INPUT_MESSAGES_RECORDED_BY, RECORDED_BY_CALLER
    )


def test_the_caller_recorded_marker_ends_with_the_context():
    span = MagicMock()
    with caller_records_conversation():
        pass
    mark_recorded_by_caller(span)

    span.set_attribute.assert_not_called()
