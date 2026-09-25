"""Unit tests for ADK suspend translation."""

from __future__ import annotations

import pytest
from agent_engine_sdk import Message
from google.adk.events import Event
from google.adk.events.event import NodeInfo
from google.genai import types

from agent_engine_sdk_adk.suspend import (
    SuspendTranslationError,
    extract_suspends,
    frontier_resume_schema,
    function_response_frontier_content,
    suspend_frontier_stream_event,
)


def _function_call_event(
    *,
    name: str,
    call_id: str,
    args: dict[str, object],
    long_running: bool = False,
) -> Event:
    return Event(
        author="agent",
        long_running_tool_ids=[call_id] if long_running else [],
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(id=call_id, name=name, args=args)
                )
            ],
        ),
    )


def test_request_input_preserves_message_payload_and_schema() -> None:
    event = _function_call_event(
        name="adk_request_input",
        call_id="manager_approval",
        args={
            "message": "Please review this time off request.",
            "payload": {"days": 5, "employee": "alex"},
            "response_schema": {
                "type": "object",
                "properties": {"approved": {"type": "boolean"}},
            },
        },
        long_running=True,
    )

    [suspend] = extract_suspends(event)
    assert suspend.function_call_id == "manager_approval"
    assert suspend.value == {
        "message": "Please review this time off request.",
        "payload": {"days": 5, "employee": "alex"},
        "response_schema": {
            "type": "object",
            "properties": {"approved": {"type": "boolean"}},
        },
    }
    assert "interrupt_id" not in suspend.semantic_input
    assert suspend.semantic_input["kind"] == "adk_request_input"


def test_confirmation_preserves_hint_payload_and_tool_call() -> None:
    event = _function_call_event(
        name="adk_request_confirmation",
        call_id="call_book_flight",
        args={
            "originalFunctionCall": {
                "id": "tool-1",
                "name": "book_flight",
                "args": {"origin": "SEA", "destination": "JFK"},
            },
            "toolConfirmation": {
                "hint": "Book SEA → JFK?",
                "payload": {"fare": 842},
            },
        },
        long_running=True,
    )

    [suspend] = extract_suspends(event)
    assert suspend.value == {
        "hint": "Book SEA → JFK?",
        "payload": {"fare": 842},
        "tool_call": {
            "name": "book_flight",
            "args": {"origin": "SEA", "destination": "JFK"},
        },
        "response_schema": {
            "type": "object",
            "required": ["confirmed"],
            "properties": {"confirmed": {"type": "boolean"}},
        },
    }
    assert "tool-1" not in suspend.semantic_input.values()
    assert suspend.semantic_input["tool_name"] == "book_flight"


def test_resume_schema_requires_resume_map_for_the_interrupt_id() -> None:
    request_input = frontier_resume_schema(["manager_approval"], ["adk_request_input"])
    confirmation = frontier_resume_schema(["call_bind"], ["adk_request_confirmation"])

    assert request_input["required"] == ["resume_map"]
    assert request_input["properties"]["resume_map"]["required"] == ["manager_approval"]
    assert request_input["properties"]["resume_map"]["additionalProperties"] is False
    assert request_input["properties"]["resume_map"]["properties"][
        "manager_approval"
    ] == {"not": {"type": "null"}}
    assert confirmation["properties"]["resume_map"]["properties"]["call_bind"] == {
        "type": "object",
        "required": ["confirmed"],
        "properties": {"confirmed": {"type": "boolean"}},
    }


def test_structured_request_input_response_is_unwrapped() -> None:
    event = _function_call_event(
        name="adk_request_input",
        call_id="manager_approval",
        args={"message": "review"},
    )
    [suspend] = extract_suspends(event)

    content = function_response_frontier_content([suspend], [{"approved": True}])

    response = content.parts[0].function_response
    assert response is not None
    assert response.id == "manager_approval"
    assert response.name == "adk_request_input"
    assert response.response == {"approved": True}


def test_scalar_request_input_response_is_wrapped() -> None:
    event = _function_call_event(
        name="adk_request_input",
        call_id="ask",
        args={"message": "name?"},
    )
    [suspend] = extract_suspends(event)

    content = function_response_frontier_content([suspend], ["Ada"])

    assert content.parts[0].function_response.response == {"result": "Ada"}


def test_confirmation_approve_and_reject_keep_adk_shape() -> None:
    event = _function_call_event(
        name="adk_request_confirmation",
        call_id="call_1",
        args={
            "originalFunctionCall": {"name": "charge", "args": {"amount": 10}},
            "toolConfirmation": {"hint": "Charge $10?"},
        },
    )
    [suspend] = extract_suspends(event)

    approved = function_response_frontier_content(
        [suspend], [{"confirmed": True, "payload": {"note": "ok"}}]
    )
    rejected = function_response_frontier_content([suspend], [{"confirmed": False}])

    assert approved.parts[0].function_response.response == {
        "confirmed": True,
        "payload": {"note": "ok"},
    }
    assert rejected.parts[0].function_response.response == {
        "confirmed": False,
        "payload": None,
    }


def test_suspend_event_includes_interrupt_envelope_and_legacy_payload() -> None:
    event = _function_call_event(
        name="adk_request_input",
        call_id="manager_approval",
        args={"message": "review"},
    )
    [suspend] = extract_suspends(event)

    streamed = suspend_frontier_stream_event(
        [suspend],
        interrupt_ids=["activity-1"],
        messages=[Message(role="assistant", content="prefix")],
        metadata={},
    )

    assert streamed.event == "suspend"
    assert streamed.data["suspend_payload"] == suspend.value
    assert streamed.data["interrupts"] == [{"id": "activity-1", "value": suspend.value}]
    assert streamed.data["resume_schema"]["required"] == ["resume_map"]
    assert streamed.data["resume_schema"]["properties"]["resume_map"]["required"] == [
        "activity-1"
    ]


def test_multiple_suspend_calls_form_one_ordered_frontier() -> None:
    event = Event(
        author="agent",
        branch="left@1",
        node_info=NodeInfo(path="parallel.left"),
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="Two reviews are required."),
                types.Part(
                    function_call=types.FunctionCall(
                        id="a", name="adk_request_input", args={}
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        id="b", name="adk_request_input", args={}
                    )
                ),
            ],
        ),
    )

    suspends = extract_suspends(event)

    assert [suspend.function_call_id for suspend in suspends] == ["a", "b"]
    assert [suspend.provenance.part_index for suspend in suspends] == [1, 2]
    assert suspends[0].provenance.order_key == (
        "parallel.left",
        "left@1",
        1,
    )


def test_named_wait_and_domain_long_running_share_one_frontier() -> None:
    event = Event(
        author="agent",
        long_running_tool_ids=["review-1"],
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="confirm-1",
                        name="adk_request_confirmation",
                        args={
                            "originalFunctionCall": {
                                "name": "create_policy",
                                "args": {},
                            }
                        },
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        id="review-1",
                        name="human_review",
                        args={"claim_id": "CLM-123"},
                    )
                ),
            ],
        ),
    )

    suspends = extract_suspends(event)

    assert [suspend.function_call_id for suspend in suspends] == [
        "confirm-1",
        "review-1",
    ]
    assert [suspend.kind for suspend in suspends] == [
        "adk_request_confirmation",
        "adk_request_input",
    ]


def test_wait_mixed_with_ordinary_tool_call_extracts_only_the_wait() -> None:
    named_wait = Event(
        author="agent",
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="a",
                        name="adk_request_input",
                        args={"message": "review"},
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        id="b", name="get_quote", args={"city": "Paris"}
                    )
                ),
            ],
        ),
    )
    domain_wait = Event(
        author="agent",
        long_running_tool_ids=["review-1"],
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="review-1",
                        name="human_review",
                        args={"claim_id": "CLM-123"},
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        id="quote-1", name="get_quote", args={"city": "Paris"}
                    )
                ),
            ],
        ),
    )

    assert [suspend.function_call_id for suspend in extract_suspends(named_wait)] == [
        "a"
    ]
    assert [suspend.function_call_id for suspend in extract_suspends(domain_wait)] == [
        "review-1"
    ]


def test_frontier_event_and_continue_require_every_exact_activity_id() -> None:
    event = Event(
        author="agent",
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="left", name="adk_request_input", args={}
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        id="right", name="adk_request_input", args={}
                    )
                ),
            ],
        ),
    )
    suspends = extract_suspends(event)

    streamed = suspend_frontier_stream_event(
        suspends,
        interrupt_ids=["activity-left", "activity-right"],
        messages=[],
    )
    schema = frontier_resume_schema(
        ["activity-left", "activity-right"],
        ["adk_request_input", "adk_request_input"],
    )
    content = function_response_frontier_content(
        suspends,
        [{"answer": "L"}, {"answer": "R"}],
    )

    assert [item["id"] for item in streamed.data["interrupts"]] == [
        "activity-left",
        "activity-right",
    ]
    assert schema["properties"]["resume_map"]["required"] == [
        "activity-left",
        "activity-right",
    ]
    assert schema["properties"]["resume_map"]["additionalProperties"] is False
    assert [part.function_response.id for part in content.parts] == ["left", "right"]


def test_credential_request_fails() -> None:
    event = _function_call_event(
        name="adk_request_credential",
        call_id="auth-1",
        args={},
        long_running=True,
    )

    with pytest.raises(SuspendTranslationError, match="credential"):
        extract_suspends(event)


def test_domain_long_running_tool_is_request_input() -> None:
    event = _function_call_event(
        name="human_review",
        call_id="review-1",
        args={
            "claim_id": "CLM-123",
            "decision": "approve",
            "reason": "High claim amount requires human approval",
            "claim_amount": 12000,
            "risk_level": "high",
            "conversation_summary": "Rear-end collision; policy POL-1.",
        },
        long_running=True,
    )

    [suspend] = extract_suspends(event)
    assert suspend.kind == "adk_request_input"
    assert suspend.function_name == "human_review"
    assert suspend.value["message"] == "High claim amount requires human approval"
    assert suspend.value["payload"]["claim_id"] == "CLM-123"

    content = function_response_frontier_content(
        [suspend], [{"decision": "approved", "reviewer_notes": "ok"}]
    )
    response = content.parts[0].function_response
    assert response is not None
    assert response.name == "human_review"
    assert response.response == {"decision": "approved", "reviewer_notes": "ok"}


def test_malformed_confirmation_fails() -> None:
    event = _function_call_event(
        name="adk_request_confirmation",
        call_id="call-1",
        args={"hint": "ok?"},
    )

    with pytest.raises(SuspendTranslationError, match="originalFunctionCall"):
        extract_suspends(event)


def test_malformed_confirmation_result_fails() -> None:
    event = _function_call_event(
        name="adk_request_confirmation",
        call_id="call-1",
        args={"originalFunctionCall": {"name": "charge", "args": {}}},
    )
    [suspend] = extract_suspends(event)

    with pytest.raises(SuspendTranslationError, match="confirmed"):
        function_response_frontier_content([suspend], ["yes"])
