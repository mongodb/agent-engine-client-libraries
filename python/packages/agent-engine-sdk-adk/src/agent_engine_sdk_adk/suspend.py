"""Translate ADK input and confirmation into Atlas Agent Engine suspend/resume."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from agent_engine_sdk import AgentOutput, Message, StreamEvent
from google.adk.events import Event
from google.genai import types
from pydantic import BaseModel, Field, JsonValue

from agent_engine_runner_shared.workflow import ChildOperationBoundary

__all__ = [
    "REQUEST_CONFIRMATION_FUNCTION_CALL_NAME",
    "REQUEST_CREDENTIAL_FUNCTION_CALL_NAME",
    "REQUEST_INPUT_FUNCTION_CALL_NAME",
    "AdkSuspend",
    "CompletedAgentResponse",
    "SuspendEventData",
    "SuspendInterrupt",
    "SuspendTranslationError",
    "SuspendedAgentResponse",
    "extract_suspends",
    "frontier_resume_schema",
    "function_response_frontier_content",
    "invoke_output",
    "suspend_frontier_stream_event",
]

REQUEST_INPUT_FUNCTION_CALL_NAME = "adk_request_input"
REQUEST_CONFIRMATION_FUNCTION_CALL_NAME = "adk_request_confirmation"
REQUEST_CREDENTIAL_FUNCTION_CALL_NAME = "adk_request_credential"

_SUSPEND_FUNCTION_NAMES = frozenset(
    {
        REQUEST_INPUT_FUNCTION_CALL_NAME,
        REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
    }
)
_RESULT_KEY = "result"


class SuspendTranslationError(ValueError):
    """An ADK event cannot become a deterministic Atlas Agent Engine wait frontier."""


class SuspendInterrupt(BaseModel):
    """One Atlas Agent Engine interrupt in a suspend envelope."""

    id: str = Field(description="Caller-facing interrupt id for resume_map.")
    value: dict[str, Any] = Field(description="Value shown for this interrupt.")


class SuspendEventData(BaseModel):
    """Payload of Atlas Agent Engine ``StreamEvent(event='suspend')``."""

    suspend_payload: dict[str, Any] = Field(
        description="Legacy single-wait payload; matches interrupts[0].value."
    )
    resumed: Literal[False] = False
    messages: list[Message] = Field(description="Visible messages before the wait.")
    metadata: dict[str, Any] = Field(default_factory=dict)
    interrupts: list[SuspendInterrupt] = Field(
        description="Complete wait frontier for this durable execution boundary."
    )
    resume_schema: dict[str, Any] = Field(
        description="JSON Schema requiring resume_map for the interrupt id."
    )
    response: str = Field(
        default="",
        description="Non-partial model text emitted before the wait.",
    )


class SuspendedAgentResponse(BaseModel):
    """``AgentOutput.response`` when the invocation suspended."""

    status: Literal["suspended"] = "suspended"
    response: str = Field(description="Text emitted before the wait, if any.")
    resumed: Literal[False] = False
    interrupts: list[SuspendInterrupt]
    resume_schema: dict[str, Any]
    suspend_payload: dict[str, Any]


class CompletedAgentResponse(BaseModel):
    """``AgentOutput.response`` when the invocation completed."""

    status: Literal["completed"] = "completed"
    response: str
    resumed: Literal[False] = False


@dataclass(frozen=True)
class SuspendProvenance:
    """Stable ADK event coordinates used to order one wait frontier.

    ``part_index`` is the zero-based position in the original ADK
    ``event.content.parts`` list, including non-function-call parts. For an
    event on node ``parallel.left`` and branch ``left@1`` whose parts are
    ``[text, request_input]``, the wait provenance is::

        SuspendProvenance(
            node_path="parallel.left",
            branch="left@1",
            part_index=1,
        )

    The frontier order key is therefore
    ``("parallel.left", "left@1", 1)``.
    """

    node_path: str
    branch: str
    part_index: int
    operation_boundaries: tuple[ChildOperationBoundary, ...] = ()

    @property
    def order_key(self) -> tuple[str, str, int]:
        return (self.node_path, self.branch, self.part_index)


@dataclass(frozen=True)
class AdkSuspend:
    """One ADK wait with native identity, provenance, and OE activity input.

    ``function_call_id`` is ADK's native ``FunctionCall.id``. It is retained
    only to correlate the eventual ``FunctionResponse`` back to that ADK call.
    The durable caller-facing interrupt id is the separate OE activity id.
    """

    kind: Literal["adk_request_input", "adk_request_confirmation"]
    function_call_id: str
    value: dict[str, Any]
    semantic_input: dict[str, Any]
    function_name: str
    provenance: SuspendProvenance


def extract_suspends(
    event: Event,
    *,
    operation_boundaries: Sequence[ChildOperationBoundary] = (),
) -> list[AdkSuspend]:
    """Return every wait in one ADK event in stable content-part order.

    This function does not assume ADK emits one final event containing every
    parallel interrupt. Each route can emit its own event. ``AdkTurn`` collects
    these per-event lists until ``Runner.run_async`` becomes quiescent, then
    finalizes the accumulated list as one wait frontier. A serial wait follows
    the exact same path with a frontier of length one.

    Builtin waits are ``adk_request_input`` and ``adk_request_confirmation``.
    A domain ``LongRunningFunctionTool`` call is the same RequestInput wait:
    OE records kind ``adk_request_input``; continue uses a FunctionResponse
    whose name matches the tool the model called.
    """
    indexed_calls = _indexed_function_calls(event)
    function_calls = [call for _, call in indexed_calls]
    if any(
        call.name == REQUEST_CREDENTIAL_FUNCTION_CALL_NAME for call in function_calls
    ):
        raise SuspendTranslationError(
            "Google ADK credential requests are not supported by the durable runtime"
        )
    long_running_ids = set(event.long_running_tool_ids or [])

    waits: list[AdkSuspend] = []
    for part_index, call in indexed_calls:
        if call.name not in _SUSPEND_FUNCTION_NAMES and call.id not in long_running_ids:
            continue
        # The GenAI wire type makes this optional, but ADK assigns missing IDs
        # while finalizing model and workflow HITL events before yielding them.
        function_call_id = cast(str, call.id)
        provenance = SuspendProvenance(
            node_path=event.node_info.path or "",
            branch=event.branch or "",
            part_index=part_index,
            operation_boundaries=tuple(operation_boundaries),
        )
        if call.name == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME:
            waits.append(_confirmation_suspend(call, function_call_id, provenance))
        else:
            waits.append(_request_input_suspend(call, function_call_id, provenance))
    return waits


def _answer_schema(
    kind: Literal["adk_request_input", "adk_request_confirmation"],
) -> dict[str, Any]:
    if kind == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME:
        answer: dict[str, Any] = {
            "type": "object",
            "required": ["confirmed"],
            "properties": {"confirmed": {"type": "boolean"}},
        }
    else:
        answer = {"not": {"type": "null"}}
    return answer


def frontier_resume_schema(
    interrupt_ids: Sequence[str],
    kinds: Sequence[Literal["adk_request_input", "adk_request_confirmation"]],
) -> dict[str, Any]:
    """Require one typed resume value for every advertised OE activity id."""
    if not interrupt_ids or len(interrupt_ids) != len(kinds):
        raise SuspendTranslationError("resume frontier ids and kinds must align")
    if any(not interrupt_id for interrupt_id in interrupt_ids) or len(
        set(interrupt_ids)
    ) != len(interrupt_ids):
        raise SuspendTranslationError("resume frontier activity ids must be unique")
    return {
        "type": "object",
        "required": ["resume_map"],
        "properties": {
            "resume_map": {
                "type": "object",
                "required": list(interrupt_ids),
                "properties": {
                    interrupt_id: _answer_schema(kind)
                    for interrupt_id, kind in zip(interrupt_ids, kinds, strict=True)
                },
                "additionalProperties": False,
            }
        },
        "additionalProperties": True,
    }


def suspend_frontier_stream_event(
    suspends: Sequence[AdkSuspend],
    *,
    interrupt_ids: Sequence[str],
    messages: list[Message],
    response: str = "",
    metadata: dict[str, Any] | None = None,
) -> StreamEvent:
    """Build one caller-facing event for a complete ADK wait frontier."""
    if not suspends or len(suspends) != len(interrupt_ids):
        raise SuspendTranslationError("suspend frontier and activity ids must align")
    data = SuspendEventData(
        suspend_payload=suspends[0].value,
        messages=messages,
        metadata=metadata or {},
        interrupts=[
            SuspendInterrupt(id=interrupt_id, value=suspend.value)
            for suspend, interrupt_id in zip(suspends, interrupt_ids, strict=True)
        ],
        resume_schema=frontier_resume_schema(
            interrupt_ids,
            [suspend.kind for suspend in suspends],
        ),
        response=response,
    )
    return StreamEvent(
        event="suspend",
        data=cast(JsonValue, data.model_dump(mode="json")),
    )


def invoke_output(
    *,
    result_text: str,
    suspend: SuspendEventData | None,
) -> AgentOutput:
    """Map the terminal stream events to ``AgentOutput``."""
    if suspend is not None:
        return AgentOutput(
            response=SuspendedAgentResponse(
                response=result_text,
                interrupts=suspend.interrupts,
                resume_schema=suspend.resume_schema,
                suspend_payload=suspend.suspend_payload,
            ).model_dump()
        )
    return AgentOutput(
        response=CompletedAgentResponse(response=result_text).model_dump()
    )


def function_response_frontier_content(
    suspends: Sequence[AdkSuspend], results: Sequence[Any]
) -> types.Content:
    """Build one ADK user Content that resolves the complete wait frontier."""
    if not suspends or len(suspends) != len(results):
        raise SuspendTranslationError("suspend frontier and results must align")
    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=suspend.function_call_id,
                    name=suspend.function_name,
                    response=_function_response_dict(suspend, result),
                )
            )
            for suspend, result in zip(suspends, results, strict=True)
        ],
    )


def _indexed_function_calls(event: Event) -> list[tuple[int, types.FunctionCall]]:
    if not event.content or not event.content.parts:
        return []
    return [
        (part_index, part.function_call)
        for part_index, part in enumerate(event.content.parts)
        if part.function_call is not None
    ]


def _request_input_suspend(
    call: types.FunctionCall,
    function_call_id: str,
    provenance: SuspendProvenance,
) -> AdkSuspend:
    args = dict(call.args) if call.args else {}
    function_name = call.name or REQUEST_INPUT_FUNCTION_CALL_NAME
    if function_name == REQUEST_INPUT_FUNCTION_CALL_NAME:
        message = _optional_str(_first_present(args, "message"))
        payload = _first_present(args, "payload")
        response_schema = _first_present(args, "response_schema", "responseSchema")
    else:
        message = _optional_str(_first_present(args, "message", "reason"))
        payload = args
        response_schema = _first_present(args, "response_schema", "responseSchema")
    value = cast(
        dict[str, Any],
        _require_jsonable(
            {
                "message": message,
                "payload": payload,
                "response_schema": response_schema,
            },
            what="RequestInput value",
        ),
    )
    semantic_input = cast(
        dict[str, Any],
        _require_jsonable(
            {
                "kind": REQUEST_INPUT_FUNCTION_CALL_NAME,
                "function_name": function_name,
                "message": message,
                "payload": payload,
                "response_schema": response_schema,
                "provenance": {
                    "node_path": provenance.node_path,
                    "branch": provenance.branch,
                    "part_index": provenance.part_index,
                },
            },
            what="RequestInput semantic input",
        ),
    )
    return AdkSuspend(
        kind=REQUEST_INPUT_FUNCTION_CALL_NAME,
        function_call_id=function_call_id,
        value=value,
        semantic_input=semantic_input,
        function_name=function_name,
        provenance=provenance,
    )


def _confirmation_suspend(
    call: types.FunctionCall,
    function_call_id: str,
    provenance: SuspendProvenance,
) -> AdkSuspend:
    args = dict(call.args) if call.args else {}
    original = _first_present(args, "originalFunctionCall", "original_function_call")
    confirmation = _first_present(args, "toolConfirmation", "tool_confirmation")
    if not isinstance(original, dict):
        raise SuspendTranslationError(
            "adk_request_confirmation is missing originalFunctionCall"
        )
    original_call = cast(dict[str, Any], original)
    confirmation_payload: dict[str, Any] = (
        cast(dict[str, Any], confirmation) if isinstance(confirmation, dict) else {}
    )
    hint = _optional_str(
        _first_present(confirmation_payload, "hint") or _first_present(args, "hint")
    )
    payload = _first_present(confirmation_payload, "payload")
    tool_name = _optional_str(_first_present(original_call, "name"))
    raw_tool_args = _first_present(original_call, "args")
    tool_args: dict[str, Any] = (
        cast(dict[str, Any], raw_tool_args) if isinstance(raw_tool_args, dict) else {}
    )
    if not tool_name:
        raise SuspendTranslationError(
            "adk_request_confirmation is missing the original tool name"
        )
    tool_call: dict[str, Any] = {"name": tool_name, "args": tool_args}
    value = cast(
        dict[str, Any],
        _require_jsonable(
            {
                "hint": hint,
                "payload": payload,
                "tool_call": tool_call,
                "response_schema": _answer_schema(
                    REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
                ),
            },
            what="tool confirmation value",
        ),
    )
    semantic_input = cast(
        dict[str, Any],
        _require_jsonable(
            {
                "kind": REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                "hint": hint,
                "payload": payload,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "provenance": {
                    "node_path": provenance.node_path,
                    "branch": provenance.branch,
                    "part_index": provenance.part_index,
                },
            },
            what="tool confirmation semantic input",
        ),
    )
    return AdkSuspend(
        kind=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
        function_call_id=function_call_id,
        value=value,
        semantic_input=semantic_input,
        function_name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
        provenance=provenance,
    )


def _function_response_dict(suspend: AdkSuspend, result: Any) -> dict[str, Any]:
    if suspend.kind == REQUEST_INPUT_FUNCTION_CALL_NAME:
        return _mapping_response(result)
    return _confirmation_response(result)


def _mapping_response(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return cast(
            dict[str, Any],
            _require_jsonable(result, what="RequestInput response"),
        )
    return {_RESULT_KEY: _require_jsonable(result, what="RequestInput response")}


def _confirmation_response(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise SuspendTranslationError(
            "tool confirmation result must be an object with confirmed"
        )
    payload = cast(dict[str, Any], result)
    if "confirmed" not in payload:
        raise SuspendTranslationError("tool confirmation result must include confirmed")
    confirmed = payload["confirmed"]
    if not isinstance(confirmed, bool):
        raise SuspendTranslationError("tool confirmation confirmed must be a boolean")
    response = {
        "confirmed": confirmed,
        "payload": payload.get("payload"),
    }
    return cast(
        dict[str, Any],
        _require_jsonable(response, what="tool confirmation response"),
    )


def _first_present(values: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in values:
            return values[key]
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _require_jsonable(value: Any, *, what: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise SuspendTranslationError(f"{what} is not JSON-serializable") from error
