"""Bidirectional conversion between platform sdk-core Messages and ADK Content."""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from agent_engine_sdk.models import LLMToolCall, LLMToolSchema, Message, Role
from google.genai import types

_logger = logging.getLogger(__name__)


def parse_tool_args(raw_args: Any) -> dict[str, Any]:
    """Parse tool call args from dict or JSON string to a dict."""
    if isinstance(raw_args, dict):
        return cast(dict[str, Any], raw_args)
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return cast(dict[str, Any], parsed)
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


_ROLE_PLATFORM_TO_ADK: dict[str, str] = {
    "user": "user",
    "assistant": "model",
    "system": "user",
}
_ROLE_ADK_TO_PLATFORM: dict[str, Role] = {
    "user": "user",
    "model": "assistant",
}
_ADK_CONTROL_FUNCTION_NAMES = frozenset(
    {
        "adk_request_confirmation",
        "adk_request_credential",
    }
)


def _canonical_json(value: Any) -> Any:
    """Normalize JSON values that lose representation details in protobuf."""
    if isinstance(value, dict):
        mapping = cast(dict[str, Any], value)
        return {key: _canonical_json(item) for key, item in mapping.items()}
    if isinstance(value, list):
        return [_canonical_json(item) for item in cast(list[Any], value)]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def platform_message_to_content(message: Message) -> types.Content:
    """Convert a platform Message to an ADK Content object."""
    if message.role == "tool":
        text = message.content if isinstance(message.content, str) else ""
        try:
            response_data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            response_data = {"result": text}
        return types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=message.tool_call_id or None,
                        name=message.name or "unknown",
                        response=response_data,
                    )
                )
            ],
        )

    adk_role = _ROLE_PLATFORM_TO_ADK.get(message.role, "user")
    text = message.content if isinstance(message.content, str) else ""

    parts: list[types.Part] = []
    if text:
        parts.append(types.Part.from_text(text=text))

    if message.tool_calls:
        for tc in message.tool_calls:
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        name=tc.name or "unknown",
                        id=tc.id or None,
                        args=parse_tool_args(tc.args),
                    )
                )
            )

    if not parts:
        parts.append(types.Part.from_text(text=""))

    return types.Content(role=adk_role, parts=parts)


def content_to_platform_messages(content: types.Content) -> list[Message]:
    """Convert an ADK Content object to platform Messages.

    A single ADK Content can carry *multiple* ``function_response`` parts when
    the model issued parallel tool calls. The platform Message model holds one
    tool result per message, so each response becomes its own ``tool`` Message.
    Collapsing them into one (keeping only the last) drops the other results,
    which makes the model re-issue those tool calls in a runaway loop.
    """
    role: Role = _ROLE_ADK_TO_PLATFORM.get(content.role or "user", "user")

    text_parts: list[str] = []
    tool_calls: list[LLMToolCall] = []
    tool_messages: list[Message] = []

    if content.parts:
        for part in content.parts:
            if part.text is not None:
                text_parts.append(part.text)
            elif part.function_call is not None:
                function_call = part.function_call
                # ADK injects confirmation and credential frames; those are not
                # model-issued tool calls. RequestInput is a real model tool
                # call and must stay paired with its function response or the
                # next LLM request is rejected as an unmatched tool message.
                if function_call.name in _ADK_CONTROL_FUNCTION_NAMES:
                    continue
                tool_calls.append(
                    LLMToolCall(
                        id=function_call.id if function_call.id else None,
                        name=function_call.name,
                        args=dict(function_call.args) if function_call.args else {},
                    )
                )
            elif part.function_response is not None:
                function_response = part.function_response
                if function_response.name in _ADK_CONTROL_FUNCTION_NAMES:
                    continue
                tool_messages.append(
                    Message(
                        role="tool",
                        content=json.dumps(
                            _canonical_json(function_response.response),
                            sort_keys=True,
                            allow_nan=False,
                        ),
                        name=function_response.name,
                        # Correlates this result with the assistant tool_call of
                        # the same id. STM records it (extract_turn_messages
                        # writes tool_call_id), so dropping it would leave ADK
                        # tool turns uncorrelatable — the LangGraph adapter
                        # preserves it and memory parity requires the same.
                        tool_call_id=function_response.id or None,
                    )
                )

    messages: list[Message] = []
    if text_parts or tool_calls:
        # ADK emits role="user" Content that carries only pseudo-function_calls
        # for its own control machinery (adk_request_credential for auth,
        # adk_request_confirmation for tool approval). That is not a user
        # utterance: persisting it would write a blank user turn to STM and
        # drop the call detail, and mapping it to assistant would fabricate a
        # tool call the model never issued. Control frames resume from ADK's
        # session service, not STM, so they are dropped here.
        if role == "user" and tool_calls and not text_parts:
            return tool_messages
        messages.append(
            Message(
                role=role,
                content="".join(text_parts),
                tool_calls=tool_calls if tool_calls else None,
            )
        )
    messages.extend(tool_messages)
    return messages


def tools_dict_to_schemas(tools_dict: dict[str, Any]) -> list[LLMToolSchema]:
    """Convert ADK tools_dict to platform LLMToolSchema list for SecureLLMProxy."""
    schemas: list[LLMToolSchema] = []
    for name, tool in tools_dict.items():
        declaration = None
        # ADK's BaseTool._get_declaration() is the only way to extract the
        # FunctionDeclaration schema.
        try:
            if hasattr(tool, "_get_declaration"):
                declaration = tool._get_declaration()
        except Exception:
            _logger.warning(
                "Failed to extract declaration for tool %s, schema will be incomplete",
                name,
                exc_info=True,
            )

        description = ""
        parameters: dict[str, Any] | None = None
        if declaration is not None:
            description = declaration.description or ""
            if declaration.parameters_json_schema is not None:
                parameters = dict(declaration.parameters_json_schema)

        schemas.append(
            LLMToolSchema(
                name=name,
                description=description,
                parameters=parameters,
            )
        )
    return schemas
