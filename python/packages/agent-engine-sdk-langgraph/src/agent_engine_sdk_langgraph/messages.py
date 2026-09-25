"""LangChain message conversion layer.

Converts between LangChain message types (AIMessage, HumanMessage, etc.)
and the framework-neutral Message model from agent-engine-sdk.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict as langchain_message_to_dict,
    messages_from_dict,
)
from langchain_core.messages.tool import ToolCall
from langchain_core.messages.tool import ToolCallChunk as LCToolCallChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from agent_engine_sdk import (
    AnyContentBlock,
    DocumentBlock,
    ImageBlock,
    LLMToolCall,
    Message,
    Role,
    TextBlock,
    collect_message_artifact_metadata,
)
from agent_engine_sdk.models import LLMResponse, LLMStreamChunk, LLMTokenUsage
from pydantic import JsonValue

from agent_engine_runner_shared.generated.workflow.v1.state_pb2 import (
    MESSAGE_ROLE_ASSISTANT,
    MESSAGE_ROLE_SYSTEM,
    MESSAGE_ROLE_TOOL,
    MESSAGE_ROLE_USER,
    WorkflowMessage,
)
from agent_engine_runner_shared.workflow.protojson import (
    json_to_proto_struct,
    json_to_proto_value,
    proto_struct_to_json,
    proto_value_to_json,
)

__all__ = [
    "lc_to_platform_message",
    "platform_to_lc_message",
    "lc_messages_to_platform",
    "platform_messages_to_lc",
    "lc_to_workflow_message",
    "workflow_to_lc_message",
    "llm_response_to_chat_result",
    "llm_stream_chunk_to_generation_chunk",
    "message_to_dict",
    "dict_to_ai_message",
    "dict_to_ai_message_chunk",
    "find_last_ai_content",
    "normalize_content",
]

_logger = logging.getLogger(__name__)

# Workflow protobuf roles are mapped here alongside the existing LangChain and
# platform message conversions so state handling does not need its own message
# representation logic.
_ROLE_TO_WORKFLOW = {
    "user": MESSAGE_ROLE_USER,
    "assistant": MESSAGE_ROLE_ASSISTANT,
    "tool": MESSAGE_ROLE_TOOL,
    "system": MESSAGE_ROLE_SYSTEM,
}
_ROLE_FROM_WORKFLOW = {value: key for key, value in _ROLE_TO_WORKFLOW.items()}


def _convert_lc_content_to_platform(
    content: Any,
) -> str | list[AnyContentBlock]:
    """Convert LangChain message content to platform format.

    LangChain messages can have multimodal content as a list of content blocks.
    This function preserves the structure for multimodal content while keeping
    simple string content as-is.

    Args:
        content: LangChain message content (str or list of content blocks)

    Returns:
        String for simple text, or list of ContentBlocks for multimodal content
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    # Multimodal content - preserve structure
    blocks: list[AnyContentBlock] = []
    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type", "text")
            if block_type == "text":
                blocks.append(TextBlock(text=block.get("text", "")))
            elif block_type in ("image", "image_url"):
                # Handle LangChain's image_url format (supports both string and dict)
                if "image_url" in block:
                    image_data = block["image_url"]
                    if isinstance(image_data, dict):
                        url = image_data.get("url")
                    else:
                        url = image_data
                else:
                    url = block.get("url")
                if url:
                    blocks.append(ImageBlock(url=url, mime_type=block.get("mime_type")))
                else:
                    _logger.warning("Image block missing url; skipping")
                    continue
            elif block_type in ("file", "document"):
                url = block.get("url")
                if url:
                    blocks.append(
                        DocumentBlock(
                            url=url,
                            mime_type=block.get("mime_type"),
                            filename=block.get("filename"),
                        )
                    )
                else:
                    _logger.warning("Document block missing url; skipping")
                    continue
            else:
                # Unknown type - log warning and serialize to text
                _logger.warning(
                    "Unknown content block type=%s; serializing to TextBlock",
                    block_type,
                )
                blocks.append(TextBlock(text=json.dumps(block)))
        else:
            blocks.append(TextBlock(text=str(block)))

    # Optimization: single text block → string
    if len(blocks) == 1 and isinstance(blocks[0], TextBlock):
        return blocks[0].text

    return blocks if blocks else ""


def _serialize_tool_calls(tool_calls: list[ToolCall]) -> list[LLMToolCall]:
    """Serialize LangChain tool calls to dicts.

    LangChain's ToolCall is a TypedDict — at runtime, tool_calls on an
    AIMessage are always plain dicts with ``name``, ``args``, ``id``, and
    ``type`` keys (enforced by LangChain's model validator).

    Args:
        tool_calls: LangChain ToolCall dicts

    Returns:
        List of typed tool calls
    """
    required_keys = {"name", "args", "id"}
    result: list[LLMToolCall] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            raise TypeError(f"Expected dict tool call, got {type(tc).__name__}")
        missing = required_keys - tc.keys()
        if missing:
            _logger.warning("Tool call missing keys %s: %s", missing, tc)
        result.append(LLMToolCall.model_validate(tc))
    return result


def lc_to_platform_message(lc_message: BaseMessage) -> Message:
    """Convert a LangChain message to platform Message.

    Args:
        lc_message: LangChain BaseMessage instance

    Returns:
        Platform Message instance
    """
    # Get the message type from LangChain's type field
    msg_type = getattr(lc_message, "type", "human")
    _ROLE_MAP: dict[str, Role] = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
    }
    role: Role = _ROLE_MAP.get(msg_type, "user")

    # Extract content (preserves multimodal structure)
    content = _convert_lc_content_to_platform(lc_message.content)

    # Handle tool calls (for AIMessage)
    tool_calls: list[LLMToolCall] | None = None
    if isinstance(lc_message, AIMessage) and lc_message.tool_calls:
        tool_calls = _serialize_tool_calls(lc_message.tool_calls)

    # Handle tool_call_id (for ToolMessage)
    tool_call_id: str | None = None
    is_error: bool | None = None
    message_id = lc_message.id
    if isinstance(lc_message, ToolMessage):
        tool_call_id = lc_message.tool_call_id
        is_error = lc_message.status == "error"

    # Preserve name (available on all BaseMessage types)
    name: str | None = lc_message.name
    additional_kwargs = (
        cast(dict[str, JsonValue], lc_message.additional_kwargs)
        if lc_message.additional_kwargs
        else None
    )
    response_metadata = (
        cast(dict[str, JsonValue], lc_message.response_metadata)
        if lc_message.response_metadata
        else None
    )

    return Message(
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
        is_error=is_error,
        name=name,
        id=message_id,
        additional_kwargs=additional_kwargs,
        response_metadata=response_metadata,
    )


def _convert_platform_content_to_lc(
    content: str | list[AnyContentBlock],
) -> str | list[dict[str, Any]]:
    """Convert platform content to LangChain format.

    Args:
        content: Platform message content (str or list of ContentBlocks)

    Returns:
        String for simple text, or list of dicts for multimodal content
    """
    if isinstance(content, str):
        return content

    # Convert ContentBlocks to LangChain dict format
    lc_blocks: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextBlock):
            lc_blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            lc_block: dict[str, Any] = {"type": "image", "url": block.url}
            if block.mime_type:
                lc_block["mime_type"] = block.mime_type
            lc_blocks.append(lc_block)
        elif isinstance(block, DocumentBlock):
            lc_block: dict[str, Any] = {"type": "document", "url": block.url}
            if block.mime_type:
                lc_block["mime_type"] = block.mime_type
            if block.filename:
                lc_block["filename"] = block.filename
            lc_blocks.append(lc_block)

    return lc_blocks if lc_blocks else ""


def platform_to_lc_message(message: Message) -> BaseMessage | None:
    """Convert a platform Message to LangChain message.

    Args:
        message: Platform Message instance

    Returns:
        LangChain BaseMessage instance, or None if the message cannot be converted
        (e.g., tool message with missing tool_call_id)
    """
    role = message.role
    lc_class = {
        "user": HumanMessage,
        "assistant": AIMessage,
        "system": SystemMessage,
        "tool": ToolMessage,
    }.get(role, HumanMessage)

    # Convert content to LangChain format
    lc_content = _convert_platform_content_to_lc(message.content)

    if lc_class == ToolMessage:
        # ToolMessage requires tool_call_id and string content
        if not message.tool_call_id:
            preview = (
                message.content[:50] if isinstance(message.content, str) else "..."
            )
            _logger.warning(
                "Skipping tool message: missing tool_call_id (content=%r)", preview
            )
            return None
        # ToolMessage typically expects string content
        str_content = (
            lc_content if isinstance(lc_content, str) else json.dumps(lc_content)
        )
        return ToolMessage(
            content=str_content,
            tool_call_id=message.tool_call_id,
            status="error" if message.is_error else "success",
            name=message.name,
            id=message.id,
            additional_kwargs=message.additional_kwargs or {},
            response_metadata=message.response_metadata or {},
        )
    elif lc_class == AIMessage:
        # AIMessage may have tool_calls
        return AIMessage(
            content=lc_content,  # type: ignore[arg-type]
            tool_calls=[
                tool_call.to_langchain_dict()
                for tool_call in (message.tool_calls or [])
            ],
            name=message.name,
            id=message.id,
            additional_kwargs=message.additional_kwargs or {},
            response_metadata=message.response_metadata or {},
        )
    else:
        # HumanMessage, SystemMessage (content can be str or list of content blocks)
        return lc_class(  # type: ignore[call-arg]
            content=lc_content,
            name=message.name,
            id=message.id,
            additional_kwargs=message.additional_kwargs or {},
            response_metadata=message.response_metadata or {},
        )


def lc_to_workflow_message(
    message: BaseMessage,
    *,
    include_legacy_source: bool = False,
) -> WorkflowMessage:
    """Convert one LangChain message to the durable workflow contract."""
    platform_message = lc_to_platform_message(message)
    result = WorkflowMessage(
        role=_ROLE_TO_WORKFLOW[platform_message.role],
        content=json_to_proto_value(message.content),
    )
    for tool_call in platform_message.tool_calls or []:
        result.tool_calls.append(
            json_to_proto_struct(tool_call.model_dump(mode="json"))
        )
    if isinstance(message, AIMessage):
        for tool_call in message.invalid_tool_calls:
            result.invalid_tool_calls.append(json_to_proto_struct(tool_call))
    additional_kwargs = dict(platform_message.additional_kwargs or {})
    platform_artifacts = additional_kwargs.pop("artifacts", [])
    if not isinstance(platform_artifacts, list):
        raise TypeError("message additional_kwargs.artifacts must be an array")
    for index, artifact in enumerate(platform_artifacts):
        if not isinstance(artifact, dict):
            raise TypeError(
                f"message additional_kwargs.artifacts[{index}] must be an object"
            )
        result.platform_artifacts.append(json_to_proto_struct(artifact))
    additional_kwargs.pop("tool_calls", None)
    if additional_kwargs:
        result.additional_kwargs.CopyFrom(json_to_proto_struct(additional_kwargs))
    if platform_message.response_metadata:
        result.response_metadata.CopyFrom(
            json_to_proto_struct(platform_message.response_metadata)
        )
    # LangGraph checkpoint serializers can materialize an omitted Pydantic field
    # as its default None value. Both shapes mean that no tool artifact exists.
    if isinstance(message, ToolMessage) and message.artifact is not None:
        result.tool_artifact.CopyFrom(json_to_proto_value(message.artifact))
    if platform_message.tool_call_id is not None:
        result.tool_call_id = platform_message.tool_call_id
    if platform_message.name is not None:
        result.name = platform_message.name
    if platform_message.id is not None:
        result.id = platform_message.id
    if platform_message.is_error is not None:
        result.is_error = platform_message.is_error
    if include_legacy_source:
        result.source_message.CopyFrom(
            json_to_proto_value(langchain_message_to_dict(message))
        )
        artifact_metadata = collect_message_artifact_metadata(message.additional_kwargs)
        if artifact_metadata is not None:
            for artifact in artifact_metadata.artifacts:
                result.artifacts.append(
                    json_to_proto_struct(
                        artifact.model_dump(exclude_none=True, mode="json")
                    )
                )
    return result


def workflow_to_lc_message(
    message: WorkflowMessage,
    *,
    path: str = "message",
) -> BaseMessage:
    """Reconstruct one LangChain message from durable workflow state."""
    if message.HasField("source_message"):
        encoded = proto_value_to_json(message.source_message)
        try:
            return messages_from_dict([encoded])[0]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{path}: previous workflow message cannot be reconstructed"
            ) from error

    role = _ROLE_FROM_WORKFLOW.get(message.role)
    if role is None:
        raise ValueError("previous workflow message has an unspecified role")
    content = proto_value_to_json(message.content)
    if not isinstance(content, (str, list)):
        raise ValueError(f"{path}: previous workflow message content is malformed")

    data: dict[str, Any] = {"content": content}
    for name in ("tool_call_id", "name", "id"):
        if message.HasField(name):
            data[name] = getattr(message, name)
    additional_kwargs: dict[str, Any] = {}
    if message.HasField("additional_kwargs"):
        additional_kwargs.update(proto_struct_to_json(message.additional_kwargs))
    if message.platform_artifacts:
        additional_kwargs["artifacts"] = [
            proto_struct_to_json(artifact) for artifact in message.platform_artifacts
        ]
    if additional_kwargs:
        data["additional_kwargs"] = additional_kwargs
    if message.HasField("response_metadata"):
        data["response_metadata"] = proto_struct_to_json(message.response_metadata)
    if message.HasField("is_error"):
        data["status"] = "error" if message.is_error else "success"
    if message.HasField("tool_artifact"):
        data["artifact"] = proto_value_to_json(message.tool_artifact)

    try:
        if role == "assistant":
            data["tool_calls"] = [
                LLMToolCall.model_validate(
                    proto_struct_to_json(tool_call)
                ).to_langchain_dict()
                for tool_call in message.tool_calls
            ]
            data["invalid_tool_calls"] = [
                proto_struct_to_json(tool_call)
                for tool_call in message.invalid_tool_calls
            ]
            return AIMessage.model_validate(data)
        if role == "tool":
            if not message.HasField("tool_call_id") or not message.tool_call_id:
                raise ValueError("tool_call_id is required")
            return ToolMessage.model_validate(data)
        if role == "system":
            return SystemMessage.model_validate(data)
        return HumanMessage.model_validate(data)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}: previous workflow message cannot be reconstructed"
        ) from error


def lc_messages_to_platform(messages: list[BaseMessage]) -> list[Message]:
    """Convert a list of LangChain messages to platform Messages.

    Args:
        messages: List of LangChain BaseMessage instances

    Returns:
        List of platform Message instances
    """
    return [lc_to_platform_message(msg) for msg in messages]


def platform_messages_to_lc(messages: list[Message]) -> list[BaseMessage]:
    """Convert a list of platform Messages to LangChain messages.

    Invalid messages (e.g., tool messages with missing tool_call_id) are
    skipped with a warning logged.

    Args:
        messages: List of platform Message instances

    Returns:
        List of LangChain BaseMessage instances (invalid messages filtered out)
    """
    result = []
    for msg in messages:
        lc_msg = platform_to_lc_message(msg)
        if lc_msg is not None:
            result.append(lc_msg)
    return result


# =============================================================================
# OE message serialization (LangChain ↔ dict)
# =============================================================================

FRAMEWORK_LANGCHAIN = "langchain"


def message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """Serialize a LangChain message to the platform wire format.

    Stamps every dict with ``"framework": "langchain"`` so downstream
    consumers can identify which framework produced the message.
    """
    data = message.model_dump(exclude_none=True)
    data["framework"] = FRAMEWORK_LANGCHAIN
    return data


def _check_framework(data: dict[str, Any]) -> None:
    """Raise if the wire-format dict came from an unsupported framework."""
    fw = data.get("framework")
    if fw is not None and fw != FRAMEWORK_LANGCHAIN:
        raise ValueError(
            f"Cannot deserialize message from framework {fw!r} "
            "with the LangChain deserializer"
        )


def dict_to_ai_message(data: Any) -> AIMessage:
    """Deserialize a platform wire-format dict to a LangChain AIMessage."""
    if isinstance(data, AIMessage):
        return data
    if isinstance(data, dict):
        _check_framework(data)
        return AIMessage.model_validate(data)
    return AIMessage(content=str(data))


def dict_to_ai_message_chunk(data: Any) -> AIMessageChunk:
    """Deserialize a platform wire-format dict to a LangChain AIMessageChunk.

    Used for streaming replay where ChatGenerationChunk expects a chunk type.
    """
    if isinstance(data, AIMessageChunk):
        return data
    if isinstance(data, dict):
        _check_framework(data)
        return AIMessageChunk.model_validate({**data, "type": "AIMessageChunk"})
    return AIMessageChunk(content=str(data))


# =============================================================================
# LLM Response conversion (sdk-core ↔ LangChain)
# =============================================================================


def _usage_from_response(response: LLMResponse) -> LLMTokenUsage | None:
    if response.usage is not None:
        return response.usage

    metadata = dict(response.metadata)
    nested_usage = metadata.get("usage")
    if isinstance(nested_usage, dict):
        return LLMTokenUsage.model_validate(nested_usage)

    token_keys = {
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "model",
    }
    if any(key in metadata for key in token_keys):
        return LLMTokenUsage.model_validate(metadata)
    return None


def llm_response_to_chat_result(response: LLMResponse) -> ChatResult:
    """Convert sdk-core LLMResponse to LangChain ChatResult.

    Args:
        response: sdk-core LLMResponse instance

    Returns:
        LangChain ChatResult with AIMessage generation
    """
    response_metadata = dict(response.metadata)
    raw_metadata = response_metadata.pop("_raw_response_metadata", {})
    raw_response_metadata: dict[str, Any] = (
        raw_metadata if isinstance(raw_metadata, dict) else {}
    )
    response_metadata.pop("usage", None)
    explicit_response_metadata = dict(response.response_metadata or {})
    usage = _usage_from_response(response)
    usage_metadata = usage.model_dump(exclude_none=True) if usage is not None else {}
    merged_metadata = {
        **response_metadata,
        **usage_metadata,
        **raw_response_metadata,
        **explicit_response_metadata,
    }
    ai_message = AIMessage(
        content=response.content,
        tool_calls=[
            tool_call.to_langchain_dict() for tool_call in (response.tool_calls or [])
        ],
        id=response.id,
        name=response.name,
        additional_kwargs=response.additional_kwargs or {},
        response_metadata=merged_metadata,
    )
    return ChatResult(generations=[ChatGeneration(message=ai_message)])


def llm_stream_chunk_to_generation_chunk(
    chunk: LLMStreamChunk,
) -> ChatGenerationChunk:
    """Convert sdk-core LLMStreamChunk to LangChain ChatGenerationChunk.

    Args:
        chunk: sdk-core LLMStreamChunk instance

    Returns:
        LangChain ChatGenerationChunk with AIMessageChunk
    """
    tool_call_chunks: list[LCToolCallChunk] = [
        LCToolCallChunk(id=tc.id, name=tc.name, args=tc.args, index=tc.index)
        for tc in (chunk.tool_calls or [])
    ]
    # Normalize usage keys: sdk-core uses prompt_tokens/completion_tokens,
    # but LangChain UsageMetadata requires input_tokens/output_tokens.
    usage_metadata = (
        chunk.usage.to_langchain_usage_metadata() if chunk.usage is not None else None
    )
    ai_chunk = AIMessageChunk(
        content=chunk.content or "",
        tool_call_chunks=tool_call_chunks,
        id=chunk.id,
        name=chunk.name,
        additional_kwargs=chunk.additional_kwargs or {},
        response_metadata=chunk.response_metadata or {},
        usage_metadata=cast(Any, usage_metadata),
    )
    return ChatGenerationChunk(message=ai_chunk)


def normalize_content(content: Any) -> str:
    """Normalize LangChain message content to a string.

    Args:
        content: Message content (str, list, or other)

    Returns:
        Normalized string content
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Multimodal content - extract text parts
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content else ""


def find_last_ai_content(messages: list[Any]) -> str:
    """Return normalized text from the last AIMessage with content.

    Skips ToolMessages, HumanMessages, and AIMessages whose text
    content is empty (common with Anthropic after tool execution).

    Args:
        messages: List of LangChain messages

    Returns:
        Content of the last AIMessage, or empty string
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = normalize_content(msg.content)
            if content:
                return content
    return ""
