"""Framework-neutral Pydantic models for Atlas Agent Engine."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Generic, Literal, TypeVar, cast

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)


class MappingCompatModel(BaseModel):
    """Pydantic model with light dict-like compatibility helpers.

    Equality to dicts is intentionally one-directional compatibility for older
    call sites that compare ``model == dict``. Prefer explicit ``model_dump()``
    comparisons in new code.
    """

    model_config = ConfigDict(populate_by_name=True)

    def __getitem__(self, key: str) -> Any:
        if key == "arguments" and hasattr(self, "args"):
            return getattr(self, "args")
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            return self.model_dump(exclude_none=True) == other
        return super().__eq__(other)


class ToolDefinition(BaseModel):
    """Framework-neutral tool definition."""

    name: str
    description: str
    args_schema: dict[str, Any]
    callable: Any
    remote: bool = True
    provider_type: str | None = None
    scopes: list[str] = []
    network: list[str] = []
    timeout_seconds: int = 30
    redact_fields: list[str] = []


class TextBlock(BaseModel):
    """Text content block."""

    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    """Image content block."""

    type: Literal["image"] = "image"
    url: str
    mime_type: str | None = None


class DocumentBlock(BaseModel):
    """Document/file content block."""

    type: Literal["document"] = "document"
    url: str
    mime_type: str | None = None
    filename: str | None = None


AnyContentBlock = TextBlock | ImageBlock | DocumentBlock


Role = Literal["user", "assistant", "tool", "system"]


class LLMTokenUsage(MappingCompatModel):
    """Typed token-usage metadata for LLM calls."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    input_tokens: int | None = None
    output_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    model: str | None = None

    @model_validator(mode="after")
    def _populate_total_tokens(self) -> LLMTokenUsage:
        if self.total_tokens is not None:
            return self

        prompt_tokens = (
            self.input_tokens if self.input_tokens is not None else self.prompt_tokens
        )
        completion_tokens = (
            self.output_tokens
            if self.output_tokens is not None
            else self.completion_tokens
        )
        if prompt_tokens is not None and completion_tokens is not None:
            self.total_tokens = prompt_tokens + completion_tokens
        return self

    def to_langchain_usage_metadata(self) -> dict[str, int]:
        """Return LangChain's expected usage-metadata keys."""
        input_tokens = (
            self.input_tokens
            if self.input_tokens is not None
            else self.prompt_tokens or 0
        )
        output_tokens = (
            self.output_tokens
            if self.output_tokens is not None
            else self.completion_tokens or 0
        )
        total_tokens = (
            self.total_tokens
            if self.total_tokens is not None
            else input_tokens + output_tokens
        )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }


class LLMToolCall(MappingCompatModel):
    """Typed final tool call requested by an LLM."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str | None = None
    name: str | None = None
    args: JsonValue | None = Field(
        default=None,
        validation_alias=AliasChoices("args", "arguments"),
        serialization_alias="args",
    )
    type: str | None = None
    index: int | None = None

    def to_langchain_dict(self) -> dict[str, JsonValue]:
        """Return a LangChain-compatible tool-call dict."""
        payload = self.model_dump(exclude_none=True)
        payload.pop("index", None)
        return payload


class LLMToolSchema(MappingCompatModel):
    """Serializable bound-tool schema forwarded with an LLM call."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    name: str | None = None
    description: str | None = None
    parameters: JsonValue | None = None
    type: str | None = None
    strict: bool | None = None
    function: JsonValue | None = None

    def to_langchain_dict(self) -> dict[str, JsonValue]:
        """Return the schema in the shape LangChain bind_tools expects."""
        return self.model_dump(exclude_none=True)


class LLMInvocationOptions(BaseModel):
    """Explicit provider/model options passed with an LLM invocation."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    max_tokens: int | None = None
    response_format: JsonValue | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    timeout: float | None = None
    parallel_tool_calls: bool | None = None
    reasoning_effort: str | None = None

    def to_model_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for underlying model invocation."""
        return self.model_dump(exclude_none=True)


class Message(MappingCompatModel):
    """Framework-neutral message.

    Content can be a simple string for text-only messages, or a list of
    content blocks for multimodal content (images, documents, etc.).
    """

    role: Role
    content: str | list[AnyContentBlock]
    tool_calls: list[LLMToolCall] | None = None
    tool_call_id: str | None = None
    is_error: bool | None = None
    name: str | None = None
    id: str | None = None
    additional_kwargs: dict[str, JsonValue] | None = None
    response_metadata: dict[str, JsonValue] | None = None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            try:
                return self.model_dump(exclude_none=True) == Message.model_validate(
                    other
                ).model_dump(exclude_none=True)
            except ValidationError:
                return False
        return super().__eq__(other)


class MessageArtifact(MappingCompatModel):
    """Structured artifact attached to an assistant message.

    Attach artifacts through ``MessageArtifactMetadata.to_message_metadata()``
    on ``Message.additional_kwargs``. Platform and custom frontends can render
    the same transported artifact payloads. Artifact fields must be JSON-safe;
    binary data should be represented as a URL or an encoded string.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str
    kind: str
    title: str | None = None


class MessageArtifactMetadata(MappingCompatModel):
    """Typed metadata contract for message-scoped artifacts."""

    model_config = ConfigDict(populate_by_name=True)

    artifacts: list[MessageArtifact] = Field(default_factory=list)

    def to_message_metadata(self) -> dict[str, JsonValue]:
        """Return the dict to attach to ``Message.additional_kwargs``."""
        metadata: dict[str, Any] = {}
        if self.artifacts:
            metadata["artifacts"] = [
                artifact.model_dump(exclude_none=True, mode="json")
                for artifact in self.artifacts
            ]
        return cast(dict[str, JsonValue], metadata)


def collect_message_artifact_metadata(
    source: Mapping[str, Any] | None,
) -> MessageArtifactMetadata | None:
    """Collect message artifact metadata from ``Message.additional_kwargs``."""

    artifacts: list[MessageArtifact] = []
    if source is None or not isinstance(source.get("artifacts"), list):
        return None

    for raw_artifact in source["artifacts"]:
        try:
            artifacts.append(MessageArtifact.model_validate(raw_artifact))
        except ValidationError as error:
            logger.warning("Skipping malformed message artifact metadata: %s", error)
            continue

    return MessageArtifactMetadata(artifacts=artifacts) if artifacts else None


class LLMResponse(MappingCompatModel):
    """Framework-neutral LLM response."""

    content: str
    tool_calls: list[LLMToolCall] | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    usage: LLMTokenUsage | None = None
    id: str | None = None
    name: str | None = None
    additional_kwargs: dict[str, JsonValue] | None = None
    response_metadata: dict[str, JsonValue] | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_usage_from_metadata(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        payload = cast(dict[str, Any], value)
        if payload.get("usage") is not None:
            return payload

        metadata_value = payload.get("metadata")
        if not isinstance(metadata_value, dict):
            return payload
        metadata = cast(dict[str, Any], metadata_value)

        usage_value = metadata.get("usage")
        usage: dict[str, Any] | None = (
            cast(dict[str, Any], usage_value) if isinstance(usage_value, dict) else None
        )
        if usage is None:
            token_keys = {
                "input_tokens",
                "output_tokens",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "model",
            }
            usage = metadata if any(key in metadata for key in token_keys) else None

        if usage is None:
            return payload

        normalized = dict(payload)
        normalized["usage"] = usage
        return normalized


class ToolCallChunk(BaseModel):
    """Partial tool call during streaming.

    All fields are optional since chunks may contain partial data
    (e.g., just the start of arguments, or just the tool name).
    """

    id: str | None = None
    name: str | None = None
    args: str | None = None
    type: str | None = None
    index: int | None = None


class LLMStreamChunk(BaseModel):
    """Rich streaming chunk following OpenAI/Anthropic patterns.

    All fields are optional since different chunks contain different data:
    - Content chunks: text deltas during generation
    - Tool call chunks: partial tool calls being built up
    - Usage chunks: token counts at end of stream
    """

    content: str | None = None
    tool_calls: list[ToolCallChunk] | None = None
    usage: LLMTokenUsage | None = None
    id: str | None = None
    name: str | None = None
    additional_kwargs: dict[str, JsonValue] | None = None
    response_metadata: dict[str, JsonValue] | None = None


class AgentInput(BaseModel):
    """Input for an agent execution. Payload is an opaque JSON value."""

    payload: JsonValue


class AgentOutput(BaseModel):
    """Output from an agent execution. Response is an opaque JSON value."""

    response: JsonValue


class StreamEvent(BaseModel):
    """Event streamed back to the caller during execution."""

    data: JsonValue
    event: str | None = None
    # Output-parser payload; None on platform-only events (additive, nullable).
    custom_event: JsonValue | None = None

    @field_validator("event")
    @classmethod
    def _reject_control_chars(cls, v: str | None) -> str | None:
        # Parity with @mongodb-js/agent-engine-sdk StreamEventNameSchema: an SSE
        # `event:` line is terminated by CR/LF, so a name carrying those (or NUL)
        # could forge extra frames in any server-sent-event serializer. Forbid
        # them at the boundary.
        if v is not None and ("\r" in v or "\n" in v or "\x00" in v):
            raise ValueError("Stream event name must not contain CR, LF, or NUL")
        return v


StreamItemT = TypeVar("StreamItemT")


class OutputParser(ABC, Generic[StreamItemT]):
    """Author-supplied map from raw stream items to consumer-facing events.

    Framework-neutral: ``StreamItemT`` is the type of one item pulled from the
    underlying stream, which a framework binding fixes when it subclasses this.
    Both methods are abstract, so a subclass that omits either cannot be
    instantiated.

    A yielded ``BaseModel`` is serialized to JSON by the producer before it
    populates ``StreamEvent.custom_event`` (typed ``JsonValue``); yielding a
    plain JSON value populates it directly.
    """

    # Opaque to core; the framework binding interprets these mode names.
    stream_modes: Sequence[str] = ()

    @abstractmethod
    async def parse(
        self, item: StreamItemT, ctx: RequestContext
    ) -> AsyncIterator[BaseModel | JsonValue]:
        """Yield zero or more custom events for ``item``."""
        raise NotImplementedError
        yield  # pragma: no cover - unreachable; makes this an async generator

    @abstractmethod
    async def on_stream_error(
        self, ctx: RequestContext, error: BaseException
    ) -> BaseModel | JsonValue | None:
        """Return a final custom event before the stream closes on error, or None."""


class RequestContext(BaseModel):
    """Request-scoped context passed to every agent execution.

    Carries framework-neutral execution identity and platform plumbing. The
    caller's invocation input lives in ``AgentInput.payload``, not here.
    Suspend/resume state is opaque to the platform: ``metadata`` is a
    framework-owned blob that the platform forwards unchanged on resume.
    """

    execution_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    # Tenant scope of the executing agent. Framework adapters use it to scope
    # state stores keyed only by session_id (e.g. the LangGraph checkpoint
    # thread_id) so agents in the same project DB cannot collide.
    workspace_id: str | None = None
    request_headers: dict[str, str] | None = None
    # Platform suspend/resume plumbing (framework-neutral).
    resume: bool = False
    resume_data: JsonValue | None = None
    # True when this session's most recent prior execution was cancelled
    # (its pod torn down mid-run). Framework adapters that persist per-step
    # state use it to fence off the killed step's partial writes before
    # running this turn.
    previous_execution_cancelled: bool = False
    # Opaque framework-owned state passed through by the SDK and AER. Only the
    # selected framework adapter may interpret its keys and values.
    metadata: dict[str, Any] | None = None


class SessionSummary(BaseModel):
    """Per-session summary sourced from framework-specific persistence.

    Returned by the AER's session-enrichment endpoint. The platform layer
    (OE) merges this with its own tenant fields (user_id, project_id,
    workspace_id, visibility) when assembling the public sessions list.
    """

    session_id: str = Field(description="Identifier of the session.")
    last_activity: str = Field(
        description=(
            "ISO 8601 timestamp of the most recent activity; empty when unavailable."
        ),
    )
    created_at: str = Field(
        description=(
            "ISO 8601 timestamp when the session was first persisted; "
            "empty when unavailable."
        ),
    )
    message_count: int = Field(
        description=(
            "Number of messages in the session — the same count the "
            "session messages endpoint returns."
        ),
    )
    first_message_preview: str = Field(
        default="",
        description=(
            "Truncated first human-authored message (up to 80 chars). "
            "Empty when no human message has been persisted."
        ),
    )


class SessionsSummaryResponse(BaseModel):
    """Response for the AER's session-summary enrichment endpoint."""

    sessions: list[SessionSummary] = Field(
        description=(
            "One summary per requested session_id that has framework-side "
            "persistence. Sessions without persistence are omitted."
        ),
    )


class SessionMessage(BaseModel):
    """A single message in a session's conversation history.

    ``role`` uses the same vocabulary as ``Message`` ("user", "assistant",
    "tool", "system"). ``name`` carries the tool name on tool messages.
    """

    id: str = Field(
        description=(
            "Stable identifier — the framework's id when available, "
            "otherwise a synthetic id derived from session_id + position."
        ),
    )
    role: str = Field(
        description=(
            'Platform role: "user", "assistant", "tool", or "system". '
            "Unknown framework types pass through unchanged."
        ),
    )
    content: str = Field(
        description=(
            "Flat-string content. Multimodal content is collapsed to its "
            "text portions only."
        ),
    )
    timestamp: str = Field(
        description=(
            "ISO 8601 persistence-boundary timestamp. Native checkpoint "
            "projections use the first checkpoint where the message appeared; "
            "durable workflow projections use the committed snapshot boundary; "
            "empty when unavailable."
        ),
    )
    session_id: str = Field(
        description="Identifier of the session this message belongs to.",
    )
    name: str = Field(
        default="",
        description="Tool name on tool messages, empty otherwise.",
    )
    tool_calls: list[LLMToolCall] | None = Field(
        default=None,
        description=(
            "Tool calls requested by an assistant message. Each carries a "
            "stable ``id`` that the matching tool result echoes back via "
            "``tool_call_id``, so consumers can join a call to its output "
            "without inferring the link from name-and-order."
        ),
    )
    tool_call_id: str | None = Field(
        default=None,
        description=(
            "On a tool message, the id of the assistant tool call this "
            "message is the result of. Empty/absent on non-tool messages."
        ),
    )
    additional_kwargs: dict[str, JsonValue] | None = Field(
        default=None,
        description=(
            "Framework message metadata that should remain attached to the "
            "message, including generic UI artifact metadata."
        ),
    )


class SessionMessagesResponse(BaseModel):
    """Response for the AER's session-messages endpoint."""

    messages: list[SessionMessage] = Field(
        description=(
            "Decoded conversation history in chronological order. Empty "
            "when the session has no persisted messages."
        ),
    )


class SessionForkResponse(BaseModel):
    """Response for forking a session into a new independent session."""

    session_id: str = Field(description="Identifier of the new branch session.")
    execution_id: str = Field(
        description="Pending execution on the new branch session.",
    )
