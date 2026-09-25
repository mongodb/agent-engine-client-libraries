# Table of Contents

* [agent\_engine\_sdk.app](#agent_engine_sdk.app)
  * [BaseApp](#agent_engine_sdk.app.BaseApp)
    * [get\_tool\_definitions](#agent_engine_sdk.app.BaseApp.get_tool_definitions)
    * [tools](#agent_engine_sdk.app.BaseApp.tools)
    * [tool](#agent_engine_sdk.app.BaseApp.tool)
    * [entrypoint](#agent_engine_sdk.app.BaseApp.entrypoint)
* [agent\_engine\_sdk.interfaces](#agent_engine_sdk.interfaces)
  * [ExecutionResult](#agent_engine_sdk.interfaces.ExecutionResult)
  * [BaseAgent](#agent_engine_sdk.interfaces.BaseAgent)
  * [BaseLLM](#agent_engine_sdk.interfaces.BaseLLM)
  * [BaseExecutionCallback](#agent_engine_sdk.interfaces.BaseExecutionCallback)
  * [NullExecutionCallback](#agent_engine_sdk.interfaces.NullExecutionCallback)
* [agent\_engine\_sdk.models](#agent_engine_sdk.models)
  * [MappingCompatModel](#agent_engine_sdk.models.MappingCompatModel)
  * [ToolDefinition](#agent_engine_sdk.models.ToolDefinition)
  * [TextBlock](#agent_engine_sdk.models.TextBlock)
  * [ImageBlock](#agent_engine_sdk.models.ImageBlock)
  * [DocumentBlock](#agent_engine_sdk.models.DocumentBlock)
  * [LLMTokenUsage](#agent_engine_sdk.models.LLMTokenUsage)
    * [to\_langchain\_usage\_metadata](#agent_engine_sdk.models.LLMTokenUsage.to_langchain_usage_metadata)
  * [LLMToolCall](#agent_engine_sdk.models.LLMToolCall)
    * [to\_langchain\_dict](#agent_engine_sdk.models.LLMToolCall.to_langchain_dict)
  * [LLMToolSchema](#agent_engine_sdk.models.LLMToolSchema)
    * [to\_langchain\_dict](#agent_engine_sdk.models.LLMToolSchema.to_langchain_dict)
  * [LLMInvocationOptions](#agent_engine_sdk.models.LLMInvocationOptions)
    * [to\_model\_kwargs](#agent_engine_sdk.models.LLMInvocationOptions.to_model_kwargs)
  * [Message](#agent_engine_sdk.models.Message)
  * [MessageArtifact](#agent_engine_sdk.models.MessageArtifact)
  * [MessageArtifactMetadata](#agent_engine_sdk.models.MessageArtifactMetadata)
    * [to\_message\_metadata](#agent_engine_sdk.models.MessageArtifactMetadata.to_message_metadata)
  * [collect\_message\_artifact\_metadata](#agent_engine_sdk.models.collect_message_artifact_metadata)
  * [LLMResponse](#agent_engine_sdk.models.LLMResponse)
  * [ToolCallChunk](#agent_engine_sdk.models.ToolCallChunk)
  * [LLMStreamChunk](#agent_engine_sdk.models.LLMStreamChunk)
  * [AgentInput](#agent_engine_sdk.models.AgentInput)
  * [AgentOutput](#agent_engine_sdk.models.AgentOutput)
  * [StreamEvent](#agent_engine_sdk.models.StreamEvent)
  * [OutputParser](#agent_engine_sdk.models.OutputParser)
    * [parse](#agent_engine_sdk.models.OutputParser.parse)
    * [on\_stream\_error](#agent_engine_sdk.models.OutputParser.on_stream_error)
  * [RequestContext](#agent_engine_sdk.models.RequestContext)
  * [SessionSummary](#agent_engine_sdk.models.SessionSummary)
  * [SessionsSummaryResponse](#agent_engine_sdk.models.SessionsSummaryResponse)
  * [SessionMessage](#agent_engine_sdk.models.SessionMessage)
  * [SessionMessagesResponse](#agent_engine_sdk.models.SessionMessagesResponse)
  * [SessionForkResponse](#agent_engine_sdk.models.SessionForkResponse)

Base application class for framework-specific SDKs.

<a id="agent_engine_sdk.app.BaseApp"></a>

## BaseApp

```python
class BaseApp(ABC)
```

Abstract base class for framework-specific SDK integrations.

Each framework SDK (LangGraph, CrewAI, etc.) subclasses ``BaseApp``
and implements the abstract methods using framework-native constructs.

The platform runtime sets ``tool_wrapper`` and ``llm_wrapper`` on the
instance before calling ``entrypoint()`` so that tools and LLMs are
routed through the secure execution layer.

<a id="agent_engine_sdk.app.BaseApp.get_tool_definitions"></a>

#### get\_tool\_definitions

```python
@abstractmethod
def get_tool_definitions() -> list[ToolDefinition]
```

Return all registered tool definitions.

Used by the OE to obtain tool execution information.

<a id="agent_engine_sdk.app.BaseApp.tools"></a>

#### tools

```python
@abstractmethod
def tools() -> Any
```

Return a list of wrapped, framework-specific tools.

<a id="agent_engine_sdk.app.BaseApp.tool"></a>

#### tool

```python
@abstractmethod
def tool(*args: Any, **kwargs: Any) -> Any
```

Decorator that registers a tool on this app.

<a id="agent_engine_sdk.app.BaseApp.entrypoint"></a>

#### entrypoint

```python
@abstractmethod
def entrypoint(fn: Any) -> Any
```

Decorator that registers the agent builder function.

Called by runtimes to obtain a ``BaseAgent``.

Framework-neutral protocols for Atlas Agent Engine.

<a id="agent_engine_sdk.interfaces.ExecutionResult"></a>

## ExecutionResult

```python
@runtime_checkable
class ExecutionResult(AsyncIterable[StreamEvent], Protocol)
```

Result of an agent execution.

Await for the final result, or async-iterate for streaming events.

Usage::

    # Non-streaming
    output = await agent.execute(ctx, input)

    # Streaming
    async for event in agent.execute(ctx, input):
        handle(event)

<a id="agent_engine_sdk.interfaces.BaseAgent"></a>

## BaseAgent

```python
@runtime_checkable
class BaseAgent(Protocol)
```

The contract between agent code and runtimes.

Agents expose a single ``execute()`` method that returns a ``ExecutionResult``.
Callers choose the execution mode:

- ``await agent.execute(ctx, input)`` for a final result (JSON-style)
- ``async for event in agent.execute(ctx, input)`` for streaming (SSE-style)

<a id="agent_engine_sdk.interfaces.BaseLLM"></a>

## BaseLLM

```python
@runtime_checkable
class BaseLLM(Protocol)
```

Framework-neutral LLM protocol.

Used at runtime by the ToolPod.

<a id="agent_engine_sdk.interfaces.BaseExecutionCallback"></a>

## BaseExecutionCallback

```python
@runtime_checkable
class BaseExecutionCallback(Protocol)
```

Framework-neutral callback for observability during agent execution.

Replaces LangChain's ``BaseCallbackHandler``. The AER injects an
implementation that forwards node events to the OE for logging.
Framework adapters (e.g. sdk-langgraph) bridge from the framework's
native callback system to this protocol.

Prefer subclassing ``NullExecutionCallback`` over implementing this
Protocol directly — it provides no-op defaults for all methods including
optional extensions (e.g. ``on_node_suspend``) so new methods never break
existing implementations.

<a id="agent_engine_sdk.interfaces.NullExecutionCallback"></a>

## NullExecutionCallback

```python
class NullExecutionCallback()
```

Concrete base class with no-op defaults for all BaseExecutionCallback methods.

Subclass this instead of implementing BaseExecutionCallback directly so that
new methods added to the protocol are automatically satisfied — you only
override the events you care about.

**Example**:


```python
class MyCallback(NullExecutionCallback):
    def on_node_start(self, node_name, inputs, *, run_id, **kwargs):
        print(f"starting {node_name}")
```

Framework-neutral Pydantic models for Atlas Agent Engine.

<a id="agent_engine_sdk.models.MappingCompatModel"></a>

## MappingCompatModel

```python
class MappingCompatModel(BaseModel)
```

Pydantic model with light dict-like compatibility helpers.

Equality to dicts is intentionally one-directional compatibility for older
call sites that compare ``model == dict``. Prefer explicit ``model_dump()``
comparisons in new code.

<a id="agent_engine_sdk.models.ToolDefinition"></a>

## ToolDefinition

```python
class ToolDefinition(BaseModel)
```

Framework-neutral tool definition.

<a id="agent_engine_sdk.models.TextBlock"></a>

## TextBlock

```python
class TextBlock(BaseModel)
```

Text content block.

<a id="agent_engine_sdk.models.ImageBlock"></a>

## ImageBlock

```python
class ImageBlock(BaseModel)
```

Image content block.

<a id="agent_engine_sdk.models.DocumentBlock"></a>

## DocumentBlock

```python
class DocumentBlock(BaseModel)
```

Document/file content block.

<a id="agent_engine_sdk.models.LLMTokenUsage"></a>

## LLMTokenUsage

```python
class LLMTokenUsage(MappingCompatModel)
```

Typed token-usage metadata for LLM calls.

<a id="agent_engine_sdk.models.LLMTokenUsage.to_langchain_usage_metadata"></a>

#### to\_langchain\_usage\_metadata

```python
def to_langchain_usage_metadata() -> dict[str, int]
```

Return LangChain's expected usage-metadata keys.

<a id="agent_engine_sdk.models.LLMToolCall"></a>

## LLMToolCall

```python
class LLMToolCall(MappingCompatModel)
```

Typed final tool call requested by an LLM.

<a id="agent_engine_sdk.models.LLMToolCall.to_langchain_dict"></a>

#### to\_langchain\_dict

```python
def to_langchain_dict() -> dict[str, JsonValue]
```

Return a LangChain-compatible tool-call dict.

<a id="agent_engine_sdk.models.LLMToolSchema"></a>

## LLMToolSchema

```python
class LLMToolSchema(MappingCompatModel)
```

Serializable bound-tool schema forwarded with an LLM call.

<a id="agent_engine_sdk.models.LLMToolSchema.to_langchain_dict"></a>

#### to\_langchain\_dict

```python
def to_langchain_dict() -> dict[str, JsonValue]
```

Return the schema in the shape LangChain bind_tools expects.

<a id="agent_engine_sdk.models.LLMInvocationOptions"></a>

## LLMInvocationOptions

```python
class LLMInvocationOptions(BaseModel)
```

Explicit provider/model options passed with an LLM invocation.

<a id="agent_engine_sdk.models.LLMInvocationOptions.to_model_kwargs"></a>

#### to\_model\_kwargs

```python
def to_model_kwargs() -> dict[str, Any]
```

Convert to kwargs for underlying model invocation.

<a id="agent_engine_sdk.models.Message"></a>

## Message

```python
class Message(MappingCompatModel)
```

Framework-neutral message.

Content can be a simple string for text-only messages, or a list of
content blocks for multimodal content (images, documents, etc.).

<a id="agent_engine_sdk.models.MessageArtifact"></a>

## MessageArtifact

```python
class MessageArtifact(MappingCompatModel)
```

Structured artifact attached to an assistant message.

Attach artifacts through ``MessageArtifactMetadata.to_message_metadata()``
on ``Message.additional_kwargs``. Platform and custom frontends can render
the same transported artifact payloads. Artifact fields must be JSON-safe;
binary data should be represented as a URL or an encoded string.

<a id="agent_engine_sdk.models.MessageArtifactMetadata"></a>

## MessageArtifactMetadata

```python
class MessageArtifactMetadata(MappingCompatModel)
```

Typed metadata contract for message-scoped artifacts.

<a id="agent_engine_sdk.models.MessageArtifactMetadata.to_message_metadata"></a>

#### to\_message\_metadata

```python
def to_message_metadata() -> dict[str, JsonValue]
```

Return the dict to attach to ``Message.additional_kwargs``.

<a id="agent_engine_sdk.models.collect_message_artifact_metadata"></a>

#### collect\_message\_artifact\_metadata

```python
def collect_message_artifact_metadata(
        source: Mapping[str, Any] | None) -> MessageArtifactMetadata | None
```

Collect message artifact metadata from ``Message.additional_kwargs``.

<a id="agent_engine_sdk.models.LLMResponse"></a>

## LLMResponse

```python
class LLMResponse(MappingCompatModel)
```

Framework-neutral LLM response.

<a id="agent_engine_sdk.models.ToolCallChunk"></a>

## ToolCallChunk

```python
class ToolCallChunk(BaseModel)
```

Partial tool call during streaming.

All fields are optional since chunks may contain partial data
(e.g., just the start of arguments, or just the tool name).

<a id="agent_engine_sdk.models.LLMStreamChunk"></a>

## LLMStreamChunk

```python
class LLMStreamChunk(BaseModel)
```

Rich streaming chunk following OpenAI/Anthropic patterns.

All fields are optional since different chunks contain different data:
- Content chunks: text deltas during generation
- Tool call chunks: partial tool calls being built up
- Usage chunks: token counts at end of stream

<a id="agent_engine_sdk.models.AgentInput"></a>

## AgentInput

```python
class AgentInput(BaseModel)
```

Input for an agent execution. Payload is an opaque JSON value.

<a id="agent_engine_sdk.models.AgentOutput"></a>

## AgentOutput

```python
class AgentOutput(BaseModel)
```

Output from an agent execution. Response is an opaque JSON value.

<a id="agent_engine_sdk.models.StreamEvent"></a>

## StreamEvent

```python
class StreamEvent(BaseModel)
```

Event streamed back to the caller during execution.

<a id="agent_engine_sdk.models.OutputParser"></a>

## OutputParser

```python
class OutputParser(ABC, Generic[StreamItemT])
```

Author-supplied map from raw stream items to consumer-facing events.

Framework-neutral: ``StreamItemT`` is the type of one item pulled from the
underlying stream, which a framework binding fixes when it subclasses this.
Both methods are abstract, so a subclass that omits either cannot be
instantiated.

A yielded ``BaseModel`` is serialized to JSON by the producer before it
populates ``StreamEvent.custom_event`` (typed ``JsonValue``); yielding a
plain JSON value populates it directly.

<a id="agent_engine_sdk.models.OutputParser.parse"></a>

#### parse

```python
@abstractmethod
async def parse(item: StreamItemT,
                ctx: RequestContext) -> AsyncIterator[BaseModel | JsonValue]
```

Yield zero or more custom events for ``item``.

<a id="agent_engine_sdk.models.OutputParser.on_stream_error"></a>

#### on\_stream\_error

```python
@abstractmethod
async def on_stream_error(
        ctx: RequestContext,
        error: BaseException) -> BaseModel | JsonValue | None
```

Return a final custom event before the stream closes on error, or None.

<a id="agent_engine_sdk.models.RequestContext"></a>

## RequestContext

```python
class RequestContext(BaseModel)
```

Request-scoped context passed to every agent execution.

Carries framework-neutral execution identity and platform plumbing. The
caller's invocation input lives in ``AgentInput.payload``, not here.
Suspend/resume state is opaque to the platform: ``metadata`` is a
framework-owned blob that the platform forwards unchanged on resume.

<a id="agent_engine_sdk.models.SessionSummary"></a>

## SessionSummary

```python
class SessionSummary(BaseModel)
```

Per-session summary sourced from framework-specific persistence.

Returned by the AER's session-enrichment endpoint. The platform layer
(OE) merges this with its own tenant fields (user_id, project_id,
workspace_id, visibility) when assembling the public sessions list.

<a id="agent_engine_sdk.models.SessionsSummaryResponse"></a>

## SessionsSummaryResponse

```python
class SessionsSummaryResponse(BaseModel)
```

Response for the AER's session-summary enrichment endpoint.

<a id="agent_engine_sdk.models.SessionMessage"></a>

## SessionMessage

```python
class SessionMessage(BaseModel)
```

A single message in a session's conversation history.

``role`` uses the same vocabulary as ``Message`` ("user", "assistant",
"tool", "system"). ``name`` carries the tool name on tool messages.

<a id="agent_engine_sdk.models.SessionMessagesResponse"></a>

## SessionMessagesResponse

```python
class SessionMessagesResponse(BaseModel)
```

Response for the AER's session-messages endpoint.

<a id="agent_engine_sdk.models.SessionForkResponse"></a>

## SessionForkResponse

```python
class SessionForkResponse(BaseModel)
```

Response for forking a session into a new independent session.
