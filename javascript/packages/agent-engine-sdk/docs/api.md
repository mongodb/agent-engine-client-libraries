# @mongodb-js/agent-engine-sdk

## Table of Contents

- **Classes**
  - [`abstract` BaseApp](#api-abstract-baseapp)
    - [entrypoint()](#api-entrypoint)
    - [getToolDefinitions()](#api-gettooldefinitions)
    - [tool()](#api-tool)
    - [tools()](#api-tools)
  - [EventClient](#api-eventclient)
    - [appendEvent()](#api-appendevent)
    - [getEvent()](#api-getevent)
    - [listEvents()](#api-listevents)
  - [LLMInvocationOptions](#api-llminvocationoptions)
    - [toJSON()](#api-tojson)
    - [toModelKwargs()](#api-tomodelkwargs)
  - [LLMResponse](#api-llmresponse)
    - [fromRaw()](#api-fromraw)
  - [LLMTokenUsage](#api-llmtokenusage)
    - [toJSON()](#api-tojson-1)
    - [toLangchainUsageMetadata()](#api-tolangchainusagemetadata)
  - [LLMToolCall](#api-llmtoolcall)
    - [toLangchainDict()](#api-tolangchaindict)
  - [LLMToolSchema](#api-llmtoolschema)
    - [toLangchainDict()](#api-tolangchaindict-1)
  - [MemoryClient](#api-memoryclient)
    - [bootstrap()](#api-bootstrap)
    - [buildContext()](#api-buildcontext)
    - [createCustom()](#api-createcustom)
    - [createEpisodic()](#api-createepisodic)
    - [createProcedural()](#api-createprocedural)
    - [createSemantic()](#api-createsemantic)
    - [createTaxonomic()](#api-createtaxonomic)
    - [deleteProcedural()](#api-deleteprocedural)
    - [discoverProcedures()](#api-discoverprocedures)
    - [fetchEpisodicMemories()](#api-fetchepisodicmemories)
    - [fetchProceduralMemories()](#api-fetchproceduralmemories)
    - [fetchSemanticMemories()](#api-fetchsemanticmemories)
    - [fetchTaxonomicMemories()](#api-fetchtaxonomicmemories)
    - [getDistinctDomains()](#api-getdistinctdomains)
    - [getProcedural()](#api-getprocedural)
    - [getSemantic()](#api-getsemantic)
    - [getTaxonomic()](#api-gettaxonomic)
    - [listEpisodic()](#api-listepisodic)
    - [retrieveCustom()](#api-retrievecustom)
    - [updateProcedural()](#api-updateprocedural)
    - [updateSemantic()](#api-updatesemantic)
    - [writeTurn()](#api-writeturn)
  - [MemoryHttpError](#api-memoryhttperror)
  - [NullExecutionCallback](#api-nullexecutioncallback)
    - [onNodeEnd()](#api-onnodeend)
    - [onNodeError()](#api-onnodeerror)
    - [onNodeStart()](#api-onnodestart)
    - [onNodeSuspend()](#api-onnodesuspend)
- **Interfaces**
  - [AgentInput](#api-agentinput)
  - [AgentOutput](#api-agentoutput)
  - [BaseAgent](#api-baseagent)
    - [execute()](#api-execute)
  - [BaseExecutionCallback](#api-baseexecutioncallback)
    - [onNodeEnd()](#api-onnodeend-1)
    - [onNodeError()](#api-onnodeerror-1)
    - [onNodeStart()](#api-onnodestart-1)
  - [BaseLLM](#api-basellm)
    - [invoke()](#api-invoke)
    - [stream()](#api-stream)
  - [ExecutionResult](#api-executionresult)
  - [LLMStreamChunk](#api-llmstreamchunk)
  - [Message](#api-message)
  - [RequestContext](#api-requestcontext)
  - [StreamEvent](#api-streamevent)
  - [ToolCallChunk](#api-toolcallchunk)
- **Type Aliases**
  - [AnyContentBlock](#api-anycontentblock)
  - [BranchRef](#api-branchref)
  - [BulkCreateSemanticResult](#api-bulkcreatesemanticresult)
  - [ContextConfig](#api-contextconfig)
  - [ContextMetadata](#api-contextmetadata)
  - [ContextResponse](#api-contextresponse)
  - [CreateEpisodicResult](#api-createepisodicresult)
  - [CreateProceduralResult](#api-createproceduralresult)
  - [CreateSemanticResult](#api-createsemanticresult)
  - [CreateSnapshotResult](#api-createsnapshotresult)
  - [CreateTaxonomicResult](#api-createtaxonomicresult)
  - [CreateUserContextResult](#api-createusercontextresult)
  - [CustomMemoryRetrieveResult](#api-custommemoryretrieveresult)
  - [CustomMemorySaveResult](#api-custommemorysaveresult)
  - [DeleteResult](#api-deleteresult)
  - [DocumentBlock](#api-documentblock)
  - [Event](#api-event)
  - [EventResponse](#api-eventresponse)
  - [EventsResponse](#api-eventsresponse)
  - [FormatStyle](#api-formatstyle)
  - [ImageBlock](#api-imageblock)
  - [InternalStateResult](#api-internalstateresult)
  - [ISGenerationResult](#api-isgenerationresult)
  - [JsonValue](#api-jsonvalue)
  - [MemoryChunk](#api-memorychunk)
  - [MemoryRouteStyle](#api-memoryroutestyle)
  - [MemorySource](#api-memorysource)
  - [ModelType](#api-modeltype)
  - [PromoteSnapshotResult](#api-promotesnapshotresult)
  - [RetrievedCustomMemory](#api-retrievedcustommemory)
  - [Role](#api-role)
  - [SessionMessage](#api-sessionmessage)
  - [SessionMessagesResponse](#api-sessionmessagesresponse)
  - [SessionsSummaryResponse](#api-sessionssummaryresponse)
  - [SessionSummary](#api-sessionsummary)
  - [TagScalar](#api-tagscalar)
  - [TextBlock](#api-textblock)
  - [ToolDefinition](#api-tooldefinition)
  - [ToolDefinitionInput](#api-tooldefinitioninput)
  - [WriteTurnResult](#api-writeturnresult)
- **Variables**
  - [AgentInputSchema](#api-agentinputschema)
  - [AgentOutputSchema](#api-agentoutputschema)
  - [AnyContentBlockSchema](#api-anycontentblockschema)
  - [BranchRefSchema](#api-branchrefschema)
  - [BulkCreateSemanticResultSchema](#api-bulkcreatesemanticresultschema)
  - [ContextConfigSchema](#api-contextconfigschema)
  - [ContextMetadataSchema](#api-contextmetadataschema)
  - [ContextResponseSchema](#api-contextresponseschema)
  - [CreateEpisodicResultSchema](#api-createepisodicresultschema)
  - [CreateProceduralResultSchema](#api-createproceduralresultschema)
  - [CreateSemanticResultSchema](#api-createsemanticresultschema)
  - [CreateSnapshotResultSchema](#api-createsnapshotresultschema)
  - [CreateTaxonomicResultSchema](#api-createtaxonomicresultschema)
  - [CreateUserContextResultSchema](#api-createusercontextresultschema)
  - [CustomMemoryRetrieveResultSchema](#api-custommemoryretrieveresultschema)
  - [CustomMemorySaveResultSchema](#api-custommemorysaveresultschema)
  - [DateFromStringSchema](#api-datefromstringschema)
  - [DeleteResultSchema](#api-deleteresultschema)
  - [DocumentBlockSchema](#api-documentblockschema)
  - [EventResponseSchema](#api-eventresponseschema)
  - [EventSchema](#api-eventschema)
  - [EventsResponseSchema](#api-eventsresponseschema)
  - [FormatStyle](#api-formatstyle-1)
  - [FormatStyleSchema](#api-formatstyleschema)
  - [ImageBlockSchema](#api-imageblockschema)
  - [InternalStateResultSchema](#api-internalstateresultschema)
  - [ISGenerationResultSchema](#api-isgenerationresultschema)
  - [JsonValueSchema](#api-jsonvalueschema)
  - [LLMInvocationOptionsSchema](#api-llminvocationoptionsschema)
  - [LLMResponseSchema](#api-llmresponseschema)
  - [LLMTokenUsageSchema](#api-llmtokenusageschema)
  - [LLMToolCallSchema](#api-llmtoolcallschema)
  - [LLMToolSchemaValidator](#api-llmtoolschemavalidator)
  - [MemoryChunkSchema](#api-memorychunkschema)
  - [MemorySource](#api-memorysource-1)
  - [MemorySourceSchema](#api-memorysourceschema)
  - [MessageSchema](#api-messageschema)
  - [ModelType](#api-modeltype-1)
  - [ModelTypeSchema](#api-modeltypeschema)
  - [PromoteSnapshotResultSchema](#api-promotesnapshotresultschema)
  - [RetrievedCustomMemorySchema](#api-retrievedcustommemoryschema)
  - [SessionMessageSchema](#api-sessionmessageschema)
  - [SessionMessagesResponseSchema](#api-sessionmessagesresponseschema)
  - [SessionsSummaryResponseSchema](#api-sessionssummaryresponseschema)
  - [SessionSummarySchema](#api-sessionsummaryschema)
  - [StreamEventSchema](#api-streameventschema)
  - [TagScalarSchema](#api-tagscalarschema)
  - [TextBlockSchema](#api-textblockschema)
  - [ToolCallChunkSchema](#api-toolcallchunkschema)
  - [ToolDefinitionSchema](#api-tooldefinitionschema)
  - [WriteTurnResultSchema](#api-writeturnresultschema)
- **Functions**
  - [createToolDefinition()](#api-createtooldefinition)
  - [serializeMessage()](#api-serializemessage)

## Classes

<a id="api-abstract-baseapp"></a>

### `abstract` BaseApp

Abstract base class for framework-specific SDK integrations.

Each framework SDK (LangGraph, CrewAI, etc.) subclasses `BaseApp`
and implements the abstract methods using framework-native constructs.

The platform runtime sets `toolWrapper` and `llmWrapper` on the
instance before calling `entrypoint()` so that tools and LLMs are
routed through the secure execution layer.

<a id="api-constructor"></a>

#### Constructor

```ts
new BaseApp(name): BaseApp;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `name` | `string` |

**Returns**

[`BaseApp`](#api-abstract-baseapp)

#### Properties

| Property | Modifier | Type | Default value |
| :------ | :------ | :------ | :------ |
| <a id="api-property-llmwrapper"></a> `llmWrapper` | `public` | `unknown` | `null` |
| <a id="api-property-name"></a> `name` | `readonly` | `string` | `undefined` |
| <a id="api-property-toolwrapper"></a> `toolWrapper` | `public` | `unknown` | `null` |

#### Methods

<a id="api-entrypoint"></a>

##### entrypoint()

```ts
abstract entrypoint(fn): unknown;
```

Decorator that registers the agent builder function. Called by runtimes to obtain a BaseAgent.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `fn` | `unknown` |

**Returns**

`unknown`

<a id="api-gettooldefinitions"></a>

##### getToolDefinitions()

```ts
abstract getToolDefinitions(): object[];
```

Returns all registered tool definitions. Used by the OE to obtain tool execution information.

**Returns**

`object`[]

<a id="api-tool"></a>

##### tool()

```ts
abstract tool(...args): unknown;
```

Decorator that registers a tool on this app.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| ...`args` | `unknown`[] |

**Returns**

`unknown`

<a id="api-tools"></a>

##### tools()

```ts
abstract tools(): unknown[];
```

Returns a list of wrapped, framework-specific tools.

**Returns**

`unknown`[]

***

<a id="api-eventclient"></a>

### EventClient

<a id="api-constructor-1"></a>

#### Constructor

```ts
new EventClient(oeUrl, timeout?): EventClient;
```

**Parameters**

| Parameter | Type | Default value |
| :------ | :------ | :------ |
| `oeUrl` | `string` | `undefined` |
| `timeout` | `number` | `30` |

**Returns**

[`EventClient`](#api-eventclient)

#### Methods

<a id="api-appendevent"></a>

##### appendEvent()

```ts
appendEvent(
   sessionId,
   actorId,
   payload,
   parentEventId?,
   metadata?,
   branch?
): Promise<{
[key: string]: unknown;
  actor_id: string;
  branch?: {
   [key: string]: unknown;
     name: string;
     root_event_id: string;
  };
  event_id: string;
  metadata?: Record<string, string>;
  parent_event_id?: string;
  payload:   | string
     | number
     | boolean
     | JsonValue[]
     | {
   [key: string]: JsonValue;
   }
     | null;
  session_id: string;
  timestamp: Date;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `sessionId` | `string` |
| `actorId` | `string` |
| `payload` | `Record`\<`string`, `unknown`\> |
| `parentEventId?` | `string` |
| `metadata?` | `Record`\<`string`, `unknown`\> |
| `branch?` | \{ \[`key`: `string`\]: `unknown`; `name`: `string`; `root_event_id`: `string`; \} |
| `branch.name?` | `string` |
| `branch.root_event_id?` | `string` |

**Returns**

`Promise`\<\{
\[`key`: `string`\]: `unknown`;
  `actor_id`: `string`;
  `branch?`: \{
   \[`key`: `string`\]: `unknown`;
     `name`: `string`;
     `root_event_id`: `string`;
  \};
  `event_id`: `string`;
  `metadata?`: `Record`\<`string`, `string`\>;
  `parent_event_id?`: `string`;
  `payload`:   \| `string`
     \| `number`
     \| `boolean`
     \| [`JsonValue`](#api-jsonvalue)[]
     \| \{
   \[`key`: `string`\]: [`JsonValue`](#api-jsonvalue);
   \}
     \| `null`;
  `session_id`: `string`;
  `timestamp`: `Date`;
\}\>

<a id="api-getevent"></a>

##### getEvent()

```ts
getEvent(sessionId, eventId): Promise<
  | {
[key: string]: unknown;
  actor_id: string;
  branch?: {
   [key: string]: unknown;
     name: string;
     root_event_id: string;
  };
  event_id: string;
  metadata?: Record<string, string>;
  parent_event_id?: string;
  payload:   | string
     | number
     | boolean
     | JsonValue[]
     | {
   [key: string]: JsonValue;
   }
     | null;
  session_id: string;
  timestamp: Date;
}
| null>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `sessionId` | `string` |
| `eventId` | `string` |

**Returns**

`Promise`\<
  \| \{
\[`key`: `string`\]: `unknown`;
  `actor_id`: `string`;
  `branch?`: \{
   \[`key`: `string`\]: `unknown`;
     `name`: `string`;
     `root_event_id`: `string`;
  \};
  `event_id`: `string`;
  `metadata?`: `Record`\<`string`, `string`\>;
  `parent_event_id?`: `string`;
  `payload`:   \| `string`
     \| `number`
     \| `boolean`
     \| [`JsonValue`](#api-jsonvalue)[]
     \| \{
   \[`key`: `string`\]: [`JsonValue`](#api-jsonvalue);
   \}
     \| `null`;
  `session_id`: `string`;
  `timestamp`: `Date`;
\}
  \| `null`\>

<a id="api-listevents"></a>

##### listEvents()

```ts
listEvents(
   sessionId,
   branch?,
   metadataFilters?
): Promise<object[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `sessionId` | `string` |
| `branch?` | `string` |
| `metadataFilters?` | `Record`\<`string`, `string`\> |

**Returns**

`Promise`\<`object`[]\>

***

<a id="api-llminvocationoptions"></a>

### LLMInvocationOptions

Explicit provider/model options passed with an LLM invocation.

<a id="api-constructor-2"></a>

#### Constructor

```ts
new LLMInvocationOptions(data): LLMInvocationOptions;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `data` | \{ \[`key`: `string`\]: `unknown`; `frequency_penalty?`: `number`; `max_tokens?`: `number`; `parallel_tool_calls?`: `boolean`; `presence_penalty?`: `number`; `reasoning_effort?`: `string`; `response_format?`: [`JsonValue`](#api-jsonvalue); `seed?`: `number`; `timeout?`: `number`; `top_k?`: `number`; `top_p?`: `number`; \} |
| `data.frequency_penalty?` | `number` |
| `data.max_tokens?` | `number` |
| `data.parallel_tool_calls?` | `boolean` |
| `data.presence_penalty?` | `number` |
| `data.reasoning_effort?` | `string` |
| `data.response_format?` | [`JsonValue`](#api-jsonvalue) |
| `data.seed?` | `number` |
| `data.timeout?` | `number` |
| `data.top_k?` | `number` |
| `data.top_p?` | `number` |

**Returns**

[`LLMInvocationOptions`](#api-llminvocationoptions)

#### Properties

| Property | Modifier | Type | Description |
| :------ | :------ | :------ | :------ |
| <a id="api-property-extras"></a> `extras` | `readonly` | `Record`\<`string`, `unknown`\> | Provider-specific kwargs not in the standard field set (e.g. temperature). Mirrors Python's extra="allow". |
| <a id="api-property-frequencypenalty"></a> `frequencyPenalty?` | `readonly` | `number` | - |
| <a id="api-property-maxtokens"></a> `maxTokens?` | `readonly` | `number` | - |
| <a id="api-property-paralleltoolcalls"></a> `parallelToolCalls?` | `readonly` | `boolean` | - |
| <a id="api-property-presencepenalty"></a> `presencePenalty?` | `readonly` | `number` | - |
| <a id="api-property-reasoningeffort"></a> `reasoningEffort?` | `readonly` | `string` | - |
| <a id="api-property-responseformat"></a> `responseFormat?` | `readonly` | [`JsonValue`](#api-jsonvalue) | - |
| <a id="api-property-seed"></a> `seed?` | `readonly` | `number` | - |
| <a id="api-property-timeout"></a> `timeout?` | `readonly` | `number` | - |
| <a id="api-property-topk"></a> `topK?` | `readonly` | `number` | - |
| <a id="api-property-topp"></a> `topP?` | `readonly` | `number` | - |

#### Methods

<a id="api-tojson"></a>

##### toJSON()

```ts
toJSON(): Record<string, unknown>;
```

Serializes to snake_case wire format so JSON.stringify round-trips through LLMInvocationOptionsSchema.

**Returns**

`Record`\<`string`, `unknown`\>

<a id="api-tomodelkwargs"></a>

##### toModelKwargs()

```ts
toModelKwargs(): Record<string, unknown>;
```

Converts to kwargs for underlying model invocation. Mirrors Python's model_dump(exclude_none=True) with extra="allow".

**Returns**

`Record`\<`string`, `unknown`\>

***

<a id="api-llmresponse"></a>

### LLMResponse

Framework-neutral LLM response.

<a id="api-constructor-3"></a>

#### Constructor

```ts
new LLMResponse(data): LLMResponse;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `data` | \{ `additionalKwargs?`: `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\>; `content`: `string`; `id?`: `string`; `metadata?`: `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\>; `name?`: `string`; `responseMetadata?`: `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\>; `toolCalls?`: [`LLMToolCall`](#api-llmtoolcall)[]; `usage?`: [`LLMTokenUsage`](#api-llmtokenusage); \} |
| `data.additionalKwargs?` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| `data.content` | `string` |
| `data.id?` | `string` |
| `data.metadata?` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| `data.name?` | `string` |
| `data.responseMetadata?` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| `data.toolCalls?` | [`LLMToolCall`](#api-llmtoolcall)[] |
| `data.usage?` | [`LLMTokenUsage`](#api-llmtokenusage) |

**Returns**

[`LLMResponse`](#api-llmresponse)

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-additionalkwargs"></a> `additionalKwargs?` | `readonly` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| <a id="api-property-content"></a> `content` | `readonly` | `string` |
| <a id="api-property-id"></a> `id?` | `readonly` | `string` |
| <a id="api-property-metadata"></a> `metadata` | `readonly` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| <a id="api-property-name-1"></a> `name?` | `readonly` | `string` |
| <a id="api-property-responsemetadata"></a> `responseMetadata?` | `readonly` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| <a id="api-property-toolcalls"></a> `toolCalls?` | `readonly` | [`LLMToolCall`](#api-llmtoolcall)[] |
| <a id="api-property-usage"></a> `usage?` | `readonly` | [`LLMTokenUsage`](#api-llmtokenusage) |

#### Methods

<a id="api-fromraw"></a>

##### fromRaw()

```ts
static fromRaw(data): LLMResponse;
```

Constructs an LLMResponse from a raw provider dict, normalizing usage
from either a top-level 'usage' key or token keys embedded in 'metadata'.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `data` | `Record`\<`string`, `unknown`\> |

**Returns**

[`LLMResponse`](#api-llmresponse)

***

<a id="api-llmtokenusage"></a>

### LLMTokenUsage

Typed token-usage metadata for LLM calls.

<a id="api-constructor-4"></a>

#### Constructor

```ts
new LLMTokenUsage(data): LLMTokenUsage;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `data` | \{ `completion_tokens?`: `number`; `input_tokens?`: `number`; `model?`: `string`; `output_tokens?`: `number`; `prompt_tokens?`: `number`; `total_tokens?`: `number`; \} |
| `data.completion_tokens?` | `number` |
| `data.input_tokens?` | `number` |
| `data.model?` | `string` |
| `data.output_tokens?` | `number` |
| `data.prompt_tokens?` | `number` |
| `data.total_tokens?` | `number` |

**Returns**

[`LLMTokenUsage`](#api-llmtokenusage)

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-completiontokens"></a> `completionTokens?` | `readonly` | `number` |
| <a id="api-property-inputtokens"></a> `inputTokens?` | `readonly` | `number` |
| <a id="api-property-model"></a> `model?` | `readonly` | `string` |
| <a id="api-property-outputtokens"></a> `outputTokens?` | `readonly` | `number` |
| <a id="api-property-prompttokens"></a> `promptTokens?` | `readonly` | `number` |
| <a id="api-property-totaltokens"></a> `totalTokens?` | `public` | `number` |

#### Methods

<a id="api-tojson-1"></a>

##### toJSON()

```ts
toJSON(): Record<string, unknown>;
```

Serializes to snake_case wire format so JSON.stringify round-trips through LLMTokenUsageSchema.

**Returns**

`Record`\<`string`, `unknown`\>

<a id="api-tolangchainusagemetadata"></a>

##### toLangchainUsageMetadata()

```ts
toLangchainUsageMetadata(): object;
```

Returns LangChain's expected usage-metadata keys.

**Returns**

`object`

| Name | Type |
| :------ | :------ |
| `input_tokens` | `number` |
| `output_tokens` | `number` |
| `total_tokens` | `number` |

***

<a id="api-llmtoolcall"></a>

### LLMToolCall

Typed final tool call requested by an LLM.

<a id="api-constructor-5"></a>

#### Constructor

```ts
new LLMToolCall(data): LLMToolCall;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `data` | \{ `args?`: [`JsonValue`](#api-jsonvalue); `arguments?`: [`JsonValue`](#api-jsonvalue); `id?`: `string`; `index?`: `number`; `name?`: `string`; `type?`: `string`; \} |
| `data.args?` | [`JsonValue`](#api-jsonvalue) |
| `data.arguments?` | [`JsonValue`](#api-jsonvalue) |
| `data.id?` | `string` |
| `data.index?` | `number` |
| `data.name?` | `string` |
| `data.type?` | `string` |

**Returns**

[`LLMToolCall`](#api-llmtoolcall)

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-args"></a> `args?` | `readonly` | [`JsonValue`](#api-jsonvalue) |
| <a id="api-property-id-1"></a> `id?` | `readonly` | `string` |
| <a id="api-property-index"></a> `index?` | `readonly` | `number` |
| <a id="api-property-name-2"></a> `name?` | `readonly` | `string` |
| <a id="api-property-type"></a> `type?` | `readonly` | `string` |

#### Methods

<a id="api-tolangchaindict"></a>

##### toLangchainDict()

```ts
toLangchainDict(): Record<string, JsonValue>;
```

Returns a LangChain-compatible tool-call dict (excludes index).

**Returns**

`Record`\<`string`, [`JsonValue`](#api-jsonvalue)\>

***

<a id="api-llmtoolschema"></a>

### LLMToolSchema

Serializable bound-tool schema forwarded with an LLM call.

<a id="api-constructor-6"></a>

#### Constructor

```ts
new LLMToolSchema(data): LLMToolSchema;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `data` | \{ `description?`: `string`; `function?`: [`JsonValue`](#api-jsonvalue); `name?`: `string`; `parameters?`: [`JsonValue`](#api-jsonvalue); `strict?`: `boolean`; `type?`: `string`; \} |
| `data.description?` | `string` |
| `data.function?` | [`JsonValue`](#api-jsonvalue) |
| `data.name?` | `string` |
| `data.parameters?` | [`JsonValue`](#api-jsonvalue) |
| `data.strict?` | `boolean` |
| `data.type?` | `string` |

**Returns**

[`LLMToolSchema`](#api-llmtoolschema)

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-description"></a> `description?` | `readonly` | `string` |
| <a id="api-property-function"></a> `function?` | `readonly` | [`JsonValue`](#api-jsonvalue) |
| <a id="api-property-name-3"></a> `name?` | `readonly` | `string` |
| <a id="api-property-parameters"></a> `parameters?` | `readonly` | [`JsonValue`](#api-jsonvalue) |
| <a id="api-property-strict"></a> `strict?` | `readonly` | `boolean` |
| <a id="api-property-type-1"></a> `type?` | `readonly` | `string` |

#### Methods

<a id="api-tolangchaindict-1"></a>

##### toLangchainDict()

```ts
toLangchainDict(): Record<string, JsonValue>;
```

Returns the schema in the shape LangChain bind_tools expects.

**Returns**

`Record`\<`string`, [`JsonValue`](#api-jsonvalue)\>

***

<a id="api-memoryclient"></a>

### MemoryClient

HTTP client for the Memory Server.

```ts
const client = new MemoryClient('http://127.0.0.1:8081')

// Write conversation turn
await client.writeTurn({ sessionId: 'thread_123', role: 'user', content: 'Hello', orgId: 'org_1', userId: 'user_1', projectId: 'proj_1' })

// Build context
const context = await client.buildContext({ query: 'What did we discuss?', sessionId: 'thread_123', orgId: 'org_1', userId: 'user_1', projectId: 'proj_1' })
```

<a id="api-constructor-7"></a>

#### Constructor

```ts
new MemoryClient(
   baseUrl,
   timeout?,
   staticHeaders?,
   apiPrefix?,
   dynamicHeaders?,
   requestExtras?,
   maxRetries?,
   routeStyle?,
   fetchImpl?
): MemoryClient;
```

**Parameters**

| Parameter | Type | Default value | Description |
| :------ | :------ | :------ | :------ |
| `baseUrl` | `string` | `undefined` | Memory server base URL (e.g. "http://127.0.0.1:8081") |
| `timeout` | `number` | `30` | Request timeout in seconds (default: 30) |
| `staticHeaders` | `Record`\<`string`, `string`\> | `{}` | Pod-level identity headers set at construction time (e.g. `X-Agent-Engine-Agent-Id` from APP_ID). Defensively copied so caller mutations after construction have no effect. |
| `apiPrefix` | `string` | `"/api/v1/memory"` | URL path prefix for memory endpoints. Defaults to "/api/v1/memory" (the OE proxy route, which strips `/memory/` before forwarding to the memory server). Use "/api/v1" when connecting directly to the memory server. |
| `dynamicHeaders?` | () => `Record`\<`string`, `string`\> | `undefined` | Called on every request; result is merged over `staticHeaders`. Used by agent-engine-runner-shared to inject execution-scoped `X-Agent-Engine-Execution-Id` from AsyncLocalStorage. |
| `requestExtras?` | () => `RequestInit` \| `undefined` | `undefined` | Called on every request; the returned `RequestInit` is spread into `fetch` init before method/headers/signal. Used to inject a custom transport (e.g. an undici mTLS `dispatcher`) for app-bound calls. |
| `maxRetries?` | `number` | `0` | Number of retries on transient 5xx (502/503/504) and network errors, with exponential backoff. Default 0 (no retry). |
| `routeStyle?` | [`MemoryRouteStyle`](#api-memoryroutestyle) | `"native"` | Memory route convention (see [MemoryRouteStyle](#api-memoryroutestyle)). Defaults to "native" — the memory server / OE proxy routes. Use "aliased" for the API Gateway's project-scoped routes (`/turns`, `/context`, single `/search` with a `type` body field). |
| `fetchImpl?` | \{ (`input`, `init?`): `Promise`\<`Response`\>; (`input`, `init?`): `Promise`\<`Response`\>; \} | `undefined` | - |

**Returns**

[`MemoryClient`](#api-memoryclient)

#### Methods

<a id="api-bootstrap"></a>

##### bootstrap()

```ts
bootstrap(): void;
```

**Returns**

`void`

<a id="api-buildcontext"></a>

##### buildContext()

```ts
buildContext(params): Promise<{
[key: string]: unknown;
  formatted_context: string | Record<string, unknown>[];
  metadata: {
   [key: string]: unknown;
     memory_counts: Record<string, number>;
     timing: Record<string, number>;
     token_count: number;
  };
  selected_memories?: object[] | null;
}>;
```

Build unified memory context.
Omitted `enabledSources` defaults to episodic and semantic; include
"stm" for recent turns.

**Parameters**

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `params` | \{ `enabledSources?`: `string`[] \| `null`; `formatStyle?`: `string` \| `null`; `maxTokens?`: `number`; `metadataFilter?`: `Record`\<`string`, `unknown`\> \| `null`; `orgId`: `string`; `projectId`: `string`; `query`: `string`; `sessionId?`: `string` \| `null`; `signal?`: `AbortSignal`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} | - |
| `params.enabledSources?` | `string`[] \| `null` | - |
| `params.formatStyle?` | `string` \| `null` | - |
| `params.maxTokens?` | `number` | Optional gross context-construction budget, serialized as `max_tokens` only when set. After retrieval and ranking, the server subtracts a 500-token formatting reserve, then greedily selects whole memory chunks that fit in the remainder. Positive values at or below 500 leave no budget for memories. Values above 500 can still yield empty context when no chunk fits. Non-positive, non-integer, and non-finite values raise before the request is sent. |
| `params.metadataFilter?` | `Record`\<`string`, `unknown`\> \| `null` | - |
| `params.orgId` | `string` | - |
| `params.projectId` | `string` | - |
| `params.query` | `string` | - |
| `params.sessionId?` | `string` \| `null` | - |
| `params.signal?` | `AbortSignal` | - |
| `params.topK?` | `number` | - |
| `params.userId?` | `string` \| `null` | - |
| `params.visibility?` | `string` \| `null` | - |

**Returns**

`Promise`\<\{
\[`key`: `string`\]: `unknown`;
  `formatted_context`: `string` \| `Record`\<`string`, `unknown`\>[];
  `metadata`: \{
   \[`key`: `string`\]: `unknown`;
     `memory_counts`: `Record`\<`string`, `number`\>;
     `timing`: `Record`\<`string`, `number`\>;
     `token_count`: `number`;
  \};
  `selected_memories?`: `object`[] \| `null`;
\}\>

<a id="api-createcustom"></a>

##### createCustom()

```ts
createCustom(params): Promise<{
  has_embedding: boolean;
  id: string;
  tags: Record<string, string | number | boolean>;
  type: string;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | \{ `content`: `string`; `contextualMetadata?`: `Record`\<`string`, `unknown`\> \| `null`; `memoryType`: `string`; `signal?`: `AbortSignal`; `tags?`: `Record`\<`string`, `unknown`\> \| `null`; \} |
| `params.content` | `string` |
| `params.contextualMetadata?` | `Record`\<`string`, `unknown`\> \| `null` |
| `params.memoryType` | `string` |
| `params.signal?` | `AbortSignal` |
| `params.tags?` | `Record`\<`string`, `unknown`\> \| `null` |

**Returns**

`Promise`\<\{
  `has_embedding`: `boolean`;
  `id`: `string`;
  `tags`: `Record`\<`string`, `string` \| `number` \| `boolean`\>;
  `type`: `string`;
\}\>

<a id="api-createepisodic"></a>

##### createEpisodic()

```ts
createEpisodic(params): Promise<{
[key: string]: unknown;
  acknowledged: boolean;
  has_embedding: boolean;
  id: string;
  title: string;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | \{ `agentId?`: `string` \| `null`; `content`: `string`; `embedding?`: `number`[] \| `null`; `extractionSource?`: `string` \| `null`; `metadata?`: `Record`\<`string`, `unknown`\> \| `null`; `orgId`: `string`; `participants?`: `string`[] \| `null`; `projectId`: `string`; `sessionId`: `string`; `signal?`: `AbortSignal`; `skipEmbedding?`: `boolean` \| `null`; `snapshotRefId?`: `string` \| `null`; `sourceAgent?`: `string` \| `null`; `summaryText`: `string`; `summaryType?`: `string` \| `null`; `tags?`: `string`[] \| `null`; `title`: `string`; `userId`: `string`; `visibility?`: `string`; \} |
| `params.agentId?` | `string` \| `null` |
| `params.content` | `string` |
| `params.embedding?` | `number`[] \| `null` |
| `params.extractionSource?` | `string` \| `null` |
| `params.metadata?` | `Record`\<`string`, `unknown`\> \| `null` |
| `params.orgId` | `string` |
| `params.participants?` | `string`[] \| `null` |
| `params.projectId` | `string` |
| `params.sessionId` | `string` |
| `params.signal?` | `AbortSignal` |
| `params.skipEmbedding?` | `boolean` \| `null` |
| `params.snapshotRefId?` | `string` \| `null` |
| `params.sourceAgent?` | `string` \| `null` |
| `params.summaryText` | `string` |
| `params.summaryType?` | `string` \| `null` |
| `params.tags?` | `string`[] \| `null` |
| `params.title` | `string` |
| `params.userId` | `string` |
| `params.visibility?` | `string` |

**Returns**

`Promise`\<\{
\[`key`: `string`\]: `unknown`;
  `acknowledged`: `boolean`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `title`: `string`;
\}\>

<a id="api-createprocedural"></a>

##### createProcedural()

```ts
createProcedural(params): Promise<{
[key: string]: unknown;
  acknowledged: boolean;
  has_embedding: boolean;
  id: string;
  procedure: string;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | \{ `agentId?`: `string` \| `null`; `allowedTools?`: `string`[] \| `null`; `compatibility?`: `string` \| `null`; `content`: `string`; `description`: `string`; `extractionSource?`: `string` \| `null`; `license?`: `string` \| `null`; `orgId`: `string`; `procedure`: `string`; `projectId`: `string`; `resources?`: `Record`\<`string`, `unknown`\>[] \| `null`; `signal?`: `AbortSignal`; `sourceFormat?`: `string` \| `null`; `sourcePath?`: `string` \| `null`; `steps?`: `Record`\<`string`, `unknown`\>[] \| `null`; `tags?`: `string`[] \| `null`; `triggerConditions?`: `string`[] \| `null`; `userId`: `string`; `visibility?`: `string`; \} |
| `params.agentId?` | `string` \| `null` |
| `params.allowedTools?` | `string`[] \| `null` |
| `params.compatibility?` | `string` \| `null` |
| `params.content` | `string` |
| `params.description` | `string` |
| `params.extractionSource?` | `string` \| `null` |
| `params.license?` | `string` \| `null` |
| `params.orgId` | `string` |
| `params.procedure` | `string` |
| `params.projectId` | `string` |
| `params.resources?` | `Record`\<`string`, `unknown`\>[] \| `null` |
| `params.signal?` | `AbortSignal` |
| `params.sourceFormat?` | `string` \| `null` |
| `params.sourcePath?` | `string` \| `null` |
| `params.steps?` | `Record`\<`string`, `unknown`\>[] \| `null` |
| `params.tags?` | `string`[] \| `null` |
| `params.triggerConditions?` | `string`[] \| `null` |
| `params.userId` | `string` |
| `params.visibility?` | `string` |

**Returns**

`Promise`\<\{
\[`key`: `string`\]: `unknown`;
  `acknowledged`: `boolean`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `procedure`: `string`;
\}\>

<a id="api-createsemantic"></a>

##### createSemantic()

```ts
createSemantic(params): Promise<{
[key: string]: unknown;
  acknowledged: boolean;
  has_embedding: boolean;
  id: string;
  label: string;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | \{ `agentId?`: `string` \| `null`; `embedding?`: `number`[] \| `null`; `label`: `string`; `metadata?`: `Record`\<`string`, `unknown`\> \| `null`; `orgId`: `string`; `projectId`: `string`; `signal?`: `AbortSignal`; `source?`: `string`; `text`: `string`; `upsert?`: `boolean`; `userId`: `string`; `visibility?`: `string`; \} |
| `params.agentId?` | `string` \| `null` |
| `params.embedding?` | `number`[] \| `null` |
| `params.label` | `string` |
| `params.metadata?` | `Record`\<`string`, `unknown`\> \| `null` |
| `params.orgId` | `string` |
| `params.projectId` | `string` |
| `params.signal?` | `AbortSignal` |
| `params.source?` | `string` |
| `params.text` | `string` |
| `params.upsert?` | `boolean` |
| `params.userId` | `string` |
| `params.visibility?` | `string` |

**Returns**

`Promise`\<\{
\[`key`: `string`\]: `unknown`;
  `acknowledged`: `boolean`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `label`: `string`;
\}\>

<a id="api-createtaxonomic"></a>

##### createTaxonomic()

```ts
createTaxonomic(params): Promise<{
[key: string]: unknown;
  acknowledged: boolean;
  domain: string;
  has_embedding: boolean;
  id: string;
  term: string;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | \{ `definition`: `string`; `domain`: `string`; `orgId`: `string`; `projectId`: `string`; `queryExpansion?`: `boolean`; `relatedTerms?`: `string`[] \| `null`; `signal?`: `AbortSignal`; `term`: `string`; `userId`: `string`; `visibility?`: `string`; \} |
| `params.definition` | `string` |
| `params.domain` | `string` |
| `params.orgId` | `string` |
| `params.projectId` | `string` |
| `params.queryExpansion?` | `boolean` |
| `params.relatedTerms?` | `string`[] \| `null` |
| `params.signal?` | `AbortSignal` |
| `params.term` | `string` |
| `params.userId` | `string` |
| `params.visibility?` | `string` |

**Returns**

`Promise`\<\{
\[`key`: `string`\]: `unknown`;
  `acknowledged`: `boolean`;
  `domain`: `string`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `term`: `string`;
\}\>

<a id="api-deleteprocedural"></a>

##### deleteProcedural()

```ts
deleteProcedural(params): Promise<{
[key: string]: unknown;
  acknowledged: boolean;
  deleted_count: number;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | \{ `id?`: `string` \| `null`; `orgId`: `string`; `procedure?`: `string` \| `null`; `projectId`: `string`; `signal?`: `AbortSignal`; `soft?`: `boolean`; \} |
| `params.id?` | `string` \| `null` |
| `params.orgId` | `string` |
| `params.procedure?` | `string` \| `null` |
| `params.projectId` | `string` |
| `params.signal?` | `AbortSignal` |
| `params.soft?` | `boolean` |

**Returns**

`Promise`\<\{
\[`key`: `string`\]: `unknown`;
  `acknowledged`: `boolean`;
  `deleted_count`: `number`;
\}\>

<a id="api-discoverprocedures"></a>

##### discoverProcedures()

```ts
discoverProcedures(params): Promise<Record<string, unknown>[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | \{ `metadataFilter?`: `Record`\<`string`, `unknown`\> \| `null`; `orgId`: `string`; `projectId`: `string`; `query`: `string`; `signal?`: `AbortSignal`; `similarityThreshold?`: `number`; `tags?`: `string`[] \| `null`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `params.metadataFilter?` | `Record`\<`string`, `unknown`\> \| `null` |
| `params.orgId` | `string` |
| `params.projectId` | `string` |
| `params.query` | `string` |
| `params.signal?` | `AbortSignal` |
| `params.similarityThreshold?` | `number` |
| `params.tags?` | `string`[] \| `null` |
| `params.topK?` | `number` |
| `params.userId?` | `string` \| `null` |
| `params.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`Record`\<`string`, `unknown`\>[]\>

<a id="api-fetchepisodicmemories"></a>

##### fetchEpisodicMemories()

```ts
fetchEpisodicMemories(params): Promise<object[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | \{ `orgId`: `string`; `projectId`: `string`; `query`: `string`; `sessionId?`: `string` \| `null`; `signal?`: `AbortSignal`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `params.orgId` | `string` |
| `params.projectId` | `string` |
| `params.query` | `string` |
| `params.sessionId?` | `string` \| `null` |
| `params.signal?` | `AbortSignal` |
| `params.topK?` | `number` |
| `params.userId?` | `string` \| `null` |
| `params.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`object`[]\>

<a id="api-fetchproceduralmemories"></a>

##### fetchProceduralMemories()

```ts
fetchProceduralMemories(params): Promise<object[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | \{ `orgId`: `string`; `projectId`: `string`; `query`: `string`; `signal?`: `AbortSignal`; `tags?`: `string`[] \| `null`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `params.orgId` | `string` |
| `params.projectId` | `string` |
| `params.query` | `string` |
| `params.signal?` | `AbortSignal` |
| `params.tags?` | `string`[] \| `null` |
| `params.topK?` | `number` |
| `params.userId?` | `string` \| `null` |
| `params.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`object`[]\>

<a id="api-fetchsemanticmemories"></a>

##### fetchSemanticMemories()

```ts
fetchSemanticMemories(params): Promise<object[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | \{ `orgId`: `string`; `projectId`: `string`; `query`: `string`; `signal?`: `AbortSignal`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `params.orgId` | `string` |
| `params.projectId` | `string` |
| `params.query` | `string` |
| `params.signal?` | `AbortSignal` |
| `params.topK?` | `number` |
| `params.userId?` | `string` \| `null` |
| `params.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`object`[]\>

<a id="api-fetchtaxonomicmemories"></a>

##### fetchTaxonomicMemories()

```ts
fetchTaxonomicMemories(params): Promise<object[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | \{ `domain?`: `string` \| `null`; `orgId`: `string`; `projectId`: `string`; `query`: `string`; `signal?`: `AbortSignal`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `params.domain?` | `string` \| `null` |
| `params.orgId` | `string` |
| `params.projectId` | `string` |
| `params.query` | `string` |
| `params.signal?` | `AbortSignal` |
| `params.topK?` | `number` |
| `params.userId?` | `string` \| `null` |
| `params.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`object`[]\>

<a id="api-getdistinctdomains"></a>

##### getDistinctDomains()

```ts
getDistinctDomains(params): Promise<string[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | \{ `orgId`: `string`; `projectId`: `string`; `signal?`: `AbortSignal`; `visibility?`: `string` \| `null`; \} |
| `params.orgId` | `string` |
| `params.projectId` | `string` |
| `params.signal?` | `AbortSignal` |
| `params.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`string`[]\>

<a id="api-getprocedural"></a>

##### getProcedural()

```ts
getProcedural(params): Promise<unknown>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | \{ `id?`: `string` \| `null`; `includeDeleted?`: `boolean`; `orgId`: `string`; `procedure?`: `string` \| `null`; `projectId`: `string`; `signal?`: `AbortSignal`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `params.id?` | `string` \| `null` |
| `params.includeDeleted?` | `boolean` |
| `params.orgId` | `string` |
| `params.procedure?` | `string` \| `null` |
| `params.projectId` | `string` |
| `params.signal?` | `AbortSignal` |
| `params.userId?` | `string` \| `null` |
| `params.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`\>

<a id="api-getsemantic"></a>

##### getSemantic()

```ts
getSemantic(params): Promise<unknown>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | \{ `id?`: `string` \| `null`; `label?`: `string` \| `null`; `orgId`: `string`; `projectId`: `string`; `signal?`: `AbortSignal`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `params.id?` | `string` \| `null` |
| `params.label?` | `string` \| `null` |
| `params.orgId` | `string` |
| `params.projectId` | `string` |
| `params.signal?` | `AbortSignal` |
| `params.userId?` | `string` \| `null` |
| `params.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`\>

<a id="api-gettaxonomic"></a>

##### getTaxonomic()

```ts
getTaxonomic(params): Promise<unknown>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | \{ `domain?`: `string` \| `null`; `orgId`: `string`; `projectId`: `string`; `signal?`: `AbortSignal`; `term?`: `string` \| `null`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `params.domain?` | `string` \| `null` |
| `params.orgId` | `string` |
| `params.projectId` | `string` |
| `params.signal?` | `AbortSignal` |
| `params.term?` | `string` \| `null` |
| `params.userId?` | `string` \| `null` |
| `params.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`\>

<a id="api-listepisodic"></a>

##### listEpisodic()

```ts
listEpisodic(params): Promise<unknown[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | \{ `limit?`: `number`; `orgId`: `string`; `projectId`: `string`; `sessionId?`: `string` \| `null`; `signal?`: `AbortSignal`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `params.limit?` | `number` |
| `params.orgId` | `string` |
| `params.projectId` | `string` |
| `params.sessionId?` | `string` \| `null` |
| `params.signal?` | `AbortSignal` |
| `params.userId?` | `string` \| `null` |
| `params.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`[]\>

<a id="api-retrievecustom"></a>

##### retrieveCustom()

```ts
retrieveCustom(params): Promise<{
  count: number;
  results: object[];
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | \{ `memoryType`: `string`; `query`: `string`; `signal?`: `AbortSignal`; `tags?`: `Record`\<`string`, `unknown`\> \| `null`; `topK?`: `number`; \} |
| `params.memoryType` | `string` |
| `params.query` | `string` |
| `params.signal?` | `AbortSignal` |
| `params.tags?` | `Record`\<`string`, `unknown`\> \| `null` |
| `params.topK?` | `number` |

**Returns**

`Promise`\<\{
  `count`: `number`;
  `results`: `object`[];
\}\>

<a id="api-updateprocedural"></a>

##### updateProcedural()

```ts
updateProcedural(params): Promise<unknown>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | \{ `allowedTools?`: `string`[] \| `null`; `content?`: `string` \| `null`; `description?`: `string` \| `null`; `id?`: `string` \| `null`; `orgId`: `string`; `procedure?`: `string` \| `null`; `projectId`: `string`; `resources?`: `Record`\<`string`, `unknown`\>[] \| `null`; `signal?`: `AbortSignal`; `steps?`: `Record`\<`string`, `unknown`\>[] \| `null`; `tags?`: `string`[] \| `null`; `triggerConditions?`: `string`[] \| `null`; `visibility?`: `string` \| `null`; \} |
| `params.allowedTools?` | `string`[] \| `null` |
| `params.content?` | `string` \| `null` |
| `params.description?` | `string` \| `null` |
| `params.id?` | `string` \| `null` |
| `params.orgId` | `string` |
| `params.procedure?` | `string` \| `null` |
| `params.projectId` | `string` |
| `params.resources?` | `Record`\<`string`, `unknown`\>[] \| `null` |
| `params.signal?` | `AbortSignal` |
| `params.steps?` | `Record`\<`string`, `unknown`\>[] \| `null` |
| `params.tags?` | `string`[] \| `null` |
| `params.triggerConditions?` | `string`[] \| `null` |
| `params.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`\>

<a id="api-updatesemantic"></a>

##### updateSemantic()

```ts
updateSemantic(params): Promise<unknown>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | \{ `label`: `string`; `orgId`: `string`; `projectId`: `string`; `signal?`: `AbortSignal`; `source?`: `string` \| `null`; `text?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `params.label` | `string` |
| `params.orgId` | `string` |
| `params.projectId` | `string` |
| `params.signal?` | `AbortSignal` |
| `params.source?` | `string` \| `null` |
| `params.text?` | `string` \| `null` |
| `params.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`\>

<a id="api-writeturn"></a>

##### writeTurn()

```ts
writeTurn(params): Promise<{
[key: string]: unknown;
  acknowledged: boolean;
  id: string;
  session_id: string;
  turn_seq: number;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | \{ `content?`: `string` \| `null`; `idempotencyKey?`: `string` \| `null`; `isError?`: `boolean`; `modelName?`: `string` \| `null`; `orgId`: `string`; `projectId`: `string`; `role`: `string`; `sessionId?`: `string` \| `null`; `signal?`: `AbortSignal`; `stopReason?`: `string` \| `null`; `tokens?`: `number` \| `null`; `toolCallId?`: `string` \| `null`; `toolCalls?`: `Record`\<`string`, `unknown`\>[] \| `null`; `toolName?`: `string` \| `null`; `userId?`: `string` \| `null`; \} |
| `params.content?` | `string` \| `null` |
| `params.idempotencyKey?` | `string` \| `null` |
| `params.isError?` | `boolean` |
| `params.modelName?` | `string` \| `null` |
| `params.orgId` | `string` |
| `params.projectId` | `string` |
| `params.role` | `string` |
| `params.sessionId?` | `string` \| `null` |
| `params.signal?` | `AbortSignal` |
| `params.stopReason?` | `string` \| `null` |
| `params.tokens?` | `number` \| `null` |
| `params.toolCallId?` | `string` \| `null` |
| `params.toolCalls?` | `Record`\<`string`, `unknown`\>[] \| `null` |
| `params.toolName?` | `string` \| `null` |
| `params.userId?` | `string` \| `null` |

**Returns**

`Promise`\<\{
\[`key`: `string`\]: `unknown`;
  `acknowledged`: `boolean`;
  `id`: `string`;
  `session_id`: `string`;
  `turn_seq`: `number`;
\}\>

***

<a id="api-memoryhttperror"></a>

### MemoryHttpError

Thrown for non-OK memory-server responses. Carries the HTTP status and body
so callers can branch on failure mode (e.g. map 401/403/404/5xx to typed
errors) instead of parsing a message string. Extends `Error`, so existing
`catch (e) { ... }` consumers are unaffected.

#### Extends

- `Error`

<a id="api-constructor-8"></a>

#### Constructor

```ts
new MemoryHttpError(status, responseText): MemoryHttpError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `status` | `number` |
| `responseText` | `string` |

**Returns**

[`MemoryHttpError`](#api-memoryhttperror)

**Overrides**

```ts
Error.constructor
```

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-responsetext"></a> `responseText` | `readonly` | `string` |
| <a id="api-property-status"></a> `status` | `readonly` | `number` |

***

<a id="api-nullexecutioncallback"></a>

### NullExecutionCallback

Concrete base class with no-op defaults for all `BaseExecutionCallback` methods.

Subclass this instead of implementing `BaseExecutionCallback` directly so that
new methods added to the interface are automatically satisfied — you only
override the events you care about.

```ts
class MyCallback extends NullExecutionCallback {
  onNodeStart(nodeName: string, _inputs, _opts) {
    console.log(`starting ${nodeName}`)
  }
}
```

#### Implements

- [`BaseExecutionCallback`](#api-baseexecutioncallback)

<a id="api-constructor-9"></a>

#### Constructor

```ts
new NullExecutionCallback(): NullExecutionCallback;
```

**Returns**

[`NullExecutionCallback`](#api-nullexecutioncallback)

#### Methods

<a id="api-onnodeend"></a>

##### onNodeEnd()

```ts
onNodeEnd(
   _nodeName,
   _outputs,
   _opts
): void;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `_nodeName` | `string` |
| `_outputs` | `Record`\<`string`, `unknown`\> |
| `_opts` | \{ `durationMs?`: `number`; `metadata?`: `Record`\<`string`, `unknown`\>; `parentRunId?`: `string`; `runId`: `string`; \} |
| `_opts.durationMs?` | `number` |
| `_opts.metadata?` | `Record`\<`string`, `unknown`\> |
| `_opts.parentRunId?` | `string` |
| `_opts.runId` | `string` |

**Returns**

`void`

**Implementation of**

[`BaseExecutionCallback`](#api-baseexecutioncallback).[`onNodeEnd`](#api-onnodeend-1)

<a id="api-onnodeerror"></a>

##### onNodeError()

```ts
onNodeError(
   _nodeName,
   _error,
   _opts
): void;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `_nodeName` | `string` |
| `_error` | `string` |
| `_opts` | \{ `metadata?`: `Record`\<`string`, `unknown`\>; `parentRunId?`: `string`; `runId`: `string`; \} |
| `_opts.metadata?` | `Record`\<`string`, `unknown`\> |
| `_opts.parentRunId?` | `string` |
| `_opts.runId` | `string` |

**Returns**

`void`

**Implementation of**

[`BaseExecutionCallback`](#api-baseexecutioncallback).[`onNodeError`](#api-onnodeerror-1)

<a id="api-onnodestart"></a>

##### onNodeStart()

```ts
onNodeStart(
   _nodeName,
   _inputs,
   _opts
): void;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `_nodeName` | `string` |
| `_inputs` | `Record`\<`string`, `unknown`\> |
| `_opts` | \{ `metadata?`: `Record`\<`string`, `unknown`\>; `parentRunId?`: `string`; `runId`: `string`; \} |
| `_opts.metadata?` | `Record`\<`string`, `unknown`\> |
| `_opts.parentRunId?` | `string` |
| `_opts.runId` | `string` |

**Returns**

`void`

**Implementation of**

[`BaseExecutionCallback`](#api-baseexecutioncallback).[`onNodeStart`](#api-onnodestart-1)

<a id="api-onnodesuspend"></a>

##### onNodeSuspend()

```ts
onNodeSuspend(_nodeName, _opts): void;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `_nodeName` | `string` |
| `_opts` | \{ `metadata?`: `Record`\<`string`, `unknown`\>; `parentRunId?`: `string`; `runId`: `string`; \} |
| `_opts.metadata?` | `Record`\<`string`, `unknown`\> |
| `_opts.parentRunId?` | `string` |
| `_opts.runId` | `string` |

**Returns**

`void`

## Interfaces

<a id="api-agentinput"></a>

### AgentInput

Input for an agent execution. Payload is an opaque JSON value.

#### Properties

| Property | Type |
| :------ | :------ |
| <a id="api-property-payload"></a> `payload` | [`JsonValue`](#api-jsonvalue) |

***

<a id="api-agentoutput"></a>

### AgentOutput

Output from an agent execution. Response is an opaque JSON value.

#### Properties

| Property | Type |
| :------ | :------ |
| <a id="api-property-response"></a> `response` | [`JsonValue`](#api-jsonvalue) |

***

<a id="api-baseagent"></a>

### BaseAgent

The contract between agent code and runtimes.

Agents expose a single `execute()` method that returns an `ExecutionResult`.
Callers choose the execution mode:
- `await agent.execute(ctx, input)` for a final result (JSON-style)
- `for await (const event of agent.execute(ctx, input))` for streaming (SSE-style)

#### Methods

<a id="api-execute"></a>

##### execute()

```ts
execute(ctx, input): ExecutionResult;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `ctx` | [`RequestContext`](#api-requestcontext) |
| `input` | [`AgentInput`](#api-agentinput) |

**Returns**

[`ExecutionResult`](#api-executionresult)

***

<a id="api-baseexecutioncallback"></a>

### BaseExecutionCallback

Framework-neutral callback for observability during agent execution.

Replaces LangChain's `BaseCallbackHandler`. The AER injects an
implementation that forwards node events to the OE for logging.
Framework adapters (e.g. sdk-langgraph) bridge from the framework's
native callback system to this interface.

Prefer subclassing `NullExecutionCallback` over implementing this
interface directly — it provides no-op defaults for all methods including
optional extensions (e.g. `onNodeSuspend`) so new methods never break
existing implementations.

#### Methods

<a id="api-onnodeend-1"></a>

##### onNodeEnd()

```ts
onNodeEnd(
   nodeName,
   outputs,
   opts
): void;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `nodeName` | `string` |
| `outputs` | `Record`\<`string`, `unknown`\> |
| `opts` | \{ `durationMs?`: `number`; `metadata?`: `Record`\<`string`, `unknown`\>; `parentRunId?`: `string`; `runId`: `string`; \} |
| `opts.durationMs?` | `number` |
| `opts.metadata?` | `Record`\<`string`, `unknown`\> |
| `opts.parentRunId?` | `string` |
| `opts.runId` | `string` |

**Returns**

`void`

<a id="api-onnodeerror-1"></a>

##### onNodeError()

```ts
onNodeError(
   nodeName,
   error,
   opts
): void;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `nodeName` | `string` |
| `error` | `string` |
| `opts` | \{ `metadata?`: `Record`\<`string`, `unknown`\>; `parentRunId?`: `string`; `runId`: `string`; \} |
| `opts.metadata?` | `Record`\<`string`, `unknown`\> |
| `opts.parentRunId?` | `string` |
| `opts.runId` | `string` |

**Returns**

`void`

<a id="api-onnodestart-1"></a>

##### onNodeStart()

```ts
onNodeStart(
   nodeName,
   inputs,
   opts
): void;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `nodeName` | `string` |
| `inputs` | `Record`\<`string`, `unknown`\> |
| `opts` | \{ `metadata?`: `Record`\<`string`, `unknown`\>; `parentRunId?`: `string`; `runId`: `string`; \} |
| `opts.metadata?` | `Record`\<`string`, `unknown`\> |
| `opts.parentRunId?` | `string` |
| `opts.runId` | `string` |

**Returns**

`void`

***

<a id="api-basellm"></a>

### BaseLLM

Framework-neutral LLM protocol.

Used at runtime by the ToolPod. In Python, this protocol exposes both `invoke`
(sync) and `ainvoke` (async) since Python supports both call styles. JS has no
sync HTTP, so this interface has a single async `invoke` plus `stream`.

#### Methods

<a id="api-invoke"></a>

##### invoke()

```ts
invoke(messages, kwargs?): Promise<LLMResponse>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `messages` | [`Message`](#api-message)[] |
| `kwargs?` | `Record`\<`string`, `unknown`\> |

**Returns**

`Promise`\<[`LLMResponse`](#api-llmresponse)\>

<a id="api-stream"></a>

##### stream()

```ts
stream(messages, kwargs?): AsyncIterable<LLMStreamChunk>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `messages` | [`Message`](#api-message)[] |
| `kwargs?` | `Record`\<`string`, `unknown`\> |

**Returns**

`AsyncIterable`\<[`LLMStreamChunk`](#api-llmstreamchunk)\>

***

<a id="api-executionresult"></a>

### ExecutionResult

Result of an agent execution.

Await for the final result, or async-iterate for streaming events.

```ts
// Non-streaming
const output = await agent.execute(ctx, input)

// Streaming
for await (const event of agent.execute(ctx, input)) {
  handle(event)
}
```

#### Extends

- `PromiseLike`\<[`AgentOutput`](#api-agentoutput)\>.`AsyncIterable`\<[`StreamEvent`](#api-streamevent)\>

***

<a id="api-llmstreamchunk"></a>

### LLMStreamChunk

Rich streaming chunk following OpenAI/Anthropic patterns.

All fields are optional since different chunks contain different data:
- Content chunks: text deltas during generation
- Tool call chunks: partial tool calls being built up
- Usage chunks: token counts at end of stream

#### Properties

| Property | Type |
| :------ | :------ |
| <a id="api-property-additionalkwargs-1"></a> `additionalKwargs?` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| <a id="api-property-content-1"></a> `content?` | `string` |
| <a id="api-property-id-2"></a> `id?` | `string` |
| <a id="api-property-name-4"></a> `name?` | `string` |
| <a id="api-property-responsemetadata-1"></a> `responseMetadata?` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| <a id="api-property-toolcalls-1"></a> `toolCalls?` | [`ToolCallChunk`](#api-toolcallchunk)[] |
| <a id="api-property-usage-1"></a> `usage?` | [`LLMTokenUsage`](#api-llmtokenusage) |

***

<a id="api-message"></a>

### Message

Framework-neutral message.

Content can be a simple string for text-only messages, or a list of
content blocks for multimodal content (images, documents, etc.).

#### Properties

| Property | Type |
| :------ | :------ |
| <a id="api-property-additionalkwargs-2"></a> `additionalKwargs?` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| <a id="api-property-content-2"></a> `content` | \| `string` \| ( \| \{ `text`: `string`; `type`: `"text"`; \} \| \{ `mime_type?`: `string`; `type`: `"image"`; `url`: `string`; \} \| \{ `filename?`: `string`; `mime_type?`: `string`; `type`: `"document"`; `url`: `string`; \})[] |
| <a id="api-property-id-3"></a> `id?` | `string` |
| <a id="api-property-iserror"></a> `isError?` | `boolean` |
| <a id="api-property-name-5"></a> `name?` | `string` |
| <a id="api-property-responsemetadata-2"></a> `responseMetadata?` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> |
| <a id="api-property-role"></a> `role` | [`Role`](#api-role) |
| <a id="api-property-toolcallid"></a> `toolCallId?` | `string` |
| <a id="api-property-toolcalls-2"></a> `toolCalls?` | [`LLMToolCall`](#api-llmtoolcall)[] |

***

<a id="api-requestcontext"></a>

### RequestContext

Request-scoped context passed to every agent execution.

Carries framework-neutral execution identity and platform plumbing. The
caller's invocation input lives in `AgentInput.payload`, not here.
Suspend/resume state is opaque to the platform: `metadata` is a
framework-owned blob that the platform forwards unchanged on resume.

#### Properties

| Property | Type | Description |
| :------ | :------ | :------ |
| <a id="api-property-executionid"></a> `executionId?` | `string` | - |
| <a id="api-property-metadata-1"></a> `metadata?` | `Record`\<`string`, [`JsonValue`](#api-jsonvalue)\> \| `null` | Opaque framework-owned state passed through by the SDK and AER. Only the selected framework adapter may interpret its keys and values. |
| <a id="api-property-previousexecutioncancelled"></a> `previousExecutionCancelled?` | `boolean` | True when this session's most recent prior execution was cancelled (its pod torn down mid-run). Framework adapters that persist per-step state use it to fence off the killed step's partial writes before running this turn. Mirrors Python's `previous_execution_cancelled`. |
| <a id="api-property-requestheaders"></a> `requestHeaders?` | `Record`\<`string`, `string`\> | - |
| <a id="api-property-resume"></a> `resume?` | `boolean` | Platform suspend/resume plumbing (framework-neutral). |
| <a id="api-property-resumedata"></a> `resumeData?` | [`JsonValue`](#api-jsonvalue) | - |
| <a id="api-property-sessionid"></a> `sessionId?` | `string` | - |
| <a id="api-property-signal"></a> `signal?` | `AbortSignal` | Cancellation signal for execution timeout or forced abort. The runtime (and any agent implementation that honours it) should unwind pending awaits and stop emitting events when this signal fires. Python relies on `asyncio.wait_for` for the same semantics; JS has no implicit cancellation, so the signal must be plumbed explicitly. |
| <a id="api-property-userid"></a> `userId?` | `string` | - |
| <a id="api-property-workspaceid"></a> `workspaceId?` | `string` | Tenant scope of the executing agent. Framework adapters use it to scope state stores keyed only by sessionId (e.g. the LangGraph checkpoint thread_id) so agents in the same project DB cannot collide. Mirrors Python's `RequestContext.workspace_id`. |

***

<a id="api-streamevent"></a>

### StreamEvent

Event streamed back to the caller during execution.

#### Properties

| Property | Type |
| :------ | :------ |
| <a id="api-property-data"></a> `data` | [`JsonValue`](#api-jsonvalue) |
| <a id="api-property-event"></a> `event?` | `string` |

***

<a id="api-toolcallchunk"></a>

### ToolCallChunk

Partial tool call during streaming.

All fields are optional since chunks may contain partial data
(e.g., just the start of arguments, or just the tool name).

#### Properties

| Property | Type |
| :------ | :------ |
| <a id="api-property-args-1"></a> `args?` | `string` |
| <a id="api-property-id-4"></a> `id?` | `string` |
| <a id="api-property-index-1"></a> `index?` | `number` |
| <a id="api-property-name-6"></a> `name?` | `string` |
| <a id="api-property-type-2"></a> `type?` | `string` |

## Type Aliases

<a id="api-anycontentblock"></a>

### AnyContentBlock

```ts
type AnyContentBlock = z.infer<typeof AnyContentBlockSchema>;
```

@mongodb-js/agent-engine-sdk — Core TypeScript protocols and models for MongoDB Agent Engine.

***

<a id="api-branchref"></a>

### BranchRef

```ts
type BranchRef = z.infer<typeof BranchRefSchema>;
```

@mongodb-js/agent-engine-sdk — Core TypeScript protocols and models for MongoDB Agent Engine.

***

<a id="api-bulkcreatesemanticresult"></a>

### BulkCreateSemanticResult

```ts
type BulkCreateSemanticResult = z.infer<typeof BulkCreateSemanticResultSchema>;
```

***

<a id="api-contextconfig"></a>

### ContextConfig

```ts
type ContextConfig = z.infer<typeof ContextConfigSchema>;
```

***

<a id="api-contextmetadata"></a>

### ContextMetadata

```ts
type ContextMetadata = z.infer<typeof ContextMetadataSchema>;
```

***

<a id="api-contextresponse"></a>

### ContextResponse

```ts
type ContextResponse = z.infer<typeof ContextResponseSchema>;
```

***

<a id="api-createepisodicresult"></a>

### CreateEpisodicResult

```ts
type CreateEpisodicResult = z.infer<typeof CreateEpisodicResultSchema>;
```

***

<a id="api-createproceduralresult"></a>

### CreateProceduralResult

```ts
type CreateProceduralResult = z.infer<typeof CreateProceduralResultSchema>;
```

***

<a id="api-createsemanticresult"></a>

### CreateSemanticResult

```ts
type CreateSemanticResult = z.infer<typeof CreateSemanticResultSchema>;
```

***

<a id="api-createsnapshotresult"></a>

### CreateSnapshotResult

```ts
type CreateSnapshotResult = z.infer<typeof CreateSnapshotResultSchema>;
```

***

<a id="api-createtaxonomicresult"></a>

### CreateTaxonomicResult

```ts
type CreateTaxonomicResult = z.infer<typeof CreateTaxonomicResultSchema>;
```

***

<a id="api-createusercontextresult"></a>

### CreateUserContextResult

```ts
type CreateUserContextResult = z.infer<typeof CreateUserContextResultSchema>;
```

***

<a id="api-custommemoryretrieveresult"></a>

### CustomMemoryRetrieveResult

```ts
type CustomMemoryRetrieveResult = z.infer<typeof CustomMemoryRetrieveResultSchema>;
```

***

<a id="api-custommemorysaveresult"></a>

### CustomMemorySaveResult

```ts
type CustomMemorySaveResult = z.infer<typeof CustomMemorySaveResultSchema>;
```

***

<a id="api-deleteresult"></a>

### DeleteResult

```ts
type DeleteResult = z.infer<typeof DeleteResultSchema>;
```

***

<a id="api-documentblock"></a>

### DocumentBlock

```ts
type DocumentBlock = z.infer<typeof DocumentBlockSchema>;
```

@mongodb-js/agent-engine-sdk — Core TypeScript protocols and models for MongoDB Agent Engine.

***

<a id="api-event"></a>

### Event

```ts
type Event = z.infer<typeof EventSchema>;
```

@mongodb-js/agent-engine-sdk — Core TypeScript protocols and models for MongoDB Agent Engine.

***

<a id="api-eventresponse"></a>

### EventResponse

```ts
type EventResponse = z.infer<typeof EventResponseSchema>;
```

***

<a id="api-eventsresponse"></a>

### EventsResponse

```ts
type EventsResponse = z.infer<typeof EventsResponseSchema>;
```

***

<a id="api-formatstyle"></a>

### FormatStyle

```ts
type FormatStyle = z.infer<typeof FormatStyleSchema>;
```

***

<a id="api-imageblock"></a>

### ImageBlock

```ts
type ImageBlock = z.infer<typeof ImageBlockSchema>;
```

@mongodb-js/agent-engine-sdk — Core TypeScript protocols and models for MongoDB Agent Engine.

***

<a id="api-internalstateresult"></a>

### InternalStateResult

```ts
type InternalStateResult = z.infer<typeof InternalStateResultSchema>;
```

***

<a id="api-isgenerationresult"></a>

### ISGenerationResult

```ts
type ISGenerationResult = z.infer<typeof ISGenerationResultSchema>;
```

***

<a id="api-jsonvalue"></a>

### JsonValue

```ts
type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | {
[key: string]: JsonValue;
};
```

@mongodb-js/agent-engine-sdk — Core TypeScript protocols and models for MongoDB Agent Engine.

***

<a id="api-memorychunk"></a>

### MemoryChunk

```ts
type MemoryChunk = z.infer<typeof MemoryChunkSchema>;
```

***

<a id="api-memoryroutestyle"></a>

### MemoryRouteStyle

```ts
type MemoryRouteStyle = "native" | "aliased";
```

Which memory route convention a backend speaks.

- `native`: the memory server's own routes (`/stm/turns`, `/retrieval/*`).
  Served directly by the memory server and by the OE proxy (which also
  accepts the aliased forms). This is the default and preserves the behaviour
  of every existing consumer.
- `aliased`: the API Gateway's project-scoped memory routes, where the core
  loop collapses to `/turns`, `/context`, and a single `/search` that selects
  the source via a `type` body field. The Gateway does not expose the native
  `/stm` or `/retrieval` paths, so a Gateway-direct client must use this.

***

<a id="api-memorysource"></a>

### MemorySource

```ts
type MemorySource = z.infer<typeof MemorySourceSchema>;
```

***

<a id="api-modeltype"></a>

### ModelType

```ts
type ModelType = z.infer<typeof ModelTypeSchema>;
```

***

<a id="api-promotesnapshotresult"></a>

### PromoteSnapshotResult

```ts
type PromoteSnapshotResult = z.infer<typeof PromoteSnapshotResultSchema>;
```

***

<a id="api-retrievedcustommemory"></a>

### RetrievedCustomMemory

```ts
type RetrievedCustomMemory = z.infer<typeof RetrievedCustomMemorySchema>;
```

***

<a id="api-role"></a>

### Role

```ts
type Role = "user" | "assistant" | "tool" | "system";
```

@mongodb-js/agent-engine-sdk — Core TypeScript protocols and models for MongoDB Agent Engine.

***

<a id="api-sessionmessage"></a>

### SessionMessage

```ts
type SessionMessage = z.infer<typeof SessionMessageSchema>;
```

@mongodb-js/agent-engine-sdk — Core TypeScript protocols and models for MongoDB Agent Engine.

***

<a id="api-sessionmessagesresponse"></a>

### SessionMessagesResponse

```ts
type SessionMessagesResponse = z.infer<typeof SessionMessagesResponseSchema>;
```

@mongodb-js/agent-engine-sdk — Core TypeScript protocols and models for MongoDB Agent Engine.

***

<a id="api-sessionssummaryresponse"></a>

### SessionsSummaryResponse

```ts
type SessionsSummaryResponse = z.infer<typeof SessionsSummaryResponseSchema>;
```

@mongodb-js/agent-engine-sdk — Core TypeScript protocols and models for MongoDB Agent Engine.

***

<a id="api-sessionsummary"></a>

### SessionSummary

```ts
type SessionSummary = z.infer<typeof SessionSummarySchema>;
```

@mongodb-js/agent-engine-sdk — Core TypeScript protocols and models for MongoDB Agent Engine.

***

<a id="api-tagscalar"></a>

### TagScalar

```ts
type TagScalar = z.infer<typeof TagScalarSchema>;
```

***

<a id="api-textblock"></a>

### TextBlock

```ts
type TextBlock = z.infer<typeof TextBlockSchema>;
```

@mongodb-js/agent-engine-sdk — Core TypeScript protocols and models for MongoDB Agent Engine.

***

<a id="api-tooldefinition"></a>

### ToolDefinition

```ts
type ToolDefinition = z.infer<typeof ToolDefinitionSchema>;
```

@mongodb-js/agent-engine-sdk — Core TypeScript protocols and models for MongoDB Agent Engine.

***

<a id="api-tooldefinitioninput"></a>

### ToolDefinitionInput

```ts
type ToolDefinitionInput = z.input<typeof ToolDefinitionSchema>;
```

@mongodb-js/agent-engine-sdk — Core TypeScript protocols and models for MongoDB Agent Engine.

***

<a id="api-writeturnresult"></a>

### WriteTurnResult

```ts
type WriteTurnResult = z.infer<typeof WriteTurnResultSchema>;
```

## Variables

<a id="api-agentinputschema"></a>

### AgentInputSchema

```ts
const AgentInputSchema: ZodObject<{
  payload: ZodUnknown & ZodType<
     | {
   }
     | null, unknown, $ZodTypeInternals<
     | {
   }
    | null, unknown>>;
}, $loose>;
```

Runtime validation schema for AgentInput wire data.

***

<a id="api-agentoutputschema"></a>

### AgentOutputSchema

```ts
const AgentOutputSchema: ZodObject<{
  response: ZodUnknown & ZodType<
     | {
   }
     | null, unknown, $ZodTypeInternals<
     | {
   }
    | null, unknown>>;
}, $loose>;
```

Runtime validation schema for AgentOutput wire data.

***

<a id="api-anycontentblockschema"></a>

### AnyContentBlockSchema

```ts
const AnyContentBlockSchema: ZodDiscriminatedUnion<[ZodObject<{
  text: ZodString;
  type: ZodDefault<ZodLiteral<"text">>;
}, $strip>, ZodObject<{
  mime_type: ZodOptional<ZodString>;
  type: ZodDefault<ZodLiteral<"image">>;
  url: ZodString;
}, $strip>, ZodObject<{
  filename: ZodOptional<ZodString>;
  mime_type: ZodOptional<ZodString>;
  type: ZodDefault<ZodLiteral<"document">>;
  url: ZodString;
}, $strip>], "type">;
```

Discriminated on `type` so a document block is never misparsed as an image
(a plain union would match `{ url }` against `ImageBlockSchema` first and
silently drop `filename`). Inputs must carry an explicit `type`.

***

<a id="api-branchrefschema"></a>

### BranchRefSchema

```ts
const BranchRefSchema: ZodObject<{
  name: ZodString;
  root_event_id: ZodString;
}, $loose>;
```

Reference to a branch and its fork point.

***

<a id="api-bulkcreatesemanticresultschema"></a>

### BulkCreateSemanticResultSchema

```ts
const BulkCreateSemanticResultSchema: ZodObject<{
  acknowledged: ZodBoolean;
  created_count: ZodNumber;
  created_ids: ZodArray<ZodString>;
  has_embeddings: ZodBoolean;
  skipped_count: ZodNumber;
  skipped_labels: ZodArray<ZodString>;
}, $loose>;
```

***

<a id="api-contextconfigschema"></a>

### ContextConfigSchema

```ts
const ContextConfigSchema: ZodObject<{
  enabled_sources: ZodDefault<ZodPipe<ZodArray<ZodEnum<{
     episodic: "episodic";
     procedural: "procedural";
     semantic: "semantic";
     stm: "stm";
     taxonomic: "taxonomic";
  }>>, ZodTransform<("stm" | "episodic" | "semantic" | "taxonomic" | "procedural")[], ("stm" | "episodic" | "semantic" | "taxonomic" | "procedural")[]>>>;
  format_style: ZodDefault<ZodPreprocess<ZodEnum<{
     claude: "claude";
     jinja2: "jinja2";
     openai: "openai";
  }>>>;
  include_memories: ZodDefault<ZodBoolean>;
  max_tokens: ZodDefault<ZodNumber>;
  model_type: ZodDefault<ZodPreprocess<ZodEnum<{
     anthropic: "anthropic";
     claude: "claude";
     gemini: "gemini";
     generic: "generic";
     openai: "openai";
  }>>>;
  ranking_weights: ZodDefault<ZodRecord<ZodString, ZodNumber>>;
  similarity_threshold: ZodDefault<ZodNumber>;
  token_buffer: ZodDefault<ZodNumber>;
}, $strip>;
```

Configuration for context building requests.

***

<a id="api-contextmetadataschema"></a>

### ContextMetadataSchema

```ts
const ContextMetadataSchema: ZodObject<{
  memory_counts: ZodDefault<ZodRecord<ZodString, ZodNumber>>;
  timing: ZodDefault<ZodRecord<ZodString, ZodNumber>>;
  token_count: ZodDefault<ZodNumber>;
}, $loose>;
```

Metadata about a context build response (token counts, per-source counts, timing).

***

<a id="api-contextresponseschema"></a>

### ContextResponseSchema

```ts
const ContextResponseSchema: ZodObject<{
  formatted_context: ZodUnion<readonly [ZodString, ZodArray<ZodRecord<ZodString, ZodUnknown>>]>;
  metadata: ZodObject<{
     memory_counts: ZodDefault<ZodRecord<ZodString, ZodNumber>>;
     timing: ZodDefault<ZodRecord<ZodString, ZodNumber>>;
     token_count: ZodDefault<ZodNumber>;
  }, $loose>;
  selected_memories: ZodOptional<ZodNullable<ZodArray<ZodObject<{
     content: ZodString;
     embedding: ZodOptional<ZodArray<ZodNumber>>;
     id: ZodString;
     metadata: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
     similarity_score: ZodOptional<ZodNumber>;
     source: ZodEnum<{
        episodic: "episodic";
        procedural: "procedural";
        semantic: "semantic";
        stm: "stm";
        taxonomic: "taxonomic";
     }>;
     timestamp: ZodPipe<ZodString, ZodTransform<Date, string>>;
  }, $loose>>>>;
}, $loose>;
```

Response from a context build request containing formatted context and retrieval metadata.

***

<a id="api-createepisodicresultschema"></a>

### CreateEpisodicResultSchema

```ts
const CreateEpisodicResultSchema: ZodObject<{
  acknowledged: ZodBoolean;
  has_embedding: ZodBoolean;
  id: ZodString;
  title: ZodString;
}, $loose>;
```

***

<a id="api-createproceduralresultschema"></a>

### CreateProceduralResultSchema

```ts
const CreateProceduralResultSchema: ZodObject<{
  acknowledged: ZodBoolean;
  has_embedding: ZodBoolean;
  id: ZodString;
  procedure: ZodString;
}, $loose>;
```

***

<a id="api-createsemanticresultschema"></a>

### CreateSemanticResultSchema

```ts
const CreateSemanticResultSchema: ZodObject<{
  acknowledged: ZodBoolean;
  has_embedding: ZodBoolean;
  id: ZodString;
  label: ZodString;
}, $loose>;
```

***

<a id="api-createsnapshotresultschema"></a>

### CreateSnapshotResultSchema

```ts
const CreateSnapshotResultSchema: ZodObject<{
  acknowledged: ZodBoolean;
  embedding_strategy: ZodOptional<ZodString>;
  has_embedding: ZodBoolean;
  id: ZodString;
  message_count: ZodNumber;
  session_id: ZodString;
  snapshot_reason: ZodString;
}, $loose>;
```

***

<a id="api-createtaxonomicresultschema"></a>

### CreateTaxonomicResultSchema

```ts
const CreateTaxonomicResultSchema: ZodObject<{
  acknowledged: ZodBoolean;
  domain: ZodString;
  has_embedding: ZodBoolean;
  id: ZodString;
  term: ZodString;
}, $loose>;
```

***

<a id="api-createusercontextresultschema"></a>

### CreateUserContextResultSchema

```ts
const CreateUserContextResultSchema: ZodObject<{
  acknowledged: ZodBoolean;
  context_key: ZodString;
  id: ZodString;
}, $loose>;
```

***

<a id="api-custommemoryretrieveresultschema"></a>

### CustomMemoryRetrieveResultSchema

```ts
const CustomMemoryRetrieveResultSchema: ZodObject<{
  count: ZodNumber;
  results: ZodArray<ZodObject<{
     agent_id: ZodOptional<ZodNullable<ZodString>>;
     content: ZodString;
     contextual_metadata: ZodOptional<ZodNullable<ZodRecord<ZodString, ZodUnknown>>>;
     id: ZodString;
     tags: ZodRecord<ZodString, ZodUnion<readonly [ZodString, ZodNumber, ZodBoolean]>>;
     type: ZodString;
     user_id: ZodOptional<ZodNullable<ZodString>>;
  }, $strip>>;
}, $strip>;
```

***

<a id="api-custommemorysaveresultschema"></a>

### CustomMemorySaveResultSchema

```ts
const CustomMemorySaveResultSchema: ZodObject<{
  has_embedding: ZodBoolean;
  id: ZodString;
  tags: ZodRecord<ZodString, ZodUnion<readonly [ZodString, ZodNumber, ZodBoolean]>>;
  type: ZodString;
}, $strip>;
```

***

<a id="api-datefromstringschema"></a>

### DateFromStringSchema

```ts
const DateFromStringSchema: ZodPipe<ZodString, ZodTransform<Date, string>>;
```

Parses an ISO datetime string into a Date. Rejects unparseable input.

***

<a id="api-deleteresultschema"></a>

### DeleteResultSchema

```ts
const DeleteResultSchema: ZodObject<{
  acknowledged: ZodBoolean;
  deleted_count: ZodNumber;
}, $loose>;
```

***

<a id="api-documentblockschema"></a>

### DocumentBlockSchema

```ts
const DocumentBlockSchema: ZodObject<{
  filename: ZodOptional<ZodString>;
  mime_type: ZodOptional<ZodString>;
  type: ZodDefault<ZodLiteral<"document">>;
  url: ZodString;
}, $strip>;
```

Document/file content block.

***

<a id="api-eventresponseschema"></a>

### EventResponseSchema

```ts
const EventResponseSchema: ZodObject<{
  event: ZodObject<{
     actor_id: ZodString;
     branch: ZodOptional<ZodObject<{
        name: ZodString;
        root_event_id: ZodString;
     }, $loose>>;
     event_id: ZodString;
     metadata: ZodOptional<ZodRecord<ZodString, ZodString>>;
     parent_event_id: ZodOptional<ZodString>;
     payload: ZodPipe<ZodUnknown, ZodTransform<
        | string
        | number
        | boolean
        | JsonValue[]
        | {
      [key: string]: JsonValue;
      }
       | null, unknown>>;
     session_id: ZodString;
     timestamp: ZodPipe<ZodString, ZodTransform<Date, string>>;
  }, $loose>;
}, $loose>;
```

***

<a id="api-eventschema"></a>

### EventSchema

```ts
const EventSchema: ZodObject<{
  actor_id: ZodString;
  branch: ZodOptional<ZodObject<{
     name: ZodString;
     root_event_id: ZodString;
  }, $loose>>;
  event_id: ZodString;
  metadata: ZodOptional<ZodRecord<ZodString, ZodString>>;
  parent_event_id: ZodOptional<ZodString>;
  payload: ZodPipe<ZodUnknown, ZodTransform<
     | string
     | number
     | boolean
     | JsonValue[]
     | {
   [key: string]: JsonValue;
   }
    | null, unknown>>;
  session_id: ZodString;
  timestamp: ZodPipe<ZodString, ZodTransform<Date, string>>;
}, $loose>;
```

Platform event — the storage envelope. Payloads are opaque.

***

<a id="api-eventsresponseschema"></a>

### EventsResponseSchema

```ts
const EventsResponseSchema: ZodObject<{
  events: ZodArray<ZodObject<{
     actor_id: ZodString;
     branch: ZodOptional<ZodObject<{
        name: ZodString;
        root_event_id: ZodString;
     }, $loose>>;
     event_id: ZodString;
     metadata: ZodOptional<ZodRecord<ZodString, ZodString>>;
     parent_event_id: ZodOptional<ZodString>;
     payload: ZodPipe<ZodUnknown, ZodTransform<
        | string
        | number
        | boolean
        | JsonValue[]
        | {
      [key: string]: JsonValue;
      }
       | null, unknown>>;
     session_id: ZodString;
     timestamp: ZodPipe<ZodString, ZodTransform<Date, string>>;
  }, $loose>>;
}, $loose>;
```

***

<a id="api-formatstyle-1"></a>

### FormatStyle

```ts
FormatStyle: object;
```

#### Type Declaration

| Name | Type |
| :------ | :------ |
| <a id="api-property-claude"></a> `CLAUDE` | `"claude"` |
| <a id="api-property-jinja2"></a> `JINJA2` | `"jinja2"` |
| <a id="api-property-openai"></a> `OPENAI` | `"openai"` |

***

<a id="api-formatstyleschema"></a>

### FormatStyleSchema

```ts
const FormatStyleSchema: ZodEnum<{
  claude: "claude";
  jinja2: "jinja2";
  openai: "openai";
}>;
```

***

<a id="api-imageblockschema"></a>

### ImageBlockSchema

```ts
const ImageBlockSchema: ZodObject<{
  mime_type: ZodOptional<ZodString>;
  type: ZodDefault<ZodLiteral<"image">>;
  url: ZodString;
}, $strip>;
```

Image content block.

***

<a id="api-internalstateresultschema"></a>

### InternalStateResultSchema

```ts
const InternalStateResultSchema: ZodObject<{
  acknowledged: ZodBoolean;
  content_hash: ZodString;
  previous_version: ZodOptional<ZodNumber>;
  session_id: ZodString;
  token_count: ZodNumber;
  version: ZodNumber;
  was_noop: ZodBoolean;
}, $loose>;
```

***

<a id="api-isgenerationresultschema"></a>

### ISGenerationResultSchema

```ts
const ISGenerationResultSchema: ZodObject<{
  coalesced: ZodBoolean;
  content_hash: ZodOptional<ZodString>;
  events_hash: ZodOptional<ZodString>;
  model_id: ZodOptional<ZodString>;
  next_update_at: ZodOptional<ZodString>;
  previous_version: ZodOptional<ZodNumber>;
  skipped_reason: ZodOptional<ZodString>;
  token_count: ZodOptional<ZodNumber>;
  updated: ZodBoolean;
  version: ZodOptional<ZodNumber>;
  was_noop: ZodBoolean;
}, $loose>;
```

***

<a id="api-jsonvalueschema"></a>

### JsonValueSchema

```ts
const JsonValueSchema: z.ZodType<JsonValue>;
```

Runtime schema for the recursive `JsonValue` type.

***

<a id="api-llminvocationoptionsschema"></a>

### LLMInvocationOptionsSchema

```ts
const LLMInvocationOptionsSchema: ZodPreprocess<ZodPipe<ZodObject<{
  frequency_penalty: ZodOptional<ZodNumber>;
  max_tokens: ZodOptional<ZodNumber>;
  parallel_tool_calls: ZodOptional<ZodBoolean>;
  presence_penalty: ZodOptional<ZodNumber>;
  reasoning_effort: ZodOptional<ZodString>;
  response_format: ZodOptional<ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>;
  seed: ZodOptional<ZodNumber>;
  timeout: ZodOptional<ZodNumber>;
  top_k: ZodOptional<ZodNumber>;
  top_p: ZodOptional<ZodNumber>;
}, $loose>, ZodTransform<LLMInvocationOptions, {
[key: string]: unknown;
  frequency_penalty?: number;
  max_tokens?: number;
  parallel_tool_calls?: boolean;
  presence_penalty?: number;
  reasoning_effort?: string;
  response_format?: JsonValue;
  seed?: number;
  timeout?: number;
  top_k?: number;
  top_p?: number;
}>>>;
```

Runtime validation schema for LLMInvocationOptions wire data.

***

<a id="api-llmresponseschema"></a>

### LLMResponseSchema

```ts
const LLMResponseSchema: ZodObject<{
  additional_kwargs: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
  content: ZodString;
  id: ZodOptional<ZodString>;
  metadata: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
  name: ZodOptional<ZodString>;
  response_metadata: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
  tool_calls: ZodOptional<ZodArray<ZodUnknown>>;
  usage: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
}, $loose>;
```

Runtime validation schema for LLMResponse wire data. `content` is the only required field.

***

<a id="api-llmtokenusageschema"></a>

### LLMTokenUsageSchema

```ts
const LLMTokenUsageSchema: ZodPreprocess<ZodPipe<ZodObject<{
  completion_tokens: ZodOptional<ZodNumber>;
  input_tokens: ZodOptional<ZodNumber>;
  model: ZodOptional<ZodString>;
  output_tokens: ZodOptional<ZodNumber>;
  prompt_tokens: ZodOptional<ZodNumber>;
  total_tokens: ZodOptional<ZodNumber>;
}, $strip>, ZodTransform<LLMTokenUsage, {
  completion_tokens?: number;
  input_tokens?: number;
  model?: string;
  output_tokens?: number;
  prompt_tokens?: number;
  total_tokens?: number;
}>>>;
```

Runtime validation schema for LLMTokenUsage wire data. All fields optional, matching the class constructor.

***

<a id="api-llmtoolcallschema"></a>

### LLMToolCallSchema

```ts
const LLMToolCallSchema: ZodPipe<ZodObject<{
  args: ZodOptional<ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>;
  arguments: ZodOptional<ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>;
  id: ZodOptional<ZodString>;
  index: ZodOptional<ZodNumber>;
  name: ZodOptional<ZodString>;
  type: ZodOptional<ZodString>;
}, $strip>, ZodTransform<LLMToolCall, {
  args?: JsonValue;
  arguments?: JsonValue;
  id?: string;
  index?: number;
  name?: string;
  type?: string;
}>>;
```

Runtime validation schema for LLMToolCall wire data.

***

<a id="api-llmtoolschemavalidator"></a>

### LLMToolSchemaValidator

```ts
const LLMToolSchemaValidator: ZodPipe<ZodObject<{
  description: ZodOptional<ZodString>;
  function: ZodOptional<ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>;
  name: ZodOptional<ZodString>;
  parameters: ZodOptional<ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>;
  strict: ZodOptional<ZodBoolean>;
  type: ZodOptional<ZodString>;
}, $strip>, ZodTransform<LLMToolSchema, {
  description?: string;
  function?: JsonValue;
  name?: string;
  parameters?: JsonValue;
  strict?: boolean;
  type?: string;
}>>;
```

Runtime validation schema for LLMToolSchema wire data.

***

<a id="api-memorychunkschema"></a>

### MemoryChunkSchema

```ts
const MemoryChunkSchema: ZodObject<{
  content: ZodString;
  embedding: ZodOptional<ZodArray<ZodNumber>>;
  id: ZodString;
  metadata: ZodOptional<ZodRecord<ZodString, ZodUnknown>>;
  similarity_score: ZodOptional<ZodNumber>;
  source: ZodEnum<{
     episodic: "episodic";
     procedural: "procedural";
     semantic: "semantic";
     stm: "stm";
     taxonomic: "taxonomic";
  }>;
  timestamp: ZodPipe<ZodString, ZodTransform<Date, string>>;
}, $loose>;
```

Unified representation of a retrieved memory chunk across all memory types.

***

<a id="api-memorysource-1"></a>

### MemorySource

```ts
MemorySource: object;
```

#### Type Declaration

| Name | Type |
| :------ | :------ |
| <a id="api-property-episodic"></a> `EPISODIC` | `"episodic"` |
| <a id="api-property-procedural"></a> `PROCEDURAL` | `"procedural"` |
| <a id="api-property-semantic"></a> `SEMANTIC` | `"semantic"` |
| <a id="api-property-stm"></a> `STM` | `"stm"` |
| <a id="api-property-taxonomic"></a> `TAXONOMIC` | `"taxonomic"` |

***

<a id="api-memorysourceschema"></a>

### MemorySourceSchema

```ts
const MemorySourceSchema: ZodEnum<{
  episodic: "episodic";
  procedural: "procedural";
  semantic: "semantic";
  stm: "stm";
  taxonomic: "taxonomic";
}>;
```

***

<a id="api-messageschema"></a>

### MessageSchema

```ts
const MessageSchema: ZodPipe<ZodObject<{
  additional_kwargs: ZodOptional<ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>>;
  content: ZodUnion<readonly [ZodString, ZodArray<ZodDiscriminatedUnion<[ZodObject<{
     text: ZodString;
     type: ZodDefault<...>;
   }, $strip>, ZodObject<{
     mime_type: ZodOptional<...>;
     type: ZodDefault<...>;
     url: ZodString;
   }, $strip>, ZodObject<{
     filename: ZodOptional<...>;
     mime_type: ZodOptional<...>;
     type: ZodDefault<...>;
     url: ZodString;
  }, $strip>], "type">>]>;
  id: ZodOptional<ZodString>;
  is_error: ZodOptional<ZodBoolean>;
  name: ZodOptional<ZodString>;
  response_metadata: ZodOptional<ZodRecord<ZodString, ZodType<JsonValue, unknown, $ZodTypeInternals<JsonValue, unknown>>>>;
  role: ZodEnum<{
     assistant: "assistant";
     system: "system";
     tool: "tool";
     user: "user";
  }>;
  tool_call_id: ZodOptional<ZodString>;
  tool_calls: ZodOptional<ZodArray<ZodPipe<ZodObject<{
     args: ZodOptional<ZodType<JsonValue, unknown, $ZodTypeInternals<..., ...>>>;
     arguments: ZodOptional<ZodType<JsonValue, unknown, $ZodTypeInternals<..., ...>>>;
     id: ZodOptional<ZodString>;
     index: ZodOptional<ZodNumber>;
     name: ZodOptional<ZodString>;
     type: ZodOptional<ZodString>;
   }, $strip>, ZodTransform<LLMToolCall, {
     args?: JsonValue;
     arguments?: JsonValue;
     id?: string;
     index?: number;
     name?: string;
     type?: string;
  }>>>>;
}, $loose>, ZodTransform<Message, {
[key: string]: unknown;
  additional_kwargs?: Record<string, JsonValue>;
  content:   | string
     | (
     | {
     text: string;
     type: "text";
   }
     | {
     mime_type?: string;
     type: "image";
     url: string;
   }
     | {
     filename?: string;
     mime_type?: string;
     type: "document";
     url: string;
   })[];
  id?: string;
  is_error?: boolean;
  name?: string;
  response_metadata?: Record<string, JsonValue>;
  role: "user" | "assistant" | "tool" | "system";
  tool_call_id?: string;
  tool_calls?: LLMToolCall[];
}>>;
```

Runtime validation schema for Message wire data. Parses snake_case wire
fields and transforms them to the camelCase Message interface shape,
matching how LLMToolCallSchema/LLMTokenUsageSchema handle the same mapping.

***

<a id="api-modeltype-1"></a>

### ModelType

```ts
ModelType: object;
```

#### Type Declaration

| Name | Type |
| :------ | :------ |
| <a id="api-property-anthropic"></a> `ANTHROPIC` | `"anthropic"` |
| <a id="api-property-claude-1"></a> `CLAUDE` | `"claude"` |
| <a id="api-property-gemini"></a> `GEMINI` | `"gemini"` |
| <a id="api-property-generic"></a> `GENERIC` | `"generic"` |
| <a id="api-property-openai-1"></a> `OPENAI` | `"openai"` |

***

<a id="api-modeltypeschema"></a>

### ModelTypeSchema

```ts
const ModelTypeSchema: ZodEnum<{
  anthropic: "anthropic";
  claude: "claude";
  gemini: "gemini";
  generic: "generic";
  openai: "openai";
}>;
```

***

<a id="api-promotesnapshotresultschema"></a>

### PromoteSnapshotResultSchema

```ts
const PromoteSnapshotResultSchema: ZodObject<{
  acknowledged: ZodBoolean;
  has_embedding: ZodDefault<ZodBoolean>;
  promoted_count: ZodNumber;
  reason: ZodString;
  session_id: ZodString;
  snapshot_id: ZodOptional<ZodString>;
}, $loose>;
```

***

<a id="api-retrievedcustommemoryschema"></a>

### RetrievedCustomMemorySchema

```ts
const RetrievedCustomMemorySchema: ZodObject<{
  agent_id: ZodOptional<ZodNullable<ZodString>>;
  content: ZodString;
  contextual_metadata: ZodOptional<ZodNullable<ZodRecord<ZodString, ZodUnknown>>>;
  id: ZodString;
  tags: ZodRecord<ZodString, ZodUnion<readonly [ZodString, ZodNumber, ZodBoolean]>>;
  type: ZodString;
  user_id: ZodOptional<ZodNullable<ZodString>>;
}, $strip>;
```

***

<a id="api-sessionmessageschema"></a>

### SessionMessageSchema

```ts
const SessionMessageSchema: ZodObject<{
  additional_kwargs: ZodDefault<ZodNullable<ZodRecord<ZodString, ZodUnknown>>>;
  content: ZodString;
  id: ZodString;
  name: ZodDefault<ZodString>;
  role: ZodString;
  session_id: ZodString;
  timestamp: ZodString;
  tool_call_id: ZodDefault<ZodNullable<ZodString>>;
  tool_calls: ZodDefault<ZodNullable<ZodArray<ZodPipe<ZodObject<{
     args: ZodOptional<ZodType<JsonValue, unknown, $ZodTypeInternals<..., ...>>>;
     arguments: ZodOptional<ZodType<JsonValue, unknown, $ZodTypeInternals<..., ...>>>;
     id: ZodOptional<ZodString>;
     index: ZodOptional<ZodNumber>;
     name: ZodOptional<ZodString>;
     type: ZodOptional<ZodString>;
   }, $strip>, ZodTransform<LLMToolCall, {
     args?: JsonValue;
     arguments?: JsonValue;
     id?: string;
     index?: number;
     name?: string;
     type?: string;
  }>>>>>;
}, $strip>;
```

A single message in a session's conversation history.

`role` uses the same vocabulary as `Message` ("user", "assistant",
"tool", "system"). `name` carries the tool name on tool messages.

Mirrors Python's `SessionMessage` in `agent_engine_sdk/models.py`.

***

<a id="api-sessionmessagesresponseschema"></a>

### SessionMessagesResponseSchema

```ts
const SessionMessagesResponseSchema: ZodObject<{
  messages: ZodArray<ZodObject<{
     additional_kwargs: ZodDefault<ZodNullable<ZodRecord<ZodString, ZodUnknown>>>;
     content: ZodString;
     id: ZodString;
     name: ZodDefault<ZodString>;
     role: ZodString;
     session_id: ZodString;
     timestamp: ZodString;
     tool_call_id: ZodDefault<ZodNullable<ZodString>>;
     tool_calls: ZodDefault<ZodNullable<ZodArray<ZodPipe<ZodObject<{
        args: ...;
        arguments: ...;
        id: ...;
        index: ...;
        name: ...;
        type: ...;
      }, $strip>, ZodTransform<LLMToolCall, {
        args?: ...;
        arguments?: ...;
        id?: ...;
        index?: ...;
        name?: ...;
        type?: ...;
     }>>>>>;
  }, $strip>>;
}, $strip>;
```

Response for the AER's session-messages endpoint.

***

<a id="api-sessionssummaryresponseschema"></a>

### SessionsSummaryResponseSchema

```ts
const SessionsSummaryResponseSchema: ZodObject<{
  sessions: ZodArray<ZodObject<{
     created_at: ZodString;
     first_message_preview: ZodDefault<ZodString>;
     last_activity: ZodString;
     message_count: ZodNumber;
     session_id: ZodString;
  }, $strip>>;
}, $strip>;
```

Response for the AER's session-summary enrichment endpoint.

***

<a id="api-sessionsummaryschema"></a>

### SessionSummarySchema

```ts
const SessionSummarySchema: ZodObject<{
  created_at: ZodString;
  first_message_preview: ZodDefault<ZodString>;
  last_activity: ZodString;
  message_count: ZodNumber;
  session_id: ZodString;
}, $strip>;
```

Per-session summary sourced from framework-specific persistence.

Returned by the AER's session-enrichment endpoint. The platform layer
(OE) merges this with its own tenant fields (user_id, project_id,
workspace_id, visibility) when assembling the public sessions list.

Mirrors Python's `SessionSummary` in `agent_engine_sdk/models.py`.

***

<a id="api-streameventschema"></a>

### StreamEventSchema

```ts
const StreamEventSchema: ZodObject<{
  data: ZodUnknown & ZodType<
     | {
   }
     | null, unknown, $ZodTypeInternals<
     | {
   }
    | null, unknown>>;
  event: ZodOptional<ZodString>;
}, $loose>;
```

Runtime validation schema for StreamEvent wire data.

***

<a id="api-tagscalarschema"></a>

### TagScalarSchema

```ts
const TagScalarSchema: ZodUnion<readonly [ZodString, ZodNumber, ZodBoolean]>;
```

***

<a id="api-textblockschema"></a>

### TextBlockSchema

```ts
const TextBlockSchema: ZodObject<{
  text: ZodString;
  type: ZodDefault<ZodLiteral<"text">>;
}, $strip>;
```

Text content block.

***

<a id="api-toolcallchunkschema"></a>

### ToolCallChunkSchema

```ts
const ToolCallChunkSchema: ZodObject<{
  args: ZodOptional<ZodString>;
  id: ZodOptional<ZodString>;
  index: ZodOptional<ZodNumber>;
  name: ZodOptional<ZodString>;
  type: ZodOptional<ZodString>;
}, $strip>;
```

Runtime validation schema for ToolCallChunk wire data.

***

<a id="api-tooldefinitionschema"></a>

### ToolDefinitionSchema

```ts
const ToolDefinitionSchema: ZodObject<{
  args_schema: ZodRecord<ZodString, ZodUnknown>;
  callable: ZodCustom<(...args) => unknown, (...args) => unknown>;
  description: ZodString;
  name: ZodString;
  network: ZodDefault<ZodArray<ZodString>>;
  provider_type: ZodOptional<ZodString>;
  redact_fields: ZodDefault<ZodArray<ZodString>>;
  remote: ZodDefault<ZodBoolean>;
  scopes: ZodDefault<ZodArray<ZodString>>;
  timeout_seconds: ZodDefault<ZodNumber>;
}, $strip>;
```

Framework-neutral tool definition.

***

<a id="api-writeturnresultschema"></a>

### WriteTurnResultSchema

```ts
const WriteTurnResultSchema: ZodObject<{
  acknowledged: ZodBoolean;
  id: ZodString;
  session_id: ZodString;
  turn_seq: ZodNumber;
}, $loose>;
```

## Functions

<a id="api-createtooldefinition"></a>

### createToolDefinition()

```ts
function createToolDefinition(input): object;
```

Constructs a ToolDefinition, applying Python-parity defaults (remote=true, network=[], timeout_seconds=30, redact_fields=[]).

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `input` | \{ `args_schema`: `Record`\<`string`, `unknown`\>; `callable`: (...`args`) => `unknown`; `description`: `string`; `name`: `string`; `network?`: `string`[]; `provider_type?`: `string`; `redact_fields?`: `string`[]; `remote?`: `boolean`; `scopes?`: `string`[]; `timeout_seconds?`: `number`; \} |
| `input.args_schema` | `Record`\<`string`, `unknown`\> |
| `input.callable` | (...`args`) => `unknown` |
| `input.description` | `string` |
| `input.name` | `string` |
| `input.network?` | `string`[] |
| `input.provider_type?` | `string` |
| `input.redact_fields?` | `string`[] |
| `input.remote?` | `boolean` |
| `input.scopes?` | `string`[] |
| `input.timeout_seconds?` | `number` |

#### Returns

`object`

| Name | Type |
| :------ | :------ |
| `args_schema` | `Record`\<`string`, `unknown`\> |
| `callable()` | (...`args`) => `unknown` |
| `description` | `string` |
| `name` | `string` |
| `network` | `string`[] |
| `provider_type?` | `string` |
| `redact_fields` | `string`[] |
| `remote` | `boolean` |
| `scopes` | `string`[] |
| `timeout_seconds` | `number` |

***

<a id="api-serializemessage"></a>

### serializeMessage()

```ts
function serializeMessage(message): Record<string, unknown>;
```

Serialize a Message to its snake_case wire shape — the camel→snake "dump"
half that mirrors Python's `model_dump(by_alias=True)`. `MessageSchema` is
the inverse snake→camel "parse" half.

Keeping these two symmetric is what makes the representation safe: messages
are camelCase internally (the `Message` interface) and snake_case on the
wire, and conversion happens exactly once at each boundary. Use this when
dumping an already-internal `Message`; never re-run an internal Message
through `MessageSchema`, whose transform would blank the camelCase fields.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `message` | [`Message`](#api-message) |

#### Returns

`Record`\<`string`, `unknown`\>
