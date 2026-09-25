# @mongodb-js/agent-engine-sdk-langgraph

## Table of Contents

- **Classes**
  - [AgentEngineToolPodBackend](#api-agentenginetoolpodbackend)
    - [downloadFiles()](#api-downloadfiles)
    - [edit()](#api-edit)
    - [execute()](#api-execute)
    - [glob()](#api-glob)
    - [grep()](#api-grep)
    - [ls()](#api-ls)
    - [read()](#api-read)
    - [readRaw()](#api-readraw)
    - [uploadFiles()](#api-uploadfiles)
    - [write()](#api-write)
  - [App](#api-app)
    - [checkpointer()](#api-checkpointer)
    - [close()](#api-close)
    - [deepAgent()](#api-deepagent)
    - [entrypoint()](#api-entrypoint)
    - [finishSession()](#api-finishsession)
    - [getAgent()](#api-getagent)
    - [getCurrentSessionId()](#api-getcurrentsessionid)
    - [getCurrentUserId()](#api-getcurrentuserid)
    - [getToolDefinitions()](#api-gettooldefinitions)
    - [getTools()](#api-gettools)
    - [getToolSchemas()](#api-gettoolschemas)
    - [llm()](#api-llm)
    - [prepareAgentInput()](#api-prepareagentinput)
    - [ready()](#api-ready)
    - [resolveThreadId()](#api-resolvethreadid)
    - [run()](#api-run)
    - [suspend()](#api-suspend)
    - [tool()](#api-tool)
    - [tools()](#api-tools)
    - [warmUp()](#api-warmup)
  - [LangGraphBaseAgent](#api-langgraphbaseagent)
    - [execute()](#api-execute-1)
    - [invoke()](#api-invoke)
    - [resume()](#api-resume)
    - [stream()](#api-stream)
  - [LangGraphQueryPlugin](#api-langgraphqueryplugin)
    - [getMessagesForSession()](#api-getmessagesforsession)
    - [getSummariesForSessions()](#api-getsummariesforsessions)
- **Interfaces**
  - [CreateAgentEngineDeepAgentOptions](#api-createagentenginedeepagentoptions)
  - [DeepAgentOptions](#api-deepagentoptions)
  - [LangGraphQueryPluginParams](#api-langgraphquerypluginparams)
- **Type Aliases**
  - [PrepareAgentInput](#api-prepareagentinput-1)
  - [ResolveThreadId](#api-resolvethreadid-2)
  - [SessionFinishStatus](#api-sessionfinishstatus)
- **Variables**
  - [MAX\_SUBAGENT\_NESTING\_DEPTH](#api-max_subagent_nesting_depth)
- **Functions**
  - [createAgentEngineDeepAgent()](#api-createagentenginedeepagent)
  - [validateSubagentTree()](#api-validatesubagenttree)
- **References**
  - [AgentEngineToolSandboxBackend](#api-agentenginetoolsandboxbackend)

## Classes

<a id="api-agentenginetoolpodbackend"></a>

### AgentEngineToolPodBackend

Backend that routes every operation through Atlas Agent Engine's secure path. Each
method delegates to `SecureToolWrapper.executeTool`, which sends the request
to the OE for policy approval and audit logging before it runs in the Tool
Pod.

#### Implements

- `SandboxBackendProtocolV2`

<a id="api-constructor"></a>

#### Constructor

```ts
new AgentEngineToolPodBackend(): AgentEngineToolPodBackend;
```

**Returns**

[`AgentEngineToolPodBackend`](#api-agentenginetoolpodbackend)

#### Properties

| Property | Modifier | Type | Default value | Description |
| :------ | :------ | :------ | :------ | :------ |
| <a id="api-property-id"></a> `id` | `readonly` | `"agent-engine-toolpod"` | `"agent-engine-toolpod"` | Unique identifier for the sandbox backend instance |

#### Methods

<a id="api-downloadfiles"></a>

##### downloadFiles()

```ts
downloadFiles(paths): Promise<FileDownloadResponse[]>;
```

Download files via the audited `filesystem_download` handler — one call per
path so each is policy-checked and logged independently — then base64-decode
the `content_base64` field into raw bytes. Responses preserve input order;
on failure `content` is null and `error` carries a classified message.

Native-mode batches fan out on a bounded worker pool so skills loading
pays one round-trip of latency instead of a sum. Durable attempts stay
sequential: unkeyed activities are exclusive under the attempt gate, and
overlapping them raises CONFLICT.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `paths` | `string`[] |

**Returns**

`Promise`\<`FileDownloadResponse`[]\>

**Implementation of**

```ts
SandboxBackendProtocolV2.downloadFiles
```

<a id="api-edit"></a>

##### edit()

```ts
edit(
   filePath,
   oldString,
   newString,
   replaceAll?
): Promise<EditResult>;
```

Edit a file by replacing string occurrences.

**Parameters**

| Parameter | Type | Default value | Description |
| :------ | :------ | :------ | :------ |
| `filePath` | `string` | `undefined` | Absolute file path |
| `oldString` | `string` | `undefined` | String to find and replace |
| `newString` | `string` | `undefined` | Replacement string |
| `replaceAll` | `boolean` | `false` | If true, replace all occurrences (default: false) |

**Returns**

`Promise`\<`EditResult`\>

EditResult with error, path, filesUpdate, and occurrences

**Implementation of**

```ts
SandboxBackendProtocolV2.edit
```

<a id="api-execute"></a>

##### execute()

```ts
execute(command): Promise<ExecuteResponse>;
```

Execute a command in the sandbox.

**Parameters**

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `command` | `string` | Full shell command string to execute |

**Returns**

`Promise`\<`ExecuteResponse`\>

ExecuteResponse with combined output, exit code, and truncation flag

**Implementation of**

```ts
SandboxBackendProtocolV2.execute
```

<a id="api-glob"></a>

##### glob()

```ts
glob(pattern, path?): Promise<GlobResult>;
```

Structured glob matching returning FileInfo objects.

**Parameters**

| Parameter | Type | Default value | Description |
| :------ | :------ | :------ | :------ |
| `pattern` | `string` | `undefined` | Glob pattern (e.g., `*.py`, `**/*.ts`) |
| `path` | `string` | `"/"` | Base path to search from (default: "/") |

**Returns**

`Promise`\<`GlobResult`\>

GlobResult with list of FileInfo objects matching the pattern on success or error on failure

**Implementation of**

```ts
SandboxBackendProtocolV2.glob
```

<a id="api-grep"></a>

##### grep()

```ts
grep(
   pattern,
   path?,
   glob?
): Promise<GrepResult>;
```

Search file contents for a literal text pattern.

Binary files (determined by MIME type) are skipped.

**Parameters**

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `pattern` | `string` | Literal text pattern to search for |
| `path?` | `string` \| `null` | Base path to search from (default: null) |
| `glob?` | `string` \| `null` | Optional glob pattern to filter files (e.g., "*.py") |

**Returns**

`Promise`\<`GrepResult`\>

GrepResult with matches on success or error on failure

**Implementation of**

```ts
SandboxBackendProtocolV2.grep
```

<a id="api-ls"></a>

##### ls()

```ts
ls(path): Promise<LsResult>;
```

Structured listing with file metadata.

Lists files and directories in the specified directory (non-recursive).
Directories have a trailing / in their path and is_dir=true.

**Parameters**

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `path` | `string` | Absolute path to directory |

**Returns**

`Promise`\<`LsResult`\>

LsResult with list of FileInfo objects on success or error on failure

**Implementation of**

```ts
SandboxBackendProtocolV2.ls
```

<a id="api-read"></a>

##### read()

```ts
read(
   filePath,
   offset?,
   limit?
): Promise<ReadResult>;
```

Read file content.

For text files, content is paginated by line offset/limit.
For binary files, the full raw Uint8Array content is returned.

**Parameters**

| Parameter | Type | Default value | Description |
| :------ | :------ | :------ | :------ |
| `filePath` | `string` | `undefined` | Absolute file path |
| `offset` | `number` | `0` | Line offset to start reading from (0-indexed), default 0 |
| `limit` | `number` | `2000` | Maximum number of lines to read, default 500 |

**Returns**

`Promise`\<`ReadResult`\>

ReadResult with content on success or error on failure

**Implementation of**

```ts
SandboxBackendProtocolV2.read
```

<a id="api-readraw"></a>

##### readRaw()

```ts
readRaw(filePath): Promise<ReadRawResult>;
```

Raw read is not backed by a dedicated wire handler; the Tool Pod exposes
only line-oriented `filesystem_read`. We wrap its text content in a v2
FileData so binary-unaware callers still work. Genuine binary reads should
use `downloadFiles`, which carries raw bytes base64-encoded.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `filePath` | `string` |

**Returns**

`Promise`\<`ReadRawResult`\>

**Implementation of**

```ts
SandboxBackendProtocolV2.readRaw
```

<a id="api-uploadfiles"></a>

##### uploadFiles()

```ts
uploadFiles(files): Promise<FileUploadResponse[]>;
```

Upload multiple files.
Optional - backends that don't support file upload can omit this.

**Parameters**

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `files` | \[`string`, `Uint8Array`\<`ArrayBufferLike`\>\][] | List of [path, content] tuples to upload |

**Returns**

`Promise`\<`FileUploadResponse`[]\>

List of FileUploadResponse objects, one per input file

**Implementation of**

```ts
SandboxBackendProtocolV2.uploadFiles
```

<a id="api-write"></a>

##### write()

```ts
write(filePath, content): Promise<WriteResult>;
```

Create a new file.

**Parameters**

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `filePath` | `string` | Absolute file path |
| `content` | `string` | File content as string |

**Returns**

`Promise`\<`WriteResult`\>

WriteResult with error populated on failure

**Implementation of**

```ts
SandboxBackendProtocolV2.write
```

***

<a id="api-app"></a>

### App

LangChain SDK for the Atlas Agent Engine.

#### Example

```ts
import { App } from '@mongodb-js/agent-engine-sdk-langgraph';

const app = new App({ appName: 'My Agent' });

app.tool()((args: { query: string }) => 'result');

app.entrypoint(() => {
  const llm = app.llm(chatModel);
  const checkpointer = app.checkpointer();
  const tools = app.getTools();
  // Build LangGraph...
  return graph;
});
```

#### Extends

- `BaseApp`

<a id="api-constructor-1"></a>

#### Constructor

```ts
new App(options): App;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `options` | `AppOptions` |

**Returns**

[`App`](#api-app)

**Overrides**

```ts
BaseApp.constructor
```

#### Properties

| Property | Modifier | Type | Inherited from |
| :------ | :------ | :------ | :------ |
| <a id="api-property-llmwrapper"></a> `llmWrapper` | `public` | `unknown` | `BaseApp.llmWrapper` |
| <a id="api-property-name"></a> `name` | `readonly` | `string` | `BaseApp.name` |
| <a id="api-property-toolwrapper"></a> `toolWrapper` | `public` | `unknown` | `BaseApp.toolWrapper` |

#### Accessors

<a id="api-agentconfig"></a>

##### agentConfig

**Get Signature**

```ts
get agentConfig(): RuntimeAgentConfig;
```

**Returns**

`RuntimeAgentConfig`

<a id="api-memory"></a>

##### memory

**Get Signature**

```ts
get memory(): Memory;
```

Unified memory facade over app-bound adapters.

Lazily constructed on first access and cached (a long-lived singleton over
per-request context). Operations resolve the end-user and session from the
ambient execution context, so agent code calls `app.memory.saveSemantic(...)`
without threading identity through. Requests route through the OE memory
proxy; memory is reachable only while handling a platform request.

**Returns**

`Memory`

#### Methods

<a id="api-checkpointer"></a>

##### checkpointer()

```ts
checkpointer(): PlatformCheckpointer | null;
```

Get the platform checkpointer for LangGraph.

Lazily constructs and wraps a `MongoDBSaver` on first call in AER mode.
Native sessions use that saver through the request-scoped platform
wrapper. The underlying `MongoClient` is closed by `App.close()`.

The database name is read from `CHECKPOINT_DB_NAME` when set (exact
override, no project scoping). Otherwise the existing
`MDB_AGENTIC_STORE_DB` / per-project store resolution is used
(default: `"mdb_store"`). The URI comes from `MONGODB_URI` — the same
source `TenantRuntime` uses internally.

Mirrors Python `runtime.py:checkpointer()`.

**Returns**

`PlatformCheckpointer` \| `null`

<a id="api-close"></a>

##### close()

```ts
close(): Promise<void>;
```

Release resources held by this App instance.

**Returns**

`Promise`\<`void`\>

<a id="api-deepagent"></a>

##### deepAgent()

```ts
deepAgent(llm, options?): DeepAgent<DeepAgentTypeConfig<ResponseFormatUndefined, undefined, InteropZodObject, readonly [AgentMiddleware<ZodObject<{
}, "strip", ZodTypeAny, {
}, {
}>, undefined, unknown, readonly [DynamicStructuredTool<ZodObject<{
}, "strip", ZodTypeAny, {
}, {
}>, {
}, {
}, Command<unknown, {
}, string>, unknown, "write_todos">]>, AgentMiddleware<StateSchema<{
}>, undefined, unknown, (
  | DynamicStructuredTool<ZodObject<{
}, $strip>, {
}, {
}, string, unknown, "ls">
  | DynamicStructuredTool<ZodObject<{
}, $strip>, {
}, {
}, object[] | object[], unknown, "read_file">
  | DynamicStructuredTool<ZodObject<{
}, $strip>, {
}, {
},
  | string
  | ToolMessage<MessageStructure<MessageToolSet>>
  | Command<unknown, {
}, string>, unknown, "write_file">
  | DynamicStructuredTool<ZodObject<{
}, $strip>, {
}, {
},
  | string
  | ToolMessage<MessageStructure<MessageToolSet>>
  | Command<unknown, {
}, string>, unknown, "edit_file">
  | DynamicStructuredTool<ZodObject<{
}, $strip>, {
}, {
}, string, unknown, "glob">
  | DynamicStructuredTool<ZodObject<{
}, $strip>, {
}, {
}, string, unknown, "grep">
  | DynamicStructuredTool<ZodObject<{
}, $strip>, {
}, {
}, string, unknown, "execute">)[]>, AgentMiddleware<undefined, undefined, unknown, readonly [DynamicStructuredTool<ZodObject<{
}, $strip>, {
}, {
},
  | string
  | Command<unknown, Record<string, unknown>, string>, unknown, "task">]>, AgentMiddleware<ZodObject<{
}, $strip>, undefined, unknown, readonly (ClientTool | ServerTool)[]>, AgentMiddleware<undefined, undefined, unknown, readonly (ClientTool | ServerTool)[]>, AgentMiddleware<any, any, any, readonly (ClientTool | ServerTool)[]>], readonly (ClientTool | ServerTool)[], readonly AnySubAgent[], readonly () => StreamTransformer<any>[]>>;
```

Build a deepagents graph pre-wired for Atlas Agent Engine AER.

Wraps `llm` in `SecureWrappedLLM`, resolves relative skill paths, validates
the subagent tree (string models are rejected — they would bypass OE
routing), then delegates to deepagents' `createDeepAgent`.

Each `skills` entry is a parent source directory. At runtime, deepagents
lists it through the configured backend and treats each immediate child
directory containing `SKILL.md` as one skill; discovery is not recursive.
deepagents skips unreadable or unparsable frontmatter and skills missing
`name` or `description`; it warns but may still load Agent Skills naming or
directory-name violations.

Mirrors Python `runtime.py:App.deep_agent()`.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `llm` | `BaseChatModel` |
| `options` | [`DeepAgentOptions`](#api-deepagentoptions) |

**Returns**

`DeepAgent`\<`DeepAgentTypeConfig`\<`ResponseFormatUndefined`, `undefined`, `InteropZodObject`, readonly \[`AgentMiddleware`\<`ZodObject`\<\{
\}, `"strip"`, `ZodTypeAny`, \{
\}, \{
\}\>, `undefined`, `unknown`, readonly \[`DynamicStructuredTool`\<`ZodObject`\<\{
\}, `"strip"`, `ZodTypeAny`, \{
\}, \{
\}\>, \{
\}, \{
\}, `Command`\<`unknown`, \{
\}, `string`\>, `unknown`, `"write_todos"`\>\]\>, `AgentMiddleware`\<`StateSchema`\<\{
\}\>, `undefined`, `unknown`, (
  \| `DynamicStructuredTool`\<`ZodObject`\<\{
\}, `$strip`\>, \{
\}, \{
\}, `string`, `unknown`, `"ls"`\>
  \| `DynamicStructuredTool`\<`ZodObject`\<\{
\}, `$strip`\>, \{
\}, \{
\}, `object`[] \| `object`[], `unknown`, `"read_file"`\>
  \| `DynamicStructuredTool`\<`ZodObject`\<\{
\}, `$strip`\>, \{
\}, \{
\},
  \| `string`
  \| `ToolMessage`\<`MessageStructure`\<`MessageToolSet`\>\>
  \| `Command`\<`unknown`, \{
\}, `string`\>, `unknown`, `"write_file"`\>
  \| `DynamicStructuredTool`\<`ZodObject`\<\{
\}, `$strip`\>, \{
\}, \{
\},
  \| `string`
  \| `ToolMessage`\<`MessageStructure`\<`MessageToolSet`\>\>
  \| `Command`\<`unknown`, \{
\}, `string`\>, `unknown`, `"edit_file"`\>
  \| `DynamicStructuredTool`\<`ZodObject`\<\{
\}, `$strip`\>, \{
\}, \{
\}, `string`, `unknown`, `"glob"`\>
  \| `DynamicStructuredTool`\<`ZodObject`\<\{
\}, `$strip`\>, \{
\}, \{
\}, `string`, `unknown`, `"grep"`\>
  \| `DynamicStructuredTool`\<`ZodObject`\<\{
\}, `$strip`\>, \{
\}, \{
\}, `string`, `unknown`, `"execute"`\>)[]\>, `AgentMiddleware`\<`undefined`, `undefined`, `unknown`, readonly \[`DynamicStructuredTool`\<`ZodObject`\<\{
\}, `$strip`\>, \{
\}, \{
\},
  \| `string`
  \| `Command`\<`unknown`, `Record`\<`string`, `unknown`\>, `string`\>, `unknown`, `"task"`\>\]\>, `AgentMiddleware`\<`ZodObject`\<\{
\}, `$strip`\>, `undefined`, `unknown`, readonly (`ClientTool` \| `ServerTool`)[]\>, `AgentMiddleware`\<`undefined`, `undefined`, `unknown`, readonly (`ClientTool` \| `ServerTool`)[]\>, `AgentMiddleware`\<`any`, `any`, `any`, readonly (`ClientTool` \| `ServerTool`)[]\>\], readonly (`ClientTool` \| `ServerTool`)[], readonly `AnySubAgent`[], readonly () => `StreamTransformer`\<`any`\>[]\>\>

**Throws**

`features.deep_agent` is not enabled, or a subagent spec
  uses a string model, or nesting exceeds the recursion cap.

<a id="api-entrypoint"></a>

##### entrypoint()

```ts
entrypoint<F>(fn): F;
```

Mark the graph builder function.

**Type Parameters**

| Type Parameter |
| :------ |
| `F` *extends* () => `unknown` |

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `fn` | `F` |

**Returns**

`F`

**Overrides**

```ts
BaseApp.entrypoint
```

<a id="api-finishsession"></a>

##### finishSession()

```ts
finishSession(): SessionFinishStatus;
```

Mark this session finished so the platform frees its compute now.

Call it when the agent is done with the session. The current turn keeps
running and returns its result normally; once it completes, the platform
cancels any live sub-agent runs and releases the session's AER and tool
pods instead of holding them until the idle timeout expires.

Safe to call more than once: the first call returns "requested", later
ones "already_requested". Outside an agent run (local scripts, tool pods)
there is no session to finish and the call returns "unavailable" without
throwing. Calling it after the turn has already ended - e.g. from a
setTimeout or a floating promise scheduled during the turn but resolving
after it - also returns "unavailable": by then nothing is listening for
the request anymore, so reporting "requested" would promise a release
that will never happen.

A turn that suspends for human review, or that fails, keeps its
resources so it stays resumable and diagnosable; the session then falls
back to the idle timeout.

**Returns**

[`SessionFinishStatus`](#api-sessionfinishstatus)

<a id="api-getagent"></a>

##### getAgent()

```ts
getAgent(opts?): LangGraphBaseAgent;
```

Build and return a `LangGraphBaseAgent` instance.

Matches `GraphBuilderLike.getAgent({ callbacks? })` — agent-engine-runner-shared's
`TenantRuntime.registerAndRun` introspects this method to compile the graph.
Accepts an options object with an optional `callbacks` array, each wrapped
in `LangGraphCallbackAdapter` before being passed to the graph.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `opts?` | \{ `callbacks?`: readonly `unknown`[]; \} |
| `opts.callbacks?` | readonly `unknown`[] |

**Returns**

[`LangGraphBaseAgent`](#api-langgraphbaseagent)

<a id="api-getcurrentsessionid"></a>

##### getCurrentSessionId()

```ts
getCurrentSessionId(): string | null;
```

**Returns**

`string` \| `null`

<a id="api-getcurrentuserid"></a>

##### getCurrentUserId()

```ts
getCurrentUserId(): string | null;
```

**Returns**

`string` \| `null`

<a id="api-gettooldefinitions"></a>

##### getToolDefinitions()

```ts
getToolDefinitions(): object[];
```

Returns all registered tool definitions. Used by the OE to obtain tool execution information.

**Returns**

`object`[]

**Overrides**

```ts
BaseApp.getToolDefinitions
```

<a id="api-gettools"></a>

##### getTools()

```ts
getTools(): readonly StructuredTool<ToolInputSchemaBase, any, any, any, unknown>[];
```

Get wrapped tools for LangGraph's `ToolNode`.

In AER mode, every tool is wrapped with `SecureToolWrapper` (via
`createSecureToolFunction`) so executions route through OE for logging
and policy enforcement. In other modes, the raw LangChain tools are
returned unchanged.

**Returns**

readonly `StructuredTool`\<`ToolInputSchemaBase`, `any`, `any`, `any`, `unknown`\>[]

<a id="api-gettoolschemas"></a>

##### getToolSchemas()

```ts
getToolSchemas(): readonly unknown[];
```

Get tool schemas for `llm.bindTools()`. Always the unwrapped LangChain tools.

**Returns**

readonly `unknown`[]

<a id="api-llm"></a>

##### llm()

```ts
llm(llm, llmId?): BaseChatModel;
```

Wrap a LangChain LLM for audited I/O through the Orchestration Engine.

In AER mode the LLM is wrapped in `SecureWrappedLLM`. In TOOL mode the raw
LLM is returned unwrapped (the tool pod is the execution end of the chain).

For agents with a single LLM, call without `llmId`. For agents with
multiple LLMs every call must supply a unique `llmId`:
  const fast    = app.llm(ChatOpenAI("gpt-5.4-mini"), "fast")
  const primary = app.llm(ChatOpenAI("gpt-5.4"),      "primary")
Unnamed calls register under the sentinel id `"__default__"`; a second
unnamed call therefore raises the same duplicate-id error as a second
named call with the same id.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `llm` | `BaseChatModel` |
| `llmId?` | `string` |

**Returns**

`BaseChatModel`

<a id="api-prepareagentinput"></a>

##### prepareAgentInput()

```ts
prepareAgentInput<F>(fn): F;
```

Register a hook that builds the graph's starting input from the caller's
`AgentInput` and `RequestContext` for a fresh execution. Resume stays
platform-managed. Returns the function unchanged so it can be used as a
decorator. Port of Python's `@app.prepare_agent_input`.

**Type Parameters**

| Type Parameter |
| :------ |
| `F` *extends* [`PrepareAgentInput`](#api-prepareagentinput-1) |

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `fn` | `F` |

**Returns**

`F`

<a id="api-ready"></a>

##### ready()

```ts
ready(): Promise<void>;
```

Resolved once configured MCP servers have been discovered and registered
as tools. `TenantRuntime.runAsync()` awaits this (via `GraphBuilderLike.ready()`)
before starting the server.

**Returns**

`Promise`\<`void`\>

<a id="api-resolvethreadid"></a>

##### resolveThreadId()

```ts
resolveThreadId<F>(fn): F;
```

Register a hook that builds the LangGraph checkpoint `thread_id`.

Callers manage Atlas Agent Engine `session_id` (and authenticated `user_id`). The
agent owns how those map to the LangGraph checkpoint key. When registered,
the hook's return value is used verbatim on every invocation — fresh and
resume — with no workspace suffix appended. When no hook is registered,
the adapter derives `session_id:workspace_id` as today.

Custom keys are invisible to Atlas Agent Engine session-history queries
(`/query/sessions*`), which still look up only the default
session/workspace-derived keys. Agents that bypass workspace scoping also
own collision isolation within the checkpoint database.

Port of Python's `@app.resolve_thread_id`.

**Type Parameters**

| Type Parameter |
| :------ |
| `F` *extends* [`ResolveThreadId`](#api-resolvethreadid-2) |

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `fn` | `F` |

**Returns**

`F`

**Example**

```ts
app.resolveThreadId((ctx) => `${ctx.sessionId}__${actorFrom(ctx.userId)}`);
```

<a id="api-run"></a>

##### run()

```ts
run(options?): Promise<void>;
```

Start the agent service.

Async because the per-project store-DB name is resolved against the live
cluster (an async listDatabases round trip in the Node driver) before the
query plugin and checkpointer are wired, so AER writes land in the same
database the OE reads. The synchronous framework hooks are still registered
before the first `await`, preserving their ordering relative to startup.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `options` | `Record`\<`string`, `unknown`\> |

**Returns**

`Promise`\<`void`\>

<a id="api-suspend"></a>

##### suspend()

```ts
suspend(reason, context): string;
```

Generate a suspend command. If a tool should suspend, return the result of
this method instead of completing normally.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `reason` | `string` |
| `context` | `Record`\<`string`, `unknown`\> |

**Returns**

`string`

<a id="api-tool"></a>

##### tool()

```ts
tool(options?): <F>(fn) => F;
```

Register a tool function.

Returns a function that, when applied to a tool function, registers it
and returns the function unchanged. Mirrors Python's `@app.tool()` shape.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `options` | `ToolDecoratorOptions` |

**Returns**

\<`F`\>(`fn`) => `F`

**Overrides**

```ts
BaseApp.tool
```

<a id="api-tools"></a>

##### tools()

```ts
tools(): unknown[];
```

Returns a list of wrapped, framework-specific tools.

**Returns**

`unknown`[]

**Overrides**

```ts
BaseApp.tools
```

<a id="api-warmup"></a>

##### warmUp()

```ts
warmUp(): void;
```

Build and cache the graph when explicitly requested. The TypeScript AER
intentionally leaves construction on the existing lazy `/execute` path:
this synchronous builder cannot run during standby warming without
blocking the Node event loop and server health.

**Returns**

`void`

***

<a id="api-langgraphbaseagent"></a>

### LangGraphBaseAgent

LangGraph adapter implementing the BaseAgent protocol.

#### Implements

- `BaseAgent`

<a id="api-constructor-2"></a>

#### Constructor

```ts
new LangGraphBaseAgent(
   graph,
   callbacks?,
   prepareInput?,
   resolveThreadId?
): LangGraphBaseAgent;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `graph` | `CompiledStateGraph`\<`unknown`, `unknown`\> |
| `callbacks?` | readonly `unknown`[] |
| `prepareInput?` | [`PrepareAgentInput`](#api-prepareagentinput-1) \| `null` |
| `resolveThreadId?` | [`ResolveThreadId`](#api-resolvethreadid-2) \| `null` |

**Returns**

[`LangGraphBaseAgent`](#api-langgraphbaseagent)

#### Accessors

<a id="api-compiledgraph"></a>

##### compiledGraph

**Get Signature**

```ts
get compiledGraph(): CompiledStateGraph<unknown, unknown>;
```

The wrapped compiled graph (Python `compiled_graph` parity).

**Returns**

`CompiledStateGraph`\<`unknown`, `unknown`\>

<a id="api-resolvethreadid-1"></a>

##### resolveThreadId

**Get Signature**

```ts
get resolveThreadId(): ResolveThreadId | null;
```

The tenant thread-id hook, or null (Python `_resolve_thread_id` parity).

**Returns**

[`ResolveThreadId`](#api-resolvethreadid-2) \| `null`

#### Methods

<a id="api-execute-1"></a>

##### execute()

```ts
execute(ctx, input): ExecutionResult;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `ctx` | `RequestContext` |
| `input` | `AgentInput` |

**Returns**

`ExecutionResult`

**Implementation of**

```ts
BaseAgent.execute
```

<a id="api-invoke"></a>

##### invoke()

```ts
invoke(ctx, input): Promise<AgentOutput>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `ctx` | `RequestContext` |
| `input` | `AgentInput` |

**Returns**

`Promise`\<`AgentOutput`\>

<a id="api-resume"></a>

##### resume()

```ts
resume(
   ctx,
   input,
   decision
): Promise<AgentOutput>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `ctx` | `RequestContext` |
| `input` | `AgentInput` |
| `decision` | `string` |

**Returns**

`Promise`\<`AgentOutput`\>

<a id="api-stream"></a>

##### stream()

```ts
stream(ctx, input): AsyncIterable<StreamEvent>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `ctx` | `RequestContext` |
| `input` | `AgentInput` |

**Returns**

`AsyncIterable`\<`StreamEvent`\>

***

<a id="api-langgraphqueryplugin"></a>

### LangGraphQueryPlugin

AERQueryPlugin backed by LangGraph's MongoDBSaver.

<a id="api-constructor-3"></a>

#### Constructor

```ts
new LangGraphQueryPlugin(params): LangGraphQueryPlugin;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `params` | [`LangGraphQueryPluginParams`](#api-langgraphquerypluginparams) |

**Returns**

[`LangGraphQueryPlugin`](#api-langgraphqueryplugin)

#### Methods

<a id="api-getmessagesforsession"></a>

##### getMessagesForSession()

```ts
getMessagesForSession(sessionId): Promise<{
  messages: object[];
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `sessionId` | `string` |

**Returns**

`Promise`\<\{
  `messages`: `object`[];
\}\>

<a id="api-getsummariesforsessions"></a>

##### getSummariesForSessions()

```ts
getSummariesForSessions(sessionIds): Promise<{
  sessions: object[];
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `sessionIds` | `string`[] |

**Returns**

`Promise`\<\{
  `sessions`: `object`[];
\}\>

## Interfaces

<a id="api-createagentenginedeepagentoptions"></a>

### CreateAgentEngineDeepAgentOptions

#### Properties

| Property | Type | Description |
| :------ | :------ | :------ |
| <a id="api-property-checkpointer"></a> `checkpointer?` | `boolean` \| `BaseCheckpointSaver`\<`number`\> | LangGraph checkpointer for state persistence, HITL, and multi-turn. |
| <a id="api-property-middleware"></a> `middleware?` | readonly `AgentMiddleware`\<`any`, `any`, `any`, readonly (`ClientTool` \| `ServerTool`)[]\>[] | Additional middleware, appended after the default `StoppedToolCallMiddleware`. |
| <a id="api-property-skills"></a> `skills?` | `string`[] | Parent directories for deepagents' one-level skill discovery. |
| <a id="api-property-skillsbasedir"></a> `skillsBaseDir?` | `string` | Base directory for relative skill paths (set by `App.deepAgent()`). |
| <a id="api-property-store"></a> `store?` | `BaseStore` | LangGraph store for skills and shared data. |
| <a id="api-property-subagents"></a> `subagents?` | readonly `AnySubAgent`[] | SubAgent specs. Plain specs with a `model` field must use a model instance, not a string — string models bypass OE routing. |
| <a id="api-property-systemprompt"></a> `systemPrompt?` | `string` | Custom system instructions. |
| <a id="api-property-tools"></a> `tools?` | `StructuredTool`\<`ToolInputSchemaBase`, `any`, `any`, `any`, `unknown`\>[] | Additional tools for the deep agent. |

***

<a id="api-deepagentoptions"></a>

### DeepAgentOptions

Options for [App.deepAgent](#api-deepagent).

#### Extends

- `Omit`\<[`CreateAgentEngineDeepAgentOptions`](#api-createagentenginedeepagentoptions), `"checkpointer"` \| `"skillsBaseDir"`\>

#### Properties

| Property | Type | Description | Inherited from |
| :------ | :------ | :------ | :------ |
| <a id="api-property-backend"></a> `backend?` | `AnyBackendProtocol` | Backend for filesystem/shell ops. Defaults to `AgentEngineToolPodBackend`, so every op is OE-audited and sandboxed in the Tool Pod. Pass a custom backend (e.g. deepagents' in-memory `StateBackend`) to override — note that doing so bypasses the OE audit path. | - |
| <a id="api-property-checkpointer-1"></a> `checkpointer?` | `boolean` \| `BaseCheckpointSaver`\<`number`\> | Checkpointer selection. Omitted (`undefined`) resolves to `app.checkpointer()` (MongoDB when configured, else none). `false` disables checkpointing. A `BaseCheckpointSaver` instance is used directly. Note the mapping differs from Python (`None` disables there): in TS, "disable" is `false`, and "use the default" is simply leaving it out. | - |
| <a id="api-property-middleware-1"></a> `middleware?` | readonly `AgentMiddleware`\<`any`, `any`, `any`, readonly (`ClientTool` \| `ServerTool`)[]\>[] | Additional middleware, appended after the default `StoppedToolCallMiddleware`. | [`CreateAgentEngineDeepAgentOptions`](#api-createagentenginedeepagentoptions).[`middleware`](#api-property-middleware) |
| <a id="api-property-skills-1"></a> `skills?` | `string`[] | Parent directories for deepagents' one-level skill discovery. | [`CreateAgentEngineDeepAgentOptions`](#api-createagentenginedeepagentoptions).[`skills`](#api-property-skills) |
| <a id="api-property-store-1"></a> `store?` | `BaseStore` | LangGraph store for skills and shared data. | [`CreateAgentEngineDeepAgentOptions`](#api-createagentenginedeepagentoptions).[`store`](#api-property-store) |
| <a id="api-property-subagents-1"></a> `subagents?` | readonly `AnySubAgent`[] | SubAgent specs. Plain specs with a `model` field must use a model instance, not a string — string models bypass OE routing. | [`CreateAgentEngineDeepAgentOptions`](#api-createagentenginedeepagentoptions).[`subagents`](#api-property-subagents) |
| <a id="api-property-systemprompt-1"></a> `systemPrompt?` | `string` | Custom system instructions. | [`CreateAgentEngineDeepAgentOptions`](#api-createagentenginedeepagentoptions).[`systemPrompt`](#api-property-systemprompt) |
| <a id="api-property-tools-1"></a> `tools?` | `StructuredTool`\<`ToolInputSchemaBase`, `any`, `any`, `any`, `unknown`\>[] | Additional tools for the deep agent. | [`CreateAgentEngineDeepAgentOptions`](#api-createagentenginedeepagentoptions).[`tools`](#api-property-tools) |

***

<a id="api-langgraphquerypluginparams"></a>

### LangGraphQueryPluginParams

#### Properties

| Property | Type |
| :------ | :------ |
| <a id="api-property-client"></a> `client` | `MongoClient` |
| <a id="api-property-dbname"></a> `dbName` | `string` |
| <a id="api-property-saver"></a> `saver` | `MongoDBSaver` |
| <a id="api-property-workspaceid"></a> `workspaceId?` | `string` |
| <a id="api-property-workspaceidresolver"></a> `workspaceIdResolver?` | () => `string` \| `null` \| `undefined` |

## Type Aliases

<a id="api-prepareagentinput-1"></a>

### PrepareAgentInput

```ts
type PrepareAgentInput = (input, ctx) => unknown;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `input` | `AgentInput` |
| `ctx` | `RequestContext` |

#### Returns

`unknown`

***

<a id="api-resolvethreadid-2"></a>

### ResolveThreadId

```ts
type ResolveThreadId = (ctx) => string;
```

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `ctx` | `RequestContext` |

#### Returns

`string`

***

<a id="api-sessionfinishstatus"></a>

### SessionFinishStatus

```ts
type SessionFinishStatus = "requested" | "already_requested" | "unavailable";
```

## Variables

<a id="api-max_subagent_nesting_depth"></a>

### MAX\_SUBAGENT\_NESTING\_DEPTH

```ts
const MAX_SUBAGENT_NESTING_DEPTH: 10 = 10;
```

Recursion is bounded so an adversarial spec (or an accidental cycle) can't
blow the stack at agent-construction time.

## Functions

<a id="api-createagentenginedeepagent"></a>

### createAgentEngineDeepAgent()

```ts
function createAgentEngineDeepAgent(
   secureLlm,
   backend,
   options?
): DeepAgent<DeepAgentTypeConfig<ResponseFormatUndefined, undefined, InteropZodObject, readonly [AgentMiddleware<ZodObject<{
}, "strip", ZodTypeAny, {
}, {
}>, undefined, unknown, readonly [DynamicStructuredTool<ZodObject<{
}, "strip", ZodTypeAny, {
}, {
}>, {
}, {
}, Command<unknown, {
}, string>, unknown, "write_todos">]>, AgentMiddleware<StateSchema<{
}>, undefined, unknown, (
  | DynamicStructuredTool<ZodObject<{
}, $strip>, {
}, {
}, string, unknown, "ls">
  | DynamicStructuredTool<ZodObject<{
}, $strip>, {
}, {
}, object[] | object[], unknown, "read_file">
  | DynamicStructuredTool<ZodObject<{
}, $strip>, {
}, {
},
  | string
  | ToolMessage<MessageStructure<MessageToolSet>>
  | Command<unknown, {
}, string>, unknown, "write_file">
  | DynamicStructuredTool<ZodObject<{
}, $strip>, {
}, {
},
  | string
  | ToolMessage<MessageStructure<MessageToolSet>>
  | Command<unknown, {
}, string>, unknown, "edit_file">
  | DynamicStructuredTool<ZodObject<{
}, $strip>, {
}, {
}, string, unknown, "glob">
  | DynamicStructuredTool<ZodObject<{
}, $strip>, {
}, {
}, string, unknown, "grep">
  | DynamicStructuredTool<ZodObject<{
}, $strip>, {
}, {
}, string, unknown, "execute">)[]>, AgentMiddleware<undefined, undefined, unknown, readonly [DynamicStructuredTool<ZodObject<{
}, $strip>, {
}, {
},
  | string
  | Command<unknown, Record<string, unknown>, string>, unknown, "task">]>, AgentMiddleware<ZodObject<{
}, $strip>, undefined, unknown, readonly (ClientTool | ServerTool)[]>, AgentMiddleware<undefined, undefined, unknown, readonly (ClientTool | ServerTool)[]>, AgentMiddleware<any, any, any, readonly (ClientTool | ServerTool)[]>], readonly (ClientTool | ServerTool)[], readonly AnySubAgent[], readonly () => StreamTransformer<any>[]>>;
```

Create a deep-agent graph pre-configured for Atlas Agent Engine AER. Resolves relative
skill paths, validates the subagent tree, then delegates to deepagents'
`createDeepAgent`.

#### Parameters

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `secureLlm` | `BaseChatModel` | A `SecureWrappedLLM` instance — every LLM call is OE-audited. |
| `backend` | `AnyBackendProtocol` \| `undefined` | Backend for filesystem/shell ops; resolved by `App.deepAgent()`. |
| `options` | [`CreateAgentEngineDeepAgentOptions`](#api-createagentenginedeepagentoptions) | - |

#### Returns

`DeepAgent`\<`DeepAgentTypeConfig`\<`ResponseFormatUndefined`, `undefined`, `InteropZodObject`, readonly \[`AgentMiddleware`\<`ZodObject`\<\{
\}, `"strip"`, `ZodTypeAny`, \{
\}, \{
\}\>, `undefined`, `unknown`, readonly \[`DynamicStructuredTool`\<`ZodObject`\<\{
\}, `"strip"`, `ZodTypeAny`, \{
\}, \{
\}\>, \{
\}, \{
\}, `Command`\<`unknown`, \{
\}, `string`\>, `unknown`, `"write_todos"`\>\]\>, `AgentMiddleware`\<`StateSchema`\<\{
\}\>, `undefined`, `unknown`, (
  \| `DynamicStructuredTool`\<`ZodObject`\<\{
\}, `$strip`\>, \{
\}, \{
\}, `string`, `unknown`, `"ls"`\>
  \| `DynamicStructuredTool`\<`ZodObject`\<\{
\}, `$strip`\>, \{
\}, \{
\}, `object`[] \| `object`[], `unknown`, `"read_file"`\>
  \| `DynamicStructuredTool`\<`ZodObject`\<\{
\}, `$strip`\>, \{
\}, \{
\},
  \| `string`
  \| `ToolMessage`\<`MessageStructure`\<`MessageToolSet`\>\>
  \| `Command`\<`unknown`, \{
\}, `string`\>, `unknown`, `"write_file"`\>
  \| `DynamicStructuredTool`\<`ZodObject`\<\{
\}, `$strip`\>, \{
\}, \{
\},
  \| `string`
  \| `ToolMessage`\<`MessageStructure`\<`MessageToolSet`\>\>
  \| `Command`\<`unknown`, \{
\}, `string`\>, `unknown`, `"edit_file"`\>
  \| `DynamicStructuredTool`\<`ZodObject`\<\{
\}, `$strip`\>, \{
\}, \{
\}, `string`, `unknown`, `"glob"`\>
  \| `DynamicStructuredTool`\<`ZodObject`\<\{
\}, `$strip`\>, \{
\}, \{
\}, `string`, `unknown`, `"grep"`\>
  \| `DynamicStructuredTool`\<`ZodObject`\<\{
\}, `$strip`\>, \{
\}, \{
\}, `string`, `unknown`, `"execute"`\>)[]\>, `AgentMiddleware`\<`undefined`, `undefined`, `unknown`, readonly \[`DynamicStructuredTool`\<`ZodObject`\<\{
\}, `$strip`\>, \{
\}, \{
\},
  \| `string`
  \| `Command`\<`unknown`, `Record`\<`string`, `unknown`\>, `string`\>, `unknown`, `"task"`\>\]\>, `AgentMiddleware`\<`ZodObject`\<\{
\}, `$strip`\>, `undefined`, `unknown`, readonly (`ClientTool` \| `ServerTool`)[]\>, `AgentMiddleware`\<`undefined`, `undefined`, `unknown`, readonly (`ClientTool` \| `ServerTool`)[]\>, `AgentMiddleware`\<`any`, `any`, `any`, readonly (`ClientTool` \| `ServerTool`)[]\>\], readonly (`ClientTool` \| `ServerTool`)[], readonly `AnySubAgent`[], readonly () => `StreamTransformer`\<`any`\>[]\>\>

#### Throws

A subagent spec uses a string model, or nesting exceeds the cap.

***

<a id="api-validatesubagenttree"></a>

### validateSubagentTree()

```ts
function validateSubagentTree(subagents): void;
```

Walk the subagent tree, throwing on any string-model spec.

String model specs would bypass OE routing because deepagents' `resolveModel`
instantiates a raw provider client with no `SecureToolWrapper`. Only model
instances (e.g. `SecureWrappedLLM`) are accepted.

Spec kinds handled:
- `CompiledSubAgent` (`runnable` set) — pre-built graph; the model is already
  bound and unreadable here. Skipped.
- `AsyncSubAgent` (`graphId` set) — remote graph we can't validate in-process.
  Logged at WARNING and skipped — the receiving end must route through the OE.
- Plain `SubAgent` — validated; a string `model` throws.

The depth counter is held in a closure rather than a public parameter so
callers cannot start recursion mid-tree and bypass the cap.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `subagents` | readonly `AnySubAgent`[] \| `undefined` |

#### Returns

`void`

#### Throws

A subagent spec uses a string `model`, or nesting exceeds
  [MAX\_SUBAGENT\_NESTING\_DEPTH](#api-max_subagent_nesting_depth).

## References

<a id="api-agentenginetoolsandboxbackend"></a>

### AgentEngineToolSandboxBackend

Renames and re-exports [AgentEngineToolPodBackend](#api-agentenginetoolpodbackend)
