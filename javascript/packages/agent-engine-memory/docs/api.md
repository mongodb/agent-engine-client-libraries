# @mongodb-js/agent-engine-sdk-memory

## Table of Contents

- **Classes**
  - [Memory](#api-memory)
    - [bind()](#api-bind)
    - [buildContext()](#api-buildcontext)
    - [close()](#api-close)
    - [discoverProcedures()](#api-discoverprocedures)
    - [getProcedure()](#api-getprocedure)
    - [getSemantic()](#api-getsemantic)
    - [getTaxonomicTerm()](#api-gettaxonomicterm)
    - [listDomains()](#api-listdomains)
    - [listEpisodes()](#api-listepisodes)
    - [recordTurn()](#api-recordturn)
    - [retrieve()](#api-retrieve)
    - [save()](#api-save)
    - [saveEpisode()](#api-saveepisode)
    - [saveProcedure()](#api-saveprocedure)
    - [saveSemantic()](#api-savesemantic)
    - [saveTaxonomic()](#api-savetaxonomic)
    - [search()](#api-search)
    - [searchEpisodes()](#api-searchepisodes)
    - [searchSemantic()](#api-searchsemantic)
    - [searchTaxonomic()](#api-searchtaxonomic)
  - [MemoryAPIError](#api-memoryapierror)
  - [MemoryAuthError](#api-memoryautherror)
  - [MemoryBadRequestError](#api-memorybadrequesterror)
  - [MemoryClientAdapter](#api-memoryclientadapter)
    - [buildContext()](#api-buildcontext-1)
    - [close()](#api-close-1)
    - [createCustom()](#api-createcustom)
    - [createEpisodic()](#api-createepisodic)
    - [createProcedural()](#api-createprocedural)
    - [createSemantic()](#api-createsemantic)
    - [createTaxonomic()](#api-createtaxonomic)
    - [discoverProcedures()](#api-discoverprocedures-1)
    - [getDistinctDomains()](#api-getdistinctdomains)
    - [getProcedural()](#api-getprocedural)
    - [getSemantic()](#api-getsemantic-1)
    - [getTaxonomic()](#api-gettaxonomic)
    - [listEpisodic()](#api-listepisodic)
    - [recordTurn()](#api-recordturn-1)
    - [retrieveCustom()](#api-retrievecustom)
    - [searchEpisodes()](#api-searchepisodes-1)
    - [searchSemantic()](#api-searchsemantic-1)
    - [searchTaxonomic()](#api-searchtaxonomic-1)
  - [MemoryClientError](#api-memoryclienterror)
  - [MemoryConnectionError](#api-memoryconnectionerror)
  - [MemoryIdentityError](#api-memoryidentityerror)
  - [MemoryNotProvisionedError](#api-memorynotprovisionederror)
  - [MemoryNotSupportedError](#api-memorynotsupportederror)
  - [MemoryRouteNotFoundError](#api-memoryroutenotfounderror)
  - [MemoryServerError](#api-memoryservererror)
- **Interfaces**
  - [AmbientIdentityRuntime](#api-ambientidentityruntime)
    - [requestContext()](#api-requestcontext)
  - [MemoryClientAdapterOptions](#api-memoryclientadapteroptions)
  - [MemoryCrudClient](#api-memorycrudclient)
    - [close()?](#api-close-2)
    - [createCustom()](#api-createcustom-1)
    - [createEpisodic()](#api-createepisodic-1)
    - [createProcedural()](#api-createprocedural-1)
    - [createSemantic()](#api-createsemantic-1)
    - [createTaxonomic()](#api-createtaxonomic-1)
    - [getDistinctDomains()](#api-getdistinctdomains-1)
    - [getProcedural()](#api-getprocedural-1)
    - [getSemantic()](#api-getsemantic-2)
    - [getTaxonomic()](#api-gettaxonomic-1)
    - [listEpisodic()](#api-listepisodic-1)
    - [retrieveCustom()](#api-retrievecustom-1)
  - [MemoryRequestContext](#api-memoryrequestcontext)
  - [MemoryRuntime](#api-memoryruntime)
    - [buildContext()](#api-buildcontext-2)
    - [close()?](#api-close-3)
    - [discoverProcedures()](#api-discoverprocedures-2)
    - [recordTurn()](#api-recordturn-2)
    - [searchEpisodes()](#api-searchepisodes-2)
    - [searchSemantic()](#api-searchsemantic-2)
    - [searchTaxonomic()](#api-searchtaxonomic-2)
- **Type Aliases**
  - [ContextMetadata](#api-contextmetadata)
  - [ContextResponse](#api-contextresponse)
  - [CreateEpisodicResult](#api-createepisodicresult)
  - [CreateProceduralResult](#api-createproceduralresult)
  - [CreateSemanticResult](#api-createsemanticresult)
  - [CreateTaxonomicResult](#api-createtaxonomicresult)
  - [CustomMemoryRetrieveResult](#api-custommemoryretrieveresult)
  - [CustomMemorySaveResult](#api-custommemorysaveresult)
  - [FetchLike](#api-fetchlike)
  - [JsonValue](#api-jsonvalue)
  - [MemoryChunk](#api-memorychunk)
  - [MemoryOptions](#api-memoryoptions)
  - [MemorySource](#api-memorysource)
  - [RequestExtras](#api-requestextras)
  - [ResolvedIdentity](#api-resolvedidentity)
  - [RetrievedCustomMemory](#api-retrievedcustommemory)
  - [SearchSource](#api-searchsource)
  - [TagMap](#api-tagmap)
  - [TagScalar](#api-tagscalar)
  - [WriteTurnResult](#api-writeturnresult)
- **Variables**
  - [ContextMetadataSchema](#api-contextmetadataschema)
  - [ContextResponseSchema](#api-contextresponseschema)
  - [CreateEpisodicResultSchema](#api-createepisodicresultschema)
  - [CreateProceduralResultSchema](#api-createproceduralresultschema)
  - [CreateSemanticResultSchema](#api-createsemanticresultschema)
  - [CreateTaxonomicResultSchema](#api-createtaxonomicresultschema)
  - [CustomMemoryRetrieveResultSchema](#api-custommemoryretrieveresultschema)
  - [CustomMemorySaveResultSchema](#api-custommemorysaveresultschema)
  - [MemoryChunkSchema](#api-memorychunkschema)
  - [MemorySource](#api-memorysource-1)
  - [RetrievedCustomMemorySchema](#api-retrievedcustommemoryschema)
  - [SearchSource](#api-searchsource-1)
  - [WriteTurnResultSchema](#api-writeturnresultschema)
- **Functions**
  - [hasAmbientIdentity()](#api-hasambientidentity)
  - [resolveIdentity()](#api-resolveidentity)
  - [toSearchSource()](#api-tosearchsource)

## Classes

<a id="api-memory"></a>

### Memory

Public, transport-free memory facade.

<a id="api-constructor"></a>

#### Constructor

```ts
new Memory(opts?): Memory;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `opts` | [`MemoryOptions`](#api-memoryoptions) |

**Returns**

[`Memory`](#api-memory)

#### Methods

<a id="api-bind"></a>

##### bind()

```ts
bind(ctx): Memory;
```

Return a new Memory scoped to `ctx` without mutating this handle.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `ctx` | [`MemoryRequestContext`](#api-memoryrequestcontext) |

**Returns**

[`Memory`](#api-memory)

<a id="api-buildcontext"></a>

##### buildContext()

```ts
buildContext(args): Promise<{
  formatted_context: string | Record<string, unknown>[];
  metadata: {
     memory_counts: Record<string, number>;
     timing: Record<string, number>;
     token_count: number;
  };
  selected_memories?: object[] | null;
}>;
```

Build a unified memory context across memory types.

**Parameters**

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `args` | \{ `enabledSources?`: `Set`\<`string`\> \| `null`; `maxTokens?`: `number`; `metadataFilter?`: `Record`\<`string`, `unknown`\> \| `null`; `query`: `string`; `sessionId?`: `string` \| `null`; `threadId?`: `string` \| `null`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} | - |
| `args.enabledSources?` | `Set`\<`string`\> \| `null` | - |
| `args.maxTokens?` | `number` | Optional gross context-construction budget. Omitted preserves prior behavior. Non-positive, non-integer, and non-finite values raise before runtime delegation. After retrieval and ranking, the server subtracts a 500-token formatting reserve, then greedily selects whole memory chunks that fit in the remainder. Positive values at or below 500 leave no budget for memories. Values above 500 can still yield empty context when no chunk fits. `metadata.token_count` reports formatted output only and excludes the reserve. |
| `args.metadataFilter?` | `Record`\<`string`, `unknown`\> \| `null` | - |
| `args.query` | `string` | - |
| `args.sessionId?` | `string` \| `null` | - |
| `args.threadId?` | `string` \| `null` | Deprecated alias for `sessionId`; a non-blank `sessionId` wins. |
| `args.userId?` | `string` \| `null` | - |
| `args.visibility?` | `string` \| `null` | - |

**Returns**

`Promise`\<\{
  `formatted_context`: `string` \| `Record`\<`string`, `unknown`\>[];
  `metadata`: \{
     `memory_counts`: `Record`\<`string`, `number`\>;
     `timing`: `Record`\<`string`, `number`\>;
     `token_count`: `number`;
  \};
  `selected_memories?`: `object`[] \| `null`;
\}\>

<a id="api-close"></a>

##### close()

```ts
close(): Promise<void>;
```

Release the underlying transport(s) when this handle owns them.

**Returns**

`Promise`\<`void`\>

<a id="api-discoverprocedures"></a>

##### discoverProcedures()

```ts
discoverProcedures(args): Promise<Record<string, unknown>[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `metadataFilter?`: `Record`\<`string`, `unknown`\> \| `null`; `query`: `string`; `similarityThreshold?`: `number`; `tags?`: `string`[] \| `null`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.metadataFilter?` | `Record`\<`string`, `unknown`\> \| `null` |
| `args.query` | `string` |
| `args.similarityThreshold?` | `number` |
| `args.tags?` | `string`[] \| `null` |
| `args.topK?` | `number` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`Record`\<`string`, `unknown`\>[]\>

<a id="api-getprocedure"></a>

##### getProcedure()

```ts
getProcedure(args): Promise<unknown>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `includeDeleted?`: `boolean`; `procedureName`: `string`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.includeDeleted?` | `boolean` |
| `args.procedureName` | `string` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`\>

<a id="api-getsemantic"></a>

##### getSemantic()

```ts
getSemantic(args): Promise<unknown>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `label`: `string`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.label` | `string` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`\>

<a id="api-gettaxonomicterm"></a>

##### getTaxonomicTerm()

```ts
getTaxonomicTerm(args): Promise<unknown>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `domain`: `string`; `term`: `string`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.domain` | `string` |
| `args.term` | `string` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`\>

<a id="api-listdomains"></a>

##### listDomains()

```ts
listDomains(args?): Promise<string[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `visibility?`: `string` \| `null`; \} |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`string`[]\>

<a id="api-listepisodes"></a>

##### listEpisodes()

```ts
listEpisodes(args?): Promise<unknown[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `limit?`: `number`; `sessionId?`: `string` \| `null`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.limit?` | `number` |
| `args.sessionId?` | `string` \| `null` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`[]\>

<a id="api-recordturn"></a>

##### recordTurn()

```ts
recordTurn(args): Promise<{
  acknowledged: boolean;
  has_embedding: boolean;
  id: string;
  session_id: string;
  turn_seq: number;
}>;
```

Record a single conversation turn (write-accepted semantics).

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `content?`: `string` \| `null`; `idempotencyKey?`: `string` \| `null`; `isError?`: `boolean`; `modelName?`: `string` \| `null`; `role`: `string`; `sessionId?`: `string` \| `null`; `toolCallId?`: `string` \| `null`; `toolCalls?`: `Record`\<`string`, `unknown`\>[] \| `null`; `toolName?`: `string` \| `null`; \} |
| `args.content?` | `string` \| `null` |
| `args.idempotencyKey?` | `string` \| `null` |
| `args.isError?` | `boolean` |
| `args.modelName?` | `string` \| `null` |
| `args.role` | `string` |
| `args.sessionId?` | `string` \| `null` |
| `args.toolCallId?` | `string` \| `null` |
| `args.toolCalls?` | `Record`\<`string`, `unknown`\>[] \| `null` |
| `args.toolName?` | `string` \| `null` |

**Returns**

`Promise`\<\{
  `acknowledged`: `boolean`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `session_id`: `string`;
  `turn_seq`: `number`;
\}\>

<a id="api-retrieve"></a>

##### retrieve()

```ts
retrieve(
   memoryType,
   query,
   options?
): Promise<{
  count: number;
  results: object[];
}>;
```

Retrieve memories of a declared custom type by semantic query.

Filters are exact-match equality on declared tag keys; result ordering
may improve between releases and is not contractual. One type per call.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `memoryType` | `string` |
| `query` | `string` |
| `options?` | \{ `tags?`: [`TagMap`](#api-tagmap) \| `null`; `topK?`: `number`; \} |
| `options.tags?` | [`TagMap`](#api-tagmap) \| `null` |
| `options.topK?` | `number` |

**Returns**

`Promise`\<\{
  `count`: `number`;
  `results`: `object`[];
\}\>

<a id="api-save"></a>

##### save()

```ts
save(
   memoryType,
   content,
   options?
): Promise<{
  has_embedding: boolean;
  id: string;
  tags: Record<string, string | number | boolean>;
  type: string;
}>;
```

Save a memory of a declared custom type.

The platform stamps identity (org, project, user) and enforces the
type's declared tag schema; this method validates only tag syntax
client-side so mistakes fail fast with server-matching messages.
Built-in types (semantic, episodic, taxonomic, procedural) are
rejected — use their dedicated methods.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `memoryType` | `string` |
| `content` | `string` |
| `options?` | \{ `contextualMetadata?`: `Record`\<`string`, `unknown`\> \| `null`; `tags?`: [`TagMap`](#api-tagmap) \| `null`; \} |
| `options.contextualMetadata?` | `Record`\<`string`, `unknown`\> \| `null` |
| `options.tags?` | [`TagMap`](#api-tagmap) \| `null` |

**Returns**

`Promise`\<\{
  `has_embedding`: `boolean`;
  `id`: `string`;
  `tags`: `Record`\<`string`, `string` \| `number` \| `boolean`\>;
  `type`: `string`;
\}\>

<a id="api-saveepisode"></a>

##### saveEpisode()

```ts
saveEpisode(args): Promise<{
  acknowledged: boolean;
  has_embedding: boolean;
  id: string;
  title: string;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `agentId?`: `string` \| `null`; `content`: `string`; `metadata?`: `Record`\<`string`, `unknown`\> \| `null`; `participants?`: `string`[] \| `null`; `sessionId?`: `string` \| `null`; `summary?`: `string` \| `null`; `tags?`: `string`[] \| `null`; `title`: `string`; `userId?`: `string` \| `null`; `visibility?`: `string`; \} |
| `args.agentId?` | `string` \| `null` |
| `args.content` | `string` |
| `args.metadata?` | `Record`\<`string`, `unknown`\> \| `null` |
| `args.participants?` | `string`[] \| `null` |
| `args.sessionId?` | `string` \| `null` |
| `args.summary?` | `string` \| `null` |
| `args.tags?` | `string`[] \| `null` |
| `args.title` | `string` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` |

**Returns**

`Promise`\<\{
  `acknowledged`: `boolean`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `title`: `string`;
\}\>

<a id="api-saveprocedure"></a>

##### saveProcedure()

```ts
saveProcedure(args): Promise<{
  acknowledged: boolean;
  has_embedding: boolean;
  id: string;
  procedure: string;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `agentId?`: `string` \| `null`; `allowedTools?`: `string`[] \| `null`; `compatibility?`: `string` \| `null`; `content`: `string`; `description`: `string`; `extractionSource?`: `string` \| `null`; `license?`: `string` \| `null`; `procedure`: `string`; `resources?`: `Record`\<`string`, `unknown`\>[] \| `null`; `sourceFormat?`: `string` \| `null`; `sourcePath?`: `string` \| `null`; `steps?`: `Record`\<`string`, `unknown`\>[] \| `null`; `tags?`: `string`[] \| `null`; `triggerConditions?`: `string`[] \| `null`; `updateExisting?`: `boolean`; `userId?`: `string` \| `null`; `visibility?`: `string`; \} |
| `args.agentId?` | `string` \| `null` |
| `args.allowedTools?` | `string`[] \| `null` |
| `args.compatibility?` | `string` \| `null` |
| `args.content` | `string` |
| `args.description` | `string` |
| `args.extractionSource?` | `string` \| `null` |
| `args.license?` | `string` \| `null` |
| `args.procedure` | `string` |
| `args.resources?` | `Record`\<`string`, `unknown`\>[] \| `null` |
| `args.sourceFormat?` | `string` \| `null` |
| `args.sourcePath?` | `string` \| `null` |
| `args.steps?` | `Record`\<`string`, `unknown`\>[] \| `null` |
| `args.tags?` | `string`[] \| `null` |
| `args.triggerConditions?` | `string`[] \| `null` |
| `args.updateExisting?` | `boolean` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` |

**Returns**

`Promise`\<\{
  `acknowledged`: `boolean`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `procedure`: `string`;
\}\>

<a id="api-savesemantic"></a>

##### saveSemantic()

```ts
saveSemantic(args): Promise<{
  acknowledged: boolean;
  has_embedding: boolean;
  id: string;
  label: string;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `agentId?`: `string` \| `null`; `label`: `string`; `metadata?`: `Record`\<`string`, `unknown`\> \| `null`; `source?`: `string`; `text`: `string`; `upsert?`: `boolean`; `userId?`: `string` \| `null`; `visibility?`: `string`; \} |
| `args.agentId?` | `string` \| `null` |
| `args.label` | `string` |
| `args.metadata?` | `Record`\<`string`, `unknown`\> \| `null` |
| `args.source?` | `string` |
| `args.text` | `string` |
| `args.upsert?` | `boolean` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` |

**Returns**

`Promise`\<\{
  `acknowledged`: `boolean`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `label`: `string`;
\}\>

<a id="api-savetaxonomic"></a>

##### saveTaxonomic()

```ts
saveTaxonomic(args): Promise<{
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
| `args` | \{ `definition`: `string`; `domain`: `string`; `relatedTerms?`: `string`[] \| `null`; `term`: `string`; `userId?`: `string` \| `null`; `visibility?`: `string`; \} |
| `args.definition` | `string` |
| `args.domain` | `string` |
| `args.relatedTerms?` | `string`[] \| `null` |
| `args.term` | `string` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` |

**Returns**

`Promise`\<\{
  `acknowledged`: `boolean`;
  `domain`: `string`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `term`: `string`;
\}\>

<a id="api-search"></a>

##### search()

```ts
search(args): Promise<object[]>;
```

Search one or more memory sources and return a single ranked list.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `domain?`: `string` \| `null`; `metadataFilter?`: `Record`\<`string`, `unknown`\> \| `null`; `query`: `string`; `sessionId?`: `string` \| `null`; `similarityThreshold?`: `number`; `sources?`: `string` \| readonly `string`[] \| `null`; `tags?`: `string`[] \| `null`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.domain?` | `string` \| `null` |
| `args.metadataFilter?` | `Record`\<`string`, `unknown`\> \| `null` |
| `args.query` | `string` |
| `args.sessionId?` | `string` \| `null` |
| `args.similarityThreshold?` | `number` |
| `args.sources?` | `string` \| readonly `string`[] \| `null` |
| `args.tags?` | `string`[] \| `null` |
| `args.topK?` | `number` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`object`[]\>

<a id="api-searchepisodes"></a>

##### searchEpisodes()

```ts
searchEpisodes(args): Promise<object[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `query`: `string`; `sessionId?`: `string` \| `null`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.query` | `string` |
| `args.sessionId?` | `string` \| `null` |
| `args.topK?` | `number` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`object`[]\>

<a id="api-searchsemantic"></a>

##### searchSemantic()

```ts
searchSemantic(args): Promise<object[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `query`: `string`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.query` | `string` |
| `args.topK?` | `number` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`object`[]\>

<a id="api-searchtaxonomic"></a>

##### searchTaxonomic()

```ts
searchTaxonomic(args): Promise<object[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `domain?`: `string` \| `null`; `query`: `string`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.domain?` | `string` \| `null` |
| `args.query` | `string` |
| `args.topK?` | `number` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`object`[]\>

***

<a id="api-memoryapierror"></a>

### MemoryAPIError

Base for typed transport errors from the HTTP memory transports.

#### Extends

- `Error`

#### Extended by

- [`MemoryAuthError`](#api-memoryautherror)
- [`MemoryNotProvisionedError`](#api-memorynotprovisionederror)
- [`MemoryBadRequestError`](#api-memorybadrequesterror)
- [`MemoryServerError`](#api-memoryservererror)
- [`MemoryConnectionError`](#api-memoryconnectionerror)

<a id="api-constructor-1"></a>

#### Constructor

```ts
new MemoryAPIError(message, opts?): MemoryAPIError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `message` | `string` |
| `opts` | \{ `code?`: `string` \| `null`; `responseText?`: `string` \| `null`; `status?`: `number` \| `null`; \} |
| `opts.code?` | `string` \| `null` |
| `opts.responseText?` | `string` \| `null` |
| `opts.status?` | `number` \| `null` |

**Returns**

[`MemoryAPIError`](#api-memoryapierror)

**Overrides**

```ts
Error.constructor
```

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-code"></a> `code` | `readonly` | `string` \| `null` |
| <a id="api-property-responsetext"></a> `responseText` | `readonly` | `string` \| `null` |
| <a id="api-property-status"></a> `status` | `readonly` | `number` \| `null` |

***

<a id="api-memoryautherror"></a>

### MemoryAuthError

Raised for 401 / 403 responses.

#### Extends

- [`MemoryAPIError`](#api-memoryapierror)

<a id="api-constructor-2"></a>

#### Constructor

```ts
new MemoryAuthError(message, opts?): MemoryAuthError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `message` | `string` |
| `opts?` | \{ `code?`: `string` \| `null`; `responseText?`: `string` \| `null`; `status?`: `number` \| `null`; \} |
| `opts.code?` | `string` \| `null` |
| `opts.responseText?` | `string` \| `null` |
| `opts.status?` | `number` \| `null` |

**Returns**

[`MemoryAuthError`](#api-memoryautherror)

**Overrides**

[`MemoryAPIError`](#api-memoryapierror).[`constructor`](#api-constructor-1)

#### Properties

| Property | Modifier | Type | Inherited from |
| :------ | :------ | :------ | :------ |
| <a id="api-property-code-1"></a> `code` | `readonly` | `string` \| `null` | [`MemoryAPIError`](#api-memoryapierror).[`code`](#api-property-code) |
| <a id="api-property-responsetext-1"></a> `responseText` | `readonly` | `string` \| `null` | [`MemoryAPIError`](#api-memoryapierror).[`responseText`](#api-property-responsetext) |
| <a id="api-property-status-1"></a> `status` | `readonly` | `number` \| `null` | [`MemoryAPIError`](#api-memoryapierror).[`status`](#api-property-status) |

***

<a id="api-memorybadrequesterror"></a>

### MemoryBadRequestError

Raised for 4xx responses other than auth / not-provisioned.

#### Extends

- [`MemoryAPIError`](#api-memoryapierror)

#### Extended by

- [`MemoryRouteNotFoundError`](#api-memoryroutenotfounderror)

<a id="api-constructor-3"></a>

#### Constructor

```ts
new MemoryBadRequestError(message, opts?): MemoryBadRequestError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `message` | `string` |
| `opts?` | \{ `code?`: `string` \| `null`; `responseText?`: `string` \| `null`; `status?`: `number` \| `null`; \} |
| `opts.code?` | `string` \| `null` |
| `opts.responseText?` | `string` \| `null` |
| `opts.status?` | `number` \| `null` |

**Returns**

[`MemoryBadRequestError`](#api-memorybadrequesterror)

**Overrides**

[`MemoryAPIError`](#api-memoryapierror).[`constructor`](#api-constructor-1)

#### Properties

| Property | Modifier | Type | Inherited from |
| :------ | :------ | :------ | :------ |
| <a id="api-property-code-2"></a> `code` | `readonly` | `string` \| `null` | [`MemoryAPIError`](#api-memoryapierror).[`code`](#api-property-code) |
| <a id="api-property-responsetext-2"></a> `responseText` | `readonly` | `string` \| `null` | [`MemoryAPIError`](#api-memoryapierror).[`responseText`](#api-property-responsetext) |
| <a id="api-property-status-2"></a> `status` | `readonly` | `number` \| `null` | [`MemoryAPIError`](#api-memoryapierror).[`status`](#api-property-status) |

***

<a id="api-memoryclientadapter"></a>

### MemoryClientAdapter

A single `MemoryClient`-backed runtime + CRUD client for the `Memory` facade.

#### Implements

- [`MemoryRuntime`](#api-memoryruntime)
- [`MemoryCrudClient`](#api-memorycrudclient)

<a id="api-constructor-4"></a>

#### Constructor

```ts
new MemoryClientAdapter(opts): MemoryClientAdapter;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `opts` | [`MemoryClientAdapterOptions`](#api-memoryclientadapteroptions) |

**Returns**

[`MemoryClientAdapter`](#api-memoryclientadapter)

#### Methods

<a id="api-buildcontext-1"></a>

##### buildContext()

```ts
buildContext(args): Promise<{
  formatted_context: string | Record<string, unknown>[];
  metadata: {
     memory_counts: Record<string, number>;
     timing: Record<string, number>;
     token_count: number;
  };
  selected_memories?: object[] | null;
}>;
```

**Parameters**

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `args` | \{ `enabledSources?`: `Set`\<`string`\> \| `null`; `maxTokens?`: `number`; `metadataFilter?`: `Record`\<`string`, `unknown`\> \| `null`; `query`: `string`; `sessionId?`: `string` \| `null`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} | - |
| `args.enabledSources?` | `Set`\<`string`\> \| `null` | - |
| `args.maxTokens?` | `number` | Optional gross context-construction budget. After retrieval and ranking, the server subtracts a 500-token formatting reserve, then greedily selects whole memory chunks that fit in the remainder. Positive values at or below 500 leave no budget for memories. Values above 500 can still yield empty context when no chunk fits. |
| `args.metadataFilter?` | `Record`\<`string`, `unknown`\> \| `null` | - |
| `args.query` | `string` | - |
| `args.sessionId?` | `string` \| `null` | - |
| `args.topK?` | `number` | - |
| `args.userId?` | `string` \| `null` | - |
| `args.visibility?` | `string` \| `null` | - |

**Returns**

`Promise`\<\{
  `formatted_context`: `string` \| `Record`\<`string`, `unknown`\>[];
  `metadata`: \{
     `memory_counts`: `Record`\<`string`, `number`\>;
     `timing`: `Record`\<`string`, `number`\>;
     `token_count`: `number`;
  \};
  `selected_memories?`: `object`[] \| `null`;
\}\>

**Implementation of**

[`MemoryRuntime`](#api-memoryruntime).[`buildContext`](#api-buildcontext-2)

<a id="api-close-1"></a>

##### close()

```ts
close(): void;
```

Optional: release underlying resources.

**Returns**

`void`

**Implementation of**

[`MemoryCrudClient`](#api-memorycrudclient).[`close`](#api-close-2)

<a id="api-createcustom"></a>

##### createCustom()

```ts
createCustom(args): Promise<{
  has_embedding: boolean;
  id: string;
  tags: Record<string, string | number | boolean>;
  type: string;
}>;
```

Identity is stamped by the platform.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `content`: `string`; `contextualMetadata?`: `Record`\<`string`, `unknown`\> \| `null`; `memoryType`: `string`; `tags?`: `Record`\<`string`, `unknown`\> \| `null`; \} |
| `args.content` | `string` |
| `args.contextualMetadata?` | `Record`\<`string`, `unknown`\> \| `null` |
| `args.memoryType` | `string` |
| `args.tags?` | `Record`\<`string`, `unknown`\> \| `null` |

**Returns**

`Promise`\<\{
  `has_embedding`: `boolean`;
  `id`: `string`;
  `tags`: `Record`\<`string`, `string` \| `number` \| `boolean`\>;
  `type`: `string`;
\}\>

**Implementation of**

[`MemoryCrudClient`](#api-memorycrudclient).[`createCustom`](#api-createcustom-1)

<a id="api-createepisodic"></a>

##### createEpisodic()

```ts
createEpisodic(args): Promise<{
  acknowledged: boolean;
  has_embedding: boolean;
  id: string;
  title: string;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `agentId?`: `string` \| `null`; `content`: `string`; `metadata?`: `Record`\<`string`, `unknown`\> \| `null`; `participants?`: `string`[] \| `null`; `sessionId`: `string`; `summaryText?`: `string` \| `null`; `tags?`: `string`[] \| `null`; `title`: `string`; `userId`: `string`; `visibility?`: `string`; \} |
| `args.agentId?` | `string` \| `null` |
| `args.content` | `string` |
| `args.metadata?` | `Record`\<`string`, `unknown`\> \| `null` |
| `args.participants?` | `string`[] \| `null` |
| `args.sessionId` | `string` |
| `args.summaryText?` | `string` \| `null` |
| `args.tags?` | `string`[] \| `null` |
| `args.title` | `string` |
| `args.userId` | `string` |
| `args.visibility?` | `string` |

**Returns**

`Promise`\<\{
  `acknowledged`: `boolean`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `title`: `string`;
\}\>

**Implementation of**

[`MemoryCrudClient`](#api-memorycrudclient).[`createEpisodic`](#api-createepisodic-1)

<a id="api-createprocedural"></a>

##### createProcedural()

```ts
createProcedural(args): Promise<{
  acknowledged: boolean;
  has_embedding: boolean;
  id: string;
  procedure: string;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `agentId?`: `string` \| `null`; `allowedTools?`: `string`[] \| `null`; `compatibility?`: `string` \| `null`; `content`: `string`; `description`: `string`; `extractionSource?`: `string` \| `null`; `license?`: `string` \| `null`; `procedure`: `string`; `resources?`: `Record`\<`string`, `unknown`\>[] \| `null`; `sourceFormat?`: `string` \| `null`; `sourcePath?`: `string` \| `null`; `steps?`: `Record`\<`string`, `unknown`\>[] \| `null`; `tags?`: `string`[] \| `null`; `triggerConditions?`: `string`[] \| `null`; `updateExisting?`: `boolean`; `userId`: `string`; `visibility?`: `string`; \} |
| `args.agentId?` | `string` \| `null` |
| `args.allowedTools?` | `string`[] \| `null` |
| `args.compatibility?` | `string` \| `null` |
| `args.content` | `string` |
| `args.description` | `string` |
| `args.extractionSource?` | `string` \| `null` |
| `args.license?` | `string` \| `null` |
| `args.procedure` | `string` |
| `args.resources?` | `Record`\<`string`, `unknown`\>[] \| `null` |
| `args.sourceFormat?` | `string` \| `null` |
| `args.sourcePath?` | `string` \| `null` |
| `args.steps?` | `Record`\<`string`, `unknown`\>[] \| `null` |
| `args.tags?` | `string`[] \| `null` |
| `args.triggerConditions?` | `string`[] \| `null` |
| `args.updateExisting?` | `boolean` |
| `args.userId` | `string` |
| `args.visibility?` | `string` |

**Returns**

`Promise`\<\{
  `acknowledged`: `boolean`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `procedure`: `string`;
\}\>

**Implementation of**

[`MemoryCrudClient`](#api-memorycrudclient).[`createProcedural`](#api-createprocedural-1)

<a id="api-createsemantic"></a>

##### createSemantic()

```ts
createSemantic(args): Promise<{
  acknowledged: boolean;
  has_embedding: boolean;
  id: string;
  label: string;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `agentId?`: `string` \| `null`; `label`: `string`; `metadata?`: `Record`\<`string`, `unknown`\> \| `null`; `source?`: `string`; `text`: `string`; `upsert?`: `boolean`; `userId`: `string`; `visibility?`: `string`; \} |
| `args.agentId?` | `string` \| `null` |
| `args.label` | `string` |
| `args.metadata?` | `Record`\<`string`, `unknown`\> \| `null` |
| `args.source?` | `string` |
| `args.text` | `string` |
| `args.upsert?` | `boolean` |
| `args.userId` | `string` |
| `args.visibility?` | `string` |

**Returns**

`Promise`\<\{
  `acknowledged`: `boolean`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `label`: `string`;
\}\>

**Implementation of**

[`MemoryCrudClient`](#api-memorycrudclient).[`createSemantic`](#api-createsemantic-1)

<a id="api-createtaxonomic"></a>

##### createTaxonomic()

```ts
createTaxonomic(args): Promise<{
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
| `args` | \{ `definition`: `string`; `domain`: `string`; `relatedTerms?`: `string`[] \| `null`; `term`: `string`; `userId`: `string`; `visibility?`: `string`; \} |
| `args.definition` | `string` |
| `args.domain` | `string` |
| `args.relatedTerms?` | `string`[] \| `null` |
| `args.term` | `string` |
| `args.userId` | `string` |
| `args.visibility?` | `string` |

**Returns**

`Promise`\<\{
  `acknowledged`: `boolean`;
  `domain`: `string`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `term`: `string`;
\}\>

**Implementation of**

[`MemoryCrudClient`](#api-memorycrudclient).[`createTaxonomic`](#api-createtaxonomic-1)

<a id="api-discoverprocedures-1"></a>

##### discoverProcedures()

```ts
discoverProcedures(args): Promise<Record<string, unknown>[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `metadataFilter?`: `Record`\<`string`, `unknown`\> \| `null`; `query`: `string`; `similarityThreshold?`: `number`; `tags?`: `string`[] \| `null`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.metadataFilter?` | `Record`\<`string`, `unknown`\> \| `null` |
| `args.query` | `string` |
| `args.similarityThreshold?` | `number` |
| `args.tags?` | `string`[] \| `null` |
| `args.topK?` | `number` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`Record`\<`string`, `unknown`\>[]\>

**Implementation of**

[`MemoryRuntime`](#api-memoryruntime).[`discoverProcedures`](#api-discoverprocedures-2)

<a id="api-getdistinctdomains"></a>

##### getDistinctDomains()

```ts
getDistinctDomains(args): Promise<string[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `visibility?`: `string` \| `null`; \} |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`string`[]\>

**Implementation of**

[`MemoryCrudClient`](#api-memorycrudclient).[`getDistinctDomains`](#api-getdistinctdomains-1)

<a id="api-getprocedural"></a>

##### getProcedural()

```ts
getProcedural(args): Promise<unknown>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `includeDeleted?`: `boolean`; `procedure?`: `string` \| `null`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.includeDeleted?` | `boolean` |
| `args.procedure?` | `string` \| `null` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`\>

**Implementation of**

[`MemoryCrudClient`](#api-memorycrudclient).[`getProcedural`](#api-getprocedural-1)

<a id="api-getsemantic-1"></a>

##### getSemantic()

```ts
getSemantic(args): Promise<unknown>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `label`: `string`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.label` | `string` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`\>

**Implementation of**

[`MemoryCrudClient`](#api-memorycrudclient).[`getSemantic`](#api-getsemantic-2)

<a id="api-gettaxonomic"></a>

##### getTaxonomic()

```ts
getTaxonomic(args): Promise<unknown>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `domain`: `string`; `term?`: `string` \| `null`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.domain` | `string` |
| `args.term?` | `string` \| `null` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`\>

**Implementation of**

[`MemoryCrudClient`](#api-memorycrudclient).[`getTaxonomic`](#api-gettaxonomic-1)

<a id="api-listepisodic"></a>

##### listEpisodic()

```ts
listEpisodic(args): Promise<unknown[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `limit?`: `number`; `sessionId?`: `string` \| `null`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.limit?` | `number` |
| `args.sessionId?` | `string` \| `null` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`[]\>

**Implementation of**

[`MemoryCrudClient`](#api-memorycrudclient).[`listEpisodic`](#api-listepisodic-1)

<a id="api-recordturn-1"></a>

##### recordTurn()

```ts
recordTurn(args): Promise<{
  acknowledged: boolean;
  has_embedding: boolean;
  id: string;
  session_id: string;
  turn_seq: number;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `agentId?`: `string` \| `null`; `content?`: `string` \| `null`; `idempotencyKey?`: `string` \| `null`; `isError?`: `boolean`; `modelName?`: `string` \| `null`; `role`: `string`; `sessionId?`: `string` \| `null`; `toolCallId?`: `string` \| `null`; `toolCalls?`: `Record`\<`string`, `unknown`\>[] \| `null`; `toolName?`: `string` \| `null`; `userId?`: `string` \| `null`; \} |
| `args.agentId?` | `string` \| `null` |
| `args.content?` | `string` \| `null` |
| `args.idempotencyKey?` | `string` \| `null` |
| `args.isError?` | `boolean` |
| `args.modelName?` | `string` \| `null` |
| `args.role` | `string` |
| `args.sessionId?` | `string` \| `null` |
| `args.toolCallId?` | `string` \| `null` |
| `args.toolCalls?` | `Record`\<`string`, `unknown`\>[] \| `null` |
| `args.toolName?` | `string` \| `null` |
| `args.userId?` | `string` \| `null` |

**Returns**

`Promise`\<\{
  `acknowledged`: `boolean`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `session_id`: `string`;
  `turn_seq`: `number`;
\}\>

**Implementation of**

[`MemoryRuntime`](#api-memoryruntime).[`recordTurn`](#api-recordturn-2)

<a id="api-retrievecustom"></a>

##### retrieveCustom()

```ts
retrieveCustom(args): Promise<{
  count: number;
  results: object[];
}>;
```

Identity is stamped by the platform.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `memoryType`: `string`; `query`: `string`; `tags?`: `Record`\<`string`, `unknown`\> \| `null`; `topK?`: `number`; \} |
| `args.memoryType` | `string` |
| `args.query` | `string` |
| `args.tags?` | `Record`\<`string`, `unknown`\> \| `null` |
| `args.topK?` | `number` |

**Returns**

`Promise`\<\{
  `count`: `number`;
  `results`: `object`[];
\}\>

**Implementation of**

[`MemoryCrudClient`](#api-memorycrudclient).[`retrieveCustom`](#api-retrievecustom-1)

<a id="api-searchepisodes-1"></a>

##### searchEpisodes()

```ts
searchEpisodes(args): Promise<object[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `query`: `string`; `sessionId?`: `string` \| `null`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.query` | `string` |
| `args.sessionId?` | `string` \| `null` |
| `args.topK?` | `number` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`object`[]\>

**Implementation of**

[`MemoryRuntime`](#api-memoryruntime).[`searchEpisodes`](#api-searchepisodes-2)

<a id="api-searchsemantic-1"></a>

##### searchSemantic()

```ts
searchSemantic(args): Promise<object[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `query`: `string`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.query` | `string` |
| `args.topK?` | `number` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`object`[]\>

**Implementation of**

[`MemoryRuntime`](#api-memoryruntime).[`searchSemantic`](#api-searchsemantic-2)

<a id="api-searchtaxonomic-1"></a>

##### searchTaxonomic()

```ts
searchTaxonomic(args): Promise<object[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `domain?`: `string` \| `null`; `query`: `string`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.domain?` | `string` \| `null` |
| `args.query` | `string` |
| `args.topK?` | `number` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`object`[]\>

**Implementation of**

[`MemoryRuntime`](#api-memoryruntime).[`searchTaxonomic`](#api-searchtaxonomic-2)

***

<a id="api-memoryclienterror"></a>

### MemoryClientError

Base for pre-HTTP client/usage errors (no status code or response body).

#### Extends

- `Error`

#### Extended by

- [`MemoryNotSupportedError`](#api-memorynotsupportederror)

<a id="api-constructor-5"></a>

#### Constructor

```ts
new MemoryClientError(message, options?): MemoryClientError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `message` | `string` |
| `options?` | `ErrorOptions` |

**Returns**

[`MemoryClientError`](#api-memoryclienterror)

**Overrides**

```ts
Error.constructor
```

***

<a id="api-memoryconnectionerror"></a>

### MemoryConnectionError

Raised when the transport fails to reach the memory backend (connect/timeout).

#### Extends

- [`MemoryAPIError`](#api-memoryapierror)

<a id="api-constructor-6"></a>

#### Constructor

```ts
new MemoryConnectionError(message, opts?): MemoryConnectionError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `message` | `string` |
| `opts?` | \{ `code?`: `string` \| `null`; `responseText?`: `string` \| `null`; `status?`: `number` \| `null`; \} |
| `opts.code?` | `string` \| `null` |
| `opts.responseText?` | `string` \| `null` |
| `opts.status?` | `number` \| `null` |

**Returns**

[`MemoryConnectionError`](#api-memoryconnectionerror)

**Overrides**

[`MemoryAPIError`](#api-memoryapierror).[`constructor`](#api-constructor-1)

#### Properties

| Property | Modifier | Type | Inherited from |
| :------ | :------ | :------ | :------ |
| <a id="api-property-code-3"></a> `code` | `readonly` | `string` \| `null` | [`MemoryAPIError`](#api-memoryapierror).[`code`](#api-property-code) |
| <a id="api-property-responsetext-3"></a> `responseText` | `readonly` | `string` \| `null` | [`MemoryAPIError`](#api-memoryapierror).[`responseText`](#api-property-responsetext) |
| <a id="api-property-status-3"></a> `status` | `readonly` | `number` \| `null` | [`MemoryAPIError`](#api-memoryapierror).[`status`](#api-property-status) |

***

<a id="api-memoryidentityerror"></a>

### MemoryIdentityError

Raised when a required identity dimension cannot be resolved.

#### Extends

- `Error`

<a id="api-constructor-7"></a>

#### Constructor

```ts
new MemoryIdentityError(message): MemoryIdentityError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `message` | `string` |

**Returns**

[`MemoryIdentityError`](#api-memoryidentityerror)

**Overrides**

```ts
Error.constructor
```

***

<a id="api-memorynotprovisionederror"></a>

### MemoryNotProvisionedError

Raised when the project's memory runtime is not yet reachable.

#### Extends

- [`MemoryAPIError`](#api-memoryapierror)

<a id="api-constructor-8"></a>

#### Constructor

```ts
new MemoryNotProvisionedError(message, opts?): MemoryNotProvisionedError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `message` | `string` |
| `opts?` | \{ `code?`: `string` \| `null`; `responseText?`: `string` \| `null`; `status?`: `number` \| `null`; \} |
| `opts.code?` | `string` \| `null` |
| `opts.responseText?` | `string` \| `null` |
| `opts.status?` | `number` \| `null` |

**Returns**

[`MemoryNotProvisionedError`](#api-memorynotprovisionederror)

**Overrides**

[`MemoryAPIError`](#api-memoryapierror).[`constructor`](#api-constructor-1)

#### Properties

| Property | Modifier | Type | Inherited from |
| :------ | :------ | :------ | :------ |
| <a id="api-property-code-4"></a> `code` | `readonly` | `string` \| `null` | [`MemoryAPIError`](#api-memoryapierror).[`code`](#api-property-code) |
| <a id="api-property-responsetext-4"></a> `responseText` | `readonly` | `string` \| `null` | [`MemoryAPIError`](#api-memoryapierror).[`responseText`](#api-property-responsetext) |
| <a id="api-property-status-4"></a> `status` | `readonly` | `number` \| `null` | [`MemoryAPIError`](#api-memoryapierror).[`status`](#api-property-status) |

***

<a id="api-memorynotsupportederror"></a>

### MemoryNotSupportedError

Raised when an operation is unavailable in the active transport mode.

This is a client-side capability check raised before any HTTP request is
attempted, so it carries no status code or response body.

#### Extends

- [`MemoryClientError`](#api-memoryclienterror)

<a id="api-constructor-9"></a>

#### Constructor

```ts
new MemoryNotSupportedError(message, options?): MemoryNotSupportedError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `message` | `string` |
| `options?` | `ErrorOptions` |

**Returns**

[`MemoryNotSupportedError`](#api-memorynotsupportederror)

**Overrides**

[`MemoryClientError`](#api-memoryclienterror).[`constructor`](#api-constructor-5)

***

<a id="api-memoryroutenotfounderror"></a>

### MemoryRouteNotFoundError

Raised when a core-loop request 404s, hinting a route-shape mismatch.

The SDK selects the route shape from `projectId` presence (set => project-scoped
Gateway routes; empty => flat OE routes). A 404 on a core-loop POST most often
means that shape does not match the backend the `baseUrl` points at, so this
carries a directional, actionable hint rather than the opaque 404. It extends
`MemoryBadRequestError` so existing 4xx handling still catches it.

#### Extends

- [`MemoryBadRequestError`](#api-memorybadrequesterror)

<a id="api-constructor-10"></a>

#### Constructor

```ts
new MemoryRouteNotFoundError(message, opts?): MemoryRouteNotFoundError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `message` | `string` |
| `opts?` | \{ `code?`: `string` \| `null`; `responseText?`: `string` \| `null`; `status?`: `number` \| `null`; \} |
| `opts.code?` | `string` \| `null` |
| `opts.responseText?` | `string` \| `null` |
| `opts.status?` | `number` \| `null` |

**Returns**

[`MemoryRouteNotFoundError`](#api-memoryroutenotfounderror)

**Overrides**

[`MemoryBadRequestError`](#api-memorybadrequesterror).[`constructor`](#api-constructor-3)

#### Properties

| Property | Modifier | Type | Inherited from |
| :------ | :------ | :------ | :------ |
| <a id="api-property-code-5"></a> `code` | `readonly` | `string` \| `null` | [`MemoryBadRequestError`](#api-memorybadrequesterror).[`code`](#api-property-code-2) |
| <a id="api-property-responsetext-5"></a> `responseText` | `readonly` | `string` \| `null` | [`MemoryBadRequestError`](#api-memorybadrequesterror).[`responseText`](#api-property-responsetext-2) |
| <a id="api-property-status-5"></a> `status` | `readonly` | `number` \| `null` | [`MemoryBadRequestError`](#api-memorybadrequesterror).[`status`](#api-property-status-2) |

***

<a id="api-memoryservererror"></a>

### MemoryServerError

Raised for 5xx responses, and for success responses whose body is unparseable
or has an unexpected shape.

#### Extends

- [`MemoryAPIError`](#api-memoryapierror)

<a id="api-constructor-11"></a>

#### Constructor

```ts
new MemoryServerError(message, opts?): MemoryServerError;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `message` | `string` |
| `opts?` | \{ `code?`: `string` \| `null`; `responseText?`: `string` \| `null`; `status?`: `number` \| `null`; \} |
| `opts.code?` | `string` \| `null` |
| `opts.responseText?` | `string` \| `null` |
| `opts.status?` | `number` \| `null` |

**Returns**

[`MemoryServerError`](#api-memoryservererror)

**Overrides**

[`MemoryAPIError`](#api-memoryapierror).[`constructor`](#api-constructor-1)

#### Properties

| Property | Modifier | Type | Inherited from |
| :------ | :------ | :------ | :------ |
| <a id="api-property-code-6"></a> `code` | `readonly` | `string` \| `null` | [`MemoryAPIError`](#api-memoryapierror).[`code`](#api-property-code) |
| <a id="api-property-responsetext-6"></a> `responseText` | `readonly` | `string` \| `null` | [`MemoryAPIError`](#api-memoryapierror).[`responseText`](#api-property-responsetext) |
| <a id="api-property-status-6"></a> `status` | `readonly` | `number` \| `null` | [`MemoryAPIError`](#api-memoryapierror).[`status`](#api-property-status) |

## Interfaces

<a id="api-ambientidentityruntime"></a>

### AmbientIdentityRuntime

Optional capability: a runtime that can supply per-call ambient identity.

Only app-bound runtimes running inside the platform stack have access to
per-request context. This is a separate interface so the facade can detect the
capability without requiring every runtime to implement it.

#### Methods

<a id="api-requestcontext"></a>

##### requestContext()

```ts
requestContext(): MemoryRequestContext | null;
```

**Returns**

[`MemoryRequestContext`](#api-memoryrequestcontext) \| `null`

***

<a id="api-memoryclientadapteroptions"></a>

### MemoryClientAdapterOptions

#### Properties

| Property | Type | Description |
| :------ | :------ | :------ |
| <a id="api-property-apiprefix"></a> `apiPrefix` | `string` | Memory path prefix passed through to `MemoryClient` — e.g. `/api/v1/memory` (OE) or `/api/v1/projects/<id>/memory` (Gateway). |
| <a id="api-property-baseurl"></a> `baseUrl` | `string` | Backend base URL (Gateway or OE). |
| <a id="api-property-fetchimpl"></a> `fetchImpl?` | [`FetchLike`](#api-fetchlike) | Injectable fetch, primarily for tests. |
| <a id="api-property-headers"></a> `headers?` | `Record`\<`string`, `string`\> | Static auth/identity headers (e.g. `Authorization`, `X-Agent-Engine-Execution-Id`). |
| <a id="api-property-projectscoped"></a> `projectScoped` | `boolean` | Whether the selected prefix is project-scoped. Used only to phrase the directional hint on a core-loop 404 (route-shape mismatch). |
| <a id="api-property-requestextras"></a> `requestExtras?` | [`RequestExtras`](#api-requestextras) | Per-request transport extras (e.g. an mTLS dispatcher for app-bound calls). |
| <a id="api-property-routestyle"></a> `routeStyle` | `MemoryRouteStyle` | Route convention the backend speaks. |
| <a id="api-property-timeout"></a> `timeout?` | `number` | Request timeout in seconds (default 30). |

***

<a id="api-memorycrudclient"></a>

### MemoryCrudClient

CRUD seam the facade's type-specific conveniences delegate to.

Like `MemoryRuntime`, tenancy is internal to each implementation.

#### Methods

<a id="api-close-2"></a>

##### close()?

```ts
optional close(): void | Promise<void>;
```

Optional: release underlying resources.

**Returns**

`void` \| `Promise`\<`void`\>

<a id="api-createcustom-1"></a>

##### createCustom()

```ts
createCustom(args): Promise<{
  has_embedding: boolean;
  id: string;
  tags: Record<string, string | number | boolean>;
  type: string;
}>;
```

Identity is stamped by the platform.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `content`: `string`; `contextualMetadata?`: `Record`\<`string`, `unknown`\> \| `null`; `memoryType`: `string`; `tags?`: `Record`\<`string`, `unknown`\> \| `null`; \} |
| `args.content` | `string` |
| `args.contextualMetadata?` | `Record`\<`string`, `unknown`\> \| `null` |
| `args.memoryType` | `string` |
| `args.tags?` | `Record`\<`string`, `unknown`\> \| `null` |

**Returns**

`Promise`\<\{
  `has_embedding`: `boolean`;
  `id`: `string`;
  `tags`: `Record`\<`string`, `string` \| `number` \| `boolean`\>;
  `type`: `string`;
\}\>

<a id="api-createepisodic-1"></a>

##### createEpisodic()

```ts
createEpisodic(args): Promise<{
  acknowledged: boolean;
  has_embedding: boolean;
  id: string;
  title: string;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `agentId?`: `string` \| `null`; `content`: `string`; `metadata?`: `Record`\<`string`, `unknown`\> \| `null`; `participants?`: `string`[] \| `null`; `sessionId`: `string`; `summaryText?`: `string` \| `null`; `tags?`: `string`[] \| `null`; `title`: `string`; `userId`: `string`; `visibility?`: `string`; \} |
| `args.agentId?` | `string` \| `null` |
| `args.content` | `string` |
| `args.metadata?` | `Record`\<`string`, `unknown`\> \| `null` |
| `args.participants?` | `string`[] \| `null` |
| `args.sessionId` | `string` |
| `args.summaryText?` | `string` \| `null` |
| `args.tags?` | `string`[] \| `null` |
| `args.title` | `string` |
| `args.userId` | `string` |
| `args.visibility?` | `string` |

**Returns**

`Promise`\<\{
  `acknowledged`: `boolean`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `title`: `string`;
\}\>

<a id="api-createprocedural-1"></a>

##### createProcedural()

```ts
createProcedural(args): Promise<{
  acknowledged: boolean;
  has_embedding: boolean;
  id: string;
  procedure: string;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `agentId?`: `string` \| `null`; `allowedTools?`: `string`[] \| `null`; `compatibility?`: `string` \| `null`; `content`: `string`; `description`: `string`; `extractionSource?`: `string` \| `null`; `license?`: `string` \| `null`; `procedure`: `string`; `resources?`: `Record`\<`string`, `unknown`\>[] \| `null`; `sourceFormat?`: `string` \| `null`; `sourcePath?`: `string` \| `null`; `steps?`: `Record`\<`string`, `unknown`\>[] \| `null`; `tags?`: `string`[] \| `null`; `triggerConditions?`: `string`[] \| `null`; `updateExisting?`: `boolean`; `userId`: `string`; `visibility?`: `string`; \} |
| `args.agentId?` | `string` \| `null` |
| `args.allowedTools?` | `string`[] \| `null` |
| `args.compatibility?` | `string` \| `null` |
| `args.content` | `string` |
| `args.description` | `string` |
| `args.extractionSource?` | `string` \| `null` |
| `args.license?` | `string` \| `null` |
| `args.procedure` | `string` |
| `args.resources?` | `Record`\<`string`, `unknown`\>[] \| `null` |
| `args.sourceFormat?` | `string` \| `null` |
| `args.sourcePath?` | `string` \| `null` |
| `args.steps?` | `Record`\<`string`, `unknown`\>[] \| `null` |
| `args.tags?` | `string`[] \| `null` |
| `args.triggerConditions?` | `string`[] \| `null` |
| `args.updateExisting?` | `boolean` |
| `args.userId` | `string` |
| `args.visibility?` | `string` |

**Returns**

`Promise`\<\{
  `acknowledged`: `boolean`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `procedure`: `string`;
\}\>

<a id="api-createsemantic-1"></a>

##### createSemantic()

```ts
createSemantic(args): Promise<{
  acknowledged: boolean;
  has_embedding: boolean;
  id: string;
  label: string;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `agentId?`: `string` \| `null`; `label`: `string`; `metadata?`: `Record`\<`string`, `unknown`\> \| `null`; `source?`: `string`; `text`: `string`; `upsert?`: `boolean`; `userId`: `string`; `visibility?`: `string`; \} |
| `args.agentId?` | `string` \| `null` |
| `args.label` | `string` |
| `args.metadata?` | `Record`\<`string`, `unknown`\> \| `null` |
| `args.source?` | `string` |
| `args.text` | `string` |
| `args.upsert?` | `boolean` |
| `args.userId` | `string` |
| `args.visibility?` | `string` |

**Returns**

`Promise`\<\{
  `acknowledged`: `boolean`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `label`: `string`;
\}\>

<a id="api-createtaxonomic-1"></a>

##### createTaxonomic()

```ts
createTaxonomic(args): Promise<{
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
| `args` | \{ `definition`: `string`; `domain`: `string`; `relatedTerms?`: `string`[] \| `null`; `term`: `string`; `userId`: `string`; `visibility?`: `string`; \} |
| `args.definition` | `string` |
| `args.domain` | `string` |
| `args.relatedTerms?` | `string`[] \| `null` |
| `args.term` | `string` |
| `args.userId` | `string` |
| `args.visibility?` | `string` |

**Returns**

`Promise`\<\{
  `acknowledged`: `boolean`;
  `domain`: `string`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `term`: `string`;
\}\>

<a id="api-getdistinctdomains-1"></a>

##### getDistinctDomains()

```ts
getDistinctDomains(args): Promise<string[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `visibility?`: `string` \| `null`; \} |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`string`[]\>

<a id="api-getprocedural-1"></a>

##### getProcedural()

```ts
getProcedural(args): Promise<unknown>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `includeDeleted?`: `boolean`; `procedure?`: `string` \| `null`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.includeDeleted?` | `boolean` |
| `args.procedure?` | `string` \| `null` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`\>

<a id="api-getsemantic-2"></a>

##### getSemantic()

```ts
getSemantic(args): Promise<unknown>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `label`: `string`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.label` | `string` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`\>

<a id="api-gettaxonomic-1"></a>

##### getTaxonomic()

```ts
getTaxonomic(args): Promise<unknown>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `domain`: `string`; `term?`: `string` \| `null`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.domain` | `string` |
| `args.term?` | `string` \| `null` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`\>

<a id="api-listepisodic-1"></a>

##### listEpisodic()

```ts
listEpisodic(args): Promise<unknown[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `limit?`: `number`; `sessionId?`: `string` \| `null`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.limit?` | `number` |
| `args.sessionId?` | `string` \| `null` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`unknown`[]\>

<a id="api-retrievecustom-1"></a>

##### retrieveCustom()

```ts
retrieveCustom(args): Promise<{
  count: number;
  results: object[];
}>;
```

Identity is stamped by the platform.

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `memoryType`: `string`; `query`: `string`; `tags?`: `Record`\<`string`, `unknown`\> \| `null`; `topK?`: `number`; \} |
| `args.memoryType` | `string` |
| `args.query` | `string` |
| `args.tags?` | `Record`\<`string`, `unknown`\> \| `null` |
| `args.topK?` | `number` |

**Returns**

`Promise`\<\{
  `count`: `number`;
  `results`: `object`[];
\}\>

***

<a id="api-memoryrequestcontext"></a>

### MemoryRequestContext

Immutable identity context carried across a memory call.

#### Properties

| Property | Modifier | Type |
| :------ | :------ | :------ |
| <a id="api-property-agentid"></a> `agentId?` | `readonly` | `string` \| `null` |
| <a id="api-property-sessionid"></a> `sessionId?` | `readonly` | `string` \| `null` |
| <a id="api-property-userid"></a> `userId?` | `readonly` | `string` \| `null` |

***

<a id="api-memoryruntime"></a>

### MemoryRuntime

Transport seam every memory backend implements.

Identity is passed as resolved arguments; tenancy is internal to each runtime.

#### Methods

<a id="api-buildcontext-2"></a>

##### buildContext()

```ts
buildContext(args): Promise<{
  formatted_context: string | Record<string, unknown>[];
  metadata: {
     memory_counts: Record<string, number>;
     timing: Record<string, number>;
     token_count: number;
  };
  selected_memories?: object[] | null;
}>;
```

**Parameters**

| Parameter | Type | Description |
| :------ | :------ | :------ |
| `args` | \{ `enabledSources?`: `Set`\<`string`\> \| `null`; `maxTokens?`: `number`; `metadataFilter?`: `Record`\<`string`, `unknown`\> \| `null`; `query`: `string`; `sessionId?`: `string` \| `null`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} | - |
| `args.enabledSources?` | `Set`\<`string`\> \| `null` | - |
| `args.maxTokens?` | `number` | Optional gross context-construction budget. After retrieval and ranking, the server subtracts a 500-token formatting reserve, then greedily selects whole memory chunks that fit in the remainder. Positive values at or below 500 leave no budget for memories. Values above 500 can still yield empty context when no chunk fits. |
| `args.metadataFilter?` | `Record`\<`string`, `unknown`\> \| `null` | - |
| `args.query` | `string` | - |
| `args.sessionId?` | `string` \| `null` | - |
| `args.topK?` | `number` | - |
| `args.userId?` | `string` \| `null` | - |
| `args.visibility?` | `string` \| `null` | - |

**Returns**

`Promise`\<\{
  `formatted_context`: `string` \| `Record`\<`string`, `unknown`\>[];
  `metadata`: \{
     `memory_counts`: `Record`\<`string`, `number`\>;
     `timing`: `Record`\<`string`, `number`\>;
     `token_count`: `number`;
  \};
  `selected_memories?`: `object`[] \| `null`;
\}\>

<a id="api-close-3"></a>

##### close()?

```ts
optional close(): void | Promise<void>;
```

Optional: release underlying resources.

**Returns**

`void` \| `Promise`\<`void`\>

<a id="api-discoverprocedures-2"></a>

##### discoverProcedures()

```ts
discoverProcedures(args): Promise<Record<string, unknown>[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `metadataFilter?`: `Record`\<`string`, `unknown`\> \| `null`; `query`: `string`; `similarityThreshold?`: `number`; `tags?`: `string`[] \| `null`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.metadataFilter?` | `Record`\<`string`, `unknown`\> \| `null` |
| `args.query` | `string` |
| `args.similarityThreshold?` | `number` |
| `args.tags?` | `string`[] \| `null` |
| `args.topK?` | `number` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`Record`\<`string`, `unknown`\>[]\>

<a id="api-recordturn-2"></a>

##### recordTurn()

```ts
recordTurn(args): Promise<{
  acknowledged: boolean;
  has_embedding: boolean;
  id: string;
  session_id: string;
  turn_seq: number;
}>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `agentId?`: `string` \| `null`; `content?`: `string` \| `null`; `idempotencyKey?`: `string` \| `null`; `isError?`: `boolean`; `modelName?`: `string` \| `null`; `role`: `string`; `sessionId?`: `string` \| `null`; `toolCallId?`: `string` \| `null`; `toolCalls?`: `Record`\<`string`, `unknown`\>[] \| `null`; `toolName?`: `string` \| `null`; `userId?`: `string` \| `null`; \} |
| `args.agentId?` | `string` \| `null` |
| `args.content?` | `string` \| `null` |
| `args.idempotencyKey?` | `string` \| `null` |
| `args.isError?` | `boolean` |
| `args.modelName?` | `string` \| `null` |
| `args.role` | `string` |
| `args.sessionId?` | `string` \| `null` |
| `args.toolCallId?` | `string` \| `null` |
| `args.toolCalls?` | `Record`\<`string`, `unknown`\>[] \| `null` |
| `args.toolName?` | `string` \| `null` |
| `args.userId?` | `string` \| `null` |

**Returns**

`Promise`\<\{
  `acknowledged`: `boolean`;
  `has_embedding`: `boolean`;
  `id`: `string`;
  `session_id`: `string`;
  `turn_seq`: `number`;
\}\>

<a id="api-searchepisodes-2"></a>

##### searchEpisodes()

```ts
searchEpisodes(args): Promise<object[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `query`: `string`; `sessionId?`: `string` \| `null`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.query` | `string` |
| `args.sessionId?` | `string` \| `null` |
| `args.topK?` | `number` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`object`[]\>

<a id="api-searchsemantic-2"></a>

##### searchSemantic()

```ts
searchSemantic(args): Promise<object[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `query`: `string`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.query` | `string` |
| `args.topK?` | `number` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`object`[]\>

<a id="api-searchtaxonomic-2"></a>

##### searchTaxonomic()

```ts
searchTaxonomic(args): Promise<object[]>;
```

**Parameters**

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `domain?`: `string` \| `null`; `query`: `string`; `topK?`: `number`; `userId?`: `string` \| `null`; `visibility?`: `string` \| `null`; \} |
| `args.domain?` | `string` \| `null` |
| `args.query` | `string` |
| `args.topK?` | `number` |
| `args.userId?` | `string` \| `null` |
| `args.visibility?` | `string` \| `null` |

**Returns**

`Promise`\<`object`[]\>

## Type Aliases

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

<a id="api-createtaxonomicresult"></a>

### CreateTaxonomicResult

```ts
type CreateTaxonomicResult = z.infer<typeof CreateTaxonomicResultSchema>;
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

<a id="api-fetchlike"></a>

### FetchLike

```ts
type FetchLike = (url, init) => Promise<Response>;
```

A minimal fetch signature so tests can inject a stub.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `url` | `string` |
| `init` | `RequestInit` |

#### Returns

`Promise`\<`Response`\>

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

***

<a id="api-memorychunk"></a>

### MemoryChunk

```ts
type MemoryChunk = z.infer<typeof MemoryChunkSchema>;
```

***

<a id="api-memoryoptions"></a>

### MemoryOptions

```ts
type MemoryOptions =
  | {
  apiKey?: string | null;
  baseUrl?: string | null;
  fetchImpl?: FetchLike;
  projectId?: string | null;
  serviceAccountToken?: string | null;
}
  | {
  client?: MemoryCrudClient | null;
  runtime: MemoryRuntime;
};
```

@mongodb-js/agent-engine-sdk-memory — unified Memory facade for Atlas Agent Engine.

#### Union Members

##### Type Literal

```ts
{
  apiKey?: string | null;
  baseUrl?: string | null;
  fetchImpl?: FetchLike;
  projectId?: string | null;
  serviceAccountToken?: string | null;
}
```

| Name | Type | Description |
| :------ | :------ | :------ |
| `apiKey?` | `string` \| `null` | **Deprecated** Use `serviceAccountToken` (or the AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN env var). Still accepted as an alias; emits a DeprecationWarning. |
| `baseUrl?` | `string` \| `null` | - |
| `fetchImpl?` | [`FetchLike`](#api-fetchlike) | Injectable fetch, primarily for tests. |
| `projectId?` | `string` \| `null` | - |
| `serviceAccountToken?` | `string` \| `null` | Auth: a service-account access token minted at POST /api/v1/oauth/token. |

***

##### Type Literal

```ts
{
  client?: MemoryCrudClient | null;
  runtime: MemoryRuntime;
}
```

***

<a id="api-memorysource"></a>

### MemorySource

```ts
type MemorySource = typeof MemorySource[keyof typeof MemorySource];
```

Source type for memory chunks.

***

<a id="api-requestextras"></a>

### RequestExtras

```ts
type RequestExtras = () => RequestInit;
```

Extra per-request options a transport implementation can layer in (e.g. mTLS agent).

#### Returns

`RequestInit`

***

<a id="api-resolvedidentity"></a>

### ResolvedIdentity

```ts
type ResolvedIdentity = Record<Field, string | null>;
```

***

<a id="api-retrievedcustommemory"></a>

### RetrievedCustomMemory

```ts
type RetrievedCustomMemory = z.infer<typeof RetrievedCustomMemorySchema>;
```

***

<a id="api-searchsource"></a>

### SearchSource

```ts
type SearchSource = typeof SearchSource[keyof typeof SearchSource];
```

A memory source that `Memory.search` can query.

Mirrors `MemorySource` minus `stm` (short-term turns are not a search target).
Callers may pass either the constant or its string value.

***

<a id="api-tagmap"></a>

### TagMap

```ts
type TagMap = Record<string,
  | TagScalar
| Record<string, TagScalar>>;
```

***

<a id="api-tagscalar"></a>

### TagScalar

```ts
type TagScalar = z.infer<typeof TagScalarSchema>;
```

***

<a id="api-writeturnresult"></a>

### WriteTurnResult

```ts
type WriteTurnResult = z.infer<typeof WriteTurnResultSchema>;
```

## Variables

<a id="api-contextmetadataschema"></a>

### ContextMetadataSchema

```ts
const ContextMetadataSchema: ZodObject<{
  memory_counts: ZodDefault<ZodRecord<ZodString, ZodNumber>>;
  timing: ZodDefault<ZodRecord<ZodString, ZodNumber>>;
  token_count: ZodDefault<ZodNumber>;
}, $strip>;
```

***

<a id="api-contextresponseschema"></a>

### ContextResponseSchema

```ts
const ContextResponseSchema: ZodObject<{
  formatted_context: ZodUnion<readonly [ZodString, ZodArray<ZodRecord<ZodString, ZodUnknown>>]>;
  metadata: ZodDefault<ZodObject<{
     memory_counts: ZodDefault<ZodRecord<ZodString, ZodNumber>>;
     timing: ZodDefault<ZodRecord<ZodString, ZodNumber>>;
     token_count: ZodDefault<ZodNumber>;
  }, $strip>>;
  selected_memories: ZodOptional<ZodNullable<ZodArray<ZodObject<{
     content: ZodString;
     embedding: ZodOptional<ZodNullable<ZodArray<ZodNumber>>>;
     id: ZodString;
     metadata: ZodOptional<ZodNullable<ZodRecord<ZodString, ZodUnknown>>>;
     similarity_score: ZodOptional<ZodNullable<ZodNumber>>;
     source: ZodEnum<{
        episodic: "episodic";
        procedural: "procedural";
        semantic: "semantic";
        stm: "stm";
        taxonomic: "taxonomic";
     }>;
     timestamp: ZodCoercedDate<unknown>;
  }, $strip>>>>;
}, $strip>;
```

Final response model for context building. `formatted_context` is a string or
a list of message dicts depending on the server's format style.

***

<a id="api-createepisodicresultschema"></a>

### CreateEpisodicResultSchema

```ts
const CreateEpisodicResultSchema: ZodObject<{
  acknowledged: ZodDefault<ZodBoolean>;
  has_embedding: ZodBoolean;
  id: ZodString;
  title: ZodString;
}, $strip>;
```

***

<a id="api-createproceduralresultschema"></a>

### CreateProceduralResultSchema

```ts
const CreateProceduralResultSchema: ZodObject<{
  acknowledged: ZodDefault<ZodBoolean>;
  has_embedding: ZodBoolean;
  id: ZodString;
  procedure: ZodString;
}, $strip>;
```

***

<a id="api-createsemanticresultschema"></a>

### CreateSemanticResultSchema

```ts
const CreateSemanticResultSchema: ZodObject<{
  acknowledged: ZodDefault<ZodBoolean>;
  has_embedding: ZodBoolean;
  id: ZodString;
  label: ZodString;
}, $strip>;
```

***

<a id="api-createtaxonomicresultschema"></a>

### CreateTaxonomicResultSchema

```ts
const CreateTaxonomicResultSchema: ZodObject<{
  acknowledged: ZodDefault<ZodBoolean>;
  domain: ZodString;
  has_embedding: ZodBoolean;
  id: ZodString;
  term: ZodString;
}, $strip>;
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

<a id="api-memorychunkschema"></a>

### MemoryChunkSchema

```ts
const MemoryChunkSchema: ZodObject<{
  content: ZodString;
  embedding: ZodOptional<ZodNullable<ZodArray<ZodNumber>>>;
  id: ZodString;
  metadata: ZodOptional<ZodNullable<ZodRecord<ZodString, ZodUnknown>>>;
  similarity_score: ZodOptional<ZodNullable<ZodNumber>>;
  source: ZodEnum<{
     episodic: "episodic";
     procedural: "procedural";
     semantic: "semantic";
     stm: "stm";
     taxonomic: "taxonomic";
  }>;
  timestamp: ZodCoercedDate<unknown>;
}, $strip>;
```

Unified representation of memory from any source (STM, episodic, semantic, …).

`timestamp` is coerced to a `Date`; a missing value on the wire is a server
contract violation and surfaces as a schema error at parse time.

`similarity_score` is on the scale of the retrieval mode that produced the
chunk: vector similarity for semantic mode, text relevance for text mode, and
the fused rank-fusion score for hybrid mode. Comparable between chunks
retrieved the same way; not comparable across modes, or across sources
searched differently. Null when the chunk did not come from a search.

***

<a id="api-memorysource-1"></a>

### MemorySource

```ts
const MemorySource: object;
```

Source type for memory chunks.

#### Type Declaration

| Name | Type | Default value |
| :------ | :------ | :------ |
| <a id="api-property-episodic"></a> `EPISODIC` | `"episodic"` | `"episodic"` |
| <a id="api-property-procedural"></a> `PROCEDURAL` | `"procedural"` | `"procedural"` |
| <a id="api-property-semantic"></a> `SEMANTIC` | `"semantic"` | `"semantic"` |
| <a id="api-property-stm"></a> `STM` | `"stm"` | `"stm"` |
| <a id="api-property-taxonomic"></a> `TAXONOMIC` | `"taxonomic"` | `"taxonomic"` |

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

<a id="api-searchsource-1"></a>

### SearchSource

```ts
const SearchSource: object;
```

A memory source that `Memory.search` can query.

Mirrors `MemorySource` minus `stm` (short-term turns are not a search target).
Callers may pass either the constant or its string value.

#### Type Declaration

| Name | Type | Default value |
| :------ | :------ | :------ |
| <a id="api-property-episodic-1"></a> `EPISODIC` | `"episodic"` | `"episodic"` |
| <a id="api-property-procedural-1"></a> `PROCEDURAL` | `"procedural"` | `"procedural"` |
| <a id="api-property-semantic-1"></a> `SEMANTIC` | `"semantic"` | `"semantic"` |
| <a id="api-property-taxonomic-1"></a> `TAXONOMIC` | `"taxonomic"` | `"taxonomic"` |

***

<a id="api-writeturnresultschema"></a>

### WriteTurnResultSchema

```ts
const WriteTurnResultSchema: ZodObject<{
  acknowledged: ZodDefault<ZodBoolean>;
  has_embedding: ZodDefault<ZodBoolean>;
  id: ZodString;
  session_id: ZodString;
  turn_seq: ZodNumber;
}, $strip>;
```

## Functions

<a id="api-hasambientidentity"></a>

### hasAmbientIdentity()

```ts
function hasAmbientIdentity(runtime): runtime is MemoryRuntime & AmbientIdentityRuntime;
```

Duck-typed check for the [AmbientIdentityRuntime](#api-ambientidentityruntime) capability.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `runtime` | [`MemoryRuntime`](#api-memoryruntime) |

#### Returns

`runtime is MemoryRuntime & AmbientIdentityRuntime`

***

<a id="api-resolveidentity"></a>

### resolveIdentity()

```ts
function resolveIdentity(args): ResolvedIdentity;
```

Resolve identity fields by precedence: call arg > bind ctx > runtime ctx.

Blank or whitespace-only values are treated as unset at every tier.

Two optional guards drop tiers for one field each:

`suppressRuntimeUserId` drops the runtime tier for `userId`. A read asking for
a broad visibility must not borrow the ambient principal as its filter. A
`userId` set by the caller or by bind is deliberate and still applies.

`suppressInheritedSessionId` drops the bind and runtime tiers for `sessionId`,
so only an explicit call arg applies. A `sessionId` scopes conversation I/O,
not long-term search; episodic search matches it exactly and consolidated
episodic memory is stored without one, so an inherited session matches nothing.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `args` | \{ `bindCtx?`: [`MemoryRequestContext`](#api-memoryrequestcontext) \| `null`; `callArgs`: `Partial`\<`Record`\<`Field`, `string` \| `null` \| `undefined`\>\>; `required?`: readonly (`"userId"` \| `"agentId"` \| `"sessionId"`)[]; `runtimeCtx?`: [`MemoryRequestContext`](#api-memoryrequestcontext) \| `null`; `suppressInheritedSessionId?`: `boolean`; `suppressRuntimeUserId?`: `boolean`; \} |
| `args.bindCtx?` | [`MemoryRequestContext`](#api-memoryrequestcontext) \| `null` |
| `args.callArgs` | `Partial`\<`Record`\<`Field`, `string` \| `null` \| `undefined`\>\> |
| `args.required?` | readonly (`"userId"` \| `"agentId"` \| `"sessionId"`)[] |
| `args.runtimeCtx?` | [`MemoryRequestContext`](#api-memoryrequestcontext) \| `null` |
| `args.suppressInheritedSessionId?` | `boolean` |
| `args.suppressRuntimeUserId?` | `boolean` |

#### Returns

[`ResolvedIdentity`](#api-resolvedidentity)

***

<a id="api-tosearchsource"></a>

### toSearchSource()

```ts
function toSearchSource(value): SearchSource;
```

Narrow an arbitrary string to a `SearchSource`, throwing on an unknown value.

#### Parameters

| Parameter | Type |
| :------ | :------ |
| `value` | `string` |

#### Returns

[`SearchSource`](#api-searchsource)
