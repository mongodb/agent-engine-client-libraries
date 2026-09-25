/** @mongodb-js/agent-engine-sdk-memory — unified Memory facade for Atlas Agent Engine. */

export { Memory, type MemoryOptions } from "./memory.js";

// Copyright 2026 MongoDB, Inc.
// SPDX-License-Identifier: Apache-2.0

export {
  MemorySource,
  SearchSource,
  toSearchSource,
  WriteTurnResultSchema,
  CreateSemanticResultSchema,
  CreateEpisodicResultSchema,
  CreateTaxonomicResultSchema,
  CreateProceduralResultSchema,
  CustomMemorySaveResultSchema,
  RetrievedCustomMemorySchema,
  CustomMemoryRetrieveResultSchema,
  MemoryChunkSchema,
  ContextMetadataSchema,
  ContextResponseSchema,
} from "./models.js";
export type {
  JsonValue,
  WriteTurnResult,
  CreateSemanticResult,
  CreateEpisodicResult,
  CreateTaxonomicResult,
  CreateProceduralResult,
  CustomMemorySaveResult,
  RetrievedCustomMemory,
  CustomMemoryRetrieveResult,
  MemoryChunk,
  ContextMetadata,
  ContextResponse,
} from "./models.js";

export type { TagScalar, TagMap } from "./tag_syntax.js";

export type {
  MemoryRuntime,
  MemoryCrudClient,
  MemoryRequestContext,
  AmbientIdentityRuntime,
  FetchLike,
  RequestExtras,
} from "./transport.js";
export { hasAmbientIdentity } from "./transport.js";

export { resolveIdentity, type ResolvedIdentity } from "./identity.js";

export {
  MemoryClientAdapter,
  type MemoryClientAdapterOptions,
} from "./http/client_adapter.js";

export {
  MemoryIdentityError,
  MemoryClientError,
  MemoryNotSupportedError,
  MemoryAPIError,
  MemoryAuthError,
  MemoryNotProvisionedError,
  MemoryBadRequestError,
  MemoryRouteNotFoundError,
  MemoryServerError,
  MemoryConnectionError,
} from "./errors.js";
