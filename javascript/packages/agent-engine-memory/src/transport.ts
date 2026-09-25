/** Seam primitives: the request context and the runtime/CRUD interfaces. */

import type {
  ContextResponse,
  CreateEpisodicResult,
  CreateProceduralResult,
  CreateSemanticResult,
  CreateTaxonomicResult,
  CustomMemoryRetrieveResult,
  CustomMemorySaveResult,
  MemoryChunk,
  WriteTurnResult,
} from "./models.js";

/** A minimal fetch signature so tests can inject a stub. */
export type FetchLike = (url: string, init: RequestInit) => Promise<Response>;

/** Extra per-request options a transport implementation can layer in (e.g. mTLS agent). */
export type RequestExtras = () => RequestInit;

/** Immutable identity context carried across a memory call. */
export interface MemoryRequestContext {
  readonly userId?: string | null;
  readonly agentId?: string | null;
  readonly sessionId?: string | null;
}

/**
 * Transport seam every memory backend implements.
 *
 * Identity is passed as resolved arguments; tenancy is internal to each runtime.
 */
export interface MemoryRuntime {
  recordTurn(args: {
    role: string;
    content?: string | null;
    sessionId?: string | null;
    userId?: string | null;
    agentId?: string | null;
    toolCalls?: Array<Record<string, unknown>> | null;
    toolCallId?: string | null;
    toolName?: string | null;
    isError?: boolean;
    modelName?: string | null;
    idempotencyKey?: string | null;
  }): Promise<WriteTurnResult>;

  buildContext(args: {
    query: string;
    sessionId?: string | null;
    userId?: string | null;
    visibility?: string | null;
    metadataFilter?: Record<string, unknown> | null;
    enabledSources?: Set<string> | null;
    topK?: number;
    /**
     * Optional gross context-construction budget. After retrieval and ranking,
     * the server subtracts a 500-token formatting reserve, then greedily
     * selects whole memory chunks that fit in the remainder. Positive values
     * at or below 500 leave no budget for memories. Values above 500
     * can still yield empty context when no chunk fits.
     */
    maxTokens?: number;
  }): Promise<ContextResponse>;

  searchSemantic(args: {
    query: string;
    userId?: string | null;
    visibility?: string | null;
    topK?: number;
  }): Promise<MemoryChunk[]>;

  searchEpisodes(args: {
    query: string;
    userId?: string | null;
    visibility?: string | null;
    sessionId?: string | null;
    topK?: number;
  }): Promise<MemoryChunk[]>;

  searchTaxonomic(args: {
    query: string;
    userId?: string | null;
    domain?: string | null;
    visibility?: string | null;
    topK?: number;
  }): Promise<MemoryChunk[]>;

  discoverProcedures(args: {
    query: string;
    userId?: string | null;
    visibility?: string | null;
    tags?: string[] | null;
    topK?: number;
    similarityThreshold?: number;
    metadataFilter?: Record<string, unknown> | null;
  }): Promise<Array<Record<string, unknown>>>;

  /** Optional: release underlying resources. */
  close?(): void | Promise<void>;
}

/**
 * Optional capability: a runtime that can supply per-call ambient identity.
 *
 * Only app-bound runtimes running inside the platform stack have access to
 * per-request context. This is a separate interface so the facade can detect the
 * capability without requiring every runtime to implement it.
 */
export interface AmbientIdentityRuntime {
  requestContext(): MemoryRequestContext | null;
}

/** Duck-typed check for the {@link AmbientIdentityRuntime} capability. */
export function hasAmbientIdentity(
  runtime: MemoryRuntime,
): runtime is MemoryRuntime & AmbientIdentityRuntime {
  return (
    typeof (runtime as Partial<AmbientIdentityRuntime>).requestContext ===
    "function"
  );
}

/**
 * CRUD seam the facade's type-specific conveniences delegate to.
 *
 * Like `MemoryRuntime`, tenancy is internal to each implementation.
 */
export interface MemoryCrudClient {
  createSemantic(args: {
    label: string;
    text: string;
    userId: string;
    source?: string;
    visibility?: string;
    agentId?: string | null;
    metadata?: Record<string, unknown> | null;
    upsert?: boolean;
  }): Promise<CreateSemanticResult>;

  getSemantic(args: {
    label: string;
    userId?: string | null;
    visibility?: string | null;
  }): Promise<unknown | null>;

  createEpisodic(args: {
    title: string;
    content: string;
    userId: string;
    sessionId: string;
    summaryText?: string | null;
    participants?: string[] | null;
    tags?: string[] | null;
    visibility?: string;
    metadata?: Record<string, unknown> | null;
    agentId?: string | null;
  }): Promise<CreateEpisodicResult>;

  listEpisodic(args: {
    userId?: string | null;
    sessionId?: string | null;
    visibility?: string | null;
    limit?: number;
  }): Promise<unknown[]>;

  createTaxonomic(args: {
    domain: string;
    term: string;
    definition: string;
    userId: string;
    relatedTerms?: string[] | null;
    visibility?: string;
  }): Promise<CreateTaxonomicResult>;

  getTaxonomic(args: {
    domain: string;
    term?: string | null;
    userId?: string | null;
    visibility?: string | null;
  }): Promise<unknown | null>;

  getDistinctDomains(args: { visibility?: string | null }): Promise<string[]>;

  createProcedural(args: {
    procedure: string;
    description: string;
    content: string;
    userId: string;
    steps?: Array<Record<string, unknown>> | null;
    resources?: Array<Record<string, unknown>> | null;
    allowedTools?: string[] | null;
    compatibility?: string | null;
    license?: string | null;
    triggerConditions?: string[] | null;
    tags?: string[] | null;
    visibility?: string;
    agentId?: string | null;
    extractionSource?: string | null;
    sourceFormat?: string | null;
    sourcePath?: string | null;
    updateExisting?: boolean;
  }): Promise<CreateProceduralResult>;

  getProcedural(args: {
    procedure?: string | null;
    userId?: string | null;
    visibility?: string | null;
    includeDeleted?: boolean;
  }): Promise<unknown | null>;

  /** Identity is stamped by the platform. */
  createCustom(args: {
    memoryType: string;
    content: string;
    tags?: Record<string, unknown> | null;
    contextualMetadata?: Record<string, unknown> | null;
  }): Promise<CustomMemorySaveResult>;

  /** Identity is stamped by the platform. */
  retrieveCustom(args: {
    memoryType: string;
    query: string;
    tags?: Record<string, unknown> | null;
    topK?: number;
  }): Promise<CustomMemoryRetrieveResult>;

  /** Optional: release underlying resources. */
  close?(): void | Promise<void>;
}
