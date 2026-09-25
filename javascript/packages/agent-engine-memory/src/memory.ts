/**
 * Transport-free public Memory facade (TypeScript port).
 *
 * `Memory` is the surface platform users interact with. It delegates the
 * high-level workflow operations to an injected `MemoryRuntime` and the CRUD
 * conveniences to an injected `MemoryCrudClient`. Identity is resolved per call;
 * tenancy never appears in any public signature. Methods are async: the
 * underlying transport is HTTP.
 */

import { randomUUID } from "node:crypto";

import { MemoryIdentityError, MemoryNotSupportedError } from "./errors.js";
import { MemoryClientAdapter } from "./http/client_adapter.js";
import { resolveIdentity, type ResolvedIdentity } from "./identity.js";
import {
  MemoryChunkSchema,
  SearchSource,
  toSearchSource,
  type ContextResponse,
  type CreateEpisodicResult,
  type CreateProceduralResult,
  type CreateSemanticResult,
  type CreateTaxonomicResult,
  type CustomMemoryRetrieveResult,
  type CustomMemorySaveResult,
  type MemoryChunk,
  type WriteTurnResult,
} from "./models.js";
import {
  validateMemoryType,
  validateTagSyntax,
  type TagMap,
} from "./tag_syntax.js";
import {
  hasAmbientIdentity,
  type FetchLike,
  type MemoryCrudClient,
  type MemoryRequestContext,
  type MemoryRuntime,
} from "./transport.js";

/** Reject non-finite / non-integer / non-positive public maxTokens budgets. */
function requirePositiveMaxTokens(maxTokens: number | undefined): void {
  if (maxTokens === undefined) {
    return;
  }
  if (!Number.isInteger(maxTokens) || maxTokens <= 0) {
    throw new RangeError("maxTokens must be a positive integer");
  }
}

/** Trim to a non-empty identity value, or `null` when blank/absent. */
function normalizeId(value: string | null | undefined): string | null {
  if (value === null || value === undefined || value.trim() === "") return null;
  return value;
}

/**
 * Return the env var's value, or `null` when unset or blank — an
 * empty-exported placeholder (common in CI/Helm/dotenv files) is not a
 * provided credential, so it must neither authenticate nor count as a second
 * input. Matches the baseUrl/projectId convention.
 */
function envOrNull(name: string): string | null {
  const value = process.env[name];
  return value !== undefined && value.trim() !== "" ? value : null;
}

const DEFAULT_GATEWAY_URL = "https://agentengine.mongodb.com";
const SERVICE_ACCOUNT_TOKEN_ENV_VAR = "AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN";
// Deprecated alias for SERVICE_ACCOUNT_TOKEN_ENV_VAR, honored with a
// DeprecationWarning so existing deployments keep working.
const API_KEY_ENV_VAR = "AGENTIC_MEMORY_API_KEY";
const BASE_URL_ENV_VAR = "AGENTIC_MEMORY_BASE_URL";
const PROJECT_ID_ENV_VAR = "AGENTIC_MEMORY_PROJECT_ID";

/** Default sources for the zero-argument search(): the two ranked similarity sources. */
const DEFAULT_SEARCH_SOURCES: readonly SearchSource[] = [
  SearchSource.SEMANTIC,
  SearchSource.EPISODIC,
];

/** Resolve requested sources to a deduped, order-preserving list. */
function normalizeSources(
  sources:
    | SearchSource
    | string
    | ReadonlyArray<SearchSource | string>
    | null
    | undefined,
): SearchSource[] {
  if (sources === null || sources === undefined) {
    return [...DEFAULT_SEARCH_SOURCES];
  }
  const list = typeof sources === "string" ? [sources] : [...sources];
  const resolved: SearchSource[] = [];
  for (const source of list) {
    const member = toSearchSource(source);
    if (!resolved.includes(member)) {
      resolved.push(member);
    }
  }
  return resolved;
}

/** Adapt a `discoverProcedures` dict into a `MemoryChunk` for unified search. */
function procedureToChunk(proc: Record<string, unknown>): MemoryChunk {
  const rawId = proc.id ?? proc._id ?? "";
  const content = proc.content ?? proc.description ?? proc.procedure ?? "";
  const score = proc.score ?? proc.similarity_score ?? null;
  const rawTs = proc.timestamp ?? proc.created_at ?? proc.updated_at;
  return MemoryChunkSchema.parse({
    id: String(rawId),
    content: String(content),
    source: "procedural",
    // Fall back to the epoch (not now()) so an absent timestamp does not
    // masquerade as a fresh result and distort recency.
    timestamp: rawTs ?? "1970-01-01T00:00:00Z",
    similarity_score: score,
    metadata: proc,
  });
}

export type MemoryOptions =
  | {
      /** Auth: a service-account access token minted at POST /api/v1/oauth/token. */
      serviceAccountToken?: string | null;
      /**
       * @deprecated Use `serviceAccountToken` (or the
       * AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN env var). Still accepted as an
       * alias; emits a DeprecationWarning.
       */
      apiKey?: string | null;
      baseUrl?: string | null;
      projectId?: string | null;
      /** Injectable fetch, primarily for tests. */
      fetchImpl?: FetchLike;
    }
  | {
      runtime: MemoryRuntime;
      client?: MemoryCrudClient | null;
    };

function isInjected(
  opts: MemoryOptions,
): opts is { runtime: MemoryRuntime; client?: MemoryCrudClient | null } {
  return "runtime" in opts && opts.runtime !== undefined;
}

/** Public, transport-free memory facade. */
export class Memory {
  private readonly runtime: MemoryRuntime;
  private readonly client: MemoryCrudClient | null;
  private boundCtx: MemoryRequestContext | null;
  private ownsRuntime: boolean;
  private crudUnsupportedReason: string | null;

  constructor(opts: MemoryOptions = {}) {
    if (isInjected(opts)) {
      this.runtime = opts.runtime;
      this.client = opts.client ?? null;
      this.boundCtx = null;
      this.ownsRuntime = false;
      this.crudUnsupportedReason = null;
      return;
    }

    const token =
      opts.serviceAccountToken ?? envOrNull(SERVICE_ACCOUNT_TOKEN_ENV_VAR);
    const legacyToken = opts.apiKey ?? envOrNull(API_KEY_ENV_VAR);
    if (token !== null && legacyToken !== null) {
      throw new Error(
        `pass only one of serviceAccountToken (${SERVICE_ACCOUNT_TOKEN_ENV_VAR}) ` +
          `and the deprecated apiKey (${API_KEY_ENV_VAR})`,
      );
    }
    const authToken = token ?? legacyToken;
    if (authToken !== null && authToken.trim() === "") {
      throw new Error(
        `${token !== null ? "serviceAccountToken" : "apiKey"} must be a non-empty string`,
      );
    }
    // Validation precedes the warning so a rejected value never also warns.
    if (legacyToken !== null) {
      process.emitWarning(
        "apiKey (AGENTIC_MEMORY_API_KEY) is deprecated and will be removed in a " +
          "future release; use serviceAccountToken " +
          "(AGENTIC_MEMORY_SERVICE_ACCOUNT_TOKEN) instead",
        "DeprecationWarning",
      );
    }
    let projectId = opts.projectId ?? process.env[PROJECT_ID_ENV_VAR] ?? null;
    // A blank projectId is treated as unset (the discriminator is presence, and
    // an empty path segment would misroute); surrounding whitespace is stripped.
    if (projectId !== null) {
      projectId = projectId.trim() || null;
    }

    const resolvedUrl = Memory.resolveUrl(opts.baseUrl ?? null, authToken);
    if (resolvedUrl === null) {
      throw new Error(
        `Memory requires baseUrl, the ${BASE_URL_ENV_VAR} environment variable, ` +
          `or an auth token (serviceAccountToken; deprecated alias apiKey), which ` +
          `falls back to the hosted default ${DEFAULT_GATEWAY_URL}`,
      );
    }

    // projectId presence, not auth, picks the route shape: set => project-scoped
    // Gateway routes, empty => flat OE routes. Both backends serve the CRUD and
    // core-loop routes, so type-specific operations are always available.
    const apiPrefix = projectId
      ? `/api/v1/projects/${encodeURIComponent(projectId)}/memory`
      : "/api/v1/memory";
    const headers = authToken
      ? { Authorization: `Bearer ${authToken}` }
      : undefined;
    const adapter = new MemoryClientAdapter({
      baseUrl: resolvedUrl,
      apiPrefix,
      routeStyle: "aliased",
      projectScoped: projectId !== null,
      headers,
      fetchImpl: opts.fetchImpl,
    });
    this.runtime = adapter;
    this.client = adapter;
    this.crudUnsupportedReason = null;
    this.boundCtx = null;
    this.ownsRuntime = true;
  }

  private static resolveUrl(
    baseUrl: string | null,
    authToken: string | null,
  ): string | null {
    if (baseUrl !== null && baseUrl.trim() === "") {
      throw new Error("baseUrl must be a non-empty string");
    }
    const resolved = baseUrl ?? process.env[BASE_URL_ENV_VAR] ?? null;
    if (resolved !== null && resolved.trim() !== "") {
      return resolved.trim();
    }
    if (authToken) {
      return DEFAULT_GATEWAY_URL;
    }
    return null;
  }

  /** Return a new Memory scoped to `ctx` without mutating this handle. */
  bind(ctx: MemoryRequestContext): Memory {
    const bound = new Memory({ runtime: this.runtime, client: this.client });
    // Preserve ownership + capability, and apply the bound context. Same-class
    // instances may access each other's private fields.
    bound.boundCtx = ctx;
    bound.ownsRuntime = this.ownsRuntime;
    bound.crudUnsupportedReason = this.crudUnsupportedReason;
    return bound;
  }

  /** Release the underlying transport(s) when this handle owns them. */
  async close(): Promise<void> {
    if (!this.ownsRuntime) {
      return;
    }
    await this.runtime.close?.();
    await this.client?.close?.();
  }

  private requireClient(): MemoryCrudClient {
    if (this.client === null) {
      throw new MemoryNotSupportedError(
        this.crudUnsupportedReason ??
          "type-specific save/get/list operations require a CRUD client; provide " +
            "one via the client option when injecting a custom runtime, or use " +
            "recordTurn, buildContext, or search",
      );
    }
    return this.client;
  }

  private runtimeCtx(): MemoryRequestContext | null {
    // Only runtimes that declare the ambient-identity capability expose per-call
    // identity; read fresh on every resolve so a long-lived facade picks up
    // context that varies per request.
    return hasAmbientIdentity(this.runtime)
      ? this.runtime.requestContext()
      : null;
  }

  private resolve(args: {
    userId?: string | null;
    agentId?: string | null;
    sessionId?: string | null;
    required?: ReadonlyArray<"userId" | "agentId" | "sessionId">;
    visibility?: string | null;
    suppressInheritedSessionId?: boolean;
  }): ResolvedIdentity {
    const runtimeCtx = this.runtimeCtx();
    // Security: when an ambient (app-bound) identity is present, it is the
    // trusted principal resolved from the execution context. A caller- or
    // bind-supplied `userId` that disagrees with it must never take effect —
    // otherwise a tool that exposes `userId` as an LLM-controllable argument
    // could read or write another user's memory through the same OE-proxied
    // route. Fail closed on a mismatch rather than silently honoring the
    // override; an equal or absent `userId` is fine (matches the documented
    // "no userId threading" contract).
    const ambientUserId = normalizeId(runtimeCtx?.userId);
    if (ambientUserId !== null) {
      const requestedUserId =
        normalizeId(args.userId) ?? normalizeId(this.boundCtx?.userId);
      if (requestedUserId !== null && requestedUserId !== ambientUserId) {
        throw new MemoryIdentityError(
          "userId does not match the ambient app-bound identity: in app-bound " +
            "mode the user is resolved from the execution context and cannot be " +
            "overridden by a caller-supplied userId",
        );
      }
    }
    return resolveIdentity({
      callArgs: {
        userId: args.userId,
        agentId: args.agentId,
        sessionId: args.sessionId,
      },
      bindCtx: this.boundCtx,
      runtimeCtx,
      required: args.required,
      suppressRuntimeUserId:
        args.visibility != null && args.visibility !== "private",
      suppressInheritedSessionId: args.suppressInheritedSessionId ?? false,
    });
  }

  // ==========================================================================
  // Conversation turns
  // ==========================================================================

  /** Record a single conversation turn (write-accepted semantics). */
  async recordTurn(args: {
    role: string;
    content?: string | null;
    sessionId?: string | null;
    toolCalls?: Array<Record<string, unknown>> | null;
    toolCallId?: string | null;
    toolName?: string | null;
    isError?: boolean;
    modelName?: string | null;
    idempotencyKey?: string | null;
  }): Promise<WriteTurnResult> {
    const resolved = this.resolve({ sessionId: args.sessionId });
    return this.runtime.recordTurn({
      role: args.role,
      content: args.content,
      sessionId: resolved.sessionId,
      userId: resolved.userId,
      agentId: resolved.agentId,
      toolCalls: args.toolCalls,
      toolCallId: args.toolCallId,
      toolName: args.toolName,
      isError: args.isError,
      modelName: args.modelName,
      // Auto-generate per call when omitted so transport-level retries beneath
      // this call are protected; cross-call dedupe needs a caller-supplied key.
      idempotencyKey: args.idempotencyKey ?? randomUUID(),
    });
  }

  // ==========================================================================
  // Semantic memory
  // ==========================================================================

  async saveSemantic(args: {
    text: string;
    label: string;
    userId?: string | null;
    source?: string;
    visibility?: string;
    metadata?: Record<string, unknown> | null;
    upsert?: boolean;
    agentId?: string | null;
  }): Promise<CreateSemanticResult> {
    const resolved = this.resolve({
      userId: args.userId,
      agentId: args.agentId,
      required: ["userId"],
    });
    return this.requireClient().createSemantic({
      label: args.label,
      text: args.text,
      userId: resolved.userId as string,
      agentId: resolved.agentId,
      source: args.source ?? "agent",
      visibility: args.visibility ?? "private",
      metadata: args.metadata,
      upsert: args.upsert ?? true,
    });
  }

  async searchSemantic(args: {
    query: string;
    userId?: string | null;
    visibility?: string | null;
    topK?: number;
  }): Promise<MemoryChunk[]> {
    const resolved = this.resolve({
      userId: args.userId,
      visibility: args.visibility,
    });
    return this.runtime.searchSemantic({
      query: args.query,
      userId: resolved.userId,
      visibility: args.visibility,
      topK: args.topK ?? 50,
    });
  }

  async getSemantic(args: {
    label: string;
    userId?: string | null;
    visibility?: string | null;
  }): Promise<unknown | null> {
    const resolved = this.resolve({
      userId: args.userId,
      visibility: args.visibility,
    });
    return this.requireClient().getSemantic({
      label: args.label,
      userId: resolved.userId,
      visibility: args.visibility,
    });
  }

  // ==========================================================================
  // Episodic memory
  // ==========================================================================

  async saveEpisode(args: {
    title: string;
    content: string;
    userId?: string | null;
    summary?: string | null;
    sessionId?: string | null;
    participants?: string[] | null;
    tags?: string[] | null;
    visibility?: string;
    metadata?: Record<string, unknown> | null;
    agentId?: string | null;
  }): Promise<CreateEpisodicResult> {
    const resolved = this.resolve({
      userId: args.userId,
      agentId: args.agentId,
      sessionId: args.sessionId,
      required: ["userId", "sessionId"],
    });
    return this.requireClient().createEpisodic({
      title: args.title,
      content: args.content,
      summaryText: args.summary,
      userId: resolved.userId as string,
      agentId: resolved.agentId,
      sessionId: resolved.sessionId as string,
      participants: args.participants,
      tags: args.tags,
      visibility: args.visibility ?? "private",
      metadata: args.metadata,
    });
  }

  async searchEpisodes(args: {
    query: string;
    userId?: string | null;
    visibility?: string | null;
    sessionId?: string | null;
    topK?: number;
  }): Promise<MemoryChunk[]> {
    const resolved = this.resolve({
      userId: args.userId,
      sessionId: args.sessionId,
      visibility: args.visibility,
      suppressInheritedSessionId: true,
    });
    return this.runtime.searchEpisodes({
      query: args.query,
      userId: resolved.userId,
      visibility: args.visibility,
      sessionId: resolved.sessionId,
      topK: args.topK ?? 50,
    });
  }

  async listEpisodes(
    args: {
      userId?: string | null;
      visibility?: string | null;
      sessionId?: string | null;
      limit?: number;
    } = {},
  ): Promise<unknown[]> {
    const resolved = this.resolve({
      userId: args.userId,
      sessionId: args.sessionId,
      visibility: args.visibility,
      suppressInheritedSessionId: true,
    });
    return this.requireClient().listEpisodic({
      userId: resolved.userId,
      visibility: args.visibility,
      sessionId: resolved.sessionId,
      limit: args.limit ?? 20,
    });
  }

  // ==========================================================================
  // Taxonomic memory
  // ==========================================================================

  async saveTaxonomic(args: {
    domain: string;
    term: string;
    definition: string;
    relatedTerms?: string[] | null;
    visibility?: string;
    userId?: string | null;
  }): Promise<CreateTaxonomicResult> {
    const resolved = this.resolve({
      userId: args.userId,
      required: ["userId"],
    });
    return this.requireClient().createTaxonomic({
      domain: args.domain,
      term: args.term,
      definition: args.definition,
      relatedTerms: args.relatedTerms,
      visibility: args.visibility ?? "org",
      userId: resolved.userId as string,
    });
  }

  async searchTaxonomic(args: {
    query: string;
    userId?: string | null;
    domain?: string | null;
    visibility?: string | null;
    topK?: number;
  }): Promise<MemoryChunk[]> {
    const resolved = this.resolve({
      userId: args.userId,
      visibility: args.visibility,
    });
    return this.runtime.searchTaxonomic({
      query: args.query,
      userId: resolved.userId,
      domain: args.domain,
      visibility: args.visibility,
      topK: args.topK ?? 50,
    });
  }

  async getTaxonomicTerm(args: {
    domain: string;
    term: string;
    userId?: string | null;
    visibility?: string | null;
  }): Promise<unknown | null> {
    const resolved = this.resolve({
      userId: args.userId,
      visibility: args.visibility,
    });
    return this.requireClient().getTaxonomic({
      domain: args.domain,
      term: args.term,
      userId: resolved.userId,
      visibility: args.visibility,
    });
  }

  async listDomains(
    args: { visibility?: string | null } = {},
  ): Promise<string[]> {
    return this.requireClient().getDistinctDomains({
      visibility: args.visibility,
    });
  }

  // ==========================================================================
  // Procedural memory
  // ==========================================================================

  async discoverProcedures(args: {
    query: string;
    userId?: string | null;
    visibility?: string | null;
    tags?: string[] | null;
    topK?: number;
    similarityThreshold?: number;
    metadataFilter?: Record<string, unknown> | null;
  }): Promise<Array<Record<string, unknown>>> {
    const resolved = this.resolve({
      userId: args.userId,
      visibility: args.visibility,
    });
    return this.runtime.discoverProcedures({
      query: args.query,
      userId: resolved.userId,
      visibility: args.visibility,
      tags: args.tags,
      topK: args.topK ?? 10,
      similarityThreshold: args.similarityThreshold ?? 0.0,
      metadataFilter: args.metadataFilter,
    });
  }

  async getProcedure(args: {
    procedureName: string;
    userId?: string | null;
    visibility?: string | null;
    includeDeleted?: boolean;
  }): Promise<unknown | null> {
    const resolved = this.resolve({
      userId: args.userId,
      visibility: args.visibility,
    });
    return this.requireClient().getProcedural({
      procedure: args.procedureName,
      userId: resolved.userId,
      visibility: args.visibility,
      includeDeleted: args.includeDeleted ?? false,
    });
  }

  async saveProcedure(args: {
    procedure: string;
    description: string;
    content: string;
    userId?: string | null;
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
  }): Promise<CreateProceduralResult> {
    const resolved = this.resolve({
      userId: args.userId,
      agentId: args.agentId,
      required: ["userId"],
    });
    return this.requireClient().createProcedural({
      procedure: args.procedure,
      description: args.description,
      content: args.content,
      userId: resolved.userId as string,
      agentId: resolved.agentId,
      steps: args.steps,
      resources: args.resources,
      allowedTools: args.allowedTools,
      compatibility: args.compatibility,
      license: args.license,
      triggerConditions: args.triggerConditions,
      tags: args.tags,
      visibility: args.visibility ?? "private",
      extractionSource: args.extractionSource,
      sourceFormat: args.sourceFormat,
      sourcePath: args.sourcePath,
      updateExisting: args.updateExisting ?? false,
    });
  }

  // ==========================================================================
  // Custom memory types
  // ==========================================================================

  /**
   * Save a memory of a declared custom type.
   *
   * The platform stamps identity (org, project, user) and enforces the
   * type's declared tag schema; this method validates only tag syntax
   * client-side so mistakes fail fast with server-matching messages.
   * Built-in types (semantic, episodic, taxonomic, procedural) are
   * rejected — use their dedicated methods.
   */
  async save(
    memoryType: string,
    content: string,
    options?: {
      tags?: TagMap | null;
      contextualMetadata?: Record<string, unknown> | null;
    },
  ): Promise<CustomMemorySaveResult> {
    validateMemoryType(memoryType);
    if (options?.tags != null) {
      validateTagSyntax(options.tags);
    }
    return this.requireClient().createCustom({
      memoryType,
      content,
      tags: options?.tags,
      contextualMetadata: options?.contextualMetadata,
    });
  }

  /**
   * Retrieve memories of a declared custom type by semantic query.
   *
   * Filters are exact-match equality on declared tag keys; result ordering
   * may improve between releases and is not contractual. One type per call.
   */
  async retrieve(
    memoryType: string,
    query: string,
    options?: { tags?: TagMap | null; topK?: number },
  ): Promise<CustomMemoryRetrieveResult> {
    validateMemoryType(memoryType);
    if (options?.tags != null) {
      validateTagSyntax(options.tags);
    }
    return this.requireClient().retrieveCustom({
      memoryType,
      query,
      tags: options?.tags,
      topK: options?.topK ?? 10,
    });
  }

  // ==========================================================================
  // Context building
  // ==========================================================================

  /** Build a unified memory context across memory types. */
  async buildContext(args: {
    query: string;
    userId?: string | null;
    sessionId?: string | null;
    visibility?: string | null;
    metadataFilter?: Record<string, unknown> | null;
    enabledSources?: Set<string> | null;
    /** Deprecated alias for `sessionId`; a non-blank `sessionId` wins. */
    threadId?: string | null;
    /**
     * Optional gross context-construction budget. Omitted preserves prior
     * behavior. Non-positive, non-integer, and non-finite values raise
     * before runtime delegation. After retrieval and ranking, the server
     * subtracts a 500-token formatting reserve, then greedily selects whole
     * memory chunks that fit in the remainder. Positive values at or below 500
     * leave no budget for memories. Values above 500 can still
     * yield empty context when no chunk fits. `metadata.token_count` reports
     * formatted output only and excludes the reserve.
     */
    maxTokens?: number;
  }): Promise<ContextResponse> {
    requirePositiveMaxTokens(args.maxTokens);
    const session =
      args.sessionId && args.sessionId.trim() ? args.sessionId : args.threadId;
    const resolved = this.resolve({
      userId: args.userId,
      sessionId: session,
      visibility: args.visibility,
    });
    // Omit maxTokens from the runtime call when unset so injected runtimes
    // that inspect key presence stay compatible with pre-maxTokens callers.
    if (args.maxTokens !== undefined) {
      return this.runtime.buildContext({
        query: args.query,
        userId: resolved.userId,
        sessionId: resolved.sessionId,
        visibility: args.visibility,
        metadataFilter: args.metadataFilter,
        enabledSources: args.enabledSources,
        maxTokens: args.maxTokens,
      });
    }
    return this.runtime.buildContext({
      query: args.query,
      userId: resolved.userId,
      sessionId: resolved.sessionId,
      visibility: args.visibility,
      metadataFilter: args.metadataFilter,
      enabledSources: args.enabledSources,
    });
  }

  // ==========================================================================
  // Unified search
  // ==========================================================================

  /** Search one or more memory sources and return a single ranked list. */
  async search(args: {
    query: string;
    sources?:
      | SearchSource
      | string
      | ReadonlyArray<SearchSource | string>
      | null;
    topK?: number;
    userId?: string | null;
    visibility?: string | null;
    sessionId?: string | null;
    domain?: string | null;
    tags?: string[] | null;
    similarityThreshold?: number;
    metadataFilter?: Record<string, unknown> | null;
  }): Promise<MemoryChunk[]> {
    const topK = args.topK ?? 10;
    // sessionId feeds only the episodic leg; suppressing inheritance on this one
    // shared resolve is safe for every leg and stops a bound/ambient session
    // from filtering episodic. An explicit sessionId still filters.
    const resolved = this.resolve({
      userId: args.userId,
      sessionId: args.sessionId,
      visibility: args.visibility,
      suppressInheritedSessionId: true,
    });
    const userId = resolved.userId;
    const sessionId = resolved.sessionId;

    const chunks: MemoryChunk[] = [];
    for (const source of normalizeSources(args.sources)) {
      if (source === SearchSource.SEMANTIC) {
        chunks.push(
          ...(await this.runtime.searchSemantic({
            query: args.query,
            userId,
            visibility: args.visibility,
            topK,
          })),
        );
      } else if (source === SearchSource.EPISODIC) {
        chunks.push(
          ...(await this.runtime.searchEpisodes({
            query: args.query,
            userId,
            visibility: args.visibility,
            sessionId,
            topK,
          })),
        );
      } else if (source === SearchSource.TAXONOMIC) {
        chunks.push(
          ...(await this.runtime.searchTaxonomic({
            query: args.query,
            userId,
            domain: args.domain,
            visibility: args.visibility,
            topK,
          })),
        );
      } else if (source === SearchSource.PROCEDURAL) {
        const procs = await this.runtime.discoverProcedures({
          query: args.query,
          userId,
          visibility: args.visibility,
          tags: args.tags,
          topK,
          similarityThreshold: args.similarityThreshold ?? 0.0,
          metadataFilter: args.metadataFilter,
        });
        chunks.push(...procs.map(procedureToChunk));
      }
    }

    // Sort by similarity_score descending; unscored chunks sort last.
    chunks.sort((a, b) => {
      const aNull =
        a.similarity_score === null || a.similarity_score === undefined;
      const bNull =
        b.similarity_score === null || b.similarity_score === undefined;
      if (aNull !== bNull) {
        return aNull ? 1 : -1;
      }
      return (b.similarity_score ?? 0) - (a.similarity_score ?? 0);
    });
    return chunks.slice(0, topK);
  }
}
