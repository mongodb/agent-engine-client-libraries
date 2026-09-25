/**
 * HTTP client for the Memory Server API.
 *
 * Makes HTTP calls to the memory server, allowing transparent replacement
 * of direct mongomem_core usage.
 */

import { z } from "zod";
import {
  ContextResponseSchema,
  CreateEpisodicResultSchema,
  CreateProceduralResultSchema,
  CreateSemanticResultSchema,
  CreateTaxonomicResultSchema,
  CustomMemoryRetrieveResultSchema,
  CustomMemorySaveResultSchema,
  DeleteResultSchema,
  MemoryChunkSchema,
  WriteTurnResultSchema,
} from "../api/v1/memory.js";
import type {
  ContextResponse,
  CreateEpisodicResult,
  CreateProceduralResult,
  CreateSemanticResult,
  CreateTaxonomicResult,
  CustomMemoryRetrieveResult,
  CustomMemorySaveResult,
  DeleteResult,
  MemoryChunk,
  WriteTurnResult,
} from "../api/v1/memory.js";

/**
 * Trim a session id, returning `null` when it is absent or blank. The memory
 * server accepts a null `session_id` (sessionless context/turns); callers that
 * require one enforce it above this layer.
 */
function normalizeSessionId(
  sessionId: string | null | undefined,
): string | null {
  if (!sessionId || !sessionId.trim()) return null;
  return sessionId.trim();
}

/** Reject non-finite / non-integer / non-positive maxTokens before serialization. */
function requirePositiveMaxTokens(maxTokens: number | undefined): void {
  if (maxTokens === undefined) {
    return;
  }
  if (!Number.isInteger(maxTokens) || maxTokens <= 0) {
    throw new RangeError("maxTokens must be a positive integer");
  }
}

/**
 * Percent-encode an id for use as a single URL path segment, refusing bare
 * dot segments: encodeURIComponent leaves '.'/'..' untouched, and the URL
 * parser would collapse them into a parent-path rewrite of this authenticated
 * request.
 */
function encodeIdSegment(id: string): string {
  if (id === "." || id === "..") {
    throw new Error(
      `Invalid memory id: bare dot segment ${JSON.stringify(id)}`,
    );
  }
  return encodeURIComponent(id);
}

/**
 * Which memory route convention a backend speaks.
 *
 * - `native`: the memory server's own routes (`/stm/turns`, `/retrieval/*`).
 *   Served directly by the memory server and by the OE proxy (which also
 *   accepts the aliased forms). This is the default and preserves the behaviour
 *   of every existing consumer.
 * - `aliased`: the API Gateway's project-scoped memory routes, where the core
 *   loop collapses to `/turns`, `/context`, and a single `/search` that selects
 *   the source via a `type` body field. The Gateway does not expose the native
 *   `/stm` or `/retrieval` paths, so a Gateway-direct client must use this.
 */
export type MemoryRouteStyle = "native" | "aliased";

function requireSessionId(sessionId: string | undefined): string {
  if (!sessionId || !sessionId.trim()) {
    throw new Error("session_id is required for memory server requests");
  }
  return sessionId.trim();
}

/**
 * Thrown for non-OK memory-server responses. Carries the HTTP status and body
 * so callers can branch on failure mode (e.g. map 401/403/404/5xx to typed
 * errors) instead of parsing a message string. Extends `Error`, so existing
 * `catch (e) { ... }` consumers are unaffected.
 */
export class MemoryHttpError extends Error {
  constructor(
    readonly status: number,
    readonly responseText: string,
  ) {
    super(`HTTP ${status}: ${responseText}`);
    this.name = "MemoryHttpError";
  }
}

/** 5xx statuses worth retrying — transient gateway/proxy failures. */
const RETRYABLE_STATUS = new Set([502, 503, 504]);
const RETRY_BASE_DELAY_MS = 200;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * HTTP client for the Memory Server.
 *
 * ```ts
 * const client = new MemoryClient('http://127.0.0.1:8081')
 *
 * // Write conversation turn
 * await client.writeTurn({ sessionId: 'thread_123', role: 'user', content: 'Hello', orgId: 'org_1', userId: 'user_1', projectId: 'proj_1' })
 *
 * // Build context
 * const context = await client.buildContext({ query: 'What did we discuss?', sessionId: 'thread_123', orgId: 'org_1', userId: 'user_1', projectId: 'proj_1' })
 * ```
 */
export class MemoryClient {
  private readonly baseUrl: string;
  private readonly timeout: number;
  private readonly staticHeaders: Record<string, string>;
  private readonly dynamicHeaders: (() => Record<string, string>) | undefined;
  private readonly apiPrefix: string;
  private readonly requestExtras: (() => RequestInit | undefined) | undefined;
  private readonly maxRetries: number;
  private readonly routeStyle: MemoryRouteStyle;
  private readonly fetchImpl: typeof fetch;

  /**
   * @param baseUrl Memory server base URL (e.g. "http://127.0.0.1:8081")
   * @param timeout Request timeout in seconds (default: 30)
   * @param staticHeaders Pod-level identity headers set at construction time
   *   (e.g. `X-Agent-Engine-Agent-Id` from APP_ID). Defensively copied so caller
   *   mutations after construction have no effect.
   * @param apiPrefix URL path prefix for memory endpoints. Defaults to
   *   "/api/v1/memory" (the OE proxy route, which strips `/memory/` before
   *   forwarding to the memory server). Use "/api/v1" when connecting
   *   directly to the memory server.
   * @param dynamicHeaders Called on every request; result is merged over
   *   `staticHeaders`. Used by agent-engine-runner-shared to inject execution-scoped
   *   `X-Agent-Engine-Execution-Id` from AsyncLocalStorage.
   * @param requestExtras Called on every request; the returned `RequestInit` is
   *   spread into `fetch` init before method/headers/signal. Used to inject a
   *   custom transport (e.g. an undici mTLS `dispatcher`) for app-bound calls.
   * @param maxRetries Number of retries on transient 5xx (502/503/504) and
   *   network errors, with exponential backoff. Default 0 (no retry).
   * @param routeStyle Memory route convention (see {@link MemoryRouteStyle}).
   *   Defaults to "native" — the memory server / OE proxy routes. Use "aliased"
   *   for the API Gateway's project-scoped routes (`/turns`, `/context`,
   *   single `/search` with a `type` body field).
   */
  constructor(
    baseUrl: string,
    timeout = 30,
    staticHeaders: Record<string, string> = {},
    apiPrefix = "/api/v1/memory",
    dynamicHeaders?: () => Record<string, string>,
    requestExtras?: () => RequestInit | undefined,
    maxRetries = 0,
    routeStyle: MemoryRouteStyle = "native",
    fetchImpl?: typeof fetch,
  ) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.timeout = timeout;
    this.staticHeaders = { ...staticHeaders };
    this.apiPrefix = "/" + apiPrefix.replace(/^\/+|\/+$/g, "");
    this.dynamicHeaders = dynamicHeaders;
    this.requestExtras = requestExtras;
    this.maxRetries = Math.max(0, maxRetries);
    this.routeStyle = routeStyle;
    // Bind so a passed method keeps its receiver; default to the global fetch.
    this.fetchImpl = fetchImpl ?? fetch;
  }

  /** Route + body for a ranked-search request, per the active route style. */
  private searchRoute(
    type: "semantic" | "episodic" | "taxonomic" | "procedural",
    body: Record<string, unknown>,
  ): { path: string; body: Record<string, unknown> } {
    return this.routeStyle === "aliased"
      ? { path: "/search", body: { type, ...body } }
      : { path: `/retrieval/${type}`, body };
  }

  /** Path for the write-turn route, per the active route style. */
  private get turnsPath(): string {
    return this.routeStyle === "aliased" ? "/turns" : "/stm/turns";
  }

  /** Path for the build-context route, per the active route style. */
  private get contextPath(): string {
    return this.routeStyle === "aliased" ? "/context" : "/retrieval/context";
  }

  /**
   * `fetch` with the configured transport extras and retry on transient 5xx /
   * network errors. All memory-server routes are safe to retry: writes are
   * idempotent (server upserts by label/session) and reads are pure.
   */
  private async fetchWithRetry(
    url: string,
    init: RequestInit,
  ): Promise<Response> {
    for (let attempt = 0; ; attempt++) {
      try {
        const resp = await this.fetchImpl(url, init);
        if (RETRYABLE_STATUS.has(resp.status) && attempt < this.maxRetries) {
          // Drain the discarded body so the connection can be reused.
          await resp.body?.cancel().catch(() => {});
          await sleep(RETRY_BASE_DELAY_MS * 2 ** attempt);
          continue;
        }
        return resp;
      } catch (err) {
        if (attempt >= this.maxRetries) throw err;
        await sleep(RETRY_BASE_DELAY_MS * 2 ** attempt);
      }
    }
  }

  private composeSignal(caller?: AbortSignal): AbortSignal {
    const timeout = AbortSignal.timeout(this.timeout * 1000);
    return caller ? AbortSignal.any([caller, timeout]) : timeout;
  }

  private requestHeaders(): Record<string, string> {
    return this.dynamicHeaders
      ? { ...this.staticHeaders, ...this.dynamicHeaders() }
      : { ...this.staticHeaders };
  }

  private buildUrl(
    path: string,
    params?: Record<string, string | number | boolean | undefined | null>,
  ): string {
    const url = new URL(`${this.baseUrl}${this.apiPrefix}${path}`);
    if (params !== undefined) {
      for (const [k, v] of Object.entries(params)) {
        if (v !== undefined && v !== null) url.searchParams.set(k, String(v));
      }
    }
    return url.toString();
  }

  private buildInit(
    method: string,
    body?: unknown,
    signal?: AbortSignal,
  ): RequestInit {
    const extras = this.requestExtras?.() ?? {};
    const init: RequestInit = {
      ...extras,
      method,
      headers: {
        ...((extras.headers as Record<string, string> | undefined) ?? {}),
        ...this.requestHeaders(),
      },
      signal: this.composeSignal(signal),
    };
    if (body !== undefined) {
      (init.headers as Record<string, string>)["Content-Type"] =
        "application/json";
      init.body = JSON.stringify(body);
    }
    return init;
  }

  private async request<T>(
    method: string,
    path: string,
    schema: z.ZodType<T>,
    body?: unknown,
    params?: Record<string, string | number | boolean | undefined | null>,
    signal?: AbortSignal,
  ): Promise<T> {
    const resp = await this.fetchWithRetry(
      this.buildUrl(path, params),
      this.buildInit(method, body, signal),
    );
    if (!resp.ok) throw new MemoryHttpError(resp.status, await resp.text());
    return schema.parse(await resp.json());
  }

  /**
   * Like `request`, but returns `null` on a 404 instead of throwing. Mirrors
   * Python's `if resp.status_code == 404: return None` on the list/lookup GETs
   * (`get_semantic`, `get_taxonomic`, `get_procedural`), so a not-found lookup
   * resolves to `null` rather than surfacing as an HTTP error.
   */
  private async requestOrNull<T>(
    method: string,
    path: string,
    schema: z.ZodType<T>,
    body?: unknown,
    params?: Record<string, string | number | boolean | undefined | null>,
    signal?: AbortSignal,
  ): Promise<T | null> {
    const resp = await this.fetchWithRetry(
      this.buildUrl(path, params),
      this.buildInit(method, body, signal),
    );
    if (resp.status === 404) return null;
    if (!resp.ok) throw new MemoryHttpError(resp.status, await resp.text());
    return schema.parse(await resp.json());
  }

  // ===========================================================================
  // Short-Term Memory (STM)
  // ===========================================================================

  async writeTurn(params: {
    sessionId?: string | null;
    role: string;
    orgId: string;
    userId?: string | null;
    projectId: string;
    content?: string | null;
    tokens?: number | null;
    stopReason?: string | null;
    modelName?: string | null;
    toolCalls?: Array<Record<string, unknown>> | null;
    toolCallId?: string | null;
    toolName?: string | null;
    isError?: boolean;
    // Server-side dedupe key. Sent when present so a retried turn write (below
    // this call or via `maxRetries`) commits at most once.
    idempotencyKey?: string | null;
    signal?: AbortSignal;
  }): Promise<WriteTurnResult> {
    const body: Record<string, unknown> = {
      session_id: normalizeSessionId(params.sessionId),
      role: params.role,
      org_id: params.orgId,
      user_id: params.userId ?? null,
      project_id: params.projectId,
      content: params.content,
      tokens: params.tokens,
      stop_reason: params.stopReason,
      model_name: params.modelName,
      tool_calls: params.toolCalls,
      tool_call_id: params.toolCallId,
      tool_name: params.toolName,
      // Default to false so the field is always present on the wire, matching
      // Python's `is_error: bool = False` (TS would otherwise drop an
      // `undefined` key via JSON.stringify).
      is_error: params.isError ?? false,
    };
    if (params.idempotencyKey != null)
      body.idempotency_key = params.idempotencyKey;
    return this.request(
      "POST",
      this.turnsPath,
      WriteTurnResultSchema,
      body,
      undefined,
      params.signal,
    );
  }

  // ===========================================================================
  // Context Building
  // ===========================================================================

  /**
   * Build unified memory context.
   * Omitted `enabledSources` defaults to episodic and semantic; include
   * "stm" for recent turns.
   */
  async buildContext(params: {
    query: string;
    sessionId?: string | null;
    orgId: string;
    userId?: string | null;
    projectId: string;
    visibility?: string | null;
    metadataFilter?: Record<string, unknown> | null;
    enabledSources?: string[] | null;
    formatStyle?: string | null;
    topK?: number;
    /**
     * Optional gross context-construction budget, serialized as `max_tokens`
     * only when set. After retrieval and ranking, the server subtracts a
     * 500-token formatting reserve, then greedily selects whole memory chunks
     * that fit in the remainder. Positive values at or below 500 leave no
     * budget for memories. Values above 500 can still yield empty
     * context when no chunk fits. Non-positive, non-integer, and non-finite
     * values raise before the request is sent.
     */
    maxTokens?: number;
    signal?: AbortSignal;
  }): Promise<ContextResponse> {
    requirePositiveMaxTokens(params.maxTokens);
    const body: Record<string, unknown> = {
      query: params.query,
      session_id: normalizeSessionId(params.sessionId),
      org_id: params.orgId,
      user_id: params.userId ?? null,
      visibility: params.visibility,
      project_id: params.projectId,
      enabled_sources: params.enabledSources,
      format_style: params.formatStyle,
      top_k: params.topK ?? 50,
    };
    // Match Python's `if metadata_filter:` (truthy) — an empty object is not
    // sent, so the server applies its default unfiltered behaviour.
    if (
      params.metadataFilter != null &&
      Object.keys(params.metadataFilter).length > 0
    )
      body["metadata_filter"] = params.metadataFilter;
    if (params.maxTokens !== undefined) body["max_tokens"] = params.maxTokens;
    return this.request(
      "POST",
      this.contextPath,
      ContextResponseSchema,
      body,
      undefined,
      params.signal,
    );
  }

  // ===========================================================================
  // Semantic Memory
  // ===========================================================================

  async createSemantic(params: {
    label: string;
    text: string;
    orgId: string;
    userId: string;
    projectId: string;
    source?: string;
    visibility?: string;
    agentId?: string | null;
    embedding?: number[] | null;
    metadata?: Record<string, unknown> | null;
    upsert?: boolean;
    signal?: AbortSignal;
  }): Promise<CreateSemanticResult> {
    const body = {
      label: params.label,
      text: params.text,
      org_id: params.orgId,
      user_id: params.userId,
      source: params.source ?? "agent",
      visibility: params.visibility ?? "private",
      project_id: params.projectId,
      agent_id: params.agentId,
      embedding: params.embedding,
      metadata: params.metadata,
      // Server-side upsert flag (Python `create_semantic(..., upsert=...)`,
      // default False at the client layer). When true, the server updates an
      // existing memory with the same label instead of erroring on duplicate.
      upsert: params.upsert ?? false,
    };
    return this.request(
      "POST",
      "/semantic",
      CreateSemanticResultSchema,
      body,
      undefined,
      params.signal,
    );
  }

  async fetchSemanticMemories(params: {
    query: string;
    orgId: string;
    projectId: string;
    userId?: string | null;
    visibility?: string | null;
    topK?: number;
    signal?: AbortSignal;
  }): Promise<MemoryChunk[]> {
    const route = this.searchRoute("semantic", {
      query: params.query,
      org_id: params.orgId,
      project_id: params.projectId,
      user_id: params.userId,
      visibility: params.visibility,
      top_k: params.topK ?? 50,
    });
    const data = await this.request(
      "POST",
      route.path,
      z.object({ memories: z.array(MemoryChunkSchema) }),
      route.body,
      undefined,
      params.signal,
    );
    return data.memories;
  }

  async getSemantic(params: {
    orgId: string;
    projectId: string;
    label?: string | null;
    id?: string | null;
    userId?: string | null;
    visibility?: string | null;
    signal?: AbortSignal;
  }): Promise<unknown | null> {
    if (params.id != null) {
      let idUrl =
        `${this.baseUrl}${this.apiPrefix}/semantic/${encodeIdSegment(String(params.id))}` +
        `?org_id=${encodeURIComponent(params.orgId)}&project_id=${encodeURIComponent(params.projectId)}`;
      if (params.userId != null)
        idUrl += `&user_id=${encodeURIComponent(params.userId)}`;
      if (params.visibility != null)
        idUrl += `&visibility=${encodeURIComponent(params.visibility)}`;
      const resp = await this.fetchWithRetry(
        idUrl,
        this.buildInit("GET", undefined, params.signal),
      );
      if (resp.status === 404) return null;
      if (!resp.ok) throw new MemoryHttpError(resp.status, await resp.text());
      return (await resp.json()) as unknown;
    }

    const urlParams: Record<string, string> = {
      org_id: params.orgId,
      project_id: params.projectId,
    };
    if (params.userId != null) urlParams["user_id"] = params.userId;
    if (params.visibility != null) urlParams["visibility"] = params.visibility;
    if (params.label != null) urlParams["label"] = params.label;
    const data = await this.requestOrNull(
      "GET",
      "/semantic",
      z.object({ entries: z.array(z.unknown()) }),
      undefined,
      urlParams,
      params.signal,
    );
    if (data == null) return null;
    return data.entries.length > 0 ? data.entries[0] : null;
  }

  async updateSemantic(params: {
    orgId: string;
    projectId: string;
    label: string;
    text?: string | null;
    source?: string | null;
    visibility?: string | null;
    signal?: AbortSignal;
  }): Promise<unknown> {
    const existing = await this.getSemantic({
      orgId: params.orgId,
      projectId: params.projectId,
      label: params.label,
      signal: params.signal,
    });
    if (existing == null)
      throw new Error(`Semantic memory not found: ${params.label}`);
    const memoryId =
      (existing as Record<string, unknown>)["id"] ??
      (existing as Record<string, unknown>)["_id"];
    if (memoryId == null)
      throw new Error(`Semantic memory has no id field: ${params.label}`);
    const body = {
      org_id: params.orgId,
      project_id: params.projectId,
      text: params.text,
      source: params.source,
      visibility: params.visibility,
    };
    return this.request(
      "PATCH",
      `/semantic/${encodeIdSegment(String(memoryId))}`,
      z.unknown(),
      body,
      undefined,
      params.signal,
    );
  }

  // ===========================================================================
  // Episodic Memory
  // ===========================================================================

  async createEpisodic(params: {
    title: string;
    content: string;
    summaryText: string;
    orgId: string;
    userId: string;
    sessionId: string;
    projectId: string;
    visibility?: string;
    agentId?: string | null;
    snapshotRefId?: string | null;
    summaryType?: string | null;
    sourceAgent?: string | null;
    participants?: string[] | null;
    tags?: string[] | null;
    metadata?: Record<string, unknown> | null;
    extractionSource?: string | null;
    embedding?: number[] | null;
    skipEmbedding?: boolean | null;
    signal?: AbortSignal;
  }): Promise<CreateEpisodicResult> {
    const body: Record<string, unknown> = {
      title: params.title,
      content: params.content,
      summary_text: params.summaryText,
      org_id: params.orgId,
      user_id: params.userId,
      session_id: requireSessionId(params.sessionId),
      visibility: params.visibility ?? "private",
      project_id: params.projectId,
      agent_id: params.agentId,
    };
    if (params.metadata != null) body["metadata"] = params.metadata;
    if (params.snapshotRefId != null)
      body["snapshot_ref_id"] = params.snapshotRefId;
    if (params.summaryType != null) body["summary_type"] = params.summaryType;
    if (params.sourceAgent != null) body["source_agent"] = params.sourceAgent;
    if (params.participants != null) body["participants"] = params.participants;
    if (params.tags != null) body["tags"] = params.tags;
    if (params.extractionSource != null)
      body["extraction_source"] = params.extractionSource;
    if (params.embedding != null) body["embedding"] = params.embedding;
    if (params.skipEmbedding != null)
      body["skip_embedding"] = params.skipEmbedding;
    return this.request(
      "POST",
      "/episodic",
      CreateEpisodicResultSchema,
      body,
      undefined,
      params.signal,
    );
  }

  async fetchEpisodicMemories(params: {
    query: string;
    orgId: string;
    projectId: string;
    userId?: string | null;
    visibility?: string | null;
    sessionId?: string | null;
    topK?: number;
    signal?: AbortSignal;
  }): Promise<MemoryChunk[]> {
    const route = this.searchRoute("episodic", {
      query: params.query,
      org_id: params.orgId,
      project_id: params.projectId,
      user_id: params.userId,
      visibility: params.visibility,
      session_id: params.sessionId,
      top_k: params.topK ?? 50,
    });
    const data = await this.request(
      "POST",
      route.path,
      z.object({ memories: z.array(MemoryChunkSchema) }),
      route.body,
      undefined,
      params.signal,
    );
    return data.memories;
  }

  async listEpisodic(params: {
    orgId: string;
    projectId: string;
    userId?: string | null;
    sessionId?: string | null;
    visibility?: string | null;
    limit?: number;
    signal?: AbortSignal;
  }): Promise<unknown[]> {
    const data = await this.request(
      "GET",
      "/episodic",
      z.object({ entries: z.array(z.unknown()) }),
      undefined,
      {
        org_id: params.orgId,
        project_id: params.projectId,
        user_id: params.userId ?? undefined,
        session_id: params.sessionId ?? undefined,
        visibility: params.visibility ?? undefined,
        limit: params.limit ?? 20,
      },
      params.signal,
    );
    return data.entries;
  }

  // ===========================================================================
  // Taxonomic Memory
  // ===========================================================================

  async createTaxonomic(params: {
    domain: string;
    term: string;
    definition: string;
    orgId: string;
    userId: string;
    projectId: string;
    relatedTerms?: string[] | null;
    queryExpansion?: boolean;
    visibility?: string;
    signal?: AbortSignal;
  }): Promise<CreateTaxonomicResult> {
    const body = {
      domain: params.domain,
      term: params.term,
      definition: params.definition,
      org_id: params.orgId,
      user_id: params.userId,
      project_id: params.projectId,
      related_terms: params.relatedTerms,
      query_expansion: params.queryExpansion ?? true,
      visibility: params.visibility ?? "org",
    };
    return this.request(
      "POST",
      "/taxonomic",
      CreateTaxonomicResultSchema,
      body,
      undefined,
      params.signal,
    );
  }

  async fetchTaxonomicMemories(params: {
    query: string;
    orgId: string;
    projectId: string;
    domain?: string | null;
    visibility?: string | null;
    topK?: number;
    // Accepted for API-surface parity with Python and intentionally NOT wired:
    // Python's fetch_taxonomic_memories absorbs user_id into **kwargs and never
    // puts it in the body — taxonomic memories are org-scoped, so the server
    // ignores it. We mirror that dead param explicitly here.
    userId?: string | null;
    signal?: AbortSignal;
  }): Promise<MemoryChunk[]> {
    // user_id is deliberately omitted from the body (see param doc above).
    const route = this.searchRoute("taxonomic", {
      query: params.query,
      org_id: params.orgId,
      project_id: params.projectId,
      domain: params.domain,
      visibility: params.visibility,
      top_k: params.topK ?? 50,
    });
    const data = await this.request(
      "POST",
      route.path,
      z.object({ memories: z.array(MemoryChunkSchema) }),
      route.body,
      undefined,
      params.signal,
    );
    return data.memories;
  }

  async getTaxonomic(params: {
    orgId: string;
    projectId: string;
    domain?: string | null;
    term?: string | null;
    userId?: string | null;
    visibility?: string | null;
    signal?: AbortSignal;
  }): Promise<unknown | null> {
    const data = await this.requestOrNull(
      "GET",
      "/taxonomic",
      z.object({ entries: z.array(z.unknown()) }),
      undefined,
      {
        org_id: params.orgId,
        project_id: params.projectId,
        domain: params.domain ?? undefined,
        user_id: params.userId ?? undefined,
        visibility: params.visibility ?? undefined,
      },
      params.signal,
    );
    if (data == null) return null;
    if (params.term != null) {
      return (
        data.entries.find(
          (e) => (e as Record<string, unknown>)["term"] === params.term,
        ) ?? null
      );
    }
    return data.entries.length > 0 ? data.entries[0] : null;
  }

  async getDistinctDomains(params: {
    orgId: string;
    projectId: string;
    visibility?: string | null;
    signal?: AbortSignal;
  }): Promise<string[]> {
    const data = await this.request(
      "GET",
      "/taxonomic/domains",
      z.object({ domains: z.array(z.string()) }),
      undefined,
      {
        org_id: params.orgId,
        project_id: params.projectId,
        visibility: params.visibility ?? undefined,
      },
      params.signal,
    );
    return data.domains;
  }

  // ===========================================================================
  // Procedural Memory
  // ===========================================================================

  async createProcedural(params: {
    procedure: string;
    description: string;
    content: string;
    orgId: string;
    userId: string;
    projectId: string;
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
    signal?: AbortSignal;
  }): Promise<CreateProceduralResult> {
    const body = {
      procedure: params.procedure,
      description: params.description,
      content: params.content,
      org_id: params.orgId,
      user_id: params.userId,
      steps: params.steps,
      resources: params.resources,
      allowed_tools: params.allowedTools,
      compatibility: params.compatibility,
      license: params.license,
      trigger_conditions: params.triggerConditions,
      tags: params.tags,
      visibility: params.visibility ?? "private",
      project_id: params.projectId,
      agent_id: params.agentId,
      extraction_source: params.extractionSource,
      source_format: params.sourceFormat,
      source_path: params.sourcePath,
    };
    return this.request(
      "POST",
      "/procedural",
      CreateProceduralResultSchema,
      body,
      undefined,
      params.signal,
    );
  }

  async getProcedural(params: {
    orgId: string;
    projectId: string;
    id?: string | null;
    procedure?: string | null;
    userId?: string | null;
    visibility?: string | null;
    includeDeleted?: boolean;
    signal?: AbortSignal;
  }): Promise<unknown | null> {
    if (params.id != null) {
      let idUrl =
        `${this.baseUrl}${this.apiPrefix}/procedural/${encodeIdSegment(String(params.id))}` +
        `?org_id=${encodeURIComponent(params.orgId)}&project_id=${encodeURIComponent(params.projectId)}` +
        `&include_deleted=${params.includeDeleted ?? false}`;
      if (params.userId != null)
        idUrl += `&user_id=${encodeURIComponent(params.userId)}`;
      if (params.visibility != null)
        idUrl += `&visibility=${encodeURIComponent(params.visibility)}`;
      const resp = await this.fetchWithRetry(
        idUrl,
        this.buildInit("GET", undefined, params.signal),
      );
      if (resp.status === 404) return null;
      if (!resp.ok) throw new MemoryHttpError(resp.status, await resp.text());
      return (await resp.json()) as unknown;
    }

    const urlParams: Record<string, string | boolean> = {
      org_id: params.orgId,
      project_id: params.projectId,
      include_deleted: params.includeDeleted ?? false,
    };
    if (params.userId != null) urlParams["user_id"] = params.userId;
    if (params.visibility != null) urlParams["visibility"] = params.visibility;
    if (params.procedure != null) urlParams["procedure"] = params.procedure;
    const data = await this.requestOrNull(
      "GET",
      "/procedural",
      z.object({ entries: z.array(z.unknown()) }),
      undefined,
      urlParams,
      params.signal,
    );
    if (data == null) return null;
    if (params.procedure != null) {
      return (
        data.entries.find(
          (e) =>
            (e as Record<string, unknown>)["procedure"] === params.procedure,
        ) ?? null
      );
    }
    return data.entries.length > 0 ? data.entries[0] : null;
  }

  async updateProcedural(params: {
    orgId: string;
    projectId: string;
    id?: string | null;
    procedure?: string | null;
    description?: string | null;
    content?: string | null;
    steps?: Array<Record<string, unknown>> | null;
    resources?: Array<Record<string, unknown>> | null;
    allowedTools?: string[] | null;
    triggerConditions?: string[] | null;
    tags?: string[] | null;
    visibility?: string | null;
    signal?: AbortSignal;
  }): Promise<unknown> {
    let memoryId = params.id;
    if (memoryId == null && params.procedure != null) {
      const existing = await this.getProcedural({
        orgId: params.orgId,
        projectId: params.projectId,
        procedure: params.procedure,
        signal: params.signal,
      });
      if (existing == null)
        throw new Error(`Procedural memory not found: ${params.procedure}`);
      const rawId =
        (existing as Record<string, unknown>)["id"] ??
        (existing as Record<string, unknown>)["_id"];
      if (rawId == null)
        throw new Error(
          `Procedural memory has no id field: ${params.procedure}`,
        );
      memoryId = String(rawId);
    }
    if (memoryId == null)
      throw new Error(
        "id or procedure is required to update a procedural memory",
      );
    const body = {
      org_id: params.orgId,
      description: params.description,
      content: params.content,
      steps: params.steps,
      resources: params.resources,
      allowed_tools: params.allowedTools,
      trigger_conditions: params.triggerConditions,
      tags: params.tags,
      visibility: params.visibility,
      project_id: params.projectId,
    };
    return this.request(
      "PATCH",
      `/procedural/${encodeIdSegment(String(memoryId))}`,
      z.unknown(),
      body,
      undefined,
      params.signal,
    );
  }

  async deleteProcedural(params: {
    orgId: string;
    projectId: string;
    id?: string | null;
    procedure?: string | null;
    soft?: boolean;
    signal?: AbortSignal;
  }): Promise<DeleteResult> {
    let memoryId = params.id;
    if (memoryId == null && params.procedure != null) {
      const existing = await this.getProcedural({
        orgId: params.orgId,
        projectId: params.projectId,
        procedure: params.procedure,
        signal: params.signal,
      });
      if (existing == null)
        throw new Error(`Procedural memory not found: ${params.procedure}`);
      const rawId =
        (existing as Record<string, unknown>)["id"] ??
        (existing as Record<string, unknown>)["_id"];
      if (rawId == null)
        throw new Error(
          `Procedural memory has no id field: ${params.procedure}`,
        );
      memoryId = String(rawId);
    }
    if (memoryId == null)
      throw new Error(
        "id or procedure is required to delete a procedural memory",
      );
    return this.request(
      "DELETE",
      `/procedural/${encodeIdSegment(String(memoryId))}`,
      DeleteResultSchema,
      undefined,
      {
        org_id: params.orgId,
        project_id: params.projectId,
        soft: params.soft ?? true,
      },
      params.signal,
    );
  }

  async discoverProcedures(params: {
    query: string;
    orgId: string;
    projectId: string;
    userId?: string | null;
    visibility?: string | null;
    tags?: string[] | null;
    topK?: number;
    similarityThreshold?: number;
    metadataFilter?: Record<string, unknown> | null;
    signal?: AbortSignal;
  }): Promise<Array<Record<string, unknown>>> {
    const body: Record<string, unknown> = {
      query: params.query,
      org_id: params.orgId,
      user_id: params.userId,
      visibility: params.visibility,
      project_id: params.projectId,
      tags: params.tags,
      top_k: params.topK ?? 10,
      similarity_threshold: params.similarityThreshold ?? 0.0,
    };
    // Match Python's `if metadata_filter:` (truthy) — skip an empty object.
    if (
      params.metadataFilter != null &&
      Object.keys(params.metadataFilter).length > 0
    )
      body["metadata_filter"] = params.metadataFilter;
    const route = this.searchRoute("procedural", body);
    const data = await this.request(
      "POST",
      route.path,
      z.object({ memories: z.array(z.record(z.string(), z.unknown())) }),
      route.body,
      undefined,
      params.signal,
    );
    return data.memories.map((m) => ({
      procedure:
        (m["metadata"] as Record<string, unknown>)?.["procedure"] ?? "",
      content: m["content"] ?? "",
      score: m["similarity_score"] ?? 0.0,
      ...(m["metadata"] as Record<string, unknown>),
    }));
  }

  async fetchProceduralMemories(params: {
    query: string;
    orgId: string;
    projectId: string;
    userId?: string | null;
    visibility?: string | null;
    tags?: string[] | null;
    topK?: number;
    signal?: AbortSignal;
  }): Promise<MemoryChunk[]> {
    const route = this.searchRoute("procedural", {
      query: params.query,
      org_id: params.orgId,
      user_id: params.userId,
      visibility: params.visibility,
      project_id: params.projectId,
      tags: params.tags,
      top_k: params.topK ?? 50,
    });
    const data = await this.request(
      "POST",
      route.path,
      z.object({ memories: z.array(MemoryChunkSchema) }),
      route.body,
      undefined,
      params.signal,
    );
    return data.memories;
  }

  // ===========================================================================
  // Custom Memory Types
  // ===========================================================================

  async createCustom(params: {
    memoryType: string;
    content: string;
    tags?: Record<string, unknown> | null;
    contextualMetadata?: Record<string, unknown> | null;
    signal?: AbortSignal;
  }): Promise<CustomMemorySaveResult> {
    const body: Record<string, unknown> = { content: params.content };
    if (params.tags != null) body.tags = params.tags;
    if (params.contextualMetadata != null) {
      body.contextual_metadata = params.contextualMetadata;
    }
    return this.request(
      "POST",
      `/types/${encodeIdSegment(params.memoryType)}`,
      CustomMemorySaveResultSchema,
      body,
      undefined,
      params.signal,
    );
  }

  async retrieveCustom(params: {
    memoryType: string;
    query: string;
    tags?: Record<string, unknown> | null;
    topK?: number;
    signal?: AbortSignal;
  }): Promise<CustomMemoryRetrieveResult> {
    const body: Record<string, unknown> = {
      query: params.query,
      top_k: params.topK ?? 10,
    };
    if (params.tags != null) body.tags = params.tags;
    return this.request(
      "POST",
      `/types/${encodeIdSegment(params.memoryType)}/retrieve`,
      CustomMemoryRetrieveResultSchema,
      body,
      undefined,
      params.signal,
    );
  }

  // ===========================================================================
  // Bootstrap
  // ===========================================================================

  bootstrap(): void {
    // No-op for HTTP client — server handles bootstrap.
  }
}
