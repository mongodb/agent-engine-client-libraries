/**
 * `MemoryRuntime` + `MemoryCrudClient` implemented over agent-engine-sdk's
 * `MemoryClient`.
 *
 * The facade needs two seams (the workflow runtime and the CRUD conveniences);
 * `MemoryClient` already owns the HTTP mechanics both share — URL building, the
 * transport-extras/dynamic-header hooks, the bounded 5xx/network retry, and the
 * 404→null lookup semantics. Rather than reimplement that stack, this adapter is
 * a thin binding: it fills the empty tenancy the backends stamp (`org_id`/
 * `project_id` come from the trusted OE execution context or the Gateway API
 * key), selects the route convention per backend, and translates
 * `MemoryClient`'s `MemoryHttpError` into the memory package's typed error
 * hierarchy (which is this package's public API).
 *
 * Two `MemoryClient` instances back each adapter: a retrying one for the write
 * turn and all reads (idempotent — turns carry an idempotency key, reads are
 * pure) and a non-retrying one for type-specific creates (no idempotency key, so
 * a retry after the server committed could duplicate the record).
 */

import {
  MemoryClient,
  MemoryHttpError,
  type MemoryRouteStyle,
} from "@mongodb-js/agent-engine-sdk";

import {
  MemoryAuthError,
  MemoryBadRequestError,
  MemoryConnectionError,
  MemoryNotProvisionedError,
  MemoryNotSupportedError,
  MemoryRouteNotFoundError,
  MemoryServerError,
} from "../errors.js";
import {
  ContextResponseSchema,
  CreateEpisodicResultSchema,
  CreateProceduralResultSchema,
  CreateSemanticResultSchema,
  CreateTaxonomicResultSchema,
  CustomMemoryRetrieveResultSchema,
  CustomMemorySaveResultSchema,
  MemoryChunkSchema,
  WriteTurnResultSchema,
  type ContextResponse,
  type CreateEpisodicResult,
  type CreateProceduralResult,
  type CreateSemanticResult,
  type CreateTaxonomicResult,
  type CustomMemoryRetrieveResult,
  type CustomMemorySaveResult,
  type MemoryChunk,
  type WriteTurnResult,
} from "../models.js";
import type {
  FetchLike,
  MemoryCrudClient,
  MemoryRuntime,
  RequestExtras,
} from "../transport.js";

/** Error codes the backends use to signal the memory runtime is not yet reachable. */
const NOT_PROVISIONED_CODES = new Set([
  "AGENT_NOT_DEPLOYED",
  "NO_WORKSPACE_ENDPOINT",
  "PROJECT_RUNTIME_FAILED",
]);

const RETRIES = 2; // initial + 2 retries, matching the previous transport

/** Empty tenancy: the backend stamps the trusted org/project (caller cannot override). */
const EMPTY_ORG = "";
const EMPTY_PROJECT = "";

export interface MemoryClientAdapterOptions {
  /** Backend base URL (Gateway or OE). */
  baseUrl: string;
  /**
   * Memory path prefix passed through to `MemoryClient` — e.g.
   * `/api/v1/memory` (OE) or `/api/v1/projects/<id>/memory` (Gateway).
   */
  apiPrefix: string;
  /** Route convention the backend speaks. */
  routeStyle: MemoryRouteStyle;
  /**
   * Whether the selected prefix is project-scoped. Used only to phrase the
   * directional hint on a core-loop 404 (route-shape mismatch).
   */
  projectScoped: boolean;
  /** Static auth/identity headers (e.g. `Authorization`, `X-Agent-Engine-Execution-Id`). */
  headers?: Record<string, string>;
  /** Injectable fetch, primarily for tests. */
  fetchImpl?: FetchLike;
  /** Per-request transport extras (e.g. an mTLS dispatcher for app-bound calls). */
  requestExtras?: RequestExtras;
  /** Request timeout in seconds (default 30). */
  timeout?: number;
}

/** A single `MemoryClient`-backed runtime + CRUD client for the `Memory` facade. */
export class MemoryClientAdapter implements MemoryRuntime, MemoryCrudClient {
  private readonly retry: MemoryClient;
  private readonly noRetry: MemoryClient;
  private readonly projectScoped: boolean;

  constructor(opts: MemoryClientAdapterOptions) {
    this.projectScoped = opts.projectScoped;
    const timeout = opts.timeout ?? 30;
    const make = (maxRetries: number): MemoryClient =>
      new MemoryClient(
        opts.baseUrl,
        timeout,
        opts.headers ?? {},
        opts.apiPrefix,
        undefined,
        opts.requestExtras,
        maxRetries,
        opts.routeStyle,
        opts.fetchImpl as typeof fetch | undefined,
      );
    this.retry = make(RETRIES);
    this.noRetry = make(0);
  }

  // ---------------------------------------------------------------------------
  // Error translation
  // ---------------------------------------------------------------------------

  /** Map a `MemoryClient` failure to this package's typed error hierarchy. */
  private mapError(err: unknown, coreLoop: boolean): Error {
    if (!(err instanceof MemoryHttpError)) {
      // A JSON-parse (SyntaxError from resp.json()) or schema-validation
      // (ZodError from schema.parse) failure means the backend answered but the
      // success body was unparseable/invalid — a server-side contract violation,
      // not a transport failure. Mirror the Python SDK and surface it as a
      // MemoryServerError rather than a MemoryConnectionError.
      if (
        err instanceof SyntaxError ||
        (err instanceof Error && err.name === "ZodError")
      ) {
        return new MemoryServerError(String(err), { status: null });
      }
      // fetch/timeout — MemoryClient rethrows the raw error after exhausting retries.
      return new MemoryConnectionError(String(err), { status: null });
    }
    const { status, responseText } = err;
    let code: string | null = null;
    let message = responseText;
    try {
      const parsed: unknown = JSON.parse(responseText);
      if (
        parsed !== null &&
        typeof parsed === "object" &&
        !Array.isArray(parsed)
      ) {
        const body = parsed as Record<string, unknown>;
        code = (body.code as string) || null;
        // `detail` is FastAPI's convention and is what the memory server
        // returns; without it a structured message surfaces as raw JSON. Only
        // strings qualify — FastAPI validation errors put a list there.
        const detail =
          typeof body.detail === "string" ? body.detail : undefined;
        message =
          (body.error as string) ||
          (body.message as string) ||
          detail ||
          responseText;
      }
    } catch {
      // Non-JSON body: keep the raw text as the message.
    }
    const opts = { status, code, responseText };
    if (status === 401 || status === 403)
      return new MemoryAuthError(message, opts);
    if (code !== null && NOT_PROVISIONED_CODES.has(code))
      return new MemoryNotProvisionedError(message, opts);
    if (status === 404 && coreLoop) return this.routeNotFoundError(opts);
    if (status >= 400 && status < 500)
      return new MemoryBadRequestError(message, opts);
    return new MemoryServerError(message, opts);
  }

  private routeNotFoundError(opts: {
    status: number;
    code: string | null;
    responseText: string;
  }): MemoryRouteNotFoundError {
    const hint = this.projectScoped
      ? "projectId is set, so the SDK used a project-scoped memory route; if you " +
        "are targeting a local or direct OE backend, unset projectId (or " +
        "AGENTIC_MEMORY_PROJECT_ID)"
      : "projectId is empty, so the SDK used a flat memory route; if you are " +
        "targeting the hosted Gateway, set projectId (or AGENTIC_MEMORY_PROJECT_ID)";
    return new MemoryRouteNotFoundError(
      "memory route not found (404). This often means the route shape does not " +
        `match the backend: ${hint}. It can also mean baseUrl points at the ` +
        "wrong host.",
      opts,
    );
  }

  // ---------------------------------------------------------------------------
  // MemoryRuntime
  // ---------------------------------------------------------------------------

  async recordTurn(
    args: Parameters<MemoryRuntime["recordTurn"]>[0],
  ): Promise<WriteTurnResult> {
    try {
      const r = await this.retry.writeTurn({
        sessionId: args.sessionId,
        role: args.role,
        orgId: EMPTY_ORG,
        userId: args.userId,
        projectId: EMPTY_PROJECT,
        content: args.content,
        modelName: args.modelName,
        toolCalls: args.toolCalls,
        toolCallId: args.toolCallId,
        toolName: args.toolName,
        isError: args.isError,
        idempotencyKey: args.idempotencyKey,
      });
      return WriteTurnResultSchema.parse(r);
    } catch (e) {
      throw this.mapError(e, true);
    }
  }

  async buildContext(
    args: Parameters<MemoryRuntime["buildContext"]>[0],
  ): Promise<ContextResponse> {
    try {
      const r = await this.retry.buildContext({
        query: args.query,
        sessionId: args.sessionId,
        orgId: EMPTY_ORG,
        userId: args.userId,
        projectId: EMPTY_PROJECT,
        visibility: args.visibility,
        metadataFilter: args.metadataFilter,
        // The facade carries enabled sources as a Set; MemoryClient takes a
        // sorted array (deterministic wire order).
        enabledSources: args.enabledSources
          ? [...args.enabledSources].sort()
          : null,
        topK: args.topK,
        maxTokens: args.maxTokens,
      });
      return ContextResponseSchema.parse(r);
    } catch (e) {
      throw this.mapError(e, true);
    }
  }

  async searchSemantic(
    args: Parameters<MemoryRuntime["searchSemantic"]>[0],
  ): Promise<MemoryChunk[]> {
    try {
      const rows = await this.retry.fetchSemanticMemories({
        query: args.query,
        orgId: EMPTY_ORG,
        projectId: EMPTY_PROJECT,
        userId: args.userId,
        visibility: args.visibility,
        topK: args.topK,
      });
      return rows.map((m) => MemoryChunkSchema.parse(m));
    } catch (e) {
      throw this.mapError(e, true);
    }
  }

  async searchEpisodes(
    args: Parameters<MemoryRuntime["searchEpisodes"]>[0],
  ): Promise<MemoryChunk[]> {
    try {
      const rows = await this.retry.fetchEpisodicMemories({
        query: args.query,
        orgId: EMPTY_ORG,
        projectId: EMPTY_PROJECT,
        userId: args.userId,
        visibility: args.visibility,
        sessionId: args.sessionId,
        topK: args.topK,
      });
      return rows.map((m) => MemoryChunkSchema.parse(m));
    } catch (e) {
      throw this.mapError(e, true);
    }
  }

  async searchTaxonomic(
    args: Parameters<MemoryRuntime["searchTaxonomic"]>[0],
  ): Promise<MemoryChunk[]> {
    try {
      const rows = await this.retry.fetchTaxonomicMemories({
        query: args.query,
        orgId: EMPTY_ORG,
        projectId: EMPTY_PROJECT,
        domain: args.domain,
        visibility: args.visibility,
        topK: args.topK,
      });
      return rows.map((m) => MemoryChunkSchema.parse(m));
    } catch (e) {
      throw this.mapError(e, true);
    }
  }

  async discoverProcedures(
    args: Parameters<MemoryRuntime["discoverProcedures"]>[0],
  ): Promise<Array<Record<string, unknown>>> {
    try {
      return await this.retry.discoverProcedures({
        query: args.query,
        orgId: EMPTY_ORG,
        projectId: EMPTY_PROJECT,
        userId: args.userId,
        visibility: args.visibility,
        tags: args.tags,
        topK: args.topK,
        similarityThreshold: args.similarityThreshold,
        metadataFilter: args.metadataFilter,
      });
    } catch (e) {
      throw this.mapError(e, true);
    }
  }

  // ---------------------------------------------------------------------------
  // MemoryCrudClient
  // ---------------------------------------------------------------------------

  async createSemantic(
    args: Parameters<MemoryCrudClient["createSemantic"]>[0],
  ): Promise<CreateSemanticResult> {
    try {
      const r = await this.noRetry.createSemantic({
        label: args.label,
        text: args.text,
        orgId: EMPTY_ORG,
        userId: args.userId,
        projectId: EMPTY_PROJECT,
        source: args.source,
        visibility: args.visibility,
        agentId: args.agentId,
        metadata: args.metadata,
        upsert: args.upsert,
      });
      return CreateSemanticResultSchema.parse(r);
    } catch (e) {
      throw this.mapError(e, false);
    }
  }

  async getSemantic(
    args: Parameters<MemoryCrudClient["getSemantic"]>[0],
  ): Promise<unknown | null> {
    try {
      return await this.retry.getSemantic({
        orgId: EMPTY_ORG,
        projectId: EMPTY_PROJECT,
        label: args.label,
        userId: args.userId,
        visibility: args.visibility,
      });
    } catch (e) {
      throw this.mapError(e, false);
    }
  }

  async createEpisodic(
    args: Parameters<MemoryCrudClient["createEpisodic"]>[0],
  ): Promise<CreateEpisodicResult> {
    try {
      const r = await this.noRetry.createEpisodic({
        title: args.title,
        content: args.content,
        summaryText: args.summaryText ?? "",
        orgId: EMPTY_ORG,
        userId: args.userId,
        sessionId: args.sessionId,
        projectId: EMPTY_PROJECT,
        visibility: args.visibility,
        agentId: args.agentId,
        participants: args.participants,
        tags: args.tags,
        metadata: args.metadata,
      });
      return CreateEpisodicResultSchema.parse(r);
    } catch (e) {
      throw this.mapError(e, false);
    }
  }

  async listEpisodic(
    args: Parameters<MemoryCrudClient["listEpisodic"]>[0],
  ): Promise<unknown[]> {
    try {
      return await this.retry.listEpisodic({
        orgId: EMPTY_ORG,
        projectId: EMPTY_PROJECT,
        userId: args.userId,
        sessionId: args.sessionId,
        visibility: args.visibility,
        limit: args.limit,
      });
    } catch (e) {
      throw this.mapError(e, false);
    }
  }

  async createTaxonomic(
    args: Parameters<MemoryCrudClient["createTaxonomic"]>[0],
  ): Promise<CreateTaxonomicResult> {
    try {
      const r = await this.noRetry.createTaxonomic({
        domain: args.domain,
        term: args.term,
        definition: args.definition,
        orgId: EMPTY_ORG,
        userId: args.userId,
        projectId: EMPTY_PROJECT,
        relatedTerms: args.relatedTerms,
        visibility: args.visibility,
      });
      return CreateTaxonomicResultSchema.parse(r);
    } catch (e) {
      throw this.mapError(e, false);
    }
  }

  async getTaxonomic(
    args: Parameters<MemoryCrudClient["getTaxonomic"]>[0],
  ): Promise<unknown | null> {
    try {
      return await this.retry.getTaxonomic({
        orgId: EMPTY_ORG,
        projectId: EMPTY_PROJECT,
        domain: args.domain,
        term: args.term,
        userId: args.userId,
        visibility: args.visibility,
      });
    } catch (e) {
      throw this.mapError(e, false);
    }
  }

  async getDistinctDomains(
    args: Parameters<MemoryCrudClient["getDistinctDomains"]>[0],
  ): Promise<string[]> {
    try {
      return await this.retry.getDistinctDomains({
        orgId: EMPTY_ORG,
        projectId: EMPTY_PROJECT,
        visibility: args.visibility,
      });
    } catch (e) {
      throw this.mapError(e, false);
    }
  }

  async createProcedural(
    args: Parameters<MemoryCrudClient["createProcedural"]>[0],
  ): Promise<CreateProceduralResult> {
    if (args.updateExisting) {
      // The HTTP backends serve only a create route that always creates;
      // app-bound mode honors updateExisting through its own runtime.
      throw new MemoryNotSupportedError(
        "updateExisting is not supported by the HTTP memory backends; the create " +
          "route always creates a new procedure",
      );
    }
    try {
      const r = await this.noRetry.createProcedural({
        procedure: args.procedure,
        description: args.description,
        content: args.content,
        orgId: EMPTY_ORG,
        userId: args.userId,
        projectId: EMPTY_PROJECT,
        steps: args.steps,
        resources: args.resources,
        allowedTools: args.allowedTools,
        compatibility: args.compatibility,
        license: args.license,
        triggerConditions: args.triggerConditions,
        tags: args.tags,
        visibility: args.visibility,
        agentId: args.agentId,
        extractionSource: args.extractionSource,
        sourceFormat: args.sourceFormat,
        sourcePath: args.sourcePath,
      });
      return CreateProceduralResultSchema.parse(r);
    } catch (e) {
      throw this.mapError(e, false);
    }
  }

  async getProcedural(
    args: Parameters<MemoryCrudClient["getProcedural"]>[0],
  ): Promise<unknown | null> {
    try {
      return await this.retry.getProcedural({
        orgId: EMPTY_ORG,
        projectId: EMPTY_PROJECT,
        procedure: args.procedure,
        userId: args.userId,
        visibility: args.visibility,
        includeDeleted: args.includeDeleted,
      });
    } catch (e) {
      throw this.mapError(e, false);
    }
  }

  /**
   * A bare 404/405 means the platform predates custom-type operations, and a
   * flag-off gateway rejects with a 400 naming the feature; the server's
   * unknown-type 404 names the type in a structured body and must pass
   * through untouched.
   */
  private mapCustomRouteError(e: unknown): unknown {
    const mapped = this.mapError(e, false);
    if (mapped instanceof MemoryBadRequestError) {
      const message = String(mapped.message);
      if (
        (mapped.status === 404 || mapped.status === 405) &&
        !message.includes("unknown custom memory type")
      ) {
        return new MemoryNotSupportedError(
          "this platform version does not support custom memory types; " +
            "save/retrieve requires a platform release with the custom memory " +
            "type operations enabled",
          { cause: mapped },
        );
      }
      // The gateway's flag-off rejection reuses the generic INVALID_REQUEST
      // code, so this message substring is the only stable discriminator for
      // that 400.
      if (
        mapped.status === 400 &&
        message.includes("custom_memory_types is not enabled")
      ) {
        return new MemoryNotSupportedError(
          "custom memory types are disabled on this deployment; " +
            "save/retrieve requires the custom_memory_types feature to be " +
            "enabled",
          { cause: mapped },
        );
      }
    }
    return mapped;
  }

  async createCustom(
    args: Parameters<MemoryCrudClient["createCustom"]>[0],
  ): Promise<CustomMemorySaveResult> {
    try {
      const r = await this.noRetry.createCustom({
        memoryType: args.memoryType,
        content: args.content,
        tags: args.tags,
        contextualMetadata: args.contextualMetadata,
      });
      return CustomMemorySaveResultSchema.parse(r);
    } catch (e) {
      throw this.mapCustomRouteError(e);
    }
  }

  async retrieveCustom(
    args: Parameters<MemoryCrudClient["retrieveCustom"]>[0],
  ): Promise<CustomMemoryRetrieveResult> {
    try {
      const r = await this.retry.retrieveCustom({
        memoryType: args.memoryType,
        query: args.query,
        tags: args.tags,
        topK: args.topK,
      });
      return CustomMemoryRetrieveResultSchema.parse(r);
    } catch (e) {
      throw this.mapCustomRouteError(e);
    }
  }

  close(): void {
    // MemoryClient holds no client to close (native fetch); present for parity.
  }
}
