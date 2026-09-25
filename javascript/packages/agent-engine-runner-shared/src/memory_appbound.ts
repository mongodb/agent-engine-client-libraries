/**
 * App-bound memory adapters for the platform runtime.
 *
 * These let the unified `Memory` facade from `@mongodb-js/agent-engine-sdk-memory`
 * run in app-bound mode (inside an agent, on the platform stack). App-bound memory
 * routes through the Orchestration Engine's memory proxy, which serves the aliased
 * memory routes (`/api/v1/memory/{turns,context,search}` plus the CRUD paths) and
 * stamps org/project from the trusted execution context.
 *
 * So the adapters are thin: each call reads the current request context
 * (`OE_URL`, execution id, user id, session id), builds a `MemoryClientAdapter`
 * pointed at the OE with mTLS and the execution-id header, and delegates. Identity
 * is ambient — `requestContext()` exposes the per-request user/session so the
 * facade resolves them without the agent author threading them through.
 */

import {
  MemoryClientAdapter,
  MemoryNotSupportedError,
  ContextResponseSchema,
  type ContextResponse,
  type CreateEpisodicResult,
  type CreateProceduralResult,
  type CreateSemanticResult,
  type CreateTaxonomicResult,
  type CustomMemoryRetrieveResult,
  type CustomMemorySaveResult,
  type MemoryChunk,
  type MemoryCrudClient,
  type MemoryRequestContext,
  type MemoryRuntime,
  type WriteTurnResult,
} from "@mongodb-js/agent-engine-sdk-memory";

import {
  getCurrentExecutionId,
  getCurrentOeUrl,
  getCurrentSessionId,
  getCurrentUserId,
  getCurrentWorkspaceId,
} from "./context.js";
import { fetchPlatform, getFetchOptionsWithTLS } from "./tls_client.js";
import { isMemoryReadOwnedByTool } from "./tool_memory_ownership.js";
import {
  ActivityKind,
  WorkflowClient,
  allocateActivityOrdinal,
  currentAttemptContext,
  runSerialActivity,
} from "./workflow/index.js";

const EXECUTION_ID_HEADER = "X-Agent-Engine-Execution-Id";
/** Legacy name, dual-sent during the rename transition; servers prefer EXECUTION_ID_HEADER. */
const EXECUTION_ID_HEADER_LEGACY = "X-Agentic-Execution-Id";
const AGENT_ID_HEADER = "X-Agent-Engine-Agent-Id";
const AGENT_ID_HEADER_LEGACY = "X-Agentic-Agent-Id";
/** The OE memory proxy serves the flat memory routes under this prefix. */
const OE_MEMORY_PREFIX = "/api/v1/memory";

/**
 * Build a `MemoryClientAdapter` bound to the current request context.
 *
 * Reads `OE_URL` fresh per call because `app.memory` is a long-lived singleton
 * over per-request context. mTLS options and the execution-id header are layered
 * in so the OE can authenticate the connection and attribute the call.
 */
function adapterFromContext(): MemoryClientAdapter {
  const oeUrl = getCurrentOeUrl();
  if (oeUrl === null || oeUrl.trim() === "") {
    throw new MemoryNotSupportedError(
      "app-bound memory is unavailable: no OE_URL in the current execution " +
        "context (memory is only reachable while handling a platform request)",
    );
  }
  const executionId = getCurrentExecutionId();
  const headers: Record<string, string> = {};
  if (executionId !== null && executionId !== "") {
    headers[EXECUTION_ID_HEADER] = executionId;
    headers[EXECUTION_ID_HEADER_LEGACY] = executionId;
  }
  const workspaceId = getCurrentWorkspaceId();
  if (workspaceId !== null && workspaceId !== "") {
    headers[AGENT_ID_HEADER] = workspaceId;
    headers[AGENT_ID_HEADER_LEGACY] = workspaceId;
  }
  return new MemoryClientAdapter({
    baseUrl: oeUrl,
    apiPrefix: OE_MEMORY_PREFIX,
    routeStyle: "aliased",
    projectScoped: false,
    headers,
    fetchImpl: fetchPlatform,
    requestExtras: () => getFetchOptionsWithTLS(oeUrl) ?? {},
  });
}

/**
 * Match the current Python durable boundary: replay context construction, but
 * leave outcome-derived STM publication to the shared outcome-delivery protocol.
 */
type BuildContextArgs = Parameters<MemoryRuntime["buildContext"]>[0];

/** Canonical JSON input used to identify a durable context read. */
function buildContextActivityInput(
  args: BuildContextArgs,
): Record<string, unknown> {
  const input: Record<string, unknown> = {
    query: args.query,
    user_id: args.userId ?? null,
    session_id: args.sessionId ?? null,
    visibility: args.visibility ?? null,
    metadata_filter: args.metadataFilter ?? null,
    enabled_sources:
      args.enabledSources == null ? null : [...args.enabledSources].sort(),
  };
  if (args.topK !== undefined) input["top_k"] = args.topK;
  if (args.maxTokens !== undefined) input["max_tokens"] = args.maxTokens;
  return input;
}

async function runBuildContextActivity(
  args: BuildContextArgs,
  execute: () => Promise<ContextResponse>,
): Promise<ContextResponse> {
  if (currentAttemptContext() === null || isMemoryReadOwnedByTool()) {
    return execute();
  }

  const oeUrl = getCurrentOeUrl();
  if (oeUrl === null || oeUrl.trim() === "") {
    throw new MemoryNotSupportedError(
      "durable app-bound context reads require OE_URL in the current execution context",
    );
  }
  const result = await runSerialActivity<unknown>({
    client: new WorkflowClient(oeUrl),
    kind: ActivityKind.MEMORY,
    name: "memory.build_context",
    activityOrdinal: allocateActivityOrdinal(),
    semanticInput: buildContextActivityInput(args),
    execute,
  });
  return ContextResponseSchema.parse(result);
}

/** `MemoryRuntime` over the OE memory proxy, exposing ambient identity. */
export class AppBoundRuntime implements MemoryRuntime {
  /** Fresh identity from the runner context; blank fields are normalized by the facade. */
  requestContext(): MemoryRequestContext | null {
    return {
      userId: getCurrentUserId(),
      sessionId: getCurrentSessionId(),
    };
  }

  async recordTurn(
    args: Parameters<MemoryRuntime["recordTurn"]>[0],
  ): Promise<WriteTurnResult> {
    return adapterFromContext().recordTurn(args);
  }

  async buildContext(
    args: Parameters<MemoryRuntime["buildContext"]>[0],
  ): Promise<ContextResponse> {
    return runBuildContextActivity(args, () =>
      adapterFromContext().buildContext(args),
    );
  }

  async searchSemantic(
    args: Parameters<MemoryRuntime["searchSemantic"]>[0],
  ): Promise<MemoryChunk[]> {
    return adapterFromContext().searchSemantic(args);
  }

  async searchEpisodes(
    args: Parameters<MemoryRuntime["searchEpisodes"]>[0],
  ): Promise<MemoryChunk[]> {
    return adapterFromContext().searchEpisodes(args);
  }

  async searchTaxonomic(
    args: Parameters<MemoryRuntime["searchTaxonomic"]>[0],
  ): Promise<MemoryChunk[]> {
    return adapterFromContext().searchTaxonomic(args);
  }

  async discoverProcedures(
    args: Parameters<MemoryRuntime["discoverProcedures"]>[0],
  ): Promise<Array<Record<string, unknown>>> {
    return adapterFromContext().discoverProcedures(args);
  }
}

/** `MemoryCrudClient` over the OE memory proxy. */
export class AppBoundCrudClient implements MemoryCrudClient {
  async createSemantic(
    args: Parameters<MemoryCrudClient["createSemantic"]>[0],
  ): Promise<CreateSemanticResult> {
    return adapterFromContext().createSemantic(args);
  }

  async getSemantic(
    args: Parameters<MemoryCrudClient["getSemantic"]>[0],
  ): Promise<unknown | null> {
    return adapterFromContext().getSemantic(args);
  }

  async createEpisodic(
    args: Parameters<MemoryCrudClient["createEpisodic"]>[0],
  ): Promise<CreateEpisodicResult> {
    return adapterFromContext().createEpisodic(args);
  }

  async listEpisodic(
    args: Parameters<MemoryCrudClient["listEpisodic"]>[0],
  ): Promise<unknown[]> {
    return adapterFromContext().listEpisodic(args);
  }

  async createTaxonomic(
    args: Parameters<MemoryCrudClient["createTaxonomic"]>[0],
  ): Promise<CreateTaxonomicResult> {
    return adapterFromContext().createTaxonomic(args);
  }

  async getTaxonomic(
    args: Parameters<MemoryCrudClient["getTaxonomic"]>[0],
  ): Promise<unknown | null> {
    return adapterFromContext().getTaxonomic(args);
  }

  async getDistinctDomains(
    args: Parameters<MemoryCrudClient["getDistinctDomains"]>[0],
  ): Promise<string[]> {
    return adapterFromContext().getDistinctDomains(args);
  }

  async createProcedural(
    args: Parameters<MemoryCrudClient["createProcedural"]>[0],
  ): Promise<CreateProceduralResult> {
    return adapterFromContext().createProcedural(args);
  }

  async getProcedural(
    args: Parameters<MemoryCrudClient["getProcedural"]>[0],
  ): Promise<unknown | null> {
    return adapterFromContext().getProcedural(args);
  }

  async createCustom(
    args: Parameters<MemoryCrudClient["createCustom"]>[0],
  ): Promise<CustomMemorySaveResult> {
    return adapterFromContext().createCustom(args);
  }

  async retrieveCustom(
    args: Parameters<MemoryCrudClient["retrieveCustom"]>[0],
  ): Promise<CustomMemoryRetrieveResult> {
    return adapterFromContext().retrieveCustom(args);
  }
}
