/**
 * LangChain SDK Runtime.
 *
 * Port of `agent_engine_sdk_langgraph/runtime.py`.
 */

import * as path from "node:path";
import { createRequire } from "node:module";
import {
  interrupt as langgraphInterrupt,
  isGraphBubbleUp,
} from "@langchain/langgraph";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import * as CallbackManagerModule from "@langchain/core/callbacks/manager";
import { tool as lcTool, type StructuredTool } from "@langchain/core/tools";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
import { LangChainInstrumentation } from "@arizeai/openinference-instrumentation-langchain";
import { MongoClient } from "mongodb";
import { z, type ZodType } from "zod";

import {
  BaseApp,
  type JsonValue,
  LLMToolSchema,
  type ToolDefinition,
} from "@mongodb-js/agent-engine-sdk";
import { Memory } from "@mongodb-js/agent-engine-sdk-memory";
import {
  AppBoundCrudClient,
  AppBoundRuntime,
  clearWorkflowAdapter,
  createSecureToolFunction,
  currentAttemptContext,
  discoverMcpTools,
  getCurrentWrapper,
  getEnvBool,
  entrypointScope,
  runWithCustomerOrigin,
  getCheckpointWorkspaceId,
  getLogger,
  getStoreDbName,
  resolveStoreDbName,
  makeMcpToolCallable,
  mcpServerNetworkHosts,
  type MCPToolBinding,
  normalizeOptionalStr,
  registerInstrumentor,
  registerLlm,
  registerLLMAdapterFactory,
  registerQueryPlugin,
  registerSuspendHandler,
  registerWorkflowAdapter,
  requestSessionFinish,
  resetLlmRegistry,
  runInstrumentor,
  RuntimeMode,
  type SessionFinishStatus,
  type SuspendPayload,
  suspendPayloadToJson,
  TenantRuntime,
  type TenantRuntimeOptions,
  type ToolResponseFormat,
  getTracer,
  GRAPH_BUILD,
  OPENINFERENCE_SPAN_KIND,
  OpenInferenceSpanKind,
  ATTR_CACHE_HIT,
} from "@mongodb-js/agent-engine-runner-shared";
import type { AnyBackendProtocol } from "deepagents";
import type { BaseCheckpointSaver } from "@langchain/langgraph";
import { withEmptyBatchGuard } from "./checkpointer.js";
import { withDurableToolResultIdentity } from "./durable_tools.js";
import { LangGraphBaseAgent } from "./agent.js";
import type { PrepareAgentInput, ResolveThreadId } from "./agent.js";
import {
  createAgentEngineDeepAgent,
  type CreateAgentEngineDeepAgentOptions,
} from "./deep_agent.js";
import { AgentEngineToolPodBackend } from "./backends/toolpod.js";
import { LangChainLLMAdapter } from "./llm_adapter.js";
import { LangGraphQueryPlugin } from "./query.js";
import { LangGraphCallbackAdapter } from "./node_logger_adapter.js";
import {
  PlatformCheckpointer,
  UnsupportedDurableGraphError,
} from "./platform_checkpointer.js";
import { SecureWrappedLLM, type SecureLLMWrapper } from "./secure_llm.js";

const logger = getLogger("agent_engine_sdk_langgraph.runtime");

/**
 * This package's version, used as the workflow adapter version OE records.
 * Mirrors Python `_adapter_version()` (installed `agent-engine-sdk-langgraph`
 * version) so the declaration tracks releases. Falls back to the same
 * "0.0.0" unknown sentinel Python uses when package.json is unreachable
 * (bundled layouts) — never a concrete release number, which would
 * masquerade as the current version once the package moves past it.
 */
function adapterVersion(): string {
  try {
    const require = createRequire(import.meta.url);
    const version = (require("../package.json") as { version?: unknown })
      .version;
    if (typeof version === "string" && version.trim() !== "") return version;
  } catch {
    // Fall through to the fallback below.
  }
  return "0.0.0";
}

/**
 * Open a `graph.build` span around graph materialization. Opens on a
 * cache hit too (near-zero duration) so a trace can show the build was
 * skipped rather than omitting the span. Covers only graph materialization,
 * not the rest of `getAgent()` (already covered by AER's `aer.build_agent`
 * span).
 */
function tracedGraphBuild<T>(cacheHit: boolean, fn: () => T): T {
  return getTracer("runner-shared.agent-engine-sdk-langgraph").startActiveSpan(
    GRAPH_BUILD,
    {
      attributes: {
        [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN,
        [ATTR_CACHE_HIT]: cacheHit,
      },
    },
    (span) => {
      try {
        return fn();
      } finally {
        span.end();
      }
    },
  );
}

/**
 * Convert a tool's Zod schema to JSON Schema, degrading to `{}` on failure.
 *
 * Called at `tool()` decoration (module-eval) time. Zod 4's `toJSONSchema`
 * throws for shapes it cannot represent, so a bad schema must warn and fall
 * back rather than crash agent startup.
 */
function deriveArgsSchema(
  name: string,
  schema: ZodType,
): Record<string, JsonValue> {
  try {
    return z.toJSONSchema(schema) as Record<string, JsonValue>;
  } catch (exc: unknown) {
    logger.warn(
      `Failed to derive args_schema for tool ${name}: ${
        exc instanceof Error ? exc.message : String(exc)
      }`,
    );
    return {};
  }
}

/** Options for {@link App.deepAgent}. */
export interface DeepAgentOptions extends Omit<
  CreateAgentEngineDeepAgentOptions,
  "checkpointer" | "skillsBaseDir"
> {
  /**
   * Checkpointer selection. Omitted (`undefined`) resolves to
   * `app.checkpointer()` (MongoDB when configured, else none). `false` disables
   * checkpointing. A `BaseCheckpointSaver` instance is used directly.
   *
   * Note the mapping differs from Python (`None` disables there): in TS,
   * "disable" is `false`, and "use the default" is simply leaving it out.
   */
  checkpointer?: BaseCheckpointSaver | boolean;
  /**
   * Backend for filesystem/shell ops. Defaults to `AgentEngineToolPodBackend`, so
   * every op is OE-audited and sandboxed in the Tool Pod. Pass a custom backend
   * (e.g. deepagents' in-memory `StateBackend`) to override — note that doing so
   * bypasses the OE audit path.
   */
  backend?: AnyBackendProtocol;
}

// ---------------------------------------------------------------------------
// Local convenience alias — matches agent-engine-sdk `ToolDefinition` shape.
// ---------------------------------------------------------------------------

export type ToolDefinitionData = ToolDefinition;

// Legacy local definition removed — `ToolDefinitionData` aliases the real
// agent-engine-sdk `ToolDefinition` shape.

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export interface AppOptions {
  appName: string;
  appVersion?: string;
  mongodbUri?: string | null;
  databaseName?: string | null;
  tracesCollectionName?: string;
  /** @deprecated Ignored. The org is taken from the `ORG_ID` env var, which the platform injects. */
  orgId?: string | null;
}

export interface ToolDecoratorOptions {
  isLocal?: boolean;
  providerType?: string;
  scopes?: readonly string[];
  network?: readonly string[];
  timeout?: number;
  redactFields?: readonly string[];
  /**
   * Human-readable description shown to the LLM (mirrors Python's docstring).
   * TypeScript has no docstring introspection, so callers pass it explicitly.
   */
  description?: string;
  /**
   * Zod schema describing the tool's arguments. Without it, the LLM has no
   * structured view of the parameters and tool calling will fail. Mirrors the
   * type hints + Pydantic-style auto-schema that Python's `@app.tool()`
   * derives from the function signature.
   */
  schema?: ZodType;
  /** Override the registered tool name. Defaults to the wrapped function's `name`. */
  name?: string;
}

/**
 * LangChain SDK for the Atlas Agent Engine.
 *
 * @example
 * ```ts
 * import { App } from '@mongodb-js/agent-engine-sdk-langgraph';
 *
 * const app = new App({ appName: 'My Agent' });
 *
 * app.tool()((args: { query: string }) => 'result');
 *
 * app.entrypoint(() => {
 *   const llm = app.llm(chatModel);
 *   const checkpointer = app.checkpointer();
 *   const tools = app.getTools();
 *   // Build LangGraph...
 *   return graph;
 * });
 * ```
 */
export class App extends BaseApp {
  // BaseApp owns `readonly name: string` — no override needed.
  private readonly runtime: TenantRuntime;
  private builderFn: (() => unknown) | null = null;
  // Cached graph.build result. Safe to reuse for the process lifetime: the
  // builder function is fixed once app.entrypoint() is called, and Node's
  // single-threaded event loop can't interleave two synchronous builds, so
  // no lock is needed (unlike the Python port).
  private graphCache: unknown | undefined;
  private prepareInputFn: PrepareAgentInput | null = null;
  private resolveThreadIdFn: ResolveThreadId | null = null;
  private readonly toolDefs: ToolDefinitionData[] = [];
  private readonly lcTools: Map<string, unknown> = new Map();
  private mongoClient: unknown = null;
  private _checkpointer: PlatformCheckpointer | null = null;
  // Resolved once configured MCP servers have been discovered and registered
  // as tools. `TenantRuntime.runAsync()` awaits `ready()` (which returns this
  // promise) before binding the server, so no request can land before MCP
  // tools are registered. Discovery is async (a `tools/list` HTTP round trip
  // per server) but the constructor itself must stay synchronous, unlike
  // Python's `App.__init__` which can block on `asyncio.run()`.
  private readonly mcpToolsReady: Promise<void>;
  // Per-project-resolved store DB name for the checkpointer + query plugin.
  // Resolved asynchronously in run() (the Node driver's listDatabases is async,
  // unlike pymongo) so the synchronous checkpointer() can read it. Null until
  // resolved → callers fall back to the unscoped base name.
  private resolvedCheckpointDb: string | null = null;
  private _memory: Memory | null = null;

  /**
   * Unified memory facade over app-bound adapters.
   *
   * Lazily constructed on first access and cached (a long-lived singleton over
   * per-request context). Operations resolve the end-user and session from the
   * ambient execution context, so agent code calls `app.memory.saveSemantic(...)`
   * without threading identity through. Requests route through the OE memory
   * proxy; memory is reachable only while handling a platform request.
   */
  get memory(): Memory {
    if (this._memory === null) {
      this._memory = new Memory({
        runtime: new AppBoundRuntime(),
        client: new AppBoundCrudClient(),
      });
    }
    return this._memory;
  }

  constructor(options: AppOptions) {
    super(options.appName);
    if (options.orgId != null) {
      logger.warn(
        "App({ orgId }) is deprecated and ignored. " +
          "Set the ORG_ID environment variable instead; " +
          "this option will be removed in a future release.",
      );
    }
    // AppOptions uses camelCase (per repo TS coding standard); pass through to
    // TenantRuntimeOptions which also uses camelCase. orgId is intentionally
    // NOT forwarded — it is deprecated and ignored; the org is taken
    // from the ORG_ID env var, which the platform injects.
    const runtimeOpts: TenantRuntimeOptions = {
      appName: options.appName,
      ...(options.appVersion !== undefined && {
        appVersion: options.appVersion,
      }),
      ...(options.mongodbUri !== undefined && {
        mongodbUri: options.mongodbUri,
      }),
      ...(options.databaseName !== undefined && {
        databaseName: options.databaseName,
      }),
      ...(options.tracesCollectionName !== undefined && {
        tracesCollectionName: options.tracesCollectionName,
      }),
    };
    this.runtime = new TenantRuntime(runtimeOpts);
    this.mcpToolsReady = this.registerMcpTools();
  }

  /**
   * Resolved once configured MCP servers have been discovered and registered
   * as tools. `TenantRuntime.runAsync()` awaits this (via `GraphBuilderLike.ready()`)
   * before starting the server.
   */
  async ready(): Promise<void> {
    return this.mcpToolsReady;
  }

  /**
   * Discover configured MCP servers' tools and register them the same way
   * `App.tool()` registers author-declared tools. Port of Python's
   * `_register_mcp_tools_from_config`.
   */
  private async registerMcpTools(): Promise<void> {
    const mcpConfig = this.runtime.getAgentConfig().mcp;
    if (Object.keys(mcpConfig.servers).length === 0) return;

    const bindings: MCPToolBinding[] = await discoverMcpTools(mcpConfig);
    for (const binding of bindings) {
      const callable = makeMcpToolCallable(binding);
      const description = binding.description.trim();
      const langchainTool = lcTool(
        (args: Record<string, unknown>) => callable(args),
        {
          name: binding.sdkToolName,
          description,
          schema: binding.inputSchema as never,
        },
      );
      this.registerToolDefinition({
        name: binding.sdkToolName,
        func: ((args: Record<string, unknown>) =>
          callable(args)) as unknown as (...input: unknown[]) => unknown,
        description,
        argsSchema: binding.inputSchema as Record<string, JsonValue>,
        isLocal: false,
        providerType: null,
        scopes: [],
        network: mcpServerNetworkHosts(binding.serverConfig),
        timeoutSeconds: binding.serverConfig.timeout_seconds,
        redactFields: [],
        langchainTool,
        mcpServer: binding.serverName,
        mcpTool: binding.toolName,
      });
    }
    logger.info(`Registered ${bindings.length} MCP tool(s)`);
  }

  // ----- Read-only properties -----

  get agentConfig(): ReturnType<typeof this.runtime.getAgentConfig> {
    return this.runtime.getAgentConfig();
  }

  // ----- BaseApp contract -----

  override getToolDefinitions(): ToolDefinitionData[] {
    return [...this.toolDefs];
  }

  override tools(): unknown[] {
    return [...this.getTools()];
  }

  // ----- Tool registration -----

  private registerToolDefinition(args: {
    name: string;
    func: (...input: unknown[]) => unknown;
    description: string;
    argsSchema: Record<string, JsonValue>;
    isLocal: boolean;
    providerType: string | null;
    scopes: readonly string[];
    network: readonly string[];
    timeoutSeconds: number;
    redactFields: readonly string[];
    langchainTool?: unknown;
    /** Set for MCP-discovered tools so `getTools()` forwards Tool-Pod call metadata. */
    mcpServer?: string;
    mcpTool?: string;
  }): void {
    if (args.name in this.runtime.tools || this.lcTools.has(args.name)) {
      throw new Error(`tool '${args.name}' is already registered`);
    }
    const description = args.description.trim();
    const metadata: Record<string, unknown> = {
      name: args.name,
      description,
      is_local: args.isLocal,
      provider_type: args.providerType,
      scopes: args.scopes,
      network: args.network,
      timeout_seconds: args.timeoutSeconds,
      redact_fields: args.redactFields,
      ...(args.mcpServer !== undefined && { mcp_server: args.mcpServer }),
      ...(args.mcpTool !== undefined && { mcp_tool: args.mcpTool }),
    };
    // His TenantRuntime.registerTool signature: (name, func, metadata) positional.
    this.runtime.registerTool(args.name, args.func as never, metadata);
    const storedLcTool =
      args.langchainTool !== undefined
        ? args.langchainTool
        : lcTool(args.func as never, { name: args.name, description });
    // getTools() reads the per-call Stop opt-in off the stored LangChain tool,
    // but withCallInterruptSupport brands the registered callable — carry the
    // brand over or every App.tool() registration would answer not_cancellable.
    if (
      (args.func as { supportsCallInterrupt?: boolean })
        .supportsCallInterrupt === true
    ) {
      (
        storedLcTool as { supportsCallInterrupt?: boolean }
      ).supportsCallInterrupt = true;
    }
    this.lcTools.set(args.name, storedLcTool);
    this.toolDefs.push({
      name: args.name,
      description,
      args_schema: args.argsSchema,
      callable: args.func,
      remote: !args.isLocal,
      ...(args.providerType !== null && { provider_type: args.providerType }),
      scopes: [...args.scopes],
      network: [...args.network],
      timeout_seconds: args.timeoutSeconds,
      redact_fields: [...args.redactFields],
    });
  }

  /**
   * Register a tool function.
   *
   * Returns a function that, when applied to a tool function, registers it
   * and returns the function unchanged. Mirrors Python's `@app.tool()` shape.
   */
  tool(
    options: ToolDecoratorOptions = {},
  ): <F extends (...input: never[]) => unknown>(fn: F) => F {
    const isLocal = options.isLocal ?? true;
    const providerType = options.providerType ?? null;
    const scopes = options.scopes ?? [];
    const network = options.network ?? [];
    const timeout = options.timeout ?? 30;
    const redactFields = options.redactFields ?? [];
    const description = options.description ?? "";
    const schema = options.schema;
    const explicitName = options.name;

    return <F extends (...input: never[]) => unknown>(fn: F): F => {
      const name =
        explicitName ?? (fn.name === "" ? "anonymous_tool" : fn.name);

      // Build a LangChain tool with description + Zod schema so the LLM sees
      // the same metadata Python derives from docstrings & type hints.
      const langchainTool = schema
        ? lcTool(fn as never, { name, description, schema })
        : lcTool(fn as never, { name, description });

      this.registerToolDefinition({
        name,
        func: fn as unknown as (...input: unknown[]) => unknown,
        description,
        // Derive JSON Schema from the tool's Zod schema so downstream consumers
        // such as API documentation see its parameters. No schema means a
        // parameterless tool. Runs at decoration (module-eval)
        // time; `toJSONSchema` throws for shapes it can't represent (recursive
        // z.lazy() without cycles, z.function(), ...), so degrade to {} and warn
        // rather than crash agent startup.
        argsSchema: schema ? deriveArgsSchema(name, schema) : {},
        isLocal,
        providerType,
        scopes,
        network,
        timeoutSeconds: timeout,
        redactFields,
        langchainTool,
      });
      return fn;
    };
  }

  /** Mark the graph builder function. */
  entrypoint<F extends () => unknown>(fn: F): F {
    this.builderFn = fn;
    return fn;
  }

  /**
   * Register a hook that builds the graph's starting input from the caller's
   * `AgentInput` and `RequestContext` for a fresh execution. Resume stays
   * platform-managed. Returns the function unchanged so it can be used as a
   * decorator. Equivalent to the Python SDK's `@app.prepare_agent_input`.
   */
  prepareAgentInput<F extends PrepareAgentInput>(fn: F): F {
    this.prepareInputFn = fn;
    return fn;
  }

  /**
   * Register a hook that builds the LangGraph checkpoint `thread_id`.
   *
   * Callers manage Atlas Agent Engine `session_id` (and authenticated `user_id`). The
   * agent owns how those map to the LangGraph checkpoint key. When registered,
   * the hook's return value is used verbatim on every invocation — fresh and
   * resume — with no workspace suffix appended. When no hook is registered,
   * the adapter derives `session_id:workspace_id` as today.
   *
   * Custom keys are invisible to Atlas Agent Engine session-history queries
   * (`/query/sessions*`), which still look up only the default
   * session/workspace-derived keys. Agents that bypass workspace scoping also
   * own collision isolation within the checkpoint database.
   *
   * Equivalent to the Python SDK's `@app.resolve_thread_id`.
   *
   * @example
   * ```ts
   * app.resolveThreadId((ctx) => `${ctx.sessionId}__${actorFrom(ctx.userId)}`);
   * ```
   */
  resolveThreadId<F extends ResolveThreadId>(fn: F): F {
    this.resolveThreadIdFn = fn;
    return fn;
  }

  /**
   * Build and return a `LangGraphBaseAgent` instance.
   *
   * Matches `GraphBuilderLike.getAgent({ callbacks? })` — agent-engine-runner-shared's
   * `TenantRuntime.registerAndRun` introspects this method to compile the graph.
   * Accepts an options object with an optional `callbacks` array, each wrapped
   * in `LangGraphCallbackAdapter` before being passed to the graph.
   */
  getAgent(opts?: { callbacks?: readonly unknown[] }): LangGraphBaseAgent {
    const adaptedCallbacks: unknown[] = [];
    if (opts?.callbacks) {
      for (const cb of opts.callbacks) {
        adaptedCallbacks.push(new LangGraphCallbackAdapter(cb as never));
      }
    }
    const graph = this.getOrBuildGraph();
    // Durable eligibility depends on the materialized graph: only the
    // PlatformCheckpointer provides fence-keyed scratch and release, so a
    // graph compiled with a custom saver (or none) must stay native.
    // Re-runs on every call, including cache hits: registration is global
    // mutable state, so skipping it on a hit could leave eligibility stale.
    // Mirrors Python `runtime.py` adapter registration.
    if (
      (graph as { checkpointer?: unknown }).checkpointer instanceof
      PlatformCheckpointer
    ) {
      registerWorkflowAdapter("langgraph", adapterVersion());
    } else {
      clearWorkflowAdapter();
    }
    return new LangGraphBaseAgent(
      graph as never,
      adaptedCallbacks,
      this.prepareInputFn,
      this.resolveThreadIdFn,
    );
  }

  /**
   * Return the materialized graph, building it at most once.
   *
   * `getAgent()` used to call the entrypoint on every `/execute` — real,
   * measurable first-invoke latency. The graph carries no per-request
   * state (callbacks/agent wrapper are still rebuilt fresh by every
   * `getAgent()` call), so caching it is safe.
   */
  private getOrBuildGraph(): unknown {
    if (this.builderFn === null) {
      throw new Error(
        "No entrypoint registered. Use app.entrypoint() to mark the graph builder function.",
      );
    }
    const builderFn = this.builderFn;
    const cacheHit = this.graphCache !== undefined;

    return tracedGraphBuild(cacheHit, () => {
      if (cacheHit) return this.graphCache;
      resetLlmRegistry();
      const graph = entrypointScope(() =>
        runWithCustomerOrigin(() => builderFn()),
      );
      this.graphCache = graph;
      return graph;
    });
  }

  /**
   * Build and cache the graph when explicitly requested. The TypeScript AER
   * intentionally leaves construction on the existing lazy `/execute` path:
   * this synchronous builder cannot run during standby warming without
   * blocking the Node event loop and server health.
   */
  warmUp(): void {
    if (this.builderFn === null) return;
    this.getOrBuildGraph();
  }

  /**
   * Start the agent service.
   *
   * Async because the per-project store-DB name is resolved against the live
   * cluster (an async listDatabases round trip in the Node driver) before the
   * query plugin and checkpointer are wired, so AER writes land in the same
   * database the OE reads. The synchronous framework hooks are still registered
   * before the first `await`, preserving their ordering relative to startup.
   */
  async run(options: Record<string, unknown> = {}): Promise<void> {
    if (this.builderFn === null) {
      throw new Error(
        "No app.entrypoint() registered. Mark your graph-builder function with " +
          "app.entrypoint() before calling run().",
      );
    }
    App.registerHooks();
    // Run the LangChain instrumentor now that its hook is registered.
    // TenantRuntime's constructor already calls `setupTracing()` at boot, but
    // our instrumentor isn't registered until `registerHooks()` above runs, and
    // `setupTracing()` early-returns once a tracer provider exists — so calling
    // it a second time here would NOT re-run the instrumentor (the original bug).
    // Invoke the instrumentor directly instead. `manuallyInstrument` patches
    // LangChain's CallbackManager independently of the tracer provider, so this
    // is correct regardless of provider-init ordering.
    runInstrumentor();
    await this.resolveCheckpointDbName();
    this.registerQueryPlugin();
    this.runtime.registerAndRun(this, options);
  }

  /**
   * Resolve the store DB name once, before the checkpointer and query plugin
   * are constructed. Honors `CHECKPOINT_DB_NAME` as an exact override (no
   * project scoping). No-op outside AER mode or without a MongoDB URI when
   * the override is unset (the checkpointer returns null in those cases).
   */
  private async resolveCheckpointDbName(): Promise<void> {
    if (
      this.runtime.mode !== RuntimeMode.AER ||
      this.resolvedCheckpointDb !== null
    ) {
      return;
    }
    // Exact database override when configured. Must not apply project
    // scoping or discovery — the value is the final DB name.
    const checkpointDbOverride = this.checkpointDbOverride();
    if (checkpointDbOverride !== null) {
      this.resolvedCheckpointDb = checkpointDbOverride;
      return;
    }
    const mongodbUri =
      this.runtime.getMongodbUri() ?? process.env["MONGODB_URI"] ?? "";
    if (mongodbUri === "") return;
    const client = new MongoClient(mongodbUri);
    try {
      this.resolvedCheckpointDb = await resolveStoreDbName(
        client as unknown as Parameters<typeof resolveStoreDbName>[0],
      );
    } finally {
      void client.close();
    }
  }

  /**
   * Exact `CHECKPOINT_DB_NAME` override when set, else null.
   * Shared by resolve / checkpointer / query plugin so the three paths cannot
   * drift on trim/precedence.
   */
  private checkpointDbOverride(): string | null {
    const override = (process.env["CHECKPOINT_DB_NAME"] ?? "").trim();
    return override !== "" ? override : null;
  }

  /**
   * Final MongoDBSaver / query-plugin database name for this app.
   * Prefer the exact env override; otherwise the resolved (or base) store DB.
   */
  private checkpointDbName(): string {
    return (
      this.checkpointDbOverride() ??
      this.resolvedCheckpointDb ??
      getStoreDbName()
    );
  }

  /**
   * Register the LangGraph-backed `AERQueryPlugin` (AER mode only). A
   * missing MongoDB URI or a failing checkpointer construction degrades to
   * "no plugin" (routes return 501) instead of failing startup.
   */
  private registerQueryPlugin(): void {
    if (this.runtime.mode !== RuntimeMode.AER) return;
    try {
      const saver = this.checkpointer();
      const native = saver?.native;
      if (
        native === null ||
        native === undefined ||
        this.mongoClient === null
      ) {
        return;
      }
      registerQueryPlugin(
        new LangGraphQueryPlugin({
          saver: native as InstanceType<typeof MongoDBSaver>,
          client: this.mongoClient as MongoClient,
          dbName: this.checkpointDbName(),
          workspaceIdResolver: getCheckpointWorkspaceId,
        }),
      );
      logger.info("Registered LangGraph session query plugin");
    } catch (exc) {
      // Log only the error class: driver URI-parse errors can echo
      // connection-string contents (credentials included) in their message.
      logger.warn(
        "Skipping session query plugin registration; checkpointer " +
          `unavailable: ${exc instanceof Error ? exc.name : typeof exc}`,
      );
    }
  }

  /**
   * Register framework-specific hooks so agent-engine-runner-shared can dispatch to
   * LangChain/LangGraph code without importing this package. Mirrors Python's
   * `App._register_hooks` (static method).
   */
  private static registerHooks(): void {
    // Suspend handler — wires LangGraph's `interrupt()` so runner-shared can
    // pause a graph from inside `SuspendPayload` flows. Lazy import keeps
    // langgraph out of the cold path for non-AER modes.
    registerSuspendHandler(((payload: unknown) => {
      if (currentAttemptContext() !== null) {
        throw new UnsupportedDurableGraphError(
          "App.suspend framework suspension is not supported on durable_workflow sessions",
        );
      }
      return langgraphInterrupt(payload as never);
    }) as never);

    // LLM adapter factory — runner-shared's Tool Pod /invoke_llm route uses
    // this to construct a LangChainLLMAdapter from the customer's BaseChatModel.
    registerLLMAdapterFactory(
      (rawLlm: unknown, opts?: { tools?: unknown[]; tool_choice?: unknown }) =>
        new LangChainLLMAdapter(
          rawLlm as never,
          opts?.tools as never,
          opts?.tool_choice,
        ) as never,
    );

    // Instrumentor — installs OpenInference LangChain instrumentation when
    // `setupTracing()` fires. Equivalent to Python's
    // `register_instrumentor(lambda: LangChainInstrumentor().instrument())`.
    //
    // JS LangChain has a non-traditional module layout, so the OpenInference
    // package patches the already-imported CallbackManager module explicitly
    // via `manuallyInstrument(...)` (see the package README).
    registerInstrumentor((() => {
      const lcInstrumentation = new LangChainInstrumentation();
      lcInstrumentation.manuallyInstrument(CallbackManagerModule);
    }) as never);
  }

  // ----- LLM API -----

  /**
   * Wrap a LangChain LLM for audited I/O through the Orchestration Engine.
   *
   * In AER mode the LLM is wrapped in `SecureWrappedLLM`. In TOOL mode the raw
   * LLM is returned unwrapped (the tool pod is the execution end of the chain).
   *
   * For agents with a single LLM, call without `llmId`. For agents with
   * multiple LLMs every call must supply a unique `llmId`:
   *   const fast    = app.llm(ChatOpenAI("gpt-5.4-mini"), "fast")
   *   const primary = app.llm(ChatOpenAI("gpt-5.4"),      "primary")
   * Unnamed calls register under the sentinel id `"__default__"`; a second
   * unnamed call therefore raises the same duplicate-id error as a second
   * named call with the same id.
   */
  llm(llm: BaseChatModel, llmId?: string): BaseChatModel {
    const resolvedId = llmId ?? "__default__";
    // Register in the named-LLM registry regardless of mode — mirrors Python's
    // App.llm() which calls register_llm() before the mode branch. The Tool Pod
    // needs it for /invoke_llm resolution; the AER side needs it for tests and
    // future named-LLM lookup.
    registerLlm(resolvedId, llm);
    if (this.runtime.mode === RuntimeMode.TOOL) {
      return llm;
    }
    return new SecureWrappedLLM(
      llm,
      () => getCurrentWrapper() as SecureLLMWrapper | null,
      resolvedId,
    ) as unknown as BaseChatModel;
  }

  // ----- Checkpointing -----

  /**
   * Get the platform checkpointer for LangGraph.
   *
   * Lazily constructs and wraps a `MongoDBSaver` on first call in AER mode.
   * Native sessions use that saver through the request-scoped platform
   * wrapper. The underlying `MongoClient` is closed by `App.close()`.
   *
   * The database name is read from `CHECKPOINT_DB_NAME` when set (exact
   * override, no project scoping). Otherwise the existing
   * `MDB_AGENTIC_STORE_DB` / per-project store resolution is used
   * (default: `"mdb_store"`). The URI comes from `MONGODB_URI` — the same
   * source `TenantRuntime` uses internally.
   *
   * Mirrors Python `runtime.py:checkpointer()`.
   */
  checkpointer(): PlatformCheckpointer | null {
    if (this.runtime.mode === RuntimeMode.AER) {
      if (this._checkpointer !== null) return this._checkpointer;

      const checkpointDb = this.checkpointDbName();
      // Prefer the runtime-resolved URI (which honors the constructor's
      // `mongodbUri` option) over the env var directly. Falls back to env for
      // any code path that hasn't been migrated to the constructor option yet.
      const mongodbUri =
        this.runtime.getMongodbUri() ?? process.env["MONGODB_URI"] ?? "";

      if (mongodbUri !== "") {
        logger.info(`Using MongoDB checkpointer: db=${checkpointDb}`);
        const client = new MongoClient(mongodbUri);
        try {
          // `@langchain/langgraph-checkpoint-mongodb` pins `mongodb@^6` while our
          // top-level dep is `mongodb@^7`. The two `MongoClient` types differ
          // only in deeply-nested option shapes that `MongoDBSaver` never
          // touches — it calls `client.db(name)` and `client.close()`, both
          // identical across versions. Cast to bridge the type mismatch.
          const native = withEmptyBatchGuard(
            new MongoDBSaver({
              client: client as unknown as ConstructorParameters<
                typeof MongoDBSaver
              >[0]["client"],
              dbName: checkpointDb,
            }),
          );
          this._checkpointer = new PlatformCheckpointer({ native });
        } catch (err) {
          // MongoDBSaver construction may fail (e.g., bad URI, auth). Close
          // the freshly-opened client before the exception propagates so
          // every failed init does not leak a pool connection.
          void client.close();
          throw err;
        }
        this.mongoClient = client;
        return this._checkpointer;
      }

      logger.warn("MongoDB URI not defined, required for checkpointer");
      return null;
    }
    logger.warn(
      `app.checkpointer() called in ${this.runtime.mode} mode; checkpointing is managed ` +
        "by the platform in non-AER modes — returning null",
    );
    return null;
  }

  // ----- Deep agents -----

  /**
   * Build a deepagents graph pre-wired for Atlas Agent Engine AER.
   *
   * Wraps `llm` in `SecureWrappedLLM`, resolves relative skill paths, validates
   * the subagent tree (string models are rejected — they would bypass OE
   * routing), then delegates to deepagents' `createDeepAgent`.
   *
   * Each `skills` entry is a parent source directory. At runtime, deepagents
   * lists it through the configured backend and treats each immediate child
   * directory containing `SKILL.md` as one skill; discovery is not recursive.
   * deepagents skips unreadable or unparsable frontmatter and skills missing
   * `name` or `description`; it warns but may still load Agent Skills naming or
   * directory-name violations.
   *
   * Mirrors Python `runtime.py:App.deep_agent()`.
   *
   * @throws {Error} `features.deep_agent` is not enabled, or a subagent spec
   *   uses a string model, or nesting exceeds the recursion cap.
   */
  deepAgent(
    llm: BaseChatModel,
    options: DeepAgentOptions = {},
  ): ReturnType<typeof createAgentEngineDeepAgent> {
    // Fail fast if the tenant hasn't opted into the Tool Pod's built-in
    // filesystem/shell handlers — otherwise a AgentEngineToolPodBackend call would
    // fail at runtime with "unknown tool", a confusing error several layers
    // from the cause. Requiring the flag at construction time gives an
    // immediate, actionable message.
    if (!this.agentConfig.featureEnabled("deep_agent", false)) {
      throw new Error(
        "App.deepAgent() requires 'features.deep_agent: true' in agent.yaml. " +
          "Without it the Tool Pod does not register the built-in filesystem + " +
          "shell handlers that deep agents rely on. Add:\n\n" +
          "    features:\n" +
          "      deep_agent: true\n\n" +
          "to your agent.yaml and redeploy the Tool Pod.",
      );
    }

    // No resetLlmRegistry() here. `getAgent()` already clears the registry
    // immediately before running the entrypoint, so a second reset is
    // redundant — and harmful: JS evaluates call arguments before the callee,
    // so a subagent model built inline as `app.llm(model, "researcher")` in the
    // `subagents` option registers "researcher" *before* this method body runs.
    // Clearing here would wipe that registration, and the Tool Pod's
    // /invoke_llm route would then throw `llm_id "researcher" not registered`
    // when the subagent's model is resolved. Registering the orchestrator's
    // __default__ below via `this.llm()` leaves prior subagent entries intact;
    // a genuine duplicate id surfaces through `registerLlm`'s own guard.
    const secureLlm = this.llm(llm);

    // Checkpointer resolution:
    //   undefined (omitted) -> app.checkpointer() (returns null outside AER)
    //   false               -> disable checkpointing
    //   instance            -> use directly
    const checkpointer =
      options.checkpointer === undefined
        ? (this.checkpointer() as BaseCheckpointSaver | null)
        : options.checkpointer;

    const skillsBaseDir = this.resolveSkillsBaseDir();

    // Default to the secure Tool Pod backend so filesystem/shell ops are
    // OE-audited and sandboxed. A caller-supplied backend still overrides —
    // matching Python `deep_agent()`. Without this, deepagents would fall back
    // to its in-memory StateBackend, which bypasses the OE entirely.
    const { checkpointer: _c, backend, ...rest } = options;
    const resolvedBackend = backend ?? new AgentEngineToolPodBackend();
    return createAgentEngineDeepAgent(secureLlm, resolvedBackend, {
      ...rest,
      ...(checkpointer != null && { checkpointer }),
      ...(skillsBaseDir !== undefined && { skillsBaseDir }),
    });
  }

  /**
   * Resolve the base directory for relative skill paths: the directory
   * containing `agent.yaml`, optionally narrowed by the source-root-relative
   * `AGENTIC_SKILLS_DIR` override.
   */
  private resolveSkillsBaseDir(): string | undefined {
    const configPath = this.agentConfig.path;
    const agentDir =
      configPath != null ? path.dirname(path.resolve(configPath)) : undefined;

    const configured = process.env["AGENTIC_SKILLS_DIR"];
    if (configured === undefined || configured === "") return agentDir;

    if (path.isAbsolute(configured)) {
      throw new Error(
        "AGENTIC_SKILLS_DIR must be relative to the agent source root; " +
          `got '${configured}'`,
      );
    }
    if (agentDir === undefined) {
      throw new Error(
        "AGENTIC_SKILLS_DIR requires agent.yaml to determine the agent source root",
      );
    }
    const resolved = path.resolve(agentDir, configured);
    const rel = path.relative(agentDir, resolved);
    if (rel.startsWith("..") || path.isAbsolute(rel)) {
      throw new Error(
        "AGENTIC_SKILLS_DIR must stay within the agent source root; " +
          `got '${configured}'`,
      );
    }
    return resolved;
  }

  /** Release resources held by this App instance. */
  async close(): Promise<void> {
    if (this.mongoClient !== null) {
      const client = this.mongoClient as {
        close?: () => Promise<void> | void;
      };
      // Clear fields before awaiting so repeated/concurrent calls are idempotent.
      this.mongoClient = null;
      this._checkpointer = null;
      // mongodb v7 `MongoClient.close()` returns Promise<void>; await it so the
      // connection pool drains before the process exits (no leaked sockets).
      if (typeof client.close === "function") await client.close();
    }
  }

  // ----- Tools -----

  /**
   * Get wrapped tools for LangGraph's `ToolNode`.
   *
   * In AER mode, every tool is wrapped with `SecureToolWrapper` (via
   * `createSecureToolFunction`) so executions route through OE for logging
   * and policy enforcement. In other modes, the raw LangChain tools are
   * returned unchanged.
   */
  getTools(): readonly StructuredTool[] {
    const toolsList = [...this.lcTools.values()] as StructuredTool[];
    if (this.runtime.mode !== RuntimeMode.AER) return toolsList;

    const allowDirect = getEnvBool("RUNNER_ALLOW_DIRECT_TOOL_EXECUTION", false);
    const wrappedTools: StructuredTool[] = [];
    for (const toolObj of toolsList) {
      const toolAny = toolObj as unknown as {
        name?: string;
        description?: string;
        schema?: unknown;
      };
      const toolName = toolAny.name ?? "anonymous";
      const toolMetadata = this.runtime.getToolMetadata(toolName);
      let toolCallMetadata: Record<string, unknown> | undefined;
      const mcpServer = toolMetadata["mcp_server"];
      const mcpTool = toolMetadata["mcp_tool"];
      if (typeof mcpServer === "string" && typeof mcpTool === "string") {
        toolCallMetadata = { mcp_server: mcpServer, mcp_tool: mcpTool };
      }
      const providerType = normalizeOptionalStr(toolMetadata["provider_type"]);
      let scopes: string[] = [];
      const rawScopes = toolMetadata["scopes"];
      if (Array.isArray(rawScopes)) {
        scopes = rawScopes
          .filter(
            (scope): scope is string =>
              typeof scope === "string" && scope.trim().length > 0,
          )
          .map((scope) => scope.trim());
      }
      const toolDeclaredFormat: ToolResponseFormat =
        (toolAny as { responseFormat?: string }).responseFormat ===
        "content_and_artifact"
          ? "content_and_artifact"
          : "content";
      // Opt-in per-call Stop support, branded by withCallInterruptSupport.
      const supportsCallInterrupt =
        (toolAny as { supportsCallInterrupt?: boolean })
          .supportsCallInterrupt === true;
      // The tool's redact_fields policy must reach the wrapper's debug
      // argument dump.
      const rawRedactFields = toolMetadata["redact_fields"];
      const redactFields = Array.isArray(rawRedactFields)
        ? rawRedactFields.filter((f): f is string => typeof f === "string")
        : [];
      const wrapperFunc = createSecureToolFunction(
        toolObj,
        toolName,
        allowDirect,
        {
          metadata: toolCallMetadata,
          isLocal: toolMetadata["is_local"] !== false,
          providerType,
          scopes,
          responseFormat: "content_and_artifact",
          toolDeclaredFormat,
          redactFields,
          isFrameworkControlFlow: isGraphBubbleUp,
          supportsCallInterrupt,
        },
      );
      const durableWrapperFunc = withDurableToolResultIdentity(
        wrapperFunc,
        toolName,
      );
      // Forward the original tool's Zod schema. Without it, lcTool() builds a
      // DynamicTool (single `input: string` shape) which strips structured
      // tool_call args before reaching the wrapper — every kwarg arrives
      // undefined at the Tool Pod.
      const wrapped = lcTool(durableWrapperFunc as never, {
        name: toolName,
        description: toolAny.description ?? "",
        responseFormat: "content_and_artifact",
        ...(toolAny.schema !== undefined && {
          schema: toolAny.schema as never,
        }),
      });
      wrappedTools.push(wrapped as unknown as StructuredTool);
      logger.debug(`Wrapped tool: ${toolName}`);
    }
    return wrappedTools;
  }

  /** Get tool schemas for `llm.bindTools()`. Always the unwrapped LangChain tools. */
  getToolSchemas(): readonly unknown[] {
    return [...this.lcTools.values()];
  }

  /**
   * Generate a suspend command. If a tool should suspend, return the result of
   * this method instead of completing normally.
   */
  suspend(reason: string, context: Record<string, unknown>): string {
    const payload: SuspendPayload = {
      suspend_reason: reason,
      suspend_context: context as never,
    };
    return suspendPayloadToJson(payload);
  }

  /**
   * Mark this session finished so the platform frees its compute now.
   *
   * Call it when the agent is done with the session. The current turn keeps
   * running and returns its result normally; once it completes, the platform
   * cancels any live sub-agent runs and releases the session's AER and tool
   * pods instead of holding them until the idle timeout expires.
   *
   * Safe to call more than once: the first call returns "requested", later
   * ones "already_requested". Outside an agent run (local scripts, tool pods)
   * there is no session to finish and the call returns "unavailable" without
   * throwing. Calling it after the turn has already ended - e.g. from a
   * setTimeout or a floating promise scheduled during the turn but resolving
   * after it - also returns "unavailable": by then nothing is listening for
   * the request anymore, so reporting "requested" would promise a release
   * that will never happen.
   *
   * A turn that suspends for human review, or that fails, keeps its
   * resources so it stays resumable and diagnosable; the session then falls
   * back to the idle timeout.
   */
  finishSession(): SessionFinishStatus {
    return requestSessionFinish();
  }

  // ----- Context -----

  getCurrentUserId(): string | null {
    return this.runtime.getCurrentUserId();
  }

  getCurrentSessionId(): string | null {
    return this.runtime.getCurrentSessionId();
  }
}

// Re-exports — callers commonly need these alongside App.
export { LLMToolSchema };
export { RuntimeMode };
