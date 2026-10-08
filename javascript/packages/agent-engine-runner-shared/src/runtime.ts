/**
 * TenantRuntime — framework-agnostic platform runtime for tenant agent
 * applications.
 *
 * Port of `agent_engine_runner_shared/runtime.py`. Same Python class shape: the runtime
 * discovers its role from `RUNNER_MODE`, wires logging / tracing / tools,
 * and delegates to a framework SDK's `App` (the `GraphBuilderLike`) for
 * graph construction.
 *
 * Scope deltas vs. Python (see `AGENTS.md` for rationale):
 *   - No memory-server mode (`RuntimeMode` has `AER`, `TOOL`, and `TOOL_FUNCTION`).
 *   - `setupTracing` owns the MongoClient build (async). Python builds it
 *     inline in `_setup_tracing` (sync pymongo).
 *   - `shutdown()` and SIGTERM/SIGINT wiring are TS additions — Python's
 *     uvicorn handles signals itself; here we explicitly close the tracing
 *     MongoClient before exit.
 */

import type { RuntimeAgentConfig } from "./agent_config.js";
import { loadRuntimeAgentConfig } from "./agent_config.js";
import { getCurrentSessionId, getCurrentUserId } from "./context.js";
import { getLogger, setupLogging } from "./logger.js";
import { AppBoundRuntime } from "./memory_appbound.js";
import { MemoryWriter } from "./memory_writer.js";
import { AERServer } from "./server/aer.js";
import {
  type GraphBuilderLike,
  type ITenantRuntime,
  type ServerToolFn,
} from "./server/base.js";
import { ToolFunctionRunner } from "./server/function.js";
import { ToolServer } from "./server/tool.js";
import {
  scrubCredentials,
  setupTracing,
  shutdownTracing,
} from "./tracing/index.js";
import {
  getEnvBool,
  getEnvInt,
  getRuntimeMode,
  RuntimeMode,
  tenantEnvVars,
} from "./utils.js";

import type {
  BaseAgent,
  BaseExecutionCallback,
} from "@mongodb-js/agent-engine-sdk";

const logger = getLogger("agent_engine_runner_shared.runtime");

/**
 * Resolve the listen host from `APP_HOST`, defaulting to `"0.0.0.0"`.
 *
 * Matches Python's `_resolve_listen_host`: ECP stamps `APP_HOST` per
 * executor type at deploy time (`"::"` for vm-mode, `"0.0.0.0"` for
 * container-mode). Empty string normalises to the default — a shell that
 * exports `APP_HOST=""` must not bind to `""`.
 */
export function resolveListenHost(): string {
  return process.env["APP_HOST"] || "0.0.0.0";
}

/** Resolve the server drain budget without changing the existing Tool policy. */
export function shutdownGracePeriodMs(mode: RuntimeMode): number {
  if (mode !== RuntimeMode.AER) {
    return getEnvInt("SHUTDOWN_GRACE_PERIOD_MS", 25000);
  }

  const raw = process.env["SHUTDOWN_GRACE_PERIOD_MS"];
  if (!raw) return 25000;
  const normalized = raw.trim();
  if (!/^\+?\d+$/.test(normalized)) {
    throw new Error(
      "SHUTDOWN_GRACE_PERIOD_MS must be a positive integer in milliseconds",
    );
  }
  const milliseconds = Number(normalized);
  if (!Number.isSafeInteger(milliseconds) || milliseconds <= 0) {
    throw new Error(
      "SHUTDOWN_GRACE_PERIOD_MS must be a positive integer in milliseconds",
    );
  }
  return milliseconds;
}

export interface TenantRuntimeOptions {
  /** Application name. Defaults to `"Agent"`. */
  appName?: string;
  /** Application version. Defaults to `"1.0.0"`. */
  appVersion?: string;
  /** MongoDB URI used for trace persistence. Falls back to `MONGODB_URI`. */
  mongodbUri?: string | null;
  /** Explicit database name for trace storage. Defaults to the platform store. */
  databaseName?: string | null;
  /** Collection name for trace storage. Defaults to `"traces"`. */
  tracesCollectionName?: string;
  /** @deprecated Ignored. The org is taken from the `ORG_ID` env var, which the platform injects. */
  orgId?: string | null;
  /** @deprecated Ignored. The project is taken from the `PROJECT_ID` env var, which the platform injects. */
  projectId?: string | null;
}

/** Public options for `registerAndRun`. */
export interface RegisterAndRunOptions {
  /** Log level passed through to the underlying server. */
  logLevel?: string;
}

const SUPPORTED_RUN_KWARGS: ReadonlySet<string> = new Set(["logLevel"]);
// Reserved slot: kwargs listed here are accepted-but-ignored (warned in
// `normalizeRunKwargs`) rather than rejected as unexpected. Empty for now —
// no run kwarg is currently deprecated; future ones get added here.
const DEPRECATED_RUN_KWARGS: ReadonlySet<string> = new Set([]);

/**
 * Tenant Runtime SDK.
 *
 * Constructed once per process. `registerAndRun` wires the framework SDK's
 * graph builder and starts the per-mode Fastify server.
 */
export class TenantRuntime implements ITenantRuntime {
  readonly appName: string;
  readonly appVersion: string;
  readonly mode: RuntimeMode;
  readonly orgId: string | null;
  readonly projectId: string | null;
  readonly memoryWriter: MemoryWriter | null;

  graphBuilder: GraphBuilderLike | null = null;
  tools: Record<string, ServerToolFn> = {};
  toolDefinitions: Record<string, Record<string, unknown>> = {};

  private readonly agentConfig: RuntimeAgentConfig;
  private readonly mongodbUri: string | null;
  private readonly databaseNameOverride: string | undefined;
  private readonly tracesCollectionName: string;

  // Guards `shutdown()` against concurrent signal-handler invocations.
  private shutdownStarted = false;

  // In-flight tracing setup. `initTracing` kicks the async Mongo wiring off in
  // the constructor; `runAsync` awaits this before binding the server so the
  // global tracer provider is attached before the first request can land
  // (matching the deterministic effect of Python's synchronous `_setup_tracing`).
  private tracingReady: Promise<void> = Promise.resolve();

  constructor(opts: TenantRuntimeOptions = {}) {
    const {
      appName = "Agent",
      appVersion = "1.0.0",
      mongodbUri,
      databaseName,
      tracesCollectionName = "traces",
      orgId,
      projectId,
    } = opts;

    this.appName = appName;
    this.appVersion = appVersion;
    this.mode = getRuntimeMode();
    // org/project come exclusively from the platform-injected env vars. The
    // `orgId`/`projectId` options are deprecated and intentionally ignored —
    // honoring an author-supplied override is what caused the silent
    // tenant-mismatch bug. Use `||` not `??` so an empty-string env
    // var falls through to null; otherwise `ORG_ID=""` would slip past the
    // `=== null` scoping guards downstream.
    this.orgId = process.env["ORG_ID"] || null;
    this.projectId = process.env["PROJECT_ID"] || null;

    // Logging first so subsequent setup phases land in the configured sink.
    setupLogging({ appName, mode: this.mode });

    // Deprecation warnings emitted after setupLogging so the log line routes
    // through the configured sink (mirrors the Python ordering).
    if (orgId != null) {
      logger.warn(
        "TenantRuntime({ orgId }) is deprecated and ignored. " +
          "Set the ORG_ID environment variable instead; " +
          "this option will be removed in a future release.",
      );
    }
    if (projectId != null) {
      logger.warn(
        "TenantRuntime({ projectId }) is deprecated and ignored. " +
          "Set the PROJECT_ID environment variable instead; " +
          "this option will be removed in a future release.",
      );
    }

    this.agentConfig = loadRuntimeAgentConfig(undefined, tenantEnvVars());
    const configuredMemory = this.agentConfig.configuredFeature("memory");
    const memoryEnabled =
      configuredMemory ?? getEnvBool("ENABLE_MEMORY", false);
    this.memoryWriter = memoryEnabled
      ? new MemoryWriter(new AppBoundRuntime())
      : null;
    this.mongodbUri = mongodbUri ?? process.env["MONGODB_URI"] ?? null;
    this.databaseNameOverride = databaseName || undefined;
    this.tracesCollectionName = tracesCollectionName;

    // Tracing is always on so every runtime mode emits consistent spans.
    this.initTracing();

    logger.info(`TenantRuntime initialized: mode=${this.mode}, app=${appName}`);
  }

  // =========================================================================
  // Accessors
  // =========================================================================

  /** Parsed `agent.yaml` configuration. */
  getAgentConfig(): RuntimeAgentConfig {
    return this.agentConfig;
  }

  /** Resolved MongoDB URI: constructor option > `MONGODB_URI` env var > null. */
  getMongodbUri(): string | null {
    return this.mongodbUri;
  }

  // =========================================================================
  // Tracing setup
  // =========================================================================

  private initTracing(): void {
    // Python builds the Mongo collection synchronously here, but Node's
    // `mongodb` driver `connect()` is async — so the collection wiring lives
    // inside `setupTracing`. Kick it off from the constructor and retain the
    // promise: `runAsync` awaits `tracingReady` before binding the server, so
    // the global tracer provider is attached before the first request lands
    // (deterministic, unlike a bare fire-and-forget). A failed setup is logged
    // and swallowed so a degraded trace store never blocks serving traffic.
    this.tracingReady = setupTracing({
      serviceName: this.appName,
      mongodbUri: this.mongodbUri,
      mongodbDatabaseName: this.databaseNameOverride,
      mongodbCollectionName: this.tracesCollectionName,
    }).catch((e) => {
      // Scrub defensively: a setup failure whose message echoes the raw store
      // URI must never write its credentials to the log sink. Rejections may
      // carry non-Error values (string throws) — keep their text so it gets
      // scrubbed rather than dropped.
      logger.warn(
        `Tracing setup failed: ${scrubCredentials((e as Error)?.message ?? e)}`,
      );
    });
  }

  // =========================================================================
  // Agent / graph builder
  // =========================================================================

  getAgent(opts?: { callbacks?: BaseExecutionCallback[] }): BaseAgent {
    if (this.graphBuilder === null) {
      throw new Error(
        "No App registered. Call registerAndRun() with a BaseApp instance first.",
      );
    }
    if (typeof this.graphBuilder.getAgent !== "function") {
      throw new Error(
        "Graph builder must be a BaseApp instance with getAgent(). " +
          "Plain callables are not supported.",
      );
    }
    return this.graphBuilder.getAgent(opts);
  }

  warmUpAgent(): void {
    this.graphBuilder?.warmUp?.();
  }

  // =========================================================================
  // Context helpers
  // =========================================================================

  /**
   * Get the current user_id from execution context.
   *
   * @returns User ID from the current execution context, or null if not available
   */
  getCurrentUserId(): string | null {
    return getCurrentUserId();
  }

  getCurrentSessionId(): string | null {
    return getCurrentSessionId();
  }

  // =========================================================================
  // Tool registration
  // =========================================================================

  /**
   * Register a raw tool function and its metadata.
   *
   * This stores the function for Tool Pod execution and the metadata
   * for routing decisions. It does NOT create any framework-specific
   * tool objects — that is the responsibility of the framework SDK.
   *
   * @param name Tool name
   * @param func Raw tool function
   * @param metadata Tool metadata (is_local, network, timeout, etc.)
   */
  registerTool(
    name: string,
    func: ServerToolFn,
    metadata: Record<string, unknown>,
  ): void {
    this.tools[name] = func;
    this.toolDefinitions[name] = metadata;
    logger.debug(`Registered tool: ${name}`);
  }

  /**
   * Get metadata for a registered tool.
   *
   * @param name Tool name
   * @returns Tool metadata object (is_local, network, timeout, etc.)
   *   or empty object if tool not found.
   */
  getToolMetadata(name: string): Record<string, unknown> {
    return this.toolDefinitions[name] ?? {};
  }

  // =========================================================================
  // Shutdown
  // =========================================================================

  /**
   * Close the tracing MongoClient. Idempotent — safe to call from multiple
   * signal handlers. TS-only addition: Python's uvicorn handles SIGTERM
   * itself, so the Python `TenantRuntime` has no equivalent method.
   */
  async shutdown(): Promise<void> {
    if (this.shutdownStarted) return;
    this.shutdownStarted = true;
    logger.info("TenantRuntime shutdown initiated");
    try {
      await shutdownTracing();
    } catch (e) {
      logger.warn(`Tracing shutdown failed: ${(e as Error).message}`);
    }
    logger.info("TenantRuntime shutdown complete");
  }

  // =========================================================================
  // Run
  // =========================================================================

  /**
   * Register the graph builder and start the mode-specific server.
   *
   * Resolves once Fastify is listening — the open socket keeps the Node
   * event loop alive, matching the practical effect of Python's
   * `asyncio.run(uvicorn.serve())` blocking until shutdown.
   */
  async registerAndRun(
    graphBuilder?: GraphBuilderLike | null,
    options: RegisterAndRunOptions = {},
  ): Promise<void> {
    if (graphBuilder) {
      this.graphBuilder = graphBuilder;
      logger.info("Graph builder registered");
    }
    const normalized = this.normalizeRunKwargs(options);
    await this.runAsync(normalized);
  }

  private normalizeRunKwargs(opts: RegisterAndRunOptions): {
    logLevel?: string;
  } {
    const raw = opts as Record<string, unknown>;
    const provided = Object.keys(raw).filter((k) => raw[k] !== undefined);
    const unexpected = provided
      .filter(
        (k) => !SUPPORTED_RUN_KWARGS.has(k) && !DEPRECATED_RUN_KWARGS.has(k),
      )
      .sort();
    if (unexpected.length > 0) {
      if (unexpected.length === 1) {
        throw new TypeError(
          `registerAndRun() got an unexpected keyword argument '${unexpected[0] ?? ""}'`,
        );
      }
      const formatted = unexpected.map((k) => `'${k}'`).join(", ");
      throw new TypeError(
        `registerAndRun() got unexpected keyword arguments: ${formatted}`,
      );
    }

    // No deprecated run kwargs are currently accepted. When one is added to
    // DEPRECATED_RUN_KWARGS, warn-and-ignore it here.

    const result: { logLevel?: string } = {};
    if (opts.logLevel !== undefined) result.logLevel = opts.logLevel;
    return result;
  }

  private async runAsync(_opts: { logLevel?: string }): Promise<void> {
    // `logLevel` is accepted for Python parity (uvicorn's log_level) but is
    // not yet wired to Fastify, which has no equivalent knob. When the wiring
    // is added, resolve it as:
    //   _opts.logLevel ?? process.env["LOG_LEVEL"] ?? "info"

    // Block startup on tracing init so the global tracer provider is attached
    // before the listener accepts requests — otherwise early spans fall through
    // to the OTel no-op tracer and are silently dropped. Setup failures are
    // already logged-and-swallowed in `initTracing`, so this never rejects.
    await this.tracingReady;

    // Function mode is not a server: one invocation, report result, return.
    // Must branch before the server dispatch so no HTTP listener is bound.
    // graphBuilder.ready() runs inside ToolFunctionRunner after the request
    // is read, so startup failures still carry platform_trace_id.
    if (this.mode === RuntimeMode.TOOL_FUNCTION) {
      try {
        await new ToolFunctionRunner(this).run();
      } finally {
        // BatchSpanProcessor's periodic flush timer is unref'd (so it can't
        // pin a long-running server's event loop) — but that same unref
        // means nothing keeps a one-shot function-mode process alive long
        // enough for that timer to ever fire. Without an explicit shutdown
        // here, this invocation's spans (Mongo and OTLP)
        // are silently lost the moment the process exits. Runs even when
        // run() throws, so a failed invocation's spans aren't lost either.
        await this.shutdown();
      }
      return;
    }

    // Block startup on the graph builder's own async setup (e.g. MCP tool
    // discovery) so no request can land before tools are registered.
    await this.graphBuilder?.ready?.();

    const ServerClass = this.mode === RuntimeMode.AER ? AERServer : ToolServer;
    const server = new ServerClass(this);
    const port = getEnvInt("APP_PORT", server.defaultPort);
    const host = resolveListenHost();
    logger.info(`listener bind resolved: host='${host}' port=${port}`);

    // K8s sends SIGTERM on pod termination. Drain the Fastify server first so
    // in-flight /execute and /invoke_llm requests can finish — emitting their
    // terminal stream chunks and /executor/callback reports — before we tear
    // down tracing and exit. Python gets this for free from uvicorn's graceful
    // shutdown; Node needs it wired explicitly. `once` so a repeated signal
    // doesn't re-enter (shutdown() is itself re-entry safe). If draining
    // exceeds the grace budget we exit anyway so termination can't hang.
    const gracePeriodMs = shutdownGracePeriodMs(this.mode);
    const onSignal = (signal: NodeJS.Signals): void => {
      logger.info(
        `Received ${signal}, draining server (grace=${gracePeriodMs}ms)...`,
      );
      const drainAndExit = async (): Promise<void> => {
        try {
          const graceTimer = new Promise<void>((resolve) => {
            const t = setTimeout(resolve, gracePeriodMs);
            if (typeof t.unref === "function") t.unref();
          });
          await Promise.race([server.close(), graceTimer]);
        } catch (e) {
          logger.warn(`Server drain error: ${(e as Error).message}`);
        }
        await this.shutdown();
      };
      void drainAndExit().finally(() => process.exit(0));
    };
    process.once("SIGTERM", onSignal);
    process.once("SIGINT", onSignal);

    await server.run(host, port);
  }
}
