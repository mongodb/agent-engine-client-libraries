/**
 * Base server class for Runner SDK components.
 *
 * Provides common Fastify setup, health/root/metrics endpoints, and
 * startup/shutdown lifecycle hooks shared by all runtime modes.
 *
 * Mirrors Python's `BaseServer` (FastAPI/uvicorn → Fastify).
 */

import fastify, { type FastifyInstance } from "fastify";
import cors from "@fastify/cors";
import { ZodError } from "zod";
import type {
  BaseAgent,
  BaseExecutionCallback,
} from "@mongodb-js/agent-engine-sdk";
import { getLogger } from "../logger.js";
import { Metrics } from "../metrics.js";
import { HealthStatus, type HealthResponse } from "../models.js";
import { tracingStatus } from "../tracing/index.js";
import type { RuntimeAgentConfig } from "../agent_config.js";
import type { TurnMemoryWriter } from "../memory_writer.js";
import { resolveCorsPolicy } from "./cors.js";
import { registerAuthHook } from "./auth.js";
import { DrainRegistry, registerDrainRoute } from "./drain.js";
import { registerCallInterruptRoute } from "./callInterrupt.js";

const logger = getLogger("agent_engine_runner_shared.server.base");

export const DEFAULT_PORTS: Record<string, number> = {
  aer: 8001,
  tool: 8002,
  "memory-server": 8081,
};

/** Tool function signature — receives the full arguments Record, may return a Promise. */
export type ServerToolFn = (args: Record<string, unknown>) => unknown;

/** Minimal graph builder interface. */
export interface GraphBuilderLike {
  getAgent?(opts?: { callbacks?: BaseExecutionCallback[] }): BaseAgent;
  /**
   * Resolves once any async setup kicked off during construction (e.g. MCP
   * tool discovery) has finished registering tools. `TenantRuntime.runAsync()`
   * awaits this before binding the server, so no request can land before
   * setup completes. Optional — builders with no async setup can omit it.
   */
  ready?(): Promise<void>;
  /**
   * Build the agent graph before first use when explicitly requested. The
   * TypeScript AER does not call this synchronous hook during standby warming
   * because doing so would block the Node event loop and server health.
   */
  warmUp?(): void;
}

/**
 * Minimal runtime interface that server classes depend on.
 */
export interface ITenantRuntime {
  readonly appName: string;
  readonly appVersion?: string;
  readonly orgId: string | null;
  /**
   * Optional so pre-existing hand-written implementations (custom runtimes,
   * test doubles) keep typechecking across the SDK bump that introduced it.
   * The capability-advertise path treats absent and null alike by skipping.
   */
  readonly projectId?: string | null;
  /** Native-checkpoint turn writer, present only when Memory is enabled. */
  readonly memoryWriter?: TurnMemoryWriter | null;
  /** Graph builder instance — null if not set at startup. */
  readonly graphBuilder: GraphBuilderLike | null;
  /** Raw tool functions registered via `app.tool()`, keyed by name. */
  readonly tools: Record<string, ServerToolFn>;
  /** Tool metadata (description, is_local, etc.), keyed by name. */
  readonly toolDefinitions: Record<string, Record<string, unknown>>;
  /** Parsed `agent.yaml` configuration. */
  getAgentConfig(): RuntimeAgentConfig;
  /** Resolved MongoDB URI: constructor option > `MONGODB_URI` env var > null. */
  getMongodbUri(): string | null;
  /** Get a `BaseAgent` instance, optionally wired with execution callbacks. */
  getAgent(opts?: { callbacks?: BaseExecutionCallback[] }): BaseAgent;
  /** Explicit pre-build via the graph builder's `warmUp()`, if it has one. */
  warmUpAgent?(): void;
}

/**
 * Abstract base server for all Runner SDK runtime modes.
 *
 * Subclasses implement:
 * - `modeName` — 'aer' | 'tool' | 'memory-server'
 * - `registerRoutes(app)` — mode-specific Fastify routes
 * - `onStartup` / `onShutdown` — optional lifecycle hooks
 * - `getHealthDetails` — optional extra fields for /health
 */
export abstract class BaseServer {
  protected readonly runtime: ITenantRuntime;
  protected app: FastifyInstance | null = null;
  private _drainRegistry?: DrainRegistry;

  constructor(runtime: ITenantRuntime) {
    this.runtime = runtime;
  }

  /**
   * Execution-scoped drain state for POST /drain; work routes register
   * against it so a cancelled execution's in-flight work can be stopped.
   * Lazily created on the prototype so tests that skip the constructor
   * (`Object.create`) still get a working registry.
   */
  get drainRegistry(): DrainRegistry {
    return (this._drainRegistry ??= new DrainRegistry());
  }

  abstract get modeName(): string;

  get defaultPort(): number {
    return DEFAULT_PORTS[this.modeName] ?? 8000;
  }

  abstract registerRoutes(app: FastifyInstance): void;

  async onStartup(): Promise<void> {}
  async onShutdown(): Promise<void> {}

  getHealthDetails(): Record<string, unknown> {
    return {};
  }

  createApp(): FastifyInstance {
    const corsPolicy = resolveCorsPolicy(process.env["CORS_ALLOWED_ORIGINS"]);
    const app = fastify({ logger: false });

    void app.register(cors, {
      origin: corsPolicy.origin,
      credentials: corsPolicy.credentials,
      methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    });

    registerAuthHook(app, this.modeName);

    app.addHook("onReady", async () => {
      logger.info(`Starting ${this.modeName} server...`);
      await this.onStartup();
      logger.info(`${this.modeName} server started`);
    });

    app.addHook("onClose", async () => {
      logger.info(`Shutting down ${this.modeName} server...`);
      await this.onShutdown();
      logger.info(`${this.modeName} server stopped`);
    });

    this.registerCommonRoutes(app);
    this.registerRoutes(app);

    // Map request-validation failures to client errors. FastAPI/Pydantic
    // returns 422 for a malformed body; without this, a ZodError thrown by a
    // route's `Schema.parse(request.body)` becomes a generic 500, which OE and
    // other callers may retry as a transient infra failure rather than reject
    // as a bad payload. Non-Zod errors keep their explicit `statusCode`
    // (e.g. the 400/504 set by the AER execute path) or default to 500.
    app.setErrorHandler((error, _request, reply) => {
      if (error instanceof ZodError) {
        return reply
          .status(422)
          .send({ error: "Validation failed", detail: error.issues });
      }
      const err = error as { statusCode?: number; message?: string };
      const statusCode = err.statusCode ?? 500;
      const message = err.message ?? "Internal Server Error";
      if (statusCode >= 500) {
        logger.error({ err: error }, `Unhandled error: ${message}`);
      }
      return reply.status(statusCode).send({ error: message });
    });

    this.app = app;
    return app;
  }

  /**
   * Gracefully close the underlying Fastify server, draining in-flight
   * requests and running `onClose` hooks. No-op if the app was never created.
   */
  async close(): Promise<void> {
    if (this.app) {
      await this.app.close();
    }
  }

  protected registerCommonRoutes(app: FastifyInstance): void {
    registerDrainRoute(app, this.drainRegistry);
    registerCallInterruptRoute(app, this.drainRegistry);

    app.get("/health", async (): Promise<HealthResponse> => {
      // Probes check the HTTP status code (2xx = pass), not the body, so a
      // `degraded` status surfaces the state for observability without
      // triggering a pod restart — e.g. the trace store being unreachable at
      // startup while the pod keeps serving traffic.
      const tracing = tracingStatus();
      const status =
        tracing.database_exporter === "degraded"
          ? HealthStatus.DEGRADED
          : HealthStatus.HEALTHY;
      return {
        status,
        component: this.runtime.appName,
        mode: this.modeName,
        version: "1.0.0",
        details: { ...this.getHealthDetails(), tracing },
      };
    });

    app.get(
      "/",
      async (): Promise<Record<string, string>> => ({
        service: this.runtime.appName,
        mode: this.modeName,
        status: "running",
      }),
    );

    app.get(
      "/metrics",
      async (): Promise<Record<string, unknown>> => ({
        status: "ok",
        component: this.modeName,
        metrics: Metrics.getAll(),
        timestamp: new Date().toISOString(),
      }),
    );
  }

  async run(host = "0.0.0.0", port?: number): Promise<void> {
    const listenPort = port ?? this.defaultPort;
    const app = this.createApp();
    logger.info(`Starting ${this.modeName} server on ${host}:${listenPort}`);
    await app.listen({ host, port: listenPort });
  }
}
