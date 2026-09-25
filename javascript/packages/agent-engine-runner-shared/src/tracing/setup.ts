/**
 * Tracing setup for Runner SDK.
 *
 * Ported from `agent_engine_runner_shared/tracing/setup.py`. Initializes OpenTelemetry
 * with `BasicTracerProvider` and optionally a Mongo exporter. Calls the
 * framework instrumentor hook
 * registered via `agent_engine_runner_shared/hooks.ts:registerInstrumentor`.
 */

import { isSpanContextValid, propagation, trace } from "@opentelemetry/api";
import type { Context, Tracer } from "@opentelemetry/api";
import type { ExportResult } from "@opentelemetry/core";
import { W3CTraceContextPropagator } from "@opentelemetry/core";
import { resourceFromAttributes } from "@opentelemetry/resources";
import {
  BasicTracerProvider,
  BatchSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import type {
  BasicTracerProvider as TracerProvider,
  ReadableSpan,
  Span,
  SpanExporter,
  SpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import type { Collection, MongoClient } from "mongodb";

import {
  getCurrentExecutionId,
  getCurrentSessionId,
  getCurrentWorkspaceId,
} from "../context.js";
import { getInstrumentor } from "../hooks.js";
import { getLogger } from "../logger.js";
import {
  ContentPolicyOTLPSpanExporter,
  getContentCaptureMode,
  MongoDBSpanExporter,
} from "./exporters.js";

const logger = getLogger("agent_engine_runner_shared.tracing");

let tracerProvider: TracerProvider | null = null;
// MongoClient instance built by `resolveTracingCollection`. Held here so
// `shutdownTracing` can close the pool — `MongoDBSpanExporter.shutdown()`
// is a no-op, so without this reference the client (and its connection
// pool, monitor threads, heartbeat intervals) would leak until process exit.
let tracingMongoClient: MongoClient | null = null;
// In-flight initialization promise. `setupTracing` sets `tracerProvider` late
// (after awaiting dynamic imports + the MongoClient connect), so the entry
// guard alone can't stop concurrent callers from both running full setup —
// which would open two Mongo clients and leak the first when the second
// overwrites `tracingMongoClient`. Sharing this promise serializes them.
let setupPromise: Promise<void> | null = null;

// MongoDB trace-store exporter state — mirrors Python's setup.py. The store
// connection can fail at pod startup (e.g. a transient Atlas TLS error); without
// retry the pod permanently lost database tracing for its whole lifetime.
// `mongodbExporterAttached` tracks whether a MongoDB span processor
// is live; `mongodbDegraded` marks a configured-but-unreachable store so /health
// can surface DEGRADED. The retry timer re-attempts the connection in the
// background and attaches the exporter on recovery.
let mongodbExporterAttached = false;
let mongodbDegraded = false;
let retryTimer: ReturnType<typeof setTimeout> | null = null;
// Stop signal for the background retry loop, mirroring Python's
// `_retry_stop_event`. `shutdownTracing` sets this so an in-flight retry
// attempt (past `retryTimer = null`, mid-`await resolveTracingCollection`)
// can't reschedule a new timer or overwrite `tracingMongoClient` with a
// client whose pool shutdown already closed — both would leak past shutdown.
let retryStopped = false;
// The lazy Mongo exporter registered with the tracer provider at setup time
// when a store URI is configured. `BasicTracerProvider` (v2.x) takes span
// processors only via its constructor — there is no `addSpanProcessor`, and
// `setGlobalTracerProvider` rejects re-registration — so the provider is built
// once with the lazy Mongo processor and the collection is wired in later by
// `attachMongoTracing` once the store recovers. While the collection
// is unset, `export` is a no-op.
let lazyMongoExporter: LazyMongoSpanExporter | null = null;

class ExecutionContextSpanProcessor implements SpanProcessor {
  onStart(span: Span, _parentContext: Context): void {
    const sessionId = getCurrentSessionId();
    const attributes = {
      "execution.id": getCurrentExecutionId(),
      "session.id": sessionId,
      "thread.id": sessionId,
      "workspace.id": getCurrentWorkspaceId(),
    };
    for (const [key, value] of Object.entries(attributes)) {
      if (value) span.setAttribute(key, value);
    }
  }

  onEnd(_span: ReadableSpan): void {}

  shutdown(): Promise<void> {
    return Promise.resolve();
  }

  forceFlush(): Promise<void> {
    return Promise.resolve();
  }
}

/** A SpanExporter whose Mongo collection is wired in after construction. */
class LazyMongoSpanExporter implements SpanExporter {
  private inner: MongoDBSpanExporter | null = null;

  setCollection(collection: Collection): void {
    this.inner = new MongoDBSpanExporter(collection);
  }

  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ): void {
    if (this.inner === null || spans.length === 0) {
      resultCallback({ code: 0 });
      return;
    }
    this.inner.export(spans, resultCallback);
  }

  shutdown(): Promise<void> {
    return this.inner?.shutdown() ?? Promise.resolve();
  }

  forceFlush(): Promise<void> {
    return this.inner?.forceFlush() ?? Promise.resolve();
  }
}

export interface SetupTracingArgs {
  serviceName?: string;
  /**
   * URI for the trace store. When provided, `setupTracing` builds the
   * MongoClient, pings `admin`, and resolves the `(database, collection)`
   * itself.
   *
   * Python's `TenantRuntime._setup_tracing` does this synchronously and
   * passes the collection in; Node's driver is fundamentally async so the
   * connect + ping live here instead. After a connection failure, a background
   * retry attaches the MongoDB exporter once the store recovers —
   * same recovery shape as Python.
   */
  mongodbUri?: string | null;
  /** Explicit database override. Omit to use the platform store default. */
  mongodbDatabaseName?: string | null;
  /** Collection name within `mongodbDatabaseName`. Defaults to `"traces"`. */
  mongodbCollectionName?: string;
}

async function resolveTracingCollection(
  args: SetupTracingArgs,
): Promise<{ collection: Collection | null; errorMessage: string | null }> {
  if (!args.mongodbUri) {
    return { collection: null, errorMessage: null };
  }
  const collName = args.mongodbCollectionName ?? "traces";

  let client: MongoClient | undefined;
  try {
    const { MongoClient } = await import("mongodb");
    client = new MongoClient(args.mongodbUri, {
      serverSelectionTimeoutMS: 5000,
    });
    await client.connect();
    // Mirrors Python's `client.admin.command("ping")` — surfaces a dead
    // server immediately rather than waiting for the first span insert.
    await client.db("admin").command({ ping: 1 });
    // Resolve the per-project-scoped store DB now that a client exists, so
    // traces land in the same database the OE reads.
    const { resolveStoreDbName } = await import("../db_config.js");
    const dbName = await resolveStoreDbName(
      client,
      args.mongodbDatabaseName ?? undefined,
    );
    // Retain the client so `shutdownTracing` can close it.
    tracingMongoClient = client;
    return {
      collection: client.db(dbName).collection(collName),
      errorMessage: null,
    };
  } catch (err) {
    // Close on failure so a half-open client doesn't leak. `connect()` may
    // not have completed, but `close()` is safe to call on a never-connected
    // client per the driver docs. The scrubbed error message is returned for
    // the caller to log at the severity it chooses (ERROR at startup, DEBUG
    // on retries) — this function stays log-free so retries don't spam.
    if (client !== undefined) {
      try {
        await client.close();
      } catch {
        // best-effort cleanup
      }
    }
    return {
      collection: null,
      errorMessage: scrubCredentials((err as Error).message),
    };
  }
}

/**
 * Mask the userinfo of any `mongodb://` / `mongodb+srv://` URI embedded in a
 * credential-bearing error message (driver parse/connect errors echo the raw
 * connection string).
 *
 * The userinfo run is GREEDY up to the LAST '@' that a host-shaped token
 * follows, so passwords containing an unescaped '@', space, or '/' are still
 * fully masked — the old `:\/\/[^@\s]+@` pattern stopped at the first '@' and
 * could not cross whitespace, leaking the password tail (or the whole
 * userinfo) to the centralized log sink. The run is tempered so it never
 * crosses into a second URI's scheme; when message text after the URI
 * contains its own '@' the mask may extend to it — over-redaction is the
 * fail-closed direction, a leak is not recoverable. Keep in sync with
 * `scrub_credentials` in Python's `agent_engine_runner_shared/tracing/setup.py`.
 *
 * Accepts `unknown` and coerces: error paths hand this whatever a rejection
 * carried (string throws, objects without a string `message`, undefined) —
 * it must never throw itself, or a degraded trace store turns into a
 * startup failure.
 */
export function scrubCredentials(message: unknown): string {
  // The run is [^"'] (any char except quotes): it still crosses embedded
  // newlines in a password (a --stdin-set secret can carry one), but stops
  // at a JSON string boundary — a credential echoed inside a compact JSON
  // body must not fold later siblings into the mask.
  return String(message).replace(
    /(mongodb(?:\+srv)?:\/\/)((?:(?!mongodb(?:\+srv)?:\/\/)[^"'])*)@(?=[^\s@/?#"'][^\s@?"']*(?:[/?#:\s"']|$))/gi,
    "$1***@",
  );
}

/**
 * Whether an OTLP endpoint was explicitly configured.
 *
 * `OTLPTraceExporter` defaults to `http://localhost:4318` when no endpoint
 * env var is set, which would silently start exporting to a collector that
 * doesn't exist in this cell architecture. Gate on explicit configuration so
 * "no endpoint configured" truly means "no OTLP traffic." Mirrors Python's
 * `_otlp_endpoint_configured`.
 */
function otlpEndpointConfigured(): boolean {
  return Boolean(
    process.env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] ||
    process.env["OTEL_EXPORTER_OTLP_ENDPOINT"],
  );
}

/**
 * Upper bound (ms) on how long a `TenantRuntime.shutdown()` flush can spend
 * waiting on the OTLP exporter. `TOOL_FUNCTION` mode calls `shutdown()`
 * synchronously on every invocation (its process-exit flush timer is
 * unref'd and would otherwise never fire), so an unreachable/slow collector
 * would otherwise add up to the OTel default export timeout (10s) of tail
 * latency to *every* one-shot invocation. Deliberately short relative to
 * that default.
 */
const OTLP_EXPORT_TIMEOUT_MILLIS = 3000;

/**
 * Build the OTLP export span processor, if an endpoint was explicitly
 * configured. Additive and never replaces Mongo. Content-capture redaction
 * (`AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE`) applies only to this path.
 * Guarded: a missing dependency or bad endpoint must not take down tracing
 * setup entirely — mirrors Python's try/except around the same block.
 */
async function buildOtlpSpanProcessor(): Promise<SpanProcessor | null> {
  if (!otlpEndpointConfigured()) return null;
  try {
    const { OTLPTraceExporter } =
      await import("@opentelemetry/exporter-trace-otlp-http");
    const contentCaptureMode = getContentCaptureMode();
    if (contentCaptureMode === "full") {
      logger.warn(
        "AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE=full — OTLP export includes " +
          "unredacted prompt/completion/tool content. This should only be " +
          "set intentionally (e.g. local debugging).",
      );
    }
    const otlpExporter = new ContentPolicyOTLPSpanExporter(
      new OTLPTraceExporter({ timeoutMillis: OTLP_EXPORT_TIMEOUT_MILLIS }),
      contentCaptureMode,
    );
    logger.info({ contentCaptureMode }, "Tracing: OTLP output enabled");
    return new BatchSpanProcessor(otlpExporter);
  } catch (err) {
    logger.warn(
      { err: (err as Error).message },
      "Failed to set up OTLP exporter — OTLP export disabled, MongoDB tracing unaffected",
    );
    return null;
  }
}

/** Retry interval (seconds) for the background trace-store reconnection. */
function getRetryIntervalSeconds(): number {
  const raw = Number(process.env["AGENTIC_TRACE_STORE_RETRY_INTERVAL_SECONDS"]);
  return Number.isFinite(raw) && raw > 0 ? raw : 30;
}

export async function setupTracing(args: SetupTracingArgs = {}): Promise<void> {
  if (tracerProvider !== null) {
    logger.debug("Tracing already initialized");
    return;
  }
  // Serialize concurrent callers onto the first call's promise. Cleared in
  // `finally` so a degraded run (e.g. OTEL deps absent) can be retried later.
  if (setupPromise !== null) return setupPromise;
  setupPromise = doSetupTracing(args).finally(() => {
    setupPromise = null;
  });
  return setupPromise;
}

async function doSetupTracing(args: SetupTracingArgs): Promise<void> {
  const serviceName = args.serviceName ?? "runner-sdk";
  const spanProcessors: SpanProcessor[] = [new ExecutionContextSpanProcessor()];

  const { collection, errorMessage } = await resolveTracingCollection(args);
  if (args.mongodbUri) {
    // A store URI was configured — register the lazy Mongo exporter up front
    // (the provider is built once, below, with it in the processor list). If
    // the store was reachable at startup, wire the collection in immediately;
    // otherwise leave it unset and retry in the background.
    const lazy = new LazyMongoSpanExporter();
    lazyMongoExporter = lazy;
    spanProcessors.push(new BatchSpanProcessor(lazy));
    if (collection) {
      lazy.setCollection(collection);
      mongodbExporterAttached = true;
      mongodbDegraded = false;
      logger.info(
        { db: collection.dbName, collection: collection.collectionName },
        "Tracing: MongoDB output enabled",
      );
    } else {
      mongodbDegraded = true;
      const intervalSec = getRetryIntervalSeconds();
      logger.error(
        `Trace store unavailable at startup; will retry every ${intervalSec}s. Error: ${errorMessage ?? "unknown"}`,
      );
      startMongoRetry(args);
    }
  }

  const otlpProcessor = await buildOtlpSpanProcessor();
  if (otlpProcessor !== null) spanProcessors.push(otlpProcessor);

  const provider = new BasicTracerProvider({
    resource: resourceFromAttributes({
      "service.name": serviceName,
    }),
    spanProcessors,
  });

  trace.setGlobalTracerProvider(provider);
  tracerProvider = provider;

  // Restrict the global propagator to W3C trace-context only, mirroring
  // Python's set_global_textmap(TraceContextTextMapPropagator()). The API's
  // own default (once anything registers a composite propagator, e.g. an
  // auto-instrumentation package) also propagates Baggage, which would leak
  // arbitrary unredacted key/value pairs onto every outbound call that goes
  // through the global propagator once something starts writing to it.
  // AER/Tool-Pod -> OE calls (tls_client.ts's fetchPlatform) already sidestep
  // this by injecting through their own local propagator instance regardless
  // of this setting, but this closes the gap for any future code path that
  // uses the global API directly.
  propagation.setGlobalPropagator(new W3CTraceContextPropagator());

  runInstrumentor();
  logger.info({ serviceName }, "Tracing enabled");
}

export function runInstrumentor(): void {
  const instrumentor = getInstrumentor();
  if (instrumentor === null) {
    logger.warn(
      "No instrumentor hook registered. Framework SDK should call registerInstrumentor() to enable tracing.",
    );
    return;
  }
  try {
    instrumentor();
    logger.info("Framework instrumentation enabled");
  } catch (err) {
    logger.warn(
      { err: (err as Error).message },
      "Framework instrumentor failed — tracing disabled",
    );
  }
}

export async function shutdownTracing(): Promise<void> {
  // Signal in-flight retry attempts to stop before tearing down the provider
  // they attach exporters to. Set before clearing the timer so an attempt
  // that resumes mid-shutdown sees the flag and cleans up its own client
  // rather than rescheduling or overwriting `tracingMongoClient`.
  retryStopped = true;
  // Cleared unconditionally so a partially-failed setup
  // (provider never created but retry scheduled) still cleans up.
  if (retryTimer !== null) {
    clearTimeout(retryTimer);
    retryTimer = null;
  }
  if (tracerProvider !== null) {
    await tracerProvider.shutdown();
    tracerProvider = null;
    logger.info("Tracing shutdown complete");
  }
  if (tracingMongoClient !== null) {
    try {
      await tracingMongoClient.close();
    } catch (err) {
      logger.warn(
        `Tracing MongoClient close failed: ${(err as Error).message}`,
      );
    }
    tracingMongoClient = null;
  }
  // Reset store-exporter state so a fresh setupTracing() in the same
  // process (e.g. tests) starts clean.
  mongodbExporterAttached = false;
  mongodbDegraded = false;
  lazyMongoExporter = null;
}

/**
 * Snapshot of the database trace-store exporter for /health. Mirrors Python's
 * `tracing_status`: `"attached"` (MongoDB span processor live), `"degraded"`
 * (store URI configured but unreachable at startup, retrying), or `"disabled"`
 * (no store URI configured, not a degradation).
 */
export function tracingStatus(): { database_exporter: string } {
  if (mongodbExporterAttached) return { database_exporter: "attached" };
  if (mongodbDegraded) return { database_exporter: "degraded" };
  return { database_exporter: "disabled" };
}

/**
 * Wire the recovered trace-store collection into the lazy Mongo exporter
 * registered at setup time. Called from the background retry loop
 * once a failed connection recovers. Idempotent — a no-op if already attached.
 * New spans created after this call route to the store; spans emitted while the
 * store was unreachable are not retroactively recovered.
 */
export function attachMongoTracing(collection: Collection): void {
  if (mongodbExporterAttached) return;
  if (lazyMongoExporter === null) {
    logger.warn(
      "Cannot attach MongoDB tracing: no store URI was configured at setup",
    );
    return;
  }
  lazyMongoExporter.setCollection(collection);
  mongodbExporterAttached = true;
  mongodbDegraded = false;
  if (retryTimer !== null) {
    clearTimeout(retryTimer);
    retryTimer = null;
  }
  logger.info(
    { db: collection.dbName, collection: collection.collectionName },
    "Tracing: MongoDB output attached (trace store recovered)",
  );
}

/**
 * Schedule a retry attempt. The timer is unref'd: TOOL_FUNCTION runtimes
 * return after a single invocation without calling shutdownTracing, so a
 * referenced retry timer would pin the event loop and keep the supposedly
 * one-shot process alive while the store is down (Python's retry thread is a
 * daemon for the same reason). The retry is best-effort background work — it
 * must never be the handle keeping the process running.
 */
function scheduleRetry(
  attempt: () => void,
  intervalMs: number,
): ReturnType<typeof setTimeout> {
  const timer = setTimeout(attempt, intervalMs);
  if (typeof timer.unref === "function") timer.unref();
  return timer;
}

/** Retry the trace-store connection in the background. */
function startMongoRetry(args: SetupTracingArgs): void {
  if (mongodbExporterAttached) return;
  if (retryTimer !== null) return;
  retryStopped = false;
  const intervalMs = getRetryIntervalSeconds() * 1000;
  const runAttempt = async (): Promise<void> => {
    retryTimer = null;
    if (mongodbExporterAttached || retryStopped) return;
    const { collection, errorMessage } = await resolveTracingCollection(args);
    // Shutdown may have raced with this attempt while it was awaiting the
    // store connection. `resolveTracingCollection` already assigned a fresh
    // client to `tracingMongoClient` on success — close it so its pool
    // doesn't leak, then bail without rescheduling.
    if (retryStopped) {
      if (tracingMongoClient !== null) {
        try {
          await tracingMongoClient.close();
        } catch {
          // best-effort cleanup
        }
        tracingMongoClient = null;
      }
      return;
    }
    if (collection) {
      attachMongoTracing(collection);
    } else {
      logger.debug(`Trace store retry failed: ${errorMessage ?? "unknown"}`);
      if (!mongodbExporterAttached && !retryStopped) {
        retryTimer = scheduleRetry(runAttempt, intervalMs);
      }
    }
  };
  retryTimer = scheduleRetry(runAttempt, intervalMs);
}

export function getCurrentTraceContext(): {
  traceId: string | null;
  spanId: string | null;
} {
  const span = trace.getActiveSpan();
  if (!span) return { traceId: null, spanId: null };
  const ctx = span.spanContext();
  if (!isSpanContextValid(ctx)) return { traceId: null, spanId: null };
  return { traceId: ctx.traceId, spanId: ctx.spanId };
}

export function getTracer(name = "runner-sdk"): Tracer {
  return trace.getTracer(name);
}
