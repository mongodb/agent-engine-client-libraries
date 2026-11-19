import {
  context,
  isSpanContextValid,
  propagation,
  trace,
} from "@opentelemetry/api";
import type { Context, Tracer } from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
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
  PlatformSpanExporter,
} from "./exporters.js";
import { markInputRecordedByCaller } from "./conversation.js";

const logger = getLogger("agent_engine_runner_shared.tracing");

const PLATFORM_RESOURCE_ATTR_PREFIX = "agentic_platform"; // open-source-refs:ignore — persisted data contract

const PLATFORM_RESOURCE_ENV_VARS: ReadonlyArray<readonly [string, string]> = [
  [`${PLATFORM_RESOURCE_ATTR_PREFIX}.org_id`, "ORG_ID"],
  [`${PLATFORM_RESOURCE_ATTR_PREFIX}.project_id`, "PROJECT_ID"],
  [`${PLATFORM_RESOURCE_ATTR_PREFIX}.workspace_id`, "APP_ID"],
  [`${PLATFORM_RESOURCE_ATTR_PREFIX}.deployment_id`, "DEPLOYMENT_ID"],
  [`${PLATFORM_RESOURCE_ATTR_PREFIX}.agent_id`, "AGENT_ID"],
  [`${PLATFORM_RESOURCE_ATTR_PREFIX}.runtime_mode`, "RUNNER_MODE"],
  [`${PLATFORM_RESOURCE_ATTR_PREFIX}.environment`, "AGENT_ENGINE_ENVIRONMENT"],
];

/**
 * Build the provider resource attributes for `serviceName`.
 *
 * Ported from Python's `_build_resource_attributes`. The platform resource
 * attribute keys carry process-level platform identity; a missing env var omits
 * its key rather than writing an empty value.
 */
export function buildResourceAttributes(
  serviceName: string,
): Record<string, string> {
  const attributes: Record<string, string> = { "service.name": serviceName };
  for (const [resourceKey, envVar] of PLATFORM_RESOURCE_ENV_VARS) {
    const value = process.env[envVar];
    if (value) attributes[resourceKey] = value;
  }
  return attributes;
}

const MONGO_APP_NAME = "Agent Engine";

let tracerProvider: TracerProvider | null = null;
let tracingMongoClient: MongoClient | null = null;
let setupPromise: Promise<void> | null = null;
let registeredContextManager: AsyncLocalStorageContextManager | null = null;

let mongodbExporterAttached = false;
let mongodbDegraded = false;
let retryTimer: ReturnType<typeof setTimeout> | null = null;
let retryStopped = false;
let lazyMongoExporter: LazyMongoSpanExporter | null = null;
let otlpSpanExporter: ContentPolicyOTLPSpanExporter | null = null;

let platformExporter: PlatformSpanExporter | null = null;

const PLATFORM_BSP_MAX_QUEUE_SIZE = 2048;
const PLATFORM_BSP_MAX_EXPORT_BATCH_SIZE = 512;
const PLATFORM_BSP_SCHEDULE_DELAY_MILLIS = 2000;

export class ExecutionContextSpanProcessor implements SpanProcessor {
  onStart(span: Span, _parentContext: Context): void {
    markInputRecordedByCaller(span);

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

  drain(): Promise<void> {
    return this.inner?.drain() ?? Promise.resolve();
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

async function closeMongoClientIgnoringErrors(
  client: MongoClient,
): Promise<void> {
  try {
    await client.close();
  } catch {
    return;
  }
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
      appName: MONGO_APP_NAME,
    });
    await client.connect();
    await client.db("admin").command({ ping: 1 });
    const { resolveStoreDbName } = await import("../db_config.js");
    const dbName = await resolveStoreDbName(
      client,
      args.mongodbDatabaseName ?? undefined,
    );
    tracingMongoClient = client;
    return {
      collection: client.db(dbName).collection(collName),
      errorMessage: null,
    };
  } catch (err) {
    if (client !== undefined) {
      await closeMongoClientIgnoringErrors(client);
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
  return String(message).replace(
    /(mongodb(?:\+srv)?:\/\/)((?:(?!mongodb(?:\+srv)?:\/\/)[^"'])*)@(?=[^\s@/?#"'][^\s@?"']*(?:[/?#:\s"']|$))/gi,
    "$1***@",
  );
}

function otlpEndpointConfigured(): boolean {
  return Boolean(
    process.env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] ||
    process.env["OTEL_EXPORTER_OTLP_ENDPOINT"],
  );
}

const OTLP_EXPORT_TIMEOUT_MILLIS = 3000;

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
    otlpSpanExporter = otlpExporter;
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

function getRetryIntervalSeconds(): number {
  const raw = Number(process.env["AGENTIC_TRACE_STORE_RETRY_INTERVAL_SECONDS"]);
  return Number.isFinite(raw) && raw > 0 ? raw : 30;
}

export async function setupTracing(args: SetupTracingArgs = {}): Promise<void> {
  if (tracerProvider !== null) {
    logger.debug("Tracing already initialized");
    return;
  }
  if (setupPromise !== null) return setupPromise;
  setupPromise = doSetupTracing(args).finally(() => {
    setupPromise = null;
  });
  return setupPromise;
}

async function doSetupTracing(args: SetupTracingArgs): Promise<void> {
  const serviceName = args.serviceName ?? "runner-sdk";
  const spanProcessors: SpanProcessor[] = [new ExecutionContextSpanProcessor()];

  const oeUrl = (process.env["OE_URL"] ?? "").trim();
  const otlpConfigured = otlpEndpointConfigured();
  if (oeUrl) {
    attachPlatformExporter(spanProcessors, oeUrl);
  } else {
    const { collection, errorMessage } = await resolveTracingCollection(args);
    if (args.mongodbUri) {
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
    } else if (!otlpConfigured) {
      logger.warn(
        "Tracing: no trace destination configured " +
          "(OE_URL unset, no MongoDB collection, no OTLP endpoint)",
      );
    }
  }

  const otlpProcessor = await buildOtlpSpanProcessor();
  if (otlpProcessor !== null) spanProcessors.push(otlpProcessor);

  const provider = new BasicTracerProvider({
    resource: resourceFromAttributes(buildResourceAttributes(serviceName)),
    spanProcessors,
  });

  const contextManager = new AsyncLocalStorageContextManager();
  contextManager.enable();
  if (context.setGlobalContextManager(contextManager)) {
    registeredContextManager = contextManager;
  } else {
    contextManager.disable();
    logger.debug(
      "A global context manager is already registered; keeping the existing one",
    );
  }

  trace.setGlobalTracerProvider(provider);
  tracerProvider = provider;

  propagation.setGlobalPropagator(new W3CTraceContextPropagator());

  runInstrumentor();
  logger.info({ serviceName }, "Tracing enabled");
}

function attachPlatformExporter(
  spanProcessors: SpanProcessor[],
  oeUrl: string,
): void {
  const exporter = new PlatformSpanExporter(oeUrl);
  spanProcessors.push(
    new BatchSpanProcessor(exporter, {
      maxQueueSize: PLATFORM_BSP_MAX_QUEUE_SIZE,
      maxExportBatchSize: PLATFORM_BSP_MAX_EXPORT_BATCH_SIZE,
      scheduledDelayMillis: PLATFORM_BSP_SCHEDULE_DELAY_MILLIS,
    }),
  );
  platformExporter = exporter;
  logger.info({ url: exporter.url }, "Tracing: platform exporter enabled");
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

const SHUTDOWN_TRACE_FLUSH_TIMEOUT_MILLIS = 5_000;

export async function shutdownTracing(): Promise<void> {
  retryStopped = true;
  if (retryTimer !== null) {
    clearTimeout(retryTimer);
    retryTimer = null;
  }
  if (tracerProvider !== null) {
    await forceFlushTracing(SHUTDOWN_TRACE_FLUSH_TIMEOUT_MILLIS);
    await tracerProvider.shutdown();
    tracerProvider = null;
    logger.info("Tracing shutdown complete");
  }
  if (registeredContextManager !== null) {
    context.disable();
    registeredContextManager = null;
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
  mongodbExporterAttached = false;
  mongodbDegraded = false;
  lazyMongoExporter = null;
  platformExporter = null;
  otlpSpanExporter = null;
}

/**
 * Drain buffered spans and wait for in-flight exporter work, bounded by
 * `timeoutMillis`.
 *
 * Called from the session-finish path before the OE releases the pod, so a
 * turn's spans are not lost to a zero-grace teardown. `provider.forceFlush()`
 * drains each processor's buffer, but a batch processor ships on a timer and
 * does not track an export it already started: a span it removed from the
 * buffer and is still shipping is invisible to it, so the flush can resolve
 * before that write lands. Await each exporter's own in-flight work too — the
 * platform exporter, the Mongo store, and (when configured) OTLP — so the
 * caller does not release on a false flush. Returns true when tracing is not
 * initialized or the flush completed within the timeout; false when it timed
 * out or raised.
 */
export async function forceFlushTracing(
  timeoutMillis = 5000,
): Promise<boolean> {
  const provider = tracerProvider;
  if (provider === null) return true;
  try {
    const providerFlushed = provider.forceFlush();
    const exportsDrained = Promise.all([
      platformExporter?.forceFlush() ?? Promise.resolve(),
      lazyMongoExporter?.drain() ?? Promise.resolve(),
      otlpSpanExporter?.drain() ?? Promise.resolve(),
    ]);
    const settled = Promise.allSettled([providerFlushed, exportsDrained]);
    if (
      !(await raceWithTimeout(
        settled.then(() => undefined),
        timeoutMillis,
      ))
    ) {
      return false;
    }
    return (await settled).every((result) => result.status === "fulfilled");
  } catch (err) {
    logger.debug({ err: (err as Error).message }, "Tracing force flush failed");
    return false;
  }
}

function raceWithTimeout(
  promise: Promise<void>,
  timeoutMillis: number,
): Promise<boolean> {
  return new Promise<boolean>((resolve) => {
    const timer = setTimeout(() => resolve(false), timeoutMillis);
    if (typeof timer.unref === "function") timer.unref();
    promise.then(
      () => {
        clearTimeout(timer);
        resolve(true);
      },
      () => {
        clearTimeout(timer);
        resolve(false);
      },
    );
  });
}

/**
 * Exporter states reported by `/health`.
 *
 * `database_exporter` is `"attached"` when a MongoDB processor is live,
 * `"degraded"` while a configured store is unreachable and retrying, or
 * `"disabled"` when no store URI was configured.
 *
 * `platform_exporter` is `"attached"` when traces are sent to the OE,
 * `"degraded"` after an incomplete export, or `"disabled"` when `OE_URL` is
 * unset. OTLP may still be configured in the disabled state.
 */
export function tracingStatus(): {
  database_exporter: string;
  platform_exporter: string;
} {
  const platform =
    platformExporter === null
      ? "disabled"
      : platformExporter.lastError !== null
        ? "degraded"
        : "attached";

  let database: string;
  if (mongodbExporterAttached) database = "attached";
  else if (mongodbDegraded) database = "degraded";
  else database = "disabled";

  return { database_exporter: database, platform_exporter: platform };
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

function scheduleRetry(
  attempt: () => void,
  intervalMs: number,
): ReturnType<typeof setTimeout> {
  const timer = setTimeout(attempt, intervalMs);
  if (typeof timer.unref === "function") timer.unref();
  return timer;
}

function startMongoRetry(args: SetupTracingArgs): void {
  if (mongodbExporterAttached) return;
  if (retryTimer !== null) return;
  retryStopped = false;
  const intervalMs = getRetryIntervalSeconds() * 1000;
  const runAttempt = async (): Promise<void> => {
    retryTimer = null;
    if (mongodbExporterAttached || retryStopped) return;
    const { collection, errorMessage } = await resolveTracingCollection(args);
    if (retryStopped) {
      if (tracingMongoClient !== null) {
        await closeMongoClientIgnoringErrors(tracingMongoClient);
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
