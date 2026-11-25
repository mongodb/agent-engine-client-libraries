import {
  context,
  isSpanContextValid,
  propagation,
  trace,
} from "@opentelemetry/api";
import type { Context, Tracer } from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
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

import {
  getCurrentExecutionId,
  getCurrentSessionId,
  getCurrentWorkspaceId,
  runOutsideExecution,
} from "../context.js";
import { getInstrumentor } from "../hooks.js";
import { getLogger } from "../logger.js";
import {
  ContentPolicyOTLPSpanExporter,
  getContentCaptureMode,
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

let tracerProvider: TracerProvider | null = null;
let setupPromise: Promise<void> | null = null;
let registeredContextManager: AsyncLocalStorageContextManager | null = null;

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

export interface SetupTracingArgs {
  serviceName?: string;
}

/**
 * Mask the userinfo of any `mongodb://` / `mongodb+srv://` URI embedded in a
 * credential-bearing error message.
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
 * it must never throw itself, or a logged setup failure turns into a
 * startup failure.
 */
export function scrubCredentials(message: unknown): string {
  return String(message).replace(
    /(mongodb(?:\+srv)?:\/\/)((?:(?!mongodb(?:\+srv)?:\/\/)[^"'])*)@(?=[^\s@/?#"'][^\s@?"']*(?:[/?#:\s"']|$))/gi,
    "$1***@",
  );
}

function exportOutsideExecution(exporter: SpanExporter): SpanExporter {
  return {
    export: (spans, resultCallback) =>
      runOutsideExecution(() => exporter.export(spans, resultCallback)),
    shutdown: () => exporter.shutdown(),
    forceFlush: exporter.forceFlush?.bind(exporter),
  };
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
      await import("@opentelemetry/exporter-trace-otlp-proto");
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
    return new BatchSpanProcessor(exportOutsideExecution(otlpExporter));
  } catch (err) {
    logger.warn(
      { err: (err as Error).message },
      "Failed to set up OTLP exporter — OTLP export disabled, platform tracing unaffected",
    );
    return null;
  }
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
  if (oeUrl) {
    attachPlatformExporter(spanProcessors, oeUrl);
  } else if (!otlpEndpointConfigured()) {
    logger.warn(
      "Tracing: no trace destination configured (OE_URL unset, no OTLP endpoint)",
    );
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
    new BatchSpanProcessor(exportOutsideExecution(exporter), {
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
 * platform exporter and (when configured) OTLP — so the
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
 * Exporter state reported by `/health`.
 *
 * `platform_exporter` is `"attached"` when traces are sent to the OE,
 * `"degraded"` after an incomplete export, or `"disabled"` when `OE_URL` is
 * unset. OTLP may still be configured in the disabled state.
 */
export function tracingStatus(): { platform_exporter: string } {
  const platform =
    platformExporter === null
      ? "disabled"
      : platformExporter.lastError !== null
        ? "degraded"
        : "attached";
  return { platform_exporter: platform };
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
