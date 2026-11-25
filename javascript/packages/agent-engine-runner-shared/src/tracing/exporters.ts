import type { Attributes } from "@opentelemetry/api";
import type { ExportResult } from "@opentelemetry/core";
import {
  ProtobufTraceSerializer,
  type IExportTraceServiceResponse,
  type ISerializer,
} from "@opentelemetry/otlp-transformer";
import type { ReadableSpan, SpanExporter } from "@opentelemetry/sdk-trace-base";

import { getLogger } from "../logger.js";
import { fetchPlatform, getFetchOptionsWithTLS } from "../tls_client.js";

import {
  ATTR_INPUT_MESSAGES_RECORDED_BY,
  RECORDED_BY_CALLER,
  ConversationDeltaTracker,
  hasInputMessages,
  omitInputMessages,
  traceStream,
} from "./conversation.js";

const logger = getLogger("agent_engine_runner_shared.tracing");

const EXPORT_SUCCESS = 0;
const EXPORT_FAILED = 1;

function conversationAttributes(
  attributes: Record<string, unknown>,
  tracker: ConversationDeltaTracker,
  traceId: string,
  resource: Record<string, unknown>,
): Record<string, unknown> | null {
  if (attributes[ATTR_INPUT_MESSAGES_RECORDED_BY] === RECORDED_BY_CALLER) {
    return omitInputMessages(attributes);
  }
  if (!hasInputMessages(attributes)) {
    return null;
  }
  return tracker.trim(attributes, traceId, traceStream(resource));
}

const CONTENT_CAPTURE_ENV = "AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE";
const CONTENT_CAPTURE_FULL = "full";
const CONTENT_CAPTURE_METADATA_ONLY = "metadata-only";

const REDACTED_KEY_FRAGMENTS: readonly string[] = [
  "input.value",
  "output.value",
  "llm.input_messages",
  "llm.output_messages",
  "llm.prompts",
  "llm.prompt_template.variables",
  "llm.prompt_template.template",
  "llm.invocation_parameters",
  "message.content",
  "message.function_call_arguments_json",
  "tool_call.function.arguments",
  "llm.function_call",
  "tool.parameters",
  "llm.tools",
  "retrieval.documents",
  "document.content",
  "reranker.query",
  "embedding.embeddings",
  "embedding.text",
  "embedding.vector",
  "embedding.invocation_parameters",
  "memory.content",
  "memory.value",
  "user.id",
  "metadata",
  "exception.message",
  "exception.stacktrace",
  "http.url",
  "url.full",
  "net.peer",
  "server.address",
  "net.sock.peer",
  "db.connection_string",
  "secret",
  "password",
  "credential",
  "api_key",
  "access_token",
  "bearer",
];

/**
 * Read the content-capture policy from `AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE`.
 * Defaults to `"metadata-only"`; any value other than `"full"` is treated as
 * `"metadata-only"` so a typo'd env var fails closed, not open.
 */
export function getContentCaptureMode(): string {
  const mode = (
    process.env[CONTENT_CAPTURE_ENV] ?? CONTENT_CAPTURE_METADATA_ONLY
  )
    .trim()
    .toLowerCase();
  return mode === CONTENT_CAPTURE_FULL
    ? CONTENT_CAPTURE_FULL
    : CONTENT_CAPTURE_METADATA_ONLY;
}

function isRedactedAttributeKey(key: string): boolean {
  const lowered = key.toLowerCase();
  return REDACTED_KEY_FRAGMENTS.some((fragment) => lowered.includes(fragment));
}

function redactAttributes(attributes: Attributes | undefined): Attributes {
  if (!attributes) return {};
  const redacted: Attributes = {};
  for (const [key, value] of Object.entries(attributes)) {
    if (!isRedactedAttributeKey(key)) redacted[key] = value;
  }
  return redacted;
}

function redactSpan(span: ReadableSpan): ReadableSpan {
  const redacted = Object.create(span) as ReadableSpan;
  Object.defineProperty(redacted, "attributes", {
    value: redactAttributes(span.attributes),
    enumerable: true,
  });
  Object.defineProperty(redacted, "events", {
    value: (span.events ?? []).map((e) => ({
      ...e,
      attributes: redactAttributes(e.attributes),
    })),
    enumerable: true,
  });
  Object.defineProperty(redacted, "status", {
    value: { code: span.status.code },
    enumerable: true,
  });
  Object.defineProperty(redacted, "links", {
    value: (span.links ?? []).map((l) => ({
      ...l,
      attributes: redactAttributes(l.attributes),
    })),
    enumerable: true,
  });
  return redacted;
}

function withAttributes(
  span: ReadableSpan,
  attributes: Record<string, unknown>,
): ReadableSpan {
  const copy = Object.create(span) as ReadableSpan;
  Object.defineProperty(copy, "attributes", {
    value: attributes,
    enumerable: true,
  });
  return copy;
}

/**
 * Wraps an OTLP `SpanExporter`, enforcing `AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE`.
 * Redaction failures fail closed: a batch that can't be safely redacted is
 * dropped rather than forwarded unredacted, matching the metadata-only
 * default's privacy-first intent. Transport failures on the wrapped exporter
 * never throw — they're surfaced only through `resultCallback`.
 */
export class ContentPolicyOTLPSpanExporter implements SpanExporter {
  private readonly contentCaptureMode: string;
  /**
   * Exports this wrapper started but has not finished. The wrapped exporter
   * reports completion through its result callback; the `BatchSpanProcessor`
   * around it ships on a timer and does not track the export it already
   * started, so its `forceFlush()` can resolve while one of these is still in
   * flight; see {@link drain}.
   */
  private readonly inFlight = new Set<Promise<void>>();

  constructor(
    private readonly inner: SpanExporter,
    contentCaptureMode?: string,
  ) {
    if (contentCaptureMode !== undefined) {
      const normalized = contentCaptureMode.trim().toLowerCase();
      this.contentCaptureMode =
        normalized === CONTENT_CAPTURE_FULL
          ? CONTENT_CAPTURE_FULL
          : CONTENT_CAPTURE_METADATA_ONLY;
    } else {
      this.contentCaptureMode = getContentCaptureMode();
    }
  }

  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ): void {
    if (spans.length === 0) {
      resultCallback({ code: EXPORT_SUCCESS });
      return;
    }

    if (this.contentCaptureMode === CONTENT_CAPTURE_FULL) {
      this.exportInner(spans, resultCallback);
      return;
    }

    let redacted: ReadableSpan[];
    try {
      redacted = spans.map((span) => redactSpan(span));
    } catch (err) {
      logger.error(
        { err: (err as Error).message },
        "Content-capture redaction failed; dropping batch rather than risk exporting unredacted content",
      );
      resultCallback({ code: EXPORT_FAILED, error: err as Error });
      return;
    }
    this.exportInner(redacted, resultCallback);
  }

  private exportInner(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ): void {
    let settle!: () => void;
    const done = new Promise<void>((resolve) => {
      settle = resolve;
    });
    this.inFlight.add(done);
    void done.finally(() => {
      this.inFlight.delete(done);
    });
    const tracked = (result: ExportResult): void => {
      settle();
      resultCallback(result);
    };
    try {
      this.inner.export(spans, tracked);
    } catch (err) {
      logger.error({ err: (err as Error).message }, "OTLP export failed");
      settle();
      resultCallback({ code: EXPORT_FAILED, error: err as Error });
    }
  }

  /**
   * Await every export this wrapper started but has not finished.
   *
   * `BatchSpanProcessor.forceFlush()` drains only the spans still buffered: an
   * export its timer already started is invisible to it, so the flush can
   * resolve while that export is still in flight. A caller about to make the
   * process disappear (a session-finish release) awaits this so it does not
   * release on a flush that only looked complete. Best-effort and never
   * rejects — a failed export is already reported through its result callback.
   */
  async drain(): Promise<void> {
    while (this.inFlight.size > 0) {
      await Promise.allSettled([...this.inFlight]);
    }
  }

  shutdown(): Promise<void> {
    return Promise.resolve(this.inner.shutdown()).catch((err: Error) => {
      logger.error({ err: err.message }, "OTLP exporter shutdown failed");
    });
  }

  forceFlush(): Promise<void> {
    if (this.inner.forceFlush === undefined) return Promise.resolve();
    return Promise.resolve(this.inner.forceFlush()).catch((err: Error) => {
      logger.error({ err: err.message }, "OTLP exporter force_flush failed");
    });
  }
}

const PLATFORM_TRACE_INGEST_PATH = "/internal/traces";
const OTLP_PROTOBUF_CONTENT_TYPE = "application/x-protobuf";

const DEFAULT_PLATFORM_MAX_REQUEST_BYTES = 8 << 20;

const DEFAULT_PLATFORM_EXPORT_TIMEOUT_SECONDS = 10.0;
const PLATFORM_PER_ATTEMPT_TIMEOUT_SECONDS = 3.0;

const PLATFORM_RETRYABLE_STATUS: ReadonlySet<number> = new Set([
  404, 429, 502, 503, 504,
]);

const PLATFORM_BACKOFF_BASE_SECONDS = 0.5;
const PLATFORM_BACKOFF_MAX_SECONDS = 5.0;

export interface PlatformSpanExporterOptions {
  fetchImpl?: typeof fetch;
  tlsOptions?: (url: string) => RequestInit | undefined;
  serializer?: ISerializer<ReadableSpan[], IExportTraceServiceResponse>;
  maxRequestBytes?: number;
  exportTimeoutSeconds?: number;
  perAttemptTimeoutSeconds?: number;
  sleep?: (seconds: number) => Promise<void>;
  clock?: () => number;
  random?: () => number;
}

function defaultSleep(seconds: number): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, seconds * 1000);
    if (typeof timer.unref === "function") timer.unref();
  });
}

function retryAfterSeconds(value: string | null): number | undefined {
  if (!value) return undefined;
  const trimmed = value.trim();
  const seconds = Number(trimmed);
  if (Number.isFinite(seconds)) return seconds >= 0 ? seconds : undefined;
  const when = Date.parse(trimmed);
  if (Number.isNaN(when)) return undefined;
  return Math.max(0, (when - Date.now()) / 1000);
}

/**
 * Export spans to the Orchestration Engine's platform trace endpoint.
 *
 * The guest POSTs OTLP-protobuf batches to `{oe_url}/internal/traces`; the OE
 * converts and stores them. Every attempt goes through the platform HTTP client
 * (`fetchPlatform`) combined with the mTLS request options, so the export call
 * creates no span of its own and presents the client certificate.
 *
 * Delivery is best-effort: a batch is retried on transport errors and retryable
 * statuses within a bounded deadline, split on 413, and dropped when the OE
 * permanently refuses it. A single span larger than the OE body limit is dropped
 * without failing the rest of its batch.
 */
export class PlatformSpanExporter implements SpanExporter {
  readonly url: string;
  private readonly fetchImpl: typeof fetch;
  private readonly tlsOptions: (url: string) => RequestInit | undefined;
  private readonly serializer: ISerializer<
    ReadableSpan[],
    IExportTraceServiceResponse
  >;
  private readonly maxRequestBytes: number;
  private readonly exportTimeoutSeconds: number;
  private readonly perAttemptTimeoutSeconds: number;
  private readonly sleep: (seconds: number) => Promise<void>;
  private readonly clock: () => number;
  private readonly random: () => number;

  private droppedSpansCount = 0;
  private lastErrorValue: string | null = null;
  private readonly pendingExports = new Set<Promise<unknown>>();
  private conversations = new ConversationDeltaTracker();

  constructor(baseUrl: string, options: PlatformSpanExporterOptions = {}) {
    const normalized = baseUrl.replace(/\/+$/, "");
    this.url = `${normalized}${PLATFORM_TRACE_INGEST_PATH}`;
    this.fetchImpl = options.fetchImpl ?? fetch;
    this.tlsOptions = options.tlsOptions ?? getFetchOptionsWithTLS;
    this.serializer = options.serializer ?? ProtobufTraceSerializer;
    this.maxRequestBytes =
      options.maxRequestBytes ?? DEFAULT_PLATFORM_MAX_REQUEST_BYTES;
    this.exportTimeoutSeconds =
      options.exportTimeoutSeconds ?? DEFAULT_PLATFORM_EXPORT_TIMEOUT_SECONDS;
    this.perAttemptTimeoutSeconds =
      options.perAttemptTimeoutSeconds ?? PLATFORM_PER_ATTEMPT_TIMEOUT_SECONDS;
    this.sleep = options.sleep ?? defaultSleep;
    this.clock = options.clock ?? (() => Date.now() / 1000);
    this.random = options.random ?? Math.random;
  }

  /** Total spans dropped for exceeding the OE body limit or OE rejection. */
  get droppedSpans(): number {
    return this.droppedSpansCount;
  }

  /** Most recent export error, or null when the last export fully succeeded. */
  get lastError(): string | null {
    return this.lastErrorValue;
  }

  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ): void {
    if (spans.length === 0) {
      resultCallback({ code: EXPORT_SUCCESS });
      return;
    }
    const pending = this.exportSpans(spans);
    this.trackPendingExport(pending);
    void pending.then(
      (ok) => resultCallback({ code: ok ? EXPORT_SUCCESS : EXPORT_FAILED }),
      (err: Error) => {
        logger.error(
          { err: err.message },
          "Platform trace export failed unexpectedly",
        );
        resultCallback({ code: EXPORT_FAILED, error: err });
      },
    );
  }

  private trackPendingExport(promise: Promise<unknown>): void {
    this.pendingExports.add(promise);
    void promise
      .finally(() => this.pendingExports.delete(promise))
      .catch(() => {});
  }

  /**
   * Wait for every export already dispatched to this exporter to settle.
   *
   * `BatchSpanProcessor.forceFlush()` drains spans still in its buffer, but a
   * batch the scheduled timer already handed to `export()` has left that
   * buffer, and the processor's flush does not call this exporter method — so a
   * caller that needs a delivery barrier must invoke it directly alongside the
   * provider flush. Loops so an export dispatched by a concurrent flush is
   * included too; the caller bounds the wait with its own timeout.
   */
  async forceFlush(): Promise<void> {
    while (this.pendingExports.size > 0) {
      await Promise.allSettled([...this.pendingExports]);
    }
  }

  shutdown(): Promise<void> {
    return Promise.resolve();
  }

  private async exportSpans(spans: ReadableSpan[]): Promise<boolean> {
    const deadline = this.clock() + this.exportTimeoutSeconds;
    const controller = new AbortController();
    const timer = setTimeout(
      () => controller.abort(),
      this.exportTimeoutSeconds * 1000,
    );
    if (typeof timer.unref === "function") timer.unref();
    try {
      const droppedBefore = this.droppedSpansCount;
      let ok = true;
      for (const chunk of this.chunkBySize(spans)) {
        const staged = this.conversations.copy();
        const trimmed = chunk.map(
          (span) => this.trimSpan(span, staged) ?? span,
        );
        const droppedBeforeChunk = this.droppedSpansCount;
        if (await this.exportChunk(trimmed, deadline, controller.signal)) {
          if (this.droppedSpansCount === droppedBeforeChunk) {
            this.conversations = staged;
          }
        } else {
          ok = false;
        }
      }
      if (ok) {
        if (this.droppedSpansCount === droppedBefore) {
          this.lastErrorValue = null;
        }
      }
      return ok;
    } finally {
      clearTimeout(timer);
    }
  }

  private trimSpan(
    span: ReadableSpan,
    tracker: ConversationDeltaTracker,
  ): ReadableSpan | null {
    const attributes = span.attributes as unknown as
      | Record<string, unknown>
      | undefined;
    if (!attributes) return null;
    const trimmed = conversationAttributes(
      attributes,
      tracker,
      span.spanContext().traceId,
      (span.resource?.attributes ?? {}) as unknown as Record<string, unknown>,
    );
    return trimmed === null ? null : withAttributes(span, trimmed);
  }

  private encode(spans: ReadableSpan[]): Uint8Array {
    return this.serializer.serializeRequest(spans) ?? new Uint8Array(0);
  }

  private encodedSize(spans: ReadableSpan[]): number {
    return this.encode(spans).byteLength;
  }

  private chunkBySize(spans: ReadableSpan[]): ReadableSpan[][] {
    if (spans.length <= 1) return [spans];
    if (this.encodedSize(spans) <= this.maxRequestBytes) return [spans];
    const mid = Math.floor(spans.length / 2);
    return [
      ...this.chunkBySize(spans.slice(0, mid)),
      ...this.chunkBySize(spans.slice(mid)),
    ];
  }

  private async exportChunk(
    chunk: ReadableSpan[],
    deadline: number,
    overallSignal: AbortSignal,
  ): Promise<boolean> {
    if (chunk.length === 1 && this.encodedSize(chunk) > this.maxRequestBytes) {
      this.dropOversized();
      return false;
    }
    let attempt = 0;
    for (;;) {
      attempt += 1;
      const remaining = deadline - this.clock();
      if (remaining <= 0) {
        this.recordError("export deadline exceeded");
        return false;
      }
      let response: Response;
      try {
        response = await this.post(
          chunk,
          Math.min(this.perAttemptTimeoutSeconds, remaining),
          overallSignal,
        );
      } catch (err) {
        if (overallSignal.aborted || this.clock() > deadline) {
          this.recordError("export deadline exceeded");
          return false;
        }
        if (!(await this.backoff(attempt, deadline, undefined))) {
          this.recordError(`transport error: ${(err as Error).message}`);
          return false;
        }
        continue;
      }

      if (this.clock() > deadline) {
        this.recordError("export deadline exceeded");
        return false;
      }

      const status = response.status;
      if (status >= 200 && status < 300) {
        await this.handlePartialSuccess(response);
        return true;
      }
      if (status === 413) {
        return this.splitAndRetry(chunk, deadline, overallSignal);
      }
      if (PLATFORM_RETRYABLE_STATUS.has(status)) {
        const retryAfter = retryAfterSeconds(
          response.headers.get("Retry-After"),
        );
        if (!(await this.backoff(attempt, deadline, retryAfter))) {
          this.recordError(`status ${status} after ${attempt} attempt(s)`);
          return false;
        }
        continue;
      }

      this.recordError(`terminal status ${status}`);
      logger.warn(
        { status, spans: chunk.length },
        "Trace export: OE returned a terminal status; dropping batch",
      );
      return false;
    }
  }

  private async splitAndRetry(
    chunk: ReadableSpan[],
    deadline: number,
    overallSignal: AbortSignal,
  ): Promise<boolean> {
    if (chunk.length <= 1) {
      this.dropOversized();
      return false;
    }
    const mid = Math.floor(chunk.length / 2);
    let ok = true;
    for (const half of [chunk.slice(0, mid), chunk.slice(mid)]) {
      if (!(await this.exportChunk(half, deadline, overallSignal))) ok = false;
    }
    return ok;
  }

  private post(
    chunk: ReadableSpan[],
    timeoutSeconds: number,
    overallSignal: AbortSignal,
  ): Promise<Response> {
    const signals = [overallSignal];
    if (timeoutSeconds > 0) {
      signals.push(
        AbortSignal.timeout(Math.max(1, Math.round(timeoutSeconds * 1000))),
      );
    }
    const signal =
      signals.length > 1 ? AbortSignal.any(signals) : overallSignal;
    const tlsOptions = this.tlsOptions(this.url) ?? {};
    return fetchPlatform(
      this.url,
      {
        method: "POST",
        headers: { "Content-Type": OTLP_PROTOBUF_CONTENT_TYPE },
        body: this.encode(chunk) as unknown as BodyInit,
        signal,
        ...tlsOptions,
      },
      this.fetchImpl,
    );
  }

  private async handlePartialSuccess(response: Response): Promise<void> {
    let bytes: Uint8Array;
    try {
      bytes = new Uint8Array(await response.arrayBuffer());
    } catch {
      return;
    }
    if (bytes.byteLength === 0) return;
    try {
      const parsed = this.serializer.deserializeResponse(bytes);
      const rejected = parsed.partialSuccess?.rejectedSpans ?? 0;
      if (rejected > 0) {
        this.droppedSpansCount += rejected;
        const message = parsed.partialSuccess?.errorMessage;
        this.recordError(
          message
            ? `OE rejected ${rejected} span(s): ${message}`
            : `OE rejected ${rejected} span(s)`,
        );
        logger.warn(
          { rejected },
          "Trace export: OE rejected span(s) as malformed or unstorable",
        );
      }
    } catch {
      return;
    }
  }

  private async backoff(
    attempt: number,
    deadline: number,
    retryAfter: number | undefined,
  ): Promise<boolean> {
    let delay: number;
    if (retryAfter !== undefined) {
      delay = retryAfter;
    } else {
      delay = Math.min(
        PLATFORM_BACKOFF_MAX_SECONDS,
        PLATFORM_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1),
      );
      delay *= 0.5 + this.random() * 0.5;
    }
    if (delay >= deadline - this.clock()) return false;
    await this.sleep(delay);
    return true;
  }

  private dropOversized(): void {
    this.droppedSpansCount += 1;
    logger.warn("Trace export: dropping a span larger than the OE body limit");
    this.recordError("dropped a span larger than the OE body limit");
  }

  private recordError(message: string): void {
    this.lastErrorValue = message;
    logger.debug({ message }, "Platform trace export failed");
  }
}
