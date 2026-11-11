/**
 * Span exporters for the Runner SDK tracing pipeline.
 *
 * Ported from `agent_engine_runner_shared/tracing/exporters.py`. Two exporters:
 *   - `JSONLSpanExporter` — optional explicit local debugging exporter.
 *   - `MongoDBSpanExporter` — inserts spans as documents into a Mongo collection.
 *
 * Both implement the `@opentelemetry/sdk-trace-base` `SpanExporter` interface.
 */

import { appendFileSync, mkdirSync, renameSync, statSync } from "node:fs";
import { dirname, join } from "node:path";

import type { Attributes } from "@opentelemetry/api";
import type { ExportResult } from "@opentelemetry/core";
import {
  ProtobufTraceSerializer,
  type IExportTraceServiceResponse,
  type ISerializer,
} from "@opentelemetry/otlp-transformer";
import type { ReadableSpan, SpanExporter } from "@opentelemetry/sdk-trace-base";

import type { Collection } from "mongodb";

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

const EXPORT_SUCCESS = 0; // ExportResultCode.SUCCESS
const EXPORT_FAILED = 1; // ExportResultCode.FAILED

const MAX_TRACE_FILE_BYTES = 50 * 1024 * 1024; // 50 MB

function rotateIfNeeded(path: string): void {
  try {
    const stat = statSync(path, { throwIfNoEntry: false });
    if (stat && stat.size >= MAX_TRACE_FILE_BYTES) {
      renameSync(path, path + ".1");
    }
  } catch {
    // best-effort — a failed rotation is better than crashing the exporter
  }
}

/** Resolve the on-disk trace file path from env vars (mirrors Python). */
export function getTracePath(): string {
  let path: string;
  if (process.env["AGENTIC_TRACES_FILE"]) {
    path = process.env["AGENTIC_TRACES_FILE"];
  } else if (process.env["AGENTIC_OBSERVABILITY_DIR"]) {
    path = join(process.env["AGENTIC_OBSERVABILITY_DIR"], "traces.jsonl");
  } else {
    path = join("observability", "traces.jsonl");
  }
  mkdirSync(dirname(path), { recursive: true });
  return path;
}

function spanToDoc(span: ReadableSpan): Record<string, unknown> {
  const ctx = span.spanContext();
  const attributes: Record<string, unknown> = { ...span.attributes };
  const sessionId = attributes["session.id"] ?? attributes["session_id"];
  const executionId = attributes["execution.id"] ?? attributes["execution_id"];

  const doc: Record<string, unknown> = {
    trace_id: ctx.traceId,
    span_id: ctx.spanId,
    name: span.name,
    kind: String(span.kind),
    start_time_ns: hrtimeToNs(span.startTime),
    end_time_ns: hrtimeToNs(span.endTime),
    duration_ns: hrtimeDurationNs(span.startTime, span.endTime),
    status: {
      code: String(span.status.code),
      description: span.status.message ?? null,
    },
    attributes,
    resource: { ...(span.resource.attributes ?? {}) },
  };

  if (sessionId !== undefined) doc["session_id"] = sessionId;
  if (executionId !== undefined) doc["execution_id"] = executionId;

  if (span.parentSpanContext?.spanId) {
    doc["parent_span_id"] = span.parentSpanContext.spanId;
  }

  if (span.events && span.events.length > 0) {
    doc["events"] = span.events.map((e) => ({
      name: e.name,
      timestamp_ns: hrtimeToNs(e.time),
      attributes: { ...(e.attributes ?? {}) },
    }));
  }

  return doc;
}

function hrtimeToNs(t: [number, number]): number {
  return t[0] * 1e9 + t[1];
}

function hrtimeDurationNs(
  start: [number, number],
  end: [number, number],
): number {
  return hrtimeToNs(end) - hrtimeToNs(start);
}

export class JSONLSpanExporter implements SpanExporter {
  private readonly path: string | null;

  constructor(path?: string) {
    this.path = path ?? null;
  }

  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ): void {
    if (spans.length === 0) {
      resultCallback({ code: EXPORT_SUCCESS });
      return;
    }
    const target = this.path ?? getTracePath();
    try {
      rotateIfNeeded(target);
      const lines =
        spans.map((s) => JSON.stringify(spanToDoc(s))).join("\n") + "\n";
      appendFileSync(target, lines, { encoding: "utf-8" });
      resultCallback({ code: EXPORT_SUCCESS });
    } catch (err) {
      logger.error(
        { path: target, err: (err as Error).message },
        "Failed to write traces",
      );
      resultCallback({ code: EXPORT_FAILED, error: err as Error });
    }
  }

  shutdown(): Promise<void> {
    return Promise.resolve();
  }

  forceFlush(): Promise<void> {
    return Promise.resolve();
  }
}

/**
 * Return the attributes to store for one span, or null to keep them as-is.
 *
 * Shared by the MongoDB and platform exporters so a span's conversation is
 * deduplicated the same way whichever store it reaches.
 */
function conversationAttributes(
  attributes: Record<string, unknown>,
  tracker: ConversationDeltaTracker,
  traceId: string,
  resource: Record<string, unknown>,
): Record<string, unknown> | null {
  if (attributes[ATTR_INPUT_MESSAGES_RECORDED_BY] === RECORDED_BY_CALLER) {
    // The caller's span holds this prompt. Dropping it here and leaving the
    // tracker alone keeps a later tool-pod span — one that records its own
    // conversation — starting from a full list rather than a bogus offset.
    return omitInputMessages(attributes);
  }
  if (!hasInputMessages(attributes)) {
    return null;
  }
  return tracker.trim(attributes, traceId, traceStream(resource));
}

export class MongoDBSpanExporter implements SpanExporter {
  private conversations = new ConversationDeltaTracker();

  constructor(private readonly collection: Collection) {}

  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ): void {
    if (spans.length === 0) {
      resultCallback({ code: EXPORT_SUCCESS });
      return;
    }
    // Trim on a copy: a failed insert must not advance past messages the store
    // never received, or later calls would omit them.
    const staged = this.conversations.copy();
    const docs = spans.map((s) => this.trimConversation(spanToDoc(s), staged));
    this.collection
      .insertMany(docs)
      .then(() => {
        this.conversations = staged;
        resultCallback({ code: EXPORT_SUCCESS });
      })
      .catch((err: Error) => {
        logger.error({ err: err.message }, "Failed to write traces to MongoDB");
        resultCallback({ code: EXPORT_FAILED, error: err });
      });
  }

  /** Keep only the input messages this call added (see `conversation.ts`). */
  private trimConversation(
    doc: Record<string, unknown>,
    tracker: ConversationDeltaTracker,
  ): Record<string, unknown> {
    const attributes = doc["attributes"] as Record<string, unknown> | undefined;
    if (!attributes) {
      return doc;
    }
    const trimmed = conversationAttributes(
      attributes,
      tracker,
      String(doc["trace_id"]),
      (doc["resource"] ?? {}) as Record<string, unknown>,
    );
    if (trimmed !== null) {
      doc["attributes"] = trimmed;
    }
    return doc;
  }

  shutdown(): Promise<void> {
    return Promise.resolve();
  }

  forceFlush(): Promise<void> {
    return Promise.resolve();
  }
}

// =============================================================================
// Content-capture policy
//
// Mongo stays unredacted (customer's own BYOC DB, canonical for the
// playground). This policy applies only to the OTLP path, which leaves the
// process bound for a third-party tool. Ported from
// `agent_engine_runner_shared/tracing/exporters.py` — keep `REDACTED_KEY_FRAGMENTS` in
// sync with that module's `_REDACTED_KEY_FRAGMENTS`; a drifted copy either
// leaks content this list was meant to block, or over-redacts metadata.
// =============================================================================

const CONTENT_CAPTURE_ENV = "AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE";
const CONTENT_CAPTURE_FULL = "full";
const CONTENT_CAPTURE_METADATA_ONLY = "metadata-only";

// Substrings matched case-insensitively against attribute keys. OpenInference
// flattens nested/indexed fields (e.g. "llm.input_messages.0.message.content"),
// so this is fragment matching, not an exact-key set. See exporters.py for the
// full rationale (allowlist deliberately deferred, etc.) — this list must stay
// identical to Python's.
const REDACTED_KEY_FRAGMENTS: readonly string[] = [
  // Prompts / completions
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
  // Tool args/results
  "tool.parameters",
  "llm.tools",
  // Retrieved docs
  "retrieval.documents",
  "document.content",
  "reranker.query",
  // Embeddings
  "embedding.embeddings",
  "embedding.text",
  "embedding.vector",
  "embedding.invocation_parameters",
  // Memory contents
  "memory.content",
  "memory.value",
  // Real end-user identifier
  "user.id",
  // Free-form metadata / exceptions
  "metadata",
  "exception.message",
  "exception.stacktrace",
  // Network / infra
  "http.url",
  "url.full",
  "net.peer",
  "server.address",
  "net.sock.peer",
  "db.connection_string",
  // Secrets
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

/**
 * Return a copy of `span` with content-bearing attributes (and event
 * attributes) stripped.
 *
 * A real SDK `Span` exposes `spanContext()` (and other members) as methods
 * on its prototype, not as own enumerable properties — a plain `{ ...span }`
 * spread silently drops them, so anything downstream that calls
 * `span.spanContext()` (e.g. the OTLP transformer) throws
 * "span.spanContext is not a function". `Object.create(span)` instead makes
 * `span` itself the new object's prototype: every unlisted member (methods
 * and data alike) falls through the prototype chain to the original span
 * unchanged, and only `attributes`/`events` are shadowed as own properties.
 */
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
  return redacted;
}

// Keep the SDK span as the prototype: its methods live there, and a spread
// copy would lose them.
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
    try {
      this.inner.export(spans, resultCallback);
    } catch (err) {
      logger.error({ err: (err as Error).message }, "OTLP export failed");
      resultCallback({ code: EXPORT_FAILED, error: err as Error });
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

// Platform trace exporter (Orchestration Engine /internal/traces).
// The guest sends OTLP-protobuf batches to the OE, which converts and stores
// them in the project traces collection. This replaces the guest's direct
// MongoDB write: it is the platform trace path, independent of customer OTLP
// export, and carries unredacted spans so the playground can show prompts and
// outputs as it did from Mongo.

const PLATFORM_TRACE_INGEST_PATH = "/internal/traces";
const OTLP_PROTOBUF_CONTENT_TYPE = "application/x-protobuf";

// The OE rejects a raw request body over its own configured limit with 413.
// Chunk at that limit before sending; an oversized single span cannot be split.
const DEFAULT_PLATFORM_MAX_REQUEST_BYTES = 8 << 20;

// Overall deadline for one export() call. The BatchSpanProcessor worker calls
// export() and waits for the callback, so a bounded deadline keeps an
// unreachable or slow OE from stalling the export queue behind it.
const DEFAULT_PLATFORM_EXPORT_TIMEOUT_SECONDS = 10.0;
const PLATFORM_PER_ATTEMPT_TIMEOUT_SECONDS = 3.0;

// Retryable: the OE is rolling (404), shedding (503/429), or a gateway hiccup
// (502/504). Any other status is terminal for the batch.
const PLATFORM_RETRYABLE_STATUS: ReadonlySet<number> = new Set([
  404, 429, 502, 503, 504,
]);

const PLATFORM_BACKOFF_BASE_SECONDS = 0.5;
const PLATFORM_BACKOFF_MAX_SECONDS = 5.0;

export interface PlatformSpanExporterOptions {
  /** Overrides the global fetch (tests). */
  fetchImpl?: typeof fetch;
  /**
   * Request options carrying mTLS for `url`, applied on every attempt.
   * `fetchPlatform` alone does not configure mTLS.
   */
  tlsOptions?: (url: string) => RequestInit | undefined;
  /** OTLP request serializer. Defaults to the protobuf serializer. */
  serializer?: ISerializer<ReadableSpan[], IExportTraceServiceResponse>;
  /** Bodies larger than this are chunked; a single larger span is dropped. */
  maxRequestBytes?: number;
  /** Overall deadline for one export() call, in seconds. */
  exportTimeoutSeconds?: number;
  /** Per-attempt request timeout, capped by the remaining export budget. */
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

/**
 * Parse a Retry-After header, accepting delay-seconds or an HTTP-date.
 * Returns undefined for a missing or unparsable value.
 */
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
      .catch(() => {
        // The result callback reports failures; prevent an unhandled rejection
        // from the cleanup chain.
      });
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
        // Stage per chunk and commit only what the OE stored, so a batch it
        // refused does not advance past messages the store never received.
        const staged = this.conversations.copy();
        const trimmed = chunk.map(
          (span) => this.trimSpan(span, staged) ?? span,
        );
        const droppedBeforeChunk = this.droppedSpansCount;
        if (await this.exportChunk(trimmed, deadline, controller.signal)) {
          // A partial success drops spans without naming them, so the next
          // call restates those messages instead of leaving a gap.
          if (this.droppedSpansCount === droppedBeforeChunk) {
            this.conversations = staged;
          }
        } else {
          ok = false;
        }
      }
      if (ok) {
        // A partial success drops spans without failing the batch, so only a
        // clean export clears the last error; /health stays degraded until a
        // later export delivers everything.
        if (this.droppedSpansCount === droppedBefore) {
          this.lastErrorValue = null;
        }
      }
      return ok;
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Return `span` with only this call's new input messages, or null when the
   * span carries no conversation to trim (see `conversation.ts`).
   *
   * The span-level twin of `MongoDBSpanExporter.trimConversation`, for the OTLP
   * path that encodes the span itself rather than building a document.
   */
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
        // Cap the attempt by the remaining export budget so the whole export
        // honours its wall-clock deadline rather than just each attempt.
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
      // AbortSignal.timeout requires an integer millisecond delay.
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
        // A Uint8Array is a valid BodyInit at runtime; the cast narrows the
        // ArrayBufferLike backing to the ArrayBuffer that BodyInit accepts.
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
        // The OE accepted the batch but discarded some spans. Delivery is
        // best-effort, so the batch is not retried (a retry would duplicate
        // the spans that were stored), but the loss is recorded so
        // tracingStatus() and /health surface it.
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
      // An unparsable response body is not actionable.
    }
  }

  /**
   * Sleep before the next attempt, or return false when none is possible.
   * Honours Retry-After when the OE sends one; otherwise uses exponential
   * backoff with jitter. Returns false when the delay would run past the export
   * deadline, so a retry is never attempted without time for it.
   */
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
    // A dropped span must keep /health degraded until a later clean export.
    this.recordError("dropped a span larger than the OE body limit");
  }

  private recordError(message: string): void {
    this.lastErrorValue = message;
    logger.debug({ message }, "Platform trace export failed");
  }
}
