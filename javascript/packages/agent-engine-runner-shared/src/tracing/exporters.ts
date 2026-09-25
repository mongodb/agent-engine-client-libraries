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
import type { ReadableSpan, SpanExporter } from "@opentelemetry/sdk-trace-base";

import type { Collection } from "mongodb";

import { getLogger } from "../logger.js";

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

export class MongoDBSpanExporter implements SpanExporter {
  constructor(private readonly collection: Collection) {}

  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ): void {
    if (spans.length === 0) {
      resultCallback({ code: EXPORT_SUCCESS });
      return;
    }
    const docs = spans.map((s) => spanToDoc(s));
    this.collection
      .insertMany(docs)
      .then(() => resultCallback({ code: EXPORT_SUCCESS }))
      .catch((err: Error) => {
        logger.error({ err: err.message }, "Failed to write traces to MongoDB");
        resultCallback({ code: EXPORT_FAILED, error: err });
      });
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

/**
 * Wraps an OTLP `SpanExporter`, enforcing `AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE`.
 * Redaction failures fail closed: a batch that can't be safely redacted is
 * dropped rather than forwarded unredacted, matching the metadata-only
 * default's privacy-first intent. Transport failures on the wrapped exporter
 * never throw — they're surfaced only through `resultCallback`. Ported from
 * `ContentPolicyOTLPSpanExporter` in `agent_engine_runner_shared/tracing/exporters.py`.
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
