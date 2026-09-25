/**
 * Unit tests for the span exporters.
 *
 * Mirrors Python's tests/unit/test_tracing_exporter.py.
 *
 * Source change to enable parity: `runInstrumentor` was exported from
 * `tracing/setup.ts`. Python imports `_run_instrumentor` directly
 * (leading-underscore = convention-private but still importable). Same
 * pattern as `resolveListenHost` — TS strict export semantics required
 * adding `export`.
 *
 * TS-vs-Python adaptations:
 *   - `SpanExporter.export()` is callback-based in TS, return-based in
 *     Python. Wrapped in `exportAndWait()` to return a promise.
 *   - Python uses `MagicMock(spec=ReadableSpan)` with manual attribute
 *     setup. TS builds a minimal structurally-typed mock and casts to
 *     `ReadableSpan` (the full OTEL interface has many fields the
 *     exporters don't touch).
 *   - Python's `caplog` fixture for log capture has no Vitest equivalent.
 *     `vi.mock` on the logger module is used instead — same pattern as
 *     `metrics.test.ts`.
 */

import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { readFileSync, existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// vi.mock the logger BEFORE source imports so the runInstrumentor warnings
// can be observed via the mock. Hoisted by Vitest.
const mocks = vi.hoisted(() => ({
  logger: {
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
    trace: vi.fn(),
    fatal: vi.fn(),
    log: vi.fn(),
  },
}));

vi.mock("../../src/logger.js", () => ({
  getLogger: () => mocks.logger,
  setupLogging: vi.fn(),
}));

import type { ReadableSpan, SpanExporter } from "@opentelemetry/sdk-trace-base";
import type { ExportResult } from "@opentelemetry/core";
import type { Collection } from "mongodb";

import {
  ContentPolicyOTLPSpanExporter,
  JSONLSpanExporter,
  MongoDBSpanExporter,
  getContentCaptureMode,
  runInstrumentor,
  registerInstrumentor,
  resetHooks,
} from "../../src/index.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeMockSpan(overrides: Partial<ReadableSpan> = {}): ReadableSpan {
  const base = {
    spanContext: () => ({
      traceId: "12345678123456781234567812345678",
      spanId: "1234567812345678",
      traceFlags: 1,
      isRemote: false,
    }),
    name: "test_span",
    kind: 0, // INTERNAL
    startTime: [1, 0] as [number, number],
    endTime: [2, 0] as [number, number],
    status: { code: 1 }, // OK
    attributes: { key: "value" },
    resource: { attributes: { "service.name": "test_service" } },
    events: [],
  };
  return { ...base, ...overrides } as unknown as ReadableSpan;
}

/** Convert the callback-based SpanExporter.export() into a promise. */
function exportAndWait(
  exporter: SpanExporter,
  spans: ReadableSpan[],
): Promise<ExportResult> {
  return new Promise((resolve) => {
    exporter.export(spans, resolve);
  });
}

let tmpDir: string;

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), "tracing-"));
  mocks.logger.warn.mockClear();
  mocks.logger.error.mockClear();
  mocks.logger.info.mockClear();
});

afterEach(() => {
  rmSync(tmpDir, { recursive: true, force: true });
  resetHooks();
});

// ---------------------------------------------------------------------------
// JSONL Exporter
// ---------------------------------------------------------------------------

describe("JSONLSpanExporter", () => {
  test("writes a span to the file", async () => {
    // test_jsonl_exporter_writes_span
    const traceFile = join(tmpDir, "traces.jsonl");
    const exporter = new JSONLSpanExporter(traceFile);

    const result = await exportAndWait(exporter, [makeMockSpan()]);

    expect(result.code).toBe(0); // EXPORT_SUCCESS
    expect(existsSync(traceFile)).toBe(true);

    const doc = JSON.parse(
      readFileSync(traceFile, "utf-8").trimEnd().split("\n")[0] ?? "",
    );
    expect(doc["trace_id"]).toBe("12345678123456781234567812345678");
    expect(doc["span_id"]).toBe("1234567812345678");
    expect(doc["name"]).toBe("test_span");
    expect(doc["attributes"]["key"]).toBe("value");
  });

  test("appends to an existing file across multiple exports", async () => {
    // test_jsonl_exporter_appends
    const traceFile = join(tmpDir, "traces.jsonl");
    const exporter = new JSONLSpanExporter(traceFile);

    await exportAndWait(exporter, [makeMockSpan({ name: "test_span" })]);
    await exportAndWait(exporter, [makeMockSpan({ name: "span_2" })]);

    const lines = readFileSync(traceFile, "utf-8").trimEnd().split("\n");
    expect(lines).toHaveLength(2);
    expect(JSON.parse(lines[0] ?? "")["name"]).toBe("test_span");
    expect(JSON.parse(lines[1] ?? "")["name"]).toBe("span_2");
  });

  test("handles an empty span list without writing the file", async () => {
    // test_jsonl_exporter_handles_empty
    const traceFile = join(tmpDir, "traces.jsonl");
    const exporter = new JSONLSpanExporter(traceFile);

    const result = await exportAndWait(exporter, []);

    expect(result.code).toBe(0); // EXPORT_SUCCESS
    expect(existsSync(traceFile)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// MongoDB Exporter
// ---------------------------------------------------------------------------

describe("MongoDBSpanExporter", () => {
  test("inserts a span via insertMany", async () => {
    // test_mongodb_exporter_inserts_span
    const insertMany = vi.fn().mockResolvedValue({ insertedCount: 1 });
    const collection = { insertMany } as unknown as Collection;
    const exporter = new MongoDBSpanExporter(collection);

    const result = await exportAndWait(exporter, [makeMockSpan()]);

    expect(result.code).toBe(0); // EXPORT_SUCCESS
    expect(insertMany).toHaveBeenCalledOnce();

    const docs = insertMany.mock.calls[0]?.[0] as Record<string, unknown>[];
    expect(docs).toHaveLength(1);
    expect(docs[0]?.["trace_id"]).toBe("12345678123456781234567812345678");
    expect(docs[0]?.["name"]).toBe("test_span");
  });

  test("handles an empty span list without calling insertMany", async () => {
    // test_mongodb_exporter_handles_empty
    const insertMany = vi.fn();
    const collection = { insertMany } as unknown as Collection;
    const exporter = new MongoDBSpanExporter(collection);

    const result = await exportAndWait(exporter, []);

    expect(result.code).toBe(0); // EXPORT_SUCCESS
    expect(insertMany).not.toHaveBeenCalled();
  });

  test("returns FAILURE when insertMany rejects", async () => {
    // test_mongodb_exporter_handles_error
    const insertMany = vi
      .fn()
      .mockRejectedValue(new Error("Connection failed"));
    const collection = { insertMany } as unknown as Collection;
    const exporter = new MongoDBSpanExporter(collection);

    const result = await exportAndWait(exporter, [makeMockSpan()]);

    expect(result.code).toBe(1); // EXPORT_FAILED
  });
});

// ---------------------------------------------------------------------------
// ContentPolicyOTLPSpanExporter
// ---------------------------------------------------------------------------

describe("getContentCaptureMode", () => {
  const ENV_KEY = "AGENTIC_PLATFORM_OTEL_CONTENT_CAPTURE";
  afterEach(() => {
    delete process.env[ENV_KEY];
  });

  test("defaults to metadata-only when unset", () => {
    delete process.env[ENV_KEY];
    expect(getContentCaptureMode()).toBe("metadata-only");
  });

  test("returns full only on an exact 'full' value", () => {
    process.env[ENV_KEY] = "full";
    expect(getContentCaptureMode()).toBe("full");
  });

  test("fails closed to metadata-only on a typo'd value", () => {
    process.env[ENV_KEY] = "FULLY";
    expect(getContentCaptureMode()).toBe("metadata-only");
  });
});

describe("ContentPolicyOTLPSpanExporter", () => {
  function makeInner(): SpanExporter & {
    export: ReturnType<typeof vi.fn>;
    shutdown: ReturnType<typeof vi.fn>;
    forceFlush: ReturnType<typeof vi.fn>;
  } {
    return {
      export: vi.fn((_spans: ReadableSpan[], cb: (r: ExportResult) => void) =>
        cb({ code: 0 }),
      ),
      shutdown: vi.fn().mockResolvedValue(undefined),
      forceFlush: vi.fn().mockResolvedValue(undefined),
    };
  }

  test("metadata-only mode strips every redacted-fragment attribute", async () => {
    const inner = makeInner();
    const exporter = new ContentPolicyOTLPSpanExporter(inner, "metadata-only");
    const span = makeMockSpan({
      attributes: {
        "llm.input_messages.0.message.content": "secret prompt",
        "tool_call.function.arguments": "{}",
        "session.id": "session-1",
        "model.name": "gpt-x",
      },
      events: [
        {
          name: "exception",
          time: [1, 0],
          attributes: { "exception.message": "leaked bad-arg content" },
        } as unknown as ReadableSpan["events"][number],
      ],
    });

    await exportAndWait(exporter, [span]);

    const forwarded = inner.export.mock.calls[0]?.[0] as ReadableSpan[];
    expect(forwarded[0]?.attributes).toEqual({
      "session.id": "session-1",
      "model.name": "gpt-x",
    });
    expect(forwarded[0]?.events[0]?.attributes).toEqual({});
  });

  test("full mode passes attributes through unredacted", async () => {
    const inner = makeInner();
    const exporter = new ContentPolicyOTLPSpanExporter(inner, "full");
    const span = makeMockSpan({
      attributes: { "llm.input_messages.0.message.content": "secret prompt" },
    });

    await exportAndWait(exporter, [span]);

    const forwarded = inner.export.mock.calls[0]?.[0] as ReadableSpan[];
    expect(forwarded[0]?.attributes).toEqual({
      "llm.input_messages.0.message.content": "secret prompt",
    });
  });

  test("an unrecognized mode string falls back to metadata-only", async () => {
    const inner = makeInner();
    const exporter = new ContentPolicyOTLPSpanExporter(inner, "unknown-mode");
    const span = makeMockSpan({
      attributes: { "llm.input_messages.0.message.content": "secret prompt" },
    });

    await exportAndWait(exporter, [span]);

    const forwarded = inner.export.mock.calls[0]?.[0] as ReadableSpan[];
    expect(forwarded[0]?.attributes).toEqual({});
  });

  test("handles an empty span list without touching the inner exporter", async () => {
    const inner = makeInner();
    const exporter = new ContentPolicyOTLPSpanExporter(inner, "metadata-only");

    const result = await exportAndWait(exporter, []);

    expect(result.code).toBe(0);
    expect(inner.export).not.toHaveBeenCalled();
  });

  test("a throwing inner exporter surfaces FAILURE instead of throwing", async () => {
    const inner = makeInner();
    inner.export.mockImplementation(() => {
      throw new Error("transport down");
    });
    const exporter = new ContentPolicyOTLPSpanExporter(inner, "metadata-only");

    const result = await exportAndWait(exporter, [makeMockSpan()]);

    expect(result.code).toBe(1); // EXPORT_FAILED
  });

  test("shutdown swallows an inner rejection", async () => {
    const inner = makeInner();
    inner.shutdown.mockRejectedValue(new Error("shutdown boom"));
    const exporter = new ContentPolicyOTLPSpanExporter(inner, "metadata-only");

    await expect(exporter.shutdown()).resolves.toBeUndefined();
  });

  test("forceFlush swallows an inner rejection", async () => {
    const inner = makeInner();
    inner.forceFlush.mockRejectedValue(new Error("flush boom"));
    const exporter = new ContentPolicyOTLPSpanExporter(inner, "metadata-only");

    await expect(exporter.forceFlush()).resolves.toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// runInstrumentor (instrumentor hook)
// ---------------------------------------------------------------------------

describe("runInstrumentor", () => {
  test("calls the registered instrumentor exactly once", () => {
    // test_calls_registered_instrumentor
    const fn = vi.fn();
    registerInstrumentor(fn);

    runInstrumentor();

    expect(fn).toHaveBeenCalledOnce();
  });

  test("logs a warning when no instrumentor is registered", () => {
    // test_logs_warning_when_no_hook
    runInstrumentor();

    expect(mocks.logger.warn).toHaveBeenCalledOnce();
    const [arg] = mocks.logger.warn.mock.calls[0] ?? [];
    expect(String(arg)).toMatch(/No instrumentor hook registered/);
  });

  test("catches instrumentor exceptions and logs a warning", () => {
    // test_instrumentor_failure_does_not_raise
    registerInstrumentor(() => {
      throw new Error("boom");
    });

    expect(() => runInstrumentor()).not.toThrow();
    expect(mocks.logger.warn).toHaveBeenCalledOnce();
    // Second arg is the message string (first is the structured fields object).
    const [, msg] = mocks.logger.warn.mock.calls[0] ?? [];
    expect(String(msg)).toMatch(/Framework instrumentor failed/);
  });
});
