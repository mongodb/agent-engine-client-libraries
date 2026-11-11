import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { trace } from "@opentelemetry/api";
import type { ExportResult } from "@opentelemetry/core";
import { ProtobufTraceSerializer } from "@opentelemetry/otlp-transformer";
import type {
  IExportTraceServiceResponse,
  ISerializer,
} from "@opentelemetry/otlp-transformer";
import { BasicTracerProvider } from "@opentelemetry/sdk-trace-base";
import { InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";
import { SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import type { ReadableSpan, SpanExporter } from "@opentelemetry/sdk-trace-base";

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

const mongo = vi.hoisted(() => {
  const collection = {
    insertMany: vi.fn().mockResolvedValue({ insertedCount: 1 }),
    dbName: "testdb",
    collectionName: "traces",
  };
  const dbObj = { command: vi.fn(), collection: vi.fn(() => collection) };
  const instance = {
    connect: vi.fn().mockResolvedValue(undefined),
    db: vi.fn(() => dbObj),
    close: vi.fn().mockResolvedValue(undefined),
  };
  return { instance, dbObj, collection };
});

vi.mock("../../src/logger.js", () => ({
  getLogger: () => mocks.logger,
  setupLogging: vi.fn(),
}));

vi.mock("mongodb", () => ({
  MongoClient: function () {
    return mongo.instance;
  },
}));

import {
  PlatformSpanExporter,
  forceFlushTracing,
  getFetchOptionsWithTLS,
  getTracer,
  platformTraceExporterEnabled,
  setupTracing,
  shutdownTracing,
  tracingStatus,
} from "../../src/index.js";
import {
  ATTR_INPUT_MESSAGES_OFFSET,
  ATTR_INPUT_MESSAGES_TOTAL,
  RUNTIME_MODE_ATTRIBUTE,
} from "../../src/tracing/conversation.js";

function makeSpans(count: number, padBytes = 0): ReadableSpan[] {
  const spans: ReadableSpan[] = [];
  for (let i = 0; i < count; i++) {
    spans.push({
      spanContext: () => ({
        traceId: "12345678123456781234567812345678",
        spanId: i.toString(16).padStart(16, "0"),
        traceFlags: 1,
        isRemote: false,
      }),
      name: `span-${i}`,
      kind: 0,
      startTime: [1, 0],
      endTime: [2, 0],
      status: { code: 1 },
      attributes: padBytes ? { pad: "x".repeat(padBytes) } : { key: "value" },
      resource: { attributes: { "service.name": "test_service" } },
      events: [],
    } as unknown as ReadableSpan);
  }
  return spans;
}

const encoder = new TextEncoder();
const decoder = new TextDecoder();

// JSON keeps payload size controllable while letting tests inspect delivered
// IDs.
function fakeSerializer(
  deserialize?: (bytes: Uint8Array) => IExportTraceServiceResponse,
): ISerializer<ReadableSpan[], IExportTraceServiceResponse> {
  return {
    serializeRequest: (spans) =>
      encoder.encode(
        JSON.stringify(
          spans.map((s) => ({
            id: s.spanContext().spanId,
            a: s.attributes,
          })),
        ),
      ),
    deserializeResponse: deserialize ?? (() => ({})),
  };
}

function spanIds(body: Uint8Array | undefined): string[] {
  if (!body) return [];
  return (JSON.parse(decoder.decode(body)) as Array<{ id: string }>).map(
    (s) => s.id,
  );
}

function conversationSpan(
  spanId: string,
  contents: string[],
  extra: Record<string, unknown> = {},
): ReadableSpan {
  const attributes: Record<string, unknown> = { ...extra };
  contents.forEach((content, index) => {
    attributes[`llm.input_messages.${index}.message.role`] = "user";
    attributes[`llm.input_messages.${index}.message.content`] = content;
  });
  return {
    spanContext: () => ({
      traceId: "12345678123456781234567812345678",
      spanId: spanId.padStart(16, "0"),
      traceFlags: 1,
      isRemote: false,
    }),
    name: "llm",
    kind: 0,
    startTime: [1, 0],
    endTime: [2, 0],
    status: { code: 1 },
    attributes,
    resource: {
      attributes: {
        "service.name": "test_service",
        [RUNTIME_MODE_ATTRIBUTE]: "aer",
      },
    },
    events: [],
  } as unknown as ReadableSpan;
}

function deliveredAttributes(
  body: Uint8Array | undefined,
): Record<string, unknown> {
  if (!body) return {};
  const parsed = JSON.parse(decoder.decode(body)) as Array<{
    a: Record<string, unknown>;
  }>;
  return parsed[0]?.a ?? {};
}

function deliveredContents(attributes: Record<string, unknown>): string[] {
  return Object.entries(attributes)
    .filter(
      ([key]) =>
        key.startsWith("llm.input_messages.") &&
        key.endsWith(".message.content"),
    )
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([, value]) => String(value));
}

interface FakeResponseInit {
  headers?: Record<string, string>;
  body?: Uint8Array;
}

function response(status: number, init: FakeResponseInit = {}): Response {
  const headers = new Map(
    Object.entries(init.headers ?? {}).map(([k, v]) => [k.toLowerCase(), v]),
  );
  return {
    status,
    ok: status >= 200 && status < 300,
    headers: { get: (key: string) => headers.get(key.toLowerCase()) ?? null },
    arrayBuffer: async () =>
      init.body
        ? init.body.buffer.slice(
            init.body.byteOffset,
            init.body.byteOffset + init.body.byteLength,
          )
        : new ArrayBuffer(0),
  } as unknown as Response;
}

function exportAndWait(
  exporter: SpanExporter,
  spans: ReadableSpan[],
): Promise<ExportResult> {
  return new Promise((resolve) => exporter.export(spans, resolve));
}

class FakeClock {
  now = 0;
  advance(seconds: number): void {
    this.now += seconds;
  }
}

describe("PlatformSpanExporter", () => {
  beforeEach(() => {
    for (const k of ["info", "error", "warn", "debug"] as const)
      mocks.logger[k].mockClear();
  });

  test("posts OTLP protobuf to the ingest path", async () => {
    const memory = new InMemorySpanExporter();
    const provider = new BasicTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(memory)],
    });
    const tracer = provider.getTracer("test");
    tracer.startSpan("real-span").end();
    const spans = memory.getFinishedSpans();
    const expected = ProtobufTraceSerializer.serializeRequest(spans);

    const calls: Array<{ url: string; init: RequestInit }> = [];
    const fetchImpl = vi.fn(async (url: string, init: RequestInit) => {
      calls.push({ url, init });
      return response(200);
    });
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
    });

    const result = await exportAndWait(exporter, spans);

    expect(result.code).toBe(0);
    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe("http://oe:8000/internal/traces");
    const headers = calls[0]?.init.headers as Record<string, string>;
    expect(headers["Content-Type"]).toBe("application/x-protobuf");
    expect(new Uint8Array(calls[0]?.init.body as Uint8Array)).toEqual(
      new Uint8Array(expected ?? new Uint8Array()),
    );
  });

  test("empty span list is a no-op", async () => {
    const fetchImpl = vi.fn();
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(),
    });

    expect((await exportAndWait(exporter, [])).code).toBe(0);
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  test.each([404, 429, 502, 503, 504])(
    "retries retryable status %i then succeeds",
    async (status) => {
      let calls = 0;
      const fetchImpl = vi.fn(async () => {
        calls += 1;
        return calls === 1 ? response(status) : response(200);
      });
      const exporter = new PlatformSpanExporter("http://oe:8000", {
        fetchImpl: fetchImpl as unknown as typeof fetch,
        tlsOptions: () => undefined,
        serializer: fakeSerializer(),
        sleep: () => Promise.resolve(),
      });

      expect((await exportAndWait(exporter, makeSpans(1))).code).toBe(0);
      expect(calls).toBe(2);
    },
  );

  test("retries a connection error then succeeds", async () => {
    let calls = 0;
    const fetchImpl = vi.fn(async () => {
      calls += 1;
      if (calls === 1) throw new Error("refused");
      return response(200);
    });
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(),
      sleep: () => Promise.resolve(),
    });

    expect((await exportAndWait(exporter, makeSpans(1))).code).toBe(0);
    expect(calls).toBe(2);
  });

  test("terminal status drops without retry", async () => {
    const fetchImpl = vi.fn(async () => response(400));
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(),
    });

    expect((await exportAndWait(exporter, makeSpans(1))).code).toBe(1);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(exporter.lastError).not.toBeNull();
  });

  test("honors a Retry-After header", async () => {
    let calls = 0;
    const sleeps: number[] = [];
    const fetchImpl = vi.fn(async () => {
      calls += 1;
      return calls === 1
        ? response(503, { headers: { "Retry-After": "2" } })
        : response(200);
    });
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(),
      exportTimeoutSeconds: 30,
      sleep: (s) => {
        sleeps.push(s);
        return Promise.resolve();
      },
    });

    expect((await exportAndWait(exporter, makeSpans(1))).code).toBe(0);
    expect(sleeps).toEqual([2]);
  });

  test("accepts an HTTP-date Retry-After without a timezone", async () => {
    let calls = 0;
    const sleeps: number[] = [];
    const fetchImpl = vi.fn(async () => {
      calls += 1;
      return calls === 1
        ? response(503, {
            headers: { "Retry-After": "Wed, 21 Oct 2015 07:28:00" },
          })
        : response(200);
    });
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(),
      exportTimeoutSeconds: 30,
      sleep: (s) => {
        sleeps.push(s);
        return Promise.resolve();
      },
    });

    expect((await exportAndWait(exporter, makeSpans(1))).code).toBe(0);
    expect(sleeps).toEqual([0]);
  });

  test("gives up when Retry-After exceeds the deadline", async () => {
    const fetchImpl = vi.fn(async () =>
      response(503, { headers: { "Retry-After": "30" } }),
    );
    const sleeps: number[] = [];
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(),
      exportTimeoutSeconds: 5,
      clock: () => 0,
      sleep: (s) => {
        sleeps.push(s);
        return Promise.resolve();
      },
    });

    expect((await exportAndWait(exporter, makeSpans(1))).code).toBe(1);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(sleeps).toEqual([]);
  });

  test("does not count a late completion as success", async () => {
    const clock = new FakeClock();
    const fetchImpl = vi.fn(async () => {
      clock.advance(10.5);
      return response(200);
    });
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(),
      exportTimeoutSeconds: 10,
      clock: () => clock.now,
    });

    expect((await exportAndWait(exporter, makeSpans(1))).code).toBe(1);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(exporter.lastError).not.toBeNull();
  });

  test("aborts an in-flight request at the export deadline", async () => {
    let started = false;
    let aborted = false;
    const fetchImpl = vi.fn((_url: string, init: RequestInit) => {
      started = true;
      return new Promise<Response>((_resolve, reject) => {
        init.signal?.addEventListener("abort", () => {
          aborted = true;
          reject(new DOMException("aborted", "AbortError"));
        });
      });
    });
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(),
      exportTimeoutSeconds: 0.05,
      perAttemptTimeoutSeconds: 0.05,
    });

    expect((await exportAndWait(exporter, makeSpans(1))).code).toBe(1);
    expect(started).toBe(true);
    expect(aborted).toBe(true);
  });

  test("retries after an attempt timeout within the budget", async () => {
    let calls = 0;
    const fetchImpl = vi.fn((_url: string, init: RequestInit) => {
      calls += 1;
      if (calls === 1) {
        return new Promise<Response>((_resolve, reject) => {
          init.signal?.addEventListener("abort", () =>
            reject(new DOMException("aborted", "AbortError")),
          );
        });
      }
      return Promise.resolve(response(200));
    });
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(),
      exportTimeoutSeconds: 5,
      perAttemptTimeoutSeconds: 0.05,
      sleep: () => Promise.resolve(),
    });

    expect((await exportAndWait(exporter, makeSpans(1))).code).toBe(0);
    expect(calls).toBe(2);
    expect(exporter.lastError).toBeNull();
  });

  test("splits a batch on 413 and delivers every span exactly once", async () => {
    const accepted: Uint8Array[] = [];
    const fetchImpl = vi.fn(async (_url: string, init: RequestInit) => {
      const body = new Uint8Array(init.body as Uint8Array);
      if (spanIds(body).length > 2) return response(413);
      accepted.push(body);
      return response(200);
    });
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(),
    });
    const spans = makeSpans(4);

    expect((await exportAndWait(exporter, spans)).code).toBe(0);
    expect(fetchImpl).toHaveBeenCalledTimes(3);
    const delivered = accepted.flatMap((body) => spanIds(body)).sort();
    const expected = spans.map((s) => s.spanContext().spanId).sort();
    expect(delivered).toEqual(expected);
  });

  test("drops a single oversized span", async () => {
    const fetchImpl = vi.fn();
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(),
      maxRequestBytes: 1,
    });

    expect((await exportAndWait(exporter, makeSpans(1))).code).toBe(1);
    expect(fetchImpl).not.toHaveBeenCalled();
    expect(exporter.droppedSpans).toBe(1);
    expect(exporter.lastError).not.toBeNull();
  });

  test("drops a single span the OE rejects with 413", async () => {
    const fetchImpl = vi.fn(async () => response(413));
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(),
    });

    expect((await exportAndWait(exporter, makeSpans(1))).code).toBe(1);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(exporter.droppedSpans).toBe(1);
    expect(exporter.lastError).not.toBeNull();
  });

  test("an oversized span degrades status even when a sibling chunk succeeds", async () => {
    const fetchImpl = vi.fn(async () => response(200));
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(),
      maxRequestBytes: 1000,
    });
    const spans = [...makeSpans(1, 2000), ...makeSpans(1)];

    expect((await exportAndWait(exporter, spans)).code).toBe(1);
    expect(exporter.droppedSpans).toBe(1);
    expect(exporter.lastError).not.toBeNull();
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  test("chunks a batch larger than the body limit", async () => {
    const fetchImpl = vi.fn(async () => {
      return response(200, { body: new Uint8Array(0) });
    });
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(),
      maxRequestBytes: 1500,
    });

    expect((await exportAndWait(exporter, makeSpans(4, 1000))).code).toBe(0);
    expect(fetchImpl.mock.calls.length).toBeGreaterThan(1);
    const delivered = fetchImpl.mock.calls.flatMap((call) =>
      spanIds((call[1] as RequestInit).body as Uint8Array),
    );
    expect(delivered).toHaveLength(4);
  });

  test("partial_success is not retried", async () => {
    const fetchImpl = vi.fn(async () =>
      response(200, { body: new Uint8Array([1]) }),
    );
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(() => ({
        partialSuccess: { rejectedSpans: 1 },
      })),
    });

    expect((await exportAndWait(exporter, makeSpans(2))).code).toBe(0);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  test("a partial rejection is recorded and a later clean export clears it", async () => {
    let rejected = 2;
    const fetchImpl = vi.fn(async () =>
      response(200, { body: new Uint8Array([1]) }),
    );
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(() =>
        rejected > 0 ? { partialSuccess: { rejectedSpans: rejected } } : {},
      ),
    });

    expect((await exportAndWait(exporter, makeSpans(2))).code).toBe(0);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(exporter.droppedSpans).toBe(2);
    expect(exporter.lastError).not.toBeNull();

    rejected = 0;
    expect((await exportAndWait(exporter, makeSpans(1))).code).toBe(0);
    expect(exporter.droppedSpans).toBe(2);
    expect(exporter.lastError).toBeNull();
  });

  test("applies mTLS options on every attempt", async () => {
    let calls = 0;
    const dispatcher = { marker: "mtls-agent" };
    const tlsOptions = vi.fn(() => ({ dispatcher }) as unknown as RequestInit);
    const seen: unknown[] = [];
    const fetchImpl = vi.fn(async (_url: string, init: RequestInit) => {
      seen.push((init as { dispatcher?: unknown }).dispatcher);
      calls += 1;
      return calls === 1 ? response(503) : response(200);
    });
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions,
      serializer: fakeSerializer(),
      sleep: () => Promise.resolve(),
    });

    expect((await exportAndWait(exporter, makeSpans(1))).code).toBe(0);
    expect(calls).toBe(2);
    expect(tlsOptions).toHaveBeenCalledTimes(2);
    expect(seen).toEqual([dispatcher, dispatcher]);
  });

  test("defaults to the platform mTLS options", () => {
    const exporter = new PlatformSpanExporter("http://oe:8000");
    expect(
      (
        exporter as unknown as {
          tlsOptions: (url: string) => RequestInit | undefined;
        }
      ).tlsOptions,
    ).toBe(getFetchOptionsWithTLS);
  });

  test("forceFlush waits for an already-dispatched export", async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const fetchImpl = vi.fn(async () => {
      await gate;
      return response(200);
    });
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(),
    });

    // A scheduled processor flush can empty its buffer before this request
    // settles.
    const inFlight = exportAndWait(exporter, makeSpans(1));
    let drained = false;
    const drain = exporter.forceFlush().then(() => {
      drained = true;
    });
    await Promise.resolve();
    expect(drained).toBe(false);

    release();
    await drain;
    expect(drained).toBe(true);
    expect((await inFlight).code).toBe(0);
  });

  test("trims the conversation to newly added messages before encoding", async () => {
    const bodies: Uint8Array[] = [];
    const fetchImpl = vi.fn(async (_url: string, init: RequestInit) => {
      bodies.push(init.body as Uint8Array);
      return response(200);
    });
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(),
    });

    await exportAndWait(exporter, [
      conversationSpan("1", ["hi"], { "input.value": "hi" }),
    ]);
    await exportAndWait(exporter, [
      conversationSpan("2", ["hi", "there"], { "input.value": "hi there" }),
    ]);

    const first = deliveredAttributes(bodies[0]);
    expect(deliveredContents(first)).toEqual(["hi"]);
    expect(first[ATTR_INPUT_MESSAGES_OFFSET]).toBe(0);
    expect(first[ATTR_INPUT_MESSAGES_TOTAL]).toBe(1);
    expect(first["input.value"]).toBeUndefined();

    const second = deliveredAttributes(bodies[1]);
    expect(deliveredContents(second)).toEqual(["there"]);
    expect(second[ATTR_INPUT_MESSAGES_OFFSET]).toBe(1);
    expect(second[ATTR_INPUT_MESSAGES_TOTAL]).toBe(2);
    expect(second["input.value"]).toBeUndefined();
  });

  test("a refused export does not advance the conversation tracker", async () => {
    const bodies: Uint8Array[] = [];
    let status = 400;
    const fetchImpl = vi.fn(async (_url: string, init: RequestInit) => {
      bodies.push(init.body as Uint8Array);
      return response(status);
    });
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(),
    });

    expect(
      (await exportAndWait(exporter, [conversationSpan("1", ["hi"])])).code,
    ).toBe(1);

    status = 200;
    await exportAndWait(exporter, [conversationSpan("2", ["hi", "there"])]);

    const second = deliveredAttributes(bodies[1]);
    expect(deliveredContents(second)).toEqual(["hi", "there"]);
    expect(second[ATTR_INPUT_MESSAGES_OFFSET]).toBe(0);
  });
});

const URI = "mongodb://user:pass@host:27017";

describe("setupTracing platform exporter wiring", () => {
  beforeEach(() => {
    for (const k of ["info", "error", "warn", "debug"] as const)
      mocks.logger[k].mockClear();
    mongo.instance.connect.mockClear();
    mongo.instance.close.mockResolvedValue(undefined);
    mongo.instance.db.mockReturnValue(mongo.dbObj);
    mongo.dbObj.collection.mockReturnValue(mongo.collection);
    mongo.dbObj.command.mockReset();
  });

  afterEach(async () => {
    await shutdownTracing();
    trace.disable();
    delete process.env["OE_URL"];
    delete process.env["AGENTIC_TRACE_EXPORTER"];
    delete process.env["OTEL_EXPORTER_OTLP_ENDPOINT"];
    delete process.env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"];
  });

  test("installs the platform exporter when OE_URL is set", async () => {
    process.env["OE_URL"] = "http://oe:8000";

    await setupTracing({ serviceName: "svc-platform" });

    expect(tracingStatus()).toEqual({
      database_exporter: "disabled",
      platform_exporter: "attached",
    });
  });

  test.each(["", "1", "true", "TRUE", "yes", "off", "garbage"])(
    "platformTraceExporterEnabled is on for %j",
    (value) => {
      expect(
        platformTraceExporterEnabled({ AGENTIC_TRACE_EXPORTER: value }),
      ).toBe(true);
      expect(platformTraceExporterEnabled({})).toBe(true);
    },
  );

  test.each(["0", "false", "FALSE", " false "])(
    "platformTraceExporterEnabled opt-out value %j",
    (value) => {
      expect(
        platformTraceExporterEnabled({ AGENTIC_TRACE_EXPORTER: value }),
      ).toBe(false);
    },
  );

  test("opt-out keeps the MongoDB exporter and warns", async () => {
    process.env["OE_URL"] = "http://oe:8000";
    process.env["AGENTIC_TRACE_EXPORTER"] = "0";
    mongo.dbObj.command.mockResolvedValue({ ok: 1 });

    await setupTracing({
      serviceName: "svc-optout",
      mongodbUri: URI,
      mongodbDatabaseName: "testdb",
    });

    expect(tracingStatus()).toEqual({
      database_exporter: "attached",
      platform_exporter: "disabled",
    });
    expect(mocks.logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("without PII redaction"),
    );
  });

  test("an unreachable OE loses the batch and reports degraded, with no Mongo fallback", async () => {
    process.env["OE_URL"] = "http://oe:8000";
    const fetchImpl = vi.fn(async () => {
      throw new Error("refused");
    });
    const clock = new FakeClock();
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(),
      clock: () => clock.now,
      sleep: async (seconds) => clock.advance(seconds),
    });

    const result = await exportAndWait(exporter, makeSpans(1));

    expect(result.code).toBe(1);
    expect(fetchImpl.mock.calls.length).toBeGreaterThan(1);
    expect(exporter.lastError).not.toBeNull();
    expect(mongo.collection.insertMany).not.toHaveBeenCalled();
  });

  test("sustained 503 backpressure drops the batch, then a clean export recovers", async () => {
    let saturated = true;
    const fetchImpl = vi.fn(async () =>
      saturated ? response(503) : response(200),
    );
    const clock = new FakeClock();
    const exporter = new PlatformSpanExporter("http://oe:8000", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      tlsOptions: () => undefined,
      serializer: fakeSerializer(),
      clock: () => clock.now,
      sleep: async (seconds) => clock.advance(seconds),
    });

    expect((await exportAndWait(exporter, makeSpans(1))).code).toBe(1);
    expect(fetchImpl.mock.calls.length).toBeGreaterThan(1);
    expect(exporter.lastError).not.toBeNull();

    saturated = false;
    expect((await exportAndWait(exporter, makeSpans(1))).code).toBe(0);
    expect(exporter.lastError).toBeNull();
  });

  test("prefers OE_URL over a MongoDB collection", async () => {
    process.env["OE_URL"] = "http://oe:8000";

    await setupTracing({
      serviceName: "svc-both",
      mongodbUri: URI,
      mongodbDatabaseName: "testdb",
    });

    expect(tracingStatus()).toEqual({
      database_exporter: "disabled",
      platform_exporter: "attached",
    });
    expect(mongo.instance.connect).not.toHaveBeenCalled();
  });

  test("without OE_URL the MongoDB exporter is used", async () => {
    mongo.dbObj.command.mockResolvedValue({ ok: 1 });

    await setupTracing({
      serviceName: "svc-mongo",
      mongodbUri: URI,
      mongodbDatabaseName: "testdb",
    });

    expect(tracingStatus()).toEqual({
      database_exporter: "attached",
      platform_exporter: "disabled",
    });
  });

  test("forceFlushTracing waits for an export the schedule timer already dispatched", async () => {
    // The scheduled flush can remove a batch from the processor buffer before
    // export settles; provider.forceFlush() alone does not wait for that export.
    vi.useFakeTimers();
    try {
      process.env["OE_URL"] = "http://oe:8000";
      let release!: () => void;
      const gate = new Promise<void>((resolve) => {
        release = resolve;
      });
      let started = false;
      const fetchImpl = vi.fn(async () => {
        started = true;
        await gate;
        return response(200);
      });
      vi.stubGlobal("fetch", fetchImpl);

      await setupTracing({ serviceName: "svc-inflight" });

      getTracer("test").startSpan("dispatched").end();
      await vi.advanceTimersByTimeAsync(2000);
      expect(started).toBe(true);

      let flushed: boolean | null = null;
      const flush = forceFlushTracing(5000).then((ok) => {
        flushed = ok;
      });
      for (let i = 0; i < 20; i++) await Promise.resolve();
      expect(flushed).toBeNull();

      release();
      await flush;
      expect(flushed).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  test("forceFlushTracing still drains in-flight exports when the buffered flush fails", async () => {
    // A buffered batch that fails makes provider.forceFlush() reject; that
    // rejection must not abandon a scheduled export still awaiting the OE.
    vi.useFakeTimers();
    try {
      process.env["OE_URL"] = "http://oe:8000";
      let release!: () => void;
      const gate = new Promise<void>((resolve) => {
        release = resolve;
      });
      let calls = 0;
      const fetchImpl = vi.fn(async () => {
        calls += 1;
        if (calls === 1) {
          await gate;
          return response(200);
        }
        return response(400);
      });
      vi.stubGlobal("fetch", fetchImpl);

      await setupTracing({ serviceName: "svc-inflight-failed-flush" });

      getTracer("test").startSpan("dispatched").end();
      await vi.advanceTimersByTimeAsync(2000);
      expect(calls).toBe(1);

      // A second batch the buffered flush will fail on.
      getTracer("test").startSpan("buffered").end();

      let flushed: boolean | null = null;
      const flush = forceFlushTracing(5000).then((ok) => {
        flushed = ok;
      });
      for (let i = 0; i < 20; i++) await Promise.resolve();
      expect(flushed).toBeNull();

      release();
      await flush;
      expect(flushed).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });
});
