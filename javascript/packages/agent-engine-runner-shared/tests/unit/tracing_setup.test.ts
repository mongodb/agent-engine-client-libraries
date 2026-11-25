import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import {
  ROOT_CONTEXT,
  TraceFlags,
  context,
  propagation,
  trace,
} from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

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

const mongo = vi.hoisted(() => ({ ctorArgs: [] as unknown[][] }));

vi.mock("../../src/logger.js", () => ({
  getLogger: () => mocks.logger,
  setupLogging: vi.fn(),
}));

vi.mock("mongodb", () => ({
  MongoClient: function (...args: unknown[]) {
    mongo.ctorArgs.push(args);
    throw new Error("tracing must not create a MongoClient");
  },
}));

import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";

import {
  setupTracing,
  shutdownTracing,
  tracingStatus,
  getTracer,
  resetHooks,
  runWithExecutionContext,
  scrubCredentials,
  TenantRuntime,
} from "../../src/index.js";
import { forceFlushTracing } from "../../src/tracing/setup.js";

const URI = "mongodb://user:pass@host:27017";
const NO_DESTINATION_WARNING =
  "Tracing: no trace destination configured (OE_URL unset, no OTLP endpoint)";

function noDestinationWarnings(): unknown[][] {
  return mocks.logger.warn.mock.calls.filter(
    (call) => call[0] === NO_DESTINATION_WARNING,
  );
}

describe("scrubCredentials", () => {
  const cases: Array<[string, string, string]> = [
    [
      "normal password",
      "connect to mongodb://user:password@host:27017/db failed: TLS",
      "connect to mongodb://***@host:27017/db failed: TLS",
    ],
    [
      "password containing an unescaped '@'",
      "parse error: mongodb://user@ssword@host/db",
      "parse error: mongodb://***@host/db",
    ],
    [
      "password containing a space",
      "parse error: mongodb://user:pass word@host/db",
      "parse error: mongodb://***@host/db",
    ],
    [
      "password containing '/'",
      "parse error: mongodb://user:pa/ss@host/db",
      "parse error: mongodb://***@host/db",
    ],
    [
      "password containing an embedded newline (stdin-set secret)",
      "parse error: mongodb://user:pass\nword@host/db",
      "parse error: mongodb://***@host/db",
    ],
    [
      "credential in a compact JSON body does not swallow later siblings",
      '{"error":"mongodb://u:p@h1/db","email":"a@b.com"}',
      '{"error":"mongodb://***@h1/db","email":"a@b.com"}',
    ],
    [
      "password containing both '@' and space",
      "parse error: mongodb://user:p@ ss word@host/db tail",
      "parse error: mongodb://***@host/db tail",
    ],
    [
      "mongodb+srv scheme",
      "mongodb+srv://user@ss@host/db failed",
      "mongodb+srv://***@host/db failed",
    ],
    [
      "no userinfo — message left intact",
      "mongodb://host/db unreachable",
      "mongodb://host/db unreachable",
    ],
    [
      "multiple URIs in one message",
      "a mongodb://u:p@h1/db b mongodb://u2:p2@h2/db2 c",
      "a mongodb://***@h1/db b mongodb://***@h2/db2 c",
    ],
    [
      "comma-separated host list stays visible",
      "mongodb://u:p@h1:27017,h2:27017/db?replicaSet=rs",
      "mongodb://***@h1:27017,h2:27017/db?replicaSet=rs",
    ],
    [
      "quoted URI",
      'Invalid connection string "mongodb://user:pass@host/db" foo',
      'Invalid connection string "mongodb://***@host/db" foo',
    ],
    [
      "uppercase scheme",
      "MongoDB://User:P@ss@Host/db",
      "MongoDB://***@Host/db",
    ],
    [
      "non-URI content unchanged",
      "something else entirely",
      "something else entirely",
    ],
  ];

  for (const [name, input, expected] of cases) {
    test(name, () => {
      expect(scrubCredentials(input)).toBe(expected);
    });
  }

  test("tolerates non-string input instead of throwing", () => {
    expect(scrubCredentials(undefined)).toBe("undefined");
    expect(scrubCredentials(null)).toBe("null");
    expect(scrubCredentials(42)).toBe("42");
    expect(scrubCredentials({})).toBe("[object Object]");
  });

  test("a string throw carrying a URI is still scrubbed", () => {
    const out = scrubCredentials(
      "driver error: mongodb://admin:p@ss word@atlas-host/db",
    );
    expect(out).toBe("driver error: mongodb://***@atlas-host/db");
    expect(out).not.toContain("p@ss");
  });

  test("no credential fragment survives any password shape", () => {
    for (const password of [
      "p@ss",
      "pass word",
      "pa/ss",
      "p@ ss word",
      "p:a@s/s w",
      "pass\nword",
    ]) {
      const out = scrubCredentials(`err mongodb://user:${password}@host/db`);
      expect(out).toBe("err mongodb://***@host/db");
      expect(out).not.toContain(password);
      expect(out).not.toContain("user");
    }
  });
});

describe("tracing setup", () => {
  beforeEach(() => {
    for (const k of ["info", "error", "warn", "debug"] as const)
      mocks.logger[k].mockClear();
    mongo.ctorArgs.length = 0;
  });

  afterEach(async () => {
    await shutdownTracing();
    trace.disable();
    propagation.disable();
    resetHooks();
    delete process.env["AGENTIC_OBSERVABILITY_DIR"];
    delete process.env["OE_URL"];
    delete process.env["MONGODB_URI"];
    delete process.env["OTEL_EXPORTER_OTLP_ENDPOINT"];
    delete process.env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"];
  });

  test("tracingStatus reports only the platform exporter", async () => {
    await setupTracing({ serviceName: "svc-no-oe" });
    expect(tracingStatus()).toEqual({ platform_exporter: "disabled" });

    await shutdownTracing();
    trace.disable();
    process.env["OE_URL"] = "http://oe:8000";
    await setupTracing({ serviceName: "svc-oe" });
    expect(tracingStatus()).toEqual({ platform_exporter: "attached" });
  });

  test("forceFlushTracing is a no-op when tracing is not set up", async () => {
    expect(await forceFlushTracing(10)).toBe(true);
  });

  test("warns once when neither OE_URL nor an OTLP endpoint is configured", async () => {
    await setupTracing({ serviceName: "svc-no-destination" });

    expect(noDestinationWarnings()).toHaveLength(1);
  });

  test("does not warn when OE_URL is unset but an OTLP endpoint is configured", async () => {
    process.env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://collector:4318";

    await setupTracing({ serviceName: "svc-otlp-only" });

    expect(noDestinationWarnings()).toHaveLength(0);
    expect(tracingStatus()).toEqual({ platform_exporter: "disabled" });
  });

  test("a runtime with MONGODB_URI and no OE_URL never connects to MongoDB for traces", async () => {
    process.env["MONGODB_URI"] = URI;

    const runtime = new TenantRuntime({ appName: "svc-mongo-uri" });
    await (runtime as unknown as { tracingReady: Promise<void> }).tracingReady;
    getTracer("test").startSpan("not-stored").end();
    await shutdownTracing();

    expect(mongo.ctorArgs).toHaveLength(0);
    expect(noDestinationWarnings()).toHaveLength(1);
    expect(mocks.logger.warn).not.toHaveBeenCalledWith(
      expect.stringContaining("Tracing setup failed"),
    );
  });

  test("does not write local trace JSONL", async () => {
    const observabilityDir = mkdtempSync(join(tmpdir(), "tracing-test-"));
    process.env["AGENTIC_OBSERVABILITY_DIR"] = observabilityDir;

    await setupTracing({ serviceName: "svc-no-local-traces" });
    const span = getTracer("test").startSpan("not-written-locally");
    span.end();
    await shutdownTracing();

    expect(existsSync(join(observabilityDir, "traces.jsonl"))).toBe(false);
  });

  test("attaches request-local execution identity to every span", async () => {
    await setupTracing({ serviceName: "svc-execution-context" });

    const spans: ReadableSpan[] = [];
    runWithExecutionContext(
      {
        executionId: "execution-1",
        wrapper: null,
        oeUrl: "http://oe",
        sessionId: "session-1",
        workspaceId: "workspace-1",
      },
      () => {
        for (const name of ["root-one", "root-two"]) {
          const span = getTracer("test").startSpan(name);
          span.end();
          spans.push(span as unknown as ReadableSpan);
        }
      },
    );
    const outside = getTracer("test").startSpan("outside-context");
    outside.end();

    for (const span of spans) {
      expect(span.attributes).toMatchObject({
        "execution.id": "execution-1",
        "session.id": "session-1",
        "thread.id": "session-1",
        "workspace.id": "workspace-1",
      });
    }
    expect((outside as unknown as ReadableSpan).attributes).not.toHaveProperty(
      "execution.id",
    );
  });

  test("OTLP endpoint configured: setup succeeds", async () => {
    process.env["OE_URL"] = "http://oe:8000";
    process.env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://bad-host:4318";

    await expect(
      setupTracing({ serviceName: "svc-otlp-configured" }),
    ).resolves.toBeUndefined();

    expect(tracingStatus()).toEqual({ platform_exporter: "attached" });
  });

  test("registers an async context manager so startActiveSpan nests", async () => {
    await setupTracing({ serviceName: "svc-context-manager" });

    const tracer = getTracer("test");
    let parentTraceId = "";
    let childTraceId = "";
    let activeInsideChild: unknown = null;
    tracer.startActiveSpan("parent", (parent) => {
      parentTraceId = parent.spanContext().traceId;
      tracer.startActiveSpan("child", (child) => {
        activeInsideChild = trace.getActiveSpan();
        childTraceId = child.spanContext().traceId;
        child.end();
      });
      parent.end();
    });

    expect(activeInsideChild).not.toBeNull();
    expect(childTraceId).toBe(parentTraceId);
  });

  test("keeps a pre-existing context manager instead of replacing it", async () => {
    const existing = new AsyncLocalStorageContextManager();
    existing.enable();
    context.setGlobalContextManager(existing);
    try {
      await setupTracing({ serviceName: "svc-existing-context-manager" });

      expect(mocks.logger.debug).toHaveBeenCalledWith(
        "A global context manager is already registered; keeping the existing one",
      );
    } finally {
      context.disable();
    }
  });

  test("global propagator is trace-context only — baggage never crosses", async () => {
    await setupTracing({ serviceName: "svc-propagator" });

    const contextWithBaggage = propagation.setBaggage(
      trace.setSpanContext(ROOT_CONTEXT, {
        traceId: "0123456789abcdef0123456789abcdef",
        spanId: "0123456789abcdef",
        traceFlags: TraceFlags.SAMPLED,
      }),
      propagation.createBaggage({
        secret: { value: "must-not-cross-service-boundary" },
      }),
    );

    const headers: Record<string, string> = {};
    propagation.inject(contextWithBaggage, headers, {
      set(carrier, key, value) {
        carrier[key] = value;
      },
    });

    expect(headers).toMatchObject({
      traceparent: "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
    });
    expect(headers).not.toHaveProperty("baggage");
  });
});
