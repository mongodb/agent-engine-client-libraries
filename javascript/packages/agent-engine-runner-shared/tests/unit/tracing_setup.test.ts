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

const mongo = vi.hoisted(() => {
  const collection = {
    insertMany: vi.fn().mockResolvedValue({ insertedCount: 1 }),
    dbName: "testdb",
    collectionName: "traces",
  };
  const dbObj = {
    command: vi.fn(),
    collection: vi.fn(() => collection),
  };
  const instance = {
    connect: vi.fn().mockResolvedValue(undefined),
    db: vi.fn(() => dbObj),
    close: vi.fn().mockResolvedValue(undefined),
  };
  const ctorArgs: unknown[][] = [];
  return { instance, dbObj, collection, ctorArgs };
});

vi.mock("../../src/logger.js", () => ({
  getLogger: () => mocks.logger,
  setupLogging: vi.fn(),
}));

vi.mock("mongodb", () => ({
  MongoClient: function (...args: unknown[]) {
    mongo.ctorArgs.push(args);
    return mongo.instance;
  },
}));

import {
  setupTracing,
  shutdownTracing,
  tracingStatus,
  attachMongoTracing,
  getTracer,
  resetHooks,
  runWithExecutionContext,
  scrubCredentials,
} from "../../src/index.js";
import { forceFlushTracing } from "../../src/tracing/setup.js";

const URI = "mongodb://user:pass@host:27017";
const MONGO_APP_NAME = "Agent Engine";

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
    mongo.instance.connect.mockResolvedValue(undefined);
    mongo.instance.close.mockResolvedValue(undefined);
    mongo.instance.db.mockReturnValue(mongo.dbObj);
    mongo.dbObj.collection.mockReturnValue(mongo.collection);
    mongo.dbObj.command.mockReset();
    mongo.collection.insertMany.mockClear();
    mongo.ctorArgs.length = 0;
  });

  afterEach(async () => {
    vi.useRealTimers();
    await shutdownTracing();
    trace.disable();
    propagation.disable();
    resetHooks();
    delete process.env["AGENTIC_TRACE_STORE_RETRY_INTERVAL_SECONDS"];
    delete process.env["AGENTIC_OBSERVABILITY_DIR"];
    delete process.env["OE_URL"];
  });

  test("disabled when no store URI is configured", async () => {
    await setupTracing({ serviceName: "svc-no-store" });
    expect(tracingStatus()).toEqual({
      database_exporter: "disabled",
      platform_exporter: "disabled",
    });
  });

  test("forceFlushTracing is a no-op when tracing is not set up", async () => {
    expect(await forceFlushTracing(10)).toBe(true);
  });

  test("forceFlushTracing ships a batched span before it resolves", async () => {
    mongo.dbObj.command.mockResolvedValue({ ok: 1 });
    await setupTracing({
      serviceName: "svc-flush",
      mongodbUri: URI,
      mongodbDatabaseName: "testdb",
    });

    let releaseExport!: () => void;
    mongo.collection.insertMany.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          releaseExport = () => resolve({ insertedCount: 1 });
        }),
    );
    getTracer("test").startSpan("flush-me").end();

    let settled = false;
    const flushed = forceFlushTracing(5_000).then((ok) => {
      settled = true;
      return ok;
    });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(settled).toBe(false);

    releaseExport();
    await expect(flushed).resolves.toBe(true);
    const docs = mongo.collection.insertMany.mock.calls.flatMap(
      (call) => call[0] as Array<{ name: string }>,
    );
    expect(docs.some((doc) => doc.name === "flush-me")).toBe(true);
  });

  test("forceFlushTracing waits for an export the batch timer already started", async () => {
    mongo.dbObj.command.mockResolvedValue({ ok: 1 });
    await setupTracing({
      serviceName: "svc-flush-inflight",
      mongodbUri: URI,
      mongodbDatabaseName: "testdb",
    });

    let releaseExport!: () => void;
    mongo.collection.insertMany.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          releaseExport = () => resolve({ insertedCount: 1 });
        }),
    );
    getTracer("test").startSpan("in-flight").end();

    const first = forceFlushTracing(5_000);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(mongo.collection.insertMany).toHaveBeenCalledTimes(1);

    let settled = false;
    const second = forceFlushTracing(5_000).then((ok) => {
      settled = true;
      return ok;
    });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(settled).toBe(false);

    releaseExport();
    await expect(first).resolves.toBe(true);
    await expect(second).resolves.toBe(true);
  });

  test("shutdownTracing joins an export the batch timer already started", async () => {
    mongo.dbObj.command.mockResolvedValue({ ok: 1 });
    await setupTracing({
      serviceName: "svc-shutdown",
      mongodbUri: URI,
      mongodbDatabaseName: "testdb",
    });

    let releaseExport!: () => void;
    mongo.collection.insertMany.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          releaseExport = () => resolve({ insertedCount: 1 });
        }),
    );
    getTracer("test").startSpan("shutdown-in-flight").end();

    const started = forceFlushTracing(5_000);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(mongo.collection.insertMany).toHaveBeenCalledTimes(1);

    let settled = false;
    const shutdown = shutdownTracing().then(() => {
      settled = true;
    });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(settled).toBe(false);

    releaseExport();
    await started;
    await shutdown;
    expect(settled).toBe(true);
    const docs = mongo.collection.insertMany.mock.calls.flatMap(
      (call) => call[0] as Array<{ name: string }>,
    );
    expect(docs.some((doc) => doc.name === "shutdown-in-flight")).toBe(true);
  });

  test("forceFlushTracing returns false when the export does not complete in time", async () => {
    mongo.dbObj.command.mockResolvedValue({ ok: 1 });
    await setupTracing({
      serviceName: "svc-flush-timeout",
      mongodbUri: URI,
      mongodbDatabaseName: "testdb",
    });

    let releaseExport!: () => void;
    mongo.collection.insertMany.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          releaseExport = () => resolve({ insertedCount: 1 });
        }),
    );
    getTracer("test").startSpan("stuck-flush").end();

    await expect(forceFlushTracing(30)).resolves.toBe(false);
    releaseExport();
  });

  test.each(["throw", "reject"])(
    "a cleanup %s preserves the degraded startup result",
    async (failure) => {
      mongo.dbObj.command.mockRejectedValue(
        new Error("trace store unavailable"),
      );
      const closesBefore = mongo.instance.close.mock.calls.length;
      mongo.instance.close.mockImplementationOnce(() => {
        if (failure === "throw") throw new Error("cleanup failed");
        return Promise.reject(new Error("cleanup failed"));
      });
      process.env["AGENTIC_TRACE_STORE_RETRY_INTERVAL_SECONDS"] = "999";

      await expect(
        setupTracing({ serviceName: "svc-cleanup-failure", mongodbUri: URI }),
      ).resolves.toBeUndefined();

      expect(mongo.instance.close.mock.calls.length).toBe(closesBefore + 1);
      expect(tracingStatus().database_exporter).toBe("degraded");
      const errorMessage = String(mocks.logger.error.mock.calls[0]?.[0] ?? "");
      expect(errorMessage).toContain("trace store unavailable");
      expect(errorMessage).not.toContain("cleanup failed");
    },
  );

  test("does not write local trace JSONL", async () => {
    const observabilityDir = mkdtempSync(join(tmpdir(), "tracing-test-"));
    process.env["AGENTIC_OBSERVABILITY_DIR"] = observabilityDir;

    await setupTracing({ serviceName: "svc-no-local-traces" });
    const span = getTracer("test").startSpan("not-written-locally");
    span.end();
    await shutdownTracing();

    expect(existsSync(join(observabilityDir, "traces.jsonl"))).toBe(false);
  });

  test("attached when the store is reachable at startup", async () => {
    mongo.dbObj.command.mockResolvedValue({ ok: 1 });
    await setupTracing({
      serviceName: "svc-attached",
      mongodbUri: URI,
      mongodbDatabaseName: "testdb",
    });
    expect(tracingStatus()).toEqual({
      database_exporter: "attached",
      platform_exporter: "disabled",
    });
  });

  test("sets the handshake appName on the trace-store client", async () => {
    mongo.dbObj.command.mockResolvedValue({ ok: 1 });
    await setupTracing({
      serviceName: "svc-appname",
      mongodbUri: URI,
      mongodbDatabaseName: "testdb",
    });

    expect(mongo.ctorArgs[0]).toEqual([
      URI,
      expect.objectContaining({ appName: MONGO_APP_NAME }),
    ]);
  });

  test("attaches request-local execution identity to every span", async () => {
    mongo.dbObj.command.mockResolvedValue({ ok: 1 });
    await setupTracing({
      serviceName: "svc-execution-context",
      mongodbUri: URI,
      mongodbDatabaseName: "testdb",
    });

    runWithExecutionContext(
      {
        executionId: "execution-1",
        wrapper: null,
        oeUrl: "http://oe",
        sessionId: "session-1",
        workspaceId: "workspace-1",
      },
      () => {
        getTracer("test").startSpan("root-one").end();
        getTracer("test").startSpan("root-two").end();
      },
    );
    getTracer("test").startSpan("outside-context").end();

    await shutdownTracing();

    const docs = mongo.collection.insertMany.mock.calls.flatMap(
      (call) =>
        call[0] as Array<{
          name: string;
          execution_id?: string;
          session_id?: string;
          attributes: Record<string, unknown>;
        }>,
    );
    const executionDocs = docs.filter((doc) => doc.name.startsWith("root-"));
    expect(executionDocs).toHaveLength(2);
    for (const doc of executionDocs) {
      expect(doc.execution_id).toBe("execution-1");
      expect(doc.session_id).toBe("session-1");
      expect(doc.attributes).toMatchObject({
        "execution.id": "execution-1",
        "session.id": "session-1",
        "thread.id": "session-1",
        "workspace.id": "workspace-1",
      });
    }
    expect(
      docs.find((doc) => doc.name === "outside-context")?.attributes,
    ).not.toHaveProperty("execution.id");
  });

  test("attachMongoTracing is a no-op when no store URI was configured", async () => {
    await setupTracing({ serviceName: "svc-no-uri" });
    expect(() => attachMongoTracing(mongo.collection as never)).not.toThrow();
    expect(mocks.logger.warn).toHaveBeenCalledWith(
      "Cannot attach MongoDB tracing: no store URI was configured at setup",
    );
    expect(tracingStatus()).toEqual({
      database_exporter: "disabled",
      platform_exporter: "disabled",
    });
  });

  test("attachMongoTracing is idempotent once attached", async () => {
    mongo.dbObj.command.mockResolvedValue({ ok: 1 });
    await setupTracing({
      serviceName: "svc-idempotent",
      mongodbUri: URI,
      mongodbDatabaseName: "testdb",
    });
    expect(tracingStatus()).toEqual({
      database_exporter: "attached",
      platform_exporter: "disabled",
    });

    const other = {
      insertMany: vi.fn().mockResolvedValue({ insertedCount: 1 }),
    };
    expect(() => attachMongoTracing(other as never)).not.toThrow();
    expect(tracingStatus()).toEqual({
      database_exporter: "attached",
      platform_exporter: "disabled",
    });
    expect(other.insertMany).not.toHaveBeenCalled();
  });

  test("retries until the store recovers", async () => {
    mongo.dbObj.command
      .mockRejectedValueOnce(
        new Error("connect to mongodb://user:pass@host:27017 failed: TLS"),
      )
      .mockResolvedValueOnce({ ok: 1 });
    process.env["AGENTIC_TRACE_STORE_RETRY_INTERVAL_SECONDS"] = "0.01";
    await setupTracing({
      serviceName: "svc-retry",
      mongodbUri: URI,
      mongodbDatabaseName: "testdb",
    });

    expect(tracingStatus()).toEqual({
      database_exporter: "degraded",
      platform_exporter: "disabled",
    });
    const errorMsg = mocks.logger.error.mock.calls[0]?.[0] ?? "";
    expect(String(errorMsg)).toMatch(/Trace store unavailable at startup/);
    expect(String(errorMsg)).toMatch(/:\/\/\*\*\*@/);
    expect(String(errorMsg)).not.toMatch(/user:pass@/);

    await vi.waitFor(
      () => {
        expect(tracingStatus().database_exporter).toBe("attached");
      },
      { timeout: 2000, interval: 5 },
    );
    expect(mongo.instance.connect).toHaveBeenCalled();

    const span = getTracer("test").startSpan("recovered-span");
    span.end();
    await shutdownTracing();

    expect(mongo.collection.insertMany).toHaveBeenCalled();
    const names = mongo.collection.insertMany.mock.calls
      .flatMap((call) => call[0] as Array<{ name: string }>)
      .map((doc) => doc.name);
    expect(names).toContain("recovered-span");
  });

  test("startup error log masks credentials when the password contains '@' or space", async () => {
    mongo.dbObj.command.mockRejectedValue(
      new Error(
        "Invalid connection string: mongodb://admin:p@ss word@atlas-host.mongodb.net/db",
      ),
    );
    process.env["AGENTIC_TRACE_STORE_RETRY_INTERVAL_SECONDS"] = "999";

    await setupTracing({
      serviceName: "svc-scrub",
      mongodbUri: URI,
      mongodbDatabaseName: "testdb",
    });

    expect(tracingStatus()).toEqual({
      database_exporter: "degraded",
      platform_exporter: "disabled",
    });
    const errorMsg = String(mocks.logger.error.mock.calls[0]?.[0] ?? "");
    expect(errorMsg).toMatch(/Trace store unavailable at startup/);
    expect(errorMsg).toContain("mongodb://***@atlas-host.mongodb.net/db");
    expect(errorMsg).not.toContain("p@ss word");
    expect(errorMsg).not.toContain("p@ss");
    expect(errorMsg).not.toContain("admin:");
  });

  test("stays degraded when the store never recovers", async () => {
    mongo.dbObj.command.mockRejectedValue(new Error("still down"));
    process.env["AGENTIC_TRACE_STORE_RETRY_INTERVAL_SECONDS"] = "0.01";

    await setupTracing({
      serviceName: "svc-stays-down",
      mongodbUri: URI,
      mongodbDatabaseName: "testdb",
    });

    expect(tracingStatus()).toEqual({
      database_exporter: "degraded",
      platform_exporter: "disabled",
    });
    const startupPings = mongo.dbObj.command.mock.calls.length;
    await vi.waitFor(() => {
      expect(mongo.dbObj.command.mock.calls.length).toBeGreaterThanOrEqual(
        startupPings + 3,
      );
    });
    expect(tracingStatus()).toEqual({
      database_exporter: "degraded",
      platform_exporter: "disabled",
    });
  });

  test("shutdown during an in-flight retry stops further retries", async () => {
    vi.useFakeTimers();
    let rejectRetry!: (err: unknown) => void;
    mongo.dbObj.command
      .mockRejectedValueOnce(new Error("startup fail"))
      .mockReturnValueOnce(
        new Promise<never>((_, reject) => {
          rejectRetry = reject;
        }),
      );
    process.env["AGENTIC_TRACE_STORE_RETRY_INTERVAL_SECONDS"] = "0.01";

    await setupTracing({
      serviceName: "svc-race-fail",
      mongodbUri: URI,
      mongodbDatabaseName: "testdb",
    });
    expect(tracingStatus()).toEqual({
      database_exporter: "degraded",
      platform_exporter: "disabled",
    });

    const startupConnects = mongo.instance.connect.mock.calls.length;
    await vi.advanceTimersByTimeAsync(10);
    await vi.waitFor(() => {
      expect(mongo.instance.connect.mock.calls.length).toBe(
        startupConnects + 1,
      );
    });
    const connectsBeforeShutdown = mongo.instance.connect.mock.calls.length;

    await shutdownTracing();
    rejectRetry(new Error("still down"));

    await vi.advanceTimersByTimeAsync(1000);
    expect(mongo.instance.connect.mock.calls.length).toBe(
      connectsBeforeShutdown,
    );
  });

  test("shutdown during an in-flight retry closes the client it opened", async () => {
    vi.useFakeTimers();
    let resolveRetry!: (v: { ok: number }) => void;
    mongo.dbObj.command
      .mockRejectedValueOnce(new Error("startup fail"))
      .mockReturnValueOnce(
        new Promise<{ ok: number }>((resolve) => {
          resolveRetry = resolve;
        }),
      );
    process.env["AGENTIC_TRACE_STORE_RETRY_INTERVAL_SECONDS"] = "0.01";

    await setupTracing({
      serviceName: "svc-race-success",
      mongodbUri: URI,
      mongodbDatabaseName: "testdb",
    });
    expect(tracingStatus()).toEqual({
      database_exporter: "degraded",
      platform_exporter: "disabled",
    });

    const startupConnects = mongo.instance.connect.mock.calls.length;
    await vi.advanceTimersByTimeAsync(10);
    await vi.waitFor(() => {
      expect(mongo.instance.connect.mock.calls.length).toBe(
        startupConnects + 1,
      );
    });
    const connectsBeforeShutdown = mongo.instance.connect.mock.calls.length;
    const closesBeforeShutdown = mongo.instance.close.mock.calls.length;

    await shutdownTracing();
    resolveRetry({ ok: 1 });

    await vi.advanceTimersByTimeAsync(1000);
    expect(mongo.instance.close.mock.calls.length).toBe(
      closesBeforeShutdown + 1,
    );
    expect(mongo.instance.connect.mock.calls.length).toBe(
      connectsBeforeShutdown,
    );
  });

  test("no OTLP endpoint configured: setup succeeds, Mongo unaffected", async () => {
    delete process.env["OTEL_EXPORTER_OTLP_ENDPOINT"];
    delete process.env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"];
    mongo.dbObj.command.mockResolvedValue({ ok: 1 });

    await setupTracing({
      serviceName: "svc-no-otlp",
      mongodbUri: URI,
      mongodbDatabaseName: "testdb",
    });

    expect(tracingStatus()).toEqual({
      database_exporter: "attached",
      platform_exporter: "disabled",
    });
  });

  test("OTLP endpoint configured: OTLP leg added without breaking Mongo setup", async () => {
    process.env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://bad-host:4318";
    mongo.dbObj.command.mockResolvedValue({ ok: 1 });

    await expect(
      setupTracing({
        serviceName: "svc-otlp-configured",
        mongodbUri: URI,
        mongodbDatabaseName: "testdb",
      }),
    ).resolves.toBeUndefined();

    expect(tracingStatus()).toEqual({
      database_exporter: "attached",
      platform_exporter: "disabled",
    });
    delete process.env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"];
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

  test("retry timers are unref'd so they never pin the event loop", async () => {
    mongo.dbObj.command.mockRejectedValue(new Error("still down"));
    process.env["AGENTIC_TRACE_STORE_RETRY_INTERVAL_SECONDS"] = "0.017";
    const RETRY_MS = 17;

    const scheduled: Array<{ timer: NodeJS.Timeout; ms?: number }> = [];
    const realSetTimeout = global.setTimeout;
    vi.spyOn(global, "setTimeout").mockImplementation(((
      cb: (...args: unknown[]) => void,
      ms?: number,
      ...args: unknown[]
    ) => {
      const timer = realSetTimeout(cb, ms, ...args);
      scheduled.push({ timer, ms });
      return timer;
    }) as typeof setTimeout);

    try {
      await setupTracing({
        serviceName: "svc-unref",
        mongodbUri: URI,
        mongodbDatabaseName: "testdb",
      });
      expect(tracingStatus()).toEqual({
        database_exporter: "degraded",
        platform_exporter: "disabled",
      });

      const startupPings = mongo.dbObj.command.mock.calls.length;
      await vi.waitFor(() => {
        expect(mongo.dbObj.command.mock.calls.length).toBeGreaterThanOrEqual(
          startupPings + 2,
        );
      });

      const retryTimers = scheduled.filter(
        (s) => s.ms !== undefined && Math.abs(s.ms - RETRY_MS) < 1,
      );
      expect(retryTimers.length).toBeGreaterThan(1);
      for (const { timer } of retryTimers) {
        expect(timer.hasRef()).toBe(false);
      }
    } finally {
      vi.restoreAllMocks();
    }
  });
});
