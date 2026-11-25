import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";

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

import {
  ContentPolicyOTLPSpanExporter,
  getContentCaptureMode,
  runInstrumentor,
  registerInstrumentor,
  resetHooks,
} from "../../src/index.js";
import { buildResourceAttributes } from "../../src/tracing/setup.js";

const RESOURCE_PREFIX = "agentic_platform"; // open-source-refs:ignore — data contract

function makeMockSpan(overrides: Partial<ReadableSpan> = {}): ReadableSpan {
  const base = {
    spanContext: () => ({
      traceId: "12345678123456781234567812345678",
      spanId: "1234567812345678",
      traceFlags: 1,
      isRemote: false,
    }),
    name: "test_span",
    kind: 0,
    startTime: [1, 0] as [number, number],
    endTime: [2, 0] as [number, number],
    status: { code: 1 },
    attributes: { key: "value" },
    resource: { attributes: { "service.name": "test_service" } },
    events: [],
  };
  return { ...base, ...overrides } as unknown as ReadableSpan;
}

function exportAndWait(
  exporter: SpanExporter,
  spans: ReadableSpan[],
): Promise<ExportResult> {
  return new Promise((resolve) => {
    exporter.export(spans, resolve);
  });
}

beforeEach(() => {
  mocks.logger.warn.mockClear();
  mocks.logger.error.mockClear();
  mocks.logger.info.mockClear();
});

afterEach(() => {
  resetHooks();
});

describe("buildResourceAttributes", () => {
  const ENV_KEYS = [
    "ORG_ID",
    "PROJECT_ID",
    "APP_ID",
    "DEPLOYMENT_ID",
    "AGENT_ID",
    "RUNNER_MODE",
    "AGENT_ENGINE_ENVIRONMENT",
  ];

  afterEach(() => {
    for (const key of ENV_KEYS) delete process.env[key];
  });

  test("maps platform env vars onto resource keys and omits absent ones", () => {
    process.env["ORG_ID"] = "org-1";
    process.env["APP_ID"] = "workspace-1";
    process.env["RUNNER_MODE"] = "aer";

    const attrs = buildResourceAttributes("svc");

    expect(attrs["service.name"]).toBe("svc");
    expect(attrs[`${RESOURCE_PREFIX}.org_id`]).toBe("org-1");
    expect(attrs[`${RESOURCE_PREFIX}.workspace_id`]).toBe("workspace-1");
    expect(attrs[`${RESOURCE_PREFIX}.runtime_mode`]).toBe("aer");
    expect(attrs).not.toHaveProperty(`${RESOURCE_PREFIX}.project_id`);
    expect(attrs).not.toHaveProperty(`${RESOURCE_PREFIX}.agent_id`);
  });

  test("maps the environment var to the environment resource key", () => {
    process.env["AGENT_ENGINE_ENVIRONMENT"] = "qa";

    const attrs = buildResourceAttributes("svc");

    expect(attrs[`${RESOURCE_PREFIX}.environment`]).toBe("qa");
  });
});

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

  test("drain waits for an export the batch timer already started", async () => {
    let finishExport!: () => void;
    const inner = makeInner();
    inner.export = vi.fn(
      (_spans: ReadableSpan[], cb: (r: ExportResult) => void) => {
        finishExport = () => cb({ code: 0 });
      },
    );
    const exporter = new ContentPolicyOTLPSpanExporter(inner, "metadata-only");

    void exportAndWait(exporter, [makeMockSpan()]);

    let settled = false;
    const drained = exporter.drain().then(() => {
      settled = true;
    });
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(settled).toBe(false);

    finishExport();
    await expect(drained).resolves.toBeUndefined();
  });

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

  test("metadata-only mode strips the free-form status message but keeps the code", async () => {
    const inner = makeInner();
    const exporter = new ContentPolicyOTLPSpanExporter(inner, "metadata-only");
    const message = "invalid payload: ssn 123-45-6789";
    const span = makeMockSpan({
      status: { code: 2, message },
      events: [
        {
          name: "exception",
          time: [1, 0],
          attributes: { "exception.message": message },
        } as unknown as ReadableSpan["events"][number],
      ],
    });

    await exportAndWait(exporter, [span]);

    const forwarded = inner.export.mock.calls[0]?.[0] as ReadableSpan[];
    expect(forwarded[0]?.status).toEqual({ code: 2 });
    expect(forwarded[0]?.status.message).toBeUndefined();
    expect(forwarded[0]?.events[0]?.attributes).toEqual({});
    expect(span.status.message).toBe(message);
  });

  test("full mode keeps the status message", async () => {
    const inner = makeInner();
    const exporter = new ContentPolicyOTLPSpanExporter(inner, "full");
    const span = makeMockSpan({ status: { code: 2, message: "boom" } });

    await exportAndWait(exporter, [span]);

    const forwarded = inner.export.mock.calls[0]?.[0] as ReadableSpan[];
    expect(forwarded[0]?.status.message).toBe("boom");
  });

  test("metadata-only mode strips content from link attributes", async () => {
    const inner = makeInner();
    const exporter = new ContentPolicyOTLPSpanExporter(inner, "metadata-only");
    const context = {
      traceId: "abcdefabcdefabcdefabcdefabcdefab",
      spanId: "000000000000beef",
      traceFlags: 1,
    };
    const span = makeMockSpan({
      links: [
        {
          context,
          attributes: { "langgraph.node": "branch_a", "input.value": "secret" },
        },
      ],
    } as Partial<ReadableSpan>);

    await exportAndWait(exporter, [span]);

    const forwarded = inner.export.mock.calls[0]?.[0] as ReadableSpan[];
    expect(forwarded[0]?.links).toEqual([
      { context, attributes: { "langgraph.node": "branch_a" } },
    ]);
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

    expect(result.code).toBe(1);
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

describe("runInstrumentor", () => {
  test("calls the registered instrumentor exactly once", () => {
    const fn = vi.fn();
    registerInstrumentor(fn);

    runInstrumentor();

    expect(fn).toHaveBeenCalledOnce();
  });

  test("logs a warning when no instrumentor is registered", () => {
    runInstrumentor();

    expect(mocks.logger.warn).toHaveBeenCalledOnce();
    const [arg] = mocks.logger.warn.mock.calls[0] ?? [];
    expect(String(arg)).toMatch(/No instrumentor hook registered/);
  });

  test("catches instrumentor exceptions and logs a warning", () => {
    registerInstrumentor(() => {
      throw new Error("boom");
    });

    expect(() => runInstrumentor()).not.toThrow();
    expect(mocks.logger.warn).toHaveBeenCalledOnce();
    const [, msg] = mocks.logger.warn.mock.calls[0] ?? [];
    expect(String(msg)).toMatch(/Framework instrumentor failed/);
  });
});
