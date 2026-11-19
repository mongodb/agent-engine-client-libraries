import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import Fastify from "fastify";
import { SpanStatusCode, context, trace } from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
  type ReadableSpan,
} from "@opentelemetry/sdk-trace-base";
import {
  AERServer,
  ContentPolicyOTLPSpanExporter,
  fetchPlatform,
  getCurrentTraceContext,
  type ExecuteRequest,
} from "../../src/index.js";
import { AER_BUILD_AGENT, AER_EXECUTE } from "../../src/span_names.js";

const INBOUND_TRACE_ID = "0af7651916cd43dd8448eb211c80319c";
const INBOUND_SPAN_ID = "b7ad6b7169203331";
const TRACEPARENT = `00-${INBOUND_TRACE_ID}-${INBOUND_SPAN_ID}-01`;

type StreamResult = { content: string; messages: unknown[] };

interface AerTracePrivates {
  executeViaAgentStream: ReturnType<typeof vi.fn>;
  sendStreamChunk: ReturnType<typeof vi.fn>;
  reportCallback: ReturnType<typeof vi.fn>;
}

let exporter: InMemorySpanExporter;
let provider: BasicTracerProvider;
let contextManager: AsyncLocalStorageContextManager;

function makeServer(stream: () => Promise<StreamResult>): AERServer {
  const runtime = {
    appName: "test-agent",
    appVersion: "0.0.1",
    orgId: null,
    projectId: null,
    graphBuilder: {},
    tools: {},
    toolDefinitions: {},
    getAgentConfig: () => ({}),
    getMongodbUri: () => null,
    getAgent: () => ({ execute: async function* () {} }),
  };
  const server = new AERServer(runtime as never) as unknown as AERServer &
    AerTracePrivates;
  server.executeViaAgentStream = vi.fn(stream);
  server.sendStreamChunk = vi.fn().mockResolvedValue(undefined);
  server.reportCallback = vi.fn().mockResolvedValue(undefined);
  return server;
}

function executeRequest(
  overrides: Partial<ExecuteRequest> = {},
): ExecuteRequest {
  return {
    execution_id: "exec-1",
    message: "hello",
    platform_api_url: "http://oe:8000",
    resume: false,
    user_id: "u-1",
    session_id: "session-1",
    workspace_id: "workspace-1",
    thread_id: "exec-1",
    ...overrides,
  } as ExecuteRequest;
}

async function injectExecute(
  server: AERServer,
  headers: Record<string, string>,
  overrides: Partial<ExecuteRequest> = {},
): Promise<{ statusCode: number }> {
  const app = Fastify();
  server.registerRoutes(app);
  try {
    return await app.inject({
      method: "POST",
      url: "/execute",
      headers,
      payload: executeRequest(overrides),
    });
  } finally {
    await app.close();
  }
}

function finished(name: string): ReadableSpan {
  const span = exporter
    .getFinishedSpans()
    .find((candidate) => candidate.name === name);
  if (span === undefined) {
    throw new Error(`no finished span named ${name}`);
  }
  return span;
}

describe("AER /execute inbound trace context", () => {
  beforeEach(() => {
    exporter = new InMemorySpanExporter();
    provider = new BasicTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
    trace.setGlobalTracerProvider(provider);

    contextManager = new AsyncLocalStorageContextManager();
    contextManager.enable();
    context.setGlobalContextManager(contextManager);
  });

  afterEach(async () => {
    await provider.shutdown();
    trace.disable();
    context.disable();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    delete process.env["APP_ID"];
    delete process.env["OE_URL"];
  });

  test("valid traceparent joins the caller's trace and stamps the run root", async () => {
    const server = makeServer(async () => ({ content: "ok", messages: [] }));

    const response = await injectExecute(server, { traceparent: TRACEPARENT });
    expect(response.statusCode).toBe(200);

    const root = finished(AER_EXECUTE);
    expect(root.spanContext().traceId).toBe(INBOUND_TRACE_ID);
    expect(root.parentSpanContext?.spanId).toBe(INBOUND_SPAN_ID);
    expect(root.attributes["openinference.span.kind"]).toBe("AGENT");
    expect(root.attributes["execution.id"]).toBe("exec-1");
    expect(root.attributes["session.id"]).toBe("session-1");
    expect(root.attributes["thread.id"]).toBe("session-1");
    expect(root.attributes["workspace.id"]).toBe("workspace-1");
  });

  test("run spans nest under aer.execute on the caller's trace", async () => {
    const server = makeServer(async () => ({ content: "ok", messages: [] }));

    await injectExecute(server, { traceparent: TRACEPARENT });

    const root = finished(AER_EXECUTE);
    const buildAgent = finished(AER_BUILD_AGENT);
    expect(buildAgent.spanContext().traceId).toBe(INBOUND_TRACE_ID);
    expect(buildAgent.parentSpanContext?.spanId).toBe(
      root.spanContext().spanId,
    );
  });

  test("missing traceparent still roots a single valid trace", async () => {
    const server = makeServer(async () => ({ content: "ok", messages: [] }));

    await injectExecute(server, {});

    const root = finished(AER_EXECUTE);
    expect(root.parentSpanContext).toBeUndefined();
    expect(root.spanContext().traceId).toMatch(/^[0-9a-f]{32}$/);
    const buildAgent = finished(AER_BUILD_AGENT);
    expect(buildAgent.spanContext().traceId).toBe(root.spanContext().traceId);
  });

  test("outbound platform calls carry the run's traceparent", async () => {
    const captured: Record<string, string> = {};
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      Object.assign(captured, init.headers ?? {});
      return new Response(null, { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    let currentTrace: { traceId: string | null; spanId: string | null } | null =
      null;
    const server = makeServer(async () => {
      currentTrace = getCurrentTraceContext();
      await fetchPlatform("http://oe:8000/stream/chunk", { method: "POST" });
      return { content: "ok", messages: [] };
    });

    await injectExecute(server, { traceparent: TRACEPARENT });

    expect(currentTrace).not.toBeNull();
    expect(currentTrace?.traceId).toBe(INBOUND_TRACE_ID);
    expect(captured["traceparent"]).toContain(INBOUND_TRACE_ID);
  });

  test("malformed traceparent does not crash and roots a fresh trace", async () => {
    const server = makeServer(async () => ({ content: "ok", messages: [] }));

    const response = await injectExecute(server, {
      traceparent: "not-a-valid-traceparent",
    });
    expect(response.statusCode).toBe(200);

    const root = finished(AER_EXECUTE);
    expect(root.parentSpanContext).toBeUndefined();
    expect(root.spanContext().traceId).toMatch(/^[0-9a-f]{32}$/);
  });
});

describe("AER run root error redaction (metadata-only OTLP)", () => {
  const PRIVATE_PAYLOAD = "ssn 123-45-6789";
  let innerExporter: InMemorySpanExporter;
  let redactingProvider: BasicTracerProvider;
  let redactionContextManager: AsyncLocalStorageContextManager;

  beforeEach(() => {
    innerExporter = new InMemorySpanExporter();
    redactingProvider = new BasicTracerProvider({
      spanProcessors: [
        new SimpleSpanProcessor(
          new ContentPolicyOTLPSpanExporter(innerExporter, "metadata-only"),
        ),
      ],
    });
    trace.setGlobalTracerProvider(redactingProvider);

    redactionContextManager = new AsyncLocalStorageContextManager();
    redactionContextManager.enable();
    context.setGlobalContextManager(redactionContextManager);
  });

  afterEach(async () => {
    await redactingProvider.shutdown();
    trace.disable();
    context.disable();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  test("a failing run exports neither the payload in status.message nor the exception event", async () => {
    const server = makeServer(async () => {
      throw new Error(`builder rejected payload ${PRIVATE_PAYLOAD}`);
    });

    const response = await injectExecute(server, { traceparent: TRACEPARENT });
    expect(response.statusCode).toBe(500);

    const root = innerExporter
      .getFinishedSpans()
      .find((span) => span.name === AER_EXECUTE);
    expect(root).toBeDefined();
    expect(root?.status.code).toBe(SpanStatusCode.ERROR);
    expect(root?.status.message).toBeUndefined();
    expect(JSON.stringify(root?.attributes ?? {})).not.toContain(
      PRIVATE_PAYLOAD,
    );
    const exceptionEvents = (root?.events ?? []).filter(
      (event) => event.name === "exception",
    );
    expect(exceptionEvents.length).toBeGreaterThan(0);
    for (const event of exceptionEvents) {
      expect(JSON.stringify(event.attributes ?? {})).not.toContain(
        PRIVATE_PAYLOAD,
      );
    }
  });
});
