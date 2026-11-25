import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import Fastify from "fastify";
import { context, propagation, trace } from "@opentelemetry/api";

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

const exportControl = vi.hoisted(() => ({
  bodies: [] as Uint8Array[],
  onTraceExport: null as null | (() => Promise<void>),
}));

vi.mock("../../src/logger.js", () => ({
  getLogger: () => mocks.logger,
  setupLogging: vi.fn(),
}));

import { AERServer, setupTracing, shutdownTracing } from "../../src/index.js";
import type { ExecuteRequest } from "../../src/index.js";
import { forceFlushTracing } from "../../src/tracing/setup.js";
import type { ExecutorCallbackRequest } from "../../src/models.js";
import { AER_EXECUTE } from "../../src/span_names.js";
import { CallbackDelivery } from "../../src/server/callback_delivery.js";

const OE_URL = "http://oe:8000";
const TRACE_INGEST_PATH = "/internal/traces";
const CALLER_TRACE_ID = "0af7651916cd43dd8448eb211c80319c";
const TRACEPARENT = `00-${CALLER_TRACE_ID}-b7ad6b7169203331-01`;

interface CallbackDeliveryPrivates {
  pending: Map<string, { oeUrl: string; callback: Record<string, unknown> }>;
  tasks: Map<string, Promise<void>>;
  retryController: AbortController;
}

function makeRouteServer(): AERServer {
  const server = Object.create(AERServer.prototype) as AERServer & {
    runtime: Record<string, unknown>;
    pendingSessionFinish: Map<string, ExecutorCallbackRequest>;
    callbackDelivery: CallbackDeliveryPrivates;
    handleExecute: ReturnType<typeof vi.fn>;
  };
  server.runtime = {};
  server.pendingSessionFinish = new Map();
  server.callbackDelivery =
    new CallbackDelivery() as unknown as CallbackDeliveryPrivates;
  server.handleExecute = vi
    .fn()
    .mockResolvedValue({ status: "completed", result: "ok" });
  return server;
}

function executeRequest(executionId: string): ExecuteRequest {
  return {
    execution_id: executionId,
    message: "hi",
    platform_api_url: "http://oe",
    resume: false,
    user_id: "u-1",
    session_id: "session-1",
    workspace_id: "workspace-1",
    thread_id: executionId,
  } as ExecuteRequest;
}

function response(status: number): Response {
  return {
    status,
    ok: status >= 200 && status < 300,
    headers: { get: () => null },
    arrayBuffer: async () => new ArrayBuffer(0),
  } as unknown as Response;
}

function deliveredSpans(body: Uint8Array | undefined): string {
  return body ? new TextDecoder().decode(body) : "";
}

describe("AER run root survives pod teardown", () => {
  beforeEach(async () => {
    exportControl.bodies = [];
    exportControl.onTraceExport = null;
    const fetchImpl = vi.fn(async (url: string, init: RequestInit) => {
      if (String(url).includes(TRACE_INGEST_PATH)) {
        exportControl.bodies.push(init.body as Uint8Array);
        if (exportControl.onTraceExport) await exportControl.onTraceExport();
      }
      return response(200);
    });
    vi.stubGlobal("fetch", fetchImpl);

    process.env["OE_URL"] = OE_URL;
    await setupTracing({ serviceName: "aer-teardown" });
  });

  afterEach(async () => {
    await shutdownTracing();
    trace.disable();
    propagation.disable();
    context.disable();
    delete process.env["OE_URL"];
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  for (const [label, headers] of [
    ["with an inbound traceparent", { traceparent: TRACEPARENT }],
    ["without an inbound traceparent", {}],
  ] as const) {
    test(`shutdownTracing delivers the run root ${label}`, async () => {
      const server = makeRouteServer();
      const app = Fastify();
      server.registerRoutes(app);
      const res = await app.inject({
        method: "POST",
        url: "/execute",
        headers,
        payload: executeRequest("exec-root"),
      });
      await app.close();
      expect(res.statusCode).toBe(200);

      let releaseExport!: () => void;
      let signalExportStarted!: () => void;
      const exportStarted = new Promise<void>((resolve) => {
        signalExportStarted = resolve;
      });
      exportControl.onTraceExport = () => {
        signalExportStarted();
        return new Promise<void>((resolve) => {
          releaseExport = resolve;
        });
      };

      const dispatched = forceFlushTracing(5_000);
      await exportStarted;

      let settled = false;
      const shutdown = shutdownTracing().then(() => {
        settled = true;
      });
      await new Promise((resolve) => setTimeout(resolve, 20));
      expect(settled).toBe(false);

      releaseExport();
      await dispatched;
      await shutdown;
      expect(settled).toBe(true);
      expect(deliveredSpans(exportControl.bodies[0])).toContain(AER_EXECUTE);
    });
  }
});
