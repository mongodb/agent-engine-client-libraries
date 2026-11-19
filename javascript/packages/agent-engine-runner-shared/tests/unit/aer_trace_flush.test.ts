import { describe, test, expect, vi, afterEach } from "vitest";
import Fastify from "fastify";

const flushControl = vi.hoisted(() => ({
  flush: null as null | (() => Promise<boolean>),
}));

vi.mock("../../src/tracing/index.js", async (importOriginal) => {
  const actual = (await importOriginal()) as Record<string, unknown>;
  return {
    ...actual,
    forceFlushTracing: () =>
      flushControl.flush ? flushControl.flush() : Promise.resolve(true),
  };
});

import { AERServer, type ExecuteRequest } from "../../src/index.js";
import type { ExecutorCallbackRequest } from "../../src/models.js";
import { CallbackDelivery } from "../../src/server/callback_delivery.js";

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
  return server;
}

function executeRequest(executionId: string): ExecuteRequest {
  return {
    execution_id: executionId,
    message: "hi",
    platform_api_url: "http://oe",
    resume: false,
    user_id: "u-1",
    thread_id: executionId,
  } as ExecuteRequest;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  flushControl.flush = null;
});

describe("AER session finish: trace flush ordering", () => {
  test("the trace flush completes before the finish POST", async () => {
    const server = makeRouteServer() as unknown as AERServer & {
      handleExecute: ReturnType<typeof vi.fn>;
      pendingSessionFinish: Map<string, ExecutorCallbackRequest>;
    };
    server.pendingSessionFinish.set("exec-flush", {
      execution_id: "exec-flush",
      status: "COMPLETED",
    });
    server.handleExecute = vi
      .fn()
      .mockResolvedValue({ status: "completed", result: "ok" });

    const order: string[] = [];
    let signalFlushStarted!: () => void;
    const flushStarted = new Promise<void>((resolve) => {
      signalFlushStarted = resolve;
    });
    let releaseFlush!: () => void;
    flushControl.flush = () => {
      order.push("flush-start");
      signalFlushStarted();
      return new Promise<boolean>((resolve) => {
        releaseFlush = () => {
          order.push("flush-end");
          resolve(true);
        };
      });
    };

    const fetchMock = vi.fn(async (url: string) => {
      order.push(String(url).includes("/finish") ? "finish" : "other");
      return new Response(null, { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const app = Fastify();
    server.registerRoutes(app);
    const res = await app.inject({
      method: "POST",
      url: "/execute",
      payload: executeRequest("exec-flush"),
    });
    expect(res.statusCode).toBe(200);

    await flushStarted;
    expect(order).toEqual(["flush-start"]);
    expect(fetchMock).not.toHaveBeenCalled();

    releaseFlush();
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(order).toEqual(["flush-start", "flush-end", "finish"]);
  });
});
