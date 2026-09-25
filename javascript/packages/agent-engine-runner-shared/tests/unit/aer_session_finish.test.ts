/**
 * Tests for the AER's post-response session-finish trigger.
 *
 * Two layers are exercised:
 *
 * Part A — `doHandleExecute`'s latch: a finish request recorded by the SDK
 * during the run must reach `pendingSessionFinish` only when the run
 * completes normally (never on suspend, never on an error).
 *
 * Part B — the `/execute` route: the finish POST must dispatch only after
 * the HTTP response has been flushed (`reply.raw`'s "finish" event), and a
 * failing finish POST must never affect the already-sent response.
 */

import { describe, test, expect, vi, afterEach } from "vitest";
import Fastify from "fastify";
import {
  AERServer,
  getCurrentTraceId,
  requestSessionFinish,
  type ExecuteRequest,
  type InterruptResult,
  type StreamingResult,
} from "../../src/index.js";
import { CallbackDelivery } from "../../src/server/callback_delivery.js";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// Part A harness — drive doHandleExecute directly, mirrors
// aer_chunk_delivery_reliability.test.ts's makeAerServer().
// ---------------------------------------------------------------------------

interface CallbackDeliveryPrivates {
  pending: Map<string, { oeUrl: string; callback: Record<string, unknown> }>;
  tasks: Map<string, Promise<void>>;
  retryController: AbortController;
}

interface AerServerPrivates {
  runtime: {
    graphBuilder?: unknown;
    orgId?: string | null;
    getAgent?: () => unknown;
    memoryWriter?: {
      pendingWrites: number;
      drain: ReturnType<typeof vi.fn>;
    } | null;
  };
  sendStreamChunk: ReturnType<typeof vi.fn>;
  reportCallback: ReturnType<typeof vi.fn>;
  executeViaAgentStream: (
    ...args: unknown[]
  ) => Promise<InterruptResult | StreamingResult>;
  doHandleExecute: (request: ExecuteRequest) => Promise<{
    status: string;
    result?: string;
  }>;
  chunkSeq: Map<string, number>;
  ownerCallbackUrl: Map<string, string>;
  pendingSessionFinish: Set<string>;
  callbackDelivery: CallbackDeliveryPrivates;
}

function makeAerServer(): AerServerPrivates {
  const server = Object.create(AERServer.prototype) as AerServerPrivates;
  server.runtime = {
    graphBuilder: {},
    orgId: null,
    getAgent: () => ({}),
  };
  server.sendStreamChunk = vi.fn().mockResolvedValue(undefined);
  server.reportCallback = vi.fn().mockResolvedValue(undefined);
  server.chunkSeq = new Map();
  server.ownerCallbackUrl = new Map();
  server.pendingSessionFinish = new Set();
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

describe("AER doHandleExecute session-finish latch", () => {
  test("no request: pendingSessionFinish stays empty on normal completion", async () => {
    const server = makeAerServer();
    const seen: (string | null)[] = [];
    server.executeViaAgentStream = async () => {
      seen.push(getCurrentTraceId());
      return { content: "42", messages: [] } as StreamingResult;
    };
    const traceId = "0123456789abcdef0123456789abcdef";
    await server.doHandleExecute({
      ...executeRequest("exec-1"),
      platform_trace_id: traceId,
    });
    expect(seen).toEqual([traceId]);
    expect(getCurrentTraceId()).toBeNull();
    expect(server.pendingSessionFinish.size).toBe(0);
  });

  test("requested + normal completion: execution id is latched", async () => {
    const server = makeAerServer();
    server.executeViaAgentStream = vi.fn(async () => {
      // Simulates the agent calling app.finishSession() mid-run: this runs
      // inside doHandleExecute's runWithExecutionContext frame.
      requestSessionFinish();
      return { content: "42", messages: [] } as StreamingResult;
    });

    await server.doHandleExecute(executeRequest("exec-2"));

    expect(server.pendingSessionFinish.has("exec-2")).toBe(true);
  });

  test("suspended run: finish request is dropped, not latched", async () => {
    const server = makeAerServer();
    server.executeViaAgentStream = vi.fn(async () => {
      requestSessionFinish();
      return {
        suspend_payload: { suspend_reason: "needs_approval" },
      } as InterruptResult;
    });

    const response = await server.doHandleExecute(executeRequest("exec-3"));

    expect(response.status).toBe("suspended");
    expect(server.pendingSessionFinish.size).toBe(0);
  });

  test("errored run: finish request is dropped, not latched", async () => {
    const server = makeAerServer();
    server.executeViaAgentStream = vi.fn(async () => {
      requestSessionFinish();
      throw new Error("agent blew up");
    });

    await expect(
      server.doHandleExecute(executeRequest("exec-4")),
    ).rejects.toMatchObject({ statusCode: 500 });

    expect(server.pendingSessionFinish.size).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Part B — the /execute route's post-response dispatch
// ---------------------------------------------------------------------------

function makeRouteServer(): AERServer {
  const server = Object.create(AERServer.prototype) as AERServer & {
    runtime: {
      memoryWriter?: {
        pendingWrites: number;
        drain: ReturnType<typeof vi.fn>;
      } | null;
    };
    pendingSessionFinish: Set<string>;
    callbackDelivery: CallbackDeliveryPrivates;
    handleExecute: ReturnType<typeof vi.fn>;
  };
  server.runtime = {};
  server.pendingSessionFinish = new Set();
  server.callbackDelivery =
    new CallbackDelivery() as unknown as CallbackDeliveryPrivates;
  return server;
}

describe("AER /execute route: post-response session finish", () => {
  test("no pending finish: no POST to /executions/*/finish", async () => {
    const server = makeRouteServer() as unknown as AERServer & {
      handleExecute: ReturnType<typeof vi.fn>;
    };
    server.handleExecute = vi
      .fn()
      .mockResolvedValue({ status: "completed", result: "ok" });
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response(null, { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const app = Fastify();
    server.registerRoutes(app);
    await app.inject({
      method: "POST",
      url: "/execute",
      payload: executeRequest("exec-5"),
    });
    await app.ready();

    expect(fetchMock).not.toHaveBeenCalled();
  });

  test("pending finish: exactly one finish POST, ordered strictly after the response is flushed", async () => {
    const server = makeRouteServer() as unknown as AERServer & {
      handleExecute: ReturnType<typeof vi.fn>;
      pendingSessionFinish: Set<string>;
    };
    server.pendingSessionFinish.add("exec-6");
    const callLog: string[] = [];
    server.handleExecute = vi.fn(async () => {
      callLog.push("handleExecute");
      return { status: "completed", result: "ok" };
    });

    const fetchMock = vi.fn(async (url: string) => {
      callLog.push(String(url).includes("/finish") ? "finish" : "other");
      return new Response(null, { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const app = Fastify();
    server.registerRoutes(app);

    const res = await app.inject({
      method: "POST",
      url: "/execute",
      payload: executeRequest("exec-6"),
    });
    expect(res.statusCode).toBe(200);

    // The finish POST is fired via a "finish" listener on reply.raw and is
    // not awaited by the route — wait for it instead of a fixed delay.
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalled());

    // handleExecute (which produces the response body) always precedes the
    // finish POST — proving the finish call is ordered after the response
    // was ready to flush, not raced against it.
    expect(callLog).toEqual(["handleExecute", "finish"]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toContain("/executions/exec-6/finish");
    // Drained so a later request for the same execution id can't double-fire.
    expect(server.pendingSessionFinish.has("exec-6")).toBe(false);
  });

  test("pending finish waits for retained terminal callback delivery", async () => {
    const server = makeRouteServer() as unknown as AERServer & {
      handleExecute: ReturnType<typeof vi.fn>;
      pendingSessionFinish: Set<string>;
      callbackDelivery: CallbackDeliveryPrivates;
    };
    const executionId = "exec-callback";
    const key = JSON.stringify(["terminal", executionId]);
    server.pendingSessionFinish.add(executionId);
    server.callbackDelivery.pending.set(key, {
      oeUrl: "http://oe",
      callback: { execution_id: executionId, status: "COMPLETED" },
    });

    let settleCallback!: () => void;
    server.callbackDelivery.tasks.set(
      key,
      new Promise<void>((resolve) => {
        settleCallback = () => {
          server.callbackDelivery.pending.delete(key);
          resolve();
        };
      }),
    );
    server.handleExecute = vi
      .fn()
      .mockResolvedValue({ status: "completed", result: "ok" });
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response(null, { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const app = Fastify();
    server.registerRoutes(app);
    const response = await app.inject({
      method: "POST",
      url: "/execute",
      payload: executeRequest(executionId),
    });

    expect(response.statusCode).toBe(200);
    expect(fetchMock).not.toHaveBeenCalled();

    settleCallback();
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(fetchMock.mock.calls[0][0]).toContain(
      `/executions/${executionId}/finish`,
    );
  });

  test("pending finish drains Memory before releasing the session", async () => {
    const server = makeRouteServer() as unknown as AERServer & {
      runtime: {
        memoryWriter: {
          pendingWrites: number;
          drain: ReturnType<typeof vi.fn>;
        };
      };
      handleExecute: ReturnType<typeof vi.fn>;
      pendingSessionFinish: Set<string>;
    };
    const executionId = "exec-memory-drain";
    server.pendingSessionFinish.add(executionId);
    let finishDrain!: (drained: boolean) => void;
    const drain = vi.fn(
      () =>
        new Promise<boolean>((resolve) => {
          finishDrain = resolve;
        }),
    );
    server.runtime.memoryWriter = { pendingWrites: 1, drain };
    server.handleExecute = vi
      .fn()
      .mockResolvedValue({ status: "completed", result: "ok" });
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response(null, { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const app = Fastify();
    server.registerRoutes(app);
    const response = await app.inject({
      method: "POST",
      url: "/execute",
      payload: executeRequest(executionId),
    });

    expect(response.statusCode).toBe(200);
    await vi.waitFor(() => expect(drain).toHaveBeenCalledOnce());
    expect(fetchMock).not.toHaveBeenCalled();

    finishDrain(true);
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    expect(fetchMock.mock.calls[0]?.[0]).toContain(
      `/executions/${executionId}/finish`,
    );
  });

  test("pending finish releases the session after the Memory drain times out", async () => {
    const server = makeRouteServer() as unknown as AERServer & {
      runtime: {
        memoryWriter: {
          pendingWrites: number;
          drain: ReturnType<typeof vi.fn>;
        };
      };
      handleExecute: ReturnType<typeof vi.fn>;
      pendingSessionFinish: Set<string>;
    };
    const executionId = "exec-memory-drain-timeout";
    server.pendingSessionFinish.add(executionId);
    const drain = vi.fn().mockResolvedValue(false);
    server.runtime.memoryWriter = { pendingWrites: 1, drain };
    server.handleExecute = vi
      .fn()
      .mockResolvedValue({ status: "completed", result: "ok" });
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response(null, { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const app = Fastify();
    server.registerRoutes(app);
    const response = await app.inject({
      method: "POST",
      url: "/execute",
      payload: executeRequest(executionId),
    });

    expect(response.statusCode).toBe(200);
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    expect(drain).toHaveBeenCalledWith(10_000);
    expect(fetchMock.mock.calls[0]?.[0]).toContain(
      `/executions/${executionId}/finish`,
    );
  });

  test("a failing finish POST is swallowed: does not affect the already-sent response", async () => {
    const server = makeRouteServer() as unknown as AERServer & {
      handleExecute: ReturnType<typeof vi.fn>;
      pendingSessionFinish: Set<string>;
    };
    server.pendingSessionFinish.add("exec-7");
    server.handleExecute = vi
      .fn()
      .mockResolvedValue({ status: "completed", result: "ok" });
    const fetchMock = vi.fn().mockRejectedValue(new Error("oe unreachable"));
    vi.stubGlobal("fetch", fetchMock);

    const app = Fastify();
    server.registerRoutes(app);

    const res = await app.inject({
      method: "POST",
      url: "/execute",
      payload: executeRequest("exec-7"),
    });

    expect(res.statusCode).toBe(200);
    expect(JSON.parse(res.payload)).toMatchObject({
      status: "completed",
      result: "ok",
    });

    // Wait for the finish POST instead of a fixed delay; the test fails via
    // an unhandled rejection if the error is not actually caught internally.
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalled());
  });
});
