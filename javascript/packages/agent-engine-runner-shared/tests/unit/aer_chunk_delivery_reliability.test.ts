/**
 * Tests for AER chunk-delivery reliability.
 *
 * Mirrors Python's tests/unit/test_aer_chunk_delivery_reliability.py.
 *
 * Two layers are exercised:
 *
 * Part A — `sendStreamChunk` retry semantics (`postWithRetries`):
 *   - Transient transport failures (network error, request-timeout abort) and
 *     5xx responses are retried with bounded exponential backoff.
 *   - Terminal-chunk (`done` / `error`) exhaustion re-raises so the caller can
 *     fall through to `reportCallback`.
 *   - Non-terminal-chunk (`text`) exhaustion logs at WARNING and returns; the
 *     run is not blocked.
 *   - Non-transient 4xx responses are not retried.
 *   - `chunkSeq` is incremented once per call and popped on success (terminal)
 *     or on terminal-failure re-raise.
 *
 * Part B — caller-side wrap in `doHandleExecute` around the terminal chunk send:
 *   - Happy path: DONE chunk succeeds, `reportCallback(COMPLETED, ...)` fires.
 *   - DONE chunk send raises after retries: caller catches, logs ERROR, and
 *     STILL fires `reportCallback(COMPLETED, result=executionOutcome.content)`
 *     so OE's `exec.Status` fallback can synthesize a `done` SSE chunk from the
 *     populated `exec.Result`.
 *   - An exception originating before the DONE chunk path (agent stream raising)
 *     still routes through the outer catch and reports status `"ERROR"`.
 *   - Timeout + ERROR chunk send failure: still reports `"ERROR"` with the
 *     timeout message and propagates 504 — the chunk-delivery error does not
 *     mask the timeout root cause.
 *
 * TS-vs-Python adaptations:
 *   - Python patches `client.post`; TS stubs the global `fetch` per attempt.
 *   - Python `caplog` → `vi.spyOn` on the shared log4js Logger prototype
 *     (`getLogger` hands out a fresh wrapper per call, but the level methods
 *     live on the prototype, so spying there intercepts the module's instance).
 *   - Python `asyncio.TimeoutError` injection → `vi.useFakeTimers()` advancing
 *     past the execution-timeout, with the agent stream rejecting on abort.
 *   - The artifact-metadata DONE test is intentionally omitted — TS does not yet
 *     port `_artifact_metadata_from_messages` (out of this PR's scope).
 */

import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import type {
  AgentInput,
  AgentOutput,
  ExecutionResult,
  RequestContext,
  StreamEvent,
} from "@mongodb-js/agent-engine-sdk";
import {
  AERServer,
  type ExecuteRequest,
  type InterruptResult,
  type StreamingResult,
} from "../../src/index.js";
import { DONE, ERROR, TEXT } from "../../src/server/chunk_types.js";
import { CallbackDelivery } from "../../src/server/callback_delivery.js";
import { runWithExecutionContext } from "../../src/context.js";
import { getLogger } from "../../src/logger.js";

const logger = getLogger("agent_engine_runner_shared.server.aer");
// log4js hands out a fresh Logger wrapper per getLogger() call, but the level
// methods live on the shared Logger prototype. Spying there intercepts the
// module's own instance too.
const loggerProto = Object.getPrototypeOf(logger) as typeof logger;

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

type CallbackBody = {
  execution_id: string;
  status: string;
  [key: string]: unknown;
};

interface CallbackDeliveryPrivates {
  pending: Map<string, { oeUrl: string; callback: CallbackBody }>;
  tasks: Map<string, Promise<void>>;
  retryController: AbortController;
  reserve: (
    key: string,
    oeUrl: string,
    callback: CallbackBody,
  ) => { oeUrl: string; callback: CallbackBody } | null;
  startRedelivery: (key: string, executionId: string) => void;
}

interface CallbackServerPrivates {
  runtime: {
    memoryWriter?: { shutdown: ReturnType<typeof vi.fn> } | null;
  };
  chunkSeq: Map<string, number>;
  ownerCallbackUrl: Map<string, string>;
  callbackDelivery: CallbackDeliveryPrivates;
  reportCallback: (
    oeUrl: string,
    executionId: string,
    status: string,
    fields?: Record<string, unknown>,
  ) => Promise<void>;
  onShutdown: () => Promise<void>;
}

function makeCallbackServer(): CallbackServerPrivates {
  const server = Object.create(AERServer.prototype) as CallbackServerPrivates;
  server.runtime = {};
  server.chunkSeq = new Map();
  server.ownerCallbackUrl = new Map();
  server.callbackDelivery =
    new CallbackDelivery() as unknown as CallbackDeliveryPrivates;
  return server;
}

function terminalKey(executionId: string): string {
  return JSON.stringify(["terminal", executionId]);
}

function suspensionKey(executionId: string, generation: number): string {
  return JSON.stringify(["suspension", executionId, generation]);
}

function startRetainedCallback(
  server: CallbackServerPrivates,
  oeUrl: string,
  callback: CallbackBody,
): void {
  const key =
    callback.status === "SUSPENDED"
      ? suspensionKey(
          callback.execution_id,
          Number(callback.suspend_generation),
        )
      : terminalKey(callback.execution_id);
  const pending = server.callbackDelivery.reserve(key, oeUrl, callback);
  if (pending !== null) {
    server.callbackDelivery.startRedelivery(key, callback.execution_id);
  }
}

describe("AER terminal callback delivery", () => {
  test("retries transient failures before succeeding", async () => {
    vi.useFakeTimers();
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("connection reset"))
      .mockResolvedValueOnce(new Response("unavailable", { status: 503 }))
      .mockResolvedValueOnce(new Response(null, { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const server = makeCallbackServer();
    const pending = server.reportCallback(
      "http://oe",
      "exec-retry",
      "COMPLETED",
      {
        result: "done",
      },
    );
    await vi.advanceTimersByTimeAsync(300);
    await pending;

    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  test.each([
    ["COMPLETED", { result: "done" }],
    ["ERROR", { error: "agent failed" }],
    ["SUSPENDED", { suspend_generation: 0, suspend_reason: "approval" }],
  ])(
    "retains %s callback after transient failure until OE acknowledges it",
    async (status, fields) => {
      vi.useFakeTimers();
      const fetchMock = scriptFetch([
        new TypeError("unavailable-1"),
        new TypeError("unavailable-2"),
        new TypeError("unavailable-3"),
        "ok",
      ]);
      vi.stubGlobal("fetch", fetchMock);
      const server = makeCallbackServer();

      const initialDelivery = server.reportCallback(
        "http://oe",
        "exec-redelivery",
        status,
        fields,
      );
      await vi.advanceTimersByTimeAsync(300);
      await initialDelivery;
      const key =
        status === "SUSPENDED"
          ? suspensionKey("exec-redelivery", 0)
          : terminalKey("exec-redelivery");
      const retryTask = server.callbackDelivery.tasks.get(key);
      expect(retryTask).toBeDefined();
      expect(server.callbackDelivery.pending.has(key)).toBe(true);

      await vi.advanceTimersByTimeAsync(100);
      await retryTask;

      expect(fetchMock).toHaveBeenCalledTimes(4);
      const bodies = fetchMock.mock.calls.map((call) =>
        JSON.parse(String((call[1] as RequestInit).body)),
      );
      expect(bodies.every((body) => body.status === status)).toBe(true);
      expect(
        bodies.every((body) => body.execution_id === "exec-redelivery"),
      ).toBe(true);
      expect(server.callbackDelivery.pending.size).toBe(0);
      expect(server.callbackDelivery.tasks.size).toBe(0);
    },
  );

  test("redelivers after a 5xx retry burst", async () => {
    vi.useFakeTimers();
    const fetchMock = scriptFetch([503, 503, 503, "ok"]);
    vi.stubGlobal("fetch", fetchMock);
    const server = makeCallbackServer();

    const initialDelivery = server.reportCallback(
      "http://oe",
      "exec-503",
      "COMPLETED",
      { result: "done" },
    );
    await vi.advanceTimersByTimeAsync(300);
    await initialDelivery;
    const retryTask = server.callbackDelivery.tasks.get(
      terminalKey("exec-503"),
    );
    expect(retryTask).toBeDefined();

    await vi.advanceTimersByTimeAsync(100);
    await retryTask;

    expect(fetchMock).toHaveBeenCalledTimes(4);
    expect(server.callbackDelivery.pending.size).toBe(0);
  });

  test("does not retain a generationless suspended callback after bounded retries", async () => {
    vi.useFakeTimers();
    const fetchMock = scriptFetch([
      new TypeError("unavailable-1"),
      new TypeError("unavailable-2"),
      new TypeError("unavailable-3"),
    ]);
    vi.stubGlobal("fetch", fetchMock);
    const server = makeCallbackServer();

    const delivery = server.reportCallback(
      "http://oe",
      "exec-suspended",
      "SUSPENDED",
      { suspend_reason: "approval" },
    );
    await vi.advanceTimersByTimeAsync(300);
    await delivery;

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(server.callbackDelivery.pending.size).toBe(0);
    expect(server.callbackDelivery.tasks.size).toBe(0);
  });

  test("in-flight suspension does not suppress a resumed terminal callback", async () => {
    let releaseSuspension!: (response: Response) => void;
    let markSuspensionStarted!: () => void;
    const suspensionStarted = new Promise<void>((resolve) => {
      markSuspensionStarted = resolve;
    });
    const suspendedResponse = new Promise<Response>((resolve) => {
      releaseSuspension = resolve;
    });
    const statuses: string[] = [];
    const fetchMock = vi.fn(async (_url: unknown, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body)) as CallbackBody;
      statuses.push(body.status);
      if (body.status === "SUSPENDED") {
        markSuspensionStarted();
        return suspendedResponse;
      }
      return new Response(null, { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);
    const server = makeCallbackServer();
    const suspended = server.reportCallback(
      "http://oe",
      "exec-resumed",
      "SUSPENDED",
      { suspend_generation: 0 },
    );
    await suspensionStarted;

    await server.reportCallback("http://oe", "exec-resumed", "COMPLETED");

    expect(statuses).toEqual(["SUSPENDED", "COMPLETED"]);
    releaseSuspension(new Response(null, { status: 200 }));
    await suspended;
  });

  test("resumed terminal callback stops generationless suspension retries", async () => {
    vi.useFakeTimers();
    let releaseSuspension!: () => void;
    let markSuspensionStarted!: () => void;
    let releaseTerminal!: () => void;
    let markTerminalStarted!: () => void;
    const suspensionStarted = new Promise<void>((resolve) => {
      markSuspensionStarted = resolve;
    });
    const suspensionResponse = new Promise<void>((resolve) => {
      releaseSuspension = resolve;
    });
    const terminalStarted = new Promise<void>((resolve) => {
      markTerminalStarted = resolve;
    });
    const terminalResponse = new Promise<void>((resolve) => {
      releaseTerminal = resolve;
    });
    const statuses: string[] = [];
    const fetchMock = vi.fn(async (_url: unknown, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body)) as CallbackBody;
      statuses.push(body.status);
      if (body.status === "SUSPENDED") {
        markSuspensionStarted();
        await suspensionResponse;
        throw new TypeError("suspension response lost");
      }
      markTerminalStarted();
      await terminalResponse;
      return new Response(null, { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);
    const server = makeCallbackServer();
    const suspended = server.reportCallback(
      "http://oe",
      "exec-stale-suspension",
      "SUSPENDED",
    );
    await suspensionStarted;

    const terminal = server.reportCallback(
      "http://oe",
      "exec-stale-suspension",
      "COMPLETED",
    );
    await terminalStarted;
    releaseSuspension();
    await vi.advanceTimersByTimeAsync(100);
    await suspended;

    expect(statuses).toEqual(["SUSPENDED", "COMPLETED"]);
    releaseTerminal();
    await terminal;
  });

  test("suspension generations and terminal redeliver independently", async () => {
    vi.useFakeTimers();
    const fetchMock = scriptFetch(["ok", "ok", "ok"]);
    vi.stubGlobal("fetch", fetchMock);
    const server = makeCallbackServer();
    startRetainedCallback(server, "http://oe", {
      execution_id: "exec-resumed",
      status: "SUSPENDED",
      suspend_generation: 0,
    });
    startRetainedCallback(server, "http://oe", {
      execution_id: "exec-resumed",
      status: "SUSPENDED",
      suspend_generation: 1,
    });
    startRetainedCallback(server, "http://oe", {
      execution_id: "exec-resumed",
      status: "COMPLETED",
    });
    const retryTasks = [...server.callbackDelivery.tasks.values()];

    await vi.advanceTimersByTimeAsync(100);
    await Promise.all(retryTasks);

    const delivered = fetchMock.mock.calls.map((call) => {
      const body = JSON.parse(
        String((call[1] as RequestInit).body),
      ) as CallbackBody;
      return [body.status, body.suspend_generation];
    });
    expect(delivered).toEqual(
      expect.arrayContaining([
        ["SUSPENDED", 0],
        ["SUSPENDED", 1],
        ["COMPLETED", undefined],
      ]),
    );
    expect(server.callbackDelivery.pending.size).toBe(0);
  });

  test("owner failure retains only the service callback target", async () => {
    vi.useFakeTimers();
    const fetchMock = scriptFetch([
      new TypeError("owner unavailable"),
      new TypeError("service unavailable-1"),
      new TypeError("service unavailable-2"),
      new TypeError("service unavailable-3"),
      "ok",
    ]);
    vi.stubGlobal("fetch", fetchMock);
    const server = makeCallbackServer();
    server.ownerCallbackUrl.set("exec-owner-redelivery", VALID_OWNER_OE);

    const initialDelivery = server.reportCallback(
      SERVICE_OE,
      "exec-owner-redelivery",
      "COMPLETED",
    );
    await vi.advanceTimersByTimeAsync(300);
    await initialDelivery;
    const retryTask = server.callbackDelivery.tasks.get(
      terminalKey("exec-owner-redelivery"),
    );
    expect(retryTask).toBeDefined();

    await vi.advanceTimersByTimeAsync(100);
    await retryTask;

    expect(fetchMock.mock.calls.map((call) => call[0])).toEqual([
      `${VALID_OWNER_OE}/executor/callback`,
      `${SERVICE_OE}/executor/callback`,
      `${SERVICE_OE}/executor/callback`,
      `${SERVICE_OE}/executor/callback`,
      `${SERVICE_OE}/executor/callback`,
    ]);
  });

  test("does not retain or retry a 4xx callback failure", async () => {
    const fetchMock = scriptFetch([400]);
    vi.stubGlobal("fetch", fetchMock);
    const server = makeCallbackServer();

    await server.reportCallback("http://oe", "exec-400", "ERROR", {
      error: "bad callback",
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(server.callbackDelivery.pending.size).toBe(0);
    expect(server.callbackDelivery.tasks.size).toBe(0);
  });

  test("redelivers distinct execution callbacks independently", async () => {
    vi.useFakeTimers();
    const fetchMock = scriptFetch(["ok", "ok"]);
    vi.stubGlobal("fetch", fetchMock);
    const server = makeCallbackServer();
    startRetainedCallback(server, "http://oe", {
      execution_id: "exec-a",
      status: "COMPLETED",
    });
    startRetainedCallback(server, "http://oe", {
      execution_id: "exec-b",
      status: "ERROR",
    });
    const retryTasks = [...server.callbackDelivery.tasks.values()];

    await vi.advanceTimersByTimeAsync(100);
    await Promise.all(retryTasks);

    const executionIds = fetchMock.mock.calls.map(
      (call) => JSON.parse(String((call[1] as RequestInit).body)).execution_id,
    );
    expect(new Set(executionIds)).toEqual(new Set(["exec-a", "exec-b"]));
    expect(server.callbackDelivery.pending.size).toBe(0);
  });

  test("equal duplicate preserves the retained callback until acknowledgement", async () => {
    vi.useFakeTimers();
    const fetchMock = scriptFetch(["ok"]);
    vi.stubGlobal("fetch", fetchMock);
    const server = makeCallbackServer();
    startRetainedCallback(server, "http://oe", {
      execution_id: "exec-duplicate",
      status: "COMPLETED",
    });
    const key = terminalKey("exec-duplicate");
    const retryTask = server.callbackDelivery.tasks.get(key);
    const retained = server.callbackDelivery.pending.get(key);

    startRetainedCallback(server, "http://oe", {
      execution_id: "exec-duplicate",
      status: "COMPLETED",
    });

    expect(server.callbackDelivery.pending.get(key)).toBe(retained);
    expect(server.callbackDelivery.tasks.get(key)).toBe(retryTask);
    await vi.advanceTimersByTimeAsync(100);
    await retryTask;
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(server.callbackDelivery.pending.size).toBe(0);
  });

  test("conflicting duplicate is rejected before delivery", async () => {
    vi.useFakeTimers();
    const fetchMock = scriptFetch(["ok"]);
    vi.stubGlobal("fetch", fetchMock);
    const server = makeCallbackServer();
    startRetainedCallback(server, "http://oe", {
      execution_id: "exec-preflight-conflict",
      status: "COMPLETED",
    });
    const key = terminalKey("exec-preflight-conflict");
    const retained = server.callbackDelivery.pending.get(key);
    const retryTask = server.callbackDelivery.tasks.get(key);

    await server.reportCallback(
      "http://oe",
      "exec-preflight-conflict",
      "ERROR",
    );

    expect(fetchMock).not.toHaveBeenCalled();
    expect(server.callbackDelivery.pending.get(key)).toBe(retained);
    expect(retained?.callback.status).toBe("COMPLETED");
    expect([...server.callbackDelivery.tasks.entries()]).toEqual([
      [key, retryTask],
    ]);
    await server.onShutdown();
  });

  test("shutdown cancels pending callback redelivery", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const server = makeCallbackServer();
    startRetainedCallback(server, "http://oe", {
      execution_id: "exec-shutdown",
      status: "COMPLETED",
    });

    await server.onShutdown();
    await vi.advanceTimersByTimeAsync(5_000);

    expect(fetchMock).not.toHaveBeenCalled();
    expect(server.callbackDelivery.pending.size).toBe(0);
    expect(server.callbackDelivery.tasks.size).toBe(0);
  });

  test("shutdown drains Memory before an app replacement completes", async () => {
    const server = makeCallbackServer();
    const shutdown = vi.fn().mockResolvedValue(undefined);
    server.runtime.memoryWriter = { shutdown };

    await server.onShutdown();

    expect(shutdown).toHaveBeenCalledOnce();
  });

  test("shutdown aborts an in-flight callback request without another attempt", async () => {
    vi.useFakeTimers();
    let requestSignal: AbortSignal | undefined;
    let rejectFirstRequest: ((reason?: unknown) => void) | undefined;
    let resolveRequestStarted!: () => void;
    const requestStarted = new Promise<void>((resolve) => {
      resolveRequestStarted = resolve;
    });
    const fetchMock = vi.fn((_url: unknown, init?: RequestInit) => {
      const signal = init?.signal;
      if (!(signal instanceof AbortSignal)) {
        return Promise.reject(new Error("callback request requires a signal"));
      }
      requestSignal = signal;
      resolveRequestStarted();
      if (fetchMock.mock.calls.length > 1) {
        return Promise.reject(
          new TypeError("callback retried after shutdown started"),
        );
      }
      return new Promise<Response>((_resolve, reject) => {
        rejectFirstRequest = reject;
        const rejectOnAbort = (): void => reject(signal.reason);
        if (signal.aborted) rejectOnAbort();
        else signal.addEventListener("abort", rejectOnAbort, { once: true });
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    const server = makeCallbackServer();
    startRetainedCallback(server, "http://oe", {
      execution_id: "exec-in-flight-shutdown",
      status: "COMPLETED",
    });

    await vi.advanceTimersByTimeAsync(100);
    await requestStarted;
    const shutdown = server.onShutdown();
    await Promise.resolve();
    try {
      expect(requestSignal?.aborted).toBe(true);
    } finally {
      rejectFirstRequest?.(new TypeError("test cleanup"));
      await vi.advanceTimersByTimeAsync(300);
      await shutdown;
    }

    await vi.advanceTimersByTimeAsync(5_000);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(server.callbackDelivery.pending.size).toBe(0);
    expect(server.callbackDelivery.tasks.size).toBe(0);
  });

  test.each([
    ["BigInt", 1n],
    ["NaN", Number.NaN],
    ["Infinity", Number.POSITIVE_INFINITY],
    ["undefined", undefined],
    ["function", () => undefined],
    ["symbol", Symbol("approval")],
    ["nested undefined", { decision: undefined }],
    ["array undefined", [undefined]],
    ["Map", new Map([["decision", "approve"]])],
    ["Set", new Set(["approve"])],
  ])(
    "surfaces lossy %s callback values to the execute error path",
    async (_description, value) => {
      const fetchMock = vi.fn();
      vi.stubGlobal("fetch", fetchMock);
      const reportCallback = (
        AERServer.prototype as unknown as {
          reportCallback: (
            oeUrl: string,
            executionId: string,
            status: string,
            fields?: Record<string, unknown>,
          ) => Promise<void>;
        }
      ).reportCallback;

      await expect(
        reportCallback.call({}, "http://oe", "exec-bad-callback", "SUSPENDED", {
          interrupts: [{ id: "approval", value }],
          resume_schema: { type: "object" },
        }),
      ).rejects.toThrow(
        'Interrupt "approval" has a non-JSON-serializable value',
      );
      expect(fetchMock).not.toHaveBeenCalled();
    },
  );
});

// ---------------------------------------------------------------------------
// Part A harness — call the REAL sendStreamChunk with a scripted global fetch
// ---------------------------------------------------------------------------

interface RealSendServer {
  chunkSeq: Map<string, number>;
  ownerCallbackUrl: Map<string, string>;
  callbackDelivery: CallbackDeliveryPrivates;
  sendStreamChunk: (
    oeUrl: string,
    executionId: string,
    chunkType: string,
    content?: string,
    error?: string,
    metadata?: Record<string, string>,
  ) => Promise<void>;
  reportCallback: (
    oeUrl: string,
    executionId: string,
    status: string,
    fields?: Record<string, unknown>,
  ) => Promise<void>;
}

function makeRealServer(): RealSendServer {
  // Object.create skips the constructor; sendStreamChunk resolves to the real
  // prototype method. Seed the callback state that the ctor would have.
  const server = Object.create(AERServer.prototype) as RealSendServer;
  server.chunkSeq = new Map();
  server.ownerCallbackUrl = new Map();
  server.callbackDelivery =
    new CallbackDelivery() as unknown as CallbackDeliveryPrivates;
  return server;
}

/** A scripted `fetch`: each entry is "ok", an HTTP status code, or an Error to throw. */
type FetchScriptEntry = "ok" | number | Error;

function scriptFetch(entries: FetchScriptEntry[]): ReturnType<typeof vi.fn> {
  let i = 0;
  return vi.fn(async () => {
    if (i >= entries.length) {
      throw new Error(
        `fetch called ${i + 1} times but script only has ${entries.length} entries`,
      );
    }
    const entry = entries[i++];
    if (entry instanceof Error) throw entry;
    const status = entry === "ok" ? 200 : entry;
    return new Response(null, { status });
  });
}

// ---------------------------------------------------------------------------
// Part A — sendStreamChunk retry semantics
// ---------------------------------------------------------------------------

describe("AER sendStreamChunk retry semantics", () => {
  test("transient transport error then success returns normally", async () => {
    const server = makeRealServer();
    const fetchMock = scriptFetch([new TypeError("network"), "ok"]);
    vi.stubGlobal("fetch", fetchMock);

    await server.sendStreamChunk("http://oe", "exec-1", TEXT, "hello");

    expect(fetchMock.mock.calls).toHaveLength(2);
    // Counter incremented exactly once; non-terminal chunks do not pop.
    expect(server.chunkSeq.get("exec-1")).toBe(1);
  });

  test("transient 503 is retried and succeeds on retry", async () => {
    const server = makeRealServer();
    const fetchMock = scriptFetch([503, "ok"]);
    vi.stubGlobal("fetch", fetchMock);

    await server.sendStreamChunk("http://oe", "exec-2", TEXT, "hi");

    expect(fetchMock.mock.calls).toHaveLength(2);
    expect(server.chunkSeq.get("exec-2")).toBe(1);
  });

  test("terminal chunk exhaustion re-raises to the caller and pops chunkSeq", async () => {
    const server = makeRealServer();
    const fetchMock = scriptFetch([
      new TypeError("t1"),
      new TypeError("t2"),
      new TypeError("t3"),
    ]);
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      server.sendStreamChunk("http://oe", "exec-3", DONE, "final answer"),
    ).rejects.toThrow();

    expect(fetchMock.mock.calls).toHaveLength(3);
    // Terminal failure must pop chunkSeq before re-raising.
    expect(server.chunkSeq.has("exec-3")).toBe(false);
  });

  test("non-terminal chunk exhaustion logs a warning and returns", async () => {
    const server = makeRealServer();
    const fetchMock = scriptFetch([
      new TypeError("t1"),
      new TypeError("t2"),
      new TypeError("t3"),
    ]);
    vi.stubGlobal("fetch", fetchMock);
    const warnSpy = vi.spyOn(loggerProto, "warn");

    // Should NOT throw — non-terminal degrades gracefully.
    await server.sendStreamChunk("http://oe", "exec-4", TEXT, "dropped");

    expect(fetchMock.mock.calls).toHaveLength(3);
    // WARNING-level (not DEBUG) so failures are visible in prod logs.
    expect(warnSpy).toHaveBeenCalled();
    expect(warnSpy.mock.calls.some((c) => String(c[0]).includes(TEXT))).toBe(
      true,
    );
    // Non-terminal chunks never pop chunkSeq.
    expect(server.chunkSeq.get("exec-4")).toBe(1);
  });

  test("non-transient 4xx is not retried", async () => {
    // Terminal 4xx re-raises after a single attempt.
    const server = makeRealServer();
    const fetchMock = scriptFetch([400]);
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      server.sendStreamChunk("http://oe", "exec-5a", DONE, "x"),
    ).rejects.toThrow();
    expect(fetchMock.mock.calls).toHaveLength(1); // no retry on 4xx

    // Non-terminal 4xx logs WARNING and returns, also without retry.
    const server2 = makeRealServer();
    const fetchMock2 = scriptFetch([400]);
    vi.stubGlobal("fetch", fetchMock2);

    await server2.sendStreamChunk("http://oe", "exec-5b", TEXT, "x");
    expect(fetchMock2.mock.calls).toHaveLength(1); // no retry on 4xx
  });

  test("chunkSeq increments once per call and pops on success and terminal failure", async () => {
    const server = makeRealServer();

    // First call: non-terminal SUCCESS — seq 0 -> 1, does NOT pop.
    vi.stubGlobal("fetch", scriptFetch(["ok"]));
    await server.sendStreamChunk("http://oe", "exec-6", TEXT, "a");
    expect(server.chunkSeq.get("exec-6")).toBe(1);

    // Second call: terminal SUCCESS — seq 1 -> 2, then pops.
    vi.stubGlobal("fetch", scriptFetch(["ok"]));
    await server.sendStreamChunk("http://oe", "exec-6", DONE, "done");
    expect(server.chunkSeq.has("exec-6")).toBe(false);

    // Third call (new execution): terminal FAILURE — incremented but popped
    // before re-raise.
    vi.stubGlobal(
      "fetch",
      scriptFetch([
        new TypeError("t1"),
        new TypeError("t2"),
        new TypeError("t3"),
      ]),
    );
    await expect(
      server.sendStreamChunk("http://oe", "exec-7", DONE, "x"),
    ).rejects.toThrow();
    expect(server.chunkSeq.has("exec-7")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Part B harness — drive doHandleExecute with the terminal chunk send stubbed
// ---------------------------------------------------------------------------

class FakeRunResult implements ExecutionResult {
  constructor(private readonly _events: StreamEvent[]) {}
  then<T1 = AgentOutput, T2 = never>(
    onfulfilled?: ((value: AgentOutput) => T1 | PromiseLike<T1>) | null,
    onrejected?: ((reason: unknown) => T2 | PromiseLike<T2>) | null,
  ): PromiseLike<T1 | T2> {
    return Promise.resolve({ response: "done" } as AgentOutput).then(
      onfulfilled,
      onrejected,
    );
  }
  [Symbol.asyncIterator](): AsyncIterator<StreamEvent> {
    const events = this._events;
    let i = 0;
    return {
      async next(): Promise<IteratorResult<StreamEvent>> {
        if (i < events.length) return { value: events[i++], done: false };
        return { value: undefined as unknown as StreamEvent, done: true };
      },
    };
  }
}

class FakeAgent {
  constructor(private readonly events: StreamEvent[]) {}
  execute(_ctx: RequestContext, _input: AgentInput): ExecutionResult {
    return new FakeRunResult(this.events);
  }
}

interface AerServerPrivates {
  runtime: {
    graphBuilder?: unknown;
    orgId?: string | null;
    getAgent?: (opts?: unknown) => FakeAgent;
  };
  sendStreamChunk: ReturnType<typeof vi.fn>;
  reportCallback: ReturnType<typeof vi.fn>;
  executeViaAgentStream: (
    agent: FakeAgent,
    ctx: RequestContext,
    agentInput: AgentInput,
    oeUrl: string,
    executionId: string,
  ) => Promise<InterruptResult | StreamingResult>;
  doHandleExecute: (request: ExecuteRequest) => Promise<{
    status: string;
    result?: string;
  }>;
  chunkSeq: Map<string, number>;
  ownerCallbackUrl: Map<string, string>;
}

function makeAerServer(): AerServerPrivates {
  const server = Object.create(AERServer.prototype) as AerServerPrivates;
  server.runtime = {
    graphBuilder: {},
    orgId: null,
    getAgent: () => new FakeAgent([]),
  };
  server.sendStreamChunk = vi.fn().mockResolvedValue(undefined);
  server.reportCallback = vi.fn().mockResolvedValue(undefined);
  server.chunkSeq = new Map();
  server.ownerCallbackUrl = new Map();
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

describe("AER doHandleExecute terminal-chunk delivery wrap", () => {
  test("happy path: DONE chunk succeeds then COMPLETED callback fires", async () => {
    const server = makeAerServer();
    server.executeViaAgentStream = vi
      .fn()
      .mockResolvedValue({ content: "42", messages: [] } as StreamingResult);

    const response = await server.doHandleExecute(executeRequest("exec-b1"));

    // DONE chunk was attempted.
    const doneCalls = server.sendStreamChunk.mock.calls.filter(
      (c) => c[2] === DONE,
    );
    expect(doneCalls).toHaveLength(1);
    // COMPLETED callback fired with the unmodified content.
    expect(server.reportCallback).toHaveBeenCalledTimes(1);
    const [, , status, fields] = server.reportCallback.mock.calls[0];
    expect(status).toBe("COMPLETED");
    expect(fields).toMatchObject({
      result: "42",
    });
    expect(response.status).toBe("completed");
    expect(response.result).toBe("42");
  });

  test("DONE chunk delivery failure still fires COMPLETED with original content", async () => {
    const server = makeAerServer();
    server.executeViaAgentStream = vi.fn().mockResolvedValue({
      content: "the actual answer",
      messages: [],
    } as StreamingResult);
    // Terminal DONE send re-raises after exhausting retries.
    server.sendStreamChunk = vi.fn(async (..._args: unknown[]) => {
      if (_args[2] === DONE) throw new Error("oe unreachable");
    });
    const errorSpy = vi.spyOn(loggerProto, "error");

    const response = await server.doHandleExecute(executeRequest("exec-b2"));

    // COMPLETED fired with the ORIGINAL content, not the exception string and
    // not status="ERROR".
    expect(server.reportCallback).toHaveBeenCalledTimes(1);
    const [, , status, fields] = server.reportCallback.mock.calls[0];
    expect(status).toBe("COMPLETED");
    expect(fields).toMatchObject({
      result: "the actual answer",
    });
    // Terminal chunk delivery failure is logged at ERROR.
    expect(errorSpy).toHaveBeenCalled();
    expect(response.status).toBe("completed");
    expect(response.result).toBe("the actual answer");
  });

  test("an exception before the DONE path routes through the outer ERROR callback", async () => {
    const server = makeAerServer();
    server.executeViaAgentStream = vi
      .fn()
      .mockRejectedValue(new Error("agent blew up before result"));

    await expect(
      server.doHandleExecute(executeRequest("exec-b3")),
    ).rejects.toMatchObject({ statusCode: 500 });

    // Outer error path fired exactly one callback with status="ERROR".
    expect(server.reportCallback).toHaveBeenCalledTimes(1);
    const [, , status, fields] = server.reportCallback.mock.calls[0];
    expect(status).toBe("ERROR");
    expect(fields).toMatchObject({
      error: "agent blew up before result",
    });
  });

  test("timeout + ERROR chunk delivery failure still reports ERROR and propagates 504", async () => {
    const server = makeAerServer();
    // Hanging agent that only unwinds when the timeout aborts it.
    server.executeViaAgentStream = vi.fn((_agent, ctx: RequestContext) => {
      return new Promise<never>((_resolve, reject) => {
        ctx.signal?.addEventListener(
          "abort",
          () => {
            const err = new Error("aborted") as Error & { name: string };
            err.name = "AbortError";
            reject(err);
          },
          { once: true },
        );
      });
    }) as unknown as typeof server.executeViaAgentStream;
    // Terminal ERROR send re-raises after exhausting retries.
    server.sendStreamChunk = vi.fn(async (..._args: unknown[]) => {
      if (_args[2] === ERROR) throw new Error("oe unreachable");
    });
    const errorSpy = vi.spyOn(loggerProto, "error");

    const timeoutMs =
      (Number(process.env["RUNNER_EXECUTION_TIMEOUT"]) || 600) * 1000;

    vi.useFakeTimers();
    let outcome: { ok: true } | { ok: false; error: unknown };
    try {
      const p = server.doHandleExecute(executeRequest("exec-b4"));
      const settled = p.then(
        () => ({ ok: true as const }),
        (e: unknown) => ({ ok: false as const, error: e }),
      );
      await vi.advanceTimersByTimeAsync(timeoutMs + 1000);
      outcome = await settled;
    } finally {
      vi.useRealTimers();
    }

    // Propagates 504 with the timeout root cause, not 500 masked by the
    // chunk-delivery error.
    expect(outcome.ok).toBe(false);
    if (!outcome.ok) {
      const err = outcome.error as Error & { statusCode?: number };
      expect(err.statusCode).toBe(504);
      expect(err.message).toContain("Execution timed out");
    }

    // Exactly one ERROR callback carrying the timeout message.
    expect(server.reportCallback).toHaveBeenCalledTimes(1);
    const [, , status, fields] = server.reportCallback.mock.calls[0];
    expect(status).toBe("ERROR");
    expect((fields as { error: string }).error).toContain(
      "Execution timed out",
    );
    // The chunk-delivery failure was logged at ERROR for ops visibility.
    expect(errorSpy).toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// Part C — owner-callback preference
//
// The AER sends stream chunks and the terminal callback to the owning OE
// replica when the request carried a valid owner URL, and falls back to the
// trusted service URL on any owner failure. Validation of the owner URL
// (server/owner_url.ts) means a forged value is never contacted.
// ---------------------------------------------------------------------------

const SERVICE_OE = "http://oe.ns.svc.cluster.local:8000";
const VALID_OWNER_OE = "http://10-1-2-3.oe-headless.ns.svc.cluster.local:8000";

/** Record the URL and parsed body of every POST; always returns 200. */
function urlRecordingFetch(): {
  mock: ReturnType<typeof vi.fn>;
  calls: Array<{ url: string; body: unknown }>;
} {
  const calls: Array<{ url: string; body: unknown }> = [];
  const mock = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, body: JSON.parse((init?.body as string) ?? "null") });
    return new Response(null, { status: 200 });
  });
  return { mock, calls };
}

describe("AER sendStreamChunk / reportCallback owner routing", () => {
  test("sendStreamChunk routes to the owner URL when one is registered", async () => {
    const server = makeRealServer();
    server.ownerCallbackUrl.set("exec-o1", VALID_OWNER_OE);
    const { mock, calls } = urlRecordingFetch();
    vi.stubGlobal("fetch", mock);

    await server.sendStreamChunk(SERVICE_OE, "exec-o1", TEXT, "hi");

    expect(calls.map((c) => c.url)).toEqual([`${VALID_OWNER_OE}/stream/chunk`]);
  });

  test("sendStreamChunk falls back to the service URL on an owner transport error", async () => {
    vi.useFakeTimers();
    try {
      const server = makeRealServer();
      server.ownerCallbackUrl.set("exec-o2", VALID_OWNER_OE);
      const calls: Array<{ url: string; body: string }> = [];
      const mock = vi.fn(async (url: string, init?: RequestInit) => {
        calls.push({ url, body: init?.body as string });
        if (url.startsWith(VALID_OWNER_OE)) throw new TypeError("connreset");
        return new Response(null, { status: 200 });
      });
      vi.stubGlobal("fetch", mock);

      const pending = server.sendStreamChunk(SERVICE_OE, "exec-o2", DONE, "x");
      await vi.runAllTimersAsync();
      await pending;

      expect(calls.map((c) => c.url)).toEqual([
        `${VALID_OWNER_OE}/stream/chunk`,
        `${SERVICE_OE}/stream/chunk`,
      ]);
      // Identical chunk body delivered to the fallback target.
      expect(calls[0].body).toBe(calls[1].body);
      expect(server.ownerCallbackUrl.size).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  test("sendStreamChunk stops retrying the owner after the first owner failure", async () => {
    vi.useFakeTimers();
    try {
      const server = makeRealServer();
      server.ownerCallbackUrl.set("exec-o2b", VALID_OWNER_OE);
      const calls: Array<{ url: string; body: string }> = [];
      const mock = vi.fn(async (url: string, init?: RequestInit) => {
        calls.push({ url, body: init?.body as string });
        if (url.startsWith(VALID_OWNER_OE)) throw new TypeError("connreset");
        return new Response(null, { status: 200 });
      });
      vi.stubGlobal("fetch", mock);

      const first = server.sendStreamChunk(SERVICE_OE, "exec-o2b", TEXT, "x");
      await vi.runAllTimersAsync();
      await first;

      const second = server.sendStreamChunk(SERVICE_OE, "exec-o2b", TEXT, "y");
      await vi.runAllTimersAsync();
      await second;

      expect(calls.map((c) => c.url)).toEqual([
        `${VALID_OWNER_OE}/stream/chunk`,
        `${SERVICE_OE}/stream/chunk`,
        `${SERVICE_OE}/stream/chunk`,
      ]);
      expect(server.ownerCallbackUrl.size).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  test("a stream owner failure suppresses later callback owner attempts", async () => {
    const server = makeRealServer();
    const executionId = "exec-o-cross-transport";
    server.ownerCallbackUrl.set(executionId, VALID_OWNER_OE);
    const calls: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        calls.push(url);
        if (url === `${VALID_OWNER_OE}/stream/chunk`) {
          throw new TypeError("connreset");
        }
        return new Response(null, { status: 200 });
      }),
    );

    await runWithExecutionContext(
      {
        executionId,
        wrapper: null,
        oeUrl: SERVICE_OE,
        oeOwnerUrl: VALID_OWNER_OE,
      },
      async () => {
        await server.sendStreamChunk(SERVICE_OE, executionId, TEXT, "x");
        await server.reportCallback(SERVICE_OE, executionId, "COMPLETED", {
          result: "done",
        });
      },
    );

    expect(calls).toEqual([
      `${VALID_OWNER_OE}/stream/chunk`,
      `${SERVICE_OE}/stream/chunk`,
      `${SERVICE_OE}/executor/callback`,
    ]);
  });

  test("sendStreamChunk falls back to the service URL on an owner HTTP error", async () => {
    const ownerStatus = 500;
    vi.useFakeTimers();
    try {
      const server = makeRealServer();
      server.ownerCallbackUrl.set("exec-oh", VALID_OWNER_OE);
      const calls: Array<{
        url: string;
        redirect: RequestRedirect | undefined;
      }> = [];
      const mock = vi.fn(async (url: string, init?: RequestInit) => {
        calls.push({ url, redirect: init?.redirect });
        if (url.startsWith(VALID_OWNER_OE))
          return new Response(null, { status: ownerStatus });
        return new Response(null, { status: 200 });
      });
      vi.stubGlobal("fetch", mock);

      const pending = server.sendStreamChunk(SERVICE_OE, "exec-oh", DONE, "x");
      await vi.runAllTimersAsync();
      await pending;

      expect(calls.map((c) => c.url)).toEqual([
        `${VALID_OWNER_OE}/stream/chunk`,
        `${SERVICE_OE}/stream/chunk`,
      ]);
      expect(
        calls.filter((c) => c.url.startsWith(VALID_OWNER_OE)),
      ).toHaveLength(1);
      expect(calls[0]?.redirect).toBe("manual");
      expect(calls[1]?.redirect).toBeUndefined();
    } finally {
      vi.useRealTimers();
    }
  });

  test("reportCallback routes to the owner /executor/callback when registered", async () => {
    const server = makeRealServer();
    server.ownerCallbackUrl.set("exec-o3", VALID_OWNER_OE);
    const { mock, calls } = urlRecordingFetch();
    vi.stubGlobal("fetch", mock);

    await server.reportCallback(SERVICE_OE, "exec-o3", "COMPLETED", {
      result: "done",
    });

    expect(calls.map((c) => c.url)).toEqual([
      `${VALID_OWNER_OE}/executor/callback`,
    ]);
  });

  test("sendStreamChunk honors a capped delta-seconds Retry-After on a service 503", async () => {
    vi.useFakeTimers();
    try {
      // No owner registered: a service 503 with Retry-After 100s waits the
      // capped 10s (>> the 100ms exponential backoff) before retrying.
      const server = makeRealServer();
      const fetchMock = vi
        .fn()
        .mockResolvedValueOnce(
          new Response(null, {
            status: 503,
            headers: { "Retry-After": "100" },
          }),
        )
        .mockResolvedValueOnce(new Response(null, { status: 200 }));
      vi.stubGlobal("fetch", fetchMock);

      const pending = server.sendStreamChunk(SERVICE_OE, "exec-ra", TEXT, "x");
      await vi.advanceTimersByTimeAsync(500);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      await vi.advanceTimersByTimeAsync(10_000);
      await pending;
      expect(fetchMock).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });
});

// ---------------------------------------------------------------------------
// Part C — handler-level owner routing through the real doHandleExecute
// ---------------------------------------------------------------------------

interface HandlerServer {
  runtime: {
    graphBuilder?: unknown;
    orgId?: string | null;
    getAgent?: (opts?: unknown) => FakeAgent;
  };
  chunkSeq: Map<string, number>;
  ownerCallbackUrl: Map<string, string>;
  callbackDelivery: CallbackDeliveryPrivates;
  executeViaAgentStream: ReturnType<typeof vi.fn>;
  doHandleExecute: (request: ExecuteRequest) => Promise<unknown>;
}

/** A double whose sendStreamChunk / reportCallback are the REAL prototype methods. */
function makeHandlerServer(): HandlerServer {
  const server = Object.create(AERServer.prototype) as HandlerServer;
  server.runtime = {
    graphBuilder: {},
    orgId: null,
    getAgent: () => new FakeAgent([]),
  };
  server.chunkSeq = new Map();
  server.ownerCallbackUrl = new Map();
  server.callbackDelivery =
    new CallbackDelivery() as unknown as CallbackDeliveryPrivates;
  server.executeViaAgentStream = vi
    .fn()
    .mockResolvedValue({ content: "42", messages: [] } as StreamingResult);
  return server;
}

function ownerExecuteRequest(
  executionId: string,
  ownerUrl?: string,
): ExecuteRequest {
  return {
    execution_id: executionId,
    message: "hi",
    platform_api_url: SERVICE_OE,
    platform_api_owner_url: ownerUrl,
    resume: false,
    user_id: "u-1",
  } as ExecuteRequest;
}

describe("AER doHandleExecute owner routing", () => {
  beforeEach(() => {
    // Unset OE_URL so resolveOeUrl keeps the request's platform_api_url; the
    // owner is then validated against that trusted value deterministically.
    vi.stubEnv("OE_URL", "");
  });
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  test("a valid owner carries both the DONE chunk and the terminal callback; registration is cleaned up", async () => {
    const server = makeHandlerServer();
    const { mock, calls } = urlRecordingFetch();
    vi.stubGlobal("fetch", mock);

    await server.doHandleExecute(
      ownerExecuteRequest("exec-h1", VALID_OWNER_OE),
    );

    expect(calls.map((c) => c.url)).toEqual([
      `${VALID_OWNER_OE}/stream/chunk`,
      `${VALID_OWNER_OE}/executor/callback`,
    ]);
    // No service traffic on those endpoints.
    expect(calls.some((c) => c.url.startsWith(SERVICE_OE))).toBe(false);
    // Popped in the finally.
    expect(server.ownerCallbackUrl.size).toBe(0);
  });

  test("when the request omits platform_api_owner_url, callbacks stay on the stable service URL", async () => {
    const server = makeHandlerServer();
    const { mock, calls } = urlRecordingFetch();
    vi.stubGlobal("fetch", mock);

    await server.doHandleExecute(ownerExecuteRequest("exec-h-miss"));

    expect(calls.map((c) => c.url)).toEqual([
      `${SERVICE_OE}/stream/chunk`,
      `${SERVICE_OE}/executor/callback`,
    ]);
    expect(server.ownerCallbackUrl.size).toBe(0);
  });

  test.each([
    ["external host", "http://attacker.example:8000"],
    ["wrong port", "http://10-1-2-3.oe-headless.ns.svc.cluster.local:9999"],
    ["wrong scheme", "https://10-1-2-3.oe-headless.ns.svc.cluster.local:8000"],
    ["non-headless shape", "http://10-1-2-3.oe.ns.svc.cluster.local:8000"],
  ])(
    "security-negative: a forged owner URL (%s) is ignored and never contacted",
    async (_desc, forged) => {
      const server = makeHandlerServer();
      const { mock, calls } = urlRecordingFetch();
      vi.stubGlobal("fetch", mock);

      await server.doHandleExecute(ownerExecuteRequest("exec-h2", forged));

      // Both endpoints went to the trusted service URL, nothing else.
      expect(calls.map((c) => c.url)).toEqual([
        `${SERVICE_OE}/stream/chunk`,
        `${SERVICE_OE}/executor/callback`,
      ]);
      // The forged host is never contacted.
      const forgedHost = new URL(forged).hostname;
      expect(calls.some((c) => c.url.includes(forgedHost))).toBe(false);
      expect(server.ownerCallbackUrl.size).toBe(0);
    },
  );

  test("owner registration is cleaned up when the agent stream errors", async () => {
    const server = makeHandlerServer();
    server.executeViaAgentStream = vi
      .fn()
      .mockRejectedValue(new Error("agent blew up"));
    const { mock } = urlRecordingFetch();
    vi.stubGlobal("fetch", mock);

    await expect(
      server.doHandleExecute(ownerExecuteRequest("exec-h3", VALID_OWNER_OE)),
    ).rejects.toMatchObject({ statusCode: 500 });

    expect(server.ownerCallbackUrl.size).toBe(0);
  });
});
