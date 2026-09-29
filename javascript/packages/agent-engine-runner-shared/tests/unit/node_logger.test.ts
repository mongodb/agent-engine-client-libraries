/**
 * Unit tests for NodeExecutionLogger (framework-neutral).
 *
 * Mirrors Python's tests/unit/test_node_logger.py.
 *
 * TS-vs-Python adaptations:
 *   - Python `patch("agent_engine_runner_shared.node_logger.httpx.Client")` → TS mocks
 *     the `fetch` global via `vi.stubGlobal`.
 *   - Python `on_node_*` methods are sync (httpx is sync). TS is async
 *     fire-and-forget: `void this._sendNodeEvent(...)`. The fetch call
 *     itself happens synchronously inside `_sendNodeEvent` before the
 *     first await, so `fetchSpy.mock.calls` is populated by the time
 *     the public method returns. Cleanup (deleting `nodeStartTimes`
 *     entries) is also sync — it happens before the await. So tests
 *     don't need to flush microtasks for these assertions.
 *   - Python uses `isinstance(logger, BaseExecutionCallback)` via Protocol
 *     class. TS `BaseExecutionCallback` is a compile-time interface; the
 *     runtime equivalent is `instanceof NullExecutionCallback` (the base
 *     class that implements the protocol).
 *   - Python `_serialize_for_json` handles Pydantic models via `.dict()`.
 *     TS handles objects with `toJSON()` methods — same intent, different
 *     duck-typing mechanism.
 *   - Method signature: Python uses keyword args (`run_id=...`); TS uses
 *     an opts object (`{ runId, parentRunId?, ... }`).
 *   - `serializeForJson` accessed via cast-through-unknown.
 */

import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import { NullExecutionCallback } from "@mongodb-js/agent-engine-sdk";
import { NodeExecutionLogger } from "../../src/index.js";

// ---------------------------------------------------------------------------
// fetch helpers
// ---------------------------------------------------------------------------

interface MockFetchResponse {
  ok: boolean;
  status: number;
  statusText: string;
  json: () => Promise<unknown>;
}

function emptyResponse(): MockFetchResponse {
  return { ok: true, status: 200, statusText: "OK", json: async () => ({}) };
}

function makeLogger(
  opts: Partial<ConstructorParameters<typeof NodeExecutionLogger>[0]> = {},
): NodeExecutionLogger {
  return new NodeExecutionLogger({
    oeUrl: "http://oe:8080",
    executionId: "exec-1",
    ...opts,
  });
}

interface CapturedCall {
  url: string;
  payload: Record<string, unknown>;
}

/** Wrap a fetch spy and decode the JSON payload from its calls. */
function captureCalls(fetchSpy: ReturnType<typeof vi.fn>): CapturedCall[] {
  return fetchSpy.mock.calls.map((call) => {
    const [url, init] = call;
    const body = (init as RequestInit).body as string;
    return { url: url as string, payload: JSON.parse(body) };
  });
}

beforeEach(() => {
  vi.unstubAllGlobals();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------
// Protocol compliance
// ---------------------------------------------------------------------------

describe("NodeExecutionLogger — BaseExecutionCallback compliance", () => {
  test("extends NullExecutionCallback (TS analog of isinstance check)", () => {
    // test_satisfies_base_execution_callback
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(emptyResponse()));

    const logger = makeLogger();

    expect(logger).toBeInstanceOf(NullExecutionCallback);
  });
});

// ---------------------------------------------------------------------------
// onNodeStart
// ---------------------------------------------------------------------------

describe("NodeExecutionLogger.onNodeStart", () => {
  test("sends POST to /node/execution with started status and full payload", () => {
    // test_sends_post_with_correct_payload
    const fetchSpy = vi.fn().mockResolvedValue(emptyResponse());
    vi.stubGlobal("fetch", fetchSpy);

    const logger = makeLogger({
      sessionId: "sess-1",
      threadId: "thread-1",
      userId: "user-1",
      orgId: "org-1",
      projectId: "group-1",
    });

    logger.onNodeStart(
      "agent_node",
      { messages: ["hello"] },
      { runId: "run-123" },
    );

    expect(fetchSpy).toHaveBeenCalledOnce();
    const [{ url, payload }] = captureCalls(fetchSpy);
    expect(url).toBe("http://oe:8080/node/execution");
    expect(payload["execution_id"]).toBe("exec-1");
    expect(payload["node_name"]).toBe("agent_node");
    expect(payload["status"]).toBe("started");
    expect(payload["run_id"]).toBe("run-123");
    expect(payload["session_id"]).toBe("sess-1");
    expect(payload["user_id"]).toBe("user-1");
    expect(payload["org_id"]).toBe("org-1");
    expect(payload["project_id"]).toBe("group-1");
    expect(payload).toHaveProperty("inputs");
  });

  test("records start time keyed by runId", () => {
    // test_records_start_time
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(emptyResponse()));
    const logger = makeLogger();

    logger.onNodeStart("agent_node", {}, { runId: "run-123" });

    const map = (logger as unknown as { nodeStartTimes: Map<string, Date> })
      .nodeStartTimes;
    expect(map.has("run-123")).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// onNodeEnd
// ---------------------------------------------------------------------------

describe("NodeExecutionLogger.onNodeEnd", () => {
  test("calculates duration_ms from recorded start time and cleans up", () => {
    // test_calculates_duration_from_start
    const fetchSpy = vi.fn().mockResolvedValue(emptyResponse());
    vi.stubGlobal("fetch", fetchSpy);

    const logger = makeLogger();
    const map = (logger as unknown as { nodeStartTimes: Map<string, Date> })
      .nodeStartTimes;
    map.set("run-123", new Date(Date.now() - 1)); // 1ms ago

    logger.onNodeEnd("agent_node", { result: "done" }, { runId: "run-123" });

    const [{ payload }] = captureCalls(fetchSpy);
    expect(payload["status"]).toBe("success");
    expect(payload["duration_ms"]).toBeDefined();
    expect(payload["duration_ms"]).toBeGreaterThan(0);
    expect(map.has("run-123")).toBe(false);
  });

  test("uses provided durationMs when given (overrides computed value)", () => {
    // test_uses_provided_duration_ms
    const fetchSpy = vi.fn().mockResolvedValue(emptyResponse());
    vi.stubGlobal("fetch", fetchSpy);

    const logger = makeLogger();

    logger.onNodeEnd("agent_node", {}, { runId: "run-456", durationMs: 42.5 });

    const [{ payload }] = captureCalls(fetchSpy);
    expect(payload["duration_ms"]).toBe(42.5);
  });

  test("works without a recorded start time (status still success)", () => {
    // test_handles_missing_start_time
    const fetchSpy = vi.fn().mockResolvedValue(emptyResponse());
    vi.stubGlobal("fetch", fetchSpy);

    const logger = makeLogger();

    logger.onNodeEnd("agent_node", {}, { runId: "run-unknown" });

    const [{ payload }] = captureCalls(fetchSpy);
    expect(payload["status"]).toBe("success");
  });
});

// ---------------------------------------------------------------------------
// onNodeError
// ---------------------------------------------------------------------------

describe("NodeExecutionLogger.onNodeError", () => {
  test("sends error status with the error message", () => {
    // test_sends_error_payload
    const fetchSpy = vi.fn().mockResolvedValue(emptyResponse());
    vi.stubGlobal("fetch", fetchSpy);

    const logger = makeLogger();

    logger.onNodeError("agent_node", "Something went wrong", {
      runId: "run-123",
    });

    const [{ payload }] = captureCalls(fetchSpy);
    expect(payload["status"]).toBe("error");
    expect(payload["error"]).toBe("Something went wrong");
    expect(payload["node_name"]).toBe("agent_node");
  });

  test("removes the recorded start time on error", () => {
    // test_cleans_up_start_time
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(emptyResponse()));

    const logger = makeLogger();
    const map = (logger as unknown as { nodeStartTimes: Map<string, Date> })
      .nodeStartTimes;
    map.set("run-123", new Date());

    logger.onNodeError("agent_node", "fail", { runId: "run-123" });

    expect(map.has("run-123")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// onNodeSuspend
// ---------------------------------------------------------------------------

describe("NodeExecutionLogger.onNodeSuspend", () => {
  test("sends suspend status with no error field", () => {
    // test_sends_suspend_payload
    const fetchSpy = vi.fn().mockResolvedValue(emptyResponse());
    vi.stubGlobal("fetch", fetchSpy);

    const logger = makeLogger();

    logger.onNodeSuspend("agent_node", { runId: "run-123" });

    const [{ payload }] = captureCalls(fetchSpy);
    expect(payload["status"]).toBe("suspend");
    expect(payload["node_name"]).toBe("agent_node");
    expect(payload).not.toHaveProperty("error");
  });

  test("computes duration_ms from start time when available", () => {
    // test_computes_duration_from_start_time
    const fetchSpy = vi.fn().mockResolvedValue(emptyResponse());
    vi.stubGlobal("fetch", fetchSpy);

    const logger = makeLogger();
    const map = (logger as unknown as { nodeStartTimes: Map<string, Date> })
      .nodeStartTimes;
    map.set("run-123", new Date(Date.now() - 1));

    logger.onNodeSuspend("agent_node", { runId: "run-123" });

    const [{ payload }] = captureCalls(fetchSpy);
    expect(payload["duration_ms"]).toBeDefined();
    expect(payload["duration_ms"]).toBeGreaterThan(0);
  });

  test("omits duration_ms from payload when no start time was recorded", () => {
    // test_duration_none_when_no_start_time
    const fetchSpy = vi.fn().mockResolvedValue(emptyResponse());
    vi.stubGlobal("fetch", fetchSpy);

    const logger = makeLogger();

    logger.onNodeSuspend("agent_node", { runId: "run-unknown" });

    const [{ payload }] = captureCalls(fetchSpy);
    expect(payload).not.toHaveProperty("duration_ms");
  });

  test("removes the recorded start time on suspend", () => {
    // test_cleans_up_start_time
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(emptyResponse()));

    const logger = makeLogger();
    const map = (logger as unknown as { nodeStartTimes: Map<string, Date> })
      .nodeStartTimes;
    map.set("run-123", new Date());

    logger.onNodeSuspend("agent_node", { runId: "run-123" });

    expect(map.has("run-123")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// onNodeInterrupted
// ---------------------------------------------------------------------------

describe("NodeExecutionLogger.onNodeInterrupted", () => {
  test("sends interrupted status with partial duration and no error field", () => {
    // test_sends_interrupted_payload_with_partial_duration
    const fetchSpy = vi.fn().mockResolvedValue(emptyResponse());
    vi.stubGlobal("fetch", fetchSpy);

    const logger = makeLogger();
    const map = (logger as unknown as { nodeStartTimes: Map<string, Date> })
      .nodeStartTimes;
    map.set("run-123", new Date(Date.now() - 150));

    logger.onNodeInterrupted("agent_node", { runId: "run-123" });

    const [{ payload }] = captureCalls(fetchSpy);
    expect(payload["status"]).toBe("interrupted");
    expect(payload["node_name"]).toBe("agent_node");
    expect(payload).not.toHaveProperty("error");
    expect(payload["duration_ms"]).toBeGreaterThanOrEqual(100);
    expect(map.has("run-123")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// HTTP failure silence
// ---------------------------------------------------------------------------

describe("NodeExecutionLogger — HTTP failure handling", () => {
  test("fetch rejection does not raise to the caller (errors swallowed)", () => {
    // test_http_error_does_not_raise
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new Error("Connection refused")),
    );

    const logger = makeLogger();

    // The fire-and-forget _sendNodeEvent catches its own errors; onNodeStart
    // must not throw even when fetch rejects.
    expect(() =>
      logger.onNodeStart("agent_node", {}, { runId: "run-123" }),
    ).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// serializeForJson — private helper
// ---------------------------------------------------------------------------

interface LoggerWithPrivates {
  serializeForJson: (obj: unknown) => unknown;
}

describe("NodeExecutionLogger.serializeForJson", () => {
  function privates(): LoggerWithPrivates {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(emptyResponse()));
    return makeLogger() as unknown as LoggerWithPrivates;
  }

  test("passes through primitives (string, number, boolean, null)", () => {
    // test_handles_primitives
    const s = privates();
    expect(s.serializeForJson("hello")).toBe("hello");
    expect(s.serializeForJson(42)).toBe(42);
    expect(s.serializeForJson(3.14)).toBe(3.14);
    expect(s.serializeForJson(true)).toBe(true);
    expect(s.serializeForJson(null)).toBe(null);
  });

  test("recursively serializes nested objects", () => {
    // test_handles_dicts
    const s = privates();
    expect(s.serializeForJson({ key: "value", nested: { a: 1 } })).toEqual({
      key: "value",
      nested: { a: 1 },
    });
  });

  test("recursively serializes arrays", () => {
    // test_handles_lists
    const s = privates();
    expect(s.serializeForJson([1, "two", { three: 3 }])).toEqual([
      1,
      "two",
      { three: 3 },
    ]);
  });

  test("uses toJSON() when present (Pydantic .dict() analog)", () => {
    // test_handles_pydantic_model
    // Python tests Pydantic BaseModel which provides .dict(). TS uses any
    // object with a toJSON() method (Date, custom classes, etc.).
    const s = privates();
    const obj = { x: 1, toJSON: () => ({ x: 1 }) };
    expect(s.serializeForJson(obj)).toEqual({ x: 1 });
  });

  test("falls back to String() for non-object types (function, symbol, bigint)", () => {
    // test_falls_back_to_str
    // Python uses `object()`; in TS, plain objects are recursively serialized,
    // so the fallback only catches genuinely non-object types (functions etc.).
    const s = privates();
    const result = s.serializeForJson(() => "x");
    expect(typeof result).toBe("string");
  });
});
