/**
 * Tests for function mode (agent_engine_runner_shared.server.function).
 *
 * Mirrors Python's tests/unit/test_function_mode.py.
 *
 * TS-vs-Python adaptations:
 *   - Python injects an httpx client; TS stubs the global `fetch` via
 *     vi.stubGlobal, matching the pattern used in aer_chunk_delivery_reliability.
 *   - Python `tmp_path` fixture → Node's `fs.mkdtempSync(os.tmpdir(), ...)`.
 *   - Python `object()` for unserializable values → circular reference
 *     (JSON.stringify throws on circular refs, the closest JS equivalent).
 *   - Python `model_dump(mode="json")` sanitises NaN/Infinity → null; JS's
 *     JSON.stringify does the same natively, so no downgrade occurs.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import type * as TracingSetup from "../../src/tracing/setup.js";

// vi.hoisted ensures `tracingMocks` exists when the hoisted vi.mock factory
// runs. Only getCurrentTraceContext is overridden; other tracing/setup.js
// exports pass through untouched via importOriginal.
const tracingMocks = vi.hoisted(() => ({
  getCurrentTraceContext: vi.fn(() => ({ traceId: null, spanId: null })),
}));

vi.mock("../../src/tracing/setup.js", async (importOriginal) => {
  const actual = await importOriginal<typeof TracingSetup>();
  return {
    ...actual,
    getCurrentTraceContext: tracingMocks.getCurrentTraceContext,
  };
});

import {
  ToolFunctionRunner,
  buildResultPayload,
  getRuntimeMode,
  RuntimeMode,
  getCurrentPayload,
  getCurrentUserId,
  getCurrentTraceId,
  emitStep,
  registerLlm,
  getNamedLlm,
  hasNamedLlms,
  resetHooks,
  entrypointScope,
  type ITenantRuntime,
  type ServerToolFn,
  type ToolResultRequest,
} from "../../src/index.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const createdDirs: string[] = [];

beforeEach(() => {
  tracingMocks.getCurrentTraceContext.mockReturnValue({
    traceId: null,
    spanId: null,
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
  for (const dir of createdDirs.splice(0)) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

interface MockRuntime {
  tools: Record<string, ServerToolFn>;
  toolDefinitions: Record<string, Record<string, unknown>>;
  graphBuilder: {
    getAgent?: () => unknown;
    ready?: () => Promise<void>;
  } | null;
}

function makeMockRuntime(tools: Record<string, ServerToolFn>): MockRuntime {
  return {
    tools,
    toolDefinitions: Object.fromEntries(Object.keys(tools).map((k) => [k, {}])),
    graphBuilder: null,
  };
}

/**
 * Write the function-mode metadata contract into a temp directory and return
 * the directory path. Mirrors Python's _write_contract / what fctr materializes
 * from RunRequest.metadata.
 */
function writeContract(opts: {
  metaDir: string;
  toolName?: string;
  arguments?: Record<string, unknown>;
  executionId?: string;
  sessionId?: string;
  oeUrl?: string | null;
  step?: unknown;
  includeStep?: boolean;
  requestJson?: string;
  extra?: Record<string, unknown>;
  writeRequest?: boolean;
}): string {
  const {
    metaDir,
    toolName = "greet",
    executionId = "exec-1",
    sessionId = "sess-1",
    oeUrl = "http://oe",
    step = 3,
    includeStep = true,
    requestJson,
    extra = {},
    writeRequest = true,
  } = opts;

  if (writeRequest) {
    let json: string;
    if (requestJson !== undefined) {
      json = requestJson;
    } else {
      const toolRequest: Record<string, unknown> = {
        execution_id: executionId,
        tool_name: toolName,
        arguments: opts.arguments ?? { name: "ada" },
        session_id: sessionId,
        ...extra,
      };
      if (oeUrl !== null) toolRequest["oe_url"] = oeUrl;
      const envelope: Record<string, unknown> = { request: toolRequest };
      if (includeStep) envelope["step"] = step;
      json = JSON.stringify(envelope);
    }
    fs.writeFileSync(path.join(metaDir, "request"), json, "utf-8");
  }
  return metaDir;
}

function makeTempDir(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "runner-fn-test-"));
  createdDirs.push(dir);
  return dir;
}

/** Script fetch so each call returns a scripted result. */
type FetchEntry = "ok" | number | Error;
function scriptFetch(entries: FetchEntry[]): ReturnType<typeof vi.fn> {
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

/** Capture the URL and body of every POST; always succeeds. */
function captureFetch(): {
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

async function run(runtime: MockRuntime, metaDir: string): Promise<void> {
  await new ToolFunctionRunner(runtime as unknown as ITenantRuntime, {
    metadataDir: metaDir,
  }).run();
}

// ---------------------------------------------------------------------------
// RuntimeMode
// ---------------------------------------------------------------------------

describe("RuntimeMode.TOOL_FUNCTION", () => {
  test("enum value is tool_function", () => {
    // test_function_is_a_valid_runtime_mode
    expect(RuntimeMode.TOOL_FUNCTION).toBe("tool_function");
  });

  test("getRuntimeMode resolves tool_function from RUNNER_MODE env", () => {
    vi.stubEnv("RUNNER_MODE", "tool_function");
    expect(getRuntimeMode()).toBe(RuntimeMode.TOOL_FUNCTION);
  });
});

// ---------------------------------------------------------------------------
// Success path
// ---------------------------------------------------------------------------

describe("ToolFunctionRunner success path", () => {
  test("reports success result to /tool/result", async () => {
    const traceId = "0123456789abcdef0123456789abcdef";
    const metaDir = makeTempDir();
    writeContract({ metaDir, extra: { platform_trace_id: traceId } });
    const { mock, calls } = captureFetch();
    vi.stubGlobal("fetch", mock);
    const seen: (string | null)[] = [];
    await run(
      makeMockRuntime({
        greet: (args) => {
          seen.push(getCurrentTraceId());
          return `hello ${args["name"]}`;
        },
      }),
      metaDir,
    );
    expect(seen).toEqual([traceId]);
    expect(getCurrentTraceId()).toBeNull();
    expect(calls).toHaveLength(1);
    const { url, body } = calls[0];
    expect(url).toBe("http://oe/tool/result");
    const b = body as Record<string, unknown>;
    expect(b["execution_id"]).toBe("exec-1");
    expect(b["step_number"]).toBe(3);
    expect(b["tool_name"]).toBe("greet");
    expect(b["status"]).toBe("success");
    expect(b["result"]).toBe("hello ada");
    expect(b["error"]).toBeUndefined();
    expect(typeof b["duration_ms"]).toBe("number");
    expect(b["trace_id"]).toBeUndefined();
    expect(b["span_id"]).toBeUndefined();
  });

  test("supports async tools", async () => {
    // test_supports_async_tools
    const metaDir = makeTempDir();
    writeContract({ metaDir });
    const { mock, calls } = captureFetch();
    vi.stubGlobal("fetch", mock);

    await run(
      makeMockRuntime({ greet: async (args) => `hi ${args["name"]}` }),
      metaDir,
    );

    const b = calls[0]?.body as Record<string, unknown>;
    expect(b["status"]).toBe("success");
    expect(b["result"]).toBe("hi ada");
  });

  test("populates trace_id/span_id from the active span", async () => {
    // test_reports_trace_context_from_active_span
    tracingMocks.getCurrentTraceContext.mockReturnValue({
      traceId: "0123456789abcdef0123456789abcdef",
      spanId: "0123456789abcdef",
    });
    const metaDir = makeTempDir();
    writeContract({ metaDir });
    const { mock, calls } = captureFetch();
    vi.stubGlobal("fetch", mock);

    await run(
      makeMockRuntime({ greet: (args) => `hello ${args["name"]}` }),
      metaDir,
    );

    const b = calls[0]?.body as Record<string, unknown>;
    expect(b["trace_id"]).toBe("0123456789abcdef0123456789abcdef");
    expect(b["span_id"]).toBe("0123456789abcdef");
  });

  test("full request fields reach tool context", async () => {
    // test_full_request_fields_reach_tool_context
    const metaDir = makeTempDir();
    writeContract({
      metaDir,
      extra: {
        user_id: "user-7",
        payload: { k: "v" },
        custom_headers: { "x-trace": "abc" },
      },
    });
    const { mock } = captureFetch();
    vi.stubGlobal("fetch", mock);

    let capturedUserId: string | null = null;
    let capturedPayload: Record<string, unknown> | null = null;

    await run(
      makeMockRuntime({
        greet: () => {
          capturedUserId = getCurrentUserId();
          capturedPayload = getCurrentPayload();
          return "ok";
        },
      }),
      metaDir,
    );

    expect(capturedUserId).toBe("user-7");
    expect(capturedPayload).toEqual({ k: "v" });
  });
});

// ---------------------------------------------------------------------------
// Error cases — tool execution
// ---------------------------------------------------------------------------

describe("ToolFunctionRunner error reporting", () => {
  test("tool exception reported as error status", async () => {
    // test_tool_exception_reported_as_error
    const metaDir = makeTempDir();
    writeContract({ metaDir, toolName: "boom" });
    const { mock, calls } = captureFetch();
    vi.stubGlobal("fetch", mock);

    await run(
      makeMockRuntime({
        boom: () => {
          throw new Error("kaboom");
        },
      }),
      metaDir,
    );

    const b = calls[0]?.body as Record<string, unknown>;
    expect(b["status"]).toBe("error");
    expect(b["error"]).toContain("kaboom");
  });

  test("redacts a credential echoed by a failed provider call", async () => {
    vi.stubEnv("CUSTOM_API_KEY", "sk-live-123");
    try {
      const metaDir = makeTempDir();
      writeContract({ metaDir, toolName: "provider" });
      const { mock, calls } = captureFetch();
      vi.stubGlobal("fetch", mock);

      await run(
        makeMockRuntime({
          provider: () => {
            const e = new Error("Request failed") as Error & {
              response: { status: number; data: unknown };
            };
            e.response = {
              status: 401,
              data: { message: "Rejected credential sk-live-123" },
            };
            throw e;
          },
        }),
        metaDir,
      );

      const b = calls[0]?.body as Record<string, unknown>;
      expect(b["status"]).toBe("error");
      expect(String(b["error"])).not.toContain("sk-live-123");
      const toolApiError = b["tool_api_error"] as
        | Record<string, unknown>
        | undefined;
      if (toolApiError) {
        expect(String(toolApiError["reason"])).not.toContain("sk-live-123");
      }
    } finally {
      vi.unstubAllEnvs();
    }
  });

  test("unknown tool reported as error status", async () => {
    // test_unknown_tool_reported_as_error
    const metaDir = makeTempDir();
    writeContract({ metaDir, toolName: "missing" });
    const { mock, calls } = captureFetch();
    vi.stubGlobal("fetch", mock);

    await run(makeMockRuntime({ greet: () => "ok" }), metaDir);

    const b = calls[0]?.body as Record<string, unknown>;
    expect(b["status"]).toBe("error");
    expect(b["error"]).toContain("missing");
  });

  test("unserializable result downgraded to error", async () => {
    // test_unserializable_result_downgraded_to_error
    // JS equivalent of Python's object(): a circular reference causes JSON.stringify to throw.
    const metaDir = makeTempDir();
    writeContract({ metaDir, toolName: "circular" });
    const { mock, calls } = captureFetch();
    vi.stubGlobal("fetch", mock);

    await run(
      makeMockRuntime({
        circular: () => {
          const obj: Record<string, unknown> = {};
          obj["self"] = obj; // circular — JSON.stringify throws
          return obj;
        },
      }),
      metaDir,
    );

    expect(calls).toHaveLength(1);
    const b = calls[0]?.body as Record<string, unknown>;
    expect(b["status"]).toBe("error");
    expect(b["result"]).toBeUndefined();
    expect(b["error"]).toContain("serializable");
  });
});

// ---------------------------------------------------------------------------
// buildResultPayload — unit tests
// ---------------------------------------------------------------------------

describe("buildResultPayload", () => {
  function makeResult(
    overrides: Partial<ToolResultRequest> = {},
  ): ToolResultRequest {
    return {
      execution_id: "exec-1",
      step_number: 1,
      tool_name: "t",
      status: "success",
      result: "ok",
      duration_ms: 1.0,
      metadata: {},
      ...overrides,
    };
  }

  test("downgrade clears unserializable metadata", () => {
    // test_result_payload_downgrade_clears_unserializable_metadata
    const circular: Record<string, unknown> = {};
    circular["bad"] = circular;
    const result = makeResult({ metadata: { bad: circular } });
    const payload = buildResultPayload(result, "exec-1");

    expect(payload["status"]).toBe("error");
    expect(payload["result"]).toBeUndefined();
    expect(payload["metadata"]).toEqual({});
    expect(typeof payload["error"]).toBe("string");
    expect(() => JSON.stringify(payload)).not.toThrow();
  });

  test("NaN and Infinity in result produce strict JSON (no downgrade)", () => {
    // test_nan_and_inf_results_serialize_to_strict_json
    // JS JSON.stringify serializes NaN/Infinity as null natively; no downgrade occurs.
    // The test verifies the JSON output (not the JS object) to match Go's json.Unmarshal.
    const result = makeResult({
      result: { score: NaN, bounds: [Infinity, -Infinity] },
      metadata: { telemetry: { ratio: NaN } },
    });
    const payload = buildResultPayload(result, "exec-1");

    expect(payload["status"]).toBe("success");
    // Verify strict JSON round-trip (Go's encoding/json accepts null, not NaN).
    expect(() => JSON.stringify(payload)).not.toThrow();
    const roundTripped = JSON.parse(JSON.stringify(payload)) as Record<
      string,
      unknown
    >;
    const r = roundTripped["result"] as Record<string, unknown>;
    expect(r["score"]).toBeNull();
    expect(r["bounds"]).toEqual([null, null]);
    const m = roundTripped["metadata"] as Record<string, unknown>;
    expect((m["telemetry"] as Record<string, unknown>)["ratio"]).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Contract validation errors
// ---------------------------------------------------------------------------

describe("ToolFunctionRunner contract errors", () => {
  test("missing request file raises", async () => {
    // test_missing_request_raises
    const metaDir = makeTempDir();
    writeContract({ metaDir, writeRequest: false });

    await expect(run(makeMockRuntime({}), metaDir)).rejects.toThrow(/request/);
  });

  test("missing step raises", async () => {
    // test_missing_step_raises
    const metaDir = makeTempDir();
    writeContract({ metaDir, includeStep: false });

    await expect(
      run(makeMockRuntime({ greet: () => "ok" }), metaDir),
    ).rejects.toThrow(/ToolFunctionRequest/);
  });

  test("non-integer step raises", async () => {
    // test_non_integer_step_raises
    const metaDir = makeTempDir();
    writeContract({ metaDir, step: "notanint" });

    await expect(
      run(makeMockRuntime({ greet: () => "ok" }), metaDir),
    ).rejects.toThrow(/ToolFunctionRequest/);
  });

  test("missing oe_url raises", async () => {
    // test_missing_oe_url_raises
    const metaDir = makeTempDir();
    writeContract({ metaDir, oeUrl: null });

    await expect(
      run(makeMockRuntime({ greet: () => "ok" }), metaDir),
    ).rejects.toThrow(/oe_url/);
  });

  test("invalid request JSON raises", async () => {
    // test_invalid_request_json_raises
    const metaDir = makeTempDir();
    writeContract({ metaDir, requestJson: "not json" });

    await expect(run(makeMockRuntime({}), metaDir)).rejects.toThrow(
      /ToolFunctionRequest/,
    );
  });

  test("inner request missing required field raises", async () => {
    // test_inner_request_missing_required_field_raises (missing session_id)
    const metaDir = makeTempDir();
    writeContract({
      metaDir,
      requestJson: JSON.stringify({
        request: {
          execution_id: "e",
          tool_name: "greet",
          arguments: {},
          oe_url: "http://oe",
          // session_id omitted — required, min_length=1
        },
        step: 1,
      }),
    });

    await expect(
      run(makeMockRuntime({ greet: () => "ok" }), metaDir),
    ).rejects.toThrow(/ToolFunctionRequest/);
  });
});

// ---------------------------------------------------------------------------
// Result delivery failure
// ---------------------------------------------------------------------------

describe("ToolFunctionRunner result delivery", () => {
  test("startup failure includes platform_trace_id", async () => {
    const traceId = "0123456789abcdef0123456789abcdef";
    const metaDir = makeTempDir();
    writeContract({ metaDir, extra: { platform_trace_id: traceId } });
    const seen: (string | null)[] = [];
    const runtime = makeMockRuntime({ greet: () => "ok" });
    runtime.graphBuilder = {
      ready: async () => {
        seen.push(getCurrentTraceId());
        throw new Error("mcp discovery failed");
      },
    };

    await expect(run(runtime, metaDir)).rejects.toThrow(/mcp discovery failed/);
    expect(seen).toEqual([traceId]);
    expect(getCurrentTraceId()).toBeNull();
  });

  test("delivery failure after retries re-raises", async () => {
    // test_result_delivery_failure_raises
    const metaDir = makeTempDir();
    writeContract({ metaDir });
    const fetchMock = scriptFetch([
      new TypeError("connection refused"),
      new TypeError("connection refused"),
      new TypeError("connection refused"),
    ]);
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      run(makeMockRuntime({ greet: () => "ok" }), metaDir),
    ).rejects.toThrow(/connection refused/);
  });

  test("result delivery failure includes platform_trace_id", async () => {
    const traceId = "0123456789abcdef0123456789abcdef";
    const metaDir = makeTempDir();
    writeContract({ metaDir, extra: { platform_trace_id: traceId } });
    const seen: (string | null)[] = [];
    const fetchMock = vi.fn(async () => {
      seen.push(getCurrentTraceId());
      throw new TypeError("connection refused");
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      run(makeMockRuntime({ greet: () => "ok" }), metaDir),
    ).rejects.toThrow(/connection refused/);
    expect(seen.length).toBeGreaterThan(0);
    expect(seen.every((id) => id === traceId)).toBe(true);
    expect(getCurrentTraceId()).toBeNull();
  });

  test("transient 5xx retried; succeeds on retry", async () => {
    const metaDir = makeTempDir();
    writeContract({ metaDir });
    const fetchMock = scriptFetch([503, "ok"]);
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      run(makeMockRuntime({ greet: () => "ok" }), metaDir),
    ).resolves.toBeUndefined();

    expect(fetchMock.mock.calls).toHaveLength(2);
  });

  test("honors a capped delta-seconds Retry-After on a service 503", async () => {
    vi.useFakeTimers();
    try {
      const metaDir = makeTempDir();
      writeContract({ metaDir });
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

      const delivery = run(makeMockRuntime({ greet: () => "ok" }), metaDir);
      await vi.advanceTimersByTimeAsync(500);
      expect(fetchMock).toHaveBeenCalledTimes(1);

      await vi.advanceTimersByTimeAsync(10_000);
      await delivery;
      expect(fetchMock).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });

  test("4xx not retried; throws immediately", async () => {
    const metaDir = makeTempDir();
    writeContract({ metaDir });
    const fetchMock = scriptFetch([404]);
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      run(makeMockRuntime({ greet: () => "ok" }), metaDir),
    ).rejects.toThrow(/404/);

    expect(fetchMock.mock.calls).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// Owner-callback preference
// ---------------------------------------------------------------------------

describe("ToolFunctionRunner owner-callback preference", () => {
  const OE = "http://oe.ns.svc.cluster.local:8000";
  const OWNER = "http://10-1-2-3.oe-headless.ns.svc.cluster.local:8000";

  test("prefers the validated owner /tool/result", async () => {
    const metaDir = makeTempDir();
    writeContract({ metaDir, oeUrl: OE, extra: { oe_owner_url: OWNER } });
    const { mock, calls } = captureFetch();
    vi.stubGlobal("fetch", mock);

    await run(makeMockRuntime({ greet: () => "ok" }), metaDir);

    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe(`${OWNER}/tool/result`);
  });

  test("falls back to the service URL on an owner transport error with an identical body", async () => {
    const metaDir = makeTempDir();
    writeContract({ metaDir, oeUrl: OE, extra: { oe_owner_url: OWNER } });
    const calls: Array<{ url: string; body: unknown }> = [];
    const mock = vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, body: JSON.parse((init?.body as string) ?? "null") });
      if (url.startsWith(OWNER)) throw new TypeError("connection refused");
      return new Response(null, { status: 200 });
    });
    vi.stubGlobal("fetch", mock);

    await run(makeMockRuntime({ greet: () => "ok" }), metaDir);

    expect(calls.map((c) => c.url)).toEqual([
      `${OWNER}/tool/result`,
      `${OE}/tool/result`,
    ]);
    expect(calls[0].body).toEqual(calls[1].body);
  });

  test("an emit owner failure suppresses the final result owner attempt", async () => {
    const metaDir = makeTempDir();
    writeContract({ metaDir, oeUrl: OE, extra: { oe_owner_url: OWNER } });
    const urls: string[] = [];
    const mock = vi.fn(async (url: string) => {
      urls.push(url);
      if (url === `${OWNER}/stream/chunk`) throw new TypeError("refused");
      return new Response(null, { status: 200 });
    });
    vi.stubGlobal("fetch", mock);

    await run(
      makeMockRuntime({
        greet: () => {
          emitStep("working");
          return "ok";
        },
      }),
      metaDir,
    );

    expect(urls).toEqual([
      `${OWNER}/stream/chunk`,
      `${OE}/stream/chunk`,
      `${OE}/tool/result`,
    ]);
  });

  test("falls back to the service URL on an owner HTTP error; owner not retried", async () => {
    const ownerStatus = 500;
    const metaDir = makeTempDir();
    writeContract({ metaDir, oeUrl: OE, extra: { oe_owner_url: OWNER } });
    const calls: Array<{
      url: string;
      redirect: RequestRedirect | undefined;
    }> = [];
    const mock = vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, redirect: init?.redirect });
      if (url.startsWith(OWNER))
        return new Response(null, { status: ownerStatus });
      return new Response(null, { status: 200 });
    });
    vi.stubGlobal("fetch", mock);

    await run(makeMockRuntime({ greet: () => "ok" }), metaDir);

    expect(calls.map((c) => c.url)).toEqual([
      `${OWNER}/tool/result`,
      `${OE}/tool/result`,
    ]);
    expect(calls.filter((c) => c.url.startsWith(OWNER))).toHaveLength(1);
    expect(calls[0]?.redirect).toBe("manual");
    expect(calls[1]?.redirect).toBeUndefined();
  });

  test("owner unreachable does not consume the service retry budget", async () => {
    const metaDir = makeTempDir();
    writeContract({ metaDir, oeUrl: OE, extra: { oe_owner_url: OWNER } });
    const urls: string[] = [];
    const serviceStatuses = [503, 503, 200];
    const mock = vi.fn(async (url: string) => {
      urls.push(url);
      if (url.startsWith(OWNER)) throw new TypeError("connection refused");
      return new Response(null, { status: serviceStatuses.shift() ?? 200 });
    });
    vi.stubGlobal("fetch", mock);

    await run(makeMockRuntime({ greet: () => "ok" }), metaDir);

    // 1 owner pre-attempt + a full 3-attempt service budget.
    expect(urls).toEqual([
      `${OWNER}/tool/result`,
      `${OE}/tool/result`,
      `${OE}/tool/result`,
      `${OE}/tool/result`,
    ]);
  });

  test("security-negative: a forged owner URL is ignored; the forged host is never contacted", async () => {
    const metaDir = makeTempDir();
    writeContract({
      metaDir,
      oeUrl: OE,
      extra: { oe_owner_url: "http://attacker.example:8000" },
    });
    const { mock, calls } = captureFetch();
    vi.stubGlobal("fetch", mock);

    await run(makeMockRuntime({ greet: () => "ok" }), metaDir);

    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe(`${OE}/tool/result`);
    expect(calls.some((c) => c.url.includes("attacker.example"))).toBe(false);
  });

  test("security-negative: with OE_URL stamped, a forged (oe_url, owner) pair cannot redirect the callback", async () => {
    const prev = process.env.OE_URL;
    process.env.OE_URL = OE;
    try {
      const metaDir = makeTempDir();
      writeContract({
        metaDir,
        oeUrl: "http://oe.attacker.svc.cluster.local:8000",
        extra: {
          oe_owner_url:
            "http://10-9-9-9.oe-headless.attacker.svc.cluster.local:8000",
        },
      });
      const { mock, calls } = captureFetch();
      vi.stubGlobal("fetch", mock);

      await run(makeMockRuntime({ greet: () => "ok" }), metaDir);

      // The trust anchor is the env-resolved OE, so neither the forged owner nor
      // the forged service host is ever contacted.
      expect(calls).toHaveLength(1);
      expect(calls[0].url).toBe(`${OE}/tool/result`);
      expect(calls.some((c) => c.url.includes("attacker"))).toBe(false);
    } finally {
      if (prev === undefined) delete process.env.OE_URL;
      else process.env.OE_URL = prev;
    }
  });
});

// ---------------------------------------------------------------------------
// _prepare() — LLM registry population (mirrors tool.test.ts onStartup group)
// ---------------------------------------------------------------------------

describe("ToolFunctionRunner._prepare — LLM registry population", () => {
  beforeEach(() => resetHooks());
  afterEach(() => resetHooks());

  function stubFetchOk(): void {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(null, { status: 200 })),
    );
  }

  test("calls getAgent to populate the LLM registry", async () => {
    // test_on_startup_calls_entrypoint_to_populate_registry
    const metaDir = makeTempDir();
    writeContract({ metaDir });
    stubFetchOk();
    const getAgent = vi.fn(() => {
      entrypointScope(() => registerLlm("primary", {}));
      return {};
    });
    const runtime = makeMockRuntime({ greet: () => "ok" });
    runtime.graphBuilder = { getAgent };

    await run(runtime, metaDir);

    expect(getAgent).toHaveBeenCalledOnce();
    expect(hasNamedLlms()).toBe(true);
  });

  test("entrypoint throws — does not crash; restores import-time LLMs", async () => {
    // test_on_startup_restores_import_time_llms_when_entrypoint_fails
    const metaDir = makeTempDir();
    writeContract({ metaDir });
    stubFetchOk();
    const mockLlm = {};
    entrypointScope(() => registerLlm("import-time", mockLlm));
    const runtime = makeMockRuntime({ greet: () => "ok" });
    runtime.graphBuilder = {
      getAgent: vi.fn(() => {
        throw new Error("MongoDB unavailable");
      }),
    };

    await run(runtime, metaDir);

    expect(getNamedLlm("import-time")).toBe(mockLlm);
  });

  test("entrypoint registers no LLMs — restores import-time LLMs", async () => {
    // test_on_startup_restores_import_time_llms_when_entrypoint_registers_none
    const metaDir = makeTempDir();
    writeContract({ metaDir });
    stubFetchOk();
    const mockLlm = {};
    entrypointScope(() => registerLlm("import-time", mockLlm));
    const runtime = makeMockRuntime({ greet: () => "ok" });
    runtime.graphBuilder = { getAgent: vi.fn(() => ({})) };

    await run(runtime, metaDir);

    expect(getNamedLlm("import-time")).toBe(mockLlm);
  });

  test("always resets registry before re-running the entrypoint", async () => {
    // test_on_startup_always_runs_entrypoint_even_when_registry_populated
    const metaDir = makeTempDir();
    writeContract({ metaDir });
    stubFetchOk();
    entrypointScope(() => registerLlm("import-time", {}));
    const getAgent = vi.fn(() => {
      entrypointScope(() => registerLlm("entrypoint-only", {}));
      return {};
    });
    const runtime = makeMockRuntime({ greet: () => "ok" });
    runtime.graphBuilder = { getAgent };

    await run(runtime, metaDir);

    expect(getAgent).toHaveBeenCalledOnce();
    expect(() => getNamedLlm("import-time")).toThrow();
    expect(() => getNamedLlm("entrypoint-only")).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// External API error classification
// ---------------------------------------------------------------------------

describe("ToolFunctionRunner external API error classification", () => {
  test("tool exception with HTTP status is classified and tool_api_error reaches /tool/result", async () => {
    // test_classified_error_reaches_tool_result
    const metaDir = makeTempDir();
    writeContract({ metaDir, toolName: "api_call" });
    const { mock, calls } = captureFetch();
    vi.stubGlobal("fetch", mock);

    await run(
      makeMockRuntime({
        api_call: () => {
          const e = new Error("rate limited") as Error & { status: number };
          e.status = 429;
          throw e;
        },
      }),
      metaDir,
    );

    const b = calls[0]?.body as Record<string, unknown>;
    expect(b["status"]).toBe("error");
    expect(b["tool_api_error"]).toBeDefined();
    const tae = b["tool_api_error"] as Record<string, unknown>;
    expect(tae["classification"]).toBe("RATE_LIMITED");
    expect(tae["http_status"]).toBe(429);
    expect(tae["retryable"]).toBe(true);
  });

  test("provider_type is resolved from toolDefinitions", async () => {
    const metaDir = makeTempDir();
    writeContract({ metaDir, toolName: "api_call" });
    const { mock, calls } = captureFetch();
    vi.stubGlobal("fetch", mock);

    const runtime = makeMockRuntime({
      api_call: () => {
        const e = new Error("forbidden") as Error & { status: number };
        e.status = 403;
        throw e;
      },
    });
    runtime.toolDefinitions = { api_call: { provider_type: "stripe" } };

    await run(runtime, metaDir);

    const b = calls[0]?.body as Record<string, unknown>;
    const tae = b["tool_api_error"] as Record<string, unknown>;
    expect(tae["provider_type"]).toBe("stripe");
  });

  test("unclassified error keeps existing error behavior (no tool_api_error)", async () => {
    const metaDir = makeTempDir();
    writeContract({ metaDir, toolName: "boom" });
    const { mock, calls } = captureFetch();
    vi.stubGlobal("fetch", mock);

    await run(
      makeMockRuntime({
        boom: () => {
          throw new Error("kaboom");
        },
      }),
      metaDir,
    );

    const b = calls[0]?.body as Record<string, unknown>;
    expect(b["status"]).toBe("error");
    expect(b["tool_api_error"]).toBeUndefined();
    expect(b["error"]).toContain("kaboom");
  });

  test("transport timeout is classified", async () => {
    const metaDir = makeTempDir();
    writeContract({ metaDir, toolName: "slow_api" });
    const { mock, calls } = captureFetch();
    vi.stubGlobal("fetch", mock);

    await run(
      makeMockRuntime({
        slow_api: () => {
          const e = new Error("timed out") as Error & { name: string };
          e.name = "TimeoutError";
          throw e;
        },
      }),
      metaDir,
    );

    const b = calls[0]?.body as Record<string, unknown>;
    const tae = b["tool_api_error"] as Record<string, unknown>;
    expect(tae["classification"]).toBe("TIMEOUT");
    expect(tae["retryable"]).toBe(true);
  });
});
