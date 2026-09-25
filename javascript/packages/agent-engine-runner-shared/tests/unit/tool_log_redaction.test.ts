/**
 * Unit tests for tool request/result logging.
 *
 * Mirrors Python's tests/unit/test_tool_log_redaction.py.
 *
 * Platform debug logs must never emit argument or result bodies.
 * The catalog redact_fields contract is satisfied by omitting values
 * entirely; redactFields itself is still unit-tested because other
 * callers use it.
 */

import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import type * as TracingSetup from "../../src/tracing/setup.js";

const mocks = vi.hoisted(() => ({
  logger: {
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
    trace: vi.fn(),
    fatal: vi.fn(),
    log: vi.fn(),
    isLevelEnabled: vi.fn(() => true),
  },
}));

vi.mock("../../src/logger.js", () => ({
  getLogger: () => mocks.logger,
  setupLogging: vi.fn(),
}));

// requestOeApproval stamps the active trace context; pin it to nulls like
// secure_wrapper.test.ts so the fetch stub is the only network surface.
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
  logToolRequest,
  logToolResult,
  redactFields,
  toolRedactFields,
} from "../../src/utils.js";
import { ToolServer } from "../../src/server/tool.js";
import { ToolFunctionRunner } from "../../src/server/function.js";
import {
  SecureToolWrapper,
  createSecureToolFunction,
} from "../../src/secure_wrapper.js";
import { runWithExecutionContext } from "../../src/context.js";
import { RuntimeAgentConfig } from "../../src/agent_config.js";
import type { ITenantRuntime, ServerToolFn } from "../../src/server/base.js";
import type { ToolPodExecuteRequest } from "../../src/models.js";

interface MockRuntime {
  tools: Record<string, ServerToolFn>;
  toolDefinitions: Record<string, Record<string, unknown>>;
  graphBuilder: { getAgent?: () => unknown } | null;
  getAgentConfig: () => RuntimeAgentConfig;
}

function makeMockRuntime(): MockRuntime {
  return {
    tools: { get_weather: () => undefined },
    toolDefinitions: { get_weather: {} },
    graphBuilder: null,
    getAgentConfig: () => new RuntimeAgentConfig(),
  };
}

function makeExecuteRequest(
  toolName: string,
  args: Record<string, unknown>,
): ToolPodExecuteRequest {
  return {
    execution_id: "exec-1",
    tool_name: toolName,
    arguments: args,
    session_id: "session-1",
    metadata: {},
  } as ToolPodExecuteRequest;
}

/** All debug-log output as one string, for credential absence assertions. */
function debugOutput(): string {
  return mocks.logger.debug.mock.calls.map((c) => String(c[0])).join("\n");
}

function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

beforeEach(() => {
  mocks.logger.isLevelEnabled.mockReturnValue(true);
});

describe("redactFields", () => {
  test("listed fields are masked, unlisted pass through", () => {
    const out = redactFields({ card_number: "4111111111111111", amount: 42 }, [
      "card_number",
    ]);
    expect(out).toEqual({ card_number: "[REDACTED]", amount: 42 });
  });

  test("empty list returns the input unchanged (no behavior change without a policy)", () => {
    const args = { card_number: "4111111111111111" };
    expect(redactFields(args, [])).toBe(args);
  });

  test("missing keys are ignored", () => {
    const out = redactFields({ amount: 42 }, ["card_number", "cvv"]);
    expect(out).toEqual({ amount: 42 });
  });
});

describe("toolRedactFields", () => {
  test("reads redact_fields from a tool definition", () => {
    expect(
      toolRedactFields(
        { stripe_charge: { redact_fields: ["card_number", "cvv"] } },
        "stripe_charge",
      ),
    ).toEqual(["card_number", "cvv"]);
  });

  test("degrades to no redaction for unknown tools and malformed metadata", () => {
    expect(toolRedactFields({}, "nope")).toEqual([]);
    expect(toolRedactFields({ t: { redact_fields: "oops" } }, "t")).toEqual([]);
    expect(toolRedactFields({ t: {} }, undefined)).toEqual([]);
  });
});

describe("logToolRequest", () => {
  beforeEach(() => {
    for (const k of ["info", "error", "warn", "debug"] as const)
      mocks.logger[k].mockClear();
  });

  test("debug lists keys and omits values", () => {
    logToolRequest(
      "stripe_charge",
      { card_number: "4111111111111111", cvv: "123", amount: 42 },
      1,
      "TOOL",
      ["card_number", "cvv"],
    );
    const out = debugOutput();
    expect(out).toContain("card_number=<redacted>");
    expect(out).toContain("cvv=<redacted>");
    expect(out).toContain("amount=<number>");
    expect(out).toContain("values_omitted=true");
    expect(out).not.toContain("4111111111111111");
    expect(out).not.toContain("123");
  });

  test("debug omits argument values without a redaction policy", () => {
    logToolRequest("get_weather", { city: "Tokyo" }, 1);
    const out = debugOutput();
    expect(out).toContain("city=<string chars=5>");
    expect(out).not.toContain("Tokyo");
  });

  test("debug does not emit a large argument body", () => {
    const blob = "PAYLOAD_BODY_" + "x".repeat(200_000);
    logToolRequest("filesystem_write", { content: blob, path: "/tmp/out" }, 1);
    const out = debugOutput();
    expect(out).not.toContain(blob);
    expect(out).not.toContain("PAYLOAD_BODY_");
    expect(out).toContain("content=<string chars=200013>");
    expect(out).toContain("path=<string chars=8>");
  });

  test("debug caps fields and field-name length", () => {
    const csi = "\x9b";
    const longKey = "field\n" + csi + "x".repeat(100);
    const args: Record<string, unknown> = { [longKey]: "hidden" };
    for (let i = 0; i < 25; i += 1) args[`key_${i}`] = i;
    logToolRequest("wide_tool", args, 1);
    const out = debugOutput();
    expect(out).toContain("field??");
    expect(out).not.toContain(longKey);
    expect(out).not.toContain(csi);
    expect(out).toContain("key_18=<number>");
    expect(out).not.toContain("key_19");
    expect(out).toContain("fields_truncated=true");
  });

  test("non-debug logging does not inspect arguments", () => {
    mocks.logger.isLevelEnabled.mockReturnValue(false);
    const args = {} as Record<string, unknown>;
    Object.defineProperty(args, "content", {
      enumerable: true,
      get: () => {
        throw new Error("argument summary should not run");
      },
    });
    expect(() => logToolRequest("filesystem_write", args, 1)).not.toThrow();
    expect(debugOutput()).toBe("");
  });
});

describe("logToolResult", () => {
  beforeEach(() => {
    for (const k of ["info", "error", "warn", "debug"] as const)
      mocks.logger[k].mockClear();
  });

  test("debug does not stringify a large result", () => {
    const blob = "RESULT_BODY_" + "y".repeat(200_000);
    logToolResult("filesystem_write", 1, "success", blob, null, 12);
    const out = debugOutput();
    expect(out).not.toContain(blob);
    expect(out).not.toContain("RESULT_BODY_");
    expect(out).toContain("type=string chars=200012");
    expect(out).toContain("values_omitted=true");
  });

  test("debug does not emit result object keys", () => {
    const sensitiveKey = "customer-token-as-key";
    logToolResult(
      "lookup_tool",
      1,
      "success",
      { [sensitiveKey]: "hidden-value" },
      null,
      12,
    );
    const out = debugOutput();
    expect(out).toContain("type=object");
    expect(out).not.toContain(sensitiveKey);
    expect(out).not.toContain("hidden-value");
  });

  test("debug does not walk a result collection", () => {
    const result = Array.from({ length: 1000 }, () => ({
      toString: () => {
        throw new Error("nested result value was stringified");
      },
    }));
    expect(() =>
      logToolResult("list_tool", 1, "success", result),
    ).not.toThrow();
    expect(debugOutput()).toContain("type=array items=1000");
  });

  test("non-debug logging does not inspect results", () => {
    mocks.logger.isLevelEnabled.mockReturnValue(false);
    const result = new Proxy(
      {},
      {
        getPrototypeOf: () => {
          throw new Error("result summary should not run");
        },
      },
    );
    expect(() =>
      logToolResult("lookup_tool", 1, "success", result),
    ).not.toThrow();
    expect(debugOutput()).toBe("");
  });
});

describe("ToolServer.doHandleExecute redaction", () => {
  beforeEach(() => {
    for (const k of ["info", "error", "warn", "debug"] as const)
      mocks.logger[k].mockClear();
  });

  test("applies the tool definition's redact_fields to the request log", async () => {
    const runtime = makeMockRuntime();
    runtime.tools = { stripe_charge: () => ({ ok: true }) };
    runtime.toolDefinitions = {
      stripe_charge: { redact_fields: ["card_number"] },
    };
    const server = new ToolServer(runtime as unknown as ITenantRuntime);

    const response = await (
      server as unknown as {
        doHandleExecute: (
          request: ToolPodExecuteRequest,
        ) => Promise<{ status: string }>;
      }
    ).doHandleExecute(
      makeExecuteRequest("stripe_charge", {
        card_number: "4111111111111111",
        amount: 42,
      }),
    );

    expect(response.status).toBe("success");
    const out = debugOutput();
    expect(out).toContain("card_number=<redacted>");
    expect(out).toContain("amount=<number>");
    expect(out).not.toContain("4111111111111111");
  });

  test("unknown tools still omit argument values", async () => {
    const runtime = makeMockRuntime();
    const server = new ToolServer(runtime as unknown as ITenantRuntime);

    await (
      server as unknown as {
        doHandleExecute: (
          request: ToolPodExecuteRequest,
        ) => Promise<{ status: string }>;
      }
    ).doHandleExecute(makeExecuteRequest("ghost_tool", { hint: "visible" }));

    expect(debugOutput()).not.toContain("visible");
    expect(debugOutput()).toContain("hint=<string chars=7>");
  });
});

describe("ToolFunctionRunner._invokeToolFn redaction", () => {
  beforeEach(() => {
    for (const k of ["info", "error", "warn", "debug"] as const)
      mocks.logger[k].mockClear();
  });

  test("applies the tool definition's redact_fields to the request log", async () => {
    const runtime = makeMockRuntime();
    runtime.tools = { stripe_charge: () => ({ ok: true }) };
    runtime.toolDefinitions = {
      stripe_charge: { redact_fields: ["card_number", "cvv"] },
    };
    const runner = new ToolFunctionRunner(runtime as unknown as ITenantRuntime);

    const response = await (
      runner as unknown as {
        _invokeToolFn: (
          request: ToolPodExecuteRequest,
        ) => Promise<{ status: string }>;
      }
    )._invokeToolFn(
      makeExecuteRequest("stripe_charge", {
        card_number: "4111111111111111",
        cvv: "123",
        amount: 42,
      }),
    );

    expect(response.status).toBe("success");
    const out = debugOutput();
    expect(out).toContain("card_number=<redacted>");
    expect(out).not.toContain("4111111111111111");
    expect(out).not.toContain("123");
  });
});

describe("SecureToolWrapper.executeTool redaction", () => {
  beforeEach(() => {
    for (const k of ["info", "error", "warn", "debug"] as const)
      mocks.logger[k].mockClear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test("redactFields reach the debug argument metadata", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          proceed: true,
          status: "success",
          result: { ok: true },
          duration_ms: 1.0,
        }),
      ),
    );

    const wrapper = new SecureToolWrapper("http://localhost:8080", "exec-123");
    await wrapper.executeTool(
      "stripe_charge",
      { card_number: "4111111111111111", amount: 42 },
      { redactFields: ["card_number"] },
    );

    const out = debugOutput();
    expect(out).toContain("card_number=<redacted>");
    expect(out).not.toContain("4111111111111111");
  });

  test("createSecureToolFunction forwards redactFields to executeTool", async () => {
    const mockWrapper = { executeTool: vi.fn().mockResolvedValue("ok") };
    await runWithExecutionContext(
      {
        executionId: "exec-123",
        wrapper: mockWrapper as unknown as SecureToolWrapper,
        oeUrl: "http://localhost:8080",
      },
      async () => {
        const wrapped = createSecureToolFunction(
          { invoke: () => "unused" },
          "stripe_charge",
          false,
          { redactFields: ["card_number"] },
        );
        await wrapped({ card_number: "4111111111111111" });
      },
    );
    expect(mockWrapper.executeTool).toHaveBeenCalledWith(
      "stripe_charge",
      { card_number: "4111111111111111" },
      {
        metadata: undefined,
        providerType: undefined,
        scopes: undefined,
        toolCallId: undefined,
        rawOnInterrupt: true,
        redactFields: ["card_number"],
        isLocal: true,
        localExecutor: expect.any(Function),
      },
    );
  });
});
