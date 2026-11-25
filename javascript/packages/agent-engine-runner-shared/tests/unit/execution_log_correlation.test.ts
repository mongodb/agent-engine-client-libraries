/**
 * Mirrors runner-shared/tests/unit/test_execution_log_correlation.py.
 *
 * Customers filter runner logs by `executionId`, so a line written while the
 * runtime works for a known execution must carry that ID even before or after
 * `runWithExecutionContext`: setup, cleanup, metrics, background work and side
 * requests. Each test finds its lines by message text, not by ID, so a missing
 * ID cannot hide a line.
 *
 * Timers stay real: fake timers run callbacks from the test's async context,
 * which would hide the inheritance these tests guard.
 */

import { once } from "node:events";
import { request as httpRequest } from "node:http";
import type { AddressInfo } from "node:net";

import log4js from "log4js";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import {
  AERServer,
  ToolServer,
  requestSessionFinish,
  type ExecuteRequest,
  type ITenantRuntime,
  type ServerToolFn,
  type StreamingResult,
} from "../../src/index.js";
import { RuntimeAgentConfig } from "../../src/agent_config.js";
import { runWithLogCorrelation } from "../../src/log_correlation.js";
import type { LLMStreamChunk } from "../../src/index.js";
import type * as McpToolsModule from "../../src/mcp_tools.js";
import { CallbackDelivery } from "../../src/server/callback_delivery.js";
import { AttemptHeartbeat } from "../../src/workflow/heartbeat.js";
import type { WorkflowClient } from "../../src/workflow/client.js";
import type { AttemptContext } from "../../src/generated/workflow/v1/runtime_pb.js";
import { captureJsonLogs, type JsonLogCapture } from "./json_log_capture.js";

const mcp = vi.hoisted(() => ({ resolveBinding: vi.fn() }));

vi.mock("../../src/mcp_tools.js", async (importOriginal) => {
  const actual = await importOriginal<typeof McpToolsModule>();
  return {
    ...actual,
    resolveConfiguredMcpToolBinding: (...args: unknown[]) =>
      mcp.resolveBinding(...args),
  };
});

const probe = log4js.getLogger("correlation_probe");
let logs: JsonLogCapture;

beforeEach(() => {
  // Set, it sends /execute through OE registration the stub runtime cannot do.
  vi.stubEnv("APP_ID", "");
  logs = captureJsonLogs();
});

afterEach(() => {
  logs.restore();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

function stdoutMessages(): Array<[string, string | null]> {
  return logs.records
    .filter((r) => r.source === "stdout")
    .map((r) => [r.message, r.executionId]);
}

interface AerPrivates {
  runtime: Record<string, unknown>;
  reportCallback: (...args: unknown[]) => Promise<void>;
  executeViaAgentStream: (...args: unknown[]) => Promise<StreamingResult>;
  sendStreamChunk: ReturnType<typeof vi.fn>;
  onStartup: () => Promise<void>;
}

function aerApp(configure: (server: AerPrivates) => void) {
  const runtime: Record<string, unknown> = {
    appName: "test-agent",
    appVersion: "0.0.1",
    orgId: null,
    projectId: null,
    graphBuilder: {},
    tools: {},
    toolDefinitions: {},
    getAgentConfig: () => ({}),
    getMongodbUri: () => null,
    getAgent: () => ({}),
  };
  const server = new AERServer(runtime as never) as unknown as AERServer &
    AerPrivates;
  server.sendStreamChunk = vi.fn().mockResolvedValue(undefined);
  server.reportCallback = async () => undefined;
  // The real error handler is under test; tenant startup is not.
  server.onStartup = async () => undefined;
  configure(server);
  return server.createApp();
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

describe("AER /execute", () => {
  test("setup, body, callback, metric and session-finish logs and output carry the execution ID", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.includes("/finish")) {
          // Well after the request, and its own output flush, have ended.
          await sleep(20);
          process.stdout.write("release partial");
        }
        throw new TypeError("oe unreachable");
      }),
    );
    const app = aerApp((server) => {
      server.executeViaAgentStream = async () => {
        requestSessionFinish();
        probe.info("agent body line");
        return { content: "ok", messages: [] } as unknown as StreamingResult;
      };
      server.reportCallback = async () => {
        probe.info("callback delivery line");
      };
    });

    const res = await app.inject({
      method: "POST",
      url: "/execute",
      payload: executeRequest("exec-lifecycle-1"),
    });
    expect(res.statusCode, res.body).toBe(200);
    // The release runs after the request's own output was flushed.
    await vi.waitFor(
      () =>
        expect(stdoutMessages()).toEqual([
          ["release partial", "exec-lifecycle-1"],
        ]),
      { timeout: 5000 },
    );
    probe.info("after request");

    for (const prefix of [
      "AER: OE callback URL",
      "agent body line",
      "callback delivery line",
      "METRIC latency",
      "Session finish request failed",
    ]) {
      expect(logs.ids(prefix), prefix).toEqual(new Set(["exec-lifecycle-1"]));
    }
    expect(logs.ids("after request")).toEqual(new Set([null]));
    await app.close();
  });

  test("a setup failure is logged with the execution ID and without credentials", async () => {
    vi.stubEnv("PAYMENTS_API_TOKEN", "tenant-token-value-123");
    const app = aerApp((server) => {
      server.runtime["getAgent"] = () => {
        throw new Error(
          "cannot reach mongodb+srv://agent:Pa55w0rd@cluster0.example.net/db " +
            "with tenant-token-value-123",
        );
      };
    });

    const res = await app.inject({
      method: "POST",
      url: "/execute",
      payload: executeRequest("exec-setup-1"),
    });

    expect(res.statusCode).toBe(500);
    expect(logs.ids("Unhandled error")).toEqual(new Set(["exec-setup-1"]));
    const logged = JSON.stringify(
      logs.records.filter((r) => r.message.startsWith("Unhandled error")),
    );
    expect(logged).toContain("cluster0.example.net");
    expect(logged).not.toContain("Pa55w0rd");
    expect(logged).not.toContain("tenant-token-value-123");
    await app.close();
  });

  test("concurrent executions never carry each other's ID", async () => {
    let arrived = 0;
    let releaseBoth!: () => void;
    const bothRunning = new Promise<void>((resolve) => (releaseBoth = resolve));
    const app = aerApp((server) => {
      server.executeViaAgentStream = async () => {
        if (++arrived === 2) releaseBoth();
        await bothRunning;
        return { content: "ok", messages: [] } as unknown as StreamingResult;
      };
      server.reportCallback = async (...args: unknown[]) => {
        await sleep(0);
        probe.info(`callback for ${String(args[1])}`);
      };
    });

    const responses = await Promise.all(
      ["execA-concurrent", "execB-concurrent"].map((id) =>
        app.inject({
          method: "POST",
          url: "/execute",
          payload: executeRequest(id),
        }),
      ),
    );

    expect(responses.map((r) => r.statusCode)).toEqual([200, 200]);
    for (const id of ["execA-concurrent", "execB-concurrent"]) {
      expect(logs.ids(`callback for ${id}`)).toEqual(new Set([id]));
    }
    await app.close();
  });
});

test("a heartbeat warning carries its attempt's execution ID wherever it started", async () => {
  let sends = 0;
  const client = {
    heartbeat: vi.fn(async () => {
      if (++sends > 1) throw new Error("OE unreachable");
    }),
  } as unknown as WorkflowClient;
  const attempt = {
    attemptId: "attempt-1",
    fencingToken: 7n,
    ownerId: "aer-1",
    workflowIdentity: { sessionId: "s-1", executionId: "exec-heartbeat-1" },
    heartbeatIntervalMs: 10n,
  } as unknown as AttemptContext;
  const heartbeat = new AttemptHeartbeat(attempt, client);

  await heartbeat.start();
  await vi.waitFor(() => logs.ids("Failed to renew durable attempt lease"));
  await heartbeat.stop();

  expect(logs.ids("Failed to renew durable attempt lease")).toEqual(
    new Set(["exec-heartbeat-1"]),
  );
});

test("callback redelivery names the retained execution, not the caller", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("oe down")));
  const delivery = new CallbackDelivery();

  await runWithLogCorrelation("exec-unrelated", () =>
    delivery.send("http://oe", {
      execution_id: "exec-retained",
      status: "COMPLETED",
    }),
  );
  await vi.waitFor(() => logs.ids("Callback redelivery remains pending"), {
    timeout: 5000,
  });
  await delivery.shutdown();

  expect(logs.ids("Callback redelivery remains pending")).toEqual(
    new Set(["exec-retained"]),
  );
});

function toolServer(tools: Record<string, ServerToolFn> = {}): ToolServer {
  return new ToolServer({
    tools,
    toolDefinitions: Object.fromEntries(Object.keys(tools).map((k) => [k, {}])),
    graphBuilder: null,
    getAgentConfig: () => new RuntimeAgentConfig(),
    getMongodbUri: () => null,
  } as unknown as ITenantRuntime);
}

function toolBody(
  executionId: string,
  toolName: string,
  extra: Record<string, unknown> = {},
) {
  return {
    execution_id: executionId,
    tool_name: toolName,
    arguments: {},
    session_id: "session-1",
    ...extra,
  };
}

describe("Tool /execute", () => {
  test("boot output is untagged; setup, resolution failure and metric logs carry the ID", async () => {
    mcp.resolveBinding.mockImplementation(() => {
      throw new Error("bad mcp url");
    });
    const app = toolServer().createApp();
    await app.ready();

    const res = await app.inject({
      method: "POST",
      url: "/execute",
      payload: toolBody("exec-tool-1", "github__search_issues", {
        metadata: { mcp_server: "github", mcp_tool: "search_issues" },
      }),
    });

    expect(res.statusCode).toBe(500);
    expect(logs.ids("Registered tools")).toEqual(new Set([null]));
    for (const prefix of [
      "TOOL_POD: Step 0 - github__search_issues",
      "Unhandled error: bad mcp url",
      "METRIC latency",
    ]) {
      expect(logs.ids(prefix), prefix).toEqual(new Set(["exec-tool-1"]));
    }
    await app.close();
  });

  test("concurrent calls for one execution keep and flush their own partial output", async () => {
    let releaseFirst!: () => void;
    const firstGate = new Promise<void>((resolve) => (releaseFirst = resolve));
    const app = toolServer({
      first: async () => {
        process.stdout.write("first partial");
        await firstGate;
        return "ok";
      },
      second: async () => {
        process.stdout.write("second partial");
        return "ok";
      },
    }).createApp();
    await app.ready();

    const first = app.inject({
      method: "POST",
      url: "/execute",
      payload: toolBody("exec-shared", "first", { step_number: 1 }),
    });
    await vi.waitFor(() =>
      expect(
        logs.records.some((r) =>
          r.message.startsWith("TOOL_POD: Step 0 - first"),
        ),
      ).toBe(true),
    );
    await app.inject({
      method: "POST",
      url: "/execute",
      payload: toolBody("exec-shared", "second", { step_number: 2 }),
    });
    expect(stdoutMessages()).toEqual([["second partial", "exec-shared"]]);

    releaseFirst();
    await first;
    expect(stdoutMessages()).toEqual([
      ["second partial", "exec-shared"],
      ["first partial", "exec-shared"],
    ]);
    await app.close();
  });
});

interface LlmSeams {
  resolveLlmForRequest: () => Promise<unknown>;
  streamLlmChunks: (...args: unknown[]) => AsyncGenerator<LLMStreamChunk>;
}

function llmApp(seams: Partial<LlmSeams>) {
  const server = toolServer() as unknown as ToolServer & LlmSeams;
  server.resolveLlmForRequest = async () => ({});
  Object.assign(server, seams);
  return server.createApp();
}

function llmBody(executionId: string) {
  return {
    execution_id: executionId,
    arguments: {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hi" }],
    },
  };
}

describe("Tool /invoke_llm", () => {
  test("output and the metric after the handler carry the execution ID", async () => {
    const app = llmApp({
      streamLlmChunks: async function* () {
        process.stdout.write("llm partial");
        yield { content: "hello" } as LLMStreamChunk;
      },
    });

    const res = await app.inject({
      method: "POST",
      url: "/invoke_llm",
      payload: llmBody("exec-llm-1"),
    });

    expect(res.json().status).toBe("success");
    expect(stdoutMessages()).toEqual([["llm partial", "exec-llm-1"]]);
    expect(logs.ids("METRIC latency")).toEqual(new Set(["exec-llm-1"]));
    await app.close();
  });
});

describe("Tool /invoke_llm/stream", () => {
  test("stream output and a model resolution failure carry the execution ID", async () => {
    const app = llmApp({
      resolveLlmForRequest: async () => {
        process.stdout.write("loading model");
        throw new Error("no such model");
      },
    });

    const res = await app.inject({
      method: "POST",
      url: "/invoke_llm/stream",
      payload: llmBody("exec-stream-fail"),
    });

    expect(res.body).toContain("error");
    expect(logs.ids("LLM stream failed")).toEqual(
      new Set(["exec-stream-fail"]),
    );
    expect(stdoutMessages()).toEqual([["loading model", "exec-stream-fail"]]);
    await app.close();
  });

  test("provider abort handling and cleanup output after a client disconnect carry the ID", async () => {
    let resumeProvider!: () => void;
    const providerGate = new Promise<void>(
      (resolve) => (resumeProvider = resolve),
    );
    let providerSignal!: AbortSignal;
    const app = llmApp({
      streamLlmChunks: async function* (_request, signal) {
        providerSignal = signal as AbortSignal;
        providerSignal.addEventListener("abort", () =>
          probe.info("provider saw abort"),
        );
        try {
          yield { content: "hello" } as LLMStreamChunk;
          // Ignores the abort signal, so the generator settles only after the
          // pipe has already returned.
          await providerGate;
          yield { content: "late" } as LLMStreamChunk;
        } finally {
          await sleep(10);
          process.stdout.write("cleanup partial");
        }
      },
    });
    await app.listen({ port: 0, host: "127.0.0.1" });
    const { port } = app.server.address() as AddressInfo;

    await new Promise<void>((resolve, reject) => {
      const req = httpRequest(
        {
          host: "127.0.0.1",
          port,
          method: "POST",
          path: "/invoke_llm/stream",
          headers: { "content-type": "application/json" },
        },
        (res) => {
          res.once("data", () => {
            req.destroy();
            resolve();
          });
        },
      );
      req.on("error", () => undefined);
      req.end(JSON.stringify(llmBody("exec-stream-1")));
      setTimeout(() => reject(new Error("no first chunk")), 5000);
    });
    // The pipe aborts this signal when it sees the disconnect, and returns
    // without waiting for the generator, whose cleanup then writes after a
    // timer: later than the route's own flush.
    if (!providerSignal.aborted) await once(providerSignal, "abort");
    resumeProvider();

    await vi.waitFor(() =>
      expect(stdoutMessages()).toEqual([["cleanup partial", "exec-stream-1"]]),
    );
    expect(logs.ids("provider saw abort")).toEqual(new Set(["exec-stream-1"]));
    await app.close();
  });
});

function guardrailCheck(executionId: string) {
  return {
    execution_id: executionId,
    stage: "llm_output",
    input: { text: "hi", metadata: {} },
    context: { org_id: "o1", project_id: "p1" },
    // An uncompilable pattern makes the evaluator warn and fail closed.
    policies: [
      {
        id: "policy-1",
        type: "output_validation",
        status: "active",
        action: "block",
        stage_filter: ["llm_output"],
        config: { regex_patterns: ["("] },
      },
    ],
  };
}

test("guardrail warnings carry only their own check's execution ID", async () => {
  const app = toolServer().createApp();
  await app.ready();
  const check = (executionId: string) =>
    app.inject({
      method: "POST",
      url: "/guardrails/check",
      payload: guardrailCheck(executionId),
    });

  const checks = await Promise.all([
    check("exec-guard-a"),
    check("exec-guard-b"),
  ]);
  // A check with no originating execution must not take its caller's.
  const unbound = await runWithLogCorrelation("exec-stale", () => check(""));

  expect([...checks, unbound].map((c) => c.json().decision)).toEqual([
    "block",
    "block",
    "block",
  ]);
  const warnings = logs.records
    .filter((r) => r.message.startsWith("Invalid guardrail regex pattern"))
    .map((r) => r.executionId);
  expect(warnings.sort()).toEqual(
    ["exec-guard-a", "exec-guard-b", null].sort(),
  );
  await app.close();
});

test("drain and interrupt logs carry the ID without splitting the running call's output", async () => {
  let releaseTool!: () => void;
  const toolGate = new Promise<void>((resolve) => (releaseTool = resolve));
  const server = toolServer({
    slow: async () => {
      process.stdout.write("running partial");
      await toolGate;
      process.stdout.write(" done\n");
      return "ok";
    },
  });
  const app = server.createApp();
  await app.ready();

  const execute = app.inject({
    method: "POST",
    url: "/execute",
    payload: toolBody("exec-drained", "slow", { step_number: 1 }),
  });
  await vi.waitFor(() =>
    expect(
      logs.records.some((r) => r.message.startsWith("TOOL_POD: Step 0 - slow")),
    ).toBe(true),
  );
  const interrupt = await app.inject({
    method: "POST",
    url: "/interrupt/call",
    payload: { execution_id: "exec-drained", step_number: 1 },
  });
  const drain = await app.inject({
    method: "POST",
    url: "/drain",
    payload: {
      request_id: "drain-aaaa",
      execution_id: "exec-drained",
      reason: "execution_cancelled",
      deadline_at_ms: Date.now() + 5000,
    },
  });
  expect(stdoutMessages()).toEqual([]);

  releaseTool();
  await execute;
  await vi.waitFor(() => logs.ids("Drain completed"));

  expect([interrupt.statusCode, drain.statusCode]).toEqual([200, 202]);
  for (const prefix of [
    "Call interrupt",
    "Drain accepted",
    "Drain completed",
  ]) {
    expect(logs.ids(prefix), prefix).toEqual(new Set(["exec-drained"]));
  }
  expect(stdoutMessages()).toEqual([["running partial done", "exec-drained"]]);
  await app.close();
});
