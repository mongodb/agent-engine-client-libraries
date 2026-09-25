/**
 * Integration tests for the SecureWrappedLLM ↔ SecureLLMProxy path.
 *
 * Ports the proxy-routing cases deferred from `secure_llm.test.ts` (see the
 * NOT-ported list in that file's header). These exercise the primary AER
 * execution path for every LLM call: `_generate` / `_streamResponseChunks`
 * routing through `SecureLLMProxy`, policy-denial propagation, latency/error
 * metrics, cached-result logging, step-counter updates, and tool-message
 * repair before the proxy.
 *
 * `SecureLLMProxy`, `Metrics`, and `logCachedResult` are mocked at the module
 * boundary so no real HTTP/metrics calls happen; everything else (message
 * conversion, exception classes) is the real implementation.
 *
 * Mirrors Python `agent-engine-sdk-langgraph/tests/test_secure_llm.py`
 * (`TestSecureWrappedLLMGenerate`).
 */

import { describe, test, expect, vi, beforeEach } from "vitest";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import {
  LLMResponse,
  LLMInvocationOptions,
  type LLMStreamChunk,
} from "@mongodb-js/agent-engine-sdk";
import type * as RunnerShared from "@mongodb-js/agent-engine-runner-shared";

// Per-test control surface for the mocked SecureLLMProxy. Hoisted so the
// `vi.mock` factory (which runs before imports) can close over it.
const proxyController = vi.hoisted(() => ({
  constructorArgs: [] as unknown[],
  streamArgs: [] as unknown[][],
  streamChunks: [] as unknown[],
  streamError: null as Error | null,
  response: null as unknown,
  lastDurationMs: 0,
  lastFromCache: false,
  lastLatestStepNumber: null as number | null,
}));

vi.mock("@mongodb-js/agent-engine-runner-shared", async (importActual) => {
  const actual = await importActual<typeof RunnerShared>();

  class MockSecureLLMProxy {
    lastDurationMs: number;
    lastFromCache: boolean;
    lastLatestStepNumber: number | null;

    constructor(config: unknown) {
      proxyController.constructorArgs.push(config);
      this.lastDurationMs = proxyController.lastDurationMs;
      this.lastFromCache = proxyController.lastFromCache;
      this.lastLatestStepNumber = proxyController.lastLatestStepNumber;
    }

    async *stream(...args: unknown[]): AsyncGenerator<unknown> {
      proxyController.streamArgs.push(args);
      if (proxyController.streamError) throw proxyController.streamError;
      for (const chunk of proxyController.streamChunks) yield chunk;
    }

    static responseFromStreamChunks(_chunks: unknown): unknown {
      return proxyController.response;
    }
  }

  return {
    ...actual,
    SecureLLMProxy: MockSecureLLMProxy,
    Metrics: { recordLatency: vi.fn(), recordError: vi.fn() },
    logCachedResult: vi.fn(),
  };
});

// Imported AFTER the mock so these resolve to the mocked module. Exception
// classes are spread through from the real module, so `instanceof` still works.
import {
  Metrics,
  logCachedResult,
  PolicyDeniedException,
  LLMInvocationError,
} from "@mongodb-js/agent-engine-runner-shared";
import { SecureWrappedLLM } from "../src/secure_llm.js";

interface FakeWrapper {
  oeUrl: string;
  executionId: string;
  readonly stepCounter: number;
  operationalSteps: {
    next(): number;
    observeAtLeast(n: number): void;
    current(): number;
  };
  nextOperationalStep(): number;
  observeOperationalStep(n: number): void;
}

function makeWrapped(
  stepCounter = 0,
  innerExtra: Record<string, unknown> = {},
): { wrapper: FakeWrapper; wrapped: SecureWrappedLLM } {
  let n = stepCounter;
  const operationalSteps = {
    next(): number {
      n += 1;
      return n;
    },
    observeAtLeast(v: number): void {
      if (v > n) n = v;
    },
    current(): number {
      return n;
    },
  };
  const wrapper: FakeWrapper = {
    oeUrl: "http://localhost:8080",
    executionId: "exec-123",
    get stepCounter() {
      return n;
    },
    operationalSteps,
    nextOperationalStep(): number {
      return operationalSteps.next();
    },
    observeOperationalStep(v: number): void {
      operationalSteps.observeAtLeast(v);
    },
  };
  const innerLlm = {
    model: "gpt-4o",
    ...innerExtra,
  } as unknown as BaseChatModel;
  const wrapped = new SecureWrappedLLM(
    innerLlm,
    () => wrapper as never,
    "primary",
  );
  return { wrapper, wrapped };
}

beforeEach(() => {
  if (!process.env["RUNNER_MODE"]) process.env["RUNNER_MODE"] = "aer";
  proxyController.constructorArgs = [];
  proxyController.streamArgs = [];
  proxyController.streamChunks = [];
  proxyController.streamError = null;
  proxyController.response = null;
  proxyController.lastDurationMs = 0;
  proxyController.lastFromCache = false;
  proxyController.lastLatestStepNumber = null;
  vi.clearAllMocks();
});

describe("SecureWrappedLLM._generate — proxy routing", () => {
  test("routes messages, stop, and kwargs to the proxy and advances the step counter", async () => {
    // Python: test_generate_routes_through_proxy_with_stop_and_kwargs.
    proxyController.streamChunks = [{ content: "Hi" } as LLMStreamChunk];
    proxyController.response = new LLMResponse({ content: "Hi" });
    proxyController.lastDurationMs = 12;
    proxyController.lastLatestStepNumber = 4;
    const { wrapper, wrapped } = makeWrapped(0);

    const result = await wrapped._generate(
      [new HumanMessage({ content: "Hello" })],
      { stop: ["END"], max_tokens: 32 } as never,
    );

    expect(result.generations[0]?.message.content).toBe("Hi");
    expect(proxyController.constructorArgs[0]).toMatchObject({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
      modelName: "gpt-4o",
      boundTools: null,
      operationalSteps: wrapper.operationalSteps,
    });
    const [, step, stop, options] = proxyController.streamArgs[0] as [
      unknown,
      number,
      unknown,
      unknown,
    ];
    expect(step).toBe(1);
    expect(stop).toEqual(["END"]);
    expect(options).toBeInstanceOf(LLMInvocationOptions);
    // step counter is bumped to the proxy's reported latest step.
    expect(wrapper.stepCounter).toBe(4);
  });

  test("drops LangGraph runtime keys from the invocation options", async () => {
    // Durable replay determinism: LangGraph injects per-attempt runtime state
    // (checkpoint/task UUIDs, nodeFirstAttemptTime) into the model call
    // options. It must never reach the OE semantic input — a replay would
    // otherwise re-admit the LLM activity with a different hash and fail with
    // "workflow is nondeterministic".
    proxyController.streamChunks = [{ content: "Hi" } as LLMStreamChunk];
    proxyController.response = new LLMResponse({ content: "Hi" });
    const { wrapped } = makeWrapped();

    await wrapped._generate([new HumanMessage({ content: "Hello" })], {
      temperature: 0,
      executionInfo: {
        checkpointId: "1f1b1459-0de1-67d0-8004-a9669df44f96",
        checkpointNs: "model_request:03bda1ab-1a16-5e35-863b-7fe97d96e833",
        taskId: "03bda1ab-1a16-5e35-863b-7fe97d96e833",
        nodeFirstAttemptTime: 1789504776660,
      },
      durability: "sync",
      control: {},
    } as never);

    const [, , , options] = proxyController.streamArgs[0] as [
      unknown,
      number,
      unknown,
      LLMInvocationOptions,
    ];
    expect(options).toBeInstanceOf(LLMInvocationOptions);
    const serialized = JSON.stringify(options.toJSON());
    expect(serialized).not.toContain("executionInfo");
    expect(serialized).not.toContain("nodeFirstAttemptTime");
    expect(serialized).not.toContain("durability");
    expect(serialized).not.toContain("checkpointId");
    // Provider kwargs still flow through.
    expect(serialized).toContain('"temperature":0');
  });

  test("propagates PolicyDeniedException without recording metrics", async () => {
    // Python: test_generate_raises_on_policy_denial.
    proxyController.streamError = new PolicyDeniedException("Rate limited");
    const { wrapped } = makeWrapped();

    await expect(
      wrapped._generate([new HumanMessage({ content: "Hello" })], {} as never),
    ).rejects.toBeInstanceOf(PolicyDeniedException);
    expect(Metrics.recordLatency).not.toHaveBeenCalled();
  });

  test("records latency metric after a successful call", async () => {
    // Python: test_generate_records_latency_metric.
    proxyController.response = new LLMResponse({ content: "Hi" });
    proxyController.lastDurationMs = 150;
    const { wrapped } = makeWrapped();

    await wrapped._generate(
      [new HumanMessage({ content: "Hello" })],
      {} as never,
    );

    expect(Metrics.recordLatency).toHaveBeenCalledWith("llm_call", 150, {
      model_name: "gpt-4o",
    });
  });

  test("records latency + error metrics and rethrows on LLMInvocationError", async () => {
    // Python: test_generate_records_error_metric_on_llm_error.
    proxyController.streamError = new LLMInvocationError("boom");
    proxyController.lastDurationMs = 25;
    const { wrapped } = makeWrapped();

    await expect(
      wrapped._generate([new HumanMessage({ content: "Hello" })], {} as never),
    ).rejects.toThrow(/boom/);
    expect(Metrics.recordLatency).toHaveBeenCalledWith("llm_call", 25, {
      model_name: "gpt-4o",
    });
    expect(Metrics.recordError).toHaveBeenCalledWith("llm_call", {
      model_name: "gpt-4o",
    });
  });

  test("logs cached results served by the proxy", async () => {
    // Python: test_generate_logs_cached_results_from_proxy.
    proxyController.response = new LLMResponse({ content: "Cached" });
    proxyController.lastFromCache = true;
    const { wrapped } = makeWrapped();

    await wrapped._generate(
      [new HumanMessage({ content: "Hello" })],
      {} as never,
    );

    expect(logCachedResult).toHaveBeenCalledWith("invoke_llm", 1, "LLM");
  });

  test("repairs missing tool messages before sending to the proxy", async () => {
    // Python: test_generate_repairs_missing_tool_messages_before_proxy.
    proxyController.response = new LLMResponse({ content: "Recovered" });
    const { wrapped } = makeWrapped();

    await wrapped._generate(
      [
        new HumanMessage({ content: "Check Sentry" }),
        new AIMessage({
          content: "",
          tool_calls: [
            {
              id: "call_sentry",
              name: "sentry__search_issues",
              args: { query: "project:agentengine-cli" },
            },
          ],
        }),
        new HumanMessage({ content: "Now use Glean" }),
      ],
      {} as never,
    );

    const sent = proxyController.streamArgs[0]?.[0] as Array<{
      role: string;
      toolCallId?: string;
      name?: string;
      content: string;
    }>;
    expect(sent.map((m) => m.role)).toEqual([
      "user",
      "assistant",
      "tool",
      "user",
    ]);
    expect(sent[2]?.toolCallId).toBe("call_sentry");
    expect(sent[2]?.name).toBe("sentry__search_issues");
  });

  test("preserves existing tool messages (no spurious repair)", async () => {
    // Python: test_generate_preserves_existing_tool_messages.
    proxyController.response = new LLMResponse({ content: "Done" });
    const { wrapped } = makeWrapped();

    await wrapped._generate(
      [
        new AIMessage({
          content: "",
          tool_calls: [
            { id: "call_search", name: "search", args: { query: "hello" } },
          ],
        }),
        new ToolMessage({
          content: "existing result",
          tool_call_id: "call_search",
          name: "search",
        }),
        new HumanMessage({ content: "continue" }),
      ],
      {} as never,
    );

    const sent = proxyController.streamArgs[0]?.[0] as Array<{
      content: string;
    }>;
    expect(sent).toHaveLength(3);
    expect(sent[1]?.content).toBe("existing result");
  });
});

describe("SecureWrappedLLM._streamResponseChunks — proxy routing", () => {
  test("streams through the proxy, yields converted chunks, and records latency", async () => {
    // Python: test_stream_* (same shape as generate).
    proxyController.streamChunks = [
      { content: "Hel" } as LLMStreamChunk,
      { content: "lo" } as LLMStreamChunk,
    ];
    proxyController.lastDurationMs = 5;
    proxyController.lastLatestStepNumber = 3;
    // Inner LLM must advertise streaming or the wrapper delegates to _generate.
    const { wrapper, wrapped } = makeWrapped(0, { stream: () => undefined });

    const chunks = [];
    for await (const chunk of wrapped._streamResponseChunks(
      [new HumanMessage({ content: "Hi" })],
      {} as never,
    )) {
      chunks.push(chunk);
    }

    expect(proxyController.streamArgs).toHaveLength(1);
    expect(chunks).toHaveLength(2);
    expect(wrapper.stepCounter).toBe(3);
    expect(Metrics.recordLatency).toHaveBeenCalledWith("llm_call", 5, {
      model_name: "gpt-4o",
    });
  });

  test("propagates PolicyDeniedException from the stream", async () => {
    proxyController.streamError = new PolicyDeniedException("Denied");
    const { wrapped } = makeWrapped(0, { stream: () => undefined });

    await expect(
      (async () => {
        for await (const _ of wrapped._streamResponseChunks(
          [new HumanMessage({ content: "Hi" })],
          {} as never,
        )) {
          // drain
        }
      })(),
    ).rejects.toBeInstanceOf(PolicyDeniedException);
  });
});
