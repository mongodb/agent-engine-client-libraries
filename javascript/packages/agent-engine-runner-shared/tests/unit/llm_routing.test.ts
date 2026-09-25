/**
 * Unit tests for LLM routing — retry, caching, invoke handler.
 *
 * Mirrors the last 4 classes of Python's tests/unit/test_llm_routing.py:
 *   - TestStreamRetryDuringIteration (6 tests)
 *   - TestStreamLLMChunks (3 tests)
 *   - TestLLMCaching (6 tests)
 *   - TestHandleInvokeLLM (3 tests)
 *
 * Schemas + extraction + stream collection live in `llm_routing_models.test.ts`.
 *
 * TS-vs-Python adaptations:
 *   - Python `patch("asyncio.sleep", new_callable=AsyncMock)` makes retry
 *     backoff instant. TS uses `setTimeout(resolve, backoff * 1000)`. We
 *     instead set `LLM_INITIAL_BACKOFF=0` + `LLM_BACKOFF_MULTIPLIER=1` via
 *     `vi.hoisted` BEFORE any source import, so module-load retry constants
 *     come out as zero-delay. Same effect as patching sleep.
 *   - Python `patch.object(server, "_create_llm_for_pod", return_value=...)`
 *     → TS spies via cast-through-unknown on the private method.
 *   - Python `_make_tool_server()` uses `__new__` to skip `__init__`. TS
 *     uses `Object.create(ToolServer.prototype)` with manual field setup.
 *   - Python `mock_adapter.astream = fake_astream` (LangChain) → TS uses
 *     `.stream(messages, kwargs)` (agent-engine-sdk BaseLLM interface).
 *   - SSE parsing: `_parseSseEvents` walks the yielded `data: {...}\n\n` strings.
 */

import { vi } from "vitest";

// Hoisted env stubs — must run BEFORE source imports so module-load
// constants (LLM_INITIAL_BACKOFF etc.) pick up zero-delay values.
vi.hoisted(() => {
  process.env["LLM_INITIAL_BACKOFF"] = "0";
  process.env["LLM_BACKOFF_MULTIPLIER"] = "1";
});

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import type { LLMStreamChunk } from "@mongodb-js/agent-engine-sdk";
import { LLMTokenUsage } from "@mongodb-js/agent-engine-sdk";
import {
  ToolServer,
  getCurrentTraceId,
  resetHooks,
  type LLMPodInvokeRequest,
  type LLMPodInvokeResponse,
} from "../../src/index.js";
import { AsyncMutex } from "../../src/server/tool.js";

// ---------------------------------------------------------------------------
// Test scaffolding
// ---------------------------------------------------------------------------

interface MockRuntime {
  tools: Record<string, unknown>;
  toolDefinitions: Record<string, Record<string, unknown>>;
  graphBuilder: unknown;
  agent_config: { featureEnabled: () => boolean };
}

interface ToolServerPrivates {
  runtime: MockRuntime;
  executeGate: AsyncMutex;
  restrictionDisabled: boolean;
  llmRegistryLock: AsyncMutex;
  llmRegistryLoaded: boolean;
  createLlmForPod: (llmId: string, tools?: unknown[]) => unknown;
  streamLlmChunks: (
    request: LLMPodInvokeRequest,
  ) => AsyncGenerator<LLMStreamChunk>;
  handleInvokeLlmStream: (
    request: LLMPodInvokeRequest,
  ) => AsyncGenerator<string>;
  doHandleInvokeLlm: (
    request: LLMPodInvokeRequest,
  ) => Promise<LLMPodInvokeResponse>;
}

function makeToolServer(): ToolServerPrivates {
  const server = Object.create(ToolServer.prototype) as ToolServerPrivates;
  server.runtime = {
    tools: {},
    toolDefinitions: {},
    graphBuilder: null,
    agent_config: { featureEnabled: () => false },
  };
  // Object.create skips field initializers, so seed the gate the LLM
  // handlers acquire, the restriction-mode flag, and
  // the lazy entrypoint-load state (ensureLlmRegistryLoaded()); mark it
  // already-loaded so these LLM-routing tests don't also need a working
  // graphBuilder.
  server.executeGate = new AsyncMutex();
  server.restrictionDisabled = false;
  server.llmRegistryLock = new AsyncMutex();
  server.llmRegistryLoaded = true;
  return server;
}

function makeRequest(
  overrides: Partial<LLMPodInvokeRequest["arguments"]> = {},
): LLMPodInvokeRequest {
  return {
    execution_id: "exec-1",
    arguments: {
      model: "test-model",
      messages: [{ role: "user", content: "Hi" }],
      llm_id: "__default__",
      ...overrides,
    },
  } as LLMPodInvokeRequest;
}

function parseSseEvents(events: string[]): Record<string, unknown>[] {
  return events
    .filter((e) => e.startsWith("data: "))
    .map(
      (e) =>
        JSON.parse(e.slice("data: ".length).trim()) as Record<string, unknown>,
    );
}

/**
 * Build an async-stream factory whose successive calls execute the next
 * generator factory in `factories`. The Python tests mutate `call_count`
 * inside a single `fake_astream`; TS uses one factory per attempt for
 * clearer assertions about how many times `stream` was invoked.
 */
function makeAdapter(
  ...factories: Array<() => AsyncGenerator<LLMStreamChunk>>
): {
  adapter: {
    stream: (msgs: unknown, kwargs?: unknown) => AsyncIterable<LLMStreamChunk>;
  };
  callCount: () => number;
} {
  let callCount = 0;
  return {
    adapter: {
      stream: () => {
        const factory = factories[Math.min(callCount, factories.length - 1)];
        callCount++;
        return factory?.() ?? asyncGenFromArray<LLMStreamChunk>([]);
      },
    },
    callCount: () => callCount,
  };
}

function spyCreateLlmForPod(
  server: ToolServerPrivates,
  adapter: ReturnType<typeof makeAdapter>["adapter"],
): void {
  server.createLlmForPod = () => adapter;
}

async function* asyncGenFromArray<T>(items: T[]): AsyncGenerator<T> {
  for (const item of items) yield item;
}

// A stream factory whose generator throws on first iteration without ever
// yielding — models an adapter attempt that fails before producing a chunk.
function throwingStream(error: Error): () => AsyncGenerator<LLMStreamChunk> {
  return async function* (): AsyncGenerator<LLMStreamChunk> {
    yield* asyncGenFromArray<LLMStreamChunk>([]);
    throw error;
  };
}

beforeEach(() => {
  resetHooks();
});

afterEach(() => {
  resetHooks();
});

// ---------------------------------------------------------------------------
// TestStreamRetryDuringIteration — handleInvokeLlmStream retry behavior
// ---------------------------------------------------------------------------

describe("handleInvokeLlmStream retry behavior", () => {
  test("retries on rate-limit error during iteration; success after 1 retry", async () => {
    // test_stream_retries_on_429_during_iteration
    const server = makeToolServer();
    const { adapter, callCount } = makeAdapter(
      // First attempt: throw a retryable error
      throwingStream(new Error("429 Too Many Requests")),
      // Second attempt: yield content
      () => asyncGenFromArray([{ content: "Hello" } as LLMStreamChunk]),
    );
    spyCreateLlmForPod(server, adapter);

    const events: string[] = [];
    for await (const e of server.handleInvokeLlmStream(makeRequest()))
      events.push(e);

    const parsed = parseSseEvents(events);
    expect(parsed.some((e) => e["content"] === "Hello")).toBe(true);
    expect(parsed.some((e) => e["done"] === true)).toBe(true);
    expect(parsed.some((e) => "error" in e)).toBe(false);
    expect(callCount()).toBe(2);
  });

  test("no retry after the first chunk is yielded; error event emitted", async () => {
    // test_stream_no_retry_after_chunks_yielded
    const server = makeToolServer();
    const { adapter, callCount } = makeAdapter(async function* () {
      yield { content: "partial" } as LLMStreamChunk;
      throw new Error("429 Too Many Requests");
    });
    spyCreateLlmForPod(server, adapter);

    const events: string[] = [];
    for await (const e of server.handleInvokeLlmStream(makeRequest()))
      events.push(e);

    const parsed = parseSseEvents(events);
    expect(parsed.some((e) => e["content"] === "partial")).toBe(true);
    expect(parsed.some((e) => "error" in e)).toBe(true);
    expect(parsed.some((e) => e["done"] === true)).toBe(false);
    expect(callCount()).toBe(1);
  });

  test("chunk with content and usage emits both the content event and a final usage event", async () => {
    // test_stream_chunk_with_content_and_usage_keeps_both
    const server = makeToolServer();
    const traceId = "0123456789abcdef0123456789abcdef";
    const seen: (string | null)[] = [];
    const { adapter } = makeAdapter(async function* () {
      seen.push(getCurrentTraceId());
      yield* asyncGenFromArray([
        {
          content: "Hello",
          usage: new LLMTokenUsage({
            input_tokens: 3,
            output_tokens: 2,
            total_tokens: 5,
          }),
        } as LLMStreamChunk,
      ]);
    });
    spyCreateLlmForPod(server, adapter);

    const events: string[] = [];
    for await (const e of server.handleInvokeLlmStream({
      ...makeRequest(),
      platform_trace_id: traceId,
    }))
      events.push(e);

    expect(seen).toEqual([traceId]);
    expect(getCurrentTraceId()).toBeNull();
    const parsed = parseSseEvents(events);
    expect(parsed.some((e) => e["content"] === "Hello")).toBe(true);
    const doneEvt = parsed.find((e) => e["done"] === true);
    expect(doneEvt).toBeDefined();
    expect(
      (doneEvt?.["usage"] as Record<string, unknown>)["total_tokens"],
    ).toBe(5);
  });

  test("merges split input and output usage onto the done event", async () => {
    const server = makeToolServer();
    const { adapter } = makeAdapter(() =>
      asyncGenFromArray([
        {
          content: "Hello",
          usage: new LLMTokenUsage({ input_tokens: 1048 }),
        } as LLMStreamChunk,
        {
          usage: new LLMTokenUsage({ output_tokens: 1222 }),
        } as LLMStreamChunk,
      ]),
    );
    spyCreateLlmForPod(server, adapter);

    const events: string[] = [];
    for await (const e of server.handleInvokeLlmStream(makeRequest()))
      events.push(e);

    const parsed = parseSseEvents(events);
    const doneEvt = parsed.find((e) => e["done"] === true);
    const usage = doneEvt?.["usage"] as Record<string, unknown>;
    expect(usage["input_tokens"]).toBe(1048);
    expect(usage["output_tokens"]).toBe(1222);
    expect(usage["total_tokens"]).toBe(2270);
  });

  test("preserves an explicit total_tokens on the done event", async () => {
    const server = makeToolServer();
    const { adapter } = makeAdapter(() =>
      asyncGenFromArray([
        {
          content: "Hello",
          usage: new LLMTokenUsage({
            input_tokens: 1,
            output_tokens: 1,
            total_tokens: 99,
          }),
        } as LLMStreamChunk,
      ]),
    );
    spyCreateLlmForPod(server, adapter);

    const events: string[] = [];
    for await (const e of server.handleInvokeLlmStream(makeRequest()))
      events.push(e);

    const parsed = parseSseEvents(events);
    const doneEvt = parsed.find((e) => e["done"] === true);
    const usage = doneEvt?.["usage"] as Record<string, unknown>;
    expect(usage["input_tokens"]).toBe(1);
    expect(usage["output_tokens"]).toBe(1);
    expect(usage["total_tokens"]).toBe(99);
  });

  test("done event keeps the adapter cumulative snapshot", async () => {
    const server = makeToolServer();
    const { adapter } = makeAdapter(() =>
      asyncGenFromArray([
        {
          content: "Hel",
          usage: new LLMTokenUsage({
            input_tokens: 18,
            output_tokens: 1,
            total_tokens: 19,
          }),
        } as LLMStreamChunk,
        {
          content: "lo",
          usage: new LLMTokenUsage({
            input_tokens: 18,
            output_tokens: 5,
            total_tokens: 23,
          }),
        } as LLMStreamChunk,
      ]),
    );
    spyCreateLlmForPod(server, adapter);

    const events: string[] = [];
    for await (const e of server.handleInvokeLlmStream(makeRequest()))
      events.push(e);

    const parsed = parseSseEvents(events);
    const doneEvt = parsed.find((e) => e["done"] === true);
    const usage = doneEvt?.["usage"] as Record<string, unknown>;
    expect(usage["input_tokens"]).toBe(18);
    expect(usage["output_tokens"]).toBe(5);
    expect(usage["total_tokens"]).toBe(23);
  });

  test("all retries exhausted emits an error event", async () => {
    // test_stream_exhausts_retries
    const server = makeToolServer();
    const { adapter, callCount } = makeAdapter(
      throwingStream(new Error("429 Too Many Requests")),
    );
    spyCreateLlmForPod(server, adapter);

    const events: string[] = [];
    for await (const e of server.handleInvokeLlmStream(makeRequest()))
      events.push(e);

    const parsed = parseSseEvents(events);
    expect(parsed.some((e) => "error" in e)).toBe(true);
    expect(callCount()).toBeGreaterThanOrEqual(2);
  });

  test("adapter camelCase chunk objects are normalized into sdk-core SSE payloads", async () => {
    // test_stream_normalizes_langchain_style_chunk_objects
    // The adapter (agent-engine-sdk-langgraph-ts) yields camelCase LLMStreamChunk
    // dicts; `coerceStreamChunk` reads camelCase only. (Python feeds raw
    // snake_case LangChain objects; the TS adapter pre-translates.)
    const server = makeToolServer();
    // Note: `usage: null` would break LLMPodStreamEventSchema.parse (null is
    // not undefined in Zod); the Python None is mapped to "field omitted" here.
    const rawChunk = {
      content: "Hello",
      toolCalls: [
        {
          id: "call-1",
          name: "lookup",
          args: '{"q":"weather"}',
          type: "tool_call_chunk",
          index: 0,
        },
      ],
      id: "msg-1",
      name: "assistant",
      responseMetadata: { finish_reason: "tool_calls" },
      additionalKwargs: { provider: "test" },
    };
    const { adapter } = makeAdapter(() =>
      asyncGenFromArray([rawChunk as unknown as LLMStreamChunk]),
    );
    spyCreateLlmForPod(server, adapter);

    const events: string[] = [];
    for await (const e of server.handleInvokeLlmStream(makeRequest()))
      events.push(e);

    const parsed = parseSseEvents(events);
    expect(parsed[0]?.["content"]).toBe("Hello");
    expect(parsed[0]?.["tool_call_chunks"]).toEqual([
      {
        id: "call-1",
        name: "lookup",
        args: '{"q":"weather"}',
        type: "tool_call_chunk",
        index: 0,
      },
    ]);
    expect(parsed[0]?.["id"]).toBe("msg-1");
    expect(parsed[0]?.["name"]).toBe("assistant");
    expect(parsed[0]?.["response_metadata"]).toEqual({
      finish_reason: "tool_calls",
    });
    expect(parsed[0]?.["additional_kwargs"]).toEqual({ provider: "test" });
    expect(parsed[1]?.["done"]).toBe(true);
  });

  test("adapter tool-call dict args are JSON-encoded before SSE emission", async () => {
    // test_stream_normalizes_langchain_tool_call_chunk_dict_args
    const server = makeToolServer();
    const rawChunk = {
      content: "",
      toolCalls: [
        {
          id: "call-1",
          name: "lookup",
          args: { q: "weather" },
          type: "tool_call_chunk",
          index: 0,
        },
      ],
      // Python's `None` for these optional fields → omit in TS (null != undefined).
    };
    const { adapter } = makeAdapter(() =>
      asyncGenFromArray([rawChunk as unknown as LLMStreamChunk]),
    );
    spyCreateLlmForPod(server, adapter);

    const events: string[] = [];
    for await (const e of server.handleInvokeLlmStream(makeRequest()))
      events.push(e);

    const parsed = parseSseEvents(events);
    expect(parsed[0]?.["tool_call_chunks"]).toEqual([
      {
        id: "call-1",
        name: "lookup",
        args: JSON.stringify({ q: "weather" }),
        type: "tool_call_chunk",
        index: 0,
      },
    ]);
  });
});

// ---------------------------------------------------------------------------
// TestStreamLLMChunks — direct streamLlmChunks retry
// ---------------------------------------------------------------------------

describe("streamLlmChunks retry behavior", () => {
  test("retry succeeds on the second attempt", async () => {
    // test_retry_succeeds_on_second_attempt
    const server = makeToolServer();
    const { adapter, callCount } = makeAdapter(
      throwingStream(new Error("429 Too Many Requests")),
      () => asyncGenFromArray([{ content: "retry success" } as LLMStreamChunk]),
    );
    spyCreateLlmForPod(server, adapter);

    const chunks: LLMStreamChunk[] = [];
    for await (const c of server.streamLlmChunks(makeRequest())) chunks.push(c);

    expect(chunks.map((c) => c.content)).toEqual(["retry success"]);
    expect(callCount()).toBe(2);
  });

  test("all retries exhausted raises the last error", async () => {
    // test_all_retries_exhausted_raises_last_error
    const server = makeToolServer();
    const { adapter, callCount } = makeAdapter(
      throwingStream(new Error("429 Too Many Requests")),
    );
    spyCreateLlmForPod(server, adapter);

    await expect(async () => {
      for await (const _ of server.streamLlmChunks(makeRequest())) {
        /* consume */
      }
    }).rejects.toThrow(/429 Too Many Requests/);

    expect(callCount()).toBeGreaterThanOrEqual(2);
  });

  test("no retry after the first chunk is yielded", async () => {
    // test_no_retry_after_first_chunk_yielded
    const server = makeToolServer();
    const { adapter, callCount } = makeAdapter(async function* () {
      yield { content: "partial" } as LLMStreamChunk;
      throw new Error("429 Too Many Requests");
    });
    spyCreateLlmForPod(server, adapter);

    const chunks: LLMStreamChunk[] = [];
    await expect(async () => {
      for await (const c of server.streamLlmChunks(makeRequest()))
        chunks.push(c);
    }).rejects.toThrow(/429 Too Many Requests/);

    expect(chunks.map((c) => c.content)).toEqual(["partial"]);
    expect(callCount()).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// TestHandleInvokeLLM — doHandleInvokeLlm (non-streaming invoke)
// ---------------------------------------------------------------------------

describe("ToolServer.doHandleInvokeLlm", () => {
  test("successful invoke returns content, tool_calls, and usage", async () => {
    // test_success_returns_content_and_usage
    const server = makeToolServer();
    const traceId = "0123456789abcdef0123456789abcdef";
    const seen: (string | null)[] = [];
    const { adapter } = makeAdapter(async function* () {
      seen.push(getCurrentTraceId());
      yield* asyncGenFromArray([
        { content: "Hello!" } as LLMStreamChunk,
        {
          toolCalls: [
            { id: "tc1", name: "search", args: '{"q": "test"}', index: 0 },
          ],
        } as LLMStreamChunk,
        {
          usage: new LLMTokenUsage({
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
          }),
        } as LLMStreamChunk,
      ]);
    });
    spyCreateLlmForPod(server, adapter);

    const response = await server.doHandleInvokeLlm({
      ...makeRequest(),
      platform_trace_id: traceId,
    });

    expect(seen).toEqual([traceId]);
    expect(getCurrentTraceId()).toBeNull();
    expect(response.status).toBe("success");
    const result = response.result as Record<string, unknown>;
    expect(result["content"]).toBe("Hello!");
    const toolCalls = result["tool_calls"] as Record<string, unknown>[];
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0]?.["name"]).toBe("search");
    expect(response.usage?.totalTokens).toBe(15);
    expect(response.pod_name).toBeDefined();
    expect(response.duration_ms ?? 0).toBeGreaterThanOrEqual(0);
  });

  test("retries on retryable error and succeeds on the second attempt", async () => {
    // test_retry_on_rate_limit
    const server = makeToolServer();
    const { adapter, callCount } = makeAdapter(
      // "429" matches TS isRetryableError patterns (Python tests patch is_retryable_error;
      // TS can't patch intra-module imports, so we include a retryable token in the message).
      throwingStream(new Error("429: Rate limit exceeded")),
      () => asyncGenFromArray([{ content: "Retry success" } as LLMStreamChunk]),
    );
    spyCreateLlmForPod(server, adapter);

    const response = await server.doHandleInvokeLlm(makeRequest());

    expect(response.status).toBe("success");
    expect((response.result as Record<string, unknown>)["content"]).toBe(
      "Retry success",
    );
    expect(callCount()).toBe(2);
  });

  test("returns an error response when all retries are exhausted", async () => {
    // test_all_retries_exhausted
    const server = makeToolServer();
    const { adapter } = makeAdapter(
      // "429" matches TS isRetryableError patterns (Python tests patch is_retryable_error;
      // TS can't patch intra-module imports, so we include a retryable token in the message).
      throwingStream(new Error("429: Rate limit exceeded")),
    );
    spyCreateLlmForPod(server, adapter);

    const response = await server.doHandleInvokeLlm(makeRequest());

    expect(response.status).toBe("error");
    expect(response.error).toMatch(/Rate limit/);
    expect(response.pod_name).toBeDefined();
    expect(response.duration_ms ?? 0).toBeGreaterThanOrEqual(0);
  });
});
