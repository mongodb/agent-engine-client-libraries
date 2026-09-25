/**
 * Unit tests for SecureLLMProxy's OE-owned invoke_llm contract.
 *
 * Mirrors Python's tests/unit/test_secure_llm_proxy.py.
 *
 * TS-vs-Python adaptations:
 *   - Python `patch("agent_engine_runner_shared.secure_llm_proxy.request_oe_approval")`
 *     and `patch("...connect_sse")` → TS mocks the `fetch` global, routing
 *     responses by URL ('/tool/execute' vs the routed SSE endpoint).
 *   - SSE streaming uses a `ReadableStream` body on a real `Response`,
 *     since TS consumes via `response.body.getReader()`.
 *   - `proxy.invoke(messages, stop=..., options=...)` (kwargs in Python) →
 *     `proxy.invoke(messages, step?, stop?, options?)` (positional in TS).
 *   - `invoke_llm` arguments use `stop_sequences` on the wire schema but
 *     `serializeInvokeLLMRequestArguments` renames to `stop` before send.
 *     Tests assert on the post-rename `body.arguments.stop`.
 */

import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import {
  LLMInvocationOptions,
  type Message,
} from "@mongodb-js/agent-engine-sdk";
import {
  SecureLLMProxy,
  PolicyDeniedException,
  LLMInvocationError,
  TerminalExecutionError,
  ToolExecutionError,
} from "../../src/index.js";
import {
  OE_DISPATCH_TAKEOVER_RETRY_DELAY_MS,
  oeStreamRetry,
} from "../../src/utils.js";

// ---------------------------------------------------------------------------
// fetch mock helpers
// ---------------------------------------------------------------------------

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function sseResponse(events: unknown[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const event of events) {
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify(event)}\n\n`),
        );
      }
      controller.close();
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

/** Truncated SSE: closes the stream before the done event. */
function truncatedSseResponse(): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(
        encoder.encode(`data: ${JSON.stringify({ content: "partial" })}\n\n`),
      );
      controller.close();
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

function emptyTruncatedSseResponse(): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.close();
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

function disconnectingSseResponse(events: unknown[]): Response {
  const encoder = new TextEncoder();
  let i = 0;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i < events.length) {
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify(events[i++])}\n\n`),
        );
        return;
      }
      controller.error(new TypeError("fetch failed"));
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

interface CapturedFetchCall {
  url: string;
  body: Record<string, unknown> | null;
}

function lastFetchCall(spy: ReturnType<typeof vi.fn>): CapturedFetchCall {
  const calls = spy.mock.calls;
  const [url, init] = calls[calls.length - 1];
  const body = (init as RequestInit | undefined)?.body;
  return {
    url: url as string,
    body: typeof body === "string" ? JSON.parse(body) : null,
  };
}

beforeEach(() => {
  vi.unstubAllGlobals();
  vi.spyOn(oeStreamRetry, "sleep").mockResolvedValue(undefined);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const baseMessage: Message = { role: "user", content: "Hello" };

// ---------------------------------------------------------------------------
// TestSecureLLMProxyConstruction
// ---------------------------------------------------------------------------

describe("SecureLLMProxy construction", () => {
  test("stores oeUrl, executionId; stepCounter starts at 0", () => {
    // test_construction
    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });
    expect(proxy.oeUrl).toBe("http://localhost:8080");
    expect(proxy.executionId).toBe("exec-123");
    expect(proxy.stepCounter).toBe(0);
  });

  test("strips trailing slash from oeUrl", () => {
    // test_strips_trailing_slash
    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080/",
      executionId: "exec-123",
      llmId: "primary",
    });
    expect(proxy.oeUrl).toBe("http://localhost:8080");
  });
});

// ---------------------------------------------------------------------------
// TestSecureLLMProxyInvoke
// ---------------------------------------------------------------------------

describe("SecureLLMProxy.invoke", () => {
  test("requests full OE execution payload (model, temp, stop, tools, options, stream, messages)", async () => {
    // test_invoke_requests_full_oe_execution_payload
    const fetchSpy = vi.fn().mockResolvedValue(
      jsonResponse({
        proceed: true,
        status: "success",
        result: { content: "Hi", tool_calls: [], metadata: {} },
        duration_ms: 12.0,
        latest_step_number: 4,
        pod_name: "tool-pod-1",
      }),
    );
    vi.stubGlobal("fetch", fetchSpy);

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
      modelName: "gpt-4o",
      boundTools: [{ name: "lookup", parameters: { type: "object" } }],
    });

    const result = await proxy.invoke(
      [baseMessage],
      null,
      ["END"],
      new LLMInvocationOptions({ max_tokens: 64 }),
    );

    expect(result.content).toBe("Hi");
    expect(proxy.stepCounter).toBe(4);
    expect(proxy.lastDurationMs).toBe(12.0);
    expect(proxy.lastFromCache).toBe(false);
    expect(proxy.lastPodName).toBe("tool-pod-1");

    const { url, body } = lastFetchCall(fetchSpy);
    expect(url).toBe("http://localhost:8080/tool/execute");
    expect(body?.["tool_name"]).toBe("invoke_llm");
    expect(body).not.toHaveProperty("custom_headers");
    const args = body?.["arguments"] as Record<string, unknown>;
    expect(args["model"]).toBe("gpt-4o");
    expect(args["llm_id"]).toBe("primary");
    expect(args["stop"]).toEqual(["END"]);
    expect(args["stream"]).toBe(true);
    expect(args["messages"]).toEqual([{ role: "user", content: "Hello" }]);
    expect(args["tools"]).toEqual([
      { name: "lookup", parameters: { type: "object" } },
    ]);
    expect(args["options"]).toEqual({ max_tokens: 64 });
    expect(args).not.toHaveProperty("custom_headers");
  });

  test("bound tool_choice rides on the invoke_llm arguments", async () => {
    const fetchSpy = vi.fn().mockResolvedValue(
      jsonResponse({
        proceed: true,
        status: "success",
        result: { content: "Hi", tool_calls: [], metadata: {} },
      }),
    );
    vi.stubGlobal("fetch", fetchSpy);

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
      modelName: "gpt-4o",
      boundTools: [{ name: "Brief", parameters: { type: "object" } }],
      boundToolChoice: "Brief",
    });

    await proxy.invoke([baseMessage], null, null, null);

    const { body } = lastFetchCall(fetchSpy);
    const args = body?.["arguments"] as Record<string, unknown>;
    expect(args["tool_choice"]).toBe("Brief");
  });

  test("omits tool_choice from the wire when no choice is bound", async () => {
    const fetchSpy = vi.fn().mockResolvedValue(
      jsonResponse({
        proceed: true,
        status: "success",
        result: { content: "Hi", tool_calls: [], metadata: {} },
      }),
    );
    vi.stubGlobal("fetch", fetchSpy);

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
      modelName: "gpt-4o",
    });

    await proxy.invoke([baseMessage], null, null, null);

    const { body } = lastFetchCall(fetchSpy);
    const args = body?.["arguments"] as Record<string, unknown>;
    expect(args).not.toHaveProperty("tool_choice");
  });

  test("preserves tool_choice=false on the wire", async () => {
    const fetchSpy = vi.fn().mockResolvedValue(
      jsonResponse({
        proceed: true,
        status: "success",
        result: { content: "Hi", tool_calls: [], metadata: {} },
      }),
    );
    vi.stubGlobal("fetch", fetchSpy);

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
      modelName: "gpt-4o",
      boundToolChoice: false,
    });

    await proxy.invoke([baseMessage], null, null, null);

    const { body } = lastFetchCall(fetchSpy);
    const args = body?.["arguments"] as Record<string, unknown>;
    expect(args["tool_choice"]).toBe(false);
  });

  test("rejects a non-JSON-serializable tool_choice instead of shipping it", async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
      modelName: "gpt-4o",
      // A function is not a JsonValue — it would be silently dropped by
      // JSON.stringify. Fail fast at the boundary instead.
      boundToolChoice: () => "nope",
    });

    await expect(proxy.invoke([baseMessage], null, null, null)).rejects.toThrow(
      /Invalid tool_choice/,
    );
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  test("dumps camelCase tool fields to snake_case on the wire", async () => {
    // Regression: internal Messages are camelCase (toolCallId/toolCalls). The
    // request must dump them to snake_case tool_call_id/tool_calls. Previously
    // buildInvokeRequest re-parsed the request through MessageSchema (the
    // snake→camel "parse" transform), which blanked these fields on
    // already-internal messages and made multi-turn tool calling loop until the
    // recursion limit. See serializeMessage / buildInvokeRequest.
    const fetchSpy = vi.fn().mockResolvedValue(
      jsonResponse({
        proceed: true,
        status: "success",
        result: { content: "done", tool_calls: [], metadata: {} },
      }),
    );
    vi.stubGlobal("fetch", fetchSpy);

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
      modelName: "gpt-4o",
    });

    const conversation: Message[] = [
      { role: "user", content: "What is the weather?" },
      {
        role: "assistant",
        content: "",
        toolCalls: [
          { id: "call_abc", name: "get_weather", args: { city: "NYC" } },
        ] as unknown as Message["toolCalls"],
      },
      { role: "tool", content: "sunny", toolCallId: "call_abc" },
    ];

    await proxy.invoke(conversation, null, null, null);

    const { body } = lastFetchCall(fetchSpy);
    const args = body?.["arguments"] as Record<string, unknown>;
    const messages = args["messages"] as Array<Record<string, unknown>>;

    const toolMsg = messages.find((m) => m["role"] === "tool");
    expect(toolMsg["tool_call_id"]).toBe("call_abc");
    expect(toolMsg).not.toHaveProperty("toolCallId");

    const aiMsg = messages.find((m) => m["role"] === "assistant");
    expect(aiMsg["tool_calls"]).toEqual([
      { id: "call_abc", name: "get_weather", args: { city: "NYC" } },
    ]);
    expect(aiMsg).not.toHaveProperty("toolCalls");
  });

  test("returns cached result via the `result` field with from_cache=true", async () => {
    // test_invoke_returns_cached_result
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          proceed: true,
          status: "success",
          result: { content: "Cached", tool_calls: null, metadata: {} },
          from_cache: true,
          duration_ms: 5.0,
        }),
      ),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });
    const result = await proxy.invoke([baseMessage]);

    expect(result.content).toBe("Cached");
    expect(proxy.lastFromCache).toBe(true);
    expect(proxy.lastDurationMs).toBe(5.0);
  });

  test("accepts the legacy `cached_result` shape", async () => {
    // test_invoke_accepts_legacy_cached_result_shape
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          proceed: true,
          cached_result: { content: "Cached", tool_calls: null, metadata: {} },
          from_cache: true,
          duration_ms: 7.0,
        }),
      ),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });
    const result = await proxy.invoke([baseMessage]);

    expect(result.content).toBe("Cached");
    expect(proxy.lastFromCache).toBe(true);
  });

  test("raises PolicyDeniedException on proceed=false", async () => {
    // test_invoke_raises_on_policy_denied
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          jsonResponse({ proceed: false, reason: "Rate limited" }),
        ),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });

    await expect(proxy.invoke([baseMessage])).rejects.toBeInstanceOf(
      PolicyDeniedException,
    );
  });

  test("a terminal execution rejection is not a policy denial", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          jsonResponse({ proceed: false, reason: "execution already error" }),
        ),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });

    const err = await proxy.invoke([baseMessage]).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(TerminalExecutionError);
    expect(err).toBeInstanceOf(ToolExecutionError);
    expect(err).not.toBeInstanceOf(PolicyDeniedException);
    expect((err as Error).message).not.toContain("Policy denied");
  });

  test("raises LLMInvocationError on status=error", async () => {
    // test_invoke_raises_on_oe_owned_error
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          proceed: true,
          status: "error",
          error: "tool pod failed",
          duration_ms: 9.0,
        }),
      ),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });

    const err = await proxy.invoke([baseMessage]).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(LLMInvocationError);
    expect(err).toMatchObject({ source: "llm" });
    expect(String(err)).toMatch(/tool pod failed/);
  });

  test("retries OE retryable error then succeeds", async () => {
    // test_invoke_retries_oe_retryable_error_then_succeeds
    const fetchSpy = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          proceed: true,
          status: "error",
          error:
            "tool pod is no longer reserved for this session; retry request",
          retryable: true,
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          proceed: true,
          status: "success",
          result: { content: "Hi", tool_calls: [] },
        }),
      );
    vi.stubGlobal("fetch", fetchSpy);

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });

    const result = await proxy.invoke([baseMessage]);
    expect(result.content).toBe("Hi");
    expect(fetchSpy).toHaveBeenCalledTimes(2);
  });

  test("yields an interrupted marker on inline/replayed interrupted status", async () => {
    // A non-streaming or durably-replayed interrupted step must yield the marker
    // rather than throwing "Unexpected OE invoke_llm status".
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          proceed: true,
          status: "interrupted",
          duration_ms: 5.0,
        }),
      ),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });

    const chunks = [];
    for await (const c of proxy.stream([baseMessage])) chunks.push(c);

    expect(chunks.at(-1)?.responseMetadata).toEqual({ interrupted: true });
  });

  test("normalizes usage into metadata on the returned LLMResponse", async () => {
    // test_invoke_normalizes_usage_into_metadata
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          proceed: true,
          status: "success",
          result: {
            content: "Hi",
            tool_calls: [
              { id: "call-1", name: "lookup", arguments: { q: "hi" } },
            ],
            usage: { input_tokens: 10, output_tokens: 4, total_tokens: 14 },
          },
        }),
      ),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });
    const result = await proxy.invoke([baseMessage]);

    expect(result.content).toBe("Hi");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls?.[0]?.name).toBe("lookup");
    expect(result.metadata).toEqual({
      input_tokens: 10,
      output_tokens: 4,
      total_tokens: 14,
    });
  });
});

// ---------------------------------------------------------------------------
// TestSecureLLMProxyStream
// ---------------------------------------------------------------------------

describe("SecureLLMProxy.stream", () => {
  test("yields synthetic chunks from a final (non-routed) result", async () => {
    // test_stream_yields_synthetic_chunks_from_final_result
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          proceed: true,
          status: "success",
          result: {
            content: "Hello!",
            tool_calls: [
              { id: "call-1", name: "lookup", arguments: { q: "weather" } },
            ],
            usage: { input_tokens: 10, output_tokens: 4, total_tokens: 14 },
          },
          duration_ms: 15.0,
        }),
      ),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });
    const chunks = [];
    for await (const c of proxy.stream([baseMessage])) chunks.push(c);

    expect(chunks).toHaveLength(2);
    expect(chunks[0]?.content).toBe("Hello!");
    expect(chunks[0]?.toolCalls).not.toBeNull();
    expect(chunks[0]?.toolCalls?.[0]?.name).toBe("lookup");
    expect(chunks[1]?.usage).toMatchObject({
      inputTokens: 10,
      outputTokens: 4,
      totalTokens: 14,
    });
    expect(proxy.lastDurationMs).toBe(15.0);
  });

  test("handles usage-only result (no content, no tool_calls)", async () => {
    // test_stream_handles_usage_only_result
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          proceed: true,
          status: "success",
          result: { content: "", tool_calls: [], usage: { total_tokens: 12 } },
        }),
      ),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });
    const chunks = [];
    for await (const c of proxy.stream([baseMessage])) chunks.push(c);

    expect(chunks).toHaveLength(1);
    expect(chunks[0]?.usage).toMatchObject({ totalTokens: 12 });
  });

  test("raises LLMInvocationError on status=error", async () => {
    // test_stream_raises_on_oe_owned_error
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          proceed: true,
          status: "error",
          error: "llm pod failed",
        }),
      ),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });

    await expect(async () => {
      for await (const _ of proxy.stream([baseMessage])) {
        /* consume */
      }
    }).rejects.toMatchObject({ source: "llm" });
  });

  test("forwards stop and options to the OE request body", async () => {
    // test_stream_forwards_stop_and_kwargs_to_oe_request
    const fetchSpy = vi.fn().mockResolvedValue(
      jsonResponse({
        proceed: true,
        status: "success",
        result: { content: "Hello!", tool_calls: [], metadata: {} },
      }),
    );
    vi.stubGlobal("fetch", fetchSpy);

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });
    const chunks = [];
    for await (const c of proxy.stream(
      [baseMessage],
      null,
      ["DONE"],
      new LLMInvocationOptions({ max_tokens: 32 }),
    )) {
      chunks.push(c);
    }

    const { body } = lastFetchCall(fetchSpy);
    const args = body?.["arguments"] as Record<string, unknown>;
    expect(args["stop"]).toEqual(["DONE"]);
    expect(args["options"]).toEqual({ max_tokens: 32 });
    expect(args["stream"]).toBe(true);
  });

  test("connects to the routed OE SSE endpoint and yields content + usage", async () => {
    // test_stream_connects_to_routed_oe_sse_endpoint
    const sseUrl = "http://oe:8000/tool/stream/exec-123/7";
    const fetchSpy = vi.fn().mockImplementation((url: string | URL) => {
      const u = String(url);
      if (u.endsWith("/tool/execute")) {
        return Promise.resolve(
          jsonResponse({
            proceed: true,
            route_to: sseUrl,
            result: null,
            cached_result: null,
          }),
        );
      }
      if (u === sseUrl) {
        return Promise.resolve(
          sseResponse([
            { content: "Hel", tool_call_chunks: [] },
            { content: "lo", tool_call_chunks: [] },
            {
              done: true,
              pod_name: "tool-pod-1",
              duration_ms: 23.0,
              usage: { input_tokens: 10, output_tokens: 4, total_tokens: 14 },
            },
          ]),
        );
      }
      return Promise.reject(new Error(`unexpected url: ${u}`));
    });
    vi.stubGlobal("fetch", fetchSpy);

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
      modelName: "gpt-4o",
    });
    const chunks = [];
    for await (const c of proxy.stream([baseMessage], 7)) chunks.push(c);

    const contents = chunks
      .map((c) => c.content)
      .filter((s): s is string => Boolean(s));
    expect(contents.slice(0, 2)).toEqual(["Hel", "lo"]);
    const usageChunk = chunks.find((c) => c.usage != null);
    expect(usageChunk?.usage).toMatchObject({
      inputTokens: 10,
      outputTokens: 4,
      totalTokens: 14,
    });
    expect(proxy.lastDurationMs).toBe(23.0);
    expect(proxy.lastPodName).toBe("tool-pod-1");

    // Verify the SSE URL was hit verbatim
    const sseCallUrls = fetchSpy.mock.calls.map((c) => String(c[0]));
    expect(sseCallUrls).toContain(sseUrl);
  });

  test("yields an interrupted marker on OE interrupt, without raising", async () => {
    // test_stream_yields_interrupted_marker_on_oe_interrupt
    const sseUrl = "http://oe:8000/tool/stream/exec-123/7";
    const fetchSpy = vi.fn().mockImplementation((url: string | URL) => {
      const u = String(url);
      if (u.endsWith("/tool/execute")) {
        return Promise.resolve(
          jsonResponse({ proceed: true, route_to: sseUrl }),
        );
      }
      if (u === sseUrl) {
        // OE aborts the in-flight call with a bare sentinel, then the stream
        // ends. The relay emits only {"interrupted": true} — no pod/duration.
        return Promise.resolve(
          sseResponse([
            { content: "Par", tool_call_chunks: [] },
            { interrupted: true },
          ]),
        );
      }
      return Promise.reject(new Error(`unexpected url: ${u}`));
    });
    vi.stubGlobal("fetch", fetchSpy);

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });
    const chunks = [];
    // Must NOT throw (unlike the truncated-SSE case): interrupt is a clean stop.
    for await (const c of proxy.stream([baseMessage], 7)) chunks.push(c);

    expect(chunks[0]?.content).toBe("Par");
    expect(chunks.at(-1)?.responseMetadata).toEqual({ interrupted: true });
  });

  test("reconnects to the same SSE URL on retryable error", async () => {
    // test_stream_reconnects_on_retryable_sse_error
    const sseUrl = "http://oe:8000/tool/stream/exec-123/7";
    let sseCalls = 0;
    const fetchSpy = vi.fn().mockImplementation((url: string | URL) => {
      const u = String(url);
      if (u.endsWith("/tool/execute")) {
        return Promise.resolve(
          jsonResponse({ proceed: true, route_to: sseUrl }),
        );
      }
      if (u === sseUrl) {
        sseCalls += 1;
        if (sseCalls === 1) {
          return Promise.resolve(
            sseResponse([
              {
                error:
                  "tool pod is no longer reserved for this session; retry request",
                retryable: true,
              },
            ]),
          );
        }
        return Promise.resolve(
          sseResponse([
            { content: "Hi", tool_call_chunks: [] },
            { done: true, pod_name: "tool-pod-1", duration_ms: 11.0 },
          ]),
        );
      }
      return Promise.reject(new Error(`unexpected url: ${u}`));
    });
    vi.stubGlobal("fetch", fetchSpy);

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });
    const chunks = [];
    for await (const c of proxy.stream([baseMessage], 7)) chunks.push(c);

    expect(chunks.map((c) => c.content).filter(Boolean)).toEqual(["Hi"]);
    expect(sseCalls).toBe(2);
  });

  test("retries transport disconnect then recovers on the same SSE URL", async () => {
    // Owner death drops the socket without ReleaseDispatch. The client waits
    // the heartbeat TTL, then reconnects on the same URL and recovers.
    const sseUrl = "http://oe:8000/tool/stream/exec-123/7";
    let sseCalls = 0;
    const fetchSpy = vi.fn().mockImplementation((url: string | URL) => {
      const u = String(url);
      if (u.endsWith("/tool/execute")) {
        return Promise.resolve(
          jsonResponse({ proceed: true, route_to: sseUrl }),
        );
      }
      if (u === sseUrl) {
        sseCalls += 1;
        if (sseCalls === 1) {
          return Promise.reject(new TypeError("fetch failed"));
        }
        if (sseCalls === 2) {
          return Promise.resolve(
            sseResponse([
              {
                error: "invoke_llm relay already active for this step",
                retryable: true,
                retry_after_ms: 0,
              },
            ]),
          );
        }
        return Promise.resolve(
          sseResponse([
            { content: "recovered", tool_call_chunks: [] },
            { done: true, pod_name: "tool-pod-1", duration_ms: 11.0 },
          ]),
        );
      }
      return Promise.reject(new Error(`unexpected url: ${u}`));
    });
    vi.stubGlobal("fetch", fetchSpy);

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });
    const chunks = [];
    for await (const c of proxy.stream([baseMessage], 7)) chunks.push(c);

    expect(chunks.map((c) => c.content).filter(Boolean)).toEqual(["recovered"]);
    expect(sseCalls).toBe(3);
    expect(oeStreamRetry.sleep).toHaveBeenNthCalledWith(
      1,
      OE_DISPATCH_TAKEOVER_RETRY_DELAY_MS,
    );
    expect(oeStreamRetry.sleep).toHaveBeenNthCalledWith(2, 0);
  });

  test("does not retry transport disconnect after content was yielded", async () => {
    const sseUrl = "http://oe:8000/tool/stream/exec-123/7";
    let sseCalls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string | URL) => {
        const u = String(url);
        if (u.endsWith("/tool/execute")) {
          return Promise.resolve(
            jsonResponse({ proceed: true, route_to: sseUrl }),
          );
        }
        if (u === sseUrl) {
          sseCalls += 1;
          return Promise.resolve(
            disconnectingSseResponse([
              { content: "Hel", tool_call_chunks: [] },
            ]),
          );
        }
        return Promise.reject(new Error(`unexpected url: ${u}`));
      }),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });
    const chunks: Array<{ content?: string }> = [];
    await expect(async () => {
      for await (const c of proxy.stream([baseMessage], 7)) chunks.push(c);
    }).rejects.toThrow(LLMInvocationError);

    expect(chunks.map((c) => c.content).filter(Boolean)).toEqual(["Hel"]);
    expect(sseCalls).toBe(1);
    expect(oeStreamRetry.sleep).not.toHaveBeenCalled();
  });

  test("does not retry transport disconnect after a tool-call fragment", async () => {
    const sseUrl = "http://oe:8000/tool/stream/exec-123/7";
    let sseCalls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string | URL) => {
        const u = String(url);
        if (u.endsWith("/tool/execute")) {
          return Promise.resolve(
            jsonResponse({ proceed: true, route_to: sseUrl }),
          );
        }
        if (u === sseUrl) {
          sseCalls += 1;
          return Promise.resolve(
            disconnectingSseResponse([
              {
                tool_call_chunks: [
                  { id: "call-1", name: "search", args: '{"q":', index: 0 },
                ],
              },
            ]),
          );
        }
        return Promise.reject(new Error(`unexpected url: ${u}`));
      }),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });
    const chunks: Array<{ toolCalls?: Array<{ name?: string }> }> = [];
    await expect(async () => {
      for await (const c of proxy.stream([baseMessage], 7)) chunks.push(c);
    }).rejects.toThrow(LLMInvocationError);

    expect(chunks).toHaveLength(1);
    expect(chunks[0]?.toolCalls?.[0]?.name).toBe("search");
    expect(sseCalls).toBe(1);
    expect(oeStreamRetry.sleep).not.toHaveBeenCalled();
  });

  test("does not retry when the consumer stops iterating", async () => {
    const sseUrl = "http://oe:8000/tool/stream/exec-123/7";
    let sseCalls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string | URL) => {
        const u = String(url);
        if (u.endsWith("/tool/execute")) {
          return Promise.resolve(
            jsonResponse({ proceed: true, route_to: sseUrl }),
          );
        }
        if (u === sseUrl) {
          sseCalls += 1;
          return Promise.resolve(
            sseResponse([
              { content: "Hel", tool_call_chunks: [] },
              { content: "lo", tool_call_chunks: [] },
              { done: true, pod_name: "tool-pod-1", duration_ms: 11.0 },
            ]),
          );
        }
        return Promise.reject(new Error(`unexpected url: ${u}`));
      }),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });
    for await (const c of proxy.stream([baseMessage], 7)) {
      if (c.content === "Hel") break;
    }

    expect(sseCalls).toBe(1);
    expect(oeStreamRetry.sleep).not.toHaveBeenCalled();
  });

  test("raises LLMInvocationError on SSE connection failure", async () => {
    // test_stream_raises_on_oe_sse_connection_error
    const sseUrl = "http://oe:8000/tool/stream/exec-123/7";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string | URL) => {
        const u = String(url);
        if (u.endsWith("/tool/execute")) {
          return Promise.resolve(
            jsonResponse({
              proceed: true,
              route_to: sseUrl,
              result: null,
              cached_result: null,
            }),
          );
        }
        return Promise.reject(new Error("relay timeout"));
      }),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });

    await expect(async () => {
      for await (const _ of proxy.stream([baseMessage], 7)) {
        /* consume */
      }
    }).rejects.toBeInstanceOf(LLMInvocationError);
  });

  test("raises LLMInvocationError when the routed SSE response has no body", async () => {
    // S2 regression: a 2xx SSE response with a null body (204/polyfill) must
    // surface a legible LLMInvocationError, not a "Cannot read properties of
    // null" TypeError from response.body.getReader().
    const sseUrl = "http://oe:8000/tool/stream/exec-123/7";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string | URL) => {
        const u = String(url);
        if (u.endsWith("/tool/execute")) {
          return Promise.resolve(
            jsonResponse({
              proceed: true,
              route_to: sseUrl,
              result: null,
              cached_result: null,
            }),
          );
        }
        // 200 OK but no body — `response.body` is null.
        return Promise.resolve(new Response(null, { status: 200 }));
      }),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });

    await expect(async () => {
      for await (const _ of proxy.stream([baseMessage], 7)) {
        /* consume */
      }
    }).rejects.toThrow(/SSE response has no body/);
  });

  test("raises LLMInvocationError on truncated SSE (no done signal)", async () => {
    // test_stream_raises_on_truncated_oe_sse
    const sseUrl = "http://oe:8000/tool/stream/exec-123/7";
    let sseCalls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string | URL) => {
        const u = String(url);
        if (u.endsWith("/tool/execute")) {
          return Promise.resolve(
            jsonResponse({
              proceed: true,
              route_to: sseUrl,
              result: null,
              cached_result: null,
            }),
          );
        }
        sseCalls += 1;
        return Promise.resolve(truncatedSseResponse());
      }),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });

    const err = await (async () => {
      for await (const _ of proxy.stream([baseMessage], 7)) {
        /* consume */
      }
    })().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(LLMInvocationError);
    expect((err as LLMInvocationError).source).toBeUndefined();
    expect(String(err)).toMatch(/truncated|done/i);
    expect(sseCalls).toBe(1);
    expect(oeStreamRetry.sleep).not.toHaveBeenCalled();
  });

  test("retries truncated SSE with no output then fails", async () => {
    const sseUrl = "http://oe:8000/tool/stream/exec-123/7";
    let sseCalls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string | URL) => {
        const u = String(url);
        if (u.endsWith("/tool/execute")) {
          return Promise.resolve(
            jsonResponse({
              proceed: true,
              route_to: sseUrl,
              result: null,
              cached_result: null,
            }),
          );
        }
        sseCalls += 1;
        return Promise.resolve(emptyTruncatedSseResponse());
      }),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });

    const err = await (async () => {
      for await (const _ of proxy.stream([baseMessage], 7)) {
        /* consume */
      }
    })().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(LLMInvocationError);
    expect(String(err)).toMatch(/truncated|done/i);
    expect(sseCalls).toBe(3);
    expect(oeStreamRetry.sleep).toHaveBeenCalledTimes(2);
    expect(oeStreamRetry.sleep).toHaveBeenCalledWith(
      OE_DISPATCH_TAKEOVER_RETRY_DELAY_MS,
    );
  });

  test("stamps source=llm on in-band SSE error", async () => {
    const sseUrl = "http://oe:8000/tool/stream/exec-123/7";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string | URL) => {
        const u = String(url);
        if (u.endsWith("/tool/execute")) {
          return Promise.resolve(
            jsonResponse({
              proceed: true,
              route_to: sseUrl,
              result: null,
              cached_result: null,
            }),
          );
        }
        return Promise.resolve(sseResponse([{ error: "Resource not found" }]));
      }),
    );

    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
    });

    await expect(async () => {
      for await (const _ of proxy.stream([baseMessage], 7)) {
        /* consume */
      }
    }).rejects.toMatchObject({ source: "llm" });
  });
});

// ---------------------------------------------------------------------------
// serializeBoundTools — ports Python TestSerializeBoundTools
// ---------------------------------------------------------------------------

describe("SecureLLMProxy.serializeBoundTools", () => {
  type ProxyInternals = {
    boundTools: unknown[];
    serializeBoundTools(): unknown;
  };

  function makeProxy(boundTools: unknown[]): ProxyInternals {
    const proxy = new SecureLLMProxy({
      oeUrl: "http://localhost:8080",
      executionId: "exec-123",
      llmId: "primary",
      boundTools: boundTools as never,
    }) as unknown as ProxyInternals;
    return proxy;
  }

  test("plain dict tool (no tool_call_schema) passes through validator", () => {
    const tool = {
      name: "calc",
      description: "Calculate",
      parameters: { type: "object" },
      type: "function",
    };

    const proxy = makeProxy([tool]);
    const result = proxy.serializeBoundTools() as Array<{
      name?: string;
      description?: string;
      type?: string;
    }>;

    expect(result).toHaveLength(1);
    expect(result[0]?.name).toBe("calc");
    expect(result[0]?.description).toBe("Calculate");
    expect(result[0]?.type).toBe("function");
  });

  test("returns null when no bound tools", () => {
    const proxy = makeProxy([]);
    expect(proxy.serializeBoundTools()).toBeNull();
  });

  test("skips unrecognized tool types (non-objects)", () => {
    const proxy = makeProxy(["not a tool", 42, null]);
    expect(proxy.serializeBoundTools()).toBeNull();
  });
});
