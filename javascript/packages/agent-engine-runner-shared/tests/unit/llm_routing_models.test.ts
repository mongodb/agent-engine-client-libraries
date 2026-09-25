/**
 * Unit tests for LLM routing — schemas, stream collection, usage extraction.
 *
 * Mirrors the first 5 classes of Python's tests/unit/test_llm_routing.py:
 *   - TestLLMPodInvokeRequestSerialization (4 tests)
 *   - TestLLMPodInvokeResponseSerialization (2 tests)
 *   - TestLLMPodStreamEventSerialization (3 tests)
 *   - TestSecureLLMProxyStreamCollection (4 tests)
 *   - TestLLMResultUsageExtraction (2 tests)
 *
 * The retry / caching / handle-invoke classes live in `llm_routing.test.ts`
 * (separate file for review-ability).
 *
 * TS-vs-Python adaptations:
 *   - Python `Model(**fields)` → TS `Schema.parse(fields)` (interfaces, not classes).
 *   - Python `model.model_dump_json()` / `Model.model_validate_json()` →
 *     TS `JSON.stringify(parsed)` + `Schema.parse(JSON.parse(...))`.
 *   - LLMTokenUsage auto-computes `total_tokens` in its class constructor,
 *     but `LLMTokenUsageSchema` (used inside LLMPodStreamEventSchema etc.)
 *     does NOT — Zod has no equivalent of Pydantic's model_validator.
 *     The done-event roundtrip test verifies the fields actually present;
 *     `total_tokens` auto-compute is NOT asserted (would require an agent-engine-sdk
 *     schema change with `.transform()`).
 *   - Private static methods on SecureLLMProxy accessed via cast-through-unknown.
 *   - `caplog` → `vi.mock` on the logger module (same pattern as metrics.test.ts).
 */

import { describe, test, expect, vi } from "vitest";

// Mock logger BEFORE the source imports so the "tool call id changed" warning
// can be observed via the mock spy. vi.hoisted ensures `mocks` exists when the
// hoisted vi.mock factory runs.
const mocks = vi.hoisted(() => ({
  logger: {
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
    trace: vi.fn(),
    fatal: vi.fn(),
    log: vi.fn(),
  },
}));

vi.mock("../../src/logger.js", () => ({
  getLogger: () => mocks.logger,
  setupLogging: vi.fn(),
}));

import {
  InvokeLLMRequestArgumentsSchema,
  LLMPodInvokeRequestSchema,
  LLMPodInvokeResponseSchema,
  LLMPodStreamEventSchema,
  LLMResult,
  accumulateStreamUsage,
  addTokenUsage,
  mergeTokenUsage,
  normalizeLLMPodInvokeResponse,
  SecureLLMProxy,
} from "../../src/index.js";
import { LLMTokenUsage } from "@mongodb-js/agent-engine-sdk";
import type { ToolCallChunk } from "@mongodb-js/agent-engine-sdk";
import { LLMToolCall } from "@mongodb-js/agent-engine-sdk";

// ---------------------------------------------------------------------------
// LLMPodInvokeRequest serialization
// ---------------------------------------------------------------------------

describe("LLMPodInvokeRequest serialization", () => {
  test("basic: roundtrips required fields, optional fields are undefined", () => {
    // test_basic_serialization
    const request = LLMPodInvokeRequestSchema.parse({
      execution_id: "exec-123",
      arguments: {
        model: "gpt-4o-mini",
        llm_id: "primary",
        messages: [{ role: "user", content: "Hello" }],
      },
    });
    const restored = LLMPodInvokeRequestSchema.parse(
      JSON.parse(JSON.stringify(request)),
    );

    expect(restored.execution_id).toBe("exec-123");
    expect(restored.arguments.model).toBe("gpt-4o-mini");
    expect(restored.arguments.messages).toEqual([
      { role: "user", content: "Hello" },
    ]);
    expect(restored.arguments.llm_id).toBe("primary");
    expect(restored.arguments.stop_sequences).toBeUndefined();
    expect(restored.arguments.tools).toBeUndefined();
    expect(restored.arguments.options).toBeUndefined();
  });

  test("full: roundtrips all fields including stop_sequences, tools, options", () => {
    // test_full_serialization
    const request = LLMPodInvokeRequestSchema.parse({
      execution_id: "exec-456",
      arguments: {
        model: "gemini-2.5-flash",
        messages: [
          { role: "system", content: "You are helpful." },
          { role: "user", content: "What is 2+2?" },
        ],
        llm_id: "primary",
        stop_sequences: ["\n", "END"],
        tools: [{ name: "calculator", parameters: { type: "object" } }],
        options: { max_tokens: 100 },
      },
    });
    const restored = LLMPodInvokeRequestSchema.parse(
      JSON.parse(JSON.stringify(request)),
    );

    expect(restored.arguments.stop_sequences).toEqual(["\n", "END"]);
    expect(restored.arguments.tools).toEqual([
      { name: "calculator", parameters: { type: "object" } },
    ]);
    expect(restored.arguments.options?.maxTokens).toBe(100);
  });

  test("JSON roundtrip preserves the request shape", () => {
    // test_json_roundtrip
    const original = LLMPodInvokeRequestSchema.parse({
      execution_id: "exec-789",
      arguments: {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hi" }],
      },
    });
    const restored = LLMPodInvokeRequestSchema.parse(
      JSON.parse(JSON.stringify(original)),
    );

    expect(restored).toEqual(original);
  });

  test("flat payload is lifted into typed arguments (legacy callers)", () => {
    // test_flat_payload_is_lifted_into_typed_arguments
    const restored = LLMPodInvokeRequestSchema.parse({
      execution_id: "exec-flat",
      model: "gpt-4o",
      llm_id: "primary",
      messages: [{ role: "user", content: "Hello" }],
      stop: ["END"],
      kwargs: { max_tokens: 32 },
    });

    expect(restored.execution_id).toBe("exec-flat");
    expect(restored.arguments.model).toBe("gpt-4o");
    expect(restored.arguments.messages).toEqual([
      { role: "user", content: "Hello" },
    ]);
    expect(restored.arguments.stop_sequences).toEqual(["END"]);
    expect(restored.arguments.options?.maxTokens).toBe(32);
  });

  test("flat payload keeps step_number out of arguments", () => {
    // test_flat_payload_step_number_stays_out_of_arguments
    // step_number is routing metadata: preserved top-level on a flat body,
    // never poured into the LLM arguments bag.
    const restored = LLMPodInvokeRequestSchema.parse({
      execution_id: "exec-flat",
      platform_trace_id: "0123456789abcdef0123456789abcdef",
      step_number: 3,
      model: "gpt-4o",
      messages: [{ role: "user", content: "Hello" }],
    });

    expect(restored.platform_trace_id).toBe("0123456789abcdef0123456789abcdef");

    expect(restored.step_number).toBe(3);
    expect(restored.arguments).not.toHaveProperty("step_number");
  });

  test("llm_id round-trips through JSON serialization", () => {
    // test_llm_id_round_trips
    const request = LLMPodInvokeRequestSchema.parse({
      execution_id: "exec-1",
      arguments: {
        model: "gpt-4o",
        llm_id: "primary",
        messages: [{ role: "user", content: "Hi" }],
      },
    });
    const restored = LLMPodInvokeRequestSchema.parse(
      JSON.parse(JSON.stringify(request)),
    );

    expect(restored.arguments.llm_id).toBe("primary");
  });

  test("missing llm_id defaults to the '__default__' sentinel", () => {
    // test_llm_id_defaults_to_sentinel
    const args = InvokeLLMRequestArgumentsSchema.parse({
      model: "gpt-4o",
      messages: [{ role: "user", content: "Hi" }],
    });
    expect(args.llm_id).toBe("__default__");
    // The default survives a JSON round-trip (the actual wire-compat scenario).
    const restored = InvokeLLMRequestArgumentsSchema.parse(
      JSON.parse(JSON.stringify(args)),
    );
    expect(restored.llm_id).toBe("__default__");
  });
});

// ---------------------------------------------------------------------------
// LLMPodInvokeResponse serialization
// ---------------------------------------------------------------------------

describe("LLMPodInvokeResponse serialization", () => {
  test("success response", () => {
    // test_success_response
    const response = LLMPodInvokeResponseSchema.parse({
      status: "success",
      result: { content: "Hello!", tool_calls: [] },
      pod_name: "pod-abc",
      duration_ms: 150.5,
      usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
    });
    const data = JSON.parse(JSON.stringify(response));

    expect(data.status).toBe("success");
    expect(data.result.content).toBe("Hello!");
    expect(data.pod_name).toBe("pod-abc");
    expect(data.usage.total_tokens).toBe(15);
  });

  test("error response: result field absent (not serialized when undefined)", () => {
    // test_error_response
    const response = LLMPodInvokeResponseSchema.parse({
      status: "error",
      error: "Rate limit exceeded",
      pod_name: "pod-xyz",
      duration_ms: 50.0,
    });
    const data = JSON.parse(JSON.stringify(response));

    expect(data.status).toBe("error");
    expect(data.error).toBe("Rate limit exceeded");
    // TS: undefined optional fields are dropped by JSON.stringify (not
    // serialized as `null` like Python's model_dump). Python checks
    // `data["result"] is None`; the TS equivalent is "key not present".
    expect(data.result).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// normalizeLLMPodInvokeResponse — usage <-> result.usage sync
// Mirrors Python's `_sync_usage_with_result` model validator. OE reads token
// usage from `result["usage"]`, so the /invoke_llm handler must mirror the
// top-level `usage` into `result` before sending.
// ---------------------------------------------------------------------------

describe("normalizeLLMPodInvokeResponse", () => {
  test("copies top-level usage into result.usage when result lacks it", () => {
    const response = LLMPodInvokeResponseSchema.parse({
      status: "success",
      result: { content: "Hello!", tool_calls: [] },
      pod_name: "pod-abc",
      duration_ms: 150.5,
      usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
    });

    const synced = normalizeLLMPodInvokeResponse(response);

    // OE reads token usage from the wire `result.usage`.
    const wire = JSON.parse(JSON.stringify(synced));
    expect(wire.result.usage.total_tokens).toBe(15);
    expect(wire.usage.total_tokens).toBe(15);
  });

  test("lifts result.usage to the top level when top-level usage is absent", () => {
    const usage = { input_tokens: 8, output_tokens: 2, total_tokens: 10 };
    const response = LLMPodInvokeResponseSchema.parse({
      status: "success",
      result: { content: "Hi", tool_calls: [], usage },
      pod_name: "pod-xyz",
      duration_ms: 12.0,
    });
    // Precondition: top-level usage really is unset before normalization.
    expect(response.usage).toBeUndefined();

    const synced = normalizeLLMPodInvokeResponse(response);

    expect(synced.usage).toEqual(usage);
    expect(synced.result?.usage).toEqual(usage);
  });

  test("leaves an error response (no result) untouched", () => {
    const response = LLMPodInvokeResponseSchema.parse({
      status: "error",
      error: "boom",
      pod_name: "pod-1",
      duration_ms: 1.0,
    });

    const synced = normalizeLLMPodInvokeResponse(response);

    expect(synced.result).toBeUndefined();
    expect(synced.usage).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// LLMPodStreamEvent serialization
// ---------------------------------------------------------------------------

describe("LLMPodStreamEvent serialization", () => {
  test("chunk event JSON roundtrip preserves tool_call_chunks", () => {
    // test_chunk_event_json_roundtrip
    const event = LLMPodStreamEventSchema.parse({
      content: "Hel",
      tool_call_chunks: [{ id: "call-1", name: "lookup", args: '{"q"' }],
      tool_calls: [],
    });
    const restored = LLMPodStreamEventSchema.parse(
      JSON.parse(JSON.stringify(event)),
    );

    expect(restored.content).toBe("Hel");
    expect(restored.tool_call_chunks).not.toBeNull();
    expect(restored.tool_call_chunks?.[0]?.name).toBe("lookup");
    expect(restored.tool_calls).toEqual([]);
  });

  test("done event JSON roundtrip preserves usage fields", () => {
    // test_done_event_json_roundtrip
    const event = LLMPodStreamEventSchema.parse({
      done: true,
      pod_name: "tool-pod-1",
      duration_ms: 23.5,
      usage: { input_tokens: 10, output_tokens: 4 },
    });
    const restored = LLMPodStreamEventSchema.parse(
      JSON.parse(JSON.stringify(event)),
    );

    expect(restored.done).toBe(true);
    expect(restored.pod_name).toBe("tool-pod-1");
    expect(restored.duration_ms).toBe(23.5);
    expect(restored.usage?.inputTokens).toBe(10);
    expect(restored.usage?.outputTokens).toBe(4);
    expect(restored.usage?.totalTokens).toBe(14);
  });

  test("message-metadata event JSON roundtrip preserves id/name/metadata", () => {
    // test_message_metadata_event_json_roundtrip
    const event = LLMPodStreamEventSchema.parse({
      content: "final",
      id: "run-1",
      name: "assistant",
      additional_kwargs: { refusal: null },
      response_metadata: { finish_reason: "stop" },
    });
    const restored = LLMPodStreamEventSchema.parse(
      JSON.parse(JSON.stringify(event)),
    );

    expect(restored.id).toBe("run-1");
    expect(restored.name).toBe("assistant");
    expect(restored.additional_kwargs).toEqual({ refusal: null });
    expect(restored.response_metadata).toEqual({ finish_reason: "stop" });
  });
});

// ---------------------------------------------------------------------------
// SecureLLMProxy stream collection (private static methods)
// ---------------------------------------------------------------------------

interface ProxyPrivates {
  convertStreamChunksToToolCalls: (
    chunks: ToolCallChunk[],
  ) => LLMToolCall[] | null;
  convertToolCallsToStreamChunks: (
    toolCalls: LLMToolCall[] | null,
  ) => ToolCallChunk[] | null;
  responsePayloadFromChunks: (
    chunks: Parameters<typeof SecureLLMProxy.responseFromStreamChunks>[0],
  ) => Record<string, unknown>;
  convertResultToResponse: (result: unknown) => {
    toolCalls?: unknown[] | null;
    additionalKwargs?: Record<string, unknown> | null;
  };
}

const proxyPrivates = SecureLLMProxy as unknown as ProxyPrivates;

describe("SecureLLMProxy stream collection", () => {
  test("absent tool_call index uses the active call (id-tracked)", () => {
    // test_absent_tool_call_index_uses_active_call
    const toolCalls = proxyPrivates.convertStreamChunksToToolCalls([
      { id: "call-1", name: "lookup", args: '{"q":' } as ToolCallChunk,
      { args: '"weather"}' } as ToolCallChunk,
    ]);

    expect(toolCalls).not.toBeNull();
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls?.[0]?.id).toBe("call-1");
    expect(toolCalls?.[0]?.args).toEqual({ q: "weather" });
  });

  test("responseFromStreamChunks preserves id/name/additionalKwargs/responseMetadata", () => {
    // test_response_from_stream_chunks_preserves_message_metadata
    const response = SecureLLMProxy.responseFromStreamChunks([
      { content: "Hello" },
      {
        id: "run-1",
        name: "assistant",
        additionalKwargs: { refusal: null },
        responseMetadata: { finish_reason: "stop" },
      },
    ] as Parameters<typeof SecureLLMProxy.responseFromStreamChunks>[0]);

    expect(response.content).toBe("Hello");
    expect(response.id).toBe("run-1");
    expect(response.name).toBe("assistant");
    expect(response.additionalKwargs).toEqual({ refusal: null });
    expect(response.responseMetadata).toEqual({ finish_reason: "stop" });
  });

  test("responseFromStreamChunks preserves the first streamed message id", () => {
    const response = SecureLLMProxy.responseFromStreamChunks([
      { id: "chatcmpl-1", content: "Hello" },
      { id: "run-1" },
    ] as Parameters<typeof SecureLLMProxy.responseFromStreamChunks>[0]);

    expect(response.id).toBe("chatcmpl-1");
  });

  test("folded stream drops provider tool-call fragments but keeps semantic calls", () => {
    const chunks = [
      {
        id: "chunk-1",
        toolCalls: [
          { id: "call-1", name: "lookup", args: '{"q":"weather"}', index: 0 },
        ],
        additionalKwargs: {
          tool_calls: [
            { index: 0, id: "call-1", function: { name: "lookup" } },
          ],
          provider: "acme",
        },
      },
    ] as Parameters<typeof SecureLLMProxy.responseFromStreamChunks>[0];

    const response = SecureLLMProxy.responseFromStreamChunks(chunks);
    expect(response.toolCalls).toHaveLength(1);
    expect(response.toolCalls?.[0]).toMatchObject({
      id: "call-1",
      name: "lookup",
    });
    expect(response.additionalKwargs).toEqual({ provider: "acme" });

    const payload = proxyPrivates.responsePayloadFromChunks(chunks);
    expect(payload["tool_calls"]).toHaveLength(1);
    expect(payload["additional_kwargs"]).toEqual({ provider: "acme" });

    const replayed = proxyPrivates.convertResultToResponse(payload);
    expect(replayed.toolCalls).toHaveLength(1);
    expect(replayed.additionalKwargs).toEqual({ provider: "acme" });
  });

  test("folded stream omits additionalKwargs when only fragments remain", () => {
    const chunks = [
      {
        id: "chunk-1",
        toolCalls: [
          { id: "call-1", name: "lookup", args: '{"q":"weather"}', index: 0 },
        ],
        additionalKwargs: {
          tool_calls: [{ index: 0, id: "call-1" }],
        },
      },
    ] as Parameters<typeof SecureLLMProxy.responseFromStreamChunks>[0];

    const response = SecureLLMProxy.responseFromStreamChunks(chunks);
    expect(response.toolCalls).toHaveLength(1);
    expect(response.additionalKwargs).toBeUndefined();

    const payload = proxyPrivates.responsePayloadFromChunks(chunks);
    expect(payload["tool_calls"]).toHaveLength(1);
    expect(payload).not.toHaveProperty("additional_kwargs");
  });

  test("synthetic tool_call encoding preserves idless calls and `type` field", () => {
    // test_synthetic_tool_call_encoding_preserves_idless_calls_and_type
    const chunks = proxyPrivates.convertToolCallsToStreamChunks([
      new LLMToolCall({
        name: "lookup",
        args: { q: "one" },
        type: "tool_call",
      }),
      new LLMToolCall({ name: "search", args: { q: "two" }, type: "function" }),
    ]);

    expect(chunks).not.toBeNull();
    expect(chunks?.map((c) => c.index)).toEqual([0, 1]);
    expect(chunks?.map((c) => c.type)).toEqual(["tool_call", "function"]);

    const response = SecureLLMProxy.responseFromStreamChunks([
      { toolCalls: chunks },
    ] as Parameters<typeof SecureLLMProxy.responseFromStreamChunks>[0]);

    expect(response.toolCalls).not.toBeUndefined();
    expect(response.toolCalls).toHaveLength(2);
    expect(response.toolCalls?.[0]?.name).toBe("lookup");
    expect(response.toolCalls?.[0]?.args).toEqual({ q: "one" });
    expect(response.toolCalls?.[0]?.type).toBe("tool_call");
    expect(response.toolCalls?.[1]?.name).toBe("search");
    expect(response.toolCalls?.[1]?.args).toEqual({ q: "two" });
    expect(response.toolCalls?.[1]?.type).toBe("function");
  });

  test("conflicting tool_call id indices log a warning", () => {
    // test_conflicting_tool_call_id_indices_log_warning
    mocks.logger.warn.mockClear();

    const toolCalls = proxyPrivates.convertStreamChunksToToolCalls([
      {
        id: "call-1",
        name: "lookup",
        args: '{"q": "one"}',
        index: 0,
      } as ToolCallChunk,
      {
        id: "call-1",
        name: "search",
        args: '{"q": "two"}',
        index: 1,
      } as ToolCallChunk,
    ]);

    expect(toolCalls).not.toBeNull();
    // Find a warn call whose message includes the conflict-id text. The
    // logger.warn(fields, msg) shape means the message is in arg[1].
    const warnCalls = mocks.logger.warn.mock.calls.map((c) =>
      String(c[1] ?? c[0] ?? ""),
    );
    expect(
      warnCalls.some((m) =>
        /Tool call id call-1 changed stream index from 0 to 1/.test(m),
      ),
    ).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// LLMResult.extractUsage
// ---------------------------------------------------------------------------

describe("LLMResult.extractUsage", () => {
  test("extracts usage from response_metadata.usage", () => {
    // test_extract_usage_from_response_metadata_usage
    const response = {
      usage_metadata: null,
      response_metadata: { usage: { prompt_tokens: 3, completion_tokens: 2 } },
      metadata: {},
    };

    const usage = LLMResult.extractUsage(response);

    expect(usage).toBeDefined();
    expect(usage?.promptTokens).toBe(3);
    expect(usage?.completionTokens).toBe(2);
    expect(usage?.totalTokens).toBe(5);
  });

  test("extracts usage from response_metadata.token_usage", () => {
    // test_extract_usage_from_response_metadata_token_usage
    const response = {
      usage_metadata: null,
      response_metadata: { token_usage: { input_tokens: 4, output_tokens: 6 } },
      metadata: {},
    };

    const usage = LLMResult.extractUsage(response);

    expect(usage).toBeDefined();
    expect(usage?.inputTokens).toBe(4);
    expect(usage?.outputTokens).toBe(6);
    expect(usage?.totalTokens).toBe(10);
  });

  test("extracts usage from an Anthropic Usage-like object", () => {
    const response = {
      usage_metadata: null,
      response_metadata: {
        usage: { input_tokens: 1048, output_tokens: 1222 },
      },
      metadata: {},
    };
    const usage = LLMResult.extractUsage(response);
    expect(usage?.inputTokens).toBe(1048);
    expect(usage?.outputTokens).toBe(1222);
    expect(usage?.totalTokens).toBe(2270);
  });

  test("extracts usage from a top-level usage attribute", () => {
    const response = {
      usage_metadata: null,
      response_metadata: {},
      metadata: {},
      usage: { input_tokens: 7, output_tokens: 3 },
    };
    const usage = LLMResult.extractUsage(response);
    expect(usage?.inputTokens).toBe(7);
    expect(usage?.outputTokens).toBe(3);
  });

  test("malformed usage returns undefined without throwing", () => {
    const response = {
      usage_metadata: "n/a",
      response_metadata: { usage: "n/a" },
      metadata: { usage: "n/a" },
      usage: "n/a",
    };
    expect(LLMResult.extractUsage(response)).toBeUndefined();
  });

  test("fromResponse coerces a usage object left in response_metadata", () => {
    const usageObj = { input_tokens: 1048, output_tokens: 1222 };
    const result = LLMResult.fromResponse({
      content: "ok",
      tool_calls: [],
      usage_metadata: null,
      response_metadata: {
        usage: usageObj,
        model_name: "claude-sonnet-4-6",
      },
      metadata: {},
    });
    expect(result.content).toBe("ok");
    expect(result.usage?.inputTokens).toBe(1048);
    expect(result.usage?.outputTokens).toBe(1222);
    expect(
      (result.responseMetadata?.["usage"] as { input_tokens?: number })[
        "input_tokens"
      ],
    ).toBe(1048);
    expect(result.responseMetadata?.["model_name"]).toBe("claude-sonnet-4-6");
    expect(() => JSON.stringify(result.responseMetadata)).not.toThrow();
  });

  test("fromResponse succeeds when usage is non-finite or a raising getter", () => {
    const raising = {
      get input_tokens(): number {
        throw new Error("nope");
      },
      output_tokens: 4,
    };
    const result = LLMResult.fromResponse({
      content: "still works",
      tool_calls: [],
      usage_metadata: null,
      response_metadata: {
        usage: raising,
        token_usage: { input_tokens: Number.NaN, output_tokens: Infinity },
      },
      additional_kwargs: { blob: { toJSON: () => ({}) } },
      metadata: { usage: { input_tokens: 3, secret: { nested: () => 1 } } },
    });
    expect(result.content).toBe("still works");
  });
});

describe("addTokenUsage", () => {
  test("adds zero-filled LangChain deltas", () => {
    const added = addTokenUsage(
      new LLMTokenUsage({
        input_tokens: 18,
        output_tokens: 1,
        total_tokens: 19,
      }),
      new LLMTokenUsage({
        input_tokens: 0,
        output_tokens: 4,
        total_tokens: 4,
      }),
    );
    expect(added?.inputTokens).toBe(18);
    expect(added?.outputTokens).toBe(5);
    expect(added?.totalTokens).toBe(23);
  });
});

describe("accumulateStreamUsage", () => {
  test("takes the last cumulative snapshot", () => {
    const accumulated = accumulateStreamUsage(
      new LLMTokenUsage({
        input_tokens: 10,
        output_tokens: 1,
        total_tokens: 11,
      }),
      new LLMTokenUsage({
        input_tokens: 10,
        output_tokens: 2,
        total_tokens: 12,
      }),
    );
    expect(accumulated?.inputTokens).toBe(10);
    expect(accumulated?.outputTokens).toBe(2);
    expect(accumulated?.totalTokens).toBe(12);
  });

  test("adds zero-filled LangChain deltas", () => {
    const accumulated = accumulateStreamUsage(
      new LLMTokenUsage({
        input_tokens: 18,
        output_tokens: 1,
        total_tokens: 19,
      }),
      new LLMTokenUsage({
        input_tokens: 0,
        output_tokens: 4,
        total_tokens: 4,
      }),
    );
    expect(accumulated?.inputTokens).toBe(18);
    expect(accumulated?.outputTokens).toBe(5);
    expect(accumulated?.totalTokens).toBe(23);
  });

  test("merges split input and output chunks", () => {
    const accumulated = accumulateStreamUsage(
      new LLMTokenUsage({ input_tokens: 1048 }),
      new LLMTokenUsage({ output_tokens: 1222 }),
    );
    expect(accumulated?.inputTokens).toBe(1048);
    expect(accumulated?.outputTokens).toBe(1222);
    expect(accumulated?.totalTokens).toBe(2270);
  });
});

describe("mergeTokenUsage", () => {
  test("merges split input and output chunks", () => {
    const merged = mergeTokenUsage(
      new LLMTokenUsage({ input_tokens: 1048 }),
      new LLMTokenUsage({ output_tokens: 1222 }),
    );
    expect(merged?.inputTokens).toBe(1048);
    expect(merged?.outputTokens).toBe(1222);
    expect(merged?.totalTokens).toBe(2270);
  });

  test("incoming undefined keeps existing", () => {
    const first = new LLMTokenUsage({ input_tokens: 10, output_tokens: 2 });
    expect(mergeTokenUsage(first, undefined)).toBe(first);
  });

  test("preserves an explicit total_tokens", () => {
    const first = new LLMTokenUsage({
      input_tokens: 1,
      output_tokens: 1,
      total_tokens: 99,
    });
    const merged = mergeTokenUsage(first, first);
    expect(merged?.totalTokens).toBe(99);
    expect(merged?.inputTokens).toBe(1);
    expect(merged?.outputTokens).toBe(1);
  });
});
