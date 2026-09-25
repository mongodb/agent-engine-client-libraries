/**
 * Tests for SecureWrappedLLM.
 *
 * Combines the original four bindTools-normalization tests with ports of the
 * Python `test_secure_llm.py` tests covering construction, model-name
 * resolution, missing-wrapper raising, and `bindTools` delegation. Deep proxy
 * behavior (policy denial, metrics, cached results, tool-message repair) is
 * exercised in integration tests where mocking SecureLLMProxy is cheap.
 *
 * Proxy-routing cases that need a module-level SecureLLMProxy mock live in
 * `secure_llm_integration.test.ts`:
 * - test_generate_routes_through_proxy_with_stop_and_kwargs
 * - test_generate_raises_on_policy_denial
 * - test_generate_records_latency_metric / error metric
 * - test_generate_logs_cached_results_from_proxy
 * - test_stream_* (same shape as generate)
 */

import { describe, test, expect, vi, beforeEach } from "vitest";
import { z } from "zod";
import { tool } from "@langchain/core/tools";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { LLMToolSchema } from "@mongodb-js/agent-engine-sdk";

import {
  SecureWrappedLLM,
  repairMissingToolMessages,
} from "../src/secure_llm.js";

beforeEach(() => {
  if (!process.env["RUNNER_MODE"]) {
    process.env["RUNNER_MODE"] = "aer";
  }
});

function makeStubInnerLlm(): BaseChatModel {
  const stub = {
    bindTools: vi.fn(function bindToolsStub(this: unknown) {
      return stub;
    }),
    model: "gpt-4o",
  };
  return stub as unknown as BaseChatModel;
}

describe("SecureWrappedLLM.bindTools — tool normalization", () => {
  test("LangChain JS tool() is converted to OpenAI shape", () => {
    const greetingTool = tool(
      async ({ name }: { name: string }) => `Hello, ${name}!`,
      {
        name: "get_greeting",
        description: "Returns a greeting for a name.",
        schema: z.object({ name: z.string() }),
      },
    );

    const wrapped = new SecureWrappedLLM(makeStubInnerLlm(), () => null);
    const next = wrapped.bindTools([greetingTool]);

    const bound = (next as unknown as { boundTools: unknown[] }).boundTools;
    expect(bound).toHaveLength(1);
    const entry = bound[0] as Record<string, unknown>;

    expect(entry["type"]).toBe("function");
    const fn = entry["function"] as Record<string, unknown>;
    expect(fn["name"]).toBe("get_greeting");
    expect(fn["description"]).toBe("Returns a greeting for a name.");
    const params = fn["parameters"] as Record<string, unknown>;
    expect(params["type"]).toBe("object");
    expect(params["properties"]).toMatchObject({ name: { type: "string" } });
  });

  test("plain dict tools pass through unchanged", () => {
    const plain = {
      name: "calc",
      description: "Calculate",
      parameters: { type: "object" },
    };

    const wrapped = new SecureWrappedLLM(makeStubInnerLlm(), () => null);
    const next = wrapped.bindTools([plain]);

    const bound = (next as unknown as { boundTools: unknown[] }).boundTools;
    expect(bound).toHaveLength(1);
    expect(bound[0]).toBe(plain);
  });

  test("LLMToolSchema instances pass through unchanged", () => {
    const schema = new LLMToolSchema({
      name: "lookup",
      description: "Look something up",
      parameters: { type: "object" },
    });

    const wrapped = new SecureWrappedLLM(makeStubInnerLlm(), () => null);
    const next = wrapped.bindTools([schema]);

    const bound = (next as unknown as { boundTools: unknown[] }).boundTools;
    expect(bound).toHaveLength(1);
    expect(bound[0]).toBe(schema);
  });

  test("inner LLM receives the original tools, not the normalized dicts", () => {
    const greetingTool = tool(async () => "ok", {
      name: "get_greeting",
      description: "Greeting",
      schema: z.object({ name: z.string() }),
    });

    const innerBindTools = vi.fn(function (
      this: unknown,
      _tools: ReadonlyArray<unknown>,
    ) {
      return this;
    });
    const innerLlm = {
      bindTools: innerBindTools,
      model: "gpt-4o",
    } as unknown as BaseChatModel;

    const wrapped = new SecureWrappedLLM(innerLlm, () => null);
    wrapped.bindTools([greetingTool]);

    expect(innerBindTools).toHaveBeenCalledTimes(1);
    const [forwardedTools] = innerBindTools.mock.calls[0] ?? [];
    expect(forwardedTools).toEqual([greetingTool]);
  });
});

describe("SecureWrappedLLM.bindTools — tool_choice", () => {
  const getChoice = (llm: SecureWrappedLLM): unknown =>
    (llm as unknown as { boundToolChoice: unknown }).boundToolChoice;

  test("captures tool_choice from kwargs so the proxy can forward it", () => {
    const wrapped = new SecureWrappedLLM(makeStubInnerLlm(), () => null);
    const next = wrapped.bindTools([{ name: "Brief" }], {
      tool_choice: "Brief",
    });

    expect(getChoice(next)).toBe("Brief");
  });

  test("defaults boundToolChoice to null when no kwargs are passed", () => {
    const wrapped = new SecureWrappedLLM(makeStubInnerLlm(), () => null);
    const next = wrapped.bindTools([{ name: "search" }]);

    expect(getChoice(next)).toBeNull();
  });

  test("preserves tool_choice=false (disables forced tool use)", () => {
    const wrapped = new SecureWrappedLLM(makeStubInnerLlm(), () => null);
    const next = wrapped.bindTools([{ name: "search" }], {
      tool_choice: false,
    });

    expect(getChoice(next)).toBe(false);
  });

  test("forwards kwargs to the inner LLM's bindTools", () => {
    const innerBindTools = vi.fn(function (
      this: unknown,
      _tools: ReadonlyArray<unknown>,
      _kwargs?: Record<string, unknown>,
    ) {
      return this;
    });
    const innerLlm = {
      bindTools: innerBindTools,
      model: "gpt-4o",
    } as unknown as BaseChatModel;

    const wrapped = new SecureWrappedLLM(innerLlm, () => null);
    wrapped.bindTools([{ name: "Brief" }], { tool_choice: "Brief" });

    const [, forwardedKwargs] = innerBindTools.mock.calls[0] ?? [];
    expect(forwardedKwargs).toEqual({ tool_choice: "Brief" });
  });
});

// ---------------------------------------------------------------------------
// Construction — model name resolution
// ---------------------------------------------------------------------------

describe("SecureWrappedLLM — construction", () => {
  test("resolves modelName from inner LLM's .model attribute", () => {
    // Python: test_model_name_from_model_name_attr (TS uses .model first).
    const inner = {
      model: "gpt-4o",
      invoke: vi.fn(),
    } as unknown as BaseChatModel;
    const wrapped = new SecureWrappedLLM(inner, () => null);
    expect(wrapped.modelName).toBe("gpt-4o");
  });

  test("falls back to inner LLM's .modelName attribute when .model is missing", () => {
    const inner = {
      modelName: "claude-3-5",
      invoke: vi.fn(),
    } as unknown as BaseChatModel;
    const wrapped = new SecureWrappedLLM(inner, () => null);
    expect(wrapped.modelName).toBe("claude-3-5");
  });

  test("uses explicit modelName argument when provided", () => {
    const inner = {
      model: "ignored",
      invoke: vi.fn(),
    } as unknown as BaseChatModel;
    // 3rd arg = llmId, 4th arg = modelName override
    const wrapped = new SecureWrappedLLM(
      inner,
      () => null,
      "__default__",
      "explicit-model",
    );
    expect(wrapped.modelName).toBe("explicit-model");
  });

  test("defaults to 'unknown' when no name source is available", () => {
    // Python: test_construction (default model name fallback).
    const inner = { invoke: vi.fn() } as unknown as BaseChatModel;
    const wrapped = new SecureWrappedLLM(inner, () => null);
    expect(wrapped.modelName).toBe("unknown");
  });
});

// ---------------------------------------------------------------------------
// bindTools — delegation paths
// ---------------------------------------------------------------------------

describe("SecureWrappedLLM.bindTools — delegation", () => {
  test("delegates to inner LLM's bindTools when supported", () => {
    // Python: test_bind_tools_delegates_when_supported.
    const innerBound = { invoke: vi.fn() };
    const inner = {
      invoke: vi.fn(),
      bindTools: vi.fn().mockReturnValue(innerBound),
    } as unknown as BaseChatModel;

    const wrapped = new SecureWrappedLLM(inner, () => null);
    const greetingTool = tool(async () => "ok", {
      name: "g",
      description: "greet",
      schema: z.object({ name: z.string() }),
    });
    const next = wrapped.bindTools([greetingTool]);

    // The new wrapper holds the inner-bound LLM, not the original inner.
    expect((next as unknown as { llm: unknown }).llm).toBe(innerBound);
  });

  test("falls back to the original inner LLM when bindTools is not supported", () => {
    // Python: test_bind_tools_falls_back_when_not_supported.
    const inner = { invoke: vi.fn() } as unknown as BaseChatModel;
    const wrapped = new SecureWrappedLLM(inner, () => null);
    const next = wrapped.bindTools([
      tool(async () => "ok", {
        name: "g",
        description: "greet",
        schema: z.object({ name: z.string() }),
      }),
    ]);

    expect((next as unknown as { llm: unknown }).llm).toBe(inner);
  });

  test("bindTools returns a new instance — does not mutate the original", () => {
    const inner = makeStubInnerLlm();
    const wrapped = new SecureWrappedLLM(inner, () => null);
    const next = wrapped.bindTools([]);
    expect(next).not.toBe(wrapped);
    expect(next).toBeInstanceOf(SecureWrappedLLM);
  });
});

// ---------------------------------------------------------------------------
// _generate / _streamResponseChunks — missing-wrapper guard
// ---------------------------------------------------------------------------

describe("SecureWrappedLLM — wrapper guard", () => {
  test("_generate raises when no wrapper is available", async () => {
    // Python: test_generate_raises_without_wrapper.
    const inner = { invoke: vi.fn() } as unknown as BaseChatModel;
    const wrapped = new SecureWrappedLLM(inner, () => null);
    await expect(
      (
        wrapped as unknown as {
          _generate: (m: unknown[], o: unknown) => Promise<unknown>;
        }
      )._generate([], {} as unknown),
    ).rejects.toThrow(/SecureToolWrapper is not initialized/);
  });

  test("_streamResponseChunks raises when no wrapper is available", async () => {
    // Python: test_stream_raises_without_wrapper.
    const inner = {
      invoke: vi.fn(),
      stream: vi.fn(),
    } as unknown as BaseChatModel;
    const wrapped = new SecureWrappedLLM(inner, () => null);
    const gen = (
      wrapped as unknown as {
        _streamResponseChunks: (
          m: unknown[],
          o: unknown,
        ) => AsyncGenerator<unknown>;
      }
    )._streamResponseChunks([], {} as unknown);
    await expect(gen.next()).rejects.toThrow(
      /SecureToolWrapper is not initialized/,
    );
  });
});

// ---------------------------------------------------------------------------
// repairMissingToolMessages — AIMessageChunk parity
// ---------------------------------------------------------------------------

describe("repairMissingToolMessages — AIMessageChunk", () => {
  test("inserts missing ToolMessage after AIMessageChunk with tool calls", () => {
    // Regression: instanceof AIMessage was false for AIMessageChunk, so the
    // repair logic silently skipped chunks and sent broken history to the LLM.
    const chunk = new AIMessageChunk({
      content: "",
      tool_calls: [
        { id: "call-1", name: "my_tool", args: {}, type: "tool_call" },
      ],
    });

    const repaired = repairMissingToolMessages([chunk]);

    expect(repaired).toHaveLength(2);
    expect(repaired[1]).toBeInstanceOf(ToolMessage);
    const inserted = repaired[1] as ToolMessage;
    expect(inserted.tool_call_id).toBe("call-1");
    expect(inserted.status).toBe("error");
  });

  test("does not insert ToolMessage when matching ToolMessage already present after AIMessageChunk", () => {
    const chunk = new AIMessageChunk({
      content: "",
      tool_calls: [
        { id: "call-2", name: "my_tool", args: {}, type: "tool_call" },
      ],
    });
    const toolMsg = new ToolMessage({ content: "ok", tool_call_id: "call-2" });

    const repaired = repairMissingToolMessages([chunk, toolMsg]);

    expect(repaired).toHaveLength(2);
    expect(repaired[1]).toBe(toolMsg);
  });

  test("passes through non-AI messages unchanged", () => {
    const human = new HumanMessage({ content: "hello" });
    const ai = new AIMessage({ content: "hi" });

    const repaired = repairMissingToolMessages([human, ai]);

    expect(repaired).toHaveLength(2);
    expect(repaired[0]).toBe(human);
    expect(repaired[1]).toBe(ai);
  });
});
