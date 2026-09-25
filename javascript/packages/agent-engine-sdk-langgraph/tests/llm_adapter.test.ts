/**
 * Port of `agent-engine-sdk-langgraph/tests/test_llm_adapter.py`.
 *
 * Tests cover constructor (bindTools wiring), invoke (Message → LLMResponse),
 * and streaming (chunks, tool_call_chunks, dict args, usage, metadata).
 *
 * Python had separate sync/async paths (`invoke`/`ainvoke`); the TS adapter
 * collapses both into a single async `invoke`. The `test_ainvoke_falls_back_*`
 * and `test_astream_falls_back_*` Python tests have no TS analog and are
 * intentionally not ported.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { AIMessage, AIMessageChunk } from "@langchain/core/messages";
import { LLMToolSchema } from "@mongodb-js/agent-engine-sdk";
import type { Message } from "@mongodb-js/agent-engine-sdk";

import { LangChainLLMAdapter } from "../src/llm_adapter.js";

beforeEach(() => {
  if (!process.env["RUNNER_MODE"]) {
    process.env["RUNNER_MODE"] = "aer";
  }
});

// ---------------------------------------------------------------------------
// Constructor — bindTools wiring
// ---------------------------------------------------------------------------

describe("LangChainLLMAdapter — constructor", () => {
  it("calls bindTools when tools are provided", () => {
    // Python: test_binds_tools_when_provided.
    const bound = { invoke: vi.fn() };
    const llm = {
      invoke: vi.fn(),
      bindTools: vi.fn().mockReturnValue(bound),
    };
    const tool = new LLMToolSchema({
      name: "search",
      description: "search",
      parameters: { type: "object" },
    });

    const adapter = new LangChainLLMAdapter(llm as never, [tool]);

    expect(llm.bindTools).toHaveBeenCalledOnce();
    const [serialized] = llm.bindTools.mock.calls[0] ?? [];
    expect(serialized[0]).toMatchObject({
      name: "search",
      description: "search",
    });
    expect((adapter as unknown as { llm: unknown }).llm).toBe(bound);
  });

  it("does NOT call bindTools when tools are omitted", () => {
    // Python: test_no_bind_tools_when_none.
    const llm = { invoke: vi.fn(), bindTools: vi.fn() };

    const adapter = new LangChainLLMAdapter(llm as never);

    expect(llm.bindTools).not.toHaveBeenCalled();
    expect((adapter as unknown as { llm: unknown }).llm).toBe(llm);
  });

  it("throws when tools are provided but the LLM has no bindTools method", () => {
    // TS-specific assertion — the adapter must fail loudly per Python parity.
    const llm = { invoke: vi.fn() };
    const tool = new LLMToolSchema({
      name: "search",
      description: "search",
      parameters: { type: "object" },
    });
    expect(() => new LangChainLLMAdapter(llm as never, [tool])).toThrow(
      /tool binding/,
    );
  });

  it("forwards tool_choice to bindTools so the model is forced to call it", () => {
    // The tool-pod force point: tool_choice must reach the underlying
    // bindTools call, otherwise a forced structured-output selection is offered
    // but never required.
    const bound = { invoke: vi.fn() };
    const llm = {
      invoke: vi.fn(),
      bindTools: vi.fn().mockReturnValue(bound),
    };
    const tool = new LLMToolSchema({
      name: "Brief",
      description: "brief",
      parameters: { type: "object" },
    });

    new LangChainLLMAdapter(llm as never, [tool], "Brief");

    const [, kwargs] = llm.bindTools.mock.calls[0] ?? [];
    expect(kwargs).toEqual({ tool_choice: "Brief" });
  });

  it("omits the tool_choice kwarg when no choice is given", () => {
    const bound = { invoke: vi.fn() };
    const llm = {
      invoke: vi.fn(),
      bindTools: vi.fn().mockReturnValue(bound),
    };
    const tool = new LLMToolSchema({
      name: "search",
      description: "search",
      parameters: { type: "object" },
    });

    new LangChainLLMAdapter(llm as never, [tool]);

    const [, kwargs] = llm.bindTools.mock.calls[0] ?? [];
    expect(kwargs).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// invoke — message conversion + response shape
// ---------------------------------------------------------------------------

describe("LangChainLLMAdapter.invoke", () => {
  it("converts Messages to LangChain, calls invoke, returns LLMResponse", async () => {
    // Python: test_converts_messages_and_returns_llm_response.
    const llm = {
      invoke: vi.fn().mockResolvedValue(new AIMessage({ content: "Hello!" })),
    };
    const adapter = new LangChainLLMAdapter(llm as never);
    const userMsg: Message = { role: "user", content: "hi" };

    const result = await adapter.invoke([userMsg]);

    expect(result.content).toBe("Hello!");
    expect(llm.invoke).toHaveBeenCalledOnce();
  });

  it("returns LLMResponse with usage when usage_metadata is present", async () => {
    // Python: test_async_invoke_returns_llm_response (TS sync/async collapsed).
    // Mock with a plain duck-typed object — `responseToLlmResponse` reads
    // properties via index access, not via AIMessage instance methods.
    const responseMsg = {
      content: "Async hello!",
      tool_calls: [],
      usage_metadata: {
        input_tokens: 5,
        output_tokens: 10,
        total_tokens: 15,
      },
    };
    const llm = { invoke: vi.fn().mockResolvedValue(responseMsg) };
    const adapter = new LangChainLLMAdapter(llm as never);

    const result = await adapter.invoke([{ role: "user", content: "hi" }]);
    expect(result.content).toBe("Async hello!");
    expect(result.usage).toBeDefined();
    // LLMTokenUsage class stores camelCase fields.
    expect(
      (result.usage as unknown as { inputTokens: number }).inputTokens,
    ).toBe(5);
  });

  it("accepts a usage object in response_metadata", async () => {
    const responseMsg = {
      content: "Hello!",
      tool_calls: [],
      usage_metadata: null,
      response_metadata: {
        usage: { input_tokens: 1048, output_tokens: 1222 },
      },
    };
    const llm = { invoke: vi.fn().mockResolvedValue(responseMsg) };
    const adapter = new LangChainLLMAdapter(llm as never);

    const result = await adapter.invoke([{ role: "user", content: "hi" }]);
    expect(result.content).toBe("Hello!");
    expect(
      (result.usage as unknown as { inputTokens: number }).inputTokens,
    ).toBe(1048);
    expect(
      (result.usage as unknown as { outputTokens: number }).outputTokens,
    ).toBe(1222);
  });

  it("preserves tool_calls on the response", async () => {
    // Python: test_ainvoke_with_tool_calls.
    const responseMsg = {
      content: "",
      tool_calls: [
        { id: "tc1", name: "search", args: { q: "test" }, type: "tool_call" },
      ],
    };
    const llm = { invoke: vi.fn().mockResolvedValue(responseMsg) };
    const adapter = new LangChainLLMAdapter(llm as never);

    const result = await adapter.invoke([
      { role: "user", content: "search for test" },
    ]);

    // LLMResponse class stores camelCase `toolCalls`.
    const toolCalls = (
      result as unknown as { toolCalls?: Array<{ name?: string }> }
    ).toolCalls;
    expect(toolCalls).toBeDefined();
    expect(toolCalls?.length).toBe(1);
    expect(toolCalls?.[0]?.name).toBe("search");
  });
});

// ---------------------------------------------------------------------------
// stream — chunk conversion
// ---------------------------------------------------------------------------

async function* yieldChunks(
  chunks: ReadonlyArray<AIMessageChunk>,
): AsyncIterable<AIMessageChunk> {
  for (const c of chunks) yield c;
}

describe("LangChainLLMAdapter.stream", () => {
  it("yields LLMStreamChunk objects in order", async () => {
    // Python: test_yields_llm_stream_chunks.
    const llm = {
      invoke: vi.fn(),
      stream: vi
        .fn()
        .mockReturnValue(
          yieldChunks([
            new AIMessageChunk({ content: "Hello" }),
            new AIMessageChunk({ content: " world" }),
          ]),
        ),
    };
    const adapter = new LangChainLLMAdapter(llm as never);

    const out: { content?: string }[] = [];
    for await (const chunk of adapter.stream([
      { role: "user", content: "hi" },
    ])) {
      out.push(chunk);
    }

    expect(out).toHaveLength(2);
    expect(out[0]?.content).toBe("Hello");
    expect(out[1]?.content).toBe(" world");
  });

  it("converts tool_call_chunks with name + index preserved", async () => {
    // Python: test_stream_with_tool_call_chunks.
    const chunk = new AIMessageChunk({
      content: "",
      tool_call_chunks: [
        {
          id: "tc1",
          name: "search",
          args: '{"q":',
          index: 0,
          type: "tool_call_chunk",
        },
      ],
    });
    const llm = {
      invoke: vi.fn(),
      stream: vi.fn().mockReturnValue(yieldChunks([chunk])),
    };
    const adapter = new LangChainLLMAdapter(llm as never);

    const out: Array<{
      toolCalls?: Array<{ name?: string; index?: number }>;
    }> = [];
    for await (const c of adapter.stream([{ role: "user", content: "hi" }])) {
      out.push(c as { toolCalls?: Array<{ name?: string; index?: number }> });
    }

    expect(out).toHaveLength(1);
    // Adapter must emit the camelCase LLMStreamChunk contract, not snake_case.
    expect(out[0]).not.toHaveProperty("tool_calls");
    expect(out[0]?.toolCalls).toBeDefined();
    expect(out[0]?.toolCalls?.[0]?.name).toBe("search");
    expect(out[0]?.toolCalls?.[0]?.index).toBe(0);
  });

  it("JSON-encodes structured tool_call_chunk args", async () => {
    // Python: test_stream_with_tool_call_chunk_dict_args.
    const chunk = new AIMessageChunk({
      content: "",
      tool_call_chunks: [
        {
          id: "tc1",
          name: "search",
          args: JSON.stringify({ q: "test" }),
          index: 0,
          type: "tool_call_chunk",
        },
      ],
    });
    const llm = {
      invoke: vi.fn(),
      stream: vi.fn().mockReturnValue(yieldChunks([chunk])),
    };
    const adapter = new LangChainLLMAdapter(llm as never);

    const out: Array<{ toolCalls?: Array<{ args?: string }> }> = [];
    for await (const c of adapter.stream([{ role: "user", content: "hi" }])) {
      out.push(c as { toolCalls?: Array<{ args?: string }> });
    }

    expect(out[0]?.toolCalls?.[0]?.args).toBe(JSON.stringify({ q: "test" }));
  });

  it("extracts usage from chunks", async () => {
    // Python: test_stream_with_usage.
    // Plain duck-typed chunk — see note on the invoke usage test above.
    const chunk = {
      content: "done",
      tool_call_chunks: [],
      tool_calls: [],
      usage_metadata: {
        input_tokens: 1,
        output_tokens: 2,
        total_tokens: 3,
      },
    } as unknown as AIMessageChunk;
    const llm = {
      invoke: vi.fn(),
      stream: vi.fn().mockReturnValue(yieldChunks([chunk])),
    };
    const adapter = new LangChainLLMAdapter(llm as never);

    const out: Array<{ usage?: unknown }> = [];
    for await (const c of adapter.stream([{ role: "user", content: "hi" }])) {
      out.push(c);
    }
    expect(out[0]?.usage).toBeDefined();
    // LLMTokenUsage uses camelCase fields.
    expect((out[0]?.usage as { inputTokens: number }).inputTokens).toBe(1);
  });

  it("keeps partial usage on each stream chunk", async () => {
    const early = {
      content: "Hel",
      tool_call_chunks: [],
      tool_calls: [],
      usage_metadata: { input_tokens: 1048 },
    } as unknown as AIMessageChunk;
    const late = {
      content: "lo",
      tool_call_chunks: [],
      tool_calls: [],
      usage_metadata: { output_tokens: 1222 },
    } as unknown as AIMessageChunk;
    const llm = {
      invoke: vi.fn(),
      stream: vi.fn().mockReturnValue(yieldChunks([early, late])),
    };
    const adapter = new LangChainLLMAdapter(llm as never);

    const out: Array<{
      usage?: { inputTokens?: number; outputTokens?: number };
    }> = [];
    for await (const c of adapter.stream([{ role: "user", content: "hi" }])) {
      out.push(c);
    }
    expect(out).toHaveLength(2);
    expect(out[0]?.usage?.inputTokens).toBe(1048);
    expect(out[1]?.usage?.inputTokens).toBe(1048);
    expect(out[1]?.usage?.outputTokens).toBe(1222);
  });

  it("accumulates zero-filled LangChain usage deltas", async () => {
    const early = {
      content: "Hel",
      tool_call_chunks: [],
      tool_calls: [],
      usage_metadata: {
        input_tokens: 18,
        output_tokens: 1,
        total_tokens: 19,
      },
    } as unknown as AIMessageChunk;
    const late = {
      content: "lo",
      tool_call_chunks: [],
      tool_calls: [],
      usage_metadata: {
        input_tokens: 0,
        output_tokens: 4,
        total_tokens: 4,
      },
    } as unknown as AIMessageChunk;
    const llm = {
      invoke: vi.fn(),
      stream: vi.fn().mockReturnValue(yieldChunks([early, late])),
    };
    const adapter = new LangChainLLMAdapter(llm as never);

    const out: Array<{
      usage?: {
        inputTokens?: number;
        outputTokens?: number;
        totalTokens?: number;
      };
    }> = [];
    for await (const c of adapter.stream([{ role: "user", content: "hi" }])) {
      out.push(c);
    }
    expect(out[1]?.usage?.inputTokens).toBe(18);
    expect(out[1]?.usage?.outputTokens).toBe(5);
    expect(out[1]?.usage?.totalTokens).toBe(23);
  });

  it("takes cumulative usage snapshots without adding them", async () => {
    const early = {
      content: "Hel",
      tool_call_chunks: [],
      tool_calls: [],
      usage_metadata: {
        input_tokens: 10,
        output_tokens: 1,
        total_tokens: 11,
      },
    } as unknown as AIMessageChunk;
    const late = {
      content: "lo",
      tool_call_chunks: [],
      tool_calls: [],
      usage_metadata: {
        input_tokens: 10,
        output_tokens: 2,
        total_tokens: 12,
      },
    } as unknown as AIMessageChunk;
    const llm = {
      invoke: vi.fn(),
      stream: vi.fn().mockReturnValue(yieldChunks([early, late])),
    };
    const adapter = new LangChainLLMAdapter(llm as never);

    const out: Array<{
      usage?: {
        inputTokens?: number;
        outputTokens?: number;
        totalTokens?: number;
      };
    }> = [];
    for await (const c of adapter.stream([{ role: "user", content: "hi" }])) {
      out.push(c);
    }
    expect(out[1]?.usage?.inputTokens).toBe(10);
    expect(out[1]?.usage?.outputTokens).toBe(2);
    expect(out[1]?.usage?.totalTokens).toBe(12);
  });

  it("normalizes content from a list of text blocks", async () => {
    // Python: test_stream_with_list_content_blocks.
    const chunk = new AIMessageChunk({
      content: [
        { type: "text", text: "Hello " },
        { type: "text", text: "world" },
      ],
    });
    const llm = {
      invoke: vi.fn(),
      stream: vi.fn().mockReturnValue(yieldChunks([chunk])),
    };
    const adapter = new LangChainLLMAdapter(llm as never);

    const out: Array<{ content?: string }> = [];
    for await (const c of adapter.stream([{ role: "user", content: "hi" }])) {
      out.push(c);
    }
    expect(out[0]?.content).toBe("Hello world");
  });

  it("preserves a single string content as-is", async () => {
    // Python: test_stream_with_single_text_content_block.
    const chunk = new AIMessageChunk({ content: "Hello" });
    const llm = {
      invoke: vi.fn(),
      stream: vi.fn().mockReturnValue(yieldChunks([chunk])),
    };
    const adapter = new LangChainLLMAdapter(llm as never);

    const out: Array<{ content?: string }> = [];
    for await (const c of adapter.stream([{ role: "user", content: "hi" }])) {
      out.push(c);
    }
    expect(out[0]?.content).toBe("Hello");
  });

  it("preserves response_metadata on chunks", async () => {
    // Python: test_stream_preserves_message_metadata.
    const chunk = new AIMessageChunk({
      content: "ok",
      response_metadata: { model_name: "gpt-4o" },
    });
    const llm = {
      invoke: vi.fn(),
      stream: vi.fn().mockReturnValue(yieldChunks([chunk])),
    };
    const adapter = new LangChainLLMAdapter(llm as never);

    const out: Array<{ responseMetadata?: Record<string, unknown> }> = [];
    for await (const c of adapter.stream([{ role: "user", content: "hi" }])) {
      out.push(c as { responseMetadata?: Record<string, unknown> });
    }
    expect(out[0]?.responseMetadata).toEqual({ model_name: "gpt-4o" });
  });

  it("accepts a usage object in stream response_metadata", async () => {
    const chunk = {
      content: "Hello",
      tool_call_chunks: [],
      tool_calls: [],
      usage_metadata: null,
      response_metadata: {
        usage: { input_tokens: 1048, output_tokens: 1222 },
      },
    } as unknown as AIMessageChunk;
    const llm = {
      invoke: vi.fn(),
      stream: vi.fn().mockReturnValue(yieldChunks([chunk])),
    };
    const adapter = new LangChainLLMAdapter(llm as never);

    const out: Array<{
      content?: string;
      usage?: { inputTokens?: number };
      responseMetadata?: Record<string, unknown>;
    }> = [];
    for await (const c of adapter.stream([{ role: "user", content: "hi" }])) {
      out.push(c);
    }
    expect(out[0]?.content).toBe("Hello");
    expect(out[0]?.usage?.inputTokens).toBe(1048);
    expect(
      (out[0]?.responseMetadata?.["usage"] as { input_tokens?: number })[
        "input_tokens"
      ],
    ).toBe(1048);
  });

  it("throws when the LLM has no stream method", async () => {
    // Python equivalent: test_astream_falls_back_to_sync_stream — TS does not
    // have a sync fallback, so the adapter must throw loudly.
    const llm = { invoke: vi.fn() };
    const adapter = new LangChainLLMAdapter(llm as never);

    const iter = adapter.stream([{ role: "user", content: "hi" }]);
    await expect(
      (async () => {
        for await (const _c of iter) break;
      })(),
    ).rejects.toThrow(/streaming/);
  });
});
