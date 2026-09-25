/** Mirrors sdk-core/tests/test_models.py. */

import { describe, expect, it } from "vitest";
import {
  AgentInputSchema,
  AgentOutputSchema,
  AnyContentBlockSchema,
  BranchRefSchema,
  createToolDefinition,
  DocumentBlockSchema,
  EventSchema,
  ImageBlockSchema,
  LLMInvocationOptions,
  LLMInvocationOptionsSchema,
  LLMResponse,
  LLMResponseSchema,
  LLMTokenUsage,
  LLMTokenUsageSchema,
  LLMToolCall,
  LLMToolCallSchema,
  LLMToolSchema,
  LLMToolSchemaValidator,
  MessageSchema,
  serializeMessage,
  StreamEventSchema,
  TextBlockSchema,
  ToolDefinitionSchema,
} from "../src/index.js";
import type {
  AnyContentBlock,
  Message,
  RequestContext,
  ToolCallChunk,
  LLMStreamChunk,
  Event,
} from "../src/index.js";

describe("ToolDefinition", () => {
  it("construction", () => {
    const td = createToolDefinition({
      name: "search",
      description: "Search",
      args_schema: {},
      callable: () => null,
    });
    expect(td.name).toBe("search");
  });

  it("defaults match Pydantic parity", () => {
    const td = createToolDefinition({
      name: "t",
      description: "d",
      args_schema: {},
      callable: () => null,
    });
    expect(td.remote).toBe(true);
    expect(td.network).toEqual([]);
    expect(td.timeout_seconds).toBe(30);
    expect(td.redact_fields).toEqual([]);
  });

  it("round-trips through JSON minus non-serializable callable", () => {
    const td = createToolDefinition({
      name: "t",
      description: "d",
      args_schema: { type: "object" },
      callable: () => null,
    });
    const { callable: _omit, ...serializable } = td;
    const data = JSON.parse(JSON.stringify(serializable));
    const restored = ToolDefinitionSchema.parse({
      ...data,
      callable: () => null,
    });
    expect(restored.name).toBe(td.name);
    expect(restored.args_schema).toEqual(td.args_schema);
  });

  it("rejects missing required fields", () => {
    expect(() => ToolDefinitionSchema.parse({})).toThrow();
  });
});

describe("Message", () => {
  it("construction with string content", () => {
    const m: Message = { role: "user", content: "hello" };
    expect(m.role).toBe("user");
    expect(m.toolCalls).toBeUndefined();
    expect(m.toolCallId).toBeUndefined();
    expect(m.isError).toBeUndefined();
  });

  it("round-trips data through MessageSchema and maps to camelCase Message shape", () => {
    const wire = {
      role: "assistant" as const,
      content: "hi",
      tool_calls: [{ id: "tc1", name: "my_tool", args: { x: 1 } }],
      tool_call_id: "tc1",
      is_error: true,
      additional_kwargs: { raw: true },
      response_metadata: { model: "claude" },
    };
    const restored = MessageSchema.parse(wire);
    expect(restored.role).toBe("assistant");
    expect(restored.content).toBe("hi");
    expect(restored.toolCalls).toHaveLength(1);
    expect(restored.toolCalls?.[0]).toBeInstanceOf(LLMToolCall);
    expect(restored.toolCalls?.[0]?.name).toBe("my_tool");
    expect(restored.toolCallId).toBe("tc1");
    expect(restored.isError).toBe(true);
    expect(restored.additionalKwargs).toEqual({ raw: true });
    expect(restored.responseMetadata).toEqual({ model: "claude" });
  });

  it("serializes tool error status to the wire shape", () => {
    expect(
      serializeMessage({
        role: "tool",
        content: "failed",
        toolCallId: "tc1",
        isError: true,
      }),
    ).toEqual({
      role: "tool",
      content: "failed",
      tool_call_id: "tc1",
      is_error: true,
    });
  });

  it("MessageSchema parses typed content blocks", () => {
    const wire = {
      role: "user" as const,
      content: [
        { type: "text", text: "Check this:" },
        {
          type: "document",
          url: "https://example.com/f.pdf",
          filename: "f.pdf",
        },
      ],
    };
    const restored = MessageSchema.parse(wire);
    expect(Array.isArray(restored.content)).toBe(true);
    const blocks = restored.content as AnyContentBlock[];
    expect(blocks[0]?.type).toBe("text");
    expect(blocks[1]?.type).toBe("document");
    if (blocks[1]?.type === "document")
      expect(blocks[1].filename).toBe("f.pdf");
  });

  it("rejects missing required fields at runtime", () => {
    expect(() => MessageSchema.parse({})).toThrow();
    expect(() => MessageSchema.parse({ role: "user" })).toThrow();
    expect(() => MessageSchema.parse({ content: "hi" })).toThrow();
  });
});

describe("LLMResponse", () => {
  it("construction", () => {
    const r = new LLMResponse({ content: "response" });
    expect(r.content).toBe("response");
    expect(r.toolCalls).toBeUndefined();
    expect(r.metadata).toEqual({});
  });

  it("round-trips via JSON + fromRaw", () => {
    const r = new LLMResponse({ content: "r", metadata: { k: "v" } });
    const data = JSON.parse(JSON.stringify(r));
    const restored = LLMResponse.fromRaw(data);
    expect(restored.metadata).toEqual({ k: "v" });
  });

  it("usage does not auto-populate metadata", () => {
    const r = new LLMResponse({
      content: "r",
      usage: new LLMTokenUsage({ input_tokens: 1, output_tokens: 2 }),
    });
    expect(r.usage).not.toBeUndefined();
    expect(r.metadata).toEqual({});
  });

  it("fromRaw normalizes usage from metadata", () => {
    const r = LLMResponse.fromRaw({
      content: "r",
      metadata: { usage: { input_tokens: 4, output_tokens: 5 } },
    });
    expect(r.usage).not.toBeUndefined();
    expect(r.usage?.totalTokens).toBe(9);
  });

  it("fromRaw normalizes flat token keys in metadata", () => {
    const r = LLMResponse.fromRaw({
      content: "r",
      metadata: { input_tokens: 3, output_tokens: 7 },
    });
    expect(r.usage?.totalTokens).toBe(10);
  });

  it("message metadata round-trips via fromRaw", () => {
    const r = new LLMResponse({
      content: "r",
      id: "run-1",
      name: "assistant",
      additionalKwargs: { refusal: null },
      responseMetadata: { finish_reason: "stop" },
    });
    const restored = LLMResponse.fromRaw(JSON.parse(JSON.stringify(r)));
    expect(restored.id).toBe("run-1");
    expect(restored.name).toBe("assistant");
    expect(restored.additionalKwargs).toEqual({ refusal: null });
    expect(restored.responseMetadata).toEqual({ finish_reason: "stop" });
  });

  it("rejects missing required fields", () => {
    expect(() => LLMResponseSchema.parse({})).toThrow();
    expect(() => LLMResponseSchema.parse({ content: 123 })).toThrow();
  });
});

describe("AgentInput", () => {
  it("construction", () => {
    const parsed = AgentInputSchema.parse({ payload: { message: "hi" } });
    expect(parsed.payload).toEqual({ message: "hi" });
  });

  it("round-trips data", () => {
    const data = { payload: { message: "hi", k: "v" } };
    const restored = AgentInputSchema.parse(JSON.parse(JSON.stringify(data)));
    expect(restored.payload).toEqual({ message: "hi", k: "v" });
  });

  it("rejects missing payload", () => {
    expect(() => AgentInputSchema.parse({})).toThrow();
  });

  it("accepts null payload", () => {
    expect(() => AgentInputSchema.parse({ payload: null })).not.toThrow();
  });
});

describe("AgentOutput", () => {
  it("construction", () => {
    const parsed = AgentOutputSchema.parse({ response: { response: "ok" } });
    expect(parsed.response).toEqual({ response: "ok" });
  });

  it("round-trips data", () => {
    const data = { response: { response: "ok", k: "v" } };
    const restored = AgentOutputSchema.parse(JSON.parse(JSON.stringify(data)));
    expect(restored.response).toEqual({ response: "ok", k: "v" });
  });

  it("rejects missing response", () => {
    expect(() => AgentOutputSchema.parse({})).toThrow();
  });
});

describe("StreamEvent", () => {
  it("construction without event tag", () => {
    const e = StreamEventSchema.parse({ data: { content: "hello" } });
    expect(e.data).toEqual({ content: "hello" });
    expect(e.event).toBeUndefined();
  });

  it("with event tag", () => {
    const e = StreamEventSchema.parse({
      data: { content: "hello" },
      event: "token",
    });
    expect(e.event).toBe("token");
  });

  it("round-trips data", () => {
    const data = { data: { content: "hello", k: "v" }, event: "token" };
    const restored = StreamEventSchema.parse(JSON.parse(JSON.stringify(data)));
    expect(restored.data).toEqual({ content: "hello", k: "v" });
  });

  it("rejects missing data", () => {
    expect(() => StreamEventSchema.parse({})).toThrow();
  });

  it("rejects CR/LF in the event name so SSE frames cannot be forged", () => {
    expect(() =>
      StreamEventSchema.parse({
        data: { content: "hi" },
        event: "token\ndata: {}",
      }),
    ).toThrow();
    expect(() =>
      StreamEventSchema.parse({
        data: { content: "hi" },
        event: "token\r\nevent: fake",
      }),
    ).toThrow();
  });
});

describe("BranchRef", () => {
  it("construction", () => {
    const br = BranchRefSchema.parse({
      name: "flight-search",
      root_event_id: "evt_001",
    });
    expect(br.name).toBe("flight-search");
    expect(br.root_event_id).toBe("evt_001");
  });

  it("round-trips", () => {
    const data = { name: "b1", root_event_id: "r1" };
    const restored = BranchRefSchema.parse(JSON.parse(JSON.stringify(data)));
    expect(restored.name).toBe("b1");
    expect(restored.root_event_id).toBe("r1");
  });
});

describe("Event", () => {
  const baseEvent = {
    event_id: "evt_001",
    session_id: "sess-1",
    actor_id: "user",
    payload: { text: "hello" },
    timestamp: "2024-01-15T10:30:00Z",
  };

  it("construction with required fields", () => {
    const e = EventSchema.parse(baseEvent);
    expect(e.event_id).toBe("evt_001");
    expect(e.session_id).toBe("sess-1");
    expect(e.parent_event_id).toBeUndefined();
    expect(e.branch).toBeUndefined();
    expect(e.metadata).toBeUndefined();
  });

  it("parses timestamp to Date", () => {
    const e: Event = EventSchema.parse(baseEvent);
    expect(e.timestamp).toBeInstanceOf(Date);
    expect(e.timestamp.toISOString()).toBe("2024-01-15T10:30:00.000Z");
  });

  it("with branch", () => {
    const e = EventSchema.parse({
      ...baseEvent,
      event_id: "evt_002",
      actor_id: "agent:flight",
      payload: { type: "search_started" },
      branch: { name: "flight-search", root_event_id: "evt_001" },
    });
    expect(e.branch).not.toBeUndefined();
    expect(e.branch?.name).toBe("flight-search");
  });

  it("round-trips full event", () => {
    const data = {
      event_id: "evt_001",
      session_id: "sess-1",
      parent_event_id: "evt_000",
      actor_id: "agent:flight",
      payload: { tool: "search" },
      metadata: { step: "search" },
      timestamp: "2024-01-15T10:30:00Z",
    };
    const e = EventSchema.parse(data);
    const dumped = { ...e, timestamp: e.timestamp.toISOString() };
    const restored = EventSchema.parse(JSON.parse(JSON.stringify(dumped)));
    expect(restored.event_id).toBe(e.event_id);
    expect(restored.metadata).toEqual(e.metadata);
    expect(restored.timestamp.toISOString()).toBe(e.timestamp.toISOString());
  });
});

describe("TextBlock", () => {
  it("construction defaults type", () => {
    const tb = TextBlockSchema.parse({ text: "Hello world" });
    expect(tb.type).toBe("text");
    expect(tb.text).toBe("Hello world");
  });

  it("round-trips", () => {
    const tb = TextBlockSchema.parse({ text: "test content" });
    const restored = TextBlockSchema.parse(JSON.parse(JSON.stringify(tb)));
    expect(restored.text).toBe("test content");
    expect(restored.type).toBe("text");
  });

  it("rejects missing text", () => {
    expect(() => TextBlockSchema.parse({})).toThrow();
  });
});

describe("ImageBlock", () => {
  it("construction with url", () => {
    const ib = ImageBlockSchema.parse({ url: "https://example.com/img.png" });
    expect(ib.type).toBe("image");
    expect(ib.url).toBe("https://example.com/img.png");
    expect(ib.mime_type).toBeUndefined();
  });

  it("construction with mime_type", () => {
    const ib = ImageBlockSchema.parse({
      url: "https://example.com/img.png",
      mime_type: "image/png",
    });
    expect(ib.mime_type).toBe("image/png");
  });

  it("accepts data URL for base64", () => {
    const dataUrl = "data:image/png;base64,aGVsbG8=";
    const ib = ImageBlockSchema.parse({ url: dataUrl });
    expect(ib.url).toBe(dataUrl);
  });

  it("rejects missing url", () => {
    expect(() => ImageBlockSchema.parse({})).toThrow();
  });

  it("round-trips", () => {
    const ib = ImageBlockSchema.parse({
      url: "https://example.com/img.png",
      mime_type: "image/jpeg",
    });
    const restored = ImageBlockSchema.parse(JSON.parse(JSON.stringify(ib)));
    expect(restored.url).toBe("https://example.com/img.png");
    expect(restored.mime_type).toBe("image/jpeg");
  });
});

describe("DocumentBlock", () => {
  it("construction with url only", () => {
    const db = DocumentBlockSchema.parse({
      url: "https://example.com/doc.pdf",
    });
    expect(db.type).toBe("document");
    expect(db.url).toBe("https://example.com/doc.pdf");
    expect(db.mime_type).toBeUndefined();
    expect(db.filename).toBeUndefined();
  });

  it("construction with all fields", () => {
    const db = DocumentBlockSchema.parse({
      url: "https://example.com/doc.pdf",
      mime_type: "application/pdf",
      filename: "document.pdf",
    });
    expect(db.mime_type).toBe("application/pdf");
    expect(db.filename).toBe("document.pdf");
  });

  it("accepts data URL for base64", () => {
    const dataUrl = "data:application/pdf;base64,dGVzdA==";
    const db = DocumentBlockSchema.parse({ url: dataUrl });
    expect(db.url).toBe(dataUrl);
  });

  it("rejects missing url", () => {
    expect(() => DocumentBlockSchema.parse({})).toThrow();
  });

  it("round-trips", () => {
    const db = DocumentBlockSchema.parse({
      url: "https://example.com/doc.pdf",
      mime_type: "application/pdf",
      filename: "doc.pdf",
    });
    const restored = DocumentBlockSchema.parse(JSON.parse(JSON.stringify(db)));
    expect(restored.url).toBe("https://example.com/doc.pdf");
    expect(restored.filename).toBe("doc.pdf");
  });
});

describe("ToolCallChunk", () => {
  it("construction all undefined is valid", () => {
    const chunk: ToolCallChunk = {};
    expect(chunk.id).toBeUndefined();
    expect(chunk.name).toBeUndefined();
    expect(chunk.args).toBeUndefined();
  });

  it("construction with values", () => {
    const chunk: ToolCallChunk = {
      id: "call_123",
      name: "search",
      args: '{"query":',
    };
    expect(chunk.id).toBe("call_123");
    expect(chunk.name).toBe("search");
    expect(chunk.args).toBe('{"query":');
  });

  it("round-trips through JSON", () => {
    const chunk: ToolCallChunk = {
      id: "call_123",
      name: "search",
      args: '{"q": "test"}',
      type: "tool_call",
    };
    const restored = JSON.parse(JSON.stringify(chunk)) as ToolCallChunk;
    expect(restored.id).toBe("call_123");
    expect(restored.args).toBe('{"q": "test"}');
    expect(restored.type).toBe("tool_call");
  });
});

describe("LLMToolCall", () => {
  it("serialization emits args key, not arguments", () => {
    const tc = new LLMToolCall({
      id: "call_1",
      name: "search",
      args: { q: "test" },
    });
    const dict = tc.toLangchainDict();
    expect("args" in dict).toBe(true);
    expect("arguments" in dict).toBe(false);
    expect(dict["args"]).toEqual({ q: "test" });
  });

  it("accepts arguments alias on construction", () => {
    const tc = new LLMToolCall({ id: "c", arguments: { q: "x" } });
    expect(tc.args).toEqual({ q: "x" });
  });

  it("does not emit index in langchain dict", () => {
    const tc = new LLMToolCall({
      id: "c",
      name: "f",
      args: { q: "x" },
      index: 0,
    });
    const dict = tc.toLangchainDict();
    expect("index" in dict).toBe(false);
  });
});

describe("LLMStreamChunk", () => {
  it("empty chunk is valid", () => {
    const chunk: LLMStreamChunk = {};
    expect(chunk.content).toBeUndefined();
    expect(chunk.toolCalls).toBeUndefined();
    expect(chunk.usage).toBeUndefined();
  });

  it("content-only chunk", () => {
    const chunk: LLMStreamChunk = { content: "Hello" };
    expect(chunk.content).toBe("Hello");
    expect(chunk.toolCalls).toBeUndefined();
  });

  it("tool-call chunk", () => {
    const chunk: LLMStreamChunk = {
      toolCalls: [{ id: "call_1", name: "search" }],
    };
    expect(chunk.toolCalls).toHaveLength(1);
    expect(chunk.toolCalls?.[0]?.name).toBe("search");
  });

  it("usage chunk", () => {
    const chunk: LLMStreamChunk = {
      usage: new LLMTokenUsage({
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
      }),
    };
    expect(chunk.usage?.totalTokens).toBe(30);
  });

  it("message metadata chunk", () => {
    const chunk: LLMStreamChunk = {
      id: "run-1",
      name: "assistant",
      additionalKwargs: { refusal: null },
      responseMetadata: { finish_reason: "stop" },
    };
    expect(chunk.id).toBe("run-1");
    expect(chunk.name).toBe("assistant");
    expect(chunk.additionalKwargs).toEqual({ refusal: null });
    expect(chunk.responseMetadata).toEqual({ finish_reason: "stop" });
  });

  it("round-trips through JSON", () => {
    const chunk: LLMStreamChunk = {
      content: "Hi",
      toolCalls: [{ id: "c1", name: "fn" }],
    };
    const restored = JSON.parse(JSON.stringify(chunk)) as LLMStreamChunk;
    expect(restored.content).toBe("Hi");
    expect(restored.toolCalls?.[0]?.name).toBe("fn");
  });
});

describe("Message multimodal content", () => {
  it("string content works", () => {
    const m: Message = { role: "user", content: "Hello" };
    expect(m.content).toBe("Hello");
  });

  it("list of content blocks works", () => {
    const m: Message = {
      role: "user",
      content: [
        TextBlockSchema.parse({ text: "What's in this image?" }),
        ImageBlockSchema.parse({ url: "https://example.com/img.png" }),
      ],
    };
    expect(Array.isArray(m.content)).toBe(true);
    expect(m.content).toHaveLength(2);
    expect((m.content as Array<{ type: string }>)[0]?.type).toBe("text");
    expect((m.content as Array<{ type: string }>)[1]?.type).toBe("image");
  });

  it("mixed content round-trips through AnyContentBlockSchema", () => {
    const blocks = [
      TextBlockSchema.parse({ text: "Check this:" }),
      DocumentBlockSchema.parse({
        url: "https://example.com/doc.pdf",
        filename: "doc.pdf",
      }),
    ];
    const data = JSON.parse(JSON.stringify(blocks));
    const restored = (data as unknown[]).map((b) =>
      AnyContentBlockSchema.parse(b),
    );
    expect(restored).toHaveLength(2);
    expect(restored[0]?.type).toBe("text");
    if (restored[0]?.type === "text")
      expect(restored[0].text).toBe("Check this:");
    expect(restored[1]?.type).toBe("document");
    if (restored[1]?.type === "document")
      expect(restored[1].url).toBe("https://example.com/doc.pdf");
  });

  it("empty content list is valid", () => {
    const m: Message = { role: "user", content: [] };
    expect(m.content).toEqual([]);
  });

  it("single text block in list is valid", () => {
    const m: Message = {
      role: "user",
      content: [TextBlockSchema.parse({ text: "Just text" })],
    };
    expect(m.content).toHaveLength(1);
    expect((m.content as Array<{ text: string }>)[0]?.text).toBe("Just text");
  });
});

describe("RequestContext", () => {
  it("all fields default to undefined", () => {
    const ctx: RequestContext = {};
    expect(ctx.executionId).toBeUndefined();
    expect(ctx.sessionId).toBeUndefined();
    expect(ctx.userId).toBeUndefined();
    expect(ctx.requestHeaders).toBeUndefined();
  });

  it("accepts all fields", () => {
    const ctx: RequestContext = {
      executionId: "inv-abc",
      sessionId: "sess-1",
      userId: "user-1",
      requestHeaders: { "x-custom": "value" },
    };
    expect(ctx.executionId).toBe("inv-abc");
    expect(ctx.sessionId).toBe("sess-1");
    expect(ctx.userId).toBe("user-1");
    expect(ctx.requestHeaders).toEqual({ "x-custom": "value" });
  });
});

describe("LLMTokenUsage", () => {
  it("auto-populates total_tokens from input + output", () => {
    const u = new LLMTokenUsage({ input_tokens: 3, output_tokens: 7 });
    expect(u.totalTokens).toBe(10);
  });

  it("falls back to prompt + completion when input/output missing", () => {
    const u = new LLMTokenUsage({ prompt_tokens: 4, completion_tokens: 6 });
    expect(u.totalTokens).toBe(10);
  });

  it("respects explicit total_tokens over derivation", () => {
    const u = new LLMTokenUsage({
      input_tokens: 1,
      output_tokens: 1,
      total_tokens: 99,
    });
    expect(u.totalTokens).toBe(99);
  });

  it("toLangchainUsageMetadata zeros out missing fields", () => {
    const u = new LLMTokenUsage({ input_tokens: 5 });
    const m = u.toLangchainUsageMetadata();
    expect(m.input_tokens).toBe(5);
    expect(m.output_tokens).toBe(0);
    expect(m.total_tokens).toBe(5);
  });

  it("parse produces a class instance with toLangchainUsageMetadata", () => {
    const u = LLMTokenUsageSchema.parse({ input_tokens: 10, output_tokens: 4 });
    expect(u).toBeInstanceOf(LLMTokenUsage);
    const m = u.toLangchainUsageMetadata();
    expect(m.input_tokens).toBe(10);
    expect(m.output_tokens).toBe(4);
    expect(m.total_tokens).toBe(14);
  });
});

describe("LLMToolCall parse path", () => {
  it("parse produces a class instance with toLangchainDict", () => {
    const tc = LLMToolCallSchema.parse({
      id: "call_1",
      name: "search",
      args: { q: "test" },
      index: 0,
    });
    expect(tc).toBeInstanceOf(LLMToolCall);
    const dict = tc.toLangchainDict();
    expect(dict["id"]).toBe("call_1");
    expect(dict["name"]).toBe("search");
    expect(dict["args"]).toEqual({ q: "test" });
    expect("index" in dict).toBe(false);
  });
});

describe("LLMToolSchema", () => {
  it("parse produces a class instance with toLangchainDict", () => {
    const tool = LLMToolSchemaValidator.parse({
      name: "get_weather",
      description: "Get current weather",
      type: "function",
      parameters: { type: "object", properties: {} },
    });
    expect(tool).toBeInstanceOf(LLMToolSchema);
    const dict = tool.toLangchainDict();
    expect(dict["name"]).toBe("get_weather");
    expect(dict["type"]).toBe("function");
  });
});

describe("LLMInvocationOptions", () => {
  it("parse produces a class instance with toModelKwargs", () => {
    const opts = LLMInvocationOptionsSchema.parse({
      max_tokens: 512,
      temperature: 0.7,
    });
    expect(opts).toBeInstanceOf(LLMInvocationOptions);
    const kwargs = opts.toModelKwargs();
    expect(kwargs["max_tokens"]).toBe(512);
  });

  it("unknown provider kwargs (e.g. temperature) survive parse and toModelKwargs", () => {
    const opts = LLMInvocationOptionsSchema.parse({
      max_tokens: 1024,
      temperature: 0.7,
      top_p: 0.9,
      custom_provider_flag: true,
    });
    const kwargs = opts.toModelKwargs();
    expect(kwargs["max_tokens"]).toBe(1024);
    expect(kwargs["top_p"]).toBe(0.9);
    expect(kwargs["temperature"]).toBe(0.7);
    expect(kwargs["custom_provider_flag"]).toBe(true);
  });

  it("extras are stored on the instance", () => {
    const opts = LLMInvocationOptionsSchema.parse({
      temperature: 0.5,
      seed: 42,
    });
    expect(opts.extras["temperature"]).toBe(0.5);
    expect(opts.extras["seed"]).toBeUndefined(); // seed is a known field, not an extra
  });

  it("round-trips through JSON.stringify + parse", () => {
    const original = LLMInvocationOptionsSchema.parse({
      max_tokens: 256,
      temperature: 0.3,
    });
    const restored = LLMInvocationOptionsSchema.parse(
      JSON.parse(JSON.stringify(original)),
    );
    expect(restored.maxTokens).toBe(256);
    expect(restored.extras["temperature"]).toBe(0.3);
  });

  it("known fields always override extras on key collision", () => {
    // seed is known — should appear under maxTokens etc, not extras
    const opts = LLMInvocationOptionsSchema.parse({
      seed: 7,
      temperature: 0.1,
    });
    expect(opts.seed).toBe(7);
    expect(opts.extras["seed"]).toBeUndefined();
    const kwargs = opts.toModelKwargs();
    expect(kwargs["seed"]).toBe(7);
  });
});
