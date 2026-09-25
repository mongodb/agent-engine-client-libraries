import { describe, expect, it } from "vitest";
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { Overwrite } from "@langchain/langgraph";

import {
  FRAMEWORK_LANGCHAIN,
  assignMissingGraphInputMessageIds,
  dictToAiMessage,
  dictToAiMessageChunk,
  lcMessagesToPlatform,
  lcToPlatformMessage,
  messageToDict,
  platformMessagesToLc,
  platformToLcMessage,
  toToolMessageContent,
} from "../src/messages.js";
import type { Message } from "@mongodb-js/agent-engine-sdk";

describe("assignMissingGraphInputMessageIds", () => {
  const missingId = (index: number) => `message-${index}`;

  it("copies messages that need ids without mutating hook-owned input", () => {
    const missing = new HumanMessage("new");
    const existing = new HumanMessage({ content: "existing", id: "keep-me" });

    const result = assignMissingGraphInputMessageIds(
      { messages: [missing, existing] },
      missingId,
    ) as { messages: HumanMessage[] };

    expect(result.messages[0]).not.toBe(missing);
    expect(result.messages[0]?.id).toBe("message-0");
    expect(result.messages[1]).toBe(existing);
    expect(result.messages[1]?.id).toBe("keep-me");
    expect(missing.id).toBeUndefined();
  });

  it("preserves a real LangGraph Overwrite wrapper", () => {
    const original = new Overwrite([["human", "hello"]]);
    const result = assignMissingGraphInputMessageIds(
      { messages: original },
      missingId,
    ) as {
      messages: Overwrite<HumanMessage[]>;
    };

    expect(result.messages).toBeInstanceOf(Overwrite);
    expect(result.messages.value[0]).toBeInstanceOf(HumanMessage);
    expect(result.messages.value[0]?.id).toBe("message-0");
    expect(original.value[0]).toEqual(["human", "hello"]);
  });

  it("preserves a serialized Overwrite wrapper", () => {
    const result = assignMissingGraphInputMessageIds(
      {
        messages: {
          __overwrite__: [["human", "hello"]],
          guard: "keep",
        },
      },
      missingId,
    ) as {
      messages: { __overwrite__: HumanMessage[]; guard: string };
    };

    expect(result.messages.guard).toBe("keep");
    expect(result.messages.__overwrite__[0]).toBeInstanceOf(HumanMessage);
    expect(result.messages.__overwrite__[0]?.id).toBe("message-0");
  });

  it("can reuse the same hook-owned message for different durable ids", () => {
    const original = new HumanMessage("hello");
    const first = assignMissingGraphInputMessageIds(
      { messages: [original] },
      (index) => `first-${index}`,
    ) as { messages: HumanMessage[] };
    const second = assignMissingGraphInputMessageIds(
      { messages: [original] },
      (index) => `second-${index}`,
    ) as { messages: HumanMessage[] };

    expect(original.id).toBeUndefined();
    expect(first.messages[0]?.id).toBe("first-0");
    expect(second.messages[0]?.id).toBe("second-0");
  });

  it("leaves null messages and caller-provided empty ids unchanged", () => {
    const withoutMessages = { messages: null, extra: "keep" };
    const emptyId = new HumanMessage({ content: "hello", id: "" });

    expect(assignMissingGraphInputMessageIds(withoutMessages, missingId)).toBe(
      withoutMessages,
    );
    const result = assignMissingGraphInputMessageIds(
      { messages: [emptyId] },
      missingId,
    ) as { messages: HumanMessage[] };
    expect(result.messages[0]).toBe(emptyId);
    expect(result.messages[0]?.id).toBe("");
  });
});

describe("toToolMessageContent", () => {
  it("preserves supported content blocks and serializes other tool results", () => {
    const blocks = [{ type: "text", text: "done" }];

    expect(toToolMessageContent("done")).toBe("done");
    expect(toToolMessageContent(blocks)).toBe(blocks);
    expect(toToolMessageContent({ ok: true })).toBe('{"ok":true}');
    expect(toToolMessageContent([{ type: "unknown", value: 1 }])).toBe(
      '[{"type":"unknown","value":1}]',
    );
  });

  it("falls back to string conversion when JSON serialization fails", () => {
    const circular: Record<string, unknown> = {};
    circular["self"] = circular;

    expect(toToolMessageContent(circular)).toBe("[object Object]");
  });
});

describe("lcToPlatformMessage", () => {
  it("maps a HumanMessage to a user-role platform Message", () => {
    const lc = new HumanMessage({ content: "hi" });
    const out = lcToPlatformMessage(lc);
    expect(out.role).toBe("user");
    expect(out.content).toBe("hi");
  });

  it("maps an AIMessage with tool_calls and preserves them", () => {
    const lc = new AIMessage({
      content: "calling tool",
      tool_calls: [
        { id: "call_1", name: "lookup", args: { x: 1 }, type: "tool_call" },
      ],
    });
    const out = lcToPlatformMessage(lc);
    expect(out.role).toBe("assistant");
    expect(out.toolCalls?.length).toBe(1);
    expect(out.toolCalls?.[0]?.id).toBe("call_1");
    expect(out.toolCalls?.[0]?.name).toBe("lookup");
  });

  it("omits redundant provider tool-call fragments from AI messages", () => {
    const out = lcToPlatformMessage(
      new AIMessage({
        content: "",
        tool_calls: [
          { id: "call_1", name: "lookup", args: { x: 1 }, type: "tool_call" },
        ],
        additional_kwargs: {
          refusal: null,
          tool_calls: [{ function: { arguments: "}" }, index: 0 }],
        },
      } as never),
    );

    expect(out.additionalKwargs).toEqual({ refusal: null });
  });

  it("maps a SystemMessage to a system-role platform Message", () => {
    const out = lcToPlatformMessage(
      new SystemMessage({ content: "be helpful" }),
    );
    expect(out.role).toBe("system");
    expect(out.content).toBe("be helpful");
  });

  it("maps a ToolMessage and carries tool_call_id", () => {
    const out = lcToPlatformMessage(
      new ToolMessage({ content: "result", tool_call_id: "call_1" }),
    );
    expect(out.role).toBe("tool");
    expect(out.toolCallId).toBe("call_1");
    expect(out.isError).toBe(false);
  });

  it("maps a failed ToolMessage status", () => {
    const out = lcToPlatformMessage(
      new ToolMessage({
        content: "failed",
        tool_call_id: "call_1",
        status: "error",
      }),
    );

    expect(out.isError).toBe(true);
  });

  it("flattens a single text-block multimodal payload back to a plain string", () => {
    const lc = new HumanMessage({ content: [{ type: "text", text: "hello" }] });
    const out = lcToPlatformMessage(lc);
    expect(out.content).toBe("hello");
  });
});

describe("platformToLcMessage", () => {
  it("returns null for a tool message missing tool_call_id", () => {
    const msg: Message = { role: "tool", content: "orphan" };
    expect(platformToLcMessage(msg)).toBeNull();
  });

  it("roundtrips Human → platform → LangChain back to HumanMessage", () => {
    const original = new HumanMessage({ content: "ping" });
    const platform = lcToPlatformMessage(original);
    const back = platformToLcMessage(platform);
    expect(back).toBeInstanceOf(HumanMessage);
    expect((back as HumanMessage).content).toBe("ping");
  });
});

describe("bulk conversion", () => {
  it("filters orphan ToolMessages out of platformMessagesToLc", () => {
    const messages: Message[] = [
      { role: "user", content: "hi" },
      { role: "tool", content: "orphan" }, // no tool_call_id → dropped
    ];
    const lc = platformMessagesToLc(messages);
    expect(lc.length).toBe(1);
    expect(lc[0]).toBeInstanceOf(HumanMessage);
  });

  it("preserves order across roundtrip", () => {
    const lcMessages = [
      new HumanMessage({ content: "a" }),
      new HumanMessage({ content: "b" }),
    ];
    const platform = lcMessagesToPlatform(lcMessages);
    expect(platform.map((m) => m.content)).toEqual(["a", "b"]);
  });
});

describe("OE dict serialization", () => {
  it("stamps the framework key onto messageToDict output", () => {
    const dict = messageToDict(new HumanMessage({ content: "hello" }));
    expect(dict["framework"]).toBe(FRAMEWORK_LANGCHAIN);
  });

  it("returns the same AIMessage instance when dictToAiMessage receives one", () => {
    const ai = new AIMessage({ content: "cached" });
    expect(dictToAiMessage(ai)).toBe(ai);
  });

  it("throws on an unsupported framework tag", () => {
    expect(() =>
      dictToAiMessage({ framework: "langflow", content: "x" }),
    ).toThrow(/Cannot deserialize message from framework/);
  });

  it("accepts a missing framework tag (treats as native)", () => {
    const ai = dictToAiMessage({ content: "no tag" });
    expect(ai).toBeInstanceOf(AIMessage);
  });
});

// ---------------------------------------------------------------------------
// messageToDict — flat shape (Python message_to_dict / model_dump parity)
// ---------------------------------------------------------------------------

describe("messageToDict — flat field preservation", () => {
  it("emits all top-level fields flat (not the nested toDict envelope)", () => {
    // Python: TestMessageToDict.test_preserves_all_fields.
    const msg = new AIMessage({
      content: "hello",
      id: "run-abc-123",
      name: "assistant",
      response_metadata: { model: "gpt-4" },
      usage_metadata: {
        input_tokens: 10,
        output_tokens: 5,
        total_tokens: 15,
      } as never,
      additional_kwargs: { refusal: null },
    });
    const result = messageToDict(msg);

    expect(result["id"]).toBe("run-abc-123");
    expect(result["name"]).toBe("assistant");
    expect(result["content"]).toBe("hello");
    expect(result["response_metadata"]).toEqual({ model: "gpt-4" });
    expect(result["usage_metadata"]).toEqual(msg.usage_metadata);
    expect(result["additional_kwargs"]).toEqual({ refusal: null });
    // Must be flat — no nested StoredMessage envelope.
    expect(result["data"]).toBeUndefined();
  });

  it("includes the framework field", () => {
    // Python: TestMessageToDict.test_includes_framework_field.
    const result = messageToDict(new AIMessage({ content: "hello" }));
    expect(result["framework"]).toBe(FRAMEWORK_LANGCHAIN);
  });
});

// ---------------------------------------------------------------------------
// Round-trip — messageToDict ↔ dictToAiMessage (Python TestRoundTrip parity)
// ---------------------------------------------------------------------------

describe("round-trip — messageToDict ↔ dictToAiMessage", () => {
  it("AIMessage survives messageToDict → dictToAiMessage", () => {
    // Python: TestRoundTrip.test_round_trip_ai_message.
    const original = new AIMessage({
      content: "hello world",
      id: "run-abc-123",
      name: "assistant",
      response_metadata: { model: "gpt-4", finish_reason: "stop" },
      tool_calls: [{ name: "foo", args: {}, id: "tc1", type: "tool_call" }],
      usage_metadata: {
        input_tokens: 10,
        output_tokens: 5,
        total_tokens: 15,
      } as never,
      additional_kwargs: { refusal: null },
    });
    const restored = dictToAiMessage(messageToDict(original));

    expect(restored.content).toBe(original.content);
    expect(restored.id).toBe(original.id);
    expect(restored.name).toBe(original.name);
    expect(restored.response_metadata).toEqual(original.response_metadata);
    expect(restored.tool_calls).toEqual(original.tool_calls);
    expect(restored.usage_metadata).toEqual(original.usage_metadata);
    expect(restored.additional_kwargs).toEqual(original.additional_kwargs);
  });

  it("AIMessage survives messageToDict → dictToAiMessageChunk", () => {
    // Python: TestRoundTrip.test_round_trip_ai_message_chunk.
    const original = new AIMessage({
      content: "hello",
      id: "run-abc-123",
      response_metadata: { finish_reason: "stop" },
      usage_metadata: {
        input_tokens: 10,
        output_tokens: 5,
        total_tokens: 15,
      } as never,
    });
    const restored = dictToAiMessageChunk(messageToDict(original));

    expect(restored).toBeInstanceOf(AIMessageChunk);
    expect(restored.content).toBe(original.content);
    expect(restored.id).toBe(original.id);
    expect(restored.response_metadata).toEqual(original.response_metadata);
    expect(restored.usage_metadata).toEqual(original.usage_metadata);
  });
});

// ---------------------------------------------------------------------------
// Round-trip tests — ports of test_messages.py round-trip suite
// ---------------------------------------------------------------------------

describe("round-trip — platform ↔ LangChain", () => {
  it("AIMessage with tool_calls survives a round-trip", () => {
    // Python: test_ai_message_with_tool_calls_round_trip.
    const original = new AIMessage({
      content: "calling",
      tool_calls: [
        { id: "tc1", name: "search", args: { q: "x" }, type: "tool_call" },
      ],
    });
    const platform = lcToPlatformMessage(original);
    const back = platformToLcMessage(platform);
    expect(back).toBeInstanceOf(AIMessage);
    expect((back as AIMessage).tool_calls?.[0]?.name).toBe("search");
  });

  it("ToolMessage with name survives a round-trip", () => {
    // Python: test_tool_message_round_trip / test_tool_message_name_round_trip.
    const original = new ToolMessage({
      content: "done",
      tool_call_id: "tc1",
      name: "search",
    });
    const platform = lcToPlatformMessage(original);
    const back = platformToLcMessage(platform);
    expect(back).toBeInstanceOf(ToolMessage);
    expect((back as ToolMessage).tool_call_id).toBe("tc1");
    expect((back as ToolMessage).name).toBe("search");
  });

  it("ToolMessage error status survives a round-trip", () => {
    const original = new ToolMessage({
      content: "failed",
      tool_call_id: "tc1",
      status: "error",
    });
    const platform = lcToPlatformMessage(original);
    const back = platformToLcMessage(platform);

    expect(back).toBeInstanceOf(ToolMessage);
    expect((back as ToolMessage).status).toBe("error");
  });

  it("SystemMessage survives a round-trip", () => {
    // Python: test_system_message round-trip (covered indirectly there).
    const original = new SystemMessage({ content: "system text" });
    const platform = lcToPlatformMessage(original);
    const back = platformToLcMessage(platform);
    expect(back).toBeInstanceOf(SystemMessage);
    expect((back as SystemMessage).content).toBe("system text");
  });

  it("batch round-trip preserves order and types", () => {
    // Python: test_batch_round_trip.
    const original = [
      new SystemMessage({ content: "sys" }),
      new HumanMessage({ content: "u1" }),
      new AIMessage({ content: "a1" }),
      new HumanMessage({ content: "u2" }),
    ];
    const platform = lcMessagesToPlatform(original);
    const back = platformMessagesToLc(platform);
    expect(back.map((m) => m.constructor.name)).toEqual(
      original.map((m) => m.constructor.name),
    );
    expect(back.map((m) => m.content)).toEqual(["sys", "u1", "a1", "u2"]);
  });
});

// ---------------------------------------------------------------------------
// Multimodal — ports of test_messages.py multimodal suite
// ---------------------------------------------------------------------------

describe("multimodal content", () => {
  it("preserves multiple text blocks as a structured payload (TS leaves them intact)", () => {
    // Python flattens to a concatenated string; the TS port keeps the
    // structured content so downstream renderers can format each block
    // independently. Verifies the array shape is preserved.
    const lc = new HumanMessage({
      content: [
        { type: "text", text: "Hello " },
        { type: "text", text: "world" },
      ],
    });
    const out = lcToPlatformMessage(lc);
    expect(Array.isArray(out.content)).toBe(true);
    expect(out.content as Array<{ text: string }>).toHaveLength(2);
  });

  it("collapses a single text block to a plain string", () => {
    // Python: test_multimodal_single_text_block_simplifies_to_string.
    const lc = new HumanMessage({ content: [{ type: "text", text: "only" }] });
    const out = lcToPlatformMessage(lc);
    expect(out.content).toBe("only");
    expect(typeof out.content).toBe("string");
  });

  it("passes mixed content blocks (text + image) through as-is", () => {
    // Python's behavior is to skip non-text blocks during string concatenation;
    // the TS port preserves the original block list since it does not flatten
    // multi-block content. Verifies the array stays intact.
    const lc = new HumanMessage({
      content: [
        { type: "text", text: "hello" },
        { type: "image" },
        { type: "text", text: " world" },
      ],
    });
    const out = lcToPlatformMessage(lc);
    expect(Array.isArray(out.content)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Tool-call edge cases — ports of test_messages.py serialize_tool_calls tests
// ---------------------------------------------------------------------------

describe("AI tool-call edge cases", () => {
  it("drops tool messages with empty tool_call_id from bulk conversion", () => {
    // Python: test_tool_message_empty_tool_call_id_skipped.
    const messages: Message[] = [
      { role: "user", content: "hi" },
      { role: "tool", content: "x", toolCallId: "" },
    ];
    const lc = platformMessagesToLc(messages);
    expect(lc).toHaveLength(1);
    expect(lc[0]).toBeInstanceOf(HumanMessage);
  });

  it("preserves AIMessage with empty content but non-empty tool_calls", () => {
    // Python: test_ai_message round-trip with empty content.
    const original = new AIMessage({
      content: "",
      tool_calls: [{ id: "tc1", name: "search", args: {}, type: "tool_call" }],
    });
    const platform = lcToPlatformMessage(original);
    expect(platform.toolCalls?.length).toBe(1);
  });
});
