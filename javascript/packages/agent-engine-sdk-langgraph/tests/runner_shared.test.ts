import { describe, expect, it } from "vitest";

import {
  LLMResult,
  normalizeContent,
  normalizeToolCallArgs,
} from "@mongodb-js/agent-engine-runner-shared";
import { LLMTokenUsage } from "@mongodb-js/agent-engine-sdk";

describe("normalizeContent", () => {
  it("passes strings through unchanged", () => {
    expect(normalizeContent("hello")).toBe("hello");
  });

  it("returns empty string for null/undefined", () => {
    expect(normalizeContent(null)).toBe("");
    expect(normalizeContent(undefined)).toBe("");
  });

  it("concatenates text blocks from a multimodal list", () => {
    expect(
      normalizeContent([
        { type: "text", text: "a " },
        { type: "text", text: "b" },
      ]),
    ).toBe("a b");
  });

  it("skips non-text blocks (image, etc.)", () => {
    expect(
      normalizeContent([
        { type: "text", text: "caption" },
        { type: "image", url: "…" },
      ]),
    ).toBe("caption");
  });

  it("stringifies non-string non-array fallback values", () => {
    expect(normalizeContent(42)).toBe("42");
  });
});

describe("normalizeToolCallArgs", () => {
  it("returns null for null/undefined", () => {
    expect(normalizeToolCallArgs(null)).toBeNull();
    expect(normalizeToolCallArgs(undefined)).toBeNull();
  });

  it("returns strings unchanged", () => {
    expect(normalizeToolCallArgs('{"a":1}')).toBe('{"a":1}');
  });

  it("JSON-serializes plain objects", () => {
    expect(normalizeToolCallArgs({ a: 1, b: "x" })).toBe('{"a":1,"b":"x"}');
  });
});

describe("LLMResult", () => {
  it("produces an LLMResponse from a LangChain-shaped object", () => {
    const lcResponse = {
      content: "hello",
      tool_calls: [],
      response_metadata: { model: "gpt-4" },
    };
    const result = LLMResult.fromResponse(lcResponse).toResponse();
    expect(result.content).toBe("hello");
    expect(result.responseMetadata?.["model"]).toBe("gpt-4");
  });

  it("extracts usage from usage_metadata", () => {
    const usage = LLMResult.extractUsage({
      usage_metadata: { input_tokens: 5, output_tokens: 7 },
    });
    expect(usage).toBeInstanceOf(LLMTokenUsage);
    expect(usage?.inputTokens).toBe(5);
    expect(usage?.outputTokens).toBe(7);
  });

  it("extracts usage from response_metadata.usage", () => {
    const usage = LLMResult.extractUsage({
      response_metadata: { usage: { input_tokens: 1, output_tokens: 2 } },
    });
    expect(usage?.inputTokens).toBe(1);
  });

  it("returns undefined when no usage shape is present", () => {
    expect(LLMResult.extractUsage({})).toBeUndefined();
  });
});
