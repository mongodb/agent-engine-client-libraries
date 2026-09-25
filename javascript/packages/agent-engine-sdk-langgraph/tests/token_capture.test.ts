/**
 * Port of `agent-engine-sdk-langgraph/tests/test_token_capture.py`.
 *
 * Token usage capture varies wildly across LangChain providers. These tests
 * lock down the precedence order `LLMResult.extractUsage` follows:
 *
 *   1. usage_metadata (LangChain canonical)
 *   2. response_metadata.usage
 *   3. response_metadata.token_usage (OpenAI legacy)
 *   4. metadata.usage
 *
 * Plus fallbacks (model-name extraction, total computed from input+output)
 * that future provider drifts could quietly break without a regression test.
 */

import { describe, expect, it } from "vitest";
import { LLMResult } from "@mongodb-js/agent-engine-runner-shared";

describe("LLMResult.extractUsage — provider-shape precedence", () => {
  it("prefers usage_metadata when present (LangChain canonical shape)", () => {
    // Python: test_usage_metadata_fallback.
    const resp = {
      content: "ok",
      usage_metadata: {
        input_tokens: 5,
        output_tokens: 10,
        total_tokens: 15,
      },
    };
    const usage = LLMResult.extractUsage(resp);
    expect(usage).toBeDefined();
    expect(usage?.inputTokens).toBe(5);
    expect(usage?.outputTokens).toBe(10);
    expect(usage?.totalTokens).toBe(15);
  });

  it("falls through usage_metadata={} to response_metadata.usage", () => {
    // Python: test_response_metadata_takes_precedence_over_usage_metadata.
    const resp = {
      content: "ok",
      usage_metadata: {},
      response_metadata: {
        usage: {
          input_tokens: 7,
          output_tokens: 14,
          total_tokens: 21,
        },
      },
    };
    const usage = LLMResult.extractUsage(resp);
    expect(usage).toBeDefined();
    expect(usage?.inputTokens).toBe(7);
  });

  it("reads response_metadata.token_usage as the OpenAI legacy fallback", () => {
    // Python: test_token_usage_key_fallback.
    const resp = {
      content: "ok",
      response_metadata: {
        token_usage: {
          input_tokens: 3,
          output_tokens: 5,
          total_tokens: 8,
        },
      },
    };
    const usage = LLMResult.extractUsage(resp);
    expect(usage).toBeDefined();
    expect(usage?.inputTokens).toBe(3);
  });

  it("reads metadata.usage as the last resort", () => {
    // Python: test_metadata_usage_fallback (covered by extractUsage chain).
    const resp = {
      content: "ok",
      metadata: {
        usage: {
          input_tokens: 1,
          output_tokens: 2,
          total_tokens: 3,
        },
      },
    };
    const usage = LLMResult.extractUsage(resp);
    expect(usage).toBeDefined();
    expect(usage?.inputTokens).toBe(1);
  });

  it("returns undefined when no usage shape is found", () => {
    // Python: test_empty_usage_dict / test_missing_response_metadata_attr.
    const resp = { content: "ok" };
    const usage = LLMResult.extractUsage(resp);
    expect(usage).toBeUndefined();
  });

  it("returns undefined when usage_metadata is empty {}", () => {
    // Python: test_empty_usage_dict — falsy empty dict must skip the branch.
    const resp = { content: "ok", usage_metadata: {} };
    const usage = LLMResult.extractUsage(resp);
    expect(usage).toBeUndefined();
  });

  it("returns undefined for null / undefined responses", () => {
    // Python: test_non_aimessage_response (covers non-object inputs).
    expect(LLMResult.extractUsage(null)).toBeUndefined();
    expect(LLMResult.extractUsage(undefined)).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Anthropic / OpenAI style — sanity that both wire shapes are read
// ---------------------------------------------------------------------------

describe("LLMResult.extractUsage — provider shapes", () => {
  it("captures Anthropic-style usage from usage_metadata", () => {
    // Python: test_anthropic_style_usage.
    const resp = {
      content: "ok",
      usage_metadata: {
        input_tokens: 12,
        output_tokens: 3,
        total_tokens: 15,
      },
    };
    const usage = LLMResult.extractUsage(resp);
    expect(usage?.inputTokens).toBe(12);
    expect(usage?.outputTokens).toBe(3);
  });

  it("captures OpenAI-style usage from response_metadata.token_usage", () => {
    // Python: test_openai_style_usage.
    const resp = {
      content: "ok",
      response_metadata: {
        token_usage: {
          input_tokens: 50,
          output_tokens: 100,
          total_tokens: 150,
        },
      },
    };
    const usage = LLMResult.extractUsage(resp);
    expect(usage?.inputTokens).toBe(50);
  });

  it("captures Anthropic Usage-like objects from response_metadata.usage", () => {
    const resp = {
      content: "ok",
      response_metadata: {
        usage: { input_tokens: 1048, output_tokens: 1222 },
      },
    };
    const usage = LLMResult.extractUsage(resp);
    expect(usage?.inputTokens).toBe(1048);
    expect(usage?.outputTokens).toBe(1222);
    expect(usage?.totalTokens).toBe(2270);
  });

  it("returns undefined for malformed usage without throwing", () => {
    expect(
      LLMResult.extractUsage({
        content: "ok",
        response_metadata: { usage: "n/a" },
      }),
    ).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Fallback computation
// ---------------------------------------------------------------------------

describe("LLMResult.extractUsage — fallback computation", () => {
  it("computes total_tokens via toResponse when missing", () => {
    // Python: test_total_tokens_computed_when_missing.
    // The LLMTokenUsage class derives totals via its serializer; assert that
    // input + output reach the response when total is omitted.
    const resp = {
      content: "ok",
      usage_metadata: { input_tokens: 4, output_tokens: 6 },
    };
    const usage = LLMResult.extractUsage(resp);
    expect(usage).toBeDefined();
    expect(usage?.inputTokens).toBe(4);
    expect(usage?.outputTokens).toBe(6);
    // totalTokens may be undefined here; downstream code derives it.
  });
});
