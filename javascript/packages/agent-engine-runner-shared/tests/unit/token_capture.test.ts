/**
 * Tests for token usage extraction (extractUsage).
 *
 * Mirrors Python's tests/unit/test_token_capture.py 1:1.
 *
 * Ported from runner-shared — uses a plain object instead of AIMessage
 * since agent-engine-runner-shared has no langchain dependency.
 *
 * Source change to enable parity: `extractUsage` and `_extractPodUsage`
 * were ported from Python's `secure_wrapper.extract_usage` /
 * `_extract_pod_usage`. The original Phase 9 port skipped them because no
 * internal caller exists, but external consumers (downstream framework
 * SDKs) expect them. See AGENTS.md for the rationale.
 */

import { describe, test, expect } from "vitest";
import { extractUsage } from "../../src/index.js";

/**
 * Build a lightweight object that quacks like an LLM response.
 * Python's `_msg` uses SimpleNamespace; TS just uses a plain object since
 * `extractUsage` reads via property access (no getattr-style fallback needed).
 */
function msg(
  responseMetadata: Record<string, unknown> | null = null,
  extra: Record<string, unknown> = {},
): Record<string, unknown> {
  return { response_metadata: responseMetadata, ...extra };
}

describe("extractUsage", () => {
  test("OpenAI-style response_metadata with usage dict", () => {
    // test_openai_style_usage
    const usage = extractUsage(
      msg({
        usage: { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
        model_name: "gpt-4o",
      }),
      "fallback",
    );

    expect(usage.prompt_tokens).toBe(100);
    expect(usage.completion_tokens).toBe(50);
    expect(usage.total_tokens).toBe(150);
    expect(usage.model).toBe("gpt-4o");
  });

  test("Anthropic-style with input_tokens/output_tokens keys", () => {
    // test_anthropic_style_usage
    const usage = extractUsage(
      msg({
        usage: { input_tokens: 100, output_tokens: 50 },
      }),
      "claude-3-5-sonnet",
    );

    expect(usage.prompt_tokens).toBe(100);
    expect(usage.completion_tokens).toBe(50);
    expect(usage.total_tokens).toBe(150);
    expect(usage.model).toBe("claude-3-5-sonnet");
  });

  test("falls back to token_usage dict when usage is missing", () => {
    // test_token_usage_key_fallback
    const usage = extractUsage(
      msg({ token_usage: { prompt_tokens: 100, completion_tokens: 50 } }),
    );

    expect(usage.prompt_tokens).toBe(100);
    expect(usage.completion_tokens).toBe(50);
  });

  test("empty usage falls through to token_usage", () => {
    const usage = extractUsage(
      msg({
        usage: {},
        token_usage: { prompt_tokens: 100, completion_tokens: 50 },
      }),
    );

    expect(usage.prompt_tokens).toBe(100);
    expect(usage.completion_tokens).toBe(50);
  });

  test("malformed usage falls through to token_usage", () => {
    const usage = extractUsage(
      msg({
        usage: { foo: 1 },
        token_usage: { prompt_tokens: 100, completion_tokens: 50 },
      }),
    );

    expect(usage.prompt_tokens).toBe(100);
    expect(usage.completion_tokens).toBe(50);
  });

  test("response_metadata=null returns all nulls with fallback model", () => {
    // test_missing_response_metadata
    const usage = extractUsage(msg(null), "test-model");

    expect(usage.prompt_tokens).toBeNull();
    expect(usage.completion_tokens).toBeNull();
    expect(usage.total_tokens).toBeNull();
    expect(usage.model).toBe("test-model");
  });

  test("empty usage dict returns nulls", () => {
    // test_empty_usage_dict
    const usage = extractUsage(msg({ usage: {} }));

    expect(usage.prompt_tokens).toBeNull();
    expect(usage.completion_tokens).toBeNull();
    expect(usage.total_tokens).toBeNull();
  });

  test("non-object response handles gracefully", () => {
    // test_non_object_response
    const usage = extractUsage("plain string response", "test");

    expect(usage.prompt_tokens).toBeNull();
    expect(usage.completion_tokens).toBeNull();
    expect(usage.model).toBe("test");
  });

  test("model name comes from response_metadata when available", () => {
    // test_model_name_from_response_metadata
    const usage = extractUsage(
      msg({
        model_name: "gpt-4o-from-metadata",
        usage: { prompt_tokens: 10, completion_tokens: 5 },
      }),
      "fallback-model",
    );

    expect(usage.model).toBe("gpt-4o-from-metadata");
  });

  test("model name falls back to fallback model when not in metadata", () => {
    // test_model_name_fallback
    const usage = extractUsage(
      msg({ usage: { prompt_tokens: 10, completion_tokens: 5 } }),
      "claude-3",
    );

    expect(usage.model).toBe("claude-3");
  });

  test('"model" key is also recognised (not just "model_name")', () => {
    // test_model_key_in_response_metadata
    const usage = extractUsage(
      msg({
        model: "gpt-4o-via-model-key",
        usage: { prompt_tokens: 10, completion_tokens: 5 },
      }),
      "fallback",
    );

    expect(usage.model).toBe("gpt-4o-via-model-key");
  });

  test("total_tokens is computed when prompt+completion are present but total is missing", () => {
    // test_total_tokens_computed_when_missing
    const usage = extractUsage(
      msg({ usage: { prompt_tokens: 100, completion_tokens: 50 } }),
    );

    expect(usage.total_tokens).toBe(150);
  });

  test("falls back to usage_metadata when response_metadata has no usage", () => {
    // test_usage_metadata_fallback
    const usage = extractUsage(
      msg(
        {},
        {
          usage_metadata: {
            prompt_tokens: 200,
            completion_tokens: 80,
            total_tokens: 280,
          },
        },
      ),
      "qwen-3-32b",
    );

    expect(usage.prompt_tokens).toBe(200);
    expect(usage.completion_tokens).toBe(80);
    expect(usage.total_tokens).toBe(280);
    expect(usage.model).toBe("qwen-3-32b");
  });

  test("usage_metadata with input_tokens/output_tokens keys works", () => {
    // test_usage_metadata_with_input_output_keys
    const usage = extractUsage(
      msg({}, { usage_metadata: { input_tokens: 150, output_tokens: 60 } }),
    );

    expect(usage.prompt_tokens).toBe(150);
    expect(usage.completion_tokens).toBe(60);
    expect(usage.total_tokens).toBe(210);
  });

  test("response_metadata.usage takes precedence over usage_metadata", () => {
    // test_response_metadata_takes_precedence_over_usage_metadata
    const usage = extractUsage(
      msg(
        { usage: { prompt_tokens: 100, completion_tokens: 50 } },
        { usage_metadata: { prompt_tokens: 999, completion_tokens: 999 } },
      ),
    );

    expect(usage.prompt_tokens).toBe(100);
    expect(usage.completion_tokens).toBe(50);
  });

  test("Anthropic Usage-like object in response_metadata.usage", () => {
    const usageObj = { input_tokens: 1048, output_tokens: 1222 };
    const usage = extractUsage(msg({ usage: usageObj }), "claude-sonnet-4-6");
    expect(usage.prompt_tokens).toBe(1048);
    expect(usage.completion_tokens).toBe(1222);
    expect(usage.total_tokens).toBe(2270);
    expect(usage.model).toBe("claude-sonnet-4-6");
  });

  test("top-level usage attribute when metadata is empty", () => {
    const usage = extractUsage(
      msg({}, { usage: { input_tokens: 7, output_tokens: 3 } }),
    );
    expect(usage.prompt_tokens).toBe(7);
    expect(usage.completion_tokens).toBe(3);
    expect(usage.total_tokens).toBe(10);
  });

  test("malformed usage string returns nulls without throwing", () => {
    const usage = extractUsage(msg({ usage: "n/a" }), "test");
    expect(usage.prompt_tokens).toBeNull();
    expect(usage.completion_tokens).toBeNull();
    expect(usage.total_tokens).toBeNull();
    expect(usage.model).toBe("test");
  });
});
