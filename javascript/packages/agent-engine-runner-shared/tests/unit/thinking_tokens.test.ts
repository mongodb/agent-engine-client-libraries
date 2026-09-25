/**
 * Tests for thinking-token stripping and filtering utilities.
 *
 * Mirrors Python's tests/unit/test_thinking_tokens.py.
 *
 * Python returns a 3-tuple (streamable, buf, inside); TS returns the same
 * shape as a tuple [streamable, newBuffer, inside]. API parity is 1:1.
 */

import { describe, test, expect } from "vitest";
import { filterThinkingTokens, stripThinking } from "../../src/index.js";

describe("filterThinkingTokens", () => {
  test("plain text passes through", () => {
    // test_plain_text_passes_through
    const [streamable, buf, inside] = filterThinkingTokens("hello", "", false);
    expect(streamable).toBe("hello");
    expect(buf).toBe("");
    expect(inside).toBe(false);
  });

  test("full <think> block is suppressed", () => {
    // test_full_think_block_suppressed
    const [streamable, buf, inside] = filterThinkingTokens(
      "<think>reasoning</think>visible",
      "",
      false,
    );
    expect(streamable).toBe("visible");
    expect(buf).toBe("");
    expect(inside).toBe(false);
  });

  test("open <think> tag buffers, marks inside=true", () => {
    // test_open_think_tag_buffers
    const [streamable, , inside] = filterThinkingTokens(
      "<think>start",
      "",
      false,
    );
    expect(streamable).toBe("");
    expect(inside).toBe(true);
  });

  test("close </think> tag resumes streaming", () => {
    // test_close_think_tag_resumes
    const [streamable, , inside] = filterThinkingTokens(
      "more</think>after",
      "buffered",
      true,
    );
    expect(streamable).toBe("after");
    expect(inside).toBe(false);
  });

  test("multi-chunk <think> block concatenates around it", () => {
    // test_multi_chunk_think_block
    let buf = "";
    let inside = false;
    const streamed: string[] = [];

    for (const token of ["He", "<think>", "reason", "</think>", "llo"]) {
      let s: string;
      [s, buf, inside] = filterThinkingTokens(token, buf, inside);
      if (s) streamed.push(s);
    }

    expect(streamed.join("")).toBe("Hello");
    expect(inside).toBe(false);
  });

  // Helper: feed a chunk sequence through the filter, return concatenated
  // streamable output and final `inside` state.
  function runChunks(chunks: string[]): { streamed: string; inside: boolean } {
    let buf = "";
    let inside = false;
    const parts: string[] = [];
    for (const token of chunks) {
      let s: string;
      [s, buf, inside] = filterThinkingTokens(token, buf, inside);
      if (s) parts.push(s);
    }
    return { streamed: parts.join(""), inside };
  }

  test("opening <think> split across chunks does not leak hidden content", () => {
    // "<think>" arrives as "<thi" + "nk>": the partial opener must be held
    // back, not flushed as plain text.
    const { streamed, inside } = runChunks(["He<thi", "nk>secret</think>llo"]);
    expect(streamed).toBe("Hello");
    expect(inside).toBe(false);
  });

  test("closing </think> split across chunks keeps content suppressed", () => {
    const { streamed, inside } = runChunks([
      "<think>a",
      "b</thi",
      "nk>visible",
    ]);
    expect(streamed).toBe("visible");
    expect(inside).toBe(false);
  });

  test("opener split into single characters does not leak", () => {
    const { streamed, inside } = runChunks([
      "<",
      "t",
      "h",
      "i",
      "n",
      "k",
      ">",
      "hidden</think>shown",
    ]);
    expect(streamed).toBe("shown");
    expect(inside).toBe(false);
  });

  test("partial-opener lookalike is not held forever", () => {
    // "<thinking>" is not a `<think>` opener — it must stream as plain text
    // rather than be held back waiting for a `>` that never completes it.
    const { streamed, inside } = runChunks(["<thi", "nking> kept"]);
    expect(streamed).toBe("<thinking> kept");
    expect(inside).toBe(false);
  });
});

describe("stripThinking", () => {
  test("removes <think> block", () => {
    // test_removes_think_block
    expect(stripThinking("<think>foo</think>bar")).toBe("bar");
  });

  test("removes unclosed <think> through end of string", () => {
    // test_removes_unclosed_think
    expect(stripThinking("before<think>trailing")).toBe("before");
  });

  test("empty string returns empty string", () => {
    // test_empty_string
    expect(stripThinking("")).toBe("");
  });

  test("plain text without tags is unchanged", () => {
    // test_no_think_tags
    expect(stripThinking("plain text")).toBe("plain text");
  });

  test("multiline <think> block is removed", () => {
    // test_multiline_think_block
    const text = "<think>\nline1\nline2\n</think>result";
    expect(stripThinking(text)).toBe("result");
  });
});
