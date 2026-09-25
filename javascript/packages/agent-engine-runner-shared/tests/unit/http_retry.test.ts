/**
 * Unit tests for the shared Retry-After helpers.
 *
 * Mirrors the delta-seconds parsing coverage in Python's
 * tests/unit/test_http_retry.py. Only the delta-seconds form is honored; the
 * HTTP-date form and garbage fall through to `null` so callers use their normal
 * backoff. The honored wait is capped at RETRY_AFTER_MAX_WAIT_MS.
 */

import { describe, test, expect, vi } from "vitest";
import {
  discardResponseBody,
  RETRY_AFTER_MAX_WAIT_MS,
  parseRetryAfterSeconds,
  retryAfterWaitMs,
} from "../../src/server/http_retry.js";

describe("discardResponseBody", () => {
  test("cancels an unread response body", async () => {
    const cancel = vi.fn(async () => {});

    await discardResponseBody({ body: { cancel } } as unknown as Response);

    expect(cancel).toHaveBeenCalledTimes(1);
  });

  test("does not replace the callback result with a cleanup error", async () => {
    const cancel = vi.fn(async () => {
      throw new Error("already closed");
    });

    await expect(
      discardResponseBody({ body: { cancel } } as unknown as Response),
    ).resolves.toBeUndefined();
  });
});

describe("parseRetryAfterSeconds", () => {
  test.each([
    ["0", 0],
    ["1", 1],
    ["120", 120],
    ["  5  ", 5],
  ])("parses delta-seconds %p", (header, expected) => {
    expect(parseRetryAfterSeconds(header)).toBe(expected);
  });

  test.each([
    ["absent", null],
    ["undefined", undefined],
    ["empty", ""],
    ["whitespace", "   "],
    ["http-date", "Wed, 21 Oct 2025 07:28:00 GMT"],
    ["negative", "-5"],
    ["decimal", "1.5"],
    ["garbage", "soon"],
    ["signed", "+5"],
  ])("returns null for non-delta-seconds (%s)", (_desc, header) => {
    const value = header === undefined ? undefined : (header as string);
    expect(parseRetryAfterSeconds(value)).toBeNull();
  });
});

describe("retryAfterWaitMs", () => {
  test("converts honored seconds to milliseconds", () => {
    expect(retryAfterWaitMs("3")).toBe(3000);
  });

  test("caps the honored wait at RETRY_AFTER_MAX_WAIT_MS", () => {
    expect(retryAfterWaitMs("100")).toBe(RETRY_AFTER_MAX_WAIT_MS);
    expect(RETRY_AFTER_MAX_WAIT_MS).toBe(10_000);
  });

  test("returns null when the header is not honorable", () => {
    expect(retryAfterWaitMs("Wed, 21 Oct 2025 07:28:00 GMT")).toBeNull();
    expect(retryAfterWaitMs(null)).toBeNull();
  });
});
