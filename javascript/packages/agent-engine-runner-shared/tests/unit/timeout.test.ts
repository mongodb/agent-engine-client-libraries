/**
 * Tests for the deadline helper used to bound the per-execution graph build.
 */

import { describe, expect, it } from "vitest";
import { TimeoutError, withTimeout } from "../../src/timeout.js";

describe("withTimeout", () => {
  it("resolves with the work's value when it settles in time", async () => {
    await expect(
      withTimeout(Promise.resolve(7), 1000, () => "too slow"),
    ).resolves.toBe(7);
  });

  it("propagates the work's own rejection", async () => {
    await expect(
      withTimeout(
        Promise.reject(new Error("builder failed")),
        1000,
        () => "too slow",
      ),
    ).rejects.toThrow("builder failed");
  });

  it("rejects with the message when the work outlives the budget", async () => {
    const never = new Promise<never>(() => {});
    await expect(
      withTimeout(never, 10, () => "the graph build did not complete"),
    ).rejects.toThrow("the graph build did not complete");
  });

  it("distinguishes the deadline passing from the work failing", async () => {
    const never = new Promise<never>(() => {});
    await expect(
      withTimeout(never, 10, () => "too slow"),
    ).rejects.toBeInstanceOf(TimeoutError);
    await expect(
      withTimeout(
        Promise.reject(new Error("builder failed")),
        1000,
        () => "late",
      ),
    ).rejects.not.toBeInstanceOf(TimeoutError);
  });
});
