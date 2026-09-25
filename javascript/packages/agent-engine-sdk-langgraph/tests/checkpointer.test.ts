/**
 * Covers withEmptyBatchGuard — the guard that stops the MongoDB checkpointer
 * from calling `bulkWrite([])` (which MongoDB rejects) when a graph step has no
 * pending writes. Driven with a fake saver so no MongoDB is required.
 */

import { describe, expect, it, vi } from "vitest";
import type { BaseCheckpointSaver } from "@langchain/langgraph-checkpoint";

import { withEmptyBatchGuard } from "../src/checkpointer.js";

/** Minimal fake saver exposing a spied putWrites; other methods are unused here. */
function fakeSaver(): BaseCheckpointSaver & {
  putWrites: ReturnType<typeof vi.fn>;
} {
  return {
    putWrites: vi.fn(async () => undefined),
  } as unknown as BaseCheckpointSaver & { putWrites: ReturnType<typeof vi.fn> };
}

const config = { configurable: { thread_id: "t1" } };

describe("withEmptyBatchGuard", () => {
  it("skips the underlying putWrites when there are no writes", async () => {
    const saver = fakeSaver();
    const original = saver.putWrites;
    const guarded = withEmptyBatchGuard(saver);

    await expect(
      guarded.putWrites(config, [], "task-1"),
    ).resolves.toBeUndefined();
    expect(original).not.toHaveBeenCalled();
  });

  it("forwards non-empty writes unchanged", async () => {
    const saver = fakeSaver();
    const original = saver.putWrites;
    const guarded = withEmptyBatchGuard(saver);
    const writes: [string, unknown][] = [["channel", { value: 1 }]];

    await guarded.putWrites(config, writes, "task-2");

    expect(original).toHaveBeenCalledTimes(1);
    expect(original).toHaveBeenCalledWith(config, writes, "task-2");
  });

  it("forwards a non-array writes value instead of swallowing it", async () => {
    const saver = fakeSaver();
    const original = saver.putWrites;
    const guarded = withEmptyBatchGuard(saver);

    await guarded.putWrites(config, undefined as never, "task-3");

    expect(original).toHaveBeenCalledWith(config, undefined, "task-3");
  });

  it("returns the same saver instance (patched in place)", () => {
    const saver = fakeSaver();
    expect(withEmptyBatchGuard(saver)).toBe(saver);
  });
});
