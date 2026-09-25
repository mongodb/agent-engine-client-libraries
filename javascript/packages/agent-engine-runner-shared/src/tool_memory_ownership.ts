/** Request-local ownership for Memory reads performed by registered Tools. */

import { AsyncLocalStorage } from "node:async_hooks";

interface ToolMemoryReadOwnership {
  active: boolean;
}

const storage = new AsyncLocalStorage<ToolMemoryReadOwnership>();

function isPromiseLike(value: unknown): value is PromiseLike<unknown> {
  if (
    value === null ||
    (typeof value !== "object" && typeof value !== "function")
  ) {
    return false;
  }
  return typeof (value as { then?: unknown }).then === "function";
}

/** Whether a live registered Tool callback owns the current Memory read. */
export function isMemoryReadOwnedByTool(): boolean {
  return storage.getStore()?.active === true;
}

/**
 * Run a registered Tool callback as the owner of Memory reads it performs.
 * The mutable holder invalidates ownership in async descendants after the
 * callback returns or its promise settles.
 */
export function runWithToolMemoryReadOwnership<T>(fn: () => T): T {
  const ownership: ToolMemoryReadOwnership = { active: true };
  try {
    const result = storage.run(ownership, fn);
    if (!isPromiseLike(result)) {
      ownership.active = false;
      return result;
    }
    return Promise.resolve(result).finally(() => {
      ownership.active = false;
    }) as T;
  } catch (error) {
    ownership.active = false;
    throw error;
  }
}
