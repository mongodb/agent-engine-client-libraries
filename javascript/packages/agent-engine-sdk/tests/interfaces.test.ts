/** Mirrors the subset of sdk-core/tests/test_protocols.py that ports to TS.
 *
 * Python-only tests (runtime-checkable Protocol isinstance assertions) are
 * deliberately omitted — TS interfaces are erased at compile time and have no
 * runtime representation. Verifying that a concrete class satisfies the
 * interface is done at compile time by tsc, not at test time.
 */

import { describe, expect, it } from "vitest";
import { NullExecutionCallback } from "../src/interfaces.js";

describe("NullExecutionCallback", () => {
  it("all methods are callable without throwing", () => {
    const cb = new NullExecutionCallback();
    expect(() => cb.onNodeStart("n", {}, { runId: "r" })).not.toThrow();
    expect(() => cb.onNodeEnd("n", {}, { runId: "r" })).not.toThrow();
    expect(() => cb.onNodeError("n", "err", { runId: "r" })).not.toThrow();
    expect(() => cb.onNodeSuspend("n", { runId: "r" })).not.toThrow();
  });
});
