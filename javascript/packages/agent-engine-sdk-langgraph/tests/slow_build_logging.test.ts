/**
 * The slow-build budget decides whether a per-execution build is reported.
 *
 * The budget is read per call, so these cases stub the environment rather than
 * re-importing the module.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import { isSlowGraphBuild } from "../src/runtime.js";

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("isSlowGraphBuild", () => {
  it("defaults to a one second budget", () => {
    expect(isSlowGraphBuild(999)).toBe(false);
    expect(isSlowGraphBuild(1001)).toBe(true);
  });

  it("honours an explicit budget", () => {
    vi.stubEnv("RUNNER_SLOW_GRAPH_BUILD_MS", "250");
    expect(isSlowGraphBuild(249)).toBe(false);
    expect(isSlowGraphBuild(251)).toBe(true);
  });

  it("reports every build when the budget is zero", () => {
    vi.stubEnv("RUNNER_SLOW_GRAPH_BUILD_MS", "0");
    expect(isSlowGraphBuild(1)).toBe(true);
  });

  it("falls back to the default when the value is unparseable", () => {
    vi.stubEnv("RUNNER_SLOW_GRAPH_BUILD_MS", "not-a-number");
    expect(isSlowGraphBuild(999)).toBe(false);
    expect(isSlowGraphBuild(1001)).toBe(true);
  });
});
