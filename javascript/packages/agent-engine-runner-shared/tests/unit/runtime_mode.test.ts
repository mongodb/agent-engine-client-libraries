import { afterEach, describe, expect, test, vi } from "vitest";

import { getRuntimeMode, RuntimeMode } from "../../src/utils.js";

describe("getRuntimeMode", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  test.each([
    ["aer", RuntimeMode.AER],
    ["tool", RuntimeMode.TOOL],
  ])("resolves RUNNER_MODE=%s", (raw, expected) => {
    vi.stubEnv("RUNNER_MODE", raw);
    expect(getRuntimeMode()).toBe(expected);
  });

  test("rejects the removed tool_function mode", () => {
    vi.stubEnv("RUNNER_MODE", "tool_function");
    expect(() => getRuntimeMode()).toThrow(/Valid values: aer, tool/);
  });
});
