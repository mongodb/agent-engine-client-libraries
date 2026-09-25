import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { RuntimeMode } from "../../src/index.js";
import { shutdownGracePeriodMs } from "../../src/runtime.js";

beforeEach(() => {
  delete process.env["SHUTDOWN_GRACE_PERIOD_MS"];
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("shutdownGracePeriodMs", () => {
  test("uses the configured AER drain budget", () => {
    vi.stubEnv("SHUTDOWN_GRACE_PERIOD_MS", "1800000");
    expect(shutdownGracePeriodMs(RuntimeMode.AER)).toBe(1_800_000);
  });

  test.each(["invalid", "0", "-1"])(
    "rejects an invalid AER drain budget: %s",
    (raw) => {
      vi.stubEnv("SHUTDOWN_GRACE_PERIOD_MS", raw);
      expect(() => shutdownGracePeriodMs(RuntimeMode.AER)).toThrow(
        "SHUTDOWN_GRACE_PERIOD_MS must be a positive integer in milliseconds",
      );
    },
  );

  test("keeps the existing default when the AER budget is absent", () => {
    expect(shutdownGracePeriodMs(RuntimeMode.AER)).toBe(25_000);
  });

  test.each([
    ["1800000", 1_800_000],
    ["invalid", 25_000],
    ["0", 0],
    ["-1", -1],
  ])("preserves Tool parsing for %s", (raw, expected) => {
    vi.stubEnv("SHUTDOWN_GRACE_PERIOD_MS", raw);
    expect(shutdownGracePeriodMs(RuntimeMode.TOOL)).toBe(expected);
  });
});
