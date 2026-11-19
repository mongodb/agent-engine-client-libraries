import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";
import type * as TracingIndex from "../../src/tracing/index.js";

const mocks = vi.hoisted(() => ({
  logger: {
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
    trace: vi.fn(),
    fatal: vi.fn(),
    log: vi.fn(),
  },
}));

vi.mock("../../src/logger.js", () => ({
  getLogger: () => mocks.logger,
  setupLogging: vi.fn(),
}));

vi.mock("../../src/tracing/index.js", async (importOriginal) => {
  const actual = await importOriginal<typeof TracingIndex>();
  return {
    ...actual,
    setupTracing: () =>
      Promise.reject(
        "driver error: mongodb://admin:p@ss word@atlas-host.mongodb.net/db",
      ),
    shutdownTracing: vi.fn(),
  };
});

import { TenantRuntime } from "../../src/index.js";

describe("TenantRuntime tracing setup failure handling", () => {
  beforeEach(() => {
    for (const k of ["info", "error", "warn", "debug"] as const)
      mocks.logger[k].mockClear();
  });

  afterEach(async () => {
    const { shutdownTracing } = await import("../../src/tracing/index.js");
    await shutdownTracing();
  });

  test("a string rejection is swallowed and scrubbed, never rethrown", async () => {
    const runtime = new TenantRuntime({ appName: "Test" });

    await expect(
      (runtime as unknown as { tracingReady: Promise<void> }).tracingReady,
    ).resolves.toBeUndefined();

    const warnOutput = mocks.logger.warn.mock.calls
      .flatMap((call) => call.map(String))
      .join("\n");
    expect(warnOutput).toContain("Tracing setup failed");
    expect(warnOutput).toContain("mongodb://***@atlas-host.mongodb.net/db");
    expect(warnOutput).not.toContain("p@ss");
    expect(warnOutput).not.toContain("admin:");
  });
});
