/**
 * Regression: a tracing setup rejection carrying a non-Error value (a string
 * throw) must stay "logged and swallowed" — the catch handler in
 * TenantRuntime.initTracing must not throw itself (turning a degraded trace
 * store into a startup failure), and the rejected content must still be
 * credential-scrubbed before it reaches the log sink.
 */

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
    // A rejection carrying a plain string (not an Error) whose text embeds a
    // credential-bearing URI — the shape that previously threw inside the
    // catch handler via (e as Error).message being undefined.
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

    // tracingReady must RESOLVE: the pre-fix catch threw on the undefined
    // .message, leaving the promise rejected and failing startup.
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
