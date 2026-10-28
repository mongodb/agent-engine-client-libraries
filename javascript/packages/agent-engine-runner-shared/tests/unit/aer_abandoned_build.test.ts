/**
 * Lifetime of a per-execution graph build whose caller gave up.
 *
 * Two properties, both driven through the real AER code rather than around it:
 * the execution stays registered as drain work while the build is live, so a
 * drain cannot report the pod idle; and a build that never settles takes the
 * pod out of service through the runner's fatal path, so it cannot sit wedged
 * while the probes keep passing.
 */

import { afterEach, describe, expect, test, vi } from "vitest";

import { AERServer } from "../../src/index.js";
import type { DrainRegistry } from "../../src/server/drain.js";
import { TimeoutError } from "../../src/timeout.js";

interface AerPrivates {
  readonly drainRegistry: DrainRegistry;
  runtime: {
    buildGraph?: (input?: unknown, ctx?: unknown) => Promise<unknown>;
  };
  abandonedBuilds: Map<string, { build: Promise<unknown> }>;
  buildGraphForInvocation: (
    executionId: string,
    agentInput: unknown,
    ctx: unknown,
    timeoutMs?: number,
  ) => Promise<unknown>;
  trackAbandonedBuild: (executionId: string, build: Promise<unknown>) => void;
  endWorkForExecution: (executionId: string) => void;
  exitForUnsettledBuild: (executionId: string) => void;
}

function makeServer(): AerPrivates {
  // `drainRegistry` is a lazy getter on the prototype, so it needs no setup.
  const server = Object.create(AERServer.prototype) as AerPrivates;
  server.abandonedBuilds = new Map();
  server.runtime = {};
  return server;
}

function flush(ms = 0): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** A promise whose settlement the test controls. */
function deferred(): { promise: Promise<void>; settle: () => void } {
  let settle: () => void = () => {};
  const promise = new Promise<void>((resolve) => {
    settle = resolve;
  });
  return { promise, settle };
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("an abandoned per-execution build", () => {
  test("keeps the execution registered as drain work until the build settles", async () => {
    const server = makeServer();
    const executionId = "exec-abandoned";
    server.drainRegistry.beginWork(executionId);

    const { promise, settle } = deferred();
    server.trackAbandonedBuild(executionId, promise);

    const endWork = vi.spyOn(server.drainRegistry, "endWork");
    server.endWorkForExecution(executionId);
    await flush();

    // The build is still live, so the execution stays registered as work.
    expect(endWork).not.toHaveBeenCalled();

    settle();
    await flush();

    expect(endWork).toHaveBeenCalledWith(executionId);
    expect(server.abandonedBuilds.size).toBe(0);
  });

  test("ends drain work immediately when no build was abandoned", () => {
    const server = makeServer();
    const executionId = "exec-normal";
    server.drainRegistry.beginWork(executionId);

    const endWork = vi.spyOn(server.drainRegistry, "endWork");
    server.endWorkForExecution(executionId);

    expect(endWork).toHaveBeenCalledWith(executionId);
  });

  test("releases the execution when the abandoned build settles by failing", async () => {
    const server = makeServer();
    const executionId = "exec-failed";
    server.drainRegistry.beginWork(executionId);

    let fail: (error: Error) => void = () => {};
    const build = new Promise<void>((_resolve, reject) => {
      fail = reject;
    });
    server.trackAbandonedBuild(executionId, build);
    // The build's own consumer is gone, so the rejection must not surface as
    // unhandled.
    build.catch(() => {});

    const endWork = vi.spyOn(server.drainRegistry, "endWork");
    server.endWorkForExecution(executionId);
    await flush();
    expect(endWork).not.toHaveBeenCalled();

    fail(new Error("builder failed"));
    await flush();

    expect(endWork).toHaveBeenCalledWith(executionId);
    expect(server.abandonedBuilds.size).toBe(0);
  });

  test("takes the pod out of service when the build never settles", async () => {
    vi.stubEnv("RUNNER_ABANDONED_BUILD_GRACE", "0.01");
    const server = makeServer();
    const exit = vi
      .spyOn(server, "exitForUnsettledBuild")
      .mockImplementation(() => {});

    server.trackAbandonedBuild("exec-wedged", new Promise<void>(() => {}));
    await flush(30);

    expect(exit).toHaveBeenCalledWith("exec-wedged");
  });

  test("stays in service when the build settles inside the grace period", async () => {
    vi.stubEnv("RUNNER_ABANDONED_BUILD_GRACE", "0.05");
    const server = makeServer();
    const exit = vi
      .spyOn(server, "exitForUnsettledBuild")
      .mockImplementation(() => {});

    const { promise, settle } = deferred();
    server.trackAbandonedBuild("exec-recovered", promise);
    settle();
    await flush(80);

    expect(exit).not.toHaveBeenCalled();
  });
});

describe("buildGraphForInvocation", () => {
  test("bounds the caller, keeps the work owned, and recovers on settlement", async () => {
    vi.stubEnv("RUNNER_ABANDONED_BUILD_GRACE", "60");
    const server = makeServer();
    const executionId = "exec-1";
    server.drainRegistry.beginWork(executionId);

    const { promise, settle } = deferred();
    server.runtime = { buildGraph: () => promise };
    const exit = vi
      .spyOn(server, "exitForUnsettledBuild")
      .mockImplementation(() => {});

    // The caller gets a bounded error rather than waiting on the build.
    await expect(
      server.buildGraphForInvocation(executionId, {}, {}, 10),
    ).rejects.toBeInstanceOf(TimeoutError);

    // The live build is now owned, so ending the request cannot report the
    // execution idle while author code runs.
    expect(server.abandonedBuilds.has(executionId)).toBe(true);
    const endWork = vi.spyOn(server.drainRegistry, "endWork");
    server.endWorkForExecution(executionId);
    await flush();
    expect(endWork).not.toHaveBeenCalled();

    // Settlement releases both, and the pod keeps serving.
    settle();
    await flush();
    expect(endWork).toHaveBeenCalledWith(executionId);
    expect(server.abandonedBuilds.size).toBe(0);
    expect(exit).not.toHaveBeenCalled();
  });

  test("takes the pod out of service when the bounded build never settles", async () => {
    vi.stubEnv("RUNNER_ABANDONED_BUILD_GRACE", "0.01");
    const server = makeServer();
    const executionId = "exec-never";
    server.drainRegistry.beginWork(executionId);
    server.runtime = { buildGraph: () => new Promise<void>(() => {}) };
    const exit = vi
      .spyOn(server, "exitForUnsettledBuild")
      .mockImplementation(() => {});

    await expect(
      server.buildGraphForInvocation(executionId, {}, {}, 10),
    ).rejects.toBeInstanceOf(TimeoutError);
    await flush(30);

    expect(exit).toHaveBeenCalledWith(executionId);
  });

  test("leaves a global agent alone: no build, no tracking", async () => {
    const server = makeServer();
    server.runtime = { buildGraph: async () => undefined };

    await expect(
      server.buildGraphForInvocation("exec-global", {}, {}, 10),
    ).resolves.toBeUndefined();
    expect(server.abandonedBuilds.size).toBe(0);
  });
});
