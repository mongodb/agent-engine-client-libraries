/**
 * Tests for TenantRuntime.buildGraph and usesPerInvocationBuild.
 *
 * The runtime's `graphBuilder` is assigned by `registerAndRun()`, which also
 * starts a Fastify server. These tests set it directly, casting through
 * `unknown` — the same pattern the other runtime unit tests use.
 */

import { describe, expect, it, vi } from "vitest";
import { TenantRuntime } from "../../src/runtime.js";

function withBuilder(builder: unknown): TenantRuntime {
  const runtime = new TenantRuntime();
  (runtime as unknown as { graphBuilder: unknown }).graphBuilder = builder;
  return runtime;
}

describe("TenantRuntime.buildGraph", () => {
  it("resolves undefined when the App declares no per-execution build", async () => {
    const runtime = withBuilder({ getAgent: () => ({}) });
    await expect(runtime.buildGraph()).resolves.toBeUndefined();
  });

  it("resolves undefined when no builder is registered", async () => {
    const runtime = new TenantRuntime();
    await expect(runtime.buildGraph()).resolves.toBeUndefined();
  });

  it("delegates to the App buildGraph with the invocation's input and context", async () => {
    const graph = { built: true };
    const buildGraph = vi.fn().mockResolvedValue(graph);
    const runtime = withBuilder({ getAgent: () => ({}), buildGraph });
    const input = { payload: {} } as never;
    const ctx = {} as never;

    await expect(runtime.buildGraph(input, ctx)).resolves.toBe(graph);
    expect(buildGraph).toHaveBeenCalledWith(input, ctx);
  });

  it("propagates a build failure to the caller", async () => {
    const runtime = withBuilder({
      getAgent: () => ({}),
      buildGraph: vi.fn().mockRejectedValue(new Error("build failed")),
    });
    await expect(runtime.buildGraph()).rejects.toThrow("build failed");
  });
});

describe("TenantRuntime.usesPerInvocationBuild", () => {
  it("is false for a builder that does not declare it", () => {
    expect(withBuilder({ getAgent: () => ({}) }).usesPerInvocationBuild()).toBe(
      false,
    );
  });

  it("is false when no builder is registered", () => {
    expect(new TenantRuntime().usesPerInvocationBuild()).toBe(false);
  });

  it("reports what the builder declares", () => {
    const runtime = withBuilder({ usesPerInvocationBuild: () => true });
    expect(runtime.usesPerInvocationBuild()).toBe(true);
  });
});
