/**
 * Tests for the Tool Pod's registry load on a failed entrypoint.
 *
 * A global agent's failed build keeps the previous contract: no half-populated
 * registry survives. An agent that opted into per-execution builds registers its
 * LLMs and only then reads invocation context, which the context-free discovery
 * path cannot supply, so what it registered before raising is kept.
 *
 * The opted-in path goes through the App's async `buildGraph`, not the cached
 * `getAgent`, because the builder is async and would otherwise leave its
 * rejection outside this boundary.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  populateLlmRegistryFromEntrypoint,
  registerLlm,
  resetHooks,
  snapshotLlmRegistry,
} from "../../src/hooks.js";

const warnings = { entrypointFailed: "failed", noLlmRegistered: "none" };
const perExecution = { usesPerInvocationBuild: true };
const globalAgent = { usesPerInvocationBuild: false };

function silentLogger() {
  return { warn: vi.fn(), info: vi.fn(), error: vi.fn() };
}

function driftError(): Error {
  const error = new Error("drifted") as Error & {
    isLlmRegistrationDrift: boolean;
  };
  error.isLlmRegistrationDrift = true;
  return error;
}

describe("populateLlmRegistryFromEntrypoint on a failed build", () => {
  beforeEach(() => resetHooks());
  afterEach(() => vi.unstubAllEnvs());

  it("discards partial registrations for a global agent", async () => {
    const graphBuilder = {
      getAgent: () => {
        registerLlm("partial", {});
        throw new Error("boom");
      },
    };

    const populated = await populateLlmRegistryFromEntrypoint(
      graphBuilder,
      silentLogger(),
      warnings,
      undefined,
      globalAgent,
    );

    expect(populated).toBe(false);
    expect(snapshotLlmRegistry().size).toBe(0);
  });

  it("keeps partial registrations for an opted-in builder", async () => {
    const graphBuilder = {
      getAgent: vi.fn(),
      buildGraph: async () => {
        registerLlm("primary", {});
        throw new Error("needs invocation context");
      },
    };

    const populated = await populateLlmRegistryFromEntrypoint(
      graphBuilder,
      silentLogger(),
      warnings,
      undefined,
      perExecution,
    );

    expect(populated).toBe(false);
    expect([...snapshotLlmRegistry().keys()]).toEqual(["primary"]);
  });

  it("discovers through buildGraph rather than the cached getAgent path", async () => {
    const graphBuilder = {
      getAgent: vi.fn(),
      buildGraph: vi.fn(async () => {
        registerLlm("primary", {});
        return { graph: true };
      }),
    };

    await populateLlmRegistryFromEntrypoint(
      graphBuilder,
      silentLogger(),
      warnings,
      undefined,
      perExecution,
    );

    expect(graphBuilder.buildGraph).toHaveBeenCalled();
    expect(graphBuilder.getAgent).not.toHaveBeenCalled();
  });

  it("discards the drifted registry and reports the drift as the load error", async () => {
    let recorded: unknown = null;
    const logger = silentLogger();
    const graphBuilder = {
      getAgent: vi.fn(),
      buildGraph: async () => {
        registerLlm("drifted", {});
        throw driftError();
      },
    };

    const populated = await populateLlmRegistryFromEntrypoint(
      graphBuilder,
      logger,
      warnings,
      (error) => {
        recorded = error;
      },
      perExecution,
    );

    expect(populated).toBe(false);
    expect(snapshotLlmRegistry().size).toBe(0);
    expect(
      (recorded as { isLlmRegistrationDrift?: boolean })
        ?.isLlmRegistrationDrift,
    ).toBe(true);
    expect(logger.error).toHaveBeenCalledWith(
      expect.stringContaining("discarding the mismatched registry"),
    );
  });

  it("keeps nothing when an opted-in builder raised before registering anything", async () => {
    const graphBuilder = {
      getAgent: vi.fn(),
      buildGraph: async () => {
        throw new Error("needs invocation context");
      },
    };

    const populated = await populateLlmRegistryFromEntrypoint(
      graphBuilder,
      silentLogger(),
      warnings,
      undefined,
      perExecution,
    );

    expect(populated).toBe(false);
    expect(snapshotLlmRegistry().size).toBe(0);
  });

  it("applies the previous contract when no strategy is supplied", async () => {
    // Pins the default for a caller that omits the strategy.
    const graphBuilder = {
      getAgent: () => {
        registerLlm("partial", {});
        throw new Error("boom");
      },
    };

    await populateLlmRegistryFromEntrypoint(
      graphBuilder,
      silentLogger(),
      warnings,
    );

    expect(snapshotLlmRegistry().size).toBe(0);
  });

  it("bounds a discovery build that never settles, keeping what it registered", async () => {
    // Discovery runs with the caller's registry lock held, so an unbounded
    // await would queue every later model request behind it.
    vi.stubEnv("RUNNER_EXECUTION_TIMEOUT", "0.02");
    const graphBuilder = {
      getAgent: vi.fn(),
      buildGraph: async () => {
        registerLlm("primary", {});
        await new Promise(() => {});
        return {};
      },
    };

    const populated = await populateLlmRegistryFromEntrypoint(
      graphBuilder,
      silentLogger(),
      warnings,
      undefined,
      perExecution,
    );

    expect(populated).toBe(false);
    expect([...snapshotLlmRegistry().keys()]).toEqual(["primary"]);
  });

  it("allows discovery beyond ten minutes when the runner timeout is unset", async () => {
    vi.stubEnv("RUNNER_EXECUTION_TIMEOUT", undefined);
    const graphBuilder = {
      getAgent: vi.fn(),
      buildGraph: async () => {
        await new Promise((resolve) => setTimeout(resolve, 11 * 60 * 1000));
        registerLlm("primary", {});
        return {};
      },
    };

    vi.useFakeTimers();
    try {
      const pending = populateLlmRegistryFromEntrypoint(
        graphBuilder,
        silentLogger(),
        warnings,
        undefined,
        perExecution,
      );
      await vi.advanceTimersByTimeAsync(11 * 60 * 1000);
      expect(await pending).toBe(true);
      expect([...snapshotLlmRegistry().keys()]).toEqual(["primary"]);
    } finally {
      vi.useRealTimers();
    }
  });

  it("still populates the registry from a successful build", async () => {
    const graphBuilder = {
      getAgent: vi.fn(),
      buildGraph: async () => {
        registerLlm("primary", {});
        return { graph: true };
      },
    };

    const populated = await populateLlmRegistryFromEntrypoint(
      graphBuilder,
      silentLogger(),
      warnings,
      undefined,
      perExecution,
    );

    expect(populated).toBe(true);
    expect([...snapshotLlmRegistry().keys()]).toEqual(["primary"]);
  });
});
