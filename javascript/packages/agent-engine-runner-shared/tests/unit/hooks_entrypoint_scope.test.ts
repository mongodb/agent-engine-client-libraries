/**
 * Tests for the entrypoint scope.
 *
 * The scope is backed by AsyncLocalStorage rather than a process-wide flag, so
 * permission to register an LLM belongs to the async context that opened it. A
 * per-execution builder holds the scope across its own I/O, and a global flag
 * would authorize any other task running during those awaits to register into
 * the registry that build's id set is validated against.
 */

import { beforeEach, describe, expect, it } from "vitest";
import {
  entrypointScope,
  entrypointScopeAsync,
  registerLlm,
  resetHooks,
  snapshotLlmRegistry,
} from "../../src/hooks.js";

describe("entrypoint scope", () => {
  beforeEach(() => resetHooks());

  it("authorizes registration inside a synchronous scope", () => {
    entrypointScope(() => {
      expect(() => registerLlm("sync", {})).not.toThrow();
    });
  });

  it("keeps the scope open across an await", async () => {
    await entrypointScopeAsync(async () => {
      await new Promise((resolve) => setTimeout(resolve, 1));
      expect(() => registerLlm("after-await", {})).not.toThrow();
    });
  });

  it("leaves the scope closed outside the callback", () => {
    expect(() => registerLlm("outside", {})).toThrow(/must be called inside/);
  });

  it("restores the previous scope state on throw", async () => {
    await expect(
      entrypointScopeAsync(async () => {
        throw new Error("builder failed");
      }),
    ).rejects.toThrow("builder failed");
    expect(() => registerLlm("outside", {})).toThrow(/must be called inside/);
  });

  it("treats a nested scope as a no-op", () => {
    entrypointScope(() => {
      entrypointScope(() => {
        expect(() => registerLlm("nested", {})).not.toThrow();
      });
      expect(() => registerLlm("after-nested", {})).not.toThrow();
    });
  });

  it("rejects a registration from a timer that outlives the evaluation", async () => {
    // AsyncLocalStorage keeps the store for async resources created inside the
    // callback, so this timer still carries it. The scope has to end with the
    // evaluation, or a stray registration would land after the build returned
    // and its id set had already been validated.
    let late: Promise<void> = Promise.resolve();
    await entrypointScopeAsync(async () => {
      late = new Promise<void>((resolve) => {
        setTimeout(() => {
          try {
            registerLlm("late", {});
          } catch {
            // Rejected, which is what the assertion below checks for.
          }
          resolve();
        }, 20);
      });
    });

    await late;

    expect(snapshotLlmRegistry().has("late")).toBe(false);
  });

  it("does not authorize an unrelated async context while a build is paused", async () => {
    let release: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });

    const build = entrypointScopeAsync(async () => {
      await gate;
      return "done";
    });

    // Started from outside the scope's context, and running while the build is
    // paused on its I/O.
    const intruder = (async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
      registerLlm("intruder", {});
    })();

    await expect(intruder).rejects.toThrow(/must be called inside/);

    release();
    await expect(build).resolves.toBe("done");
  });
});
