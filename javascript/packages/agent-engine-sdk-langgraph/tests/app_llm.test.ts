/**
 * Port of `agent-engine-sdk-langgraph/tests/test_app_llm.py`.
 *
 * Verifies `App.llm()` behavior across runtime modes — wrapping in
 * SecureWrappedLLM in AER mode, and registering in the named-LLM registry
 * in TOOL mode.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  entrypointScope,
  getNamedLlm,
  hasNamedLlms,
  resetHooks,
  RuntimeMode,
  type TenantRuntime,
} from "@mongodb-js/agent-engine-runner-shared";

import { App } from "../src/runtime.js";
import { SecureWrappedLLM } from "../src/secure_llm.js";

afterEach(() => {
  resetHooks();
});

describe("App.llm() — AER mode", () => {
  let savedMode: string | undefined;

  beforeEach(() => {
    savedMode = process.env["RUNNER_MODE"];
    process.env["RUNNER_MODE"] = "aer";
  });

  afterEach(() => {
    if (savedMode === undefined) delete process.env["RUNNER_MODE"];
    else process.env["RUNNER_MODE"] = savedMode;
  });

  it("wraps the LLM in SecureWrappedLLM", () => {
    const app = new App({ appName: "test-agent" });
    const fakeLlm = { invoke: vi.fn() } as never;

    const wrapped = entrypointScope(() => app.llm(fakeLlm));

    expect(wrapped).toBeInstanceOf(SecureWrappedLLM);
    expect(wrapped).not.toBe(fakeLlm);
  });

  it("unnamed call uses __default__ sentinel id", () => {
    const app = new App({ appName: "test-agent" });
    const fakeLlm = { invoke: vi.fn() } as never;

    const wrapped = entrypointScope(
      () => app.llm(fakeLlm) as unknown as SecureWrappedLLM,
    );

    // SecureWrappedLLM stores the llmId — defaults to "__default__"
    expect((wrapped as unknown as { llmId: string }).llmId).toBe("__default__");
  });

  it("named call stores the given llmId on SecureWrappedLLM", () => {
    const app = new App({ appName: "test-agent" });
    const fakeLlm = { invoke: vi.fn() } as never;

    const wrapped = entrypointScope(
      () =>
        app.llm(fakeLlm, "primary") as unknown as {
          llmId: string;
        },
    );

    expect(wrapped.llmId).toBe("primary");
  });

  it("every call returns a fresh wrapper (no shared state)", () => {
    const app = new App({ appName: "test-agent" });
    const [a, b] = entrypointScope(() => [
      app.llm({ invoke: vi.fn() } as never),
      app.llm({ invoke: vi.fn() } as never, "second"),
    ]);
    expect(a).not.toBe(b);
  });
});

describe("App.llm() — TOOL mode", () => {
  let savedMode: string | undefined;

  beforeEach(() => {
    savedMode = process.env["RUNNER_MODE"];
    process.env["RUNNER_MODE"] = "tool";
  });

  afterEach(() => {
    if (savedMode === undefined) delete process.env["RUNNER_MODE"];
    else process.env["RUNNER_MODE"] = savedMode;
  });

  it("returns the raw LLM unwrapped", () => {
    const app = new App({ appName: "test-agent" });
    const fakeLlm = { invoke: vi.fn(), tag: "raw" } as never;
    const result = entrypointScope(() => app.llm(fakeLlm));
    expect(result).toBe(fakeLlm);
  });

  it("unnamed call registers under __default__ sentinel", () => {
    const app = new App({ appName: "test-agent" });
    const fakeLlm = { invoke: vi.fn() } as never;

    entrypointScope(() => app.llm(fakeLlm));

    expect(getNamedLlm("__default__")).toBe(fakeLlm);
  });

  it("named call registers under given llmId", () => {
    const app = new App({ appName: "test-agent" });
    const fakeLlm = { invoke: vi.fn() } as never;

    entrypointScope(() => app.llm(fakeLlm, "primary"));

    expect(getNamedLlm("primary")).toBe(fakeLlm);
  });

  it("two different llmIds both survive — no overwrite", () => {
    const app = new App({ appName: "test-agent" });
    const llmA = { invoke: vi.fn(), tag: "a" } as never;
    const llmB = { invoke: vi.fn(), tag: "b" } as never;

    entrypointScope(() => {
      app.llm(llmA, "primary");
      app.llm(llmB, "fast");
    });

    expect(getNamedLlm("primary")).toBe(llmA);
    expect(getNamedLlm("fast")).toBe(llmB);
  });

  it("duplicate llmId throws", () => {
    const app = new App({ appName: "test-agent" });
    entrypointScope(() => {
      app.llm({ invoke: vi.fn() } as never, "primary");

      expect(() => app.llm({ invoke: vi.fn() } as never, "primary")).toThrow(
        "already registered",
      );
    });
  });

  it("second unnamed call throws (both collide on __default__)", () => {
    const app = new App({ appName: "test-agent" });
    entrypointScope(() => {
      app.llm({ invoke: vi.fn() } as never);

      expect(() => app.llm({ invoke: vi.fn() } as never)).toThrow(
        "already registered",
      );
    });
  });

  it("hasNamedLlms() returns true after registration", () => {
    const app = new App({ appName: "test-agent" });
    expect(hasNamedLlms()).toBe(false);
    entrypointScope(() => app.llm({ invoke: vi.fn() } as never, "primary"));
    expect(hasNamedLlms()).toBe(true);
  });

  it("runtime mode is TOOL when RUNNER_MODE=tool", () => {
    const app = new App({ appName: "test-agent" });
    expect((app as unknown as { runtime: TenantRuntime }).runtime.mode).toBe(
      RuntimeMode.TOOL,
    );
  });
});
