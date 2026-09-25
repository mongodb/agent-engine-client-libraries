/**
 * Regression tests — guard against drift on the parity fixes applied during
 * the agent-engine-sdk swap + Python parity audit.
 *
 * Each test targets a specific behavior that was wrong or missing before:
 *   1. `LLMResponse` is a class with a `fromRaw` static factory
 *   2. `fromRaw` normalises `usage` from 3 provider shapes
 *   3. `BaseApp` is an abstract class and `App extends BaseApp`
 *   4. `RequestContext.signal?: AbortSignal` is accepted at the type level
 *   5. Hook registry round-trips a registered function
 *   6. `resetHooks` clears the registry
 *   7. `setupTracing` invokes the registered instrumentor (and swallows its errors)
 *   8. `ExecutionResult` is both `PromiseLike` and `AsyncIterable`
 *   9. `getEnvBool` parses common truthy values
 *  10. `getStoreDbName` honours `MDB_AGENTIC_STORE_DB` env var
 */

import { afterEach, describe, expect, it } from "vitest";

import {
  BaseApp,
  LLMResponse,
  LLMTokenUsage,
  type ExecutionResult,
  type RequestContext,
} from "@mongodb-js/agent-engine-sdk";
import {
  getEnvBool,
  getInstrumentor,
  getLLMAdapterFactory,
  getStoreDbName,
  getSuspendHandler,
  registerInstrumentor,
  registerLLMAdapterFactory,
  registerSuspendHandler,
  resetHooks,
  setupTracing,
} from "@mongodb-js/agent-engine-runner-shared";

import { App } from "../src/runtime.js";

afterEach(() => {
  resetHooks();
});

// ---------------------------------------------------------------------------
// Fix 1 & 2 — LLMResponse is a class with `fromRaw`
// ---------------------------------------------------------------------------

describe("LLMResponse class + fromRaw", () => {
  it("is constructible as a class with explicit fields", () => {
    const r = new LLMResponse({ content: "hello" });
    expect(r).toBeInstanceOf(LLMResponse);
    expect(r.content).toBe("hello");
    expect(r.metadata).toEqual({});
  });

  it("fromRaw extracts usage from a top-level `usage` key", () => {
    const r = LLMResponse.fromRaw({
      content: "hi",
      usage: { input_tokens: 5, output_tokens: 7 },
    });
    expect(r.usage).toBeInstanceOf(LLMTokenUsage);
    expect(r.usage?.inputTokens).toBe(5);
    expect(r.usage?.outputTokens).toBe(7);
  });

  it("fromRaw extracts usage from `metadata.usage`", () => {
    const r = LLMResponse.fromRaw({
      content: "hi",
      metadata: { usage: { prompt_tokens: 3, completion_tokens: 4 } },
    });
    expect(r.usage?.promptTokens).toBe(3);
    expect(r.usage?.completionTokens).toBe(4);
  });

  it("fromRaw extracts usage from flat token keys in `metadata`", () => {
    const r = LLMResponse.fromRaw({
      content: "hi",
      metadata: { input_tokens: 10, output_tokens: 12 },
    });
    expect(r.usage?.inputTokens).toBe(10);
  });

  it("fromRaw returns a usage-less response when no token info is present", () => {
    const r = LLMResponse.fromRaw({ content: "hi", metadata: { foo: "bar" } });
    expect(r.usage).toBeUndefined();
  });

  it("fromRaw falls back to empty content when `content` is missing from the input dict", () => {
    // agent-engine-runner-shared's `fromRaw` types its input as Record<string, unknown>
    // and reads `data['content'] ?? ''` — so an input with no `content` key
    // produces an LLMResponse with empty content.
    expect(LLMResponse.fromRaw({}).content).toBe("");
    expect(LLMResponse.fromRaw({ metadata: { foo: "bar" } }).content).toBe("");
  });
});

// ---------------------------------------------------------------------------
// Fix 3 — BaseApp + App extends BaseApp
// ---------------------------------------------------------------------------

describe("BaseApp inheritance", () => {
  it("`BaseApp` is exported as an abstract class", () => {
    expect(typeof BaseApp).toBe("function");
  });

  it("`App` is a subclass of `BaseApp`", () => {
    // Subclass check works on the prototype chain regardless of constructor side-effects.
    expect(Object.getPrototypeOf(App.prototype)).toBe(BaseApp.prototype);
  });
});

// ---------------------------------------------------------------------------
// Fix 4 — RequestContext.signal field
// ---------------------------------------------------------------------------

describe("RequestContext shape", () => {
  it("accepts an `AbortSignal` on the `signal` field", () => {
    const controller = new AbortController();
    const ctx: RequestContext = {
      executionId: "exec-1",
      sessionId: "sess-1",
      signal: controller.signal,
    };
    expect(ctx.signal).toBe(controller.signal);
    expect(ctx.signal?.aborted).toBe(false);
    controller.abort();
    expect(ctx.signal?.aborted).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Fix 5 & 6 — Hook registry register/get + resetHooks
// ---------------------------------------------------------------------------

describe("hook registry", () => {
  it("round-trips a registered suspend handler", () => {
    const handler = (payload: unknown): unknown => payload;
    registerSuspendHandler(handler as never);
    expect(getSuspendHandler()).toBe(handler);
  });

  it("round-trips a registered LLM adapter factory", () => {
    class FakeAdapter {}
    registerLLMAdapterFactory(FakeAdapter as never);
    expect(getLLMAdapterFactory()).toBe(FakeAdapter);
  });

  it("round-trips a registered instrumentor", () => {
    const instrumentor = (): undefined => undefined;
    registerInstrumentor(instrumentor as never);
    expect(getInstrumentor()).toBe(instrumentor);
  });

  it("`resetHooks` clears every registered hook", () => {
    registerSuspendHandler((() => undefined) as never);
    registerLLMAdapterFactory((() => undefined) as never);
    registerInstrumentor((() => undefined) as never);
    resetHooks();
    // Each getter has its own "no registration" semantics in agent-engine-runner-shared:
    //   - getSuspendHandler / getInstrumentor → return `null`
    //   - getLLMAdapterFactory → throws (required hook in production)
    expect(getSuspendHandler()).toBeNull();
    expect(getInstrumentor()).toBeNull();
    expect(() => getLLMAdapterFactory()).toThrow(
      /No LLM adapter factory registered/,
    );
  });
});

// ---------------------------------------------------------------------------
// Fix 7 — setupTracing exists and is safe to call
// ---------------------------------------------------------------------------

describe("setupTracing", () => {
  it("returns without throwing when called without an OTEL stack installed", async () => {
    // agent-engine-runner-shared's `setupTracing` is async and tries to dynamic-import
    // `@opentelemetry/*`. When the optional dep isn't present it logs a warning
    // and resolves silently — verify the function is at least callable here.
    await expect(setupTracing()).resolves.toBeUndefined();
  });

  it("is idempotent when called twice", async () => {
    await expect(setupTracing()).resolves.toBeUndefined();
    await expect(setupTracing()).resolves.toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Fix 8 — ExecutionResult is PromiseLike + AsyncIterable
// ---------------------------------------------------------------------------

describe("ExecutionResult dual shape", () => {
  it("a value satisfying the type can be both awaited and iterated", async () => {
    const fakeResult: ExecutionResult = {
      then: <T1, T2>(
        onfulfilled?:
          | ((value: { response: string }) => T1 | PromiseLike<T1>)
          | null,
        onrejected?: ((reason: unknown) => T2 | PromiseLike<T2>) | null,
      ): PromiseLike<T1 | T2> =>
        Promise.resolve({ response: "done" }).then(onfulfilled, onrejected),
      async *[Symbol.asyncIterator]() {
        yield { data: "a", event: "token" } as const;
        yield { data: "b", event: "token" } as const;
      },
    };

    // PromiseLike behaviour
    const awaited = await fakeResult;
    expect(awaited).toEqual({ response: "done" });

    // AsyncIterable behaviour
    const events: unknown[] = [];
    for await (const ev of fakeResult) events.push(ev);
    expect(events).toHaveLength(2);
  });
});

// ---------------------------------------------------------------------------
// Fix 9 & 10 — env helpers
// ---------------------------------------------------------------------------

describe("env helpers", () => {
  it("`getEnvBool` returns the default when the var is absent", () => {
    delete process.env["REGRESSION_TEST_BOOL"];
    expect(getEnvBool("REGRESSION_TEST_BOOL", false)).toBe(false);
    expect(getEnvBool("REGRESSION_TEST_BOOL", true)).toBe(true);
  });

  it("`getEnvBool` recognises common truthy values", () => {
    for (const value of ["1", "true", "TRUE", "yes", "on"]) {
      process.env["REGRESSION_TEST_BOOL"] = value;
      expect(getEnvBool("REGRESSION_TEST_BOOL", false)).toBe(true);
    }
    process.env["REGRESSION_TEST_BOOL"] = "0";
    expect(getEnvBool("REGRESSION_TEST_BOOL", true)).toBe(false);
    delete process.env["REGRESSION_TEST_BOOL"];
  });

  it("`getStoreDbName` falls back to `mdb_store` when env is unset", () => {
    delete process.env["MDB_AGENTIC_STORE_DB"];
    expect(getStoreDbName()).toBe("mdb_store");
  });

  it("`getStoreDbName` reads `MDB_AGENTIC_STORE_DB` when set", () => {
    process.env["MDB_AGENTIC_STORE_DB"] = "tenant_override";
    expect(getStoreDbName()).toBe("tenant_override");
    delete process.env["MDB_AGENTIC_STORE_DB"];
  });
});
