/**
 * Tests for agent_engine_runner_shared.hooks — framework hook registry.
 *
 * Mirrors Python's tests/unit/test_hooks.py.
 *
 * Hooks are module-level singletons; mutations persist across tests within
 * a file. afterEach calls resetHooks() to mirror Python's autouse
 * `_reset_hooks` fixture (which yields then resets after the test).
 *
 * Note on Python test naming: `test_get_raises_when_unregistered` is used
 * for both SuspendHandler and Instrumentor in Python, but those actually
 * return None (don't raise). Only get_llm_factory / get_llm_adapter_factory
 * throw. TS test names describe the actual behaviour for clarity.
 */

import { describe, test, expect, afterEach, vi } from "vitest";
import {
  registerSuspendHandler,
  getSuspendHandler,
  registerLlm,
  getNamedLlm,
  hasNamedLlms,
  registerLLMAdapterFactory,
  getLLMAdapterFactory,
  registerInstrumentor,
  getInstrumentor,
  registerQueryPlugin,
  getQueryPlugin,
  registerWorkflowAdapter,
  clearWorkflowAdapter,
  getWorkflowAdapter,
  resetHooks,
  entrypointScope,
  type SuspendHandler,
  type LLMAdapterFactory,
  type Instrumentor,
  type AERQueryPlugin,
} from "../../src/index.js";
import { populateLlmRegistryFromEntrypoint } from "../../src/hooks.js";

afterEach(() => {
  resetHooks();
});

// ---------------------------------------------------------------------------
// Stubs (mirror Python's _stub_*)
// ---------------------------------------------------------------------------

const stubSuspend: SuspendHandler = () => ({ decision: "approved" });
// LLMAdapterFactory is typed to return BaseLLM; the test never invokes it,
// it only verifies storage/retrieval. Cast through unknown to satisfy the
// type without constructing a real BaseLLM.
const stubAdapter = (() => "fake-adapter") as unknown as LLMAdapterFactory;
const stubInstrumentor: Instrumentor = () => undefined;
const stubQueryPlugin: AERQueryPlugin = {
  getSummariesForSessions: async () => ({ sessions: [] }),
  getMessagesForSession: async () => ({ messages: [] }),
};

// ---------------------------------------------------------------------------
// SuspendHandler
// ---------------------------------------------------------------------------

describe("SuspendHandler", () => {
  test("get returns null when unregistered", () => {
    // test_get_raises_when_unregistered (Python name; actually returns None)
    expect(getSuspendHandler()).toBeNull();
  });

  test("register then get returns the handler", () => {
    // test_register_and_get
    registerSuspendHandler(stubSuspend);
    expect(getSuspendHandler()).toBe(stubSuspend);
  });
});

// ---------------------------------------------------------------------------
// LLMAdapterFactory
// ---------------------------------------------------------------------------

describe("LLMAdapterFactory", () => {
  test("get throws when unregistered", () => {
    // test_get_raises_when_unregistered
    expect(() => getLLMAdapterFactory()).toThrow(
      /No LLM adapter factory registered/,
    );
  });

  test("register then get returns the factory", () => {
    // test_register_and_get
    registerLLMAdapterFactory(stubAdapter);
    expect(getLLMAdapterFactory()).toBe(stubAdapter);
  });
});

// ---------------------------------------------------------------------------
// Instrumentor
// ---------------------------------------------------------------------------

describe("Instrumentor", () => {
  test("get returns null when unregistered", () => {
    // test_get_returns_none_when_unregistered
    expect(getInstrumentor()).toBeNull();
  });

  test("register then get returns the instrumentor", () => {
    // test_register_and_get
    registerInstrumentor(stubInstrumentor);
    expect(getInstrumentor()).toBe(stubInstrumentor);
  });
});

// ---------------------------------------------------------------------------
// QueryPlugin
// ---------------------------------------------------------------------------

describe("QueryPlugin", () => {
  test("get returns null when unregistered", () => {
    expect(getQueryPlugin()).toBeNull();
  });

  test("register then get returns the plugin", () => {
    registerQueryPlugin(stubQueryPlugin);
    expect(getQueryPlugin()).toBe(stubQueryPlugin);
  });
});

// ---------------------------------------------------------------------------
// resetHooks
// ---------------------------------------------------------------------------

describe("resetHooks", () => {
  test("clears all registered hooks", () => {
    // test_reset_clears_all
    registerSuspendHandler(stubSuspend);
    registerLLMAdapterFactory(stubAdapter);
    registerInstrumentor(stubInstrumentor);
    registerQueryPlugin(stubQueryPlugin);
    registerWorkflowAdapter("langgraph", "1");
    entrypointScope(() => registerLlm("primary", {}));

    resetHooks();

    expect(getSuspendHandler()).toBeNull();
    expect(() => getLLMAdapterFactory()).toThrow(
      /No LLM adapter factory registered/,
    );
    expect(getInstrumentor()).toBeNull();
    expect(getQueryPlugin()).toBeNull();
    expect(getWorkflowAdapter()).toBeNull();
    expect(hasNamedLlms()).toBe(false);
  });
});

describe("WorkflowAdapter", () => {
  test("get returns null when unregistered", () => {
    expect(getWorkflowAdapter()).toBeNull();
  });

  test("register then get returns the adapter identity", () => {
    registerWorkflowAdapter("langgraph", "1.2.3");
    expect(getWorkflowAdapter()).toEqual({
      name: "langgraph",
      version: "1.2.3",
    });
  });

  test("rejects a blank name or version", () => {
    expect(() => registerWorkflowAdapter(" ", "1")).toThrow(/non-empty/);
    expect(() => registerWorkflowAdapter("langgraph", " ")).toThrow(
      /non-empty/,
    );
  });

  test("clear drops eligibility without resetting other hooks", () => {
    registerSuspendHandler(stubSuspend);
    registerWorkflowAdapter("langgraph", "1");
    clearWorkflowAdapter();
    expect(getWorkflowAdapter()).toBeNull();
    expect(getSuspendHandler()).toBe(stubSuspend);
  });
});

// ---------------------------------------------------------------------------
// Named-LLM registry (mirrors Python's TestLLMRegistry)
// ---------------------------------------------------------------------------

describe("LLMRegistry", () => {
  test("getNamedLlm throws when not registered", () => {
    // test_get_raises_when_not_registered
    expect(() => getNamedLlm("missing")).toThrow(
      /llm_id "missing" not registered/,
    );
  });

  test("register then get returns the llm", () => {
    // test_register_and_get
    const fakeLlm = {};
    entrypointScope(() => registerLlm("primary", fakeLlm));
    expect(getNamedLlm("primary")).toBe(fakeLlm);
  });

  test("hasNamedLlms is false when empty", () => {
    // test_has_named_llms_false_when_empty
    expect(hasNamedLlms()).toBe(false);
  });

  test("hasNamedLlms is true after register", () => {
    // test_has_named_llms_true_after_register
    entrypointScope(() => registerLlm("primary", {}));
    expect(hasNamedLlms()).toBe(true);
  });

  test("registers multiple ids", () => {
    // test_register_multiple_ids
    const llmA = {};
    const llmB = {};
    entrypointScope(() => {
      registerLlm("a", llmA);
      registerLlm("b", llmB);
    });
    expect(getNamedLlm("a")).toBe(llmA);
    expect(getNamedLlm("b")).toBe(llmB);
  });

  test("duplicate id throws", () => {
    // test_register_duplicate_id_raises
    entrypointScope(() => {
      registerLlm("primary", {});
      expect(() => registerLlm("primary", {})).toThrow(/already registered/);
    });
  });

  test("reset clears the registry", () => {
    // test_reset_clears_registry
    entrypointScope(() => registerLlm("x", {}));
    resetHooks();
    expect(hasNamedLlms()).toBe(false);
    expect(() => getNamedLlm("x")).toThrow();
  });
});

// ---------------------------------------------------------------------------
// populateLlmRegistryFromEntrypoint
// ---------------------------------------------------------------------------

describe("populateLlmRegistryFromEntrypoint", () => {
  const warnings = {
    entrypointFailed:
      "Failed to run entrypoint for LLM registration; /invoke_llm will fail if no LLM was registered",
    noLlmRegistered: "No LLM registered after entrypoint",
  };

  test("preserves this binding when calling getAgent", () => {
    const logger = { warn: vi.fn(), info: vi.fn() };
    class MockGraphBuilder {
      builderFn = (): unknown => ({});
      getAgent(): unknown {
        if (this.builderFn === null) {
          throw new Error("No entrypoint registered");
        }
        registerLlm("__default__", { model: "test" });
        return {};
      }
    }

    populateLlmRegistryFromEntrypoint(new MockGraphBuilder(), logger, warnings);

    expect(logger.warn).not.toHaveBeenCalledWith(warnings.entrypointFailed);
    expect(getNamedLlm("__default__")).toEqual({ model: "test" });
  });
});

// ---------------------------------------------------------------------------
// entrypointScope enforcement (mirrors Python's TestEntrypointScopeEnforcement)
// ---------------------------------------------------------------------------

describe("entrypointScope enforcement", () => {
  test("registerLlm outside entrypointScope throws", () => {
    expect(() => registerLlm("primary", {})).toThrow(/must be called inside/);
  });

  test("registerLlm inside entrypointScope succeeds", () => {
    entrypointScope(() => registerLlm("primary", {}));
    expect(getNamedLlm("primary")).toBeDefined();
  });

  test("nested entrypointScope restores outer state on exit", () => {
    entrypointScope(() => {
      entrypointScope(() => registerLlm("nested", {}));
      // Still inside the outer scope: should not throw.
      registerLlm("outer", {});
    });
    expect(() => registerLlm("after", {})).toThrow(/must be called inside/);
  });

  test("entrypointScope restores state when fn throws", () => {
    expect(() =>
      entrypointScope(() => {
        throw new Error("boom");
      }),
    ).toThrow("boom");
    expect(() => registerLlm("primary", {})).toThrow(/must be called inside/);
  });
});
