/**
 * Port of `agent-engine-sdk-langgraph/tests/test_hook_registration.py`.
 *
 * Verifies `App.run()` registers the framework hooks (suspend handler, LLM
 * adapter factory, instrumentor, query plugin) with runner-shared BEFORE the
 * per-mode Fastify server starts.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { create } from "@bufbuild/protobuf";

import { LangChainInstrumentation } from "@arizeai/openinference-instrumentation-langchain";

import {
  getInstrumentor,
  getLLMAdapterFactory,
  getQueryPlugin,
  getSuspendHandler,
  resetHooks,
  runInstrumentor,
  AttemptContextSchema,
  runWithAttemptContext,
  type TenantRuntime,
} from "@mongodb-js/agent-engine-runner-shared";

import { App } from "../src/runtime.js";
import { LangGraphQueryPlugin } from "../src/query.js";
import { PlatformCheckpointer } from "../src/platform_checkpointer.js";

let savedMode: string | undefined;
let savedMongoUri: string | undefined;

beforeEach(() => {
  savedMode = process.env["RUNNER_MODE"];
  process.env["RUNNER_MODE"] = "aer";
  // Clear MONGODB_URI so resolveCheckpointDbName returns early and the suite
  // stays hermetic even when a URI is set in the developer's environment.
  savedMongoUri = process.env["MONGODB_URI"];
  delete process.env["MONGODB_URI"];
  resetHooks();
});

afterEach(() => {
  if (savedMode === undefined) delete process.env["RUNNER_MODE"];
  else process.env["RUNNER_MODE"] = savedMode;
  if (savedMongoUri === undefined) delete process.env["MONGODB_URI"];
  else process.env["MONGODB_URI"] = savedMongoUri;
  resetHooks();
});

/** Build an App with a no-op entrypoint and a TenantRuntime.registerAndRun stubbed. */
function makeApp(): { app: App; runSpy: ReturnType<typeof vi.fn> } {
  const app = new App({ appName: "test-agent" });
  app.entrypoint(() => ({}));
  const runSpy = vi.fn();
  // Stub `registerAndRun` so `app.run()` doesn't actually bind a port.
  const runtime = (app as unknown as { runtime: TenantRuntime }).runtime;
  (runtime as unknown as { registerAndRun: unknown }).registerAndRun = runSpy;
  return { app, runSpy };
}

describe("App.run() registers framework hooks", () => {
  beforeEach(() => {
    resetHooks();
  });

  it("registers the suspend handler", async () => {
    const { app } = makeApp();
    await app.run();
    expect(getSuspendHandler()).not.toBeNull();
  });

  it("routes the suspend handler through LangGraph inside a durable attempt", async () => {
    const { app } = makeApp();
    await app.run();
    const handler = getSuspendHandler();
    if (handler === null) throw new Error("suspend handler was not registered");
    // The durable refusal is gone: the handler always delegates to LangGraph,
    // so a guardrail review can pause after a durable LLM activity. Outside a
    // running graph that surfaces as LangGraph's own error, not an
    // adapter-level rejection.
    expect(() =>
      runWithAttemptContext(
        create(AttemptContextSchema, { attemptId: "attempt" }),
        () => handler({ suspend_reason: "review", suspend_context: {} }),
      ),
    ).toThrow("outside the context of a graph");
  });

  it("registers the LLM adapter factory", async () => {
    const { app } = makeApp();
    await app.run();
    // `getLLMAdapterFactory` throws when nothing is registered; a clean
    // call means a factory exists.
    expect(() => getLLMAdapterFactory()).not.toThrow();
  });

  it("registers the instrumentor", async () => {
    const { app } = makeApp();
    await app.run();
    expect(getInstrumentor()).not.toBeNull();
  });

  it("registered instrumentor is a callable that runInstrumentor invokes", async () => {
    // The hook is registered as a callback `(() => { ... }) as never`, NOT an
    // IIFE — so runInstrumentor() must be able to invoke it and actually run
    // `manuallyInstrument(...)`. Guards against regressing to an immediately
    // invoked form (which would store `undefined` and silently no-op).
    const manuallyInstrumentSpy = vi
      .spyOn(LangChainInstrumentation.prototype, "manuallyInstrument")
      .mockImplementation(() => undefined);
    try {
      const { app } = makeApp();
      await app.run();
      // app.run() already runs the instrumentor once via tracing setup; clear
      // so we can assert the explicit runInstrumentor() invocation in isolation.
      manuallyInstrumentSpy.mockClear();

      expect(() => runInstrumentor()).not.toThrow();
      expect(manuallyInstrumentSpy).toHaveBeenCalledTimes(1);
    } finally {
      manuallyInstrumentSpy.mockRestore();
    }
  });

  it("all hooks are registered BEFORE registerAndRun fires", async () => {
    const { app, runSpy } = makeApp();
    const stateAtServerStart: {
      suspend: unknown;
      adapter: unknown;
      instrumentor: unknown;
    } = { suspend: null, adapter: null, instrumentor: null };

    runSpy.mockImplementation(() => {
      stateAtServerStart.suspend = getSuspendHandler();
      // `getLLMAdapterFactory` throws if nothing registered — capture via try/catch.
      try {
        stateAtServerStart.adapter = getLLMAdapterFactory();
      } catch {
        stateAtServerStart.adapter = null;
      }
      stateAtServerStart.instrumentor = getInstrumentor();
    });

    await app.run();

    expect(stateAtServerStart.suspend).not.toBeNull();
    expect(stateAtServerStart.adapter).not.toBeNull();
    expect(stateAtServerStart.instrumentor).not.toBeNull();
  });

  it("rejects if no @app.entrypoint was registered", async () => {
    const app = new App({ appName: "test-agent" });
    // No entrypoint registration
    await expect(app.run()).rejects.toThrow(
      /No app\.entrypoint\(\) registered/,
    );
  });
});

// ---------------------------------------------------------------------------
// Query plugin registration (mirrors Python's test_runtime.py query-plugin
// lifecycle tests)
// ---------------------------------------------------------------------------

/** Saver/client stand-ins satisfying what LangGraphQueryPlugin's ctor reads. */
const fakeSaver = {
  checkpointCollectionName: "checkpoints",
  checkpointWritesCollectionName: "checkpoint_writes",
};
const fakeClient = { db: () => ({ collection: () => ({}) }) };

function stubCheckpointer(
  app: App,
  impl: () => unknown,
  mongoClient: unknown = fakeClient,
): void {
  const target = app as unknown as {
    checkpointer: () => unknown;
    mongoClient: unknown;
  };
  target.checkpointer = impl;
  target.mongoClient = mongoClient;
}

describe("App.run() registers the query plugin", () => {
  it("registers a LangGraphQueryPlugin in AER mode when the checkpointer is available", async () => {
    const { app } = makeApp();
    stubCheckpointer(
      app,
      () => new PlatformCheckpointer({ native: fakeSaver as never }),
    );
    await app.run();
    expect(getQueryPlugin()).toBeInstanceOf(LangGraphQueryPlugin);
  });

  it("registers nothing when the checkpointer returns null (no MongoDB URI)", async () => {
    const { app } = makeApp();
    stubCheckpointer(app, () => null);
    await app.run();
    expect(getQueryPlugin()).toBeNull();
  });

  it("degrades to no plugin when checkpointer construction throws", async () => {
    const { app } = makeApp();
    stubCheckpointer(app, () => {
      throw new Error("bad URI");
    });
    await expect(app.run()).resolves.toBeUndefined();
    expect(getQueryPlugin()).toBeNull();
  });

  it("skips registration outside AER mode", async () => {
    process.env["RUNNER_MODE"] = "tool";
    const { app } = makeApp();
    const checkpointerSpy = vi.fn(() => fakeSaver);
    stubCheckpointer(app, checkpointerSpy);
    await app.run();
    expect(checkpointerSpy).not.toHaveBeenCalled();
    expect(getQueryPlugin()).toBeNull();
  });
});
