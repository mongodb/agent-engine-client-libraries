/**
 * Composed lifecycle coverage for an opted-in build, driven through the real
 * AER /execute route with a real App and its registry mutex.
 *
 * This file exists because the focused unit tests next to it supply each
 * connection themselves: they stub the runtime's buildGraph, call the build
 * helper directly, end drain work by hand, and no-op the exit. That leaves the
 * production wiring between the invocation deadline, the route's drain
 * cleanup, and the fatal path unobserved. Everything here goes through
 * `fastify.inject` on the registered routes so those connections are exercised:
 *
 *   - removing the AER's deadline abort leaves ctx.signal live, so the expired
 *     queued invocation enters the builder and the `entered` assertion fails
 *   - ending drain work immediately instead of deferring it fails the
 *     `endWork` assertion while the build is live
 *   - dropping the fatal call from exitForUnsettledBuild fails the exit-code
 *     and termination-message assertions
 *
 * The modules are re-imported per test because the execution timeout and the
 * abandonment grace are read when the AER module loads.
 */

import { spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

import { AIMessage } from "@langchain/core/messages";
import {
  END,
  MessagesAnnotation,
  START,
  StateGraph,
} from "@langchain/langgraph";
import Fastify from "fastify";
import { afterEach, describe, expect, test, vi } from "vitest";

const EXECUTION_TIMEOUT_SECONDS = 0.3;

const NEVER_SETTLES_FIXTURE = fileURLToPath(
  new URL("./fixtures/per_invocation_never_settles.mjs", import.meta.url),
);

/**
 * A compiled graph the adapter can actually execute, so a served invocation is
 * observed end to end rather than inferred from the builder having run.
 */
function compiledGraph(label: string) {
  return new StateGraph(MessagesAnnotation)
    .addNode("respond", async () => ({
      messages: [new AIMessage({ content: `reply:${label}` })],
    }))
    .addEdge(START, "respond")
    .addEdge("respond", END)
    .compile();
}

interface ServerHarness {
  runtime: Record<string, unknown>;
  drainRegistry: {
    beginWork(executionId: string): void;
    endWork(executionId: string): void;
    apply(request: unknown): unknown;
  };
  abandonedBuilds: Map<string, { build: Promise<unknown> }>;
  handleExecute: (request: unknown) => Promise<unknown>;
  registerRoutes(app: Fastify.FastifyInstance): void;
  chunkSeq: Map<string, number>;
  ownerCallbackUrl: Map<string, string>;
  sendStreamChunk: unknown;
  reportCallback: unknown;
}

async function loadModules(
  graceSeconds: number,
  executionTimeoutSeconds: number = EXECUTION_TIMEOUT_SECONDS,
) {
  vi.resetModules();
  vi.stubEnv("RUNNER_MODE", "aer");
  vi.stubEnv("RUNNER_EXECUTION_TIMEOUT", String(executionTimeoutSeconds));
  vi.stubEnv("RUNNER_ABANDONED_BUILD_GRACE", String(graceSeconds));
  vi.stubEnv("OE_URL", "http://oe.test:8000");
  delete process.env["APP_ID"];

  const shared = await import("@mongodb-js/agent-engine-runner-shared");
  const { App } = await import("../src/runtime.js");
  return { shared, App };
}

function makeServer(
  shared: { AERServer: unknown },
  app: unknown,
): ServerHarness {
  const server = Object.create(
    (shared.AERServer as { prototype: object }).prototype,
  ) as ServerHarness;
  server.runtime = {
    appName: "test-agent",
    appVersion: "1.0.0",
    memoryEnabled: false,
    graphBuilder: app,
    tools: {},
    toolDefinitions: {},
    getAgentConfig: () => ({ featureEnabled: () => false, mcp: undefined }),
    getMongodbUri: () => null,
    getCurrentUserId: () => null,
    getAgent: (opts: unknown) =>
      (app as { getAgent(o: unknown): unknown }).getAgent(opts),
    buildGraph: (input: unknown, ctx: unknown) =>
      (
        app as { buildGraph(i: unknown, c: unknown): Promise<unknown> }
      ).buildGraph(input, ctx),
    usesPerInvocationBuild: () =>
      (app as { usesPerInvocationBuild(): boolean }).usesPerInvocationBuild(),
  };
  server.chunkSeq = new Map();
  server.ownerCallbackUrl = new Map();
  server.abandonedBuilds = new Map();
  server.sendStreamChunk = vi.fn().mockResolvedValue(undefined);
  server.reportCallback = vi.fn().mockResolvedValue(undefined);
  // `handleExecute` is an instance field (arrow function), so Object.create
  // does not install it. The route calls it; point it at the real prototype
  // method the field wraps.
  server.handleExecute = (request: unknown) =>
    (
      server as unknown as {
        doHandleExecute(r: unknown): Promise<unknown>;
      }
    ).doHandleExecute(request);
  return server;
}

/** A ToolServer over a real App, so its registry lock and discovery are real. */
function makeToolServer(
  shared: { ToolServer: unknown },
  app: unknown,
): { resolveLlmForRequest(llmId: string): Promise<unknown> } {
  const runtime = {
    appName: "test-agent",
    appVersion: "1.0.0",
    memoryEnabled: false,
    graphBuilder: app,
    tools: {},
    toolDefinitions: {},
    getAgentConfig: () => ({ featureEnabled: () => false, mcp: undefined }),
    getMongodbUri: () => null,
    getCurrentUserId: () => null,
    getAgent: (opts: unknown) =>
      (app as { getAgent(o: unknown): unknown }).getAgent(opts),
    buildGraph: (input: unknown, ctx: unknown) =>
      (
        app as { buildGraph(i: unknown, c: unknown): Promise<unknown> }
      ).buildGraph(input, ctx),
    usesPerInvocationBuild: () =>
      (app as { usesPerInvocationBuild(): boolean }).usesPerInvocationBuild(),
  };
  return new (shared.ToolServer as new (r: unknown) => {
    resolveLlmForRequest(llmId: string): Promise<unknown>;
  })(runtime);
}

async function waitFor(
  predicate: () => boolean,
  timeoutMs = 5000,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error("waitFor timed out");
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
  vi.resetModules();
});

describe("opted-in build lifecycle through the AER route", () => {
  test("bounds an expired build, refuses an expired queued one, and recovers after settlement", async () => {
    const { shared, App } = await loadModules(30);
    const app = new App({ appName: "test-agent" });

    const entered: string[] = [];
    const built: string[] = [];
    let releaseA: () => void = () => {};
    const gateA = new Promise<void>((resolve) => {
      releaseA = resolve;
    });

    app.entrypoint(
      async (_input: unknown, ctx: unknown) => {
        const id = (ctx as { executionId: string }).executionId;
        entered.push(id);
        if (id === "exec-a") await gateA;
        built.push(id);
        return compiledGraph(id);
      },
      { loadStrategy: "per-execution" },
    );

    const server = makeServer(shared, app);
    const fastify = Fastify();
    server.registerRoutes(fastify);
    await fastify.ready();

    const endWork = vi.spyOn(server.drainRegistry, "endWork");
    const post = (id: string) =>
      fastify.inject({
        method: "POST",
        url: "/execute",
        payload: {
          execution_id: id,
          message: "hi",
          platform_api_url: "http://oe.test:8000",
        },
      });

    // A holds the build lock and runs past its invocation deadline.
    const a = post("exec-a");
    await waitFor(() => entered.includes("exec-a"));

    // B queues behind A and also reaches its own deadline while waiting.
    const b = post("exec-b");
    await waitFor(() => server.abandonedBuilds.has("exec-a"));

    // A is live, so its execution must stay registered as drain work.
    expect(server.abandonedBuilds.has("exec-a")).toBe(true);
    expect(endWork).not.toHaveBeenCalledWith("exec-a");

    // Both callers get a bounded error rather than waiting on the build, and
    // both are reported as the deadline breach they are rather than as generic
    // 500s.
    const [aResponse, bResponse] = await Promise.all([a, b]);
    expect(aResponse.statusCode).toBe(504);
    expect(bResponse.statusCode).toBe(504);
    // B never entered the builder: its deadline had passed when the lock freed.
    expect(entered).toEqual(["exec-a"]);

    // Releasing A settles it, which releases both the lock and its drain work.
    releaseA();
    await waitFor(() => !server.abandonedBuilds.has("exec-a"));
    expect(endWork).toHaveBeenCalledWith("exec-a");

    // C arrives with a live deadline, so it builds and is actually served: the
    // graph materializes and executes, which builder entry alone would not show.
    const c = await post("exec-c");
    expect(c.statusCode).toBe(200);
    expect(built).toContain("exec-c");
    expect(entered).toEqual(["exec-a", "exec-c"]);

    await fastify.close();
  }, 20000);
});

describe("a builder that honors the invocation deadline", () => {
  test("reports the deadline as a timeout rather than a generic failure", async () => {
    const { shared, App } = await loadModules(30);
    const app = new App({ appName: "test-agent" });
    app.entrypoint(
      async (_input: unknown, ctx: unknown) => {
        // Abort-aware author I/O: it rejects on the signal rather than running
        // past the deadline, so the failure arrives as an ordinary error and
        // only the controller says it was the deadline.
        await new Promise<never>((_resolve, reject) => {
          (ctx as { signal: AbortSignal }).signal.addEventListener(
            "abort",
            () => reject(new Error("aborted")),
          );
        });
        return compiledGraph("unreachable");
      },
      { loadStrategy: "per-execution" },
    );

    const server = makeServer(shared, app);
    const fastify = Fastify();
    server.registerRoutes(fastify);
    await fastify.ready();

    const response = await fastify.inject({
      method: "POST",
      url: "/execute",
      payload: {
        execution_id: "exec-cooperative",
        message: "hi",
        platform_api_url: "http://oe.test:8000",
      },
    });

    expect(response.statusCode).toBe(504);
    const sendChunk = (server as unknown as { sendStreamChunk: unknown })
      .sendStreamChunk as { mock: { calls: unknown[][] } };
    const errorChunks = sendChunk.mock.calls.filter(
      (args) => args[2] === "error",
    );
    expect(errorChunks).toHaveLength(1);
    expect(errorChunks[0]?.[5]).toEqual({ error_code: "timeout" });

    await fastify.close();
  }, 20000);
});

describe("a queued build that is drained before its deadline", () => {
  test("acknowledges the cancellation instead of reporting a timeout", async () => {
    const { shared, App } = await loadModules(30);
    const app = new App({ appName: "test-agent" });
    const entered: string[] = [];
    let releaseA: () => void = () => {};
    const gateA = new Promise<void>((resolve) => {
      releaseA = resolve;
    });

    app.entrypoint(
      async (_input: unknown, ctx: unknown) => {
        const id = (ctx as { executionId: string }).executionId;
        entered.push(id);
        if (id === "exec-a") await gateA;
        return compiledGraph(id);
      },
      { loadStrategy: "per-execution" },
    );

    const server = makeServer(shared, app);
    const fastify = Fastify();
    server.registerRoutes(fastify);
    await fastify.ready();

    const post = (id: string) =>
      fastify.inject({
        method: "POST",
        url: "/execute",
        payload: {
          execution_id: id,
          message: "hi",
          platform_api_url: "http://oe.test:8000",
        },
      });

    const a = post("exec-a");
    await waitFor(() => entered.includes("exec-a"));

    // B queues behind A and is then drained, well before its deadline.
    const b = post("exec-b");
    await new Promise((resolve) => setTimeout(resolve, 10));
    server.drainRegistry.apply({
      execution_id: "exec-b",
      request_id: "req-drain",
      reason: "test",
      deadline_at_ms: Date.now() + 5000,
    } as never);

    const bResponse = await b;
    // A drained invocation is a cancellation, not a deadline breach: no
    // timeout-coded chunk, no 504.
    expect(bResponse.statusCode).toBe(200);
    expect(JSON.parse(bResponse.body)).toEqual({ status: "cancelled" });
    expect(entered).toEqual(["exec-a"]);
    const sendChunk = (server as unknown as { sendStreamChunk: unknown })
      .sendStreamChunk as { mock: { calls: unknown[][] } };
    expect(
      sendChunk.mock.calls.filter((args) => args[2] === "error"),
    ).toHaveLength(0);

    releaseA();
    await a;
    await fastify.close();
  }, 20000);
});

describe("a running build whose invocation is stopped", () => {
  test("acknowledges the cancellation without starting author preparation", async () => {
    // Roomier deadline so the drain and the release both land before it.
    const { shared, App } = await loadModules(30, 3);
    const app = new App({ appName: "test-agent" });

    const prepared: number[] = [];
    app.prepareAgentInput(((input: unknown) => {
      prepared.push(1);
      return input;
    }) as never);

    const entered: number[] = [];
    let release: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    app.entrypoint(
      async () => {
        entered.push(1);
        await gate;
        return compiledGraph("stopped");
      },
      { loadStrategy: "per-execution" },
    );

    const server = makeServer(shared, app);
    const fastify = Fastify();
    server.registerRoutes(fastify);
    await fastify.ready();

    const pending = fastify.inject({
      method: "POST",
      url: "/execute",
      payload: {
        execution_id: "exec-stopped",
        message: "hi",
        platform_api_url: "http://oe.test:8000",
      },
    });
    await waitFor(() => entered.length === 1);

    // Stop while the builder is held, then let it settle normally.
    server.drainRegistry.apply({
      execution_id: "exec-stopped",
      request_id: "req-stop",
      reason: "test",
      deadline_at_ms: Date.now() + 5000,
    } as never);
    release();

    const response = await pending;
    expect(response.statusCode).toBe(200);
    expect(JSON.parse(response.body)).toEqual({ status: "cancelled" });
    // The build settled normally, so without the post-settlement recheck the
    // author's input-preparation hook would have run after the Stop.
    expect(prepared).toEqual([]);
    const sendChunk = (server as unknown as { sendStreamChunk: unknown })
      .sendStreamChunk as { mock: { calls: unknown[][] } };
    expect(
      sendChunk.mock.calls.filter((args) => args[2] === "error"),
    ).toHaveLength(0);

    await fastify.close();
  }, 20000);
});

describe("concurrent model resolution on one Tool Pod", () => {
  test("both requests capture the model their own discovery retained", async () => {
    const { shared, App } = await loadModules(30);
    const app = new App({ appName: "test-agent" });
    app.entrypoint(
      async () => {
        app.llm({ invoke: () => undefined } as never, "primary");
        await new Promise((resolve) => setTimeout(resolve, 5));
        throw new Error("needs invocation context");
      },
      { loadStrategy: "per-execution" },
    );

    const server = makeToolServer(shared, app);
    const seenAtLookup: boolean[] = [];
    (
      server as unknown as {
        createLlmForPod: (llmId: string) => unknown;
      }
    ).createLlmForPod = (llmId: string) => {
      seenAtLookup.push(
        (shared.snapshotLlmRegistry() as Map<string, unknown>).has(llmId),
      );
      return { llmId };
    };

    await Promise.all([
      server.resolveLlmForRequest("primary"),
      server.resolveLlmForRequest("primary"),
    ]);

    // Discovery fails by design for these agents, so it never latches and every
    // request re-runs it, clearing the shared registry before repopulating it.
    // A lookup outside the discovery critical section would see it empty and
    // the request would lose the registration its own discovery retained.
    expect(seenAtLookup).toEqual([true, true]);
  });
});

describe("fatal withdrawal through the AER route", () => {
  test("exits 1 with a termination message naming the execution when a build never settles", () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ap7601-fatal-"));
    const terminationLog = path.join(tmpDir, "termination-log");

    // Out of process, so the real exit code and the real termination message
    // are observed with nothing replaced. Mocking process.exit in-process made
    // fatalWithTermination's own guard throw, which left an unhandled rejection
    // rather than evidence.
    const result = spawnSync(process.execPath, [NEVER_SETTLES_FIXTURE], {
      env: {
        ...process.env,
        RUNNER_MODE: "aer",
        RUNNER_EXECUTION_TIMEOUT: "0.2",
        RUNNER_ABANDONED_BUILD_GRACE: "0.05",
        OE_URL: "http://oe.test:8000",
        TERMINATION_LOG_PATH: terminationLog,
      },
      encoding: "utf-8",
      timeout: 30000,
    });

    expect(result.status).toBe(1);
    expect(fs.readFileSync(terminationLog, "utf8")).toContain("exec-never");

    fs.rmSync(tmpDir, { recursive: true, force: true });
  }, 40000);
});
