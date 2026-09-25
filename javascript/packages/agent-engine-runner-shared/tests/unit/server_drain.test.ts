/**
 * Execution-scoped drain receiver (POST /drain), TypeScript side.
 *
 * Mirrors Python's tests/unit/test_drain.py — the two runner runtimes are
 * parallel implementations, so the twins must stay behaviourally identical.
 * The cross-language request/response vectors live in
 * `client-libraries/test-fixtures/drain/contract.json` and are pinned by
 * server_drain_contract.test.ts.
 *
 * TS-vs-Python divergence (mirrors tool_cancellation_boundary.test.ts):
 * promises cannot be cancelled, so "notify active work" aborts an
 * AbortController, which reaches only work with a signal channel (the AER
 * execution context, LLM streams). Plain tool functions are tracked and
 * admission-blocked; outliving the deadline is an honest timed_out.
 */

import { describe, expect, test, vi } from "vitest";

import {
  DRAIN_ABORT_REASON,
  DrainRegistry,
  type DrainRequest,
} from "../../src/server/drain.js";
import { ToolServer } from "../../src/server/tool.js";
import { AERServer } from "../../src/server/aer.js";
import { RuntimeAgentConfig } from "../../src/agent_config.js";
import type { ITenantRuntime } from "../../src/server/base.js";

function makeRequest(overrides: Partial<DrainRequest> = {}): DrainRequest {
  return {
    request_id: "drain-aaaa",
    execution_id: "exec-1",
    reason: "execution_cancelled",
    deadline_at_ms: Date.now() + 5000,
    ...overrides,
  };
}

/** Re-send the drain request until a final outcome is recorded — the same
 * polling the caller performs. */
async function settled(registry: DrainRegistry, request: DrainRequest) {
  for (let i = 0; i < 200; i++) {
    const verdict = registry.apply(request);
    if (verdict.body.outcome !== "accepted") return verdict;
    await new Promise((r) => setTimeout(r, 10));
  }
  throw new Error("drain did not finalize");
}

/** Poll until work for the execution is registered — deterministic, unlike a
 * fixed sleep that a slow dispatch can outlast. */
async function waitForActiveWork(
  registry: DrainRegistry,
  executionId: string,
): Promise<void> {
  const entries = (
    registry as unknown as {
      entries: Map<string, { activeCount: number }>;
    }
  ).entries;
  for (let i = 0; i < 500; i++) {
    if ((entries.get(executionId)?.activeCount ?? 0) > 0) return;
    await new Promise((r) => setTimeout(r, 5));
  }
  throw new Error("work was never registered");
}

/** Poll until every registered unit of work for the execution has ended. */
async function waitForNoActiveWork(
  registry: DrainRegistry,
  executionId: string,
): Promise<void> {
  const entries = (
    registry as unknown as {
      entries: Map<string, { activeCount: number }>;
    }
  ).entries;
  for (let i = 0; i < 500; i++) {
    if ((entries.get(executionId)?.activeCount ?? 0) === 0) return;
    await new Promise((r) => setTimeout(r, 5));
  }
  throw new Error("work never settled");
}

describe("DrainRegistry", () => {
  test("unknown execution is delivery_failed", () => {
    const verdict = new DrainRegistry().apply(makeRequest());
    expect(verdict.status).toBe(200);
    expect(verdict.body).toEqual({
      outcome: "delivery_failed",
      reason_code: "execution_not_found",
    });
  });

  test("a restarted runtime loses all drain state", () => {
    // Registry loss is not completion evidence: the restarted runtime's work
    // is gone, but only the caller's durable record can reconcile the drain.
    const first = new DrainRegistry();
    first.beginWork("exec-1");
    expect(first.apply(makeRequest()).status).toBe(202);

    const restarted = new DrainRegistry();
    expect(restarted.apply(makeRequest()).body.outcome).toBe("delivery_failed");
  });

  test("known ended execution completes", () => {
    const registry = new DrainRegistry();
    registry.beginWork("exec-1");
    registry.endWork("exec-1");

    const verdict = registry.apply(makeRequest());
    expect(verdict.status).toBe(200);
    expect(verdict.body).toEqual({ outcome: "completed" });
  });

  test("active execution accepts, then completes when the work ends", async () => {
    const registry = new DrainRegistry();
    registry.beginWork("exec-1");

    expect(registry.apply(makeRequest()).status).toBe(202);
    registry.endWork("exec-1");

    const verdict = await settled(registry, makeRequest());
    expect(verdict.status).toBe(200);
    expect(verdict.body).toEqual({ outcome: "completed" });
  });

  test("drain aborts a controller blocked on an await", async () => {
    // The AER case: a turn parked on an outbound interaction (an OE callback
    // that will never answer, an LLM stream) observes the abort through the
    // execution-context signal — the OE admission gate alone cannot unblock it.
    const registry = new DrainRegistry();
    const controller = new AbortController();
    const blocked = new Promise((_, reject) => {
      controller.signal.addEventListener("abort", () =>
        reject(new DOMException("This operation was aborted", "AbortError")),
      );
    });
    registry.beginWork("exec-1", controller);

    expect(registry.apply(makeRequest()).status).toBe(202);
    await expect(blocked).rejects.toMatchObject({ name: "AbortError" });

    registry.endWork("exec-1", controller);
    expect((await settled(registry, makeRequest())).body.outcome).toBe(
      "completed",
    );
  });

  test("work with no signal channel times out and stays timed out", async () => {
    const registry = new DrainRegistry();
    registry.beginWork("exec-1"); // no controller: nothing to signal

    expect(
      registry.apply(makeRequest({ deadline_at_ms: Date.now() + 50 })).status,
    ).toBe(202);

    const verdict = await settled(registry, makeRequest());
    expect(verdict.body).toEqual({
      outcome: "timed_out",
      reason_code: "deadline_exceeded",
    });

    registry.endWork("exec-1"); // the work finishes after the deadline
    expect(registry.apply(makeRequest()).body.outcome).toBe("timed_out");
  });

  test("duplicate request_id never re-runs side effects", () => {
    const registry = new DrainRegistry();
    const controller = new AbortController();
    const abort = vi.spyOn(controller, "abort");
    registry.beginWork("exec-1", controller);

    expect(registry.apply(makeRequest()).status).toBe(202);
    expect(registry.apply(makeRequest()).status).toBe(202);
    expect(abort).toHaveBeenCalledTimes(1);
  });

  test("different request_ids coalesce into one drain", async () => {
    const registry = new DrainRegistry();
    const controller = new AbortController();
    const abort = vi.spyOn(controller, "abort");
    registry.beginWork("exec-1", controller);

    expect(
      registry.apply(makeRequest({ request_id: "drain-aaaa" })).status,
    ).toBe(202);
    expect(
      registry.apply(makeRequest({ request_id: "drain-bbbb" })).status,
    ).toBe(202);
    expect(abort).toHaveBeenCalledTimes(1);

    registry.endWork("exec-1", controller);
    for (const requestId of ["drain-aaaa", "drain-bbbb"]) {
      expect(
        (await settled(registry, makeRequest({ request_id: requestId }))).body
          .outcome,
      ).toBe("completed");
    }
  });

  test("a retry never extends the absolute deadline", async () => {
    const registry = new DrainRegistry();
    registry.beginWork("exec-1");

    expect(
      registry.apply(makeRequest({ deadline_at_ms: Date.now() + 50 })).status,
    ).toBe(202);
    // A retry (new request_id) carrying a far-later deadline must coalesce
    // without moving the original deadline.
    expect(
      registry.apply(
        makeRequest({
          request_id: "drain-bbbb",
          deadline_at_ms: Date.now() + 60_000,
        }),
      ).status,
    ).toBe(202);

    expect((await settled(registry, makeRequest())).body.outcome).toBe(
      "timed_out",
    );
  });

  test("a sibling execution is unaffected", () => {
    const registry = new DrainRegistry();
    const sibling = new AbortController();
    const drained = new AbortController();
    registry.beginWork("exec-A", sibling);
    registry.beginWork("exec-B", drained);

    expect(registry.apply(makeRequest({ execution_id: "exec-B" })).status).toBe(
      202,
    );

    expect(sibling.signal.aborted).toBe(false);
    expect(drained.signal.aborted).toBe(true);
    // exec-A still admits work; exec-B does not.
    registry.beginWork("exec-A");
    registry.endWork("exec-A");
    expect(() => registry.beginWork("exec-B")).toThrowError(
      expect.objectContaining({ statusCode: 409 }),
    );
  });

  test("attachController aborts immediately when the drain landed first", () => {
    const registry = new DrainRegistry();
    registry.beginWork("exec-1"); // admitted, controller not yet attached
    expect(registry.apply(makeRequest()).status).toBe(202);

    const late = new AbortController();
    registry.attachController("exec-1", late);
    expect(late.signal.aborted).toBe(true);
    // The AER tells a drain apart from its execution timeout by this reason —
    // a bare abort() would surface as a phantom timeout.
    expect(late.signal.reason).toBe(DRAIN_ABORT_REASON);
  });

  test("rejects malformed or non-positive drain configuration", () => {
    // Number("bogus") is NaN: as a TTL it evicts immediately, as a deadline
    // ceiling it disables the check. Fail fast instead. Mirrors Python's
    // _positive.
    expect(
      () => new DrainRegistry({ env: { RUNNER_DRAIN_RECORD_TTL_S: "bogus" } }),
    ).toThrow(/RUNNER_DRAIN_RECORD_TTL_S/);
    expect(
      () => new DrainRegistry({ env: { RUNNER_DRAIN_RECORD_TTL_S: "-5" } }),
    ).toThrow(/RUNNER_DRAIN_RECORD_TTL_S/);
    expect(
      () => new DrainRegistry({ env: { RUNNER_DRAIN_MAX_DEADLINE_MS: "0" } }),
    ).toThrow(/RUNNER_DRAIN_MAX_DEADLINE_MS/);
    // Unset env keeps the defaults.
    expect(new DrainRegistry({ env: {} }).maxDeadlineMs).toBe(60_000);
  });

  test("unknown execution drain latches admission and stays stable", () => {
    // The drain-before-dispatch race: a dispatch that slips past the OE gate
    // and lands after the drain must not start, and a retry of the same
    // request must see the same (not-found) answer for the retention window.
    const registry = new DrainRegistry();
    expect(registry.apply(makeRequest()).body.outcome).toBe("delivery_failed");

    expect(() => registry.beginWork("exec-1")).toThrowError(
      expect.objectContaining({ statusCode: 409 }),
    );
    expect(registry.apply(makeRequest()).body.outcome).toBe("delivery_failed");
  });

  test("tombstone drain latches admission", () => {
    // A drain finalizing on an ended execution still latches: a resume
    // reusing the execution id must not run while retries report completed.
    const registry = new DrainRegistry();
    registry.beginWork("exec-1");
    registry.endWork("exec-1");
    expect(registry.apply(makeRequest()).body.outcome).toBe("completed");

    expect(() => registry.beginWork("exec-1")).toThrowError(
      expect.objectContaining({ statusCode: 409 }),
    );
  });

  test("tombstone drain record lives a full TTL from finalization", async () => {
    // A drain landing near tombstone expiry restarts retention: the record
    // must survive the full TTL from finalization, not the tombstone's
    // original eviction time.
    const registry = new DrainRegistry({ recordTtlMs: 400 });
    registry.beginWork("exec-1");
    registry.endWork("exec-1"); // tombstone eviction scheduled at t+400
    await new Promise((r) => setTimeout(r, 300));

    expect(registry.apply(makeRequest()).body.outcome).toBe("completed");
    await new Promise((r) => setTimeout(r, 200)); // t=500: past original eviction
    expect(registry.apply(makeRequest()).body.outcome).toBe("completed");
    await new Promise((r) => setTimeout(r, 300)); // t=800: past restarted TTL
    expect(registry.apply(makeRequest()).body.outcome).toBe("delivery_failed");
  });

  test("endWork drops tracked controllers when no work remains", () => {
    // The AER attaches its controller mid-handler but ends work without it; a
    // tombstone must not retain controllers and their listeners for its TTL.
    const registry = new DrainRegistry();
    const controller = new AbortController();
    registry.beginWork("exec-1", controller);
    registry.endWork("exec-1");

    const entries = (
      registry as unknown as {
        entries: Map<string, { controllers: Set<AbortController> }>;
      }
    ).entries;
    expect(entries.get("exec-1")?.controllers.size).toBe(0);
  });

  test("the record outlives completion for the TTL, then expires", async () => {
    const registry = new DrainRegistry({ recordTtlMs: 50 });
    registry.beginWork("exec-1");
    registry.endWork("exec-1");

    expect(registry.apply(makeRequest()).body.outcome).toBe("completed");
    await new Promise((r) => setTimeout(r, 100)); // past the TTL

    expect(registry.apply(makeRequest()).body.outcome).toBe("delivery_failed");
  });
});

// ---------------------------------------------------------------------------
// Wiring through the real servers
// ---------------------------------------------------------------------------

function fakeRuntime(tools: ITenantRuntime["tools"]): ITenantRuntime {
  return {
    appName: "test-app",
    orgId: null,
    graphBuilder: {},
    tools,
    toolDefinitions: Object.fromEntries(Object.keys(tools).map((k) => [k, {}])),
    // A real config: capability advertise reads language/framework/features
    // off it, the tool server calls featureEnabled on it.
    getAgentConfig: () => new RuntimeAgentConfig(),
    getMongodbUri: () => null,
    getAgent: () => {
      throw new Error("no agent in tool tests");
    },
  };
}

describe("ToolServer drain wiring", () => {
  test("a tool call outliving the deadline completes the work, not the drain", async () => {
    let release!: () => void;
    const gate = new Promise<string>((r) => {
      release = () => r("external work completed");
    });
    const server = new ToolServer(fakeRuntime({ tool: () => gate }));
    const app = server.createApp();
    await app.ready();

    const execute = app.inject({
      method: "POST",
      url: "/execute",
      payload: {
        execution_id: "exec-1",
        tool_name: "tool",
        arguments: {},
        session_id: "session-1",
      },
    });
    await waitForActiveWork(server.drainRegistry, "exec-1");

    const drained = await app.inject({
      method: "POST",
      url: "/drain",
      payload: makeRequest({ deadline_at_ms: Date.now() + 100 }),
    });
    expect(drained.statusCode).toBe(202);

    expect(
      (await settled(server.drainRegistry, makeRequest())).body.outcome,
    ).toBe("timed_out");

    // New work for the drained execution is refused at admission.
    const replay = await app.inject({
      method: "POST",
      url: "/execute",
      payload: {
        execution_id: "exec-1",
        tool_name: "tool",
        arguments: {},
        session_id: "session-1",
      },
    });
    expect(replay.statusCode).toBe(409);

    // The in-flight tool call itself was never interrupted.
    release();
    const finished = await execute;
    expect(finished.json().status).toBe("success");
    await app.close();
  });

  test("unscoped runtime accepts a drain that omits workspace_id", async () => {
    // No APP_ID (local development): there is nothing to check the scope
    // claim against, so a request without the key is accepted.
    const savedAppId = process.env["APP_ID"];
    delete process.env["APP_ID"];
    try {
      const server = new ToolServer(fakeRuntime({}));
      const app = server.createApp();
      await app.ready();
      const resp = await app.inject({
        method: "POST",
        url: "/drain",
        payload: makeRequest(), // no workspace_id key at all
      });
      expect(resp.statusCode).toBe(200);
      expect(resp.json().outcome).toBe("delivery_failed");

      // JSON null is an accepted spelling of omission (matches Python and
      // the published OpenAPI contract).
      const nullResp = await app.inject({
        method: "POST",
        url: "/drain",
        payload: { ...makeRequest(), workspace_id: null },
      });
      expect(nullResp.statusCode).toBe(200);
      expect(nullResp.json().outcome).toBe("delivery_failed");
      await app.close();
    } finally {
      if (savedAppId === undefined) delete process.env["APP_ID"];
      else process.env["APP_ID"] = savedAppId;
    }
  });

  test("a signal-ignoring LLM stream times the drain out instead of reporting completed", async () => {
    // The generator parks on a promise the abort cannot settle: force-close
    // destroys the socket promptly, but the drain registration must survive
    // until the generator truly unwinds — completed while it pends would be
    // false quiescence evidence.
    const server = new ToolServer(fakeRuntime({}));
    let releaseGenerator!: () => void;
    const parked = new Promise<void>((r) => {
      releaseGenerator = r;
    });
    (
      server as unknown as {
        loadRegistryThenStreamLlmChunks: () => AsyncGenerator<never>;
      }
    ).loadRegistryThenStreamLlmChunks = async function* () {
      await parked; // never observes the abort signal it was handed
      yield undefined as never;
    };
    const app = server.createApp();
    await app.ready();

    // Swallow the socket-destroy outcome; what matters is that it settles.
    const streamSettled = app
      .inject({
        method: "POST",
        url: "/invoke_llm/stream",
        payload: {
          execution_id: "exec-1",
          arguments: {
            model: "gpt-4o",
            messages: [{ role: "user", content: "hi" }],
          },
        },
      })
      .then(
        () => true,
        () => true,
      );
    await waitForActiveWork(server.drainRegistry, "exec-1");

    const drain = server.drainRegistry.apply(
      makeRequest({ deadline_at_ms: Date.now() + 300 }),
    );
    expect(drain.status).toBe(202);

    // Socket teardown is prompt even though the generator is parked.
    await streamSettled;

    // The drain waits out the deadline and reports timed_out — never
    // completed while the generator was parked.
    const final = await settled(server.drainRegistry, makeRequest());
    expect(final.body).toEqual({
      outcome: "timed_out",
      reason_code: "deadline_exceeded",
    });

    // When the generator eventually unwinds, the registration is released —
    // and the recorded timeout stands (write-once).
    releaseGenerator();
    await waitForNoActiveWork(server.drainRegistry, "exec-1");
    expect(server.drainRegistry.apply(makeRequest()).body.outcome).toBe(
      "timed_out",
    );
    await app.close();
  });
});

describe("AERServer drain wiring", () => {
  test("a turn blocked mid-execution is stopped by the drain", async () => {
    // A turn parked on an await — the shape of being blocked inside an OE
    // interaction — unwinds via the execution-context abort signal.
    const hangingAgent = {
      execute: () =>
        (async function* () {
          await new Promise(() => {});
          yield; // unreachable by construction
        })(),
    };
    const runtime = {
      ...fakeRuntime({}),
      orgId: "org-1",
      projectId: "proj-1",
      getAgent: () => hangingAgent,
    } as unknown as ITenantRuntime;
    const server = new AERServer(runtime);
    // Hermetic: stub the two OE-facing network surfaces the error path uses.
    (server as unknown as Record<string, unknown>)["sendStreamChunk"] = vi
      .fn()
      .mockResolvedValue(undefined);
    (server as unknown as Record<string, unknown>)["reportCallback"] = vi
      .fn()
      .mockResolvedValue(undefined);
    // Startup requires OE_URL/APP_ID/ORG_ID/PROJECT_ID for capability
    // registration; a stubbed fetch answers the advertise POST.
    process.env["OE_URL"] = "http://oe.test";
    process.env["APP_ID"] = "ws-1";
    process.env["ORG_ID"] = "org-1";
    process.env["PROJECT_ID"] = "proj-1";
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("{}", { status: 200 })),
    );
    let app;
    try {
      app = server.createApp();
      await app.ready();

      const execute = app.inject({
        method: "POST",
        url: "/execute",
        payload: {
          execution_id: "exec-1",
          platform_api_url: "http://127.0.0.1:9",
        },
      });
      // Deterministic: wait until the turn has registered its work rather
      // than racing a fixed sleep against a slow dispatch.
      await waitForActiveWork(server.drainRegistry, "exec-1");

      const drained = await app.inject({
        method: "POST",
        url: "/drain",
        // APP_ID is set: the scope claim is mandatory on a scoped runtime.
        payload: makeRequest({
          deadline_at_ms: Date.now() + 2000,
          workspace_id: "ws-1",
        }),
      });
      expect(drained.statusCode).toBe(202);

      const finished = await execute; // the abort unwinds the turn into an error
      expect(finished.statusCode).toBeGreaterThanOrEqual(500);

      expect(
        (await settled(server.drainRegistry, makeRequest())).body.outcome,
      ).toBe("completed");

      const replay = await app.inject({
        method: "POST",
        url: "/execute",
        payload: {
          execution_id: "exec-1",
          platform_api_url: "http://127.0.0.1:9",
        },
      });
      expect(replay.statusCode).toBe(409);
    } finally {
      vi.unstubAllGlobals();
      delete process.env["OE_URL"];
      delete process.env["APP_ID"];
      delete process.env["ORG_ID"];
      delete process.env["PROJECT_ID"];
      if (app) await app.close();
    }
  });

  test("a drain landing between admission and controller attach is not a timeout", async () => {
    // The AER registers work at admission but builds its execution-wide
    // AbortController inside the handler. A drain in that window fires at
    // attach time — with DRAIN_ABORT_REASON, so the catch path reports a
    // cancellation and never walks the timeout path (504 + timeout chunk).
    const hangingAgent = {
      execute: () =>
        (async function* () {
          await new Promise(() => {});
          yield; // unreachable by construction
        })(),
    };
    const runtime = {
      ...fakeRuntime({}),
      orgId: "org-1",
      projectId: "proj-1",
      getAgent: () => hangingAgent,
    } as unknown as ITenantRuntime;
    const server = new AERServer(runtime);
    const sendStreamChunk = vi.fn().mockResolvedValue(undefined);
    (server as unknown as Record<string, unknown>)["sendStreamChunk"] =
      sendStreamChunk;
    (server as unknown as Record<string, unknown>)["reportCallback"] = vi
      .fn()
      .mockResolvedValue(undefined);
    process.env["OE_URL"] = "http://oe.test";
    process.env["APP_ID"] = "ws-1";
    process.env["ORG_ID"] = "org-1";
    process.env["PROJECT_ID"] = "proj-1";
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("{}", { status: 200 })),
    );
    let app;
    try {
      // Interpose the drain deterministically: it lands after admission and
      // just before the handler attaches its controller.
      const registry = server.drainRegistry;
      const original = registry.attachController.bind(registry);
      registry.attachController = (executionId, controller) => {
        registry.apply(makeRequest());
        original(executionId, controller);
      };
      app = server.createApp();
      await app.ready();

      const finished = await app.inject({
        method: "POST",
        url: "/execute",
        payload: {
          execution_id: "exec-1",
          platform_api_url: "http://127.0.0.1:9",
        },
      });
      expect(finished.statusCode).not.toBe(504);
      expect(finished.body).toContain("drained");
      expect(
        sendStreamChunk.mock.calls.some((call) =>
          JSON.stringify(call).includes('"error_code":"timeout"'),
        ),
      ).toBe(false);
    } finally {
      vi.unstubAllGlobals();
      delete process.env["OE_URL"];
      delete process.env["APP_ID"];
      delete process.env["ORG_ID"];
      delete process.env["PROJECT_ID"];
      if (app) await app.close();
    }
  });
});
