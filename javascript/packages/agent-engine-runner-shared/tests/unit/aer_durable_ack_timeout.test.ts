/**
 * Negotiated whole-turn timeout through the real AER wiring.
 *
 * Lives in its own file because it re-imports the module graph with a short
 * `RUNNER_EXECUTION_TIMEOUT`, and `vi.resetModules()` must not disturb the
 * other AER suites.
 *
 * Twin obligation of Python's mid-turn failure tests: a stalled chunk POST
 * with a full waiting queue must not strand the handler past the execution
 * deadline. The timeout releases blocked production promptly, the turn settles
 * with the original timeout discriminator and a 504, and no fresh retry budget
 * is granted.
 */

import { afterEach, describe, expect, test, vi } from "vitest";

import { DRAIN_ABORT_REASON } from "../../src/server/drain.js";
import {
  DURABLE_ACK_INCOMPLETE_ERROR_CODE,
  RUN_TIMEOUT_ERROR_CODE,
} from "../../src/server/chunk_types.js";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.resetModules();
  vi.restoreAllMocks();
});

type ChunkBody = {
  chunk_type?: string;
  chunks?: Array<Record<string, unknown>>;
  metadata?: Record<string, unknown>;
};

describe("AER negotiated execution timeout", () => {
  test("releases blocked production and settles with the timeout error", async () => {
    vi.stubEnv("RUNNER_EXECUTION_TIMEOUT", "0.5");
    vi.resetModules();
    const fresh = await import("../../src/index.js");
    const FreshAERServer = fresh.AERServer;

    const bodies: ChunkBody[] = [];
    const fetchMock = vi.fn(async (_url: string | URL, init?: RequestInit) => {
      bodies.push(JSON.parse(String(init?.body)) as ChunkBody);
      if (bodies.length === 1) {
        // The first batch POST stalls until the execution aborts it.
        return new Promise<Response>((_resolve, reject) => {
          const abort = (): void => reject(new Error("request aborted"));
          if (init?.signal?.aborted === true) {
            abort();
            return;
          }
          init?.signal?.addEventListener("abort", abort);
        });
      }
      return new Response(null, { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const server = Object.create(FreshAERServer.prototype) as {
      runtime: Record<string, unknown>;
      chunkSeq: Map<string, number>;
      durableAckSenders: Map<string, unknown>;
      ownerCallbackUrl: Map<string, string>;
      reportCallback: ReturnType<typeof vi.fn>;
      executeViaAgentStream: (...args: unknown[]) => Promise<unknown>;
      doHandleExecute: (request: unknown) => Promise<{ status: string }>;
    };
    server.runtime = {
      getAgent: vi.fn(() => ({})),
      memoryWriter: null,
      orgId: "org-1",
      projectId: "proj-1",
    };
    server.chunkSeq = new Map();
    server.durableAckSenders = new Map();
    server.ownerCallbackUrl = new Map();
    server.reportCallback = vi.fn().mockResolvedValue(undefined);

    let blockedResolve: (() => void) | undefined;
    const blocked = new Promise<void>((resolve) => {
      blockedResolve = resolve;
    });
    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      const send = (
        server as unknown as {
          sendStreamChunk: (
            oe: string,
            exec: string,
            type: string,
            content: string,
          ) => Promise<void>;
        }
      ).sendStreamChunk.bind(server);
      await send(oeUrl, executionId, "text", "c1");
      // Fill the 96-slot queue behind the stalled POST; the next send blocks.
      for (let seq = 2; seq <= 97; seq++) {
        await send(oeUrl, executionId, "text", `c${seq}`);
      }
      blockedResolve?.();
      await send(oeUrl, executionId, "text", "blocked");
      return { content: "done", messages: [] };
    };

    const started = performance.now();
    const request = {
      execution_id: "exec-timeout",
      message: "hi",
      platform_api_url: "http://oe",
      user_id: "u-1",
      custom_headers: {},
      resume: false,
      durable_ack_enabled: true,
    };
    const error = (await server
      .doHandleExecute(request)
      .catch((e: unknown) => e)) as Error & { statusCode?: number };
    await blocked;

    // Prompt entry into timeout cleanup: far below the sender's full budget
    // and the test's own guard.
    expect(performance.now() - started).toBeLessThan(4000);
    expect(error.message).toContain("Execution timed out after 0.5 seconds");
    expect(error.statusCode).toBe(504);
    // The timeout discriminator rides the ERROR terminal and the callback.
    const errorTerminals = bodies.filter((b) => b.chunk_type === "error");
    expect(errorTerminals.length).toBeGreaterThanOrEqual(1);
    expect(errorTerminals[0]?.metadata?.["error_code"]).toBe(
      RUN_TIMEOUT_ERROR_CODE,
    );
    const callbackFields = server.reportCallback.mock.calls.map(
      (call) => call[3] as Record<string, unknown>,
    );
    expect(
      callbackFields.some(
        (fields) =>
          (fields["metadata"] as Record<string, unknown> | undefined)?.[
            "error_code"
          ] === RUN_TIMEOUT_ERROR_CODE,
      ),
    ).toBe(true);
    // The timeout abort is not a drain: the reason the AER used is its own.
    expect(error.message).not.toContain(DRAIN_ABORT_REASON);
  });

  test("delivers all accepted output after the timeout within the original budget", async () => {
    // Controlled preparation: the agent materialization blocks for 300ms, so
    // the sender's original deadline (~t0+300ms+500ms) extends beyond the
    // execution timer (t0+500ms). Every accepted chunk must still be
    // acknowledged after the timeout fires and before that original deadline,
    // with no fresh retry budget, and the timeout must keep the settlement.
    vi.stubEnv("RUNNER_EXECUTION_TIMEOUT", "0.5");
    vi.resetModules();
    const fresh = await import("../../src/index.js");
    const FreshAERServer = fresh.AERServer;

    const events: string[] = [];
    const ackedContents: string[] = [];
    let acksAtTerminal = -1;
    let acksAtCallback = -1;
    const bodies: ChunkBody[] = [];
    const fetchMock = vi.fn(async (_url: string | URL, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body)) as ChunkBody;
      bodies.push(body);
      if (body.chunk_type === "error") {
        acksAtTerminal = ackedContents.length;
        events.push("terminal");
        return new Response(null, { status: 200 });
      }
      if (bodies.length === 1) {
        // The stalled first POST is abort-aware (a cancelled poster must
        // fail this test) and is released shortly after the execution
        // timeout, relative to POST entry, while the sender's original
        // deadline still has room.
        return new Promise<Response>((resolve, reject) => {
          const abort = (): void => reject(new Error("request aborted"));
          if (init?.signal?.aborted === true) {
            abort();
            return;
          }
          init?.signal?.addEventListener("abort", abort);
          setTimeout(() => {
            for (const chunk of body.chunks ?? []) {
              ackedContents.push(String(chunk["content"]));
            }
            resolve(new Response(null, { status: 200 }));
          }, 250);
        });
      }
      for (const chunk of body.chunks ?? []) {
        ackedContents.push(String(chunk["content"]));
      }
      return new Response(null, { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const server = Object.create(FreshAERServer.prototype) as {
      runtime: Record<string, unknown>;
      chunkSeq: Map<string, number>;
      durableAckSenders: Map<string, unknown>;
      ownerCallbackUrl: Map<string, string>;
      reportCallback: ReturnType<typeof vi.fn>;
      executeViaAgentStream: (...args: unknown[]) => Promise<unknown>;
      doHandleExecute: (request: unknown) => Promise<{ status: string }>;
    };
    server.runtime = {
      getAgent: vi.fn(() => {
        const until = performance.now() + 300;
        while (performance.now() < until) {
          /* deliberate stall */
        }
        return {};
      }),
      memoryWriter: null,
      orgId: "org-1",
      projectId: "proj-1",
    };
    server.chunkSeq = new Map();
    server.durableAckSenders = new Map();
    server.ownerCallbackUrl = new Map();
    server.reportCallback = vi.fn(async () => {
      acksAtCallback = ackedContents.length;
      events.push("callback");
    });

    let blockedResolve: (() => void) | undefined;
    const blocked = new Promise<void>((resolve) => {
      blockedResolve = resolve;
    });
    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      const send = (
        server as unknown as {
          sendStreamChunk: (
            oe: string,
            exec: string,
            type: string,
            content: string,
          ) => Promise<void>;
        }
      ).sendStreamChunk.bind(server);
      await send(oeUrl, executionId, "text", "c1");
      for (let seq = 2; seq <= 97; seq++) {
        await send(oeUrl, executionId, "text", `c${seq}`);
      }
      blockedResolve?.();
      await send(oeUrl, executionId, "text", "blocked");
      return { content: "done", messages: [] };
    };

    const error = (await server
      .doHandleExecute({
        execution_id: "exec-timeout-recovery",
        message: "hi",
        platform_api_url: "http://oe",
        user_id: "u-1",
        custom_headers: {},
        resume: false,
        durable_ack_enabled: true,
      })
      .catch((e: unknown) => e)) as Error & { statusCode?: number };
    await blocked;

    // The timeout still owns the settlement...
    expect(error.message).toContain("Execution timed out after 0.5 seconds");
    expect(error.statusCode).toBe(504);
    // ...and every accepted chunk was acknowledged before the timeout ERROR
    // terminal and its callback, inside the sender's original budget.
    const accepted = Array.from({ length: 97 }, (_, i) => `c${i + 1}`);
    expect(ackedContents.slice().sort()).toEqual(accepted.slice().sort());
    expect(acksAtTerminal).toBe(97);
    expect(acksAtCallback).toBe(97);
    expect(events).toContain("terminal");
    expect(events).toContain("callback");
    const errorTerminals = bodies.filter((b) => b.chunk_type === "error");
    expect(errorTerminals.length).toBeGreaterThanOrEqual(1);
    expect(errorTerminals[0]?.metadata?.["error_code"]).toBe(
      RUN_TIMEOUT_ERROR_CODE,
    );
  });

  test("settlement-budget exhaustion is never presented as completed", async () => {
    // The stream finishes, but the completion settlement is still waiting for
    // an acknowledgement when the execution deadline passes: the turn must
    // settle with the timeout outcome, never COMPLETED.
    vi.stubEnv("RUNNER_EXECUTION_TIMEOUT", "0.5");
    vi.resetModules();
    const fresh = await import("../../src/index.js");
    const FreshAERServer = fresh.AERServer;

    const bodies: ChunkBody[] = [];
    const fetchMock = vi.fn(async (_url: string | URL, init?: RequestInit) => {
      bodies.push(JSON.parse(String(init?.body)) as ChunkBody);
      if (bodies.length === 1) {
        // The batch POST never acknowledges on its own; only the close
        // abort ends it. Later posts (the timeout ERROR terminal) succeed.
        return new Promise<Response>((_resolve, reject) => {
          const abort = (): void => reject(new Error("request aborted"));
          if (init?.signal?.aborted === true) {
            abort();
            return;
          }
          init?.signal?.addEventListener("abort", abort);
        });
      }
      return new Response(null, { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const server = Object.create(FreshAERServer.prototype) as {
      runtime: Record<string, unknown>;
      chunkSeq: Map<string, number>;
      durableAckSenders: Map<string, unknown>;
      ownerCallbackUrl: Map<string, string>;
      reportCallback: ReturnType<typeof vi.fn>;
      executeViaAgentStream: (...args: unknown[]) => Promise<unknown>;
      doHandleExecute: (request: unknown) => Promise<{ status: string }>;
    };
    server.runtime = {
      getAgent: vi.fn(() => ({})),
      memoryWriter: null,
      orgId: "org-1",
      projectId: "proj-1",
    };
    server.chunkSeq = new Map();
    server.durableAckSenders = new Map();
    server.ownerCallbackUrl = new Map();
    server.reportCallback = vi.fn().mockResolvedValue(undefined);

    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      const send = (
        server as unknown as {
          sendStreamChunk: (
            oe: string,
            exec: string,
            type: string,
            content: string,
          ) => Promise<void>;
        }
      ).sendStreamChunk.bind(server);
      await send(oeUrl, executionId, "text", "kept");
      // The stream is done; the completion settlement now waits on the ack.
      return { content: "kept", messages: [] };
    };

    const error = (await server
      .doHandleExecute({
        execution_id: "exec-timeout-settle",
        message: "hi",
        platform_api_url: "http://oe",
        user_id: "u-1",
        custom_headers: {},
        resume: false,
        durable_ack_enabled: true,
      })
      .catch((e: unknown) => e)) as Error & { statusCode?: number };

    // Settlement-budget exhaustion with chunks still unacknowledged is the
    // incomplete-turn failure (never COMPLETED), and it keeps its
    // durable_ack_incomplete discriminator in both runtimes.
    expect(error.message).toContain(
      "Stream chunk delivery could not be durably acknowledged",
    );
    expect(error.statusCode).toBe(500);
    const statuses = server.reportCallback.mock.calls.map((call) => call[2]);
    expect(statuses).not.toContain("COMPLETED");
    expect(statuses).toEqual(["ERROR"]);
    const recorded = server.reportCallback.mock.calls.map(
      (call) => call[3] as Record<string, unknown>,
    );
    const metadata = recorded[0]?.["metadata"] as
      | Record<string, unknown>
      | undefined;
    expect(metadata?.["error_code"]).toBe(DURABLE_ACK_INCOMPLETE_ERROR_CODE);
    const errorTerminals = bodies.filter((b) => b.chunk_type === "error");
    expect(errorTerminals.length).toBeGreaterThanOrEqual(1);
    expect(errorTerminals[0]?.metadata?.["error_code"]).toBe(
      DURABLE_ACK_INCOMPLETE_ERROR_CODE,
    );
  });
});
