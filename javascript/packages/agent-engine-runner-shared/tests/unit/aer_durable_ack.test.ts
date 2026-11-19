/**
 * Tests for ordered durable chunk acknowledgement through the AER.
 *
 * Twin of Python's `tests/unit/test_aer_durable_ack.py` Part B:
 *  - with the negotiation on, `sendStreamChunk` posts batches through the
 *    sender; the DONE frame and the settlement wait for the drain; an
 *    unacknowledged drain fails the turn (ERROR terminal + ERROR callback,
 *    never COMPLETED);
 *  - a suspension waits for the drain before the SUSPENDED settlement;
 *  - without the negotiation, the legacy one-chunk-per-POST path is untouched
 *    and no sender is registered.
 */

import { afterEach, describe, expect, test, vi } from "vitest";
import { mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  AERServer,
  LLMInvocationError,
  PolicyDeniedException,
  emitStep,
} from "../../src/index.js";
import type { ExecuteRequest } from "../../src/index.js";
import { DurableChunkSender } from "../../src/server/chunk_delivery.js";
import {
  StreamJournalManager,
  decodeJournalRecords,
  journalFileName,
} from "../../src/server/stream_journal.js";
import { TimeoutError } from "../../src/timeout.js";
import {
  DONE,
  DURABLE_ACK_INCOMPLETE_ERROR_CODE,
  TEXT,
} from "../../src/server/chunk_types.js";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

type ChunkBody = {
  execution_id?: string;
  chunk_type?: string;
  content?: string;
  generation?: number;
  chunks?: Array<Record<string, unknown>>;
};

interface ServerPrivates {
  runtime: Record<string, unknown>;
  chunkSeq: Map<string, number>;
  durableAckSenders: Map<string, DurableChunkSender>;
  ownerCallbackUrl: Map<string, string>;
  executeViaAgentStream: (...args: unknown[]) => Promise<unknown>;
  reportCallback: ReturnType<typeof vi.fn>;
  doHandleExecute: (request: ExecuteRequest) => Promise<{ status: string }>;
}

function makeServer(): ServerPrivates {
  const server = Object.create(AERServer.prototype) as ServerPrivates;
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
  return server;
}

function executeRequest(
  executionId: string,
  durableAckEnabled: boolean,
  streamJournalEnabled = false,
): ExecuteRequest {
  return {
    execution_id: executionId,
    message: "hi",
    platform_api_url: "http://oe",
    user_id: "u-1",
    custom_headers: {},
    resume: false,
    durable_ack_enabled: durableAckEnabled,
    stream_journal_enabled: streamJournalEnabled,
  } as unknown as ExecuteRequest;
}

type FetchScriptEntry = number | ((init?: RequestInit) => Promise<number>);

function scriptFetch(
  statuses: FetchScriptEntry[],
  defaultStatus = 200,
): { bodies: ChunkBody[] } {
  const bodies: ChunkBody[] = [];
  let index = 0;
  const fetchMock = vi.fn(async (_url: string | URL, init?: RequestInit) => {
    bodies.push(JSON.parse(String(init?.body)) as ChunkBody);
    let entry = statuses[index] ?? defaultStatus;
    index += 1;
    if (typeof entry === "function") entry = await entry(init);
    return new Response(null, { status: entry });
  });
  vi.stubGlobal("fetch", fetchMock);
  return { bodies };
}

function sendChunk(
  server: ServerPrivates,
  oeUrl: string,
  executionId: string,
  content: string,
): Promise<void> {
  const real = server as unknown as {
    sendStreamChunk: (
      oe: string,
      exec: string,
      type: string,
      content: string,
    ) => Promise<void>;
  };
  return real.sendStreamChunk.call(server, oeUrl, executionId, TEXT, content);
}

async function waitUntil(
  predicate: () => boolean,
  timeoutMs = 5000,
): Promise<void> {
  const deadline = performance.now() + timeoutMs;
  while (!predicate()) {
    if (performance.now() > deadline) {
      throw new Error("condition was not reached in time");
    }
    await new Promise((resolve) => setTimeout(resolve, 1));
  }
}

function callbacks(server: ServerPrivates): Array<Record<string, unknown>> {
  return server.reportCallback.mock.calls.map(
    (call) =>
      ({ status: call[2], ...(call[3] as Record<string, unknown>) }) as Record<
        string,
        unknown
      >,
  );
}

describe("AER durable chunk acknowledgement", () => {
  test("waits for the acknowledgement before DONE and COMPLETED", async () => {
    const server = makeServer();
    const recorder = scriptFetch([]);

    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      for (const token of ["a", "b", "c"]) {
        await sendChunk(server, oeUrl, executionId, token);
      }
      return { content: "abc", messages: [] };
    };

    const response = await server.doHandleExecute(
      executeRequest("exec-ordered", true),
    );

    expect(response.status).toBe("completed");
    expect(callbacks(server).map((c) => c["status"])).toEqual(["COMPLETED"]);
    const chunkBodies = recorder.bodies.filter((b) => b.chunks !== undefined);
    const doneBodies = recorder.bodies.filter((b) => b.chunk_type === DONE);
    expect(doneBodies).toHaveLength(1);
    expect(
      chunkBodies.flatMap((b) => (b.chunks ?? []).map((c) => c["content"])),
    ).toEqual(["a", "b", "c"]);
    // The DONE frame is the last POST and is not batched.
    expect(recorder.bodies[recorder.bodies.length - 1]?.chunk_type).toBe(DONE);
    expect(server.durableAckSenders.size).toBe(0);
  });

  test("fails the turn when chunks are not acknowledged", async () => {
    const server = makeServer();
    scriptFetch([], 503);
    // A tiny budget so the drain fails fast in the test; the AER's own call
    // site builds the sender, so override it on the instance.
    (
      server as unknown as {
        startDurableChunkSender: (request: ExecuteRequest) => void;
      }
    ).startDurableChunkSender = (request: ExecuteRequest): void => {
      server.durableAckSenders.set(
        request.execution_id,
        new DurableChunkSender({
          executionId: request.execution_id,
          chunkUrl: `${request.platform_api_url}/stream/chunk`,
          deadlineMs: performance.now() + 200,
        }),
      );
    };

    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      await sendChunk(server, oeUrl, executionId, "lost");
      return { content: "lost", messages: [] };
    };

    await expect(
      server.doHandleExecute(executeRequest("exec-unacked", true)),
    ).rejects.toThrow(/could not be durably acknowledged/);

    const recorded = callbacks(server);
    expect(recorded.map((c) => c["status"])).toEqual(["ERROR"]);
    const metadata = recorded[0]?.["metadata"] as Record<string, unknown>;
    expect(metadata["error_code"]).toBe(DURABLE_ACK_INCOMPLETE_ERROR_CODE);
  });

  test("suspension waits for the acknowledgement", async () => {
    const server = makeServer();
    const recorder = scriptFetch([]);

    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      await sendChunk(server, oeUrl, executionId, "pre");
      return {
        suspend_payload: { suspend_reason: "approval" },
        messages: [],
      };
    };

    const response = await server.doHandleExecute(
      executeRequest("exec-suspend", true),
    );

    expect(response.status).toBe("suspended");
    expect(callbacks(server)[0]?.["status"]).toBe("SUSPENDED");
    const chunkBodies = recorder.bodies.filter((b) => b.chunks !== undefined);
    expect(
      chunkBodies.flatMap((b) => (b.chunks ?? []).map((c) => c["content"])),
    ).toEqual(["pre"]);
  });

  test("a mid-stream sender failure takes the coded incomplete-turn path", async () => {
    // A gated permanent delivery failure: production observes the recorded
    // failure (not a sleep) and must not return normally.
    let releaseGate: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    const server = makeServer();
    const recorder = scriptFetch(
      [
        async () => {
          await gate;
          return 400;
        },
      ],
      200,
    );
    (
      server as unknown as {
        startDurableChunkSender: (request: ExecuteRequest) => void;
      }
    ).startDurableChunkSender = (request: ExecuteRequest): void => {
      server.durableAckSenders.set(
        request.execution_id,
        new DurableChunkSender({
          executionId: request.execution_id,
          chunkUrl: `${request.platform_api_url}/stream/chunk`,
          deadlineMs: performance.now() + 30_000,
        }),
      );
    };

    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      await sendChunk(server, oeUrl, executionId, "a");
      releaseGate?.();
      const sender = server.durableAckSenders.get(executionId);
      await waitUntil(() => sender?.failed != null);
      // Production does not return normally: the failed sender raises rather
      // than silently dropping the chunk.
      await sendChunk(server, oeUrl, executionId, "b");
      throw new Error("production must not continue past a failed sender");
    };

    await expect(
      server.doHandleExecute(executeRequest("exec-mid", true)),
    ).rejects.toThrow(/could not be durably acknowledged/);

    const recorded = callbacks(server);
    expect(recorded.map((c) => c["status"])).toEqual(["ERROR"]);
    const metadata = recorded[0]?.["metadata"] as Record<string, unknown>;
    expect(metadata["error_code"]).toBe(DURABLE_ACK_INCOMPLETE_ERROR_CODE);
    // The attempted coded ERROR terminal; never a DONE frame or COMPLETED.
    const errorTerminals = recorder.bodies.filter(
      (body) => body.chunk_type === "error",
    );
    expect(errorTerminals).toHaveLength(1);
    const terminalMeta = (
      errorTerminals[0] as { metadata?: Record<string, unknown> }
    ).metadata;
    expect(terminalMeta?.["error_code"]).toBe(
      DURABLE_ACK_INCOMPLETE_ERROR_CODE,
    );
    expect(recorder.bodies.some((body) => body.chunk_type === DONE)).toBe(
      false,
    );
  });

  test("an ordinary execution error drains gated output within the window", async () => {
    let releaseGate: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    const events: string[] = [];
    const server = makeServer();
    scriptFetch([
      (init?: RequestInit) =>
        new Promise<number>((resolve, reject) => {
          const abort = (): void => reject(new Error("request aborted"));
          if (init?.signal?.aborted === true) {
            abort();
            return;
          }
          init?.signal?.addEventListener("abort", abort);
          void gate.then(() => {
            if (init?.signal?.aborted === true) {
              abort();
              return;
            }
            events.push("acked");
            resolve(200);
          });
        }),
    ]);
    server.reportCallback = vi.fn(async () => {
      events.push("callback");
    });
    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      await sendChunk(server, oeUrl, executionId, "kept");
      // Release the acknowledgement inside the error-drain window.
      setTimeout(() => releaseGate?.(), 50);
      throw new Error("agent blew up");
    };

    await expect(
      server.doHandleExecute(executeRequest("exec-generic", true)),
    ).rejects.toThrow("agent blew up");

    // The accepted output was acknowledged before the settlement callback, so
    // it is not lost with the sender.
    expect(events).toContain("acked");
    expect(events).toContain("callback");
    expect(events.indexOf("acked")).toBeLessThan(events.indexOf("callback"));
    // The generic error keeps today's callback shape (no durable-ack code).
    const recorded = callbacks(server);
    expect(recorded).toEqual([{ status: "ERROR", error: "agent blew up" }]);
  });

  test("a drained completion acknowledgement is not a durable-ack failure", async () => {
    // Completion waits on a gated acknowledgement; a real DrainRegistry drain
    // fires: the cancelled execution must keep its own settlement and must not
    // be converted into a durable_ack_incomplete ERROR.
    let releaseGate: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    const server = makeServer();
    const recorder = scriptFetch([
      async (init?: RequestInit) => {
        // Gated until the drain aborts it: an acknowledged chunk must not win
        // the race against the cancellation.
        return new Promise<number>((_resolve, reject) => {
          const abort = (): void => reject(new Error("request aborted"));
          if (init?.signal?.aborted === true) {
            abort();
            return;
          }
          init?.signal?.addEventListener("abort", abort);
          void gate.then(() => undefined);
        });
      },
    ]);
    void releaseGate;
    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      await sendChunk(server, oeUrl, executionId, "a");
      return { content: "a", messages: [] };
    };

    const registry = (
      server as unknown as {
        drainRegistry: {
          beginWork: (executionId: string) => void;
          apply: (request: {
            execution_id: string;
            request_id: string;
            reason: string;
            deadline_at_ms: number;
          }) => unknown;
        };
      }
    ).drainRegistry;
    // The route arms the registry before the handler runs, which is what makes
    // attachController register the handler's controller.
    registry.beginWork("exec-drained");
    const execution = server.doHandleExecute(
      executeRequest("exec-drained", true),
    );
    await waitUntil(() => recorder.bodies.length === 1); // settle is waiting
    registry.apply({
      execution_id: "exec-drained",
      request_id: "drain-1",
      reason: "execution_cancelled",
      deadline_at_ms: Date.now() + 5_000,
    });

    const response = await execution;
    expect(response.status).toBe("cancelled");
    const recorded = callbacks(server);
    expect(
      recorded.some(
        (c) =>
          (c["metadata"] as Record<string, unknown> | undefined)?.[
            "error_code"
          ] === DURABLE_ACK_INCOMPLETE_ERROR_CODE,
      ),
    ).toBe(false);
  });

  test("a drained suspension acknowledgement is not a durable-ack failure", async () => {
    // Suspension waits on a gated acknowledgement; a real DrainRegistry drain
    // fires: the cancelled execution must not settle SUSPENDED and must not
    // produce a durable_ack_incomplete ERROR.
    const server = makeServer();
    const recorder = scriptFetch([
      (init?: RequestInit) =>
        new Promise<number>((_resolve, reject) => {
          const abort = (): void => reject(new Error("request aborted"));
          if (init?.signal?.aborted === true) {
            abort();
            return;
          }
          init?.signal?.addEventListener("abort", abort);
        }),
    ]);
    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      await sendChunk(server, oeUrl, executionId, "pre");
      return {
        suspend_payload: { suspend_reason: "approval" },
        messages: [],
      };
    };

    const registry = (
      server as unknown as {
        drainRegistry: {
          beginWork: (executionId: string) => void;
          apply: (request: {
            execution_id: string;
            request_id: string;
            reason: string;
            deadline_at_ms: number;
          }) => unknown;
        };
      }
    ).drainRegistry;
    registry.beginWork("exec-drained-suspend");
    const execution = server.doHandleExecute(
      executeRequest("exec-drained-suspend", true),
    );
    await waitUntil(() => recorder.bodies.length === 1); // settle is waiting
    registry.apply({
      execution_id: "exec-drained-suspend",
      request_id: "drain-3",
      reason: "execution_cancelled",
      deadline_at_ms: Date.now() + 5_000,
    });

    const response = await execution;
    expect(response.status).toBe("cancelled");
    const recorded = callbacks(server);
    expect(recorded.some((c) => c["status"] === "SUSPENDED")).toBe(false);
    expect(
      recorded.some(
        (c) =>
          (c["metadata"] as Record<string, unknown> | undefined)?.[
            "error_code"
          ] === DURABLE_ACK_INCOMPLETE_ERROR_CODE,
      ),
    ).toBe(false);
  });

  test("a full queue releases blocked production on a drain", async () => {
    // Production is blocked on the queue cap when the execution is drained:
    // the drain must release it promptly and keep the cancelled settlement.
    let releaseGate: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    let blockedResolve: (() => void) | undefined;
    const blocked = new Promise<void>((resolve) => {
      blockedResolve = resolve;
    });
    const server = makeServer();
    const recorder = scriptFetch([
      async (init?: RequestInit) => {
        return new Promise<number>((_resolve, reject) => {
          const abort = (): void => reject(new Error("request aborted"));
          if (init?.signal?.aborted === true) {
            abort();
            return;
          }
          init?.signal?.addEventListener("abort", abort);
          void gate.then(() => undefined);
        });
      },
    ]);
    void releaseGate;
    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      await sendChunk(server, oeUrl, executionId, "c1");
      // Fill the 96-slot queue behind the gated in-flight POST; the next send
      // blocks on the cap.
      for (let seq = 2; seq <= 97; seq++) {
        await sendChunk(server, oeUrl, executionId, `c${seq}`);
      }
      blockedResolve?.();
      await sendChunk(server, oeUrl, executionId, "blocked");
      return { content: "done", messages: [] };
    };

    const registry = (
      server as unknown as {
        drainRegistry: {
          beginWork: (executionId: string) => void;
          apply: (request: {
            execution_id: string;
            request_id: string;
            reason: string;
            deadline_at_ms: number;
          }) => unknown;
        };
      }
    ).drainRegistry;
    registry.beginWork("exec-blocked");
    const execution = server.doHandleExecute(
      executeRequest("exec-blocked", true),
    );
    await blocked;
    registry.apply({
      execution_id: "exec-blocked",
      request_id: "drain-2",
      reason: "execution_cancelled",
      deadline_at_ms: Date.now() + 5_000,
    });

    const response = await execution;
    expect(response.status).toBe("cancelled");
    // The blocked producer never posted past the accepted output.
    expect(recorder.bodies).toHaveLength(1);
  });

  test("an uncoded LLM relay error drains gated output within the window", async () => {
    let releaseGate: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    const events: string[] = [];
    const server = makeServer();
    scriptFetch([
      (init?: RequestInit) =>
        new Promise<number>((resolve, reject) => {
          const abort = (): void => reject(new Error("request aborted"));
          if (init?.signal?.aborted === true) {
            abort();
            return;
          }
          init?.signal?.addEventListener("abort", abort);
          void gate.then(() => {
            // The acknowledgement only counts if the gate wins without a
            // cancellation.
            if (init?.signal?.aborted === true) {
              abort();
              return;
            }
            events.push("acked");
            resolve(200);
          });
        }),
    ]);
    server.reportCallback = vi.fn(async () => {
      events.push("callback");
    });
    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      await sendChunk(server, oeUrl, executionId, "kept");
      // Release the acknowledgement inside the error-drain window.
      setTimeout(() => releaseGate?.(), 50);
      throw new LLMInvocationError("truncated relay response");
    };

    await expect(
      server.doHandleExecute(executeRequest("exec-uncoded", true)),
    ).rejects.toThrow("truncated relay response");

    // The accepted output was acknowledged before the settlement callback.
    expect(events).toContain("acked");
    expect(events).toContain("callback");
    expect(events.indexOf("acked")).toBeLessThan(events.indexOf("callback"));
    // The original uncoded error is preserved (no durable-ack discriminator).
    const recorded = callbacks(server);
    expect(recorded).toEqual([
      {
        status: "ERROR",
        error: "LLM invocation error: truncated relay response",
      },
    ]);
  });

  test("a withheld acknowledgement preserves the uncoded error and 500", async () => {
    const server = makeServer();
    scriptFetch([
      async (init?: RequestInit) => {
        // Never acknowledged: the bounded drain must give up.
        return new Promise<number>((_resolve, reject) => {
          const abort = (): void => reject(new Error("request aborted"));
          if (init?.signal?.aborted === true) {
            abort();
            return;
          }
          init?.signal?.addEventListener("abort", abort);
        });
      },
    ]);
    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      await sendChunk(server, oeUrl, executionId, "kept");
      throw new LLMInvocationError("truncated relay response");
    };

    const error = (await server
      .doHandleExecute(executeRequest("exec-uncoded-withheld", true))
      .catch((e: unknown) => e)) as Error & { statusCode?: number };
    expect(error.message).toContain("truncated relay response");
    expect(error.statusCode).toBe(500);
    const recorded = callbacks(server);
    expect(recorded).toEqual([
      {
        status: "ERROR",
        error: "LLM invocation error: truncated relay response",
      },
    ]);
  }, 20_000);

  test("not-negotiated path keeps single-chunk posts", async () => {
    const server = makeServer();
    const recorder = scriptFetch([]);

    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      await sendChunk(server, oeUrl, executionId, "x");
      await sendChunk(server, oeUrl, executionId, "y");
      return { content: "xy", messages: [] };
    };

    const response = await server.doHandleExecute(
      executeRequest("exec-legacy", false),
    );

    expect(response.status).toBe("completed");
    // Today's shape: one chunk per POST, no batch envelope anywhere.
    expect(recorder.bodies.every((b) => b.chunks === undefined)).toBe(true);
    expect(
      recorder.bodies
        .filter((b) => b.chunk_type === TEXT)
        .map((b) => b.content),
    ).toEqual(["x", "y"]);
    expect(callbacks(server).map((c) => c["status"])).toEqual(["COMPLETED"]);
    expect(server.durableAckSenders.size).toBe(0);
  });
});

describe("AER journal-mode chunk delivery", () => {
  test("does not gate settlement on the acknowledgement", async () => {
    // The decisive journal-mode behavior: a turn whose chunks are never
    // acknowledged still completes, and never produces durable_ack_incomplete.
    const server = makeServer();
    const recorder = scriptFetch([], 400);

    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      for (const token of ["a", "b", "c"]) {
        await sendChunk(server, oeUrl, executionId, token);
      }
      return { content: "abc", messages: [] };
    };

    const response = await server.doHandleExecute(
      executeRequest("exec-journal", false, true),
    );

    expect(response.status).toBe("completed");
    expect(callbacks(server).map((c) => c["status"])).toEqual(["COMPLETED"]);
    // Journal mode posts one raw chunk per request, the legacy shape the
    // unnegotiated engine parses.
    const chunkBodies = recorder.bodies.filter((b) => b.chunks === undefined);
    expect(chunkBodies.flatMap((b) => b.content ?? [])).toEqual(
      expect.arrayContaining(["a", "b", "c"]),
    );
    // Sender torn down with the execution, never presented as incomplete.
    expect(server.durableAckSenders.size).toBe(0);
  });

  test("wins over a negotiated durable acknowledgement", async () => {
    // A dispatch that carries both fields is a journal dispatch: settlement is
    // not gated even though the engine also negotiated durable acknowledgement.
    const server = makeServer();
    scriptFetch([], 400);

    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      await sendChunk(server, oeUrl, executionId, "x");
      return { content: "x", messages: [] };
    };

    const response = await server.doHandleExecute(
      executeRequest("exec-both", true, true),
    );

    expect(response.status).toBe("completed");
    expect(callbacks(server).map((c) => c["status"])).toEqual(["COMPLETED"]);
  });

  test("does not gate the suspension settlement", async () => {
    const server = makeServer();
    scriptFetch([], 400);

    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      await sendChunk(server, oeUrl, executionId, "pre");
      return {
        suspend_payload: { suspend_reason: "approval" },
        messages: [],
      };
    };

    const response = await server.doHandleExecute(
      executeRequest("exec-journal-suspend", false, true),
    );

    expect(response.status).toBe("suspended");
    expect(callbacks(server)[0]?.["status"]).toBe("SUSPENDED");
  });

  test("a completion past the deadline is not gated", async () => {
    // In journal mode settlement does not wait for the drain, so a turn that
    // finishes at (or past) its original deadline is still completed, not
    // failed as incomplete.
    const server = makeServer();
    scriptFetch([], 200);
    const realNow = performance.now.bind(performance);
    let jumped = false;
    vi.spyOn(performance, "now").mockImplementation(() =>
      jumped ? realNow() + 3_600_000 : realNow(),
    );

    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      await sendChunk(server, oeUrl, executionId, "late");
      // The turn ran past its deadline before it settled.
      jumped = true;
      return { content: "late", messages: [] };
    };

    const response = await server.doHandleExecute(
      executeRequest("exec-late", false, true),
    );

    expect(response.status).toBe("completed");
    expect(callbacks(server).map((c) => c["status"])).toEqual(["COMPLETED"]);
  });
});

describe("AER stream journal dispatch", () => {
  test("records every chunk with its generation and closes with the outcome", async () => {
    // The runner-side wiring: a journal dispatch opens the journal, stamps the
    // generation on every chunk (terminal included), records them before they
    // are posted, and closes with the lifecycle reason.
    const dir = mkdtempSync(join(tmpdir(), "aer-journal-dispatch-"));
    const server = makeServer();
    const generation = 5;
    (
      server as unknown as { _streamJournal?: StreamJournalManager }
    )._streamJournal = new StreamJournalManager({ dir });
    const recorder = scriptFetch([], 200);

    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      for (const token of ["a", "b"]) {
        await sendChunk(server, oeUrl, executionId, token);
      }
      return { content: "ab", messages: [] };
    };

    try {
      const response = await server.doHandleExecute({
        ...executeRequest("exec-journal-record", false, true),
        stream_generation: generation,
      } as ExecuteRequest);

      expect(response.status).toBe("completed");
      const records = decodeJournalRecords(
        readFileSync(
          join(dir, journalFileName("exec-journal-record", generation)),
        ),
        0,
      );
      expect(records.entries.map((entry) => entry.seq)).toEqual([1, 2, 3]);
      expect(records.entries.map((entry) => entry.generation)).toEqual([
        generation,
        generation,
        generation,
      ]);
      expect(records.entries.map((entry) => entry.chunk["content"])).toEqual([
        "a",
        "b",
        "ab",
      ]);
      expect(records.end).toMatchObject({
        kind: "end",
        generation,
        reason: "completed",
      });

      // Live delivery keeps the legacy raw shape and carries the generation.
      for (const body of recorder.bodies) {
        expect(body.chunks).toBeUndefined();
        expect(body.generation).toBe(generation);
      }
      // The journal is no longer registered as in flight.
      expect(
        (server as unknown as { _journalGeneration?: Map<string, number> })
          ._journalGeneration?.size ?? 0,
      ).toBe(0);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test("an explicit zero generation leaves the chunk path untouched", async () => {
    // The generation names the journal body. A dispatch that explicitly carries
    // 0 must behave exactly like a non-journal one: an unaddressable body would
    // have no generation to follow and no key in the index.
    const dir = mkdtempSync(join(tmpdir(), "aer-journal-off-"));
    const server = makeServer();
    (
      server as unknown as { _streamJournal?: StreamJournalManager }
    )._streamJournal = new StreamJournalManager({ dir });
    const recorder = scriptFetch([], 200);

    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      await sendChunk(server, oeUrl, executionId, "a");
      return { content: "a", messages: [] };
    };

    try {
      const response = await server.doHandleExecute({
        ...executeRequest("exec-journal-zero", false, true),
        stream_generation: 0,
      } as ExecuteRequest);

      expect(response.status).toBe("completed");
      // No index entry (so no generation for a reader to follow) and no file.
      expect(
        (
          server as unknown as { streamJournal: StreamJournalManager }
        ).streamJournal.getIndex("exec-journal-zero"),
      ).toBeNull();
      expect(readdirSync(dir)).toEqual([]);
      for (const body of recorder.bodies) {
        expect(body.generation).toBeUndefined();
      }
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test("a suspended journal dispatch ends its journal as suspended", async () => {
    // The lifecycle reason must reach the journal: a follower that replays a
    // suspended generation must not be told the turn errored.
    const dir = mkdtempSync(join(tmpdir(), "aer-journal-suspend-"));
    const server = makeServer();
    const generation = 4;
    (
      server as unknown as { _streamJournal?: StreamJournalManager }
    )._streamJournal = new StreamJournalManager({ dir });
    scriptFetch([], 200);

    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      await sendChunk(server, oeUrl, executionId, "pre");
      return { suspend_payload: { suspend_reason: "approval" }, messages: [] };
    };

    try {
      const response = await server.doHandleExecute({
        ...executeRequest("exec-journal-suspend", false, true),
        stream_generation: generation,
      } as ExecuteRequest);

      expect(response.status).toBe("suspended");
      const records = decodeJournalRecords(
        readFileSync(
          join(dir, journalFileName("exec-journal-suspend", generation)),
        ),
        0,
      );
      expect(records.entries.map((entry) => entry.chunk["content"])).toEqual([
        "pre",
      ]);
      expect(records.entries.map((entry) => entry.generation)).toEqual([
        generation,
      ]);
      expect(records.end).toMatchObject({ reason: "suspended" });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("AER stream journal preparation failures", () => {
  test("a graph-build timeout journals its stamped error and closes once", async () => {
    // Preparation can emit the terminal ERROR before execution starts. The
    // journal is opened before it, so that frame is recorded and the generation
    // is still indexed and closed for a follower.
    const dir = mkdtempSync(join(tmpdir(), "aer-journal-prep-"));
    const server = makeServer();
    const generation = 6;
    (
      server as unknown as { _streamJournal?: StreamJournalManager }
    )._streamJournal = new StreamJournalManager({ dir });
    const recorder = scriptFetch([], 200);
    server.runtime = {
      ...server.runtime,
      buildGraph: vi.fn(async () => {
        throw new TimeoutError("build deadline");
      }),
    } as typeof server.runtime;

    try {
      const error = (await server
        .doHandleExecute({
          ...executeRequest("exec-journal-build-timeout", false, true),
          stream_generation: generation,
        } as ExecuteRequest)
        .catch((e: unknown) => e)) as Error & { statusCode?: number };

      expect(error.statusCode).toBe(504);
      // Exactly one terminal ERROR frame was sent to the engine.
      expect(
        recorder.bodies.filter((body) => body.chunk_type === "error"),
      ).toHaveLength(1);
      const records = decodeJournalRecords(
        readFileSync(
          join(dir, journalFileName("exec-journal-build-timeout", generation)),
        ),
        0,
      );
      expect(records.entries.map((entry) => entry.chunk["chunk_type"])).toEqual(
        ["error"],
      );
      expect(records.entries[0]?.generation).toBe(generation);
      expect(records.end).toMatchObject({ reason: "error" });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("AER stream journal preparation outcomes", () => {
  test("a policy denial from the graph builder is journaled before it closes", async () => {
    // A preparation failure the outer handler classifies emits its terminal
    // ERROR; the journal must still be open for it, or replay would lose the
    // discriminator the live stream carried.
    const dir = mkdtempSync(join(tmpdir(), "aer-journal-policy-"));
    const server = makeServer();
    const generation = 8;
    (
      server as unknown as { _streamJournal?: StreamJournalManager }
    )._streamJournal = new StreamJournalManager({ dir });
    const recorder = scriptFetch([], 200);
    server.runtime = {
      ...server.runtime,
      buildGraph: vi.fn(async () => {
        throw new PolicyDeniedException("blocked by guardrail");
      }),
    } as typeof server.runtime;

    try {
      await server
        .doHandleExecute({
          ...executeRequest("exec-journal-policy", false, true),
          stream_generation: generation,
        } as ExecuteRequest)
        .catch(() => undefined);

      const live = recorder.bodies.filter(
        (body) => body.chunk_type === "error",
      );
      expect(live).toHaveLength(1);
      expect(live[0]?.generation).toBe(generation);
      expect(
        (live[0] as { metadata?: Record<string, unknown> }).metadata?.[
          "error_code"
        ],
      ).toBe("policy_denied");

      const records = decodeJournalRecords(
        readFileSync(
          join(dir, journalFileName("exec-journal-policy", generation)),
        ),
        0,
      );
      expect(records.entries.map((entry) => entry.chunk["chunk_type"])).toEqual(
        ["error"],
      );
      expect(
        (records.entries[0]?.chunk["metadata"] as Record<string, unknown>)[
          "error_code"
        ],
      ).toBe("policy_denied");
      expect(records.end).toMatchObject({ reason: "error" });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test("a drain during the graph build settles cancelled and records aborted", async () => {
    const dir = mkdtempSync(join(tmpdir(), "aer-journal-drain-"));
    const server = makeServer();
    const generation = 9;
    (
      server as unknown as { _streamJournal?: StreamJournalManager }
    )._streamJournal = new StreamJournalManager({ dir });
    scriptFetch([], 200);
    // The /execute route registers the execution as drain work before the
    // handler runs, which is what lets a drain reach the execution controller.
    const work = server.drainRegistry.beginWork("exec-journal-drain");
    server.runtime = {
      ...server.runtime,
      buildGraph: vi.fn(async () => {
        server.drainRegistry.apply({
          request_id: "drain-prep",
          execution_id: "exec-journal-drain",
          reason: "execution_cancelled",
          deadline_at_ms: Date.now() + 5000,
        });
        throw new Error("Execution stopped: drained while building");
      }),
    } as typeof server.runtime;

    try {
      const response = await server.doHandleExecute({
        ...executeRequest("exec-journal-drain", false, true),
        stream_generation: generation,
      } as ExecuteRequest);

      expect(response.status).toBe("cancelled");
      const records = decodeJournalRecords(
        readFileSync(
          join(dir, journalFileName("exec-journal-drain", generation)),
        ),
        0,
      );
      expect(records.end).toMatchObject({ reason: "aborted" });
    } finally {
      server.drainRegistry.endWork("exec-journal-drain", undefined, work);
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("AER stream journal local progress", () => {
  test("an in-process emitStep is journaled in order before DONE", async () => {
    // A local tool calls the public progress API. Its step must be sequenced
    // and recorded on the turn's journal, or a replay would miss a step the
    // client saw and still call the turn complete.
    const dir = mkdtempSync(join(tmpdir(), "aer-journal-progress-"));
    const server = makeServer();
    const generation = 11;
    (
      server as unknown as { _streamJournal?: StreamJournalManager }
    )._streamJournal = new StreamJournalManager({ dir });
    scriptFetch([], 200);

    server.executeViaAgentStream = async (...args: unknown[]) => {
      const oeUrl = args[3] as string;
      const executionId = args[4] as string;
      await sendChunk(server, oeUrl, executionId, "before");
      await emitStep("halfway");
      await sendChunk(server, oeUrl, executionId, "after");
      return { content: "beforeafter", messages: [] };
    };

    try {
      const response = await server.doHandleExecute({
        ...executeRequest("exec-journal-progress", false, true),
        stream_generation: generation,
      } as ExecuteRequest);

      expect(response.status).toBe("completed");
      const records = decodeJournalRecords(
        readFileSync(
          join(dir, journalFileName("exec-journal-progress", generation)),
        ),
        0,
      );
      expect(records.entries.map((entry) => entry.seq)).toEqual([1, 2, 3, 4]);
      expect(
        records.entries.map((entry) => [
          entry.chunk["chunk_type"],
          entry.chunk["content"],
        ]),
      ).toEqual([
        ["text", "before"],
        ["step", "halfway"],
        ["text", "after"],
        [DONE, "beforeafter"],
      ]);
      expect(
        records.entries.every((entry) => entry.generation === generation),
      ).toBe(true);
      expect(records.end).toMatchObject({ reason: "completed" });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("AER preparation failure reporting", () => {
  test("an ordinary graph-builder failure reports one ERROR callback", async () => {
    // Preparation failures keep their established reporting: one ERROR callback
    // carrying the original message, before the dispatch is rejected. Journaling
    // must not change that, and the generation still closes as error.
    for (const journaled of [false, true]) {
      const dir = mkdtempSync(join(tmpdir(), "aer-prep-error-"));
      const server = makeServer();
      const generation = 12;
      if (journaled) {
        (
          server as unknown as { _streamJournal?: StreamJournalManager }
        )._streamJournal = new StreamJournalManager({ dir });
      }
      scriptFetch([], 200);
      server.runtime = {
        ...server.runtime,
        buildGraph: vi.fn(async () => {
          throw new Error("missing configuration");
        }),
      } as typeof server.runtime;

      try {
        await server
          .doHandleExecute({
            ...executeRequest("exec-prep-error", false, journaled),
            ...(journaled ? { stream_generation: generation } : {}),
          } as ExecuteRequest)
          .catch(() => undefined);

        const recorded = callbacks(server);
        expect(recorded.map((c) => c["status"])).toEqual(["ERROR"]);
        expect(recorded[0]?.["error"]).toContain("missing configuration");

        if (journaled) {
          const records = decodeJournalRecords(
            readFileSync(
              join(dir, journalFileName("exec-prep-error", generation)),
            ),
            0,
          );
          expect(records.end).toMatchObject({ reason: "error" });
        }
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    }
  });
});
