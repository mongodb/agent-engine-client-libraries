/**
 * Tests for `DurableChunkSender` (ordered durable chunk acknowledgement).
 *
 * Twin of Python's `tests/unit/test_aer_durable_ack.py` Part A:
 *  - chunks are posted in production order, coalescing everything produced
 *    while a request is in flight into the next request (bounded batch);
 *  - a batch is retried unchanged until the engine acknowledges it, and an
 *    acknowledged chunk is never resent;
 *  - batches respect the engine's per-request bound and the waiting queue is
 *    bounded (producers block when it is full);
 *  - a failed sender throws from `send`/`settle` instead of dropping chunks,
 *    and wakes blocked producers;
 *  - the turn budget bounds retries; `settle(timeoutMs)` bounds the error-path
 *    drain below the budget;
 *  - the owner pre-attempt follows the shared owner-callback policy (one
 *    best-effort attempt, service fallback, failure latched).
 */

import { afterEach, describe, expect, test, vi } from "vitest";

import {
  DURABLE_ACK_MAX_BATCH_CHUNKS,
  DurableAckError,
  DurableChunkSender,
  type StreamChunkPayload,
} from "../../src/server/chunk_delivery.js";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

type ScriptEntry =
  | number
  | Error
  | ((init?: RequestInit) => Promise<number | Error>);

/**
 * Recorded request bodies. Durable-ack dispatches post a batch envelope
 * (`{execution_id, chunks}`); journal mode posts one raw chunk per request, the
 * legacy shape the unnegotiated engine parses.
 */
type RecordedBody = Record<string, unknown> & {
  chunks?: StreamChunkPayload[];
};

interface FetchRecorder {
  fetchMock: ReturnType<typeof vi.fn>;
  urls: string[];
  bodies: RecordedBody[];
}

function scriptFetch(
  script: ScriptEntry[],
  defaultStatus = 200,
): FetchRecorder {
  const urls: string[] = [];
  const bodies: RecordedBody[] = [];
  const fetchMock = vi.fn(async (url: string | URL, init?: RequestInit) => {
    urls.push(String(url));
    bodies.push(JSON.parse(String(init?.body)) as RecordedBody);
    let entry: number | Error =
      script.length > 0 ? (script.shift() as ScriptEntry) : defaultStatus;
    if (typeof entry === "function") entry = await entry(init);
    if (entry instanceof Error) throw entry;
    return new Response(null, { status: entry });
  });
  vi.stubGlobal("fetch", fetchMock);
  return { fetchMock, urls, bodies };
}

function chunk(seq: number, content?: string): StreamChunkPayload {
  return {
    execution_id: "exec-durable",
    chunk_type: "text",
    content: content ?? `c${seq}`,
    error: "",
    metadata: {},
    seq,
  };
}

function makeSender(
  options: Partial<ConstructorParameters<typeof DurableChunkSender>[0]> = {},
): DurableChunkSender {
  return new DurableChunkSender({
    executionId: "exec-durable",
    chunkUrl: "http://oe/stream/chunk",
    deadlineMs: performance.now() + 30_000,
    ...options,
  });
}

/** The chunks a request carried, in either wire shape. */
function chunksOf(body: RecordedBody): StreamChunkPayload[] {
  return body.chunks ?? [body as StreamChunkPayload];
}

function contents(body: RecordedBody): string[] {
  return chunksOf(body).map((c) => String(c["content"]));
}

function bodyAt(recorder: FetchRecorder, index: number): RecordedBody {
  const body = recorder.bodies[index];
  if (body === undefined) throw new Error(`no recorded body at index ${index}`);
  return body;
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

describe("DurableChunkSender", () => {
  test("posts in order and coalesces while a request is in flight", async () => {
    let releaseGate: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    const recorder = scriptFetch([
      async () => {
        await gate;
        return 200;
      },
    ]);
    const sender = makeSender();
    try {
      await sender.send(chunk(1));
      await waitUntil(() => recorder.bodies.length === 1);
      // Produced while the first POST is in flight: coalesced, not posted.
      await sender.send(chunk(2));
      await sender.send(chunk(3));
      expect(recorder.bodies).toHaveLength(1);
      releaseGate?.();
      await sender.settle();
    } finally {
      await sender.close();
    }

    expect(recorder.bodies).toHaveLength(2);
    expect(contents(bodyAt(recorder, 0))).toEqual(["c1"]);
    expect(contents(bodyAt(recorder, 1))).toEqual(["c2", "c3"]);
    for (const body of recorder.bodies) {
      expect(body.execution_id).toBe("exec-durable");
      for (const element of chunksOf(body)) {
        expect(element["execution_id"]).toBe("exec-durable");
      }
    }
  });

  test("retries the same batch until acknowledged", async () => {
    let releaseGate: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    const recorder = scriptFetch([
      async () => {
        await gate;
        return 503;
      },
      503,
      new TypeError("down"),
      200,
    ]);
    const sender = makeSender();
    try {
      await sender.send(chunk(1));
      await waitUntil(() => recorder.bodies.length === 1);
      releaseGate?.();
      await sender.settle();
    } finally {
      await sender.close();
    }

    // The same batch is retried unchanged until acknowledged.
    expect(recorder.bodies).toHaveLength(4);
    const first = bodyAt(recorder, 0);
    for (const body of recorder.bodies) {
      expect(body).toEqual(first);
    }
    expect(contents(first)).toEqual(["c1"]);
  });

  test("never resends acknowledged chunks", async () => {
    let releaseGate: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    const recorder = scriptFetch([
      async () => {
        await gate;
        return 200;
      },
      503,
      200,
    ]);
    const sender = makeSender();
    try {
      await sender.send(chunk(1));
      await waitUntil(() => recorder.bodies.length === 1);
      await sender.send(chunk(2));
      releaseGate?.();
      await sender.settle();
    } finally {
      await sender.close();
    }

    expect(contents(bodyAt(recorder, 0))).toEqual(["c1"]);
    expect(contents(bodyAt(recorder, 1))).toEqual(["c2"]);
    expect(contents(bodyAt(recorder, 2))).toEqual(["c2"]);
    for (const body of recorder.bodies.slice(1)) {
      expect(contents(body)).not.toContain("c1");
    }
  });

  test("batches respect the engine bound", async () => {
    let releaseGate: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    const recorder = scriptFetch([
      async () => {
        await gate;
        return 200;
      },
    ]);
    const sender = makeSender();
    try {
      await sender.send(chunk(1));
      await waitUntil(() => recorder.bodies.length === 1);
      // Queued while the first POST is in flight: coalesced into bounded
      // batches in production order.
      for (let seq = 2; seq <= 40; seq++) {
        await sender.send(chunk(seq));
      }
      releaseGate?.();
      await sender.settle();
    } finally {
      await sender.close();
    }

    expect(recorder.bodies.map((body) => chunksOf(body).length)).toEqual([
      1,
      DURABLE_ACK_MAX_BATCH_CHUNKS,
      7,
    ]);
    for (const body of recorder.bodies) {
      expect(chunksOf(body).length).toBeLessThanOrEqual(
        DURABLE_ACK_MAX_BATCH_CHUNKS,
      );
    }
    const flattened = recorder.bodies.flatMap(contents);
    expect(flattened).toEqual(
      Array.from({ length: 40 }, (_, i) => `c${i + 1}`),
    );
  });

  test("bounds the queue and blocks producers", async () => {
    let releaseGate: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    const recorder = scriptFetch([
      async () => {
        await gate;
        return 200;
      },
    ]);
    const sender = makeSender({ maxQueuedChunks: 2 });
    try {
      await sender.send(chunk(1));
      await waitUntil(() => recorder.bodies.length === 1);
      await sender.send(chunk(2));
      await sender.send(chunk(3));
      // The queue is full (2 waiting): a third enqueue must block.
      let blockedResolved = false;
      const blocked = sender.send(chunk(4)).then(() => {
        blockedResolved = true;
      });
      await new Promise((resolve) => setTimeout(resolve, 10));
      expect(blockedResolved).toBe(false);
      releaseGate?.();
      await blocked;
      await sender.settle();
    } finally {
      await sender.close();
    }

    const flattened = recorder.bodies.flatMap(contents);
    expect(flattened).toEqual(["c1", "c2", "c3", "c4"]);
  });

  test("failure wakes a blocked producer", async () => {
    let releaseGate: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    const recorder = scriptFetch(
      [
        async () => {
          await gate;
          return 503;
        },
      ],
      503,
    );
    const sender = makeSender({
      maxQueuedChunks: 1,
      deadlineMs: performance.now() + 100,
    });
    try {
      await sender.send(chunk(1));
      await waitUntil(() => recorder.bodies.length === 1);
      await sender.send(chunk(2));
      const blocked = sender.send(chunk(3));
      releaseGate?.();
      await expect(blocked).rejects.toBeInstanceOf(DurableAckError);
    } finally {
      await sender.close();
    }
  });

  test("settle raises when the budget is exhausted", async () => {
    scriptFetch([], 503);
    const sender = makeSender({ deadlineMs: performance.now() + 150 });
    try {
      await sender.send(chunk(1));
      // The poster may fail (timed out) or this wait may hit the deadline first
      // (incomplete); both mean unacknowledged chunks.
      await expect(sender.settle()).rejects.toThrow(
        /acknowledgement (timed out|incomplete)/,
      );
      // The failure is sticky: later sends throw instead of queueing.
      await expect(sender.send(chunk(2))).rejects.toBeInstanceOf(
        DurableAckError,
      );
    } finally {
      await sender.close();
    }
  });

  test("never retries a client error", async () => {
    const recorder = scriptFetch([400]);
    const sender = makeSender();
    try {
      await sender.send(chunk(1));
      await expect(sender.settle()).rejects.toThrow(/failed/);
    } finally {
      await sender.close();
    }
    expect(recorder.bodies).toHaveLength(1);
  });

  test("settle timeout bounds the error drain", async () => {
    scriptFetch([], 503);
    const sender = makeSender();
    try {
      await sender.send(chunk(1));
      const started = performance.now();
      await expect(sender.settle(150)).rejects.toBeInstanceOf(DurableAckError);
      expect(performance.now() - started).toBeLessThan(5000);
    } finally {
      await sender.close();
    }
  });

  test("owner attempt then service fallback", async () => {
    const recorder = scriptFetch([503, 200]);
    const ownerFailures: boolean[] = [];
    const sender = makeSender({
      ownerUrlGetter: () => "http://owner-0/stream/chunk",
      onOwnerFailure: () => ownerFailures.push(true),
    });
    try {
      await sender.send(chunk(1));
      await sender.settle();
    } finally {
      await sender.close();
    }

    expect(ownerFailures).toEqual([true]);
    expect(recorder.urls[0]).toBe("http://owner-0/stream/chunk");
    expect(recorder.urls[1]).toBe("http://oe/stream/chunk");
  });

  test("batches respect the engine size bound", async () => {
    let releaseGate: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    const recorder = scriptFetch([
      async () => {
        await gate;
        return 200;
      },
    ]);
    // Each ~100-char chunk serializes to ~217 bytes with its envelope, so
    // three fit the 700-byte cap and a fourth must split — a size-blind
    // coalescer would post all six in one oversized request.
    const sender = makeSender({ maxBatchBytes: 700 });
    const encoder = new TextEncoder();
    try {
      await sender.send(chunk(1, "x".repeat(100)));
      await waitUntil(() => recorder.bodies.length === 1);
      for (let seq = 2; seq <= 6; seq++) {
        await sender.send(chunk(seq, "x".repeat(100)));
      }
      releaseGate?.();
      await sender.settle();
    } finally {
      await sender.close();
    }

    expect(recorder.bodies.map((body) => chunksOf(body).length)).toEqual([
      1, 3, 2,
    ]);
    for (const body of recorder.bodies) {
      expect(encoder.encode(JSON.stringify(body)).length).toBeLessThanOrEqual(
        700,
      );
    }
    expect(recorder.bodies.flatMap(contents)).toEqual(
      Array.from({ length: 6 }, () => "x".repeat(100)),
    );
  });

  test("byte accounting covers escaped and multibyte content", async () => {
    let releaseGate: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    const recorder = scriptFetch([
      async () => {
        await gate;
        return 200;
      },
    ]);
    const sender = makeSender({ maxBatchBytes: 700 });
    const encoder = new TextEncoder();
    // Quotes, backslashes, newlines, and multibyte characters all change the
    // wire size; the accounting must count UTF-8 bytes, not code points.
    const tricky = 'a"b\\c\n\u00e9\u{1F600}'.repeat(10);
    try {
      await sender.send(chunk(1, tricky));
      await waitUntil(() => recorder.bodies.length === 1);
      for (let seq = 2; seq <= 5; seq++) {
        await sender.send(chunk(seq, tricky));
      }
      releaseGate?.();
      await sender.settle();
    } finally {
      await sender.close();
    }

    for (const body of recorder.bodies) {
      expect(encoder.encode(JSON.stringify(body)).length).toBeLessThanOrEqual(
        700,
      );
    }
    expect(recorder.bodies.flatMap(contents)).toEqual(
      Array.from({ length: 5 }, () => tricky),
    );
  });

  test("fails explicitly on an oversized chunk", async () => {
    const recorder = scriptFetch([]);
    const sender = makeSender({ maxBatchBytes: 420 });
    try {
      await sender.send(chunk(1, "x".repeat(500)));
      await expect(sender.settle()).rejects.toThrow(
        /exceeds the engine's request size limit/,
      );
    } finally {
      await sender.close();
    }
    // The oversized chunk is never posted.
    expect(recorder.bodies).toHaveLength(0);
  });

  test("stops posting after a terminal failure with later batches queued", async () => {
    // A fatal (non-retryable) response ends delivery for the turn: later
    // queued output must never be posted, and a producer blocked by the queue
    // cap is released with the failure.
    let releaseGate: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    const recorder = scriptFetch(
      [
        async () => {
          await gate;
          return 400;
        },
      ],
      200,
    );
    const sender = makeSender({ maxQueuedChunks: 2 });
    try {
      await sender.send(chunk(1));
      await waitUntil(() => recorder.bodies.length === 1); // batch 1 in flight
      await sender.send(chunk(2));
      await sender.send(chunk(3));
      const blocked = sender.send(chunk(4));
      releaseGate?.();
      await expect(blocked).rejects.toBeInstanceOf(DurableAckError);
      await expect(sender.settle()).rejects.toBeInstanceOf(DurableAckError);
    } finally {
      await sender.close();
    }
    // Exactly one POST: the failed batch, and nothing after it.
    expect(recorder.bodies).toHaveLength(1);
  });

  test("an aborted execution unwinds blocked sends and settles", async () => {
    // The POST never answers on its own; it only settles when the request's
    // abort signal fires, like a real in-flight fetch.
    scriptFetch([
      (init?: RequestInit) =>
        new Promise<number | Error>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () =>
            reject(new Error("request aborted")),
          );
        }),
    ]);
    const controller = new AbortController();
    const sender = makeSender({
      maxQueuedChunks: 1,
      cancellationSignal: controller.signal,
      productionSignal: controller.signal,
    });
    try {
      await sender.send(chunk(1));
      // Let the poster pick chunk 1 up so the queue has room for chunk 2 and
      // chunk 3 is the one that blocks on the cap.
      await new Promise((resolve) => setTimeout(resolve, 10));
      await sender.send(chunk(2));
      const blocked = sender.send(chunk(3));
      const settling = sender.settle();
      controller.abort(new Error("execution cancelled"));
      await expect(blocked).rejects.toThrow("execution cancelled");
      await expect(settling).rejects.toThrow("execution cancelled");
      // The cancellation reason is preserved, not converted into a
      // durable-ack failure.
      await expect(sender.settle()).rejects.toThrow("execution cancelled");
    } finally {
      await sender.close();
    }
  });

  test("an execution timeout releases blocked production but not the drain", async () => {
    // A whole-turn timeout aborts production (a producer blocked on the queue
    // cap must not strand the handler past its deadline) but is not a Stop:
    // accepted output may still use the bounded drain.
    let releaseGate: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    const recorder = scriptFetch([
      async () => {
        await gate;
        return 200;
      },
    ]);
    const drainController = new AbortController();
    const productionController = new AbortController();
    const sender = makeSender({
      maxQueuedChunks: 1,
      cancellationSignal: drainController.signal,
      productionSignal: productionController.signal,
    });
    try {
      await sender.send(chunk(1));
      await waitUntil(() => recorder.bodies.length === 1); // batch 1 in flight
      await sender.send(chunk(2));
      const blocked = sender.send(chunk(3));
      productionController.abort(new Error("Execution timed out"));
      await expect(blocked).rejects.toThrow("Execution timed out");
      // The drain signal is untouched: the accepted output still lands.
      releaseGate?.();
      await sender.settle();
    } finally {
      await sender.close();
    }

    expect(recorder.bodies.flatMap(contents)).toEqual(["c1", "c2"]);
  });

  test("a cancelled owner attempt does not fall back to the service URL", async () => {
    const controller = new AbortController();
    const urls: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string | URL, init?: RequestInit) => {
        urls.push(String(url));
        // The owner request only settles when the cancellation fires.
        return new Promise<Response>((_resolve, reject) => {
          const fail = (): void => reject(new Error("request aborted"));
          if (init?.signal?.aborted === true) {
            fail();
            return;
          }
          init?.signal?.addEventListener("abort", fail);
        });
      }),
    );
    const sender = makeSender({
      ownerUrlGetter: () => "http://owner-0/stream/chunk",
      cancellationSignal: controller.signal,
    });
    try {
      await sender.send(chunk(1));
      const settling = sender.settle();
      controller.abort(new Error("execution cancelled"));
      await expect(settling).rejects.toThrow("execution cancelled");
    } finally {
      await sender.close();
    }
    // Exactly one request: the owner attempt; no service fallback after the
    // cancellation.
    expect(urls).toEqual(["http://owner-0/stream/chunk"]);
  });

  test("close stops the poster and rejects later sends", async () => {
    scriptFetch([]);
    const sender = makeSender();
    await sender.close();
    await expect(sender.send(chunk(1))).rejects.toThrow(/closed/);
  });
});

describe("DurableChunkSender journal mode", () => {
  function makeJournalSender(
    options: Partial<ConstructorParameters<typeof DurableChunkSender>[0]> = {},
  ): DurableChunkSender {
    return makeSender({ journalMode: true, ...options });
  }

  test("posts one raw chunk per request in the legacy shape", async () => {
    // A journal dispatch never negotiates durable acknowledgement, and the
    // engine parses a batch envelope only under that negotiation. Live delivery
    // therefore keeps the legacy single-chunk shape; only the failure policy
    // changes in journal mode.
    let releaseGate: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    const recorder = scriptFetch([
      async () => {
        await gate;
        return 200;
      },
    ]);
    const sender = makeJournalSender();
    try {
      await sender.send(chunk(1));
      await waitUntil(() => recorder.bodies.length === 1);
      // Produced while the first POST is in flight: queued in order, not
      // coalesced into an envelope the engine would not parse.
      await sender.send(chunk(2));
      await sender.send(chunk(3));
      releaseGate?.();
      await waitUntil(() => recorder.bodies.length === 3);
    } finally {
      await sender.close();
    }

    expect(recorder.bodies.flatMap(contents)).toEqual(["c1", "c2", "c3"]);
    for (const body of recorder.bodies) {
      expect(body).not.toHaveProperty("chunks");
      expect(body["execution_id"]).toBe("exec-durable");
      expect(body["chunk_type"]).toBe("text");
    }
    expect(sender.dropped).toBe(0);
    expect(sender.failed).toBeNull();
  });

  test("drops instead of blocking when the queue is full", async () => {
    let releaseGate: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    const recorder = scriptFetch([
      async () => {
        await gate;
        return 200;
      },
    ]);
    const sender = makeJournalSender({ maxQueuedChunks: 2 });
    try {
      // Batch 1 is in flight; the next two fill the queue.
      await sender.send(chunk(1));
      await waitUntil(() => recorder.bodies.length === 1);
      await sender.send(chunk(2));
      await sender.send(chunk(3));
      // Every later send returns at once instead of blocking on the cap.
      for (let seq = 4; seq <= 8; seq += 1) {
        await Promise.race([
          sender.send(chunk(seq)),
          new Promise((_, reject) =>
            setTimeout(() => reject(new Error("send blocked")), 1000),
          ),
        ]);
      }
      expect(sender.dropped).toBe(5);
      expect(sender.failed).toBeNull();
      releaseGate?.();
    } finally {
      await sender.close();
    }
  });

  test("settle does not wait for the acknowledgement", async () => {
    // The engine never acknowledges; journal mode settlement still returns at
    // once and never throws, because the journal holds the turn's chunks.
    scriptFetch([], 503);
    const sender = makeJournalSender({ deadlineMs: performance.now() + 200 });
    try {
      await sender.send(chunk(1));
      await sender.settle();
      expect(sender.failed).toBeNull();
      // A later send still queues rather than throwing on the failure.
      await sender.send(chunk(2));
    } finally {
      await sender.close();
    }
  });

  test("drops a failed batch and keeps delivering", async () => {
    // A non-retryable response drops the batch instead of failing the sender,
    // so delivery continues with what follows it.
    const recorder = scriptFetch([400, 200]);
    const sender = makeJournalSender();
    try {
      await sender.send(chunk(1));
      await waitUntil(() => recorder.bodies.length === 1);
      await sender.send(chunk(2));
      await waitUntil(() => recorder.bodies.length === 2);
    } finally {
      await sender.close();
    }

    expect(recorder.bodies.flatMap((body) => contents(body))).toEqual([
      "c1",
      "c2",
    ]);
    expect(sender.dropped).toBe(1);
    expect(sender.failed).toBeNull();
  });

  test("drops an oversized chunk and keeps delivering", async () => {
    // The durable path fails the sender on a chunk that can never fit the
    // request cap; journal mode drops it and delivers the rest.
    const recorder = scriptFetch([]);
    const sender = makeJournalSender({ maxBatchBytes: 420 });
    try {
      await sender.send(chunk(1, "x".repeat(500)));
      await sender.send(chunk(2, "ok"));
      await waitUntil(() => recorder.bodies.length === 1);
    } finally {
      await sender.close();
    }

    expect(recorder.bodies.flatMap((body) => contents(body))).toEqual(["ok"]);
    expect(sender.dropped).toBe(1);
    expect(sender.failed).toBeNull();
  });
});
