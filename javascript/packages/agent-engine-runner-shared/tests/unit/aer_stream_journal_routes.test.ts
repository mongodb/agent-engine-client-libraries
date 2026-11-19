/**
 * Tests for the runner's stream-journal read routes (aer.ts).
 *
 * Mirrors Python's tests/unit/test_aer_stream_journal_routes.py. The runner
 * owns the journal; these routes serve it back to whichever engine replica
 * resumes the turn, so the cases pin the wire contract a resume depends on:
 *
 *   - the index lists every generation this runner opened, and 404s for an
 *     execution it never saw;
 *   - a generation streams as SSE: `chunk` frames after the cursor, then
 *     exactly one `end` (with the close reason) or `incomplete`;
 *   - an unknown or evicted generation is a 404, not an empty stream;
 *   - the reader cap is a 429;
 *   - the routes sit behind the same auth hook as every peer route, which
 *     refuses a tokenless caller only when a token is configured.
 *
 * Python→TS adaptation notes:
 *   - FastAPI TestClient → Fastify app + app.inject.
 *   - AERServer is built via Object.create(prototype) to skip the ctor; the
 *     journal manager and per-execution generation map are lazy, so they are
 *     injected through the same cast-through-unknown pattern the other AER
 *     route tests use.
 */

import Fastify, { type FastifyInstance } from "fastify";
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { get as httpGet } from "node:http";
import { Writable } from "node:stream";
import type { ServerResponse } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { AERServer } from "../../src/server/aer.js";
import {
  StreamJournalManager,
  journalSseLines,
  type JournalReason,
} from "../../src/server/stream_journal.js";
import { pipeSseLinesToResponse } from "../../src/server/tool.js";
import { registerAuthHook } from "../../src/server/auth.js";

let tmpDir: string;
let manager: StreamJournalManager;

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), "aer-journal-routes-"));
  manager = new StreamJournalManager({ dir: tmpDir });
});

afterEach(() => {
  rmSync(tmpDir, { recursive: true, force: true });
});

interface JournalServerPrivates {
  _streamJournal?: StreamJournalManager;
  registerRoutes: (app: FastifyInstance) => void;
}

function makeServer(
  withManager: StreamJournalManager = manager,
): JournalServerPrivates {
  const server = Object.create(AERServer.prototype) as JournalServerPrivates;
  server._streamJournal = withManager;
  return server;
}

async function makeApp(
  server: JournalServerPrivates,
  options: { env?: NodeJS.ProcessEnv } = {},
): Promise<FastifyInstance> {
  const app = Fastify();
  if (options.env) registerAuthHook(app, "aer", options.env);
  server.registerRoutes(app);
  await app.ready();
  return app;
}

function chunk(executionId: string, seq: number, content: string) {
  return {
    kind: "chunk" as const,
    seq,
    generation: 1,
    chunk: {
      execution_id: executionId,
      chunk_type: "text",
      content,
      seq,
    },
  };
}

async function seed(
  executionId = "exec-1",
  {
    generation = 1,
    contents = ["a", "b"],
    reason = "completed" as JournalReason,
  } = {},
  target: StreamJournalManager = manager,
): Promise<void> {
  const journal = await target.openJournal(executionId, "s-1", generation);
  contents.forEach((content, index) => {
    journal.append({
      ...chunk(executionId, index + 1, content),
      generation,
    });
  });
  await target.closeJournal(executionId, generation, reason);
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

/** Fetch one SSE frame over a real socket, then drop the connection. */
function firstSseFrame(url: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const request = httpGet(url, (response) => {
      let buffer = "";
      response.setEncoding("utf8");
      response.on("data", (piece: string) => {
        buffer += piece;
        if (buffer.includes("\n\n")) {
          // A peer drop mid-follow: the journal is still open.
          request.destroy();
          resolve(buffer);
        }
      });
    });
    request.on("error", (error) => {
      // Destroying the request is the test's own disconnect.
      if ((error as NodeJS.ErrnoException).code === "ECONNRESET") return;
      reject(error);
    });
  });
}

function sseEvents(body: string): Array<{ event: string; data: unknown }> {
  return body
    .trim()
    .split("\n\n")
    .map((block) => {
      const lines = Object.fromEntries(
        block.split("\n").map((line) => {
          const [key, ...rest] = line.split(": ");
          return [key, rest.join(": ")];
        }),
      );
      return {
        event: lines["event"] ?? "",
        data: JSON.parse(lines["data"] ?? "null") as unknown,
      };
    });
}

describe("GET /stream/journal/:execution_id", () => {
  test("lists each generation with its close state", async () => {
    const app = await makeApp(makeServer());
    await seed("exec-1", { generation: 1 });
    await seed("exec-1", { generation: 2, contents: ["c"] });

    const res = await app.inject({
      method: "GET",
      url: "/stream/journal/exec-1",
    });

    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({
      execution_id: "exec-1",
      generations: [
        { generation: 1, last_seq: 2, reason: "completed", retained: true },
        { generation: 2, last_seq: 1, reason: "completed", retained: true },
      ],
    });
    await app.close();
  });

  test("is 404 for an execution the runner never opened", async () => {
    const app = await makeApp(makeServer());
    const res = await app.inject({
      method: "GET",
      url: "/stream/journal/exec-unknown",
    });
    expect(res.statusCode).toBe(404);
    await app.close();
  });

  test("reports an open generation with a null reason", async () => {
    // A live generation has no close reason yet. Both runtimes must report the
    // same value for that state (null), so a consumer never decodes two shapes.
    const app = await makeApp(makeServer());
    const journal = await manager.openJournal("exec-open", "s-1", 1);
    journal.append(chunk("exec-open", 1, "a"));
    await waitUntil(() => journal.lastSeq === 1);

    const res = await app.inject({
      method: "GET",
      url: "/stream/journal/exec-open",
    });

    expect(res.json().generations).toEqual([
      { generation: 1, last_seq: 1, reason: null, retained: true },
    ]);
    await journal.close("completed");
    await app.close();
  });
});

describe("GET /stream/journal/:execution_id/:generation", () => {
  test("streams chunks then exactly one end frame", async () => {
    const app = await makeApp(makeServer());
    await seed("exec-1", { contents: ["a", "b", "c"] });

    const res = await app.inject({
      method: "GET",
      url: "/stream/journal/exec-1/1",
    });

    expect(res.statusCode).toBe(200);
    expect(res.headers["content-type"]).toContain("text/event-stream");
    const events = sseEvents(res.body);
    expect(events.map((event) => event.event)).toEqual([
      "chunk",
      "chunk",
      "chunk",
      "end",
    ]);
    expect(events.at(-1)?.data).toEqual({
      generation: 1,
      last_seq: 3,
      reason: "completed",
    });
    await app.close();
  });

  test("after_seq replays only what the client has not seen", async () => {
    const app = await makeApp(makeServer());
    await seed("exec-1", { contents: ["a", "b", "c"] });

    const res = await app.inject({
      method: "GET",
      url: "/stream/journal/exec-1/1?after_seq=2",
    });

    const events = sseEvents(res.body);
    expect(
      events
        .filter((event) => event.event === "chunk")
        .map((event) => (event.data as { seq: number }).seq),
    ).toEqual([3]);
    expect(events.at(-1)?.event).toBe("end");
    await app.close();
  });

  test("an incomplete journal streams an incomplete frame, not an end", async () => {
    // A per-journal chunk bound hit makes the prefix untrustworthy while the
    // turn keeps running; the follower must be told so, never handed an `end`.
    const bounded = new StreamJournalManager({ dir: tmpDir, maxChunks: 1 });
    const app = await makeApp(makeServer(bounded));
    const journal = await bounded.openJournal("exec-1", "s-1", 1);
    journal.append(chunk("exec-1", 1, "a"));
    journal.append(chunk("exec-1", 2, "b"));
    await bounded.closeJournal("exec-1", 1, "completed");

    const res = await app.inject({
      method: "GET",
      url: "/stream/journal/exec-1/1",
    });

    const events = sseEvents(res.body);
    expect(events.at(-1)?.event).toBe("incomplete");
    expect((events.at(-1)?.data as { reason: string }).reason).toBe(
      "incomplete",
    );
    await app.close();
  });

  test("an unknown generation is 404", async () => {
    const app = await makeApp(makeServer());
    await seed("exec-1", { generation: 1 });

    expect(
      (await app.inject({ method: "GET", url: "/stream/journal/exec-1/2" }))
        .statusCode,
    ).toBe(404);
    expect(
      (await app.inject({ method: "GET", url: "/stream/journal/exec-1/-1" }))
        .statusCode,
    ).toBe(404);
    await app.close();
  });

  test("an invalid path or query parameter is 422", async () => {
    // Mirrors the Python route's typed parameters: a malformed generation or
    // cursor is rejected, never reinterpreted as a different replay range.
    const app = await makeApp(makeServer());
    await seed("exec-1");

    const invalid = [
      "/stream/journal/exec-1/not-a-number",
      "/stream/journal/exec-1/1?after_seq=-1",
      "/stream/journal/exec-1/1?after_seq=abc",
    ];
    for (const url of invalid) {
      expect((await app.inject({ method: "GET", url })).statusCode).toBe(422);
    }
    await app.close();
  });

  test("an evicted generation is 404", async () => {
    // The index entry survives eviction, but the body it describes is gone, so
    // a follow must fail closed instead of serving an empty stream as complete.
    const app = await makeApp(makeServer());
    await seed("exec-1");
    await manager.getJournal("exec-1", 1)?.evict();

    const index = await app.inject({
      method: "GET",
      url: "/stream/journal/exec-1",
    });
    expect(index.json().generations[0].retained).toBe(false);
    expect(
      (await app.inject({ method: "GET", url: "/stream/journal/exec-1/1" }))
        .statusCode,
    ).toBe(404);
    await app.close();
  });

  test("the reader limit is 429 while a follow holds the slot", async () => {
    const limited = new StreamJournalManager({
      dir: tmpDir,
      maxReadersPerJournal: 1,
    });
    const app = await makeApp(makeServer(limited));
    await seed("exec-1", {}, limited);

    const release = limited.acquireReader("exec-1", 1);
    expect(release).not.toBeNull();
    expect(
      (await app.inject({ method: "GET", url: "/stream/journal/exec-1/1" }))
        .statusCode,
    ).toBe(429);

    release?.();
    expect(
      (await app.inject({ method: "GET", url: "/stream/journal/exec-1/1" }))
        .statusCode,
    ).toBe(200);
    // A finished follow gives its slot back: this second full GET succeeds even
    // though the cap is one.
    expect(
      (await app.inject({ method: "GET", url: "/stream/journal/exec-1/1" }))
        .statusCode,
    ).toBe(200);
    await app.close();
  });

  test("refuses a tokenless caller only when a token is configured", async () => {
    // The control on a deployed runner is the executor network policy, which
    // admits only the per-project OE; an unset token must not refuse service.
    const server = makeServer();
    await seed("exec-1");
    const app = await makeApp(server, { env: { RUNNER_AUTH_TOKEN: "s3cret" } });

    expect(
      (await app.inject({ method: "GET", url: "/stream/journal/exec-1" }))
        .statusCode,
    ).toBe(401);
    expect(
      (
        await app.inject({
          method: "GET",
          url: "/stream/journal/exec-1",
          headers: { authorization: "Bearer s3cret" },
        })
      ).statusCode,
    ).toBe(200);
    await app.close();

    const openApp = await makeApp(server, { env: {} });
    expect(
      (await openApp.inject({ method: "GET", url: "/stream/journal/exec-1" }))
        .statusCode,
    ).toBe(200);
    await openApp.close();
  });
});

describe("stream journal route lifecycle", () => {
  test("releases the reader slot when the client disconnects mid-follow", async () => {
    // The route pipes the journal's SSE lines into the hijacked socket with the
    // shared pipe, which owns the socket lifecycle. A peer drop while the
    // journal is still open must free the reader slot, not strand the cap.
    const limited = new StreamJournalManager({
      dir: tmpDir,
      maxReadersPerJournal: 1,
    });
    const journal = await limited.openJournal("exec-live", "s-1", 1);
    journal.append(chunk("exec-live", 1, "a"));
    await waitUntil(() => journal.lastSeq === 1);

    const release = limited.acquireReader("exec-live", 1);
    expect(release).not.toBeNull();

    const written: string[] = [];
    const res = new Writable({
      write: (data, _encoding, callback) => {
        written.push(String(data));
        callback();
        // The peer drops after the first frame arrives.
        res.destroy();
      },
    }) as unknown as ServerResponse & Writable;

    try {
      await pipeSseLinesToResponse(
        res,
        (signal) => journalSseLines(journal, 0, signal),
        { onGeneratorSettled: release ?? (() => {}) },
      );
    } finally {
      await journal.close("aborted");
    }

    expect(written.join("")).toContain("event: chunk");
    // The cap is one: this only succeeds if the disconnect released the slot.
    const replacement = limited.acquireReader("exec-live", 1);
    expect(replacement).not.toBeNull();
    replacement?.();
  });
});

describe("registered live route lifecycle", () => {
  test("a live follow that disconnects returns both reader budgets", async () => {
    // The registered route, over a real socket: the follow parks on an open
    // journal, the peer drops after the first frame, and a replacement follow
    // must still be admitted while that journal is open.
    const limited = new StreamJournalManager({
      dir: tmpDir,
      maxReadersPerJournal: 1,
      maxReadersPerRunner: 1,
    });
    const app = await makeApp(makeServer(limited));
    const base = await app.listen({ port: 0, host: "127.0.0.1" });
    const journal = await limited.openJournal("exec-live", "s-1", 1);
    journal.append(chunk("exec-live", 1, "a"));
    await waitUntil(() => journal.lastSeq === 1);
    const capacity = limited as unknown as {
      runnerReaders: number;
      journalReaders: Map<string, number>;
    };

    try {
      expect(
        await firstSseFrame(`${base}/stream/journal/exec-live/1`),
      ).toContain("event: chunk");
      // The disconnect is observed when the pipe unwinds: both budgets return.
      await waitUntil(() => capacity.runnerReaders === 0);
      expect(capacity.journalReaders.size).toBe(0);
      // The journal stayed open across both follows.
      expect(journal.closed).toBe(false);

      expect(
        await firstSseFrame(`${base}/stream/journal/exec-live/1`),
      ).toContain("event: chunk");
      await waitUntil(() => capacity.runnerReaders === 0);
      expect(journal.closed).toBe(false);
    } finally {
      await journal.close("aborted");
      await app.close();
    }
  });
});
