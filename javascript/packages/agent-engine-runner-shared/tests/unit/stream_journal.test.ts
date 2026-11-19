/**
 * Tests for the runner stream journal (`server/stream_journal.ts`).
 *
 * Twin of Python's `tests/unit/test_stream_journal.py`. Covers the writer
 * layer only (the read routes and the AER wiring are separate changes):
 *  - a torn or corrupted final record is ignored on read;
 *  - every lifecycle end appends a framed end record naming the reason;
 *  - overflowing the per-journal bounds marks the journal incomplete;
 *  - an evicted journal survives in the index with `retained: false`;
 *  - an open journal reserves runner capacity, and closed journals are evicted
 *    oldest-first to fit a new one;
 *  - a draining writer keeps its reservation and is never evicted, and a
 *    delayed writer settles into creation-order retention;
 *  - a body is charged until its file is confirmed deleted;
 *  - a storage failure stops recording without failing the caller, and the
 *    index never advertises a record that did not reach the file.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
  appendFileSync,
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type * as FsPromises from "node:fs/promises";

import {
  decodeJournalRecords,
  encodeChunkEntry,
  encodeJournalRecord,
  journalFileName,
  STREAM_JOURNAL_END_MARKER_RESERVE,
  StreamJournalManager,
  type JournalChunkEntry,
  type StreamJournal,
} from "../../src/server/stream_journal.js";
import { TEXT } from "../../src/server/chunk_types.js";

// Gate `appendFile` and `rm` so tests can hold a journal write or removal
// pending across close, admission and retention. The defaults are the real
// implementations.
const fsMock = vi.hoisted(() => ({
  appendFile: undefined as
    | ((path: string, data: Buffer) => Promise<void>)
    | undefined,
  rm: undefined as
    | ((path: string, options?: { force?: boolean }) => Promise<void>)
    | undefined,
}));

vi.mock("node:fs/promises", async (importOriginal) => {
  const actual = await importOriginal<typeof FsPromises>();
  fsMock.appendFile = actual.appendFile as never;
  fsMock.rm = actual.rm as never;
  return {
    ...actual,
    appendFile: (path: string, data: Buffer) =>
      (fsMock.appendFile as (p: string, d: Buffer) => Promise<void>)(
        path,
        data,
      ),
    rm: (path: string, options?: { force?: boolean }) =>
      (fsMock.rm as (p: string, o?: { force?: boolean }) => Promise<void>)(
        path,
        options,
      ),
  };
});

let tmpDir: string;

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), "stream-journal-"));
});

afterEach(() => {
  rmSync(tmpDir, { recursive: true, force: true });
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function chunkEntry(
  generation: number,
  seq: number,
  content: string,
): JournalChunkEntry {
  return {
    kind: "chunk",
    seq,
    generation,
    chunk: {
      execution_id: "exec",
      chunk_type: TEXT,
      content,
      error: "",
      metadata: {},
      seq,
    },
  };
}

/** Encoded size of one sample record, so budget tests can be exact. */
function sampleRecordBytes(): number {
  return encodeChunkEntry(chunkEntry(1, 1, "a")).length;
}

function endRecordBytes(lastSeq: number, reason: string): number {
  return encodeJournalRecord(
    Buffer.from(
      JSON.stringify({ kind: "end", generation: 1, last_seq: lastSeq, reason }),
      "utf8",
    ),
  ).length;
}

function entriesOf(journal: { filePath: string }): JournalChunkEntry[] {
  if (!existsSync(journal.filePath)) return [];
  return decodeJournalRecords(readFileSync(journal.filePath), 0).entries;
}

function endOf(journal: { filePath: string }) {
  if (!existsSync(journal.filePath)) return null;
  return decodeJournalRecords(readFileSync(journal.filePath), 0).end;
}

function chargedOf(manager: StreamJournalManager): number {
  return (manager as unknown as { totalBytes: number }).totalBytes;
}

/**
 * Wait for a journal's writer and retention cleanup to finish. A writer-less
 * journal has nothing to await. An expired deadline throws, so non-settlement
 * is reported rather than treated as success.
 */
async function awaitSettled(journal: StreamJournal): Promise<void> {
  const writer = (journal as unknown as { writer: Promise<void> | null })
    .writer;
  if (writer === null) return;
  const done = (journal as unknown as { done: Promise<void> }).done;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const settled = await Promise.race([
    done.then(() => true),
    new Promise<boolean>((resolve) => {
      timer = setTimeout(() => resolve(false), 5000);
    }),
  ]);
  if (timer !== undefined) clearTimeout(timer);
  if (!settled) {
    throw new Error(
      `stream journal cleanup did not settle: ${journal.executionId}`,
    );
  }
}

/** Close and await every owned journal, reporting any that fail to settle. */
async function cleanupOwned(journals: StreamJournal[]): Promise<void> {
  const unsettled: string[] = [];
  for (const journal of journals) {
    try {
      await journal.close("aborted");
    } catch {
      unsettled.push(journal.executionId);
    }
  }
  for (const journal of journals) {
    try {
      await awaitSettled(journal);
    } catch {
      if (!unsettled.includes(journal.executionId)) {
        unsettled.push(journal.executionId);
      }
    }
  }
  if (unsettled.length > 0) {
    throw new Error(`stream journal cleanup did not settle: ${unsettled}`);
  }
}

async function waitUntil(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 5000; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 1));
  }
  throw new Error("condition was not met");
}

// ---------------------------------------------------------------------------
// Journal file format
// ---------------------------------------------------------------------------

describe("stream journal file format", () => {
  test("a torn final record is ignored on read", async () => {
    const manager = new StreamJournalManager({ dir: tmpDir });
    const journal = await manager.openJournal("exec-torn", "s", 1);
    journal.append(chunkEntry(1, 1, "one"));
    journal.append(chunkEntry(1, 2, "two"));
    await manager.closeJournal("exec-torn", 1, "completed");

    // A crash mid-record: the length prefix is written, the body is not.
    const partial = Buffer.alloc(4 + 3);
    partial.writeUInt32BE(64, 0);
    partial.write("abc", 4);
    appendFileSync(journal.filePath, partial);

    expect(entriesOf(journal).map((entry) => entry.seq)).toEqual([1, 2]);
  });

  test("a record with a corrupted CRC is ignored", async () => {
    const manager = new StreamJournalManager({ dir: tmpDir });
    const journal = await manager.openJournal("exec-crc", "s", 1);
    journal.append(chunkEntry(1, 1, "one"));
    journal.append(chunkEntry(1, 2, "two"));
    await manager.closeJournal("exec-crc", 1, "completed");

    const record = encodeChunkEntry(chunkEntry(1, 3, "three"));
    record[record.length - 1] ^= 0xff;
    appendFileSync(journal.filePath, record);

    const { entries } = decodeJournalRecords(readFileSync(journal.filePath), 0);
    expect(entries.map((entry) => entry.seq)).toEqual([1, 2]);
  });
});

// ---------------------------------------------------------------------------
// Lifecycle end marker
// ---------------------------------------------------------------------------

describe("stream journal end marker", () => {
  test.each(["completed", "error", "suspended", "aborted"] as const)(
    "close appends an end record with reason %s",
    async (reason) => {
      const manager = new StreamJournalManager({ dir: tmpDir });
      const journal = await manager.openJournal("exec-end", "s", 1);
      journal.append(chunkEntry(1, 1, "a"));
      journal.append(chunkEntry(1, 2, "b"));
      await manager.closeJournal("exec-end", 1, reason);

      expect(entriesOf(journal).map((entry) => entry.seq)).toEqual([1, 2]);
      expect(endOf(journal)).toEqual({
        kind: "end",
        generation: 1,
        last_seq: 2,
        reason,
      });
    },
  );

  test("an empty lifecycle still writes the end marker", async () => {
    const manager = new StreamJournalManager({ dir: tmpDir });
    const journal = await manager.openJournal("exec-empty", "s", 2);
    await manager.closeJournal("exec-empty", 2, "suspended");

    expect(entriesOf(journal)).toEqual([]);
    expect(endOf(journal)).toEqual({
      kind: "end",
      generation: 2,
      last_seq: 0,
      reason: "suspended",
    });
  });

  test("close is idempotent", async () => {
    const manager = new StreamJournalManager({ dir: tmpDir });
    const journal = await manager.openJournal("exec-idem", "s", 1);
    journal.append(chunkEntry(1, 1, "a"));
    await manager.closeJournal("exec-idem", 1, "completed");
    const size = readFileSync(journal.filePath).length;
    await manager.closeJournal("exec-idem", 1, "error");
    expect(readFileSync(journal.filePath).length).toBe(size);
  });
});

// ---------------------------------------------------------------------------
// Bounds, reservations and eviction
// ---------------------------------------------------------------------------

describe("stream journal bounds", () => {
  test("overflowing the chunk bound marks the journal incomplete", async () => {
    const manager = new StreamJournalManager({ dir: tmpDir, maxChunks: 3 });
    const journal = await manager.openJournal("exec-overflow", "s", 1);
    for (let seq = 1; seq <= 5; seq += 1) {
      journal.append(chunkEntry(1, seq, `c${seq}`));
    }
    await manager.closeJournal("exec-overflow", 1, "completed");

    expect(journal.incomplete).toBe(true);
    expect(journal.lastSeq).toBe(3);
    expect(manager.getIndex("exec-overflow")?.generations).toEqual([
      { generation: 1, last_seq: 3, reason: "incomplete", retained: true },
    ]);
  });

  test("overflowing the byte bound marks the journal incomplete", async () => {
    const sample = chunkEntry(1, 1, "x".repeat(80));
    const oneRecord = encodeChunkEntry(sample).length;
    const manager = new StreamJournalManager({
      dir: tmpDir,
      maxBytes: oneRecord + 10,
    });
    const journal = await manager.openJournal("exec-bytes", "s", 1);
    journal.append(sample);
    journal.append(chunkEntry(1, 2, "y".repeat(80)));
    await manager.closeJournal("exec-bytes", 1, "completed");

    expect(journal.incomplete).toBe(true);
    expect(journal.lastSeq).toBe(1);
  });

  test("an open journal reserves capacity so two cannot over-admit", async () => {
    const one = sampleRecordBytes();
    const manager = new StreamJournalManager({
      dir: tmpDir,
      maxBytes: one,
      runnerMaxBytes: one + STREAM_JOURNAL_END_MARKER_RESERVE,
    });
    const first = await manager.openJournal("exec-1", "s", 1);
    expect(first.incomplete).toBe(false);

    // The runner budget fits exactly one journal's reservation; the second
    // starts incomplete and records nothing, even though the first has not
    // written yet.
    const second = await manager.openJournal("exec-2", "s", 1);
    expect(second.incomplete).toBe(true);
    expect(second.append(chunkEntry(1, 1, "a"))).toBe(0);
    expect(second.lastSeq).toBe(0);
  });

  test("the oldest closed journal is evicted past the per-session cap", async () => {
    const manager = new StreamJournalManager({
      dir: tmpDir,
      maxClosedPerSession: 1,
    });
    const first = await manager.openJournal("exec-a", "s", 1);
    first.append(chunkEntry(1, 1, "a"));
    await manager.closeJournal("exec-a", 1, "completed");

    const second = await manager.openJournal("exec-b", "s", 1);
    second.append(chunkEntry(1, 1, "b"));
    await manager.closeJournal("exec-b", 1, "completed");

    expect(manager.getIndex("exec-a")?.generations).toEqual([
      { generation: 1, last_seq: 1, reason: "completed", retained: false },
    ]);
    expect(manager.getIndex("exec-b")?.generations).toEqual([
      { generation: 1, last_seq: 1, reason: "completed", retained: true },
    ]);
    expect(existsSync(first.filePath)).toBe(false);
    expect(existsSync(second.filePath)).toBe(true);
  });

  test("a body from another session is not evicted by the per-session cap", async () => {
    const manager = new StreamJournalManager({
      dir: tmpDir,
      maxClosedPerSession: 1,
    });
    const other = await manager.openJournal("exec-other", "other-session", 1);
    other.append(chunkEntry(1, 1, "x"));
    await manager.closeJournal("exec-other", 1, "completed");

    const first = await manager.openJournal("exec-a", "s", 1);
    first.append(chunkEntry(1, 1, "a"));
    await manager.closeJournal("exec-a", 1, "completed");
    const second = await manager.openJournal("exec-b", "s", 1);
    second.append(chunkEntry(1, 1, "b"));
    await manager.closeJournal("exec-b", 1, "completed");

    expect(other.retained).toBe(true);
    expect(first.retained).toBe(false);
    expect(second.retained).toBe(true);
  });

  test("a closed journal is evicted to fit a new journal, oldest-first", async () => {
    // A settled journal's body plus one more reservation must not fit, so
    // opening the second journal evicts the first (oldest-first).
    const one = sampleRecordBytes();
    const end = endRecordBytes(3, "completed");
    const capacity = 3 * one + STREAM_JOURNAL_END_MARKER_RESERVE;
    const manager = new StreamJournalManager({
      dir: tmpDir,
      maxBytes: 3 * one,
      runnerMaxBytes: 3 * one + end + capacity - 1,
    });

    const first = await manager.openJournal("exec-1", "s", 1);
    for (const seq of [1, 2, 3]) first.append(chunkEntry(1, seq, "a"));
    await manager.closeJournal("exec-1", 1, "completed");

    const second = await manager.openJournal("exec-2", "s", 1);
    second.append(chunkEntry(1, 1, "a"));

    expect(first.retained).toBe(false);
    expect(second.incomplete).toBe(false);
    expect(existsSync(first.filePath)).toBe(false);

    // Close the still-open journal so no writer outlives the test.
    await manager.closeJournal("exec-2", 1, "completed");
  });

  test("a write failure stops at the written prefix and releases the pending payloads", async () => {
    const real = fsMock.appendFile;
    const manager = new StreamJournalManager({ dir: tmpDir, maxChunks: 1000 });
    const journal = await manager.openJournal("exec-wf", "s", 1);
    const targetPath = join(tmpDir, journalFileName("exec-wf", 1));
    let calls = 0;
    fsMock.appendFile = async (path, data) => {
      // Only this journal's writes are injected: another test's outstanding
      // writer must not consume the first-success allowance.
      if (path !== targetPath) {
        await (real as (p: string, d: Buffer) => Promise<void>)(path, data);
        return;
      }
      calls += 1;
      if (calls === 1) {
        // The first frame reaches the real file; the second fails.
        await (real as (p: string, d: Buffer) => Promise<void>)(path, data);
        return;
      }
      throw new Error("ENOSPC");
    };

    try {
      // Enqueue three frames before the writer yields, so a later accepted
      // payload is still pending when the second append fails.
      journal.append(chunkEntry(1, 1, "a"));
      journal.append(chunkEntry(1, 2, "b"));
      journal.append(chunkEntry(1, 3, "c"));
      await manager.closeJournal("exec-wf", 1, "error");

      // Only sequence 1 reached the file; the index must not advertise the
      // accepted-but-unwritten sequences.
      expect(journal.lastSeq).toBe(1);
      expect(journal.incomplete).toBe(true);
      expect(entriesOf(journal).map((entry) => entry.seq)).toEqual([1]);
      expect(manager.getIndex("exec-wf")?.generations).toEqual([
        { generation: 1, last_seq: 1, reason: "incomplete", retained: true },
      ]);
      // The retired writer released the payloads it could no longer write.
      expect((journal as unknown as { queue: unknown[] }).queue.length).toBe(0);
    } finally {
      fsMock.appendFile = real;
    }
  });

  test("duplicate open awaits the original admission", async () => {
    const realRm = fsMock.rm;
    const one = sampleRecordBytes();
    const end = endRecordBytes(1, "completed");
    const capacity = one + STREAM_JOURNAL_END_MARKER_RESERVE;
    const manager = new StreamJournalManager({
      dir: tmpDir,
      maxBytes: one,
      runnerMaxBytes: one + end + capacity - 1,
      admissionTimeoutMs: 5000,
      closeDrainTimeoutMs: 500,
    });
    // A settled body the next admission must reclaim.
    const prior = await manager.openJournal("exec-prior", "s", 1);
    prior.append(chunkEntry(1, 1, "a"));
    await manager.closeJournal("exec-prior", 1, "completed");

    let entered = false;
    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    fsMock.rm = async (path, options) => {
      entered = true;
      await held;
      await (realRm as (p: string, o?: { force?: boolean }) => Promise<void>)(
        path,
        options,
      );
    };

    try {
      // Two concurrent opens of the same tuple while the first admission is
      // reclaiming capacity.
      const first = manager.openJournal("exec-dup", "s", 1);
      await waitUntil(() => entered);
      const second = manager.openJournal("exec-dup", "s", 1);
      // The duplicate must not resolve while reclamation is still held: it
      // awaits the same admission instead of exposing an unadmitted journal.
      let secondResolved = false;
      void second.then(() => {
        secondResolved = true;
      });
      await new Promise((resolve) => setTimeout(resolve, 50));
      expect(secondResolved).toBe(false);

      release();
      const a = await first;
      const b = await second;

      // The duplicate awaited the same admission: it did not observe a
      // writer-less, unadmitted journal.
      expect(b).toBe(a);
      expect(b.finalized).toBe(false);
      expect(b.incomplete).toBe(false);
      expect(b.reserved).toBe(capacity);

      // Closing through the duplicate settles the same admitted journal and
      // reconciles its reservation to the file exactly once.
      b.append(chunkEntry(1, 1, "a"));
      await manager.closeJournal("exec-dup", 1, "completed");
      expect(a.finalized).toBe(true);
      expect(a.retained).toBe(true);
      expect(endOf(a)?.reason).toBe("completed");
      expect(a.reserved).toBe(readFileSync(a.filePath).length);
      expect(chargedOf(manager)).toBe(a.reserved);
    } finally {
      release();
      fsMock.rm = realRm;
    }
  });

  test("manager close respects a pending admission", async () => {
    const realRm = fsMock.rm;
    const one = sampleRecordBytes();
    const end = endRecordBytes(1, "completed");
    const capacity = one + STREAM_JOURNAL_END_MARKER_RESERVE;
    const manager = new StreamJournalManager({
      dir: tmpDir,
      maxBytes: one,
      runnerMaxBytes: one + end + capacity - 1,
      admissionTimeoutMs: 5000,
      closeDrainTimeoutMs: 500,
    });
    const prior = await manager.openJournal("exec-prior", "s", 1);
    prior.append(chunkEntry(1, 1, "a"));
    await manager.closeJournal("exec-prior", 1, "completed");

    let entered = false;
    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    fsMock.rm = async (path, options) => {
      entered = true;
      await held;
      await (realRm as (p: string, o?: { force?: boolean }) => Promise<void>)(
        path,
        options,
      );
    };

    try {
      const opening = manager.openJournal("exec-dup", "s", 1);
      await waitUntil(() => entered);
      // An unadmitted journal is not exposed, and a manager close cannot
      // finalize it before its writer starts.
      expect(manager.getJournal("exec-dup", 1)).toBeNull();
      const closing = manager.closeJournal("exec-dup", 1, "aborted");

      release();
      const a = await opening;
      await closing;
      expect(a.finalized).toBe(true);
      expect(endOf(a)?.reason).toBe("aborted");
      expect(a.reserved).toBe(readFileSync(a.filePath).length);
      expect(chargedOf(manager)).toBe(a.reserved);
    } finally {
      release();
      fsMock.rm = realRm;
    }
  });

  test("a failed removal stays registered for a later retry", async () => {
    const realRm = fsMock.rm;
    const one = sampleRecordBytes();
    const end = endRecordBytes(1, "completed");
    const capacity = one + STREAM_JOURNAL_END_MARKER_RESERVE;
    const manager = new StreamJournalManager({
      dir: tmpDir,
      maxBytes: one,
      runnerMaxBytes: one + end + capacity - 1,
      admissionTimeoutMs: 50,
      closeDrainTimeoutMs: 500,
    });
    const a = await manager.openJournal("exec-a", "s", 1);
    a.append(chunkEntry(1, 1, "a"));
    await manager.closeJournal("exec-a", 1, "completed");
    const charged = chargedOf(manager);

    let entered = false;
    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    let attempted = false;
    fsMock.rm = async (path, options) => {
      if (!attempted) {
        attempted = true;
        entered = true;
        await held;
        throw new Error("EPERM"); // the removal fails
      }
      await (realRm as (p: string, o?: { force?: boolean }) => Promise<void>)(
        path,
        options,
      );
    };

    try {
      // B's admission selects A for removal and holds it, then starts
      // incomplete.
      const b = await manager.openJournal("exec-b", "s", 1);
      expect(entered).toBe(true);
      expect(b.incomplete).toBe(true);
      // A same-session writer-less close during the hold must not drop A's
      // registration.
      await manager.closeJournal("exec-b", 1, "completed");

      // The held removal now fails: A keeps its charge and stays discoverable.
      release();
      await waitUntil(() => a.retained);
      expect(existsSync(a.filePath)).toBe(true);
      expect(chargedOf(manager)).toBe(charged);

      // Storage recovers: a later admission retries and reclaims A.
      const c = await manager.openJournal("exec-c", "s", 1);
      expect(c.incomplete).toBe(false);
      expect(existsSync(a.filePath)).toBe(false);
      expect(a.reserved).toBe(0);
    } finally {
      release();
      fsMock.rm = realRm;
    }
  });

  test("an unwritable journal directory does not fail the caller", async () => {
    // A file where the journal directory should be makes mkdir fail.
    const blocked = join(tmpDir, "blocked");
    writeFileSync(blocked, "not a directory");
    const manager = new StreamJournalManager({
      dir: join(blocked, "journals"),
    });
    const journal = await manager.openJournal("exec-blocked", "s", 1);
    expect(() => journal.append(chunkEntry(1, 1, "a"))).not.toThrow();
    await manager.closeJournal("exec-blocked", 1, "completed");

    expect(manager.getIndex("exec-blocked")?.generations).toEqual([
      { generation: 1, last_seq: 0, reason: "incomplete", retained: true },
    ]);
  });
});

// ---------------------------------------------------------------------------
// Draining writers, delayed settlement and reclamation ownership
// ---------------------------------------------------------------------------

describe("stream journal settlement and reclamation", () => {
  test("a draining journal keeps its reservation and is not evicted", async () => {
    const real = fsMock.appendFile;
    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    let entered = false;
    fsMock.appendFile = async (path, data) => {
      entered = true;
      await held;
      await (real as (p: string, d: Buffer) => Promise<void>)(path, data);
    };

    try {
      const one = sampleRecordBytes();
      const capacity = one + STREAM_JOURNAL_END_MARKER_RESERVE;
      const manager = new StreamJournalManager({
        dir: tmpDir,
        maxBytes: one,
        runnerMaxBytes: capacity + 1,
        closeDrainTimeoutMs: 50,
        admissionTimeoutMs: 50,
      });

      const draining = await manager.openJournal("exec-1", "s", 1);
      draining.append(chunkEntry(1, 1, "a"));
      await waitUntil(() => entered);

      // Close times out while the append is held: the journal is incomplete,
      // not finalized, and still holds its whole reservation.
      await manager.closeJournal("exec-1", 1, "completed");
      expect(draining.incomplete).toBe(true);
      expect(draining.finalized).toBe(false);
      expect(draining.reserved).toBe(capacity);
      expect(chargedOf(manager)).toBe(capacity);

      // Admission while draining: the draining body cannot be evicted, so the
      // new journal starts incomplete instead of reusing its capacity.
      const second = await manager.openJournal("exec-2", "s", 1);
      expect(second.incomplete).toBe(true);
      expect(draining.retained).toBe(true);
      expect(chargedOf(manager)).toBe(capacity);

      // Release the append: the writer settles exactly once, releases the
      // unused reservation, and keeps the readable prefix.
      release();
      await waitUntil(() => draining.finalized);
      expect(draining.reserved).toBe(draining.bytes);
      expect(chargedOf(manager)).toBe(draining.bytes);
      expect(entriesOf(draining).map((entry) => entry.seq)).toEqual([1]);
      expect(endOf(draining)?.reason).toBe("incomplete");
    } finally {
      fsMock.appendFile = real;
      release();
    }
  });

  test("a delayed writer settles into creation-order retention", async () => {
    const real = fsMock.appendFile;
    const aPath = join(tmpDir, journalFileName("exec-a", 1));
    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    let entered = false;
    fsMock.appendFile = async (path, data) => {
      // Only A is gated, so B and C settle normally while A stays pending.
      if (path === aPath) {
        entered = true;
        await held;
      }
      await (real as (p: string, d: Buffer) => Promise<void>)(path, data);
    };

    const owned: StreamJournal[] = [];
    let bodyError: unknown;
    let cleanupError: unknown;
    try {
      const one = sampleRecordBytes();
      const capacity = one + STREAM_JOURNAL_END_MARKER_RESERVE;
      const manager = new StreamJournalManager({
        dir: tmpDir,
        maxBytes: one,
        maxClosedPerSession: 2,
        runnerMaxBytes: 4 * capacity,
        closeDrainTimeoutMs: 1000,
      });

      // A starts first but its writer stays pending past its bounded close.
      const a = await manager.openJournal("exec-a", "s", 1);
      owned.push(a);
      a.append(chunkEntry(1, 1, "a"));
      await waitUntil(() => entered);
      await manager.closeJournal("exec-a", 1, "completed");
      expect(a.finalized).toBe(false);

      // B and C settle in the same session while A is still draining.
      const b = await manager.openJournal("exec-b", "s", 1);
      owned.push(b);
      b.append(chunkEntry(1, 1, "a"));
      await manager.closeJournal("exec-b", 1, "completed");
      const c = await manager.openJournal("exec-c", "s", 1);
      owned.push(c);
      c.append(chunkEntry(1, 1, "a"));
      await manager.closeJournal("exec-c", 1, "completed");
      expect(b.finalized).toBe(true);
      expect(c.finalized).toBe(true);
      expect(a.finalized).toBe(false);

      // A settles last: retention is by creation order, so the two newer
      // bodies (B, C) survive and the oldest (A) is evicted. Wait for the
      // deletion and the released charge, not just the selection flag.
      release();
      await waitUntil(() => a.reserved === 0);
      expect(a.retained).toBe(false);
      expect(b.retained).toBe(true);
      expect(c.retained).toBe(true);
      expect(existsSync(a.filePath)).toBe(false);
      expect(chargedOf(manager)).toBe(
        readFileSync(b.filePath).length + readFileSync(c.filePath).length,
      );
      // The surviving tails are still readable.
      expect(entriesOf(b).map((entry) => entry.seq)).toEqual([1]);
      expect(endOf(b)?.reason).toBe("completed");
      expect(entriesOf(c).map((entry) => entry.seq)).toEqual([1]);
      expect(endOf(c)?.reason).toBe("completed");
    } catch (error) {
      bodyError = error;
    } finally {
      // Own the cleanup: release the gate, then close and await every owned
      // journal before the fixture tears down. Mock restoration is
      // unconditional. Both failures are reported outside the finally.
      release();
      try {
        await cleanupOwned(owned);
      } catch (error) {
        cleanupError = error;
      } finally {
        fsMock.appendFile = real;
      }
    }
    if (cleanupError !== undefined) {
      if (bodyError !== undefined) {
        throw new AggregateError(
          [bodyError, cleanupError],
          "journal cleanup failed after a body failure",
          { cause: bodyError },
        );
      }
      throw cleanupError;
    }
    if (bodyError !== undefined) {
      throw bodyError;
    }
  }, 30000);

  test("retention counts only bodies still on disk", async () => {
    const real = fsMock.appendFile;
    const aPath = join(tmpDir, journalFileName("exec-a", 1));
    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    let entered = false;
    fsMock.appendFile = async (path, data) => {
      if (path === aPath) {
        entered = true;
        await held;
      }
      await (real as (p: string, d: Buffer) => Promise<void>)(path, data);
    };

    const owned: StreamJournal[] = [];
    let bodyError: unknown;
    let cleanupError: unknown;
    try {
      const one = sampleRecordBytes();
      const capacity = one + STREAM_JOURNAL_END_MARKER_RESERVE;
      const manager = new StreamJournalManager({
        dir: tmpDir,
        maxBytes: one,
        maxClosedPerSession: 2,
        runnerMaxBytes: 2 * capacity + 1,
        closeDrainTimeoutMs: 1000,
      });

      const a = await manager.openJournal("exec-a", "s", 1);
      owned.push(a);
      a.append(chunkEntry(1, 1, "a"));
      await waitUntil(() => entered);
      await manager.closeJournal("exec-a", 1, "completed");

      // C's admission budget-evicts B while A is still draining.
      const b = await manager.openJournal("exec-b", "s", 1);
      owned.push(b);
      b.append(chunkEntry(1, 1, "a"));
      await manager.closeJournal("exec-b", 1, "completed");
      const c = await manager.openJournal("exec-c", "s", 1);
      owned.push(c);
      c.append(chunkEntry(1, 1, "a"));
      await manager.closeJournal("exec-c", 1, "completed");
      expect(b.retained).toBe(false);

      // A settling must not evict another body to make room for the
      // already-gone B: the cap counts only bodies still on disk.
      release();
      await waitUntil(() => a.reserved === a.bytes);
      expect(a.retained).toBe(true);
      expect(c.retained).toBe(true);
      expect(chargedOf(manager)).toBe(
        readFileSync(a.filePath).length + readFileSync(c.filePath).length,
      );
      // A's timed-out close still left its accepted chunk and an incomplete
      // end record; C is a normal completed tail.
      expect(entriesOf(a).map((entry) => entry.seq)).toEqual([1]);
      expect(endOf(a)?.reason).toBe("incomplete");
      expect(entriesOf(c).map((entry) => entry.seq)).toEqual([1]);
      expect(endOf(c)?.reason).toBe("completed");
    } catch (error) {
      bodyError = error;
    } finally {
      // Own the cleanup: release the gate, then close and await every owned
      // journal before the fixture tears down. Mock restoration is
      // unconditional. Both failures are reported outside the finally.
      release();
      try {
        await cleanupOwned(owned);
      } catch (error) {
        cleanupError = error;
      } finally {
        fsMock.appendFile = real;
      }
    }
    if (cleanupError !== undefined) {
      if (bodyError !== undefined) {
        throw new AggregateError(
          [bodyError, cleanupError],
          "journal cleanup failed after a body failure",
          { cause: bodyError },
        );
      }
      throw cleanupError;
    }
    if (bodyError !== undefined) {
      throw bodyError;
    }
  }, 30000);

  test("a torn append keeps the body charged", async () => {
    const real = fsMock.appendFile;
    const one = sampleRecordBytes();
    const capacity = 10 * one + STREAM_JOURNAL_END_MARKER_RESERVE;
    const manager = new StreamJournalManager({
      dir: tmpDir,
      maxBytes: 10 * one,
      maxChunks: 1000,
      runnerMaxBytes: 2 * capacity - 1,
    });
    const journal = await manager.openJournal("exec-torn", "s", 1);
    fsMock.appendFile = async (path, data) => {
      // Write a real prefix before failing: the file is left torn on disk.
      await (real as (p: string, d: Buffer) => Promise<void>)(
        path,
        data.subarray(0, Math.floor(data.length / 2)),
      );
      throw new Error("ENOSPC");
    };

    try {
      journal.append(chunkEntry(1, 1, "a"));
      await manager.closeJournal("exec-torn", 1, "error");

      expect(journal.incomplete).toBe(true);
      expect(journal.lastSeq).toBe(0);
      expect(readFileSync(journal.filePath).length).toBeGreaterThan(0);
      expect(
        decodeJournalRecords(readFileSync(journal.filePath), 0).end,
      ).toBeNull();
      // The occupied file stays charged at the captured reservation, so a
      // later admission must reclaim it before it can use that capacity.
      expect(chargedOf(manager)).toBe(capacity);

      const second = await manager.openJournal("exec-2", "s", 1);
      expect(second.incomplete).toBe(false);
      expect(journal.reserved).toBe(0);
      expect(existsSync(journal.filePath)).toBe(false);
      expect(chargedOf(manager)).toBe(capacity);
    } finally {
      fsMock.appendFile = real;
    }
  });

  test("a torn end marker keeps the body charged", async () => {
    const real = fsMock.appendFile;
    const one = sampleRecordBytes();
    const capacity = 10 * one + STREAM_JOURNAL_END_MARKER_RESERVE;
    const manager = new StreamJournalManager({
      dir: tmpDir,
      maxBytes: 10 * one,
      maxChunks: 1000,
      runnerMaxBytes: 2 * capacity - 1,
    });
    const journal = await manager.openJournal("exec-torn2", "s", 1);
    let calls = 0;
    fsMock.appendFile = async (path, data) => {
      calls += 1;
      if (calls === 1) {
        await (real as (p: string, d: Buffer) => Promise<void>)(path, data);
        return;
      }
      await (real as (p: string, d: Buffer) => Promise<void>)(
        path,
        data.subarray(0, Math.floor(data.length / 2)),
      );
      throw new Error("ENOSPC");
    };

    try {
      journal.append(chunkEntry(1, 1, "a"));
      await manager.closeJournal("exec-torn2", 1, "completed");

      expect(journal.incomplete).toBe(true);
      expect(journal.lastSeq).toBe(1);
      const { entries, end } = decodeJournalRecords(
        readFileSync(journal.filePath),
        0,
      );
      expect(entries.map((entry) => entry.seq)).toEqual([1]);
      expect(end).toBeNull();
      // The partial end frame is on disk, so the reservation stays charged.
      expect(chargedOf(manager)).toBe(capacity);
    } finally {
      fsMock.appendFile = real;
    }
  });

  test("a pending eviction keeps its charge until the file is gone", async () => {
    const realRm = fsMock.rm;
    const one = sampleRecordBytes();
    const end = endRecordBytes(1, "completed");
    const capacity = one + STREAM_JOURNAL_END_MARKER_RESERVE;
    const manager = new StreamJournalManager({
      dir: tmpDir,
      maxBytes: one,
      runnerMaxBytes: 2 * (one + end) + capacity - 1,
      admissionTimeoutMs: 50,
      closeDrainTimeoutMs: 500,
    });
    const first = await manager.openJournal("exec-1", "s", 1);
    first.append(chunkEntry(1, 1, "a"));
    await manager.closeJournal("exec-1", 1, "completed");
    const second = await manager.openJournal("exec-2", "s", 1);
    second.append(chunkEntry(1, 1, "a"));
    await manager.closeJournal("exec-2", 1, "completed");
    const charged = chargedOf(manager);

    let entered = false;
    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    fsMock.rm = async (path, options) => {
      entered = true;
      await held;
      await (realRm as (p: string, o?: { force?: boolean }) => Promise<void>)(
        path,
        options,
      );
    };

    try {
      // Admission needs room, selects the oldest body, and times out while its
      // removal is held: the new journal starts incomplete and the selected
      // body keeps its charge because its file is still there.
      const third = await manager.openJournal("exec-3", "s", 1);
      expect(entered).toBe(true);
      expect(third.incomplete).toBe(true);
      expect(first.retained).toBe(false);
      expect(existsSync(first.filePath)).toBe(true);
      expect(chargedOf(manager)).toBe(charged);

      const freed = first.reserved;
      release();
      await waitUntil(() => !existsSync(first.filePath));
      await waitUntil(() => first.reserved === 0);
      expect(chargedOf(manager)).toBe(charged - freed);
    } finally {
      fsMock.rm = realRm;
      release();
    }
  });

  test("a failed removal keeps the body charged", async () => {
    const realRm = fsMock.rm;
    const one = sampleRecordBytes();
    const end = endRecordBytes(1, "completed");
    const capacity = one + STREAM_JOURNAL_END_MARKER_RESERVE;
    const manager = new StreamJournalManager({
      dir: tmpDir,
      maxBytes: one,
      runnerMaxBytes: 2 * (one + end) + capacity - 1,
      admissionTimeoutMs: 50,
    });
    const first = await manager.openJournal("exec-1", "s", 1);
    first.append(chunkEntry(1, 1, "a"));
    await manager.closeJournal("exec-1", 1, "completed");
    const second = await manager.openJournal("exec-2", "s", 1);
    second.append(chunkEntry(1, 1, "a"));
    await manager.closeJournal("exec-2", 1, "completed");
    const charged = chargedOf(manager);

    fsMock.rm = async () => {
      throw new Error("EPERM");
    };
    try {
      const third = await manager.openJournal("exec-3", "s", 1);
      expect(third.incomplete).toBe(true);
      expect(existsSync(first.filePath)).toBe(true);
      expect(chargedOf(manager)).toBe(charged);
    } finally {
      fsMock.rm = realRm;
    }
  });

  test("a writer-less close is bounded when retention removal stalls", async () => {
    const realRm = fsMock.rm;
    // The budget fits nothing, so every journal starts incomplete with no
    // writer, and every finalized body is selected for retention eviction.
    const manager = new StreamJournalManager({
      dir: tmpDir,
      maxBytes: 1,
      maxClosedPerSession: 0,
      runnerMaxBytes: 0,
      admissionTimeoutMs: 50,
      closeDrainTimeoutMs: 50,
    });
    const journal = await manager.openJournal("exec-a", "s", 1);
    expect(journal.incomplete).toBe(true);

    let entered = false;
    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    fsMock.rm = async (path, options) => {
      entered = true;
      await held;
      await (realRm as (p: string, o?: { force?: boolean }) => Promise<void>)(
        path,
        options,
      );
    };

    try {
      // Closing triggers retention eviction, which is held: the close must
      // still return within its bound.
      await manager.closeJournal("exec-a", 1, "completed");
      expect(entered).toBe(true);
      release();
      await waitUntil(() => journal.reserved === 0);
    } finally {
      fsMock.rm = realRm;
      release();
    }
  });
});

// ---------------------------------------------------------------------------
// Read layer: replay-then-follow, the terminal event, and reader accounting
// ---------------------------------------------------------------------------

describe("stream journal read layer", () => {
  test("replays from the cursor and ends cleanly", async () => {
    const manager = new StreamJournalManager({ dir: tmpDir });
    const journal = await manager.openJournal("exec-read", "s", 4);
    for (const seq of [1, 2, 3]) {
      journal.append(chunkEntry(4, seq, `c${seq}`));
    }
    await manager.closeJournal("exec-read", 4, "completed");

    const events = [];
    for await (const event of journal.iterEvents(0)) events.push(event);

    expect(events.map((event) => event.event)).toEqual([
      "chunk",
      "chunk",
      "chunk",
      "end",
    ]);
    expect(
      events
        .filter((event) => event.event === "chunk")
        .map((event) => event.data["seq"]),
    ).toEqual([1, 2, 3]);
    expect(events.at(-1)?.data).toEqual({
      generation: 4,
      last_seq: 3,
      reason: "completed",
    });

    // A resume cursor replays only what the client has not seen.
    const resumed = [];
    for await (const event of journal.iterEvents(2)) resumed.push(event);
    expect(
      resumed
        .filter((event) => event.event === "chunk")
        .map((event) => event.data["seq"]),
    ).toEqual([3]);
    expect(resumed.at(-1)?.event).toBe("end");
  });

  test("follows chunks appended after the reader started", async () => {
    const manager = new StreamJournalManager({ dir: tmpDir });
    const journal = await manager.openJournal("exec-follow", "s", 1);
    journal.append(chunkEntry(1, 1, "a"));

    const iterator = journal.iterEvents(0);
    const first = await iterator.next();
    expect(first.value?.data["chunk"]).toMatchObject({ content: "a" });

    // Appended while the follower is parked: the wake-up must deliver it, and
    // the close must terminate the stream after the terminal chunk.
    journal.append(chunkEntry(1, 2, "b"));
    await manager.closeJournal("exec-follow", 1, "completed");

    const rest = [];
    for (
      let step = await iterator.next();
      !step.done;
      step = await iterator.next()
    ) {
      rest.push(step.value);
    }
    expect(rest.map((event) => event.event)).toEqual(["chunk", "end"]);
    expect(rest[0]?.data["chunk"]).toMatchObject({ content: "b" });
  });

  test("an incomplete journal ends the stream without waiting for the turn", async () => {
    // A bound hit means nothing more can be recorded, so a follower must learn
    // the prefix is untrustworthy now rather than hang until the turn ends.
    const manager = new StreamJournalManager({ dir: tmpDir, maxChunks: 1 });
    const journal = await manager.openJournal("exec-incomplete", "s", 1);
    journal.append(chunkEntry(1, 1, "a"));
    journal.append(chunkEntry(1, 2, "b"));
    expect(journal.incomplete).toBe(true);

    const events = [];
    for await (const event of journal.iterEvents(0)) events.push(event);
    expect(events.at(-1)?.event).toBe("incomplete");
    expect(events.at(-1)?.data["reason"]).toBe("incomplete");
    await manager.closeJournal("exec-incomplete", 1, "completed");
  });

  test("a torn end record is reported incomplete rather than a clean end", async () => {
    // An interrupted end-marker append: the chunks are intact but the journal
    // cannot prove it closed, so the reader must not present it as complete.
    const manager = new StreamJournalManager({ dir: tmpDir });
    const journal = await manager.openJournal("exec-torn-end", "s", 1);
    journal.append(chunkEntry(1, 1, "a"));
    await manager.closeJournal("exec-torn-end", 1, "completed");
    const bytes = readFileSync(journal.filePath);
    writeFileSync(journal.filePath, bytes.subarray(0, bytes.length - 2));

    const events = [];
    for await (const event of journal.iterEvents(0)) events.push(event);
    expect(events.map((event) => event.event)).toEqual(["chunk", "incomplete"]);
    expect(events.at(-1)?.data["last_seq"]).toBe(1);
  });

  test("reader slots are capped per journal and released idempotently", () => {
    const manager = new StreamJournalManager({
      dir: tmpDir,
      maxReadersPerJournal: 1,
      maxReadersPerRunner: 4,
    });
    const release = manager.acquireReader("exec-readers", 1);
    expect(release).not.toBeNull();
    expect(manager.acquireReader("exec-readers", 1)).toBeNull();

    release?.();
    // A disconnect path may release twice without stealing another slot.
    release?.();
    const again = manager.acquireReader("exec-readers", 1);
    expect(again).not.toBeNull();
    again?.();
  });

  test("reader slots are capped across the runner", () => {
    const manager = new StreamJournalManager({
      dir: tmpDir,
      maxReadersPerJournal: 4,
      maxReadersPerRunner: 1,
    });
    const release = manager.acquireReader("exec-a", 1);
    expect(release).not.toBeNull();
    expect(manager.acquireReader("exec-b", 1)).toBeNull();
    release?.();
    const next = manager.acquireReader("exec-b", 1);
    expect(next).not.toBeNull();
    next?.();
  });
});

describe("stream journal read layer races", () => {
  test("re-reads when the journal moves while a read is in flight", async () => {
    // A follower's read can complete just after the end marker was appended.
    // The reader must re-read the settled file, not judge the journal from a
    // state its read raced past.
    const manager = new StreamJournalManager({ dir: tmpDir });
    const journal = await manager.openJournal("exec-race", "s", 1);
    journal.append(chunkEntry(1, 1, "a"));
    await manager.closeJournal("exec-race", 1, "completed");

    const realRead = journal.readFrom.bind(journal);
    let reads = 0;
    const spy = vi
      .spyOn(journal, "readFrom")
      .mockImplementation(async (offset) => {
        reads += 1;
        if (reads === 1) {
          // Model the writer moving during the read: this read saw nothing.
          (journal as unknown as { changeVersion: number }).changeVersion += 1;
          return { entries: [], end: null, nextOffset: offset, torn: false };
        }
        return realRead(offset);
      });

    const events = [];
    for await (const event of journal.iterEvents(0)) events.push(event);
    spy.mockRestore();

    expect(reads).toBeGreaterThanOrEqual(2);
    expect(events.at(-1)?.event).toBe("end");
    expect(events.at(-1)?.data["reason"]).toBe("completed");
  });
});
