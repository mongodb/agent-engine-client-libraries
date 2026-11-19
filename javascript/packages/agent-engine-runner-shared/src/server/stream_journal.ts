/**
 * Runner-side stream journal.
 *
 * TypeScript counterpart of Python's
 * `agent_engine_runner_shared.server.stream_journal`. The two runtimes are
 * parallel implementations of the same contract, so this file and its Python
 * twin must be changed together.
 *
 * When the engine dispatches a turn with `stream_journal_enabled` and a
 * `stream_generation`, the runner keeps a journal of the chunks it emits on
 * its own scratch disk, so any engine replica can read the turn's tail back
 * after the dispatching replica dies, without the engine writing anything per
 * chunk.
 *
 * Layout: one append-only file per `(execution_id, generation)`, named from a
 * path-safe base64url encoding of the execution id. Each record is a 4-byte
 * big-endian length prefix, the UTF-8 JSON body, then a 4-byte big-endian
 * CRC32 of the body. A chunk record carries `kind: "chunk"`, its `seq`, its
 * `generation` and the `chunk`; a lifecycle end is a `kind: "end"` record
 * carrying the `generation`, the last written `seq` and the close `reason`. A
 * torn final record is ignored on read. Appends are not fsynced.
 *
 * The producer never blocks: a chunk goes onto a bounded queue and a single
 * writer appends it with async filesystem calls, so no blocking disk call runs
 * on the event loop. The writer advances the written-prefix watermark only
 * after a record reaches the file.
 *
 * Bounds (all inert when the feature is off):
 * - a journal is capped at 64 MiB or 50,000 chunks of chunk records, after
 *   which it is marked incomplete and stops recording; the lifecycle end record
 *   is additional, bounded by the end-marker reserve;
 * - the whole runner is capped at 256 MiB; each open journal reserves its
 *   potential size (chunk bound plus end-marker reserve), settled closed
 *   journals are evicted oldest-first to make room for a new journal, and a
 *   journal that cannot fit even after evicting every closed one starts
 *   incomplete and records nothing (an open or still-draining journal is never
 *   evicted);
 * - at most two closed journal bodies are retained per session.
 *
 * Accounting settles once, after the writer stops: a journal still draining
 * keeps its full reservation and is not evictable, so the runner disk bound
 * holds and a body is never deleted while it is still being written.
 *
 * The read routes that serve a journal back to the engine live in the AER
 * server; this module owns only the record format, the writer, and the
 * whole-runner budget, eviction and retention.
 */

import { mkdir, appendFile, open, rm } from "node:fs/promises";
import { join } from "node:path";

import { getLogger } from "../logger.js";

const logger = getLogger("agent_engine_runner_shared.server.stream_journal");

/** Directory holding every journal file, overridable per deployment. */
export const STREAM_JOURNAL_DIR_ENV = "AGENTENGINE_STREAM_JOURNAL_DIR";
export const STREAM_JOURNAL_DEFAULT_DIR =
  "/scratch/agent-engine-stream-journal";

/** Per-journal bounds on chunk records. */
export const STREAM_JOURNAL_MAX_BYTES = 64 * 1024 * 1024;
export const STREAM_JOURNAL_MAX_CHUNKS = 50_000;

/** Bytes reserved beyond the chunk bound for the lifecycle end record. */
export const STREAM_JOURNAL_END_MARKER_RESERVE = 512;

/** Whole-runner byte budget; closed journals are evicted to stay under it. */
export const STREAM_JOURNAL_RUNNER_MAX_BYTES = 256 * 1024 * 1024;

/** At most this many closed journal bodies are retained per session. */
export const STREAM_JOURNAL_MAX_CLOSED_PER_SESSION = 2;

/** Bounded producer queue ahead of the writer. */
export const STREAM_JOURNAL_QUEUE_MAX_CHUNKS = 4096;

/** How long close waits for the writer to settle before giving up. */
export const STREAM_JOURNAL_CLOSE_DRAIN_TIMEOUT_MS = 2000;

/** How long admission waits for reclamation before starting incomplete. */
export const STREAM_JOURNAL_ADMISSION_TIMEOUT_MS = 1000;

/**
 * Concurrent live-follow readers, per journal and per runner. A follower is a
 * long-lived connection, so the cap is what keeps a stuck or hostile engine
 * replica from pinning unbounded read loops on one runner.
 */
export const STREAM_JOURNAL_MAX_READERS_PER_JOURNAL = 4;
export const STREAM_JOURNAL_MAX_READERS_PER_RUNNER = 32;

/**
 * A follower is woken on every append and close; this bound is only the
 * backstop that keeps it re-reading if a wake-up is ever missed.
 */
export const STREAM_JOURNAL_READER_WAKE_TIMEOUT_MS = 500;

/** Why a journal stopped recording. */
export type JournalReason =
  | "completed"
  | "error"
  | "suspended"
  | "aborted"
  | "incomplete";

/** One recorded chunk, as carried in the file and the index. */
export interface JournalChunkEntry {
  kind: "chunk";
  seq: number;
  generation: number;
  chunk: Record<string, unknown>;
}

/** One encoded chunk frame queued for the writer. */
interface QueuedFrame {
  seq: number;
  data: Buffer;
}

/** The lifecycle end record written when a journal closes. */
export interface JournalEndRecord {
  kind: "end";
  generation: number;
  last_seq: number;
  reason: JournalReason;
}

/** One generation's summary in the per-execution index. */
export interface JournalIndexEntry {
  generation: number;
  last_seq: number;
  reason: JournalReason | null;
  retained: boolean;
}

/** The runner's per-execution index of the journals it opened. */
export interface JournalIndex {
  execution_id: string;
  generations: JournalIndexEntry[];
}

/** One event in a journal's replay-then-follow stream. */
export interface JournalStreamEvent {
  /** `chunk` while events remain; then exactly one `end` or `incomplete`. */
  event: "chunk" | "end" | "incomplete";
  data: Record<string, unknown>;
}

/**
 * Encode one Server-Sent Event frame.
 *
 * Event names are fixed literals and the payload is JSON-encoded, so a raw
 * newline carried inside any field value is escaped and cannot forge a frame
 * boundary.
 */
export function sseFrame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/**
 * A journal's replay-then-follow events, encoded as SSE lines.
 *
 * Yielding the frames here (rather than writing to the socket) keeps the wire
 * format on the journal contract and lets the server pipe own disconnects and
 * backpressure. `signal` aborts a parked follower promptly.
 */
export async function* journalSseLines(
  journal: StreamJournal,
  afterSeq: number,
  signal: AbortSignal,
): AsyncGenerator<string, void, void> {
  for await (const { event, data } of journal.iterEvents(afterSeq, signal)) {
    yield sseFrame(event, data);
  }
}

const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let index = 0; index < 256; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) {
      value = (value & 1) !== 0 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    }
    table[index] = value >>> 0;
  }
  return table;
})();

/** Standard CRC32 (IEEE 802.3), matching Python's `zlib.crc32`. */
export function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (let index = 0; index < bytes.length; index += 1) {
    const byte = bytes[index] ?? 0;
    crc = (CRC32_TABLE[(crc ^ byte) & 0xff] ?? 0) ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/** Frame a record: length prefix + body + CRC32 of the body. */
export function encodeJournalRecord(body: Buffer): Buffer {
  const record = Buffer.allocUnsafe(body.length + 8);
  record.writeUInt32BE(body.length, 0);
  body.copy(record, 4);
  record.writeUInt32BE(crc32(body), body.length + 4);
  return record;
}

/** Encode one chunk entry into its framed on-disk form. */
export function encodeChunkEntry(entry: JournalChunkEntry): Buffer {
  return encodeJournalRecord(Buffer.from(JSON.stringify(entry), "utf8"));
}

function isJournalEndRecord(value: unknown): value is JournalEndRecord {
  if (typeof value !== "object" || value === null) return false;
  const record = value as Record<string, unknown>;
  return (
    record["kind"] === "end" &&
    typeof record["generation"] === "number" &&
    typeof record["last_seq"] === "number"
  );
}

function isJournalChunkEntry(value: unknown): value is JournalChunkEntry {
  if (typeof value !== "object" || value === null) return false;
  const entry = value as Record<string, unknown>;
  return (
    typeof entry["seq"] === "number" &&
    typeof entry["generation"] === "number" &&
    typeof entry["chunk"] === "object" &&
    entry["chunk"] !== null
  );
}

/**
 * Decode complete records starting at `offset`.
 *
 * Stops at the first incomplete or corrupt record and reports its start as the
 * next offset, so a later read can pick it up once the writer has finished it.
 * A torn final record is therefore ignored, never surfaced as a chunk.
 */
export function decodeJournalRecords(
  buffer: Buffer,
  offset: number,
): {
  entries: JournalChunkEntry[];
  end: JournalEndRecord | null;
  nextOffset: number;
} {
  const entries: JournalChunkEntry[] = [];
  let end: JournalEndRecord | null = null;
  let position = offset;
  while (position + 8 <= buffer.length) {
    const length = buffer.readUInt32BE(position);
    const bodyStart = position + 4;
    const bodyEnd = bodyStart + length;
    if (bodyEnd + 4 > buffer.length) break;
    const body = buffer.subarray(bodyStart, bodyEnd);
    const storedCrc = buffer.readUInt32BE(bodyEnd);
    if (crc32(body) !== storedCrc) break;
    let parsed: unknown;
    try {
      parsed = JSON.parse(body.toString("utf8"));
    } catch {
      break;
    }
    if (isJournalEndRecord(parsed)) {
      end = parsed;
    } else if (isJournalChunkEntry(parsed)) {
      entries.push(parsed);
    } else {
      break;
    }
    position = bodyEnd + 4;
  }
  return { entries, end, nextOffset: position };
}

/** Encode a journal filename from tenant-influenced identifiers. */
export function journalFileName(
  executionId: string,
  generation: number,
): string {
  // base64url is path-safe by construction: it never emits `/`, `.`, or `..`,
  // so an execution id can never escape the journal directory.
  const encoded = Buffer.from(executionId, "utf8").toString("base64url");
  return `${encoded}.${generation}.journal`;
}

/**
 * A one-shot wake-up signal.
 *
 * `notify` before `wait` is never lost: the pending flag carries the wake-up to
 * the next waiter. `notify` with a waiter resolves it directly and leaves no
 * pending flag, so a waiter loop does not spin.
 */
class JournalSignal {
  private pending = false;
  private waiters: Array<() => void> = [];

  notify(): void {
    const waiters = this.waiters;
    if (waiters.length === 0) {
      this.pending = true;
      return;
    }
    this.waiters = [];
    for (const resolve of waiters) resolve();
  }

  async wait(): Promise<void> {
    if (this.pending) {
      this.pending = false;
      return;
    }
    await new Promise<void>((resolve) => this.waiters.push(resolve));
  }
}

/** Resolve when `promise` settles or `timeoutMs` elapses, whichever is first. */
async function bounded(
  promise: Promise<void>,
  timeoutMs: number,
): Promise<void> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<void>((resolve) => {
    timer = setTimeout(resolve, timeoutMs);
  });
  try {
    await Promise.race([promise, timeout]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

/** Per-journal bounds, injectable so tests can exercise overflow cheaply. */
interface JournalLimits {
  maxBytes: number;
  maxChunks: number;
  maxQueueChunks: number;
}

/**
 * One `(execution_id, generation)` journal.
 *
 * The journal keeps the written-prefix watermark (`lastSeq`, `bytes`) separate
 * from the producer admission counters (`acceptedChunks`, `acceptedBytes`), so
 * the bounds still hold while records are queued and the index never advertises
 * a record that has not reached the file.
 */
export class StreamJournal {
  readonly executionId: string;
  readonly sessionId: string;
  readonly generation: number;
  readonly dir: string;
  readonly filePath: string;
  /** Set once the journal closes; null while it is still recording. */
  reason: JournalReason | null = null;
  /** The written prefix: advanced only after a record reaches the file. */
  lastSeq = 0;
  bytes = 0;
  /** Producer admission counters, separate from the written prefix. */
  acceptedChunks = 0;
  acceptedBytes = 0;
  /** Capacity reserved against the runner budget while the journal is open. */
  reserved = 0;
  /** False once the body has been evicted (the index entry survives). */
  retained = true;
  /** True once the journal can no longer guarantee a complete prefix. */
  incomplete = false;
  closed = false;
  /** True once the writer has settled and the journal has been finalized. */
  finalized = false;
  /** True while a removal is in flight, so it is never selected twice. */
  evicting = false;
  /** Creation order across the whole runner, for oldest-first eviction. */
  order = 0;

  // Encoded frames queued for the writer. Encoding once at admission keeps the
  // byte bound honest and makes the record immutable to a caller that mutates
  // the chunk after append() returns.
  private readonly queue: QueuedFrame[] = [];
  private readonly signal = new JournalSignal();
  private readonly limits: JournalLimits;
  private readonly closeDrainTimeoutMs: number;
  private readonly onFinalize: (journal: StreamJournal) => Promise<void>;
  private writer: Promise<void> | null = null;
  private resolveDone: () => void = () => {};
  private readonly done: Promise<void>;
  /** True once an append or end-marker write failed: the file may be torn. */
  writeFailed = false;
  // Live followers are woken individually (a shared signal would let one
  // follower consume another's wake-up), and the version lets a follower tell
  // "changed while I was reading" from "nothing new".
  private changeVersion = 0;
  private changeWaiters: Array<() => void> = [];

  constructor(
    executionId: string,
    sessionId: string,
    generation: number,
    dir: string,
    limits: JournalLimits,
    closeDrainTimeoutMs: number = STREAM_JOURNAL_CLOSE_DRAIN_TIMEOUT_MS,
    onFinalize: (journal: StreamJournal) => Promise<void> = async () => {},
  ) {
    this.executionId = executionId;
    this.sessionId = sessionId;
    this.generation = generation;
    this.dir = dir;
    this.filePath = join(dir, journalFileName(executionId, generation));
    this.limits = limits;
    this.closeDrainTimeoutMs = closeDrainTimeoutMs;
    this.onFinalize = onFinalize;
    this.done = new Promise<void>((resolve) => {
      this.resolveDone = resolve;
    });
  }

  /** Start the writer. Filesystem work happens on the writer, not the caller. */
  start(): void {
    this.writer = this.runWriter();
  }

  /** Start a journal that records nothing (the runner budget is full). */
  startIncomplete(): void {
    this.incomplete = true;
    this.reason = "incomplete";
  }

  /** True while the journal can still accept a chunk. */
  private get writable(): boolean {
    return !this.closed && !this.incomplete && this.retained;
  }

  /**
   * Accept one chunk for the journal, returning the accepted bytes (0 when it
   * was not accepted). Never blocks: the frame is encoded once, here, so the
   * byte bound reflects exactly what reaches the file and a caller mutating
   * `entry` afterwards cannot change what is journaled.
   */
  append(entry: JournalChunkEntry): number {
    if (!this.writable) return 0;
    const data = encodeChunkEntry(entry);
    if (this.queue.length >= this.limits.maxQueueChunks) {
      this.markIncomplete();
      return 0;
    }
    if (
      this.acceptedChunks + 1 > this.limits.maxChunks ||
      this.acceptedBytes + data.length > this.limits.maxBytes
    ) {
      this.markIncomplete();
      return 0;
    }
    this.acceptedChunks += 1;
    this.acceptedBytes += data.length;
    this.queue.push({ seq: entry.seq, data });
    this.signal.notify();
    return data.length;
  }

  /** Stop recording; the prefix can no longer be trusted. */
  markIncomplete(): void {
    if (this.incomplete) return;
    this.incomplete = true;
    this.reason = "incomplete";
    this.signal.notify();
    this.bumpChange();
    logger.debug(
      `Stream journal for execution ${this.executionId} generation ` +
        `${this.generation} exceeded its bounds and is now incomplete`,
    );
  }

  /** Wake every live follower after the file or the journal state moved. */
  private bumpChange(): void {
    this.changeVersion += 1;
    const waiters = this.changeWaiters;
    this.changeWaiters = [];
    for (const wake of waiters) wake();
  }

  /**
   * Resolve once the journal has changed after `sinceVersion`, the signal
   * aborts, or the backstop timeout elapses.
   *
   * The version check and the waiter registration run without an `await`
   * between them, so a change cannot slip through the gap.
   */
  async waitForChange(
    sinceVersion: number,
    signal?: AbortSignal,
  ): Promise<void> {
    if (this.changeVersion !== sinceVersion || this.closed || this.incomplete) {
      return;
    }
    let wake: () => void = () => {};
    const changed = new Promise<void>((resolve) => {
      wake = resolve;
    });
    // Registered directly as the waiter and the abort listener, so cleanup has
    // one stable reference to remove.
    const onAbort = (): void => {
      wake();
    };
    this.changeWaiters.push(onAbort);
    signal?.addEventListener("abort", onAbort, { once: true });
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      await Promise.race([
        changed,
        new Promise<void>((resolve) => {
          timer = setTimeout(resolve, STREAM_JOURNAL_READER_WAKE_TIMEOUT_MS);
        }),
      ]);
    } finally {
      if (timer !== undefined) clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
      this.changeWaiters = this.changeWaiters.filter(
        (entry) => entry !== onAbort,
      );
    }
  }

  /**
   * Read the records appended after `offset`.
   *
   * `nextOffset` advances only past records that were fully written, so a live
   * follower re-reads a partial trailing record once the writer finishes it
   * instead of serving a truncated body. `torn` is true when the bytes after
   * `nextOffset` are a short, CRC-mismatched or undecodable record: harmless
   * while the journal is live, and `incomplete` once it has closed.
   */
  async readFrom(offset: number): Promise<{
    entries: JournalChunkEntry[];
    end: JournalEndRecord | null;
    nextOffset: number;
    torn: boolean;
  }> {
    let buffer: Buffer;
    try {
      const handle = await open(this.filePath, "r");
      try {
        const { size } = await handle.stat();
        if (size <= offset) {
          return { entries: [], end: null, nextOffset: offset, torn: false };
        }
        const chunk = Buffer.allocUnsafe(size - offset);
        let read = 0;
        while (read < chunk.length) {
          const { bytesRead } = await handle.read(
            chunk,
            read,
            chunk.length - read,
            offset + read,
          );
          if (bytesRead <= 0) break;
          read += bytesRead;
        }
        buffer = chunk.subarray(0, read);
      } finally {
        await handle.close();
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        return { entries: [], end: null, nextOffset: offset, torn: false };
      }
      return { entries: [], end: null, nextOffset: offset, torn: true };
    }
    const decoded = decodeJournalRecords(buffer, 0);
    return {
      entries: decoded.entries,
      end: decoded.end,
      nextOffset: offset + decoded.nextOffset,
      torn: decoded.nextOffset < buffer.length,
    };
  }

  /**
   * Replay this journal's events past `afterSeq`, then follow live appends
   * until it closes.
   *
   * The final event is `end` for a clean close and `incomplete` whenever the
   * prefix cannot be trusted — a bound hit, a lost write, a torn final record,
   * or a close that never wrote an end marker. An `incomplete` journal yields
   * it as soon as the flag is set rather than after the turn ends: nothing more
   * can be recorded, so a follower that waited would hang for the rest of the
   * turn to learn the prefix is untrustworthy.
   */
  async *iterEvents(
    afterSeq: number,
    signal?: AbortSignal,
  ): AsyncGenerator<JournalStreamEvent, void, void> {
    let sent = afterSeq;
    let offset = 0;
    // The end record may be decoded in the same read that still had chunks to
    // hand the client, so carry it to the terminal decision below.
    let lastEnd: JournalEndRecord | null = null;
    while (true) {
      // close() flips `closed` before the writer drains; a reader must judge
      // the *settled* file, or it would report the last queued chunk as
      // missing. Bounded, so a writer that never settles is read as-is (and
      // then reported incomplete).
      if (this.closed && this.writer !== null) {
        await this.waitDone(this.closeDrainTimeoutMs);
      }
      const version = this.changeVersion;
      const { entries, end, nextOffset, torn } = await this.readFrom(offset);
      offset = nextOffset;
      if (end !== null) lastEnd = end;
      let emitted = 0;
      for (const entry of entries) {
        if (entry.seq <= sent) continue;
        sent = entry.seq;
        emitted += 1;
        yield {
          event: "chunk",
          data: {
            generation: this.generation,
            seq: entry.seq,
            chunk: entry.chunk,
          },
        };
      }
      if (emitted > 0 || version !== this.changeVersion) {
        // The journal moved while this read was in flight — a chunk or the end
        // marker was appended, or `incomplete` was set. Re-read so
        // `end`/`torn`/`incomplete` describe the file as it is now instead of a
        // state this read raced past.
        continue;
      }
      if (this.incomplete) {
        yield {
          event: "incomplete",
          data: {
            generation: this.generation,
            last_seq: sent,
            reason: "incomplete",
          },
        };
        return;
      }
      if (this.closed) {
        yield torn || lastEnd === null
          ? {
              event: "incomplete",
              data: {
                generation: this.generation,
                last_seq: sent,
                reason: "incomplete",
              },
            }
          : {
              event: "end",
              data: {
                generation: this.generation,
                last_seq: sent,
                reason: lastEnd.reason,
              },
            };
        return;
      }
      if (signal?.aborted) return;
      await this.waitForChange(this.changeVersion, signal);
      if (signal?.aborted) return;
    }
  }

  /**
   * Close the journal and wait a bounded time for the writer to settle.
   *
   * On timeout the journal is marked incomplete and close returns: the writer
   * keeps its reservation until it settles, and finalizes then. Idempotent.
   */
  async close(reason: JournalReason): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    if (!this.incomplete) this.reason = reason;
    this.signal.notify();
    this.bumpChange();
    if (this.writer !== null) {
      const settled = await this.waitDone(this.closeDrainTimeoutMs);
      if (!settled) {
        this.markIncomplete();
        return;
      }
      return;
    }
    // No writer: bound the finalization too, so retention removal on slow
    // storage cannot stall the turn. The outstanding cleanup keeps its charge
    // until it settles.
    const task = this.finalize();
    void task.catch(() => {});
    await bounded(task, this.closeDrainTimeoutMs);
  }

  /** Resolve once the writer has settled, or false on timeout. */
  private async waitDone(timeoutMs: number): Promise<boolean> {
    if (this.writer === null) return true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<boolean>((resolve) => {
      timer = setTimeout(() => resolve(false), timeoutMs);
    });
    try {
      return await Promise.race([this.done.then(() => true), timeout]);
    } finally {
      if (timer !== undefined) clearTimeout(timer);
    }
  }

  /**
   * Select the body for removal and remove the file.
   *
   * `retained` flips immediately so the body is never selected twice while the
   * removal is in flight. Returns true once the file is confirmed absent; on
   * failure the charge is kept and `retained` is restored, so a later admission
   * can retry instead of stranding the body.
   */
  async evict(): Promise<boolean> {
    if (this.evicting) return false;
    if (!this.retained) return true;
    this.evicting = true;
    this.retained = false;
    try {
      await rm(this.filePath, { force: true });
      return true;
    } catch (error) {
      logger.warn(
        `Stream journal eviction failed for execution ` +
          `${this.executionId} generation ${this.generation}: ${String(error)}`,
      );
      this.retained = true;
      return false;
    } finally {
      this.evicting = false;
    }
  }

  private async runWriter(): Promise<void> {
    try {
      try {
        await mkdir(this.dir, { recursive: true });
      } catch (error) {
        // The journal must never fail the turn: stop recording and let the
        // turn continue. The journal reports incomplete.
        logger.debug(
          `Stream journal dir unavailable for execution ${this.executionId}: ` +
            `${String(error)}`,
        );
        this.markIncomplete();
        return;
      }
      while (true) {
        while (this.queue.length > 0) {
          const frame = this.queue.shift();
          if (frame === undefined) break;
          try {
            await appendFile(this.filePath, frame.data);
          } catch (error) {
            logger.debug(
              `Stream journal write failed for execution ${this.executionId}: ` +
                `${String(error)}`,
            );
            this.writeFailed = true;
            this.markIncomplete();
            return;
          }
          this.bytes += frame.data.length;
          this.lastSeq = frame.seq;
          this.bumpChange();
        }
        if (this.writeFailed) return;
        if (this.closed) {
          await this.writeEndMarker();
          return;
        }
        await this.signal.wait();
      }
    } finally {
      // The writer will consume nothing more: drop any payload it can no longer
      // write so a retired journal does not pin the memory.
      this.queue.length = 0;
      await this.finalize();
      this.resolveDone();
      // Last, so a follower that wakes on it sees the settled state.
      this.bumpChange();
    }
  }

  private async writeEndMarker(): Promise<void> {
    const record: JournalEndRecord = {
      kind: "end",
      generation: this.generation,
      last_seq: this.lastSeq,
      reason: this.reason ?? "completed",
    };
    const data = encodeJournalRecord(
      Buffer.from(JSON.stringify(record), "utf8"),
    );
    try {
      await appendFile(this.filePath, data);
    } catch (error) {
      logger.debug(
        `Stream journal end marker write failed for execution ` +
          `${this.executionId}: ${String(error)}`,
      );
      this.writeFailed = true;
      this.markIncomplete();
      return;
    }
    this.bytes += data.length;
  }

  /**
   * Release the unused reservation and register the body for retention.
   *
   * Runs exactly once, after the writer has settled, so a journal that is still
   * draining keeps its reservation and is never evicted.
   */
  private async finalize(): Promise<void> {
    if (this.finalized) return;
    this.finalized = true;
    await this.onFinalize(this);
  }
}

/** Options for {@link StreamJournalManager}. */
export interface StreamJournalManagerOptions {
  /** Journal directory; defaults to the deployment env or the scratch path. */
  dir?: string;
  maxBytes?: number;
  maxChunks?: number;
  maxQueueChunks?: number;
  runnerMaxBytes?: number;
  maxClosedPerSession?: number;
  endMarkerReserve?: number;
  closeDrainTimeoutMs?: number;
  admissionTimeoutMs?: number;
  maxReadersPerJournal?: number;
  maxReadersPerRunner?: number;
}

/**
 * Owns every journal for one runner and enforces the runner-wide bounds.
 *
 * `totalBytes` counts each open or draining journal's reservation plus each
 * settled journal's written size, so admitted capacity never exceeds the runner
 * budget and no body is deleted while it is still being written.
 */
export class StreamJournalManager {
  readonly dir: string;
  private readonly limits: JournalLimits;
  private readonly runnerMaxBytes: number;
  private readonly maxClosedPerSession: number;
  private readonly endMarkerReserve: number;
  private readonly closeDrainTimeoutMs: number;
  private readonly admissionTimeoutMs: number;
  private readonly maxReadersPerJournal: number;
  private readonly maxReadersPerRunner: number;
  private readonly journals = new Map<string, Map<number, StreamJournal>>();
  /** Creation order, oldest first, for oldest-first eviction. */
  private readonly order: StreamJournal[] = [];
  /**
   * In-flight admissions per `(executionId, generation)`, so a duplicate open
   * of the same tuple awaits the original decision instead of observing a
   * journal that is not admitted yet.
   */
  private readonly admissions = new Map<string, Promise<StreamJournal>>();
  private totalBytes = 0;
  private sequence = 0;
  /** Live-follow reader slots, per journal and per runner. */
  private runnerReaders = 0;
  private readonly journalReaders = new Map<string, number>();

  constructor(options: StreamJournalManagerOptions = {}) {
    this.dir =
      options.dir ??
      (process.env[STREAM_JOURNAL_DIR_ENV]?.trim() ||
        STREAM_JOURNAL_DEFAULT_DIR);
    this.limits = {
      maxBytes: options.maxBytes ?? STREAM_JOURNAL_MAX_BYTES,
      maxChunks: options.maxChunks ?? STREAM_JOURNAL_MAX_CHUNKS,
      maxQueueChunks: options.maxQueueChunks ?? STREAM_JOURNAL_QUEUE_MAX_CHUNKS,
    };
    this.runnerMaxBytes =
      options.runnerMaxBytes ?? STREAM_JOURNAL_RUNNER_MAX_BYTES;
    this.maxClosedPerSession =
      options.maxClosedPerSession ?? STREAM_JOURNAL_MAX_CLOSED_PER_SESSION;
    this.endMarkerReserve =
      options.endMarkerReserve ?? STREAM_JOURNAL_END_MARKER_RESERVE;
    this.closeDrainTimeoutMs =
      options.closeDrainTimeoutMs ?? STREAM_JOURNAL_CLOSE_DRAIN_TIMEOUT_MS;
    this.admissionTimeoutMs =
      options.admissionTimeoutMs ?? STREAM_JOURNAL_ADMISSION_TIMEOUT_MS;
    this.maxReadersPerJournal =
      options.maxReadersPerJournal ?? STREAM_JOURNAL_MAX_READERS_PER_JOURNAL;
    this.maxReadersPerRunner =
      options.maxReadersPerRunner ?? STREAM_JOURNAL_MAX_READERS_PER_RUNNER;
  }

  /** Capacity one open journal reserves against the runner budget. */
  get journalReservation(): number {
    return this.limits.maxBytes + this.endMarkerReserve;
  }

  /**
   * Open (or return the existing) journal for one lifecycle. The caller opens
   * one per `/execute` that negotiated a journal; a redelivered dispatch for
   * the same generation reuses it.
   *
   * A duplicate open while the first admission is still reclaiming capacity
   * awaits that same admission and receives the same journal, so neither caller
   * can close a journal whose writer has not been started yet.
   */
  async openJournal(
    executionId: string,
    sessionId: string,
    generation: number,
  ): Promise<StreamJournal> {
    // The in-flight admission is authoritative: the journal is published into
    // `journals` before the writer starts, so a duplicate must consult the
    // admission before the published journal.
    const key = this.admissionKey(executionId, generation);
    const inflight = this.admissions.get(key);
    if (inflight !== undefined) return inflight;
    const existing = this.journals.get(executionId)?.get(generation);
    if (existing !== undefined) return existing;
    const admission = this.admitJournal(executionId, sessionId, generation);
    this.admissions.set(key, admission);
    try {
      return await admission;
    } finally {
      this.admissions.delete(key);
    }
  }

  private async admitJournal(
    executionId: string,
    sessionId: string,
    generation: number,
  ): Promise<StreamJournal> {
    const existing = this.journals.get(executionId)?.get(generation);
    if (existing !== undefined) return existing;

    const journal = new StreamJournal(
      executionId,
      sessionId,
      generation,
      this.dir,
      this.limits,
      this.closeDrainTimeoutMs,
      (target) => this.finalizeJournal(target),
    );
    let byGeneration = this.journals.get(executionId);
    if (byGeneration === undefined) {
      byGeneration = new Map();
      this.journals.set(executionId, byGeneration);
    }
    byGeneration.set(generation, journal);
    journal.order = this.sequence;
    this.sequence += 1;
    this.order.push(journal);

    // Make room for the new journal's whole potential size by evicting settled
    // closed journals oldest-first. Bounded: if scratch storage is too slow to
    // reclaim in time, the new journal starts incomplete and the turn proceeds;
    // the outstanding removal keeps its charge until it settles. A draining
    // journal is never evicted.
    try {
      await this.evictClosedToFitBounded(this.journalReservation);
    } catch (error) {
      // The admission never completed: withdraw the published journal so no
      // reader can follow a body nothing will ever write or close.
      byGeneration.delete(generation);
      const index = this.order.indexOf(journal);
      if (index >= 0) this.order.splice(index, 1);
      throw error;
    }
    if (this.totalBytes + this.journalReservation > this.runnerMaxBytes) {
      journal.startIncomplete();
      return journal;
    }
    journal.reserved = this.journalReservation;
    this.totalBytes += this.journalReservation;
    journal.start();
    logger.debug(
      `Opened stream journal for execution ${executionId} generation ` +
        `${generation} at ${journal.filePath}`,
    );
    return journal;
  }

  /** Record one chunk on the journal for `(executionId, generation)`. */
  append(
    executionId: string,
    generation: number,
    entry: JournalChunkEntry,
  ): void {
    const journal = this.journals.get(executionId)?.get(generation);
    if (journal === undefined) return;
    journal.append(entry);
  }

  /** Close a journal with an end reason. No-op for an unknown generation. */
  async closeJournal(
    executionId: string,
    generation: number,
    reason: JournalReason,
  ): Promise<void> {
    const inflight = this.admissions.get(
      this.admissionKey(executionId, generation),
    );
    if (inflight !== undefined) {
      // Respect the pending admission: never finalize a journal whose writer
      // has not been started yet.
      let journal: StreamJournal;
      try {
        journal = await inflight;
      } catch {
        return;
      }
      await journal.close(reason);
      return;
    }
    const journal = this.journals.get(executionId)?.get(generation);
    if (journal === undefined) return;
    await journal.close(reason);
  }

  getJournal(executionId: string, generation: number): StreamJournal | null {
    // A journal is not exposed until its admission has decided: before that it
    // has no writer and no reservation.
    if (this.admissions.has(this.admissionKey(executionId, generation))) {
      return null;
    }
    return this.journals.get(executionId)?.get(generation) ?? null;
  }

  /** The per-execution index, or null for an execution never opened. */
  getIndex(executionId: string): JournalIndex | null {
    const byGeneration = this.journals.get(executionId);
    if (byGeneration === undefined || byGeneration.size === 0) return null;
    const generations = [...byGeneration.values()]
      .sort((a, b) => a.generation - b.generation)
      .map((journal) => ({
        generation: journal.generation,
        last_seq: journal.lastSeq,
        reason: journal.incomplete ? ("incomplete" as const) : journal.reason,
        retained: journal.retained,
      }));
    return { execution_id: executionId, generations };
  }

  /**
   * Reserve a live-follow reader slot for one generation.
   *
   * Returns a release function, or null when the per-journal or per-runner cap
   * is already taken. The caller must release once when the follow ends,
   * including on a client disconnect; the returned function is idempotent so a
   * disconnect path can call it twice.
   */
  acquireReader(executionId: string, generation: number): (() => void) | null {
    if (this.runnerReaders >= this.maxReadersPerRunner) return null;
    const key = this.admissionKey(executionId, generation);
    const held = this.journalReaders.get(key) ?? 0;
    if (held >= this.maxReadersPerJournal) return null;
    this.journalReaders.set(key, held + 1);
    this.runnerReaders += 1;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.runnerReaders = Math.max(0, this.runnerReaders - 1);
      const remaining = (this.journalReaders.get(key) ?? 1) - 1;
      if (remaining <= 0) this.journalReaders.delete(key);
      else this.journalReaders.set(key, remaining);
    };
  }

  /** Close every open journal as aborted and drop all state (shutdown). */
  async shutdown(): Promise<void> {
    // Let any pending admission settle before closing, so no writer starts
    // after shutdown.
    await Promise.all(
      [...this.admissions.values()].map((admission) =>
        admission.then(
          () => undefined,
          () => undefined,
        ),
      ),
    );
    await Promise.all(this.order.map((journal) => journal.close("aborted")));
    this.journals.clear();
    this.order.length = 0;
    this.totalBytes = 0;
  }

  private admissionKey(executionId: string, generation: number): string {
    return `${executionId}\u0000${generation}`;
  }

  /**
   * Release a settled journal's unused reservation and register its body for
   * retention. Called once, by the journal, after its writer has settled.
   */
  private async finalizeJournal(journal: StreamJournal): Promise<void> {
    if (!journal.writeFailed) {
      this.totalBytes = Math.max(
        0,
        this.totalBytes - (journal.reserved - journal.bytes),
      );
      journal.reserved = journal.bytes;
    }
    // On an uncertain append the file may hold a torn prefix longer than the
    // written prefix; keep the reservation until the body is deleted.
    await this.enforceClosedLimit(journal.sessionId);
  }

  private async enforceClosedLimit(sessionId: string): Promise<void> {
    const closed = this.order
      .filter(
        (journal) =>
          journal.sessionId === sessionId &&
          journal.finalized &&
          journal.retained,
      )
      .sort((a, b) => a.order - b.order);
    const excess = closed.length - this.maxClosedPerSession;
    for (let index = 0; index < excess; index += 1) {
      const victim = closed[index];
      if (victim !== undefined && !(await this.evict(victim))) {
        // A failed removal keeps the body and its charge; stop rather than
        // spin, and let a later close retry.
        break;
      }
    }
  }

  /**
   * Evict settled closed journals, oldest first, until `need` more bytes fit
   * under the runner budget. A draining journal is never a candidate, so this
   * stops as soon as the only journals left are open or unsettled. The caller
   * decides what to do if the budget still cannot fit `need`.
   */
  private async evictClosedToFit(need: number): Promise<void> {
    while (this.totalBytes + need > this.runnerMaxBytes) {
      const victim = this.oldestClosedRetained();
      if (victim === null) return;
      if (!(await this.evict(victim))) {
        // The removal failed; the body stays charged and selectable for a later
        // retry. Stop rather than select it again immediately.
        return;
      }
    }
  }

  /**
   * Run reclamation with a deadline so admission cannot stall a turn.
   *
   * On timeout the reclamation keeps running in the background: the outstanding
   * removal holds its charge until it settles, and the caller starts the new
   * journal incomplete.
   */
  private async evictClosedToFitBounded(need: number): Promise<void> {
    const task = this.evictClosedToFit(need);
    void task.catch(() => {});
    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<void>((resolve) => {
      timer = setTimeout(resolve, this.admissionTimeoutMs);
    });
    try {
      await Promise.race([task, timeout]);
    } finally {
      if (timer !== undefined) clearTimeout(timer);
    }
  }

  private oldestClosedRetained(): StreamJournal | null {
    for (const journal of this.order) {
      if (journal.finalized && journal.retained) return journal;
    }
    return null;
  }

  private async evict(journal: StreamJournal): Promise<boolean> {
    const gone = await journal.evict();
    if (!gone) {
      // The file is still on disk: keep the charge and leave it selectable for
      // a later retry.
      return false;
    }
    this.totalBytes = Math.max(0, this.totalBytes - journal.reserved);
    journal.reserved = 0;
    return true;
  }
}
