/**
 * Ordered, coalescing, budget-bounded delivery of a turn's stream chunks.
 *
 * TypeScript counterpart of Python's `agent_engine_runner_shared.server.chunk_delivery`.
 * When the engine negotiates durable acknowledgement (it stamps
 * `durable_ack_enabled` on the execute request because the workspace advertised
 * the `durable_ack` capability), a 200 response to `/stream/chunk` means every
 * chunk in the request is durably persisted — a delivery-identity dedup hit
 * counts as persisted. This module implements the runner half of that contract:
 *
 * - chunks are posted **in production order** through one poster, so durability
 *   order matches production order even across failed requests and takeovers;
 * - chunks produced while a request is in flight are **coalesced** into the
 *   next request (bounded batch), so a slow ack does not add a POST per token;
 * - a batch is **retried unchanged** until the engine acknowledges it or the
 *   turn's budget is exhausted — never resent after a 200, never dropped
 *   silently;
 * - the waiting queue is **bounded**, and producers block while it is full, so
 *   memory stays bounded at any token rate; a failed sender wakes blocked
 *   producers with the failure;
 * - a failed sender is terminal for the turn: the settlement paths refuse to
 *   present the turn as complete.
 *
 * **Journal mode** (`journalMode: true`, set when the engine asked for the
 * runner's stream journal) keeps the poster and its ordering guarantees but
 * reverses the durability obligations above, because the journal — not live
 * delivery — is the durable copy of the turn:
 *
 * - producers never block and never fail the turn: a full queue drops the chunk
 *   from live delivery (the journal keeps it);
 * - an exhausted retry budget drops the chunk and delivery continues with the
 *   next one, instead of failing the sender;
 * - settlement does not wait for delivery to drain, so an engine outage that
 *   exhausts live delivery never turns a successful execution into a failed one.
 *
 * Journal mode also posts one chunk per request in the legacy raw shape: a
 * journal dispatch never negotiates durable acknowledgement, and the engine
 * parses a batch envelope only under that negotiation, so a batch would be
 * ingested as a single unknown frame. One chunk in flight at a time is also all
 * live delivery needs — the journal supplies everything older.
 *
 * The not-negotiated path does not construct a sender and is untouched.
 */

import { getLogger } from "../logger.js";
import { postWithRetriesUntil, RetryBudgetExhausted } from "./http_retry.js";

const logger = getLogger("agent_engine_runner_shared.server.chunk_delivery");

// One request maps onto at most one engine publication batch: the engine's
// own per-batch bound is 32 chunks, and a request over it is rejected.
export const DURABLE_ACK_MAX_BATCH_CHUNKS = 32;
// The engine caps a request body at 1 MiB and rejects an oversized batch, so
// a batch is also bounded by its serialized request size. The estimate is
// exact: the payload is compact JSON with UTF-8 encoding, which is what the
// HTTP client puts on the wire.
export const DURABLE_ACK_MAX_BATCH_BYTES = 1 << 20;
// Bounded waiting memory. The queue cap plus one in-flight batch keeps the
// total unacknowledged window at or below the engine's resume-dedup budget
// (its FIFO dedup set is capped at 1024, and only unacknowledged chunks are
// ever resent): 96 queued + 32 in flight = 128.
export const DURABLE_ACK_MAX_QUEUED_CHUNKS = 96;
// Retry backoff for the in-order batch loop. The cap keeps the settlement
// drain's wake-up latency bounded when the engine is slow or down.
export const DURABLE_ACK_BASE_DELAY_MS = 100;
export const DURABLE_ACK_MAX_DELAY_MS = 2000;

const textEncoder = new TextEncoder();

/** Wire size of one JSON value in the request body (compact, UTF-8). */
function serializedSize(value: unknown): number {
  return textEncoder.encode(JSON.stringify(value)).length;
}

/** The turn's chunk stream could not be acknowledged; never present it complete. */
export class DurableAckError extends Error {}

/**
 * A versioned wake-up gate: waiters re-check their condition after every
 * notification, and a notification that lands between a condition check and
 * the wait is never missed (the version read happens first). A wait also
 * resolves on the cancellation signal so an aborted execution never leaves a
 * producer or settler parked.
 */
class Gate {
  private version = 0;
  private resolvers = new Set<() => void>();

  get current(): number {
    return this.version;
  }

  notify(): void {
    this.version += 1;
    const pending = [...this.resolvers];
    this.resolvers.clear();
    for (const resolve of pending) resolve();
  }

  wait(
    seenVersion: number,
    timeoutMs?: number,
    signal?: AbortSignal | null,
  ): Promise<void> {
    if (this.version !== seenVersion) return Promise.resolve();
    return new Promise((resolve) => {
      let timer: ReturnType<typeof setTimeout> | undefined;
      const finish = (): void => {
        if (timer !== undefined) clearTimeout(timer);
        signal?.removeEventListener("abort", finish);
        this.resolvers.delete(finish);
        resolve();
      };
      if (timeoutMs !== undefined) {
        timer = setTimeout(finish, Math.max(0, timeoutMs));
      }
      signal?.addEventListener("abort", finish, { once: true });
      this.resolvers.add(finish);
    });
  }
}

/** A chunk as built by the AER (`sendStreamChunk`), carried verbatim in a batch. */
export type StreamChunkPayload = Record<string, unknown>;

/** Options for {@link DurableChunkSender}. */
export interface DurableChunkSenderOptions {
  executionId: string;
  chunkUrl: string;
  /** Monotonic deadline (`performance.now()` based) bounding all retries. */
  deadlineMs: number;
  /** Full owner URL for the chunk route, resolved per POST. */
  ownerUrlGetter?: (() => string | null) | null;
  onOwnerFailure?: (() => void) | null;
  /**
   * Drain/Stop only: cancels the poster, its in-flight attempts, and the
   * settlement waits promptly. A whole-turn timeout must not appear here, so
   * already accepted output can still use the bounded error drain.
   */
  cancellationSignal?: AbortSignal | null;
  /**
   * Any execution abort (including the whole-turn timeout): releases a
   * producer blocked on the queue cap so the existing timeout owner can run.
   * Surfaced as its own reason — never converted into a durable-ack failure.
   */
  productionSignal?: AbortSignal | null;
  maxBatchChunks?: number;
  maxBatchBytes?: number;
  maxQueuedChunks?: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
  /**
   * Journal mode: live delivery is best effort because the journal, not live
   * delivery, is the durable copy of the turn. `send` never blocks or throws
   * (a full queue drops the chunk), a failed batch is dropped and delivery
   * continues, and `settle` returns without waiting for the drain.
   */
  journalMode?: boolean;
}

/**
 * One ordered poster per execution for the negotiated durable-ack path.
 *
 * `send` enqueues a chunk (blocking while the bounded queue is full, which is
 * the backpressure that bounds memory); the poster drains the queue in order,
 * coalescing up to `maxBatchChunks` per request. `settle` waits until
 * everything enqueued so far is acknowledged; the terminal/settlement paths
 * call it before presenting the turn, and a throw from it means the turn must
 * be failed instead.
 *
 * In journal mode (`journalMode: true`) live delivery is best effort: `send`
 * never blocks or throws (a full queue drops the chunk), a failed chunk is
 * dropped and delivery continues, `settle` returns without waiting for the
 * drain, and each request carries exactly one chunk in the legacy raw shape.
 */
export class DurableChunkSender {
  /** True when live delivery is best effort (the journal is the durable copy). */
  readonly journalMode: boolean;
  private readonly executionId: string;
  private readonly chunkUrl: string;
  private readonly deadlineMs: number;
  private readonly ownerUrlGetter: (() => string | null) | null;
  private readonly onOwnerFailure: (() => void) | null;
  private readonly cancellationSignal: AbortSignal | null;
  private readonly productionSignal: AbortSignal | null;
  private readonly maxBatchChunks: number;
  private readonly maxBatchBytes: number;
  private readonly maxQueuedChunks: number;
  private readonly baseDelayMs: number;
  private readonly maxDelayMs: number;

  /** Envelope size plus each element's serialized size plus commas. */
  private readonly envelopeBytes: number;
  private pending: Array<[StreamChunkPayload, number]> = [];
  private enqueued = 0;
  private acked = 0;
  private droppedCount = 0;
  private failure: DurableAckError | null = null;
  private closed = false;
  private readonly gate = new Gate();
  private readonly closeController = new AbortController();
  private readonly poster: Promise<void>;

  constructor(options: DurableChunkSenderOptions) {
    this.executionId = options.executionId;
    this.chunkUrl = options.chunkUrl;
    this.deadlineMs = options.deadlineMs;
    this.ownerUrlGetter = options.ownerUrlGetter ?? null;
    this.onOwnerFailure = options.onOwnerFailure ?? null;
    this.cancellationSignal = options.cancellationSignal ?? null;
    this.productionSignal = options.productionSignal ?? null;
    this.maxBatchChunks =
      options.maxBatchChunks ?? DURABLE_ACK_MAX_BATCH_CHUNKS;
    this.maxBatchBytes = options.maxBatchBytes ?? DURABLE_ACK_MAX_BATCH_BYTES;
    this.maxQueuedChunks =
      options.maxQueuedChunks ?? DURABLE_ACK_MAX_QUEUED_CHUNKS;
    this.baseDelayMs = options.baseDelayMs ?? DURABLE_ACK_BASE_DELAY_MS;
    this.maxDelayMs = options.maxDelayMs ?? DURABLE_ACK_MAX_DELAY_MS;
    this.journalMode = options.journalMode ?? false;
    this.envelopeBytes = serializedSize({
      execution_id: this.executionId,
      chunks: [],
    });
    this.poster = this.run();
  }

  get failed(): DurableAckError | null {
    return this.failure;
  }

  /** Chunks abandoned by journal-mode live delivery (kept in the journal). */
  get dropped(): number {
    return this.droppedCount;
  }

  /**
   * Queue one chunk for ordered delivery.
   *
   * Resolves once the chunk is queued — not once it is acknowledged — so a
   * POST in flight does not stall production; it blocks only while the bounded
   * queue is full (backpressure). Throws {@link DurableAckError} when the
   * sender has failed or been closed: the chunk can no longer be delivered,
   * and the caller must fail the turn rather than drop it.
   */
  async send(chunk: StreamChunkPayload): Promise<void> {
    if (this.journalMode) {
      // Journal mode: live delivery is best effort. Never block a producer and
      // never fail the turn — a full queue, a failed poster, or a closed
      // sender drops the chunk (the journal keeps it).
      if (
        this.pending.length >= this.maxQueuedChunks ||
        this.failure !== null ||
        this.closed
      ) {
        this.drop("queue full, failed, or closed");
        return;
      }
      this.pending.push([chunk, serializedSize(chunk)]);
      this.enqueued += 1;
      this.gate.notify();
      return;
    }
    while (true) {
      const version = this.gate.current;
      this.raiseIfUnusable();
      // Production observes any execution abort (a Stop or the whole-turn
      // timeout) so the handler's existing owner can run; the poster and
      // settlement waits observe only a Stop.
      this.raiseIfAborted(this.productionSignal);
      if (this.pending.length < this.maxQueuedChunks) break;
      await this.gate.wait(version, undefined, this.productionSignal);
    }
    this.pending.push([chunk, serializedSize(chunk)]);
    this.enqueued += 1;
    this.gate.notify();
  }

  /**
   * Wait until every chunk enqueued so far is acknowledged.
   *
   * `timeoutMs` bounds the wait below the sender's turn budget; the error
   * paths use it so a failing turn's terminal is not held on the network.
   * Throws {@link DurableAckError} when the sender failed or the budget ran
   * out with chunks still unacknowledged — the turn must not be presented as
   * complete.
   *
   * In journal mode settlement never waits for the poster to drain (and so
   * never throws): an engine outage that exhausts live delivery must not turn
   * a successful execution into a failed one, because the journal already
   * holds the turn's chunks.
   */
  async settle(timeoutMs?: number): Promise<void> {
    if (this.journalMode) return;
    const deadline =
      timeoutMs === undefined
        ? this.deadlineMs
        : Math.min(this.deadlineMs, performance.now() + timeoutMs);
    while (true) {
      const version = this.gate.current;
      this.raiseIfFailed();
      this.raiseIfAborted(this.cancellationSignal);
      if (this.acked >= this.enqueued) return;
      if (this.closed) throw this.raiseIncomplete();
      const remainingMs = deadline - performance.now();
      if (remainingMs <= 0) throw this.raiseIncomplete();
      await this.gate.wait(version, remainingMs, this.cancellationSignal);
    }
  }

  /**
   * Stop the poster and abandon anything still queued.
   *
   * Called from the execution teardown: the terminal paths have already
   * settled (or failed the turn), so unacknowledged chunks can no longer be
   * presented.
   */
  async close(): Promise<void> {
    this.closed = true;
    this.gate.notify();
    this.closeController.abort();
    await this.poster.catch(() => {
      // The poster observes the abort as a close, not a failure.
    });
  }

  private raiseIfFailed(): void {
    if (this.failure !== null) throw this.failure;
  }

  private raiseIfAborted(signal: AbortSignal | null): void {
    if (signal?.aborted === true) {
      // Surface the execution's own cancellation reason: the existing drain
      // and cancel handling owns the settlement, so this must never look like
      // a durable-ack failure.
      throw signal.reason;
    }
  }

  private raiseIfUnusable(): void {
    this.raiseIfFailed();
    if (this.closed) {
      throw new DurableAckError(
        `durable chunk sender for execution ${this.executionId} is closed`,
      );
    }
  }

  private incompleteError(): DurableAckError {
    return new DurableAckError(
      `durable chunk acknowledgement incomplete for execution ` +
        `${this.executionId}: ${this.enqueued - this.acked} chunk(s) unacknowledged`,
    );
  }

  /**
   * Build the incomplete error, poisoning the sender when the turn's own
   * budget (not a bounded drain timeout) is what expired: any later send must
   * fail rather than queue into a turn that can no longer be presented.
   */
  private raiseIncomplete(): DurableAckError {
    const error = this.incompleteError();
    if (performance.now() >= this.deadlineMs) this.fail(error);
    return error;
  }

  private async run(): Promise<void> {
    try {
      while (this.failure === null) {
        const batch = await this.nextBatch();
        if (batch === null) return;
        await this.deliver(batch);
      }
    } catch (error) {
      if (this.closed || this.cancellationSignal?.aborted === true) return;
      if (this.journalMode) {
        // Journal mode never fails the turn: stop live delivery and leave the
        // chunks to the journal.
        logger.warn(
          `Journal-mode live delivery for execution ${this.executionId} ` +
            `stopped after an unexpected failure; the journal keeps the ` +
            `turn's chunks: ${
              error instanceof Error ? error.message : String(error)
            }`,
        );
        return;
      }
      this.fail(
        error instanceof DurableAckError
          ? error
          : new DurableAckError(
              `durable chunk delivery for execution ${this.executionId} ` +
                `failed unexpectedly: ${error instanceof Error ? error.message : String(error)}`,
            ),
      );
    }
  }

  /**
   * Pop the next coalesced batch, or `null` once delivery is over.
   *
   * The batch is the largest ordered prefix that fits both the engine's
   * chunk-count bound and its request-size bound, so one request maps onto at
   * most one publication batch. A chunk that cannot fit the size bound even
   * alone can never be delivered: it fails the sender, or — in journal mode,
   * where live delivery is best effort — is dropped and delivery moves on to
   * the rest of the queue.
   */
  private async nextBatch(): Promise<StreamChunkPayload[] | null> {
    while (true) {
      const version = this.gate.current;
      if (this.pending.length > 0) {
        const batch: StreamChunkPayload[] = [];
        // Journal mode posts the legacy single-chunk shape. A journal dispatch
        // never negotiates durable acknowledgement, and the engine parses a
        // batch envelope only under that negotiation, so a batch would be
        // ingested as one unknown frame. One raw chunk per request keeps live
        // delivery on the shape the engine already accepts, which is what the
        // journal replaces.
        let size = this.journalMode ? 0 : this.envelopeBytes;
        const maxChunks = this.journalMode ? 1 : this.maxBatchChunks;
        while (this.pending.length > 0 && batch.length < maxChunks) {
          const head = this.pending[0];
          if (head === undefined) break;
          const [chunk, chunkBytes] = head;
          const cost = chunkBytes + (batch.length > 0 ? 1 : 0);
          if (size + cost > this.maxBatchBytes) break;
          this.pending.shift();
          batch.push(chunk);
          size += cost;
        }
        if (batch.length > 0) {
          // Popping freed queue space: wake any producer blocked on the cap.
          this.gate.notify();
          return batch;
        }
        // The head chunk alone exceeds the request cap: it can never be sent.
        const head = this.pending.shift();
        if (this.journalMode) {
          this.drop(
            `chunk of ${head?.[1] ?? 0} bytes exceeds the request size limit`,
          );
          continue;
        }
        this.fail(
          new DurableAckError(
            `chunk of ${head?.[1] ?? 0} bytes for execution ` +
              `${this.executionId} exceeds the engine's request size limit ` +
              `of ${this.maxBatchBytes} bytes`,
          ),
        );
        return null;
      }
      if (this.closed || this.failure !== null) return null;
      await this.gate.wait(version, undefined, this.cancellationSignal);
      if (this.cancellationSignal?.aborted === true) return null;
    }
  }

  /** Post one batch, retrying the same chunks in order until acknowledged. */
  private async deliver(batch: StreamChunkPayload[]): Promise<void> {
    const ownerUrl =
      this.ownerUrlGetter !== null ? this.ownerUrlGetter() : null;
    const signal = this.cancellationSignal
      ? AbortSignal.any([this.closeController.signal, this.cancellationSignal])
      : this.closeController.signal;
    try {
      await postWithRetriesUntil(
        this.chunkUrl,
        // Journal mode always carries exactly one chunk (see nextBatch) and
        // posts it in the legacy raw shape the unnegotiated engine parses.
        this.journalMode
          ? (batch[0] ?? {})
          : { execution_id: this.executionId, chunks: batch },
        {
          deadlineMs: this.deadlineMs,
          baseDelayMs: this.baseDelayMs,
          maxDelayMs: this.maxDelayMs,
          ownerUrl,
          onOwnerFailure: ownerUrl !== null ? this.onOwnerFailure : null,
          cancellationSignal: signal,
        },
      );
    } catch (error) {
      if (this.closed || this.cancellationSignal?.aborted === true) {
        // A close or an execution cancellation is not a delivery failure;
        // surface it so the existing teardown owns the settlement.
        throw error;
      }
      if (this.journalMode) {
        // Journal mode: live delivery is best effort, so an exhausted retry
        // budget (or any other failure) drops this batch and delivery moves on.
        this.drop(
          error instanceof Error ? error.message : String(error),
          batch.length,
        );
        return;
      }
      if (error instanceof RetryBudgetExhausted) {
        this.fail(
          new DurableAckError(
            `durable chunk acknowledgement timed out for execution ` +
              `${this.executionId} after ${this.acked} of ${this.enqueued} ` +
              `chunk(s) were acknowledged`,
          ),
        );
      } else {
        this.fail(
          new DurableAckError(
            `durable chunk delivery for execution ${this.executionId} failed: ` +
              `${error instanceof Error ? error.message : String(error)}`,
          ),
        );
      }
      return;
    }
    this.acked += batch.length;
    this.gate.notify();
  }

  /**
   * Abandon `count` chunk(s) from journal-mode live delivery.
   *
   * Journal mode only: the chunks are already recorded in the runner's stream
   * journal, so dropping them from live delivery costs at most a live `seq`
   * gap, never a turn.
   */
  private drop(reason: string, count = 1): void {
    this.droppedCount += count;
    logger.debug(
      `Journal-mode live delivery for execution ${this.executionId} dropped ` +
        `${count} chunk(s) (${reason}); the journal keeps them`,
    );
  }

  private fail(error: DurableAckError): void {
    if (this.failure === null) {
      this.failure = error;
      logger.warn(
        `Durable chunk delivery failed for execution ${this.executionId} with ` +
          `${this.acked} of ${this.enqueued} chunk(s) acknowledged; the turn ` +
          `must not be presented as complete: ${error.message}`,
      );
    }
    this.gate.notify();
  }
}
