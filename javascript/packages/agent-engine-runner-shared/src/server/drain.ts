/**
 * Execution-scoped drain receiver for Runner SDK servers.
 *
 * Mirrors Python's `agent_engine_runner_shared.server.drain`. Both runtimes are parallel
 * implementations of the same server architecture, so this file and its
 * Python twin must be changed together.
 *
 * When the OE durably cancels an execution it POSTs `/drain` to the Tool/AER
 * runtime owning that execution. The runtime blocks new work for the
 * execution, aborts tracked in-flight work where a signal channel exists,
 * and records an idempotent final outcome the caller retrieves by re-sending
 * the same request.
 *
 * Contract (pinned by the shared fixture in
 * `client-libraries/test-fixtures/drain/contract.json`):
 *
 * - `POST /drain {request_id, execution_id, reason, deadline_at_ms, workspace_id}`
 *   — `workspace_id` is mandatory on workspace-scoped runtimes (`APP_ID`
 *   set) and must match it exactly.
 * - `202 {"outcome": "accepted"}` while draining; `200` with the final
 *   outcome (`completed` / `timed_out` / `delivery_failed`) afterwards.
 * - `deadline_at_ms` is absolute. The runtime never extends it.
 * - `accepted` is HTTP-level acceptance only; `completed` is the only
 *   quiescence signal.
 * - An execution this runtime never served is `delivery_failed` with
 *   `reason_code=execution_not_found` — exact OE targeting does not prove a
 *   false `completed` safe, since one execution can span AER and Tool
 *   runtimes independently.
 *
 * TS-vs-Python divergence, matching the existing boundary tests: Python can
 * cancel coroutine tasks outright; a promise has no cancellation, so "notify
 * active work" here means aborting an AbortController. That reaches work
 * only where a signal channel already exists (AER execution context, LLM
 * streams). Plain tool functions receive no signal — they are tracked and
 * admission-blocked, and outliving the deadline is an honest `timed_out`.
 *
 * State is process-local by design: a restarted runtime has lost its active
 * work and answers `delivery_failed`; reconciliation across restarts belongs
 * to the caller's durable record, not to this registry.
 */

import { z } from "zod";
import type { FastifyInstance } from "fastify";
import { getLogger } from "../logger.js";

const logger = getLogger("agent_engine_runner_shared.server.drain");

/** How long a finalized drain record (and never-drained tombstone) survives
 * for idempotent retries. Must be >= the caller-side retry window. */
const DEFAULT_RECORD_TTL_S = 900;
/** Upper bound on the caller-chosen drain window; the runtime never extends
 * a deadline, and rejects ones set unreasonably far out. */
const DEFAULT_MAX_DEADLINE_MS = 60_000;

export const DrainRequestSchema = z.object({
  request_id: z
    .string()
    .min(1)
    .max(128)
    .regex(/^[A-Za-z0-9._:-]+$/),
  execution_id: z.string().min(1).max(256),
  reason: z.enum(["execution_cancelled", "rolling_restart"]),
  deadline_at_ms: z.number().int().positive(),
  // Null is an accepted spelling of omission, matching the Python twin and
  // the published OpenAPI contract (string|null). The route's truthiness
  // checks treat it as a missing scope claim either way.
  workspace_id: z.string().max(256).nullish(),
});

export type DrainRequest = z.infer<typeof DrainRequestSchema>;

export type DrainOutcome =
  | "accepted"
  | "completed"
  | "timed_out"
  | "delivery_failed";

export type DrainReasonCode =
  | "execution_not_found"
  | "deadline_exceeded"
  | "execution_draining";

export interface DrainVerdict {
  status: number;
  body: { outcome: DrainOutcome; reason_code?: DrainReasonCode };
}

/**
 * Outcome of one per-call abort (POST /interrupt/call). Unlike a drain, an
 * interrupt is surgical: the execution is never admission-latched and later
 * calls of the same execution proceed. `not_cancellable` is the honest
 * answer for tracked work with no signal channel (a plain tool function) —
 * never a claimed abandonment the runtime cannot perform.
 */
export type CallInterruptOutcome =
  | "interrupted"
  | "not_found"
  | "already_settled"
  | "not_cancellable";

/**
 * `AbortController.abort()` reason used by the per-call abort, distinct from
 * DRAIN_ABORT_REASON so error paths can tell the two apart.
 */
export const CALL_INTERRUPT_REASON = "call_interrupted";

interface DrainRecord {
  requestIds: Set<string>;
  reason: DrainRequest["reason"];
  deadlineAtMs: number;
  outcome: DrainOutcome | null; // null = still draining
  reasonCode?: DrainReasonCode;
  finalizedAtMs?: number;
}

/**
 * One step-addressable unit of in-flight work. `controller` absent means
 * track-only work the runtime cannot signal. Settled work stays mapped so a
 * late abort reads already_settled; the entry's TTL eviction reclaims it.
 */
interface StepWork {
  controller?: AbortController;
  settled: boolean;
  aborted: boolean;
}

interface ExecutionEntry {
  activeCount: number;
  controllers: Set<AbortController>;
  draining: boolean;
  drain: DrainRecord | null;
  idle: { promise: Promise<void>; resolve: () => void };
  eviction?: NodeJS.Timeout;
  /** Live or settled work by step number, for the per-call abort. */
  byStep: Map<number, StepWork>;
}

/**
 * `AbortController.abort()` reason used by the drain, so error paths can tell
 * "stopped by cancellation" apart from the execution timeout, which shares
 * the same controller in the AER.
 */
export const DRAIN_ABORT_REASON = "execution_drained";

/**
 * How long an abort that preceded its call's registration stays claimable by
 * `beginWork`. The window covers the OE-claim → in-graph-dispatch handoff; the
 * platform's dispatch claim makes a same-step collision inside it impossible.
 * Mirrors Python's `PRE_ABORT_RETENTION_S`.
 */
const PRE_ABORT_RETENTION_MS = 60_000;

function newIdle(): ExecutionEntry["idle"] {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

function drainingError(): Error & { statusCode: number } {
  return Object.assign(
    new Error("execution_draining" satisfies DrainReasonCode),
    {
      statusCode: 409,
    },
  );
}

/**
 * Reject malformed drain tunables rather than silently collapsing retention
 * or the deadline ceiling — `Number("bogus")` is NaN, which would evict
 * records immediately or disable the deadline check entirely. Mirrors
 * Python's fail-fast `_positive`.
 */
function parsePositiveNumber(
  raw: string | undefined,
  fallback: number,
  name: string,
): number {
  if (raw === undefined || raw.trim() === "") return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a positive finite number, got ${raw}`);
  }
  return value;
}

/**
 * Execution-scoped admission + active-work registry for one runtime process.
 *
 * Mutating methods are synchronous, so check-and-set is atomic on Node's
 * single thread. The registry is created per server; per-process state only.
 */
export class DrainRegistry {
  private readonly recordTtlMs: number;
  private readonly maxDeadlineMsValue: number;
  private readonly entries = new Map<string, ExecutionEntry>();
  /**
   * Aborts that arrived before their call registered, keyed by execution and
   * step, valued by expiry. The window is the OE-claim → in-graph-dispatch
   * handoff; the retention covers graph scheduling delays, not call lifetimes.
   */
  private readonly preAborted = new Map<string, number>();

  constructor(
    opts: {
      recordTtlMs?: number;
      maxDeadlineMs?: number;
      env?: NodeJS.ProcessEnv;
    } = {},
  ) {
    const env = opts.env ?? process.env;
    this.recordTtlMs =
      opts.recordTtlMs ??
      parsePositiveNumber(
        env["RUNNER_DRAIN_RECORD_TTL_S"],
        DEFAULT_RECORD_TTL_S,
        "RUNNER_DRAIN_RECORD_TTL_S",
      ) * 1000;
    this.maxDeadlineMsValue =
      opts.maxDeadlineMs ??
      parsePositiveNumber(
        env["RUNNER_DRAIN_MAX_DEADLINE_MS"],
        DEFAULT_MAX_DEADLINE_MS,
        "RUNNER_DRAIN_MAX_DEADLINE_MS",
      );
  }

  get maxDeadlineMs(): number {
    return this.maxDeadlineMsValue;
  }

  /** Reject work for a draining execution without registering it. */
  checkAdmission(executionId: string): void {
    const entry = this.entries.get(executionId);
    if (entry?.draining) throw drainingError();
  }

  /**
   * Register in-flight work for an execution, rejecting drained ones.
   *
   * `controller` is the abort handle the drain fires; omit it for work the
   * runtime can only wait out (e.g. a tool function with no signal channel).
   * `stepNumber` makes the work addressable by the per-call abort
   * (`abortCall`); work registered without it is only ever drained
   * execution-wide.
   *
   * Returns true when a per-call abort preceded this registration (the Stop
   * landed in the OE-claim → in-graph-dispatch handoff): the caller must not
   * start the body and must settle the call interrupted.
   */
  beginWork(
    executionId: string,
    controller?: AbortController,
    stepNumber?: number,
  ): boolean {
    let entry = this.entries.get(executionId);
    if (entry === undefined) {
      entry = {
        activeCount: 0,
        controllers: new Set(),
        draining: false,
        drain: null,
        idle: newIdle(),
        byStep: new Map(),
      };
      this.entries.set(executionId, entry);
    } else {
      if (entry.draining) throw drainingError();
      this.cancelEviction(entry);
      if (entry.activeCount === 0) {
        // Reviving a tombstone (e.g. a HITL resume reusing the id).
        entry.idle = newIdle();
      }
    }
    entry.activeCount += 1;
    if (controller !== undefined) entry.controllers.add(controller);
    let preAborted = false;
    if (stepNumber !== undefined) {
      preAborted = this.popPreAborted(executionId, stepNumber);
      // A step maps to one in-flight call by the platform's dispatch claim,
      // so this can only replace a settled entry.
      entry.byStep.set(stepNumber, {
        controller,
        settled: false,
        aborted: preAborted,
      });
      if (preAborted && controller !== undefined) {
        controller.abort(CALL_INTERRUPT_REASON);
      }
    }
    return preAborted;
  }

  private preAbortKey(executionId: string, stepNumber: number): string {
    return `${executionId}\n${stepNumber}`;
  }

  private popPreAborted(executionId: string, stepNumber: number): boolean {
    const key = this.preAbortKey(executionId, stepNumber);
    const expiry = this.preAborted.get(key);
    this.preAborted.delete(key);
    this.sweepPreAborted();
    return expiry !== undefined && expiry > Date.now();
  }

  private sweepPreAborted(): void {
    if (this.preAborted.size === 0) return;
    const now = Date.now();
    for (const [key, expiry] of this.preAborted) {
      if (expiry <= now) this.preAborted.delete(key);
    }
  }

  /**
   * Signal exactly one in-flight call, leaving the execution open — the
   * surgical sibling of a drain: no admission latch, no record, no deadline,
   * and later steps of the same execution proceed. Idempotent: a repeat
   * abort of the same step re-reads the same outcome.
   */
  abortCall(executionId: string, stepNumber: number): CallInterruptOutcome {
    const entry = this.entries.get(executionId);
    const work = entry?.byStep.get(stepNumber);
    if (work === undefined) {
      // The abort can race the call's registration: OE claims the call, then
      // the runtime registers only when the in-graph body starts. Retain it
      // briefly so beginWork settles the late registration as interrupted
      // instead of running a call the caller stopped.
      this.sweepPreAborted();
      this.preAborted.set(
        this.preAbortKey(executionId, stepNumber),
        Date.now() + PRE_ABORT_RETENTION_MS,
      );
      return "not_found";
    }
    if (work.settled) return "already_settled";
    // The missing-controller check precedes the repeat check: a repeat abort
    // of track-only work must keep answering not_cancellable, matching the
    // Python receiver.
    if (work.controller === undefined) return "not_cancellable";
    if (work.aborted) return "interrupted";
    work.aborted = true;
    work.controller.abort(CALL_INTERRUPT_REASON);
    return "interrupted";
  }

  /**
   * Close a call's interruptibility as its result report begins: the body
   * already produced its outcome, so a late abort must read already_settled
   * rather than claim a stop the durable record will contradict.
   */
  claimSettlement(executionId: string, stepNumber: number): void {
    const work = this.entries.get(executionId)?.byStep.get(stepNumber);
    if (work !== undefined) work.settled = true;
  }

  /**
   * Attach an abort handle to already-registered work. The AER builds its
   * execution-wide controller inside the handler, after admission — a drain
   * that landed in between aborts it here, at attach time.
   */
  attachController(executionId: string, controller: AbortController): void {
    const entry = this.entries.get(executionId);
    if (entry === undefined) return;
    entry.controllers.add(controller);
    // Abort with the drain reason: the AER shares this controller with its
    // execution timeout and tells them apart by signal.reason.
    if (entry.draining) controller.abort(DRAIN_ABORT_REASON);
  }

  endWork(
    executionId: string,
    controller?: AbortController,
    stepNumber?: number,
  ): void {
    const entry = this.entries.get(executionId);
    if (entry === undefined) return;
    if (controller !== undefined) entry.controllers.delete(controller);
    if (stepNumber !== undefined) {
      // Keep the settled step mapped: a late abort must read already_settled,
      // not not_found. The entry's TTL eviction reclaims it.
      const work = entry.byStep.get(stepNumber);
      if (work !== undefined) work.settled = true;
    }
    entry.activeCount -= 1;
    if (entry.activeCount > 0) return;
    // No work remains: drop every tracked handle (drained or not) so a
    // tombstone never retains controllers and their listeners for its TTL.
    entry.controllers.clear();
    entry.idle.resolve();
    // A pending drain's finalizer owns eviction from here; otherwise the
    // tombstone (and any finalized drain record) expires on the TTL.
    if (entry.drain === null || entry.drain.outcome !== null) {
      this.scheduleEviction(executionId, entry);
    }
  }

  /** Apply one drain request; idempotent by request_id, coalescing per execution. */
  apply(request: DrainRequest): DrainVerdict {
    const entry = this.entries.get(request.execution_id);
    if (entry?.drain != null) {
      // Same request_id or a different one: one drain per execution.
      entry.drain.requestIds.add(request.request_id);
      return this.current(entry.drain);
    }
    if (entry === undefined) {
      // Unknown execution: not applied here, but still record the outcome
      // and latch admission. A dispatch that slips past the OE gate and lands
      // after this drain must not start, and a retry of the same request must
      // see the same answer for the retention window.
      const record: DrainRecord = {
        requestIds: new Set([request.request_id]),
        reason: request.reason,
        deadlineAtMs: request.deadline_at_ms,
        outcome: "delivery_failed",
        reasonCode: "execution_not_found",
        finalizedAtMs: Date.now(),
      };
      const latched: ExecutionEntry = {
        activeCount: 0,
        controllers: new Set(),
        draining: true,
        drain: record,
        idle: newIdle(),
        byStep: new Map(),
      };
      this.entries.set(request.execution_id, latched);
      this.scheduleEviction(request.execution_id, latched);
      return this.final("delivery_failed", "execution_not_found");
    }
    if (entry.activeCount === 0) {
      // Known, already-ended execution: nothing to drain here. Latch
      // admission anyway — a drained execution never accepts work again — and
      // restart retention so the record lives a full TTL from now.
      entry.draining = true;
      const record: DrainRecord = {
        requestIds: new Set([request.request_id]),
        reason: request.reason,
        deadlineAtMs: request.deadline_at_ms,
        outcome: "completed",
        finalizedAtMs: Date.now(),
      };
      entry.drain = record;
      this.scheduleEviction(request.execution_id, entry);
      return this.final("completed");
    }
    entry.draining = true;
    const record: DrainRecord = {
      requestIds: new Set([request.request_id]),
      reason: request.reason,
      deadlineAtMs: request.deadline_at_ms,
      outcome: null,
    };
    entry.drain = record;
    for (const controller of entry.controllers) {
      controller.abort(DRAIN_ABORT_REASON);
    }
    void this.finalize(request.execution_id, entry, record);
    logger.info(
      `Drain accepted for execution ${request.execution_id} (reason=${request.reason}, active=${entry.activeCount})`,
    );
    return { status: 202, body: { outcome: "accepted" } };
  }

  private async finalize(
    executionId: string,
    entry: ExecutionEntry,
    record: DrainRecord,
  ): Promise<void> {
    const remainingMs = record.deadlineAtMs - Date.now();
    let timer: NodeJS.Timeout | undefined;
    const timedOut = await Promise.race([
      entry.idle.promise.then(() => false),
      new Promise<true>((r) => {
        timer = setTimeout(() => r(true), Math.max(0, remainingMs));
      }),
    ]);
    clearTimeout(timer);
    if (timedOut) {
      // Late completion must not overwrite the recorded timeout.
      record.outcome = "timed_out";
      record.reasonCode = "deadline_exceeded";
    } else {
      record.outcome = "completed";
    }
    record.finalizedAtMs = Date.now();
    logger.info(`Drain ${record.outcome} for execution ${executionId}`);
    if (entry.activeCount === 0) {
      this.scheduleEviction(executionId, entry);
    }
  }

  private scheduleEviction(executionId: string, entry: ExecutionEntry): void {
    this.cancelEviction(entry);
    const timer = setTimeout(
      () => this.evict(executionId, entry),
      this.recordTtlMs,
    );
    // The timer is memory hygiene, not work the process must stay alive for.
    timer.unref();
    entry.eviction = timer;
  }

  private cancelEviction(entry: ExecutionEntry): void {
    if (entry.eviction !== undefined) {
      clearTimeout(entry.eviction);
      entry.eviction = undefined;
    }
  }

  private evict(executionId: string, entry: ExecutionEntry): void {
    // A retried drain after eviction finds no entry and gets
    // execution_not_found; it never re-runs side effects.
    if (this.entries.get(executionId) === entry && entry.activeCount === 0) {
      this.entries.delete(executionId);
    }
  }

  private current(record: DrainRecord): DrainVerdict {
    if (record.outcome === null) {
      return { status: 202, body: { outcome: "accepted" } };
    }
    return this.final(record.outcome, record.reasonCode);
  }

  private final(
    outcome: DrainOutcome,
    reasonCode?: DrainReasonCode,
  ): DrainVerdict {
    return {
      status: 200,
      body:
        reasonCode === undefined
          ? { outcome }
          : { outcome, reason_code: reasonCode },
    };
  }
}

/**
 * Register `POST /drain` on a runner server.
 *
 * The bearer-auth hook already gates the path; this adds the
 * execution/workspace-scoped validation on top. Callers of this endpoint
 * must not follow redirects — the bearer token must never be forwarded to
 * another host.
 *
 * Scope check: the bearer token authenticates the caller but does not
 * establish that the named drain belongs to this runtime, so when the
 * platform scopes this process to a workspace (`APP_ID` set — every managed
 * runtime), the request must name that workspace exactly; a missing
 * `workspace_id` is a 400, a mismatch a 403. When `APP_ID` is unset (local
 * development against a single unscoped runtime) no workspace check is
 * possible and none is enforced.
 */
export function registerDrainRoute(
  app: FastifyInstance,
  registry: DrainRegistry,
  env: NodeJS.ProcessEnv = process.env,
): void {
  const workspaceId = (env["APP_ID"] ?? "").trim();

  app.post("/drain", async (request, reply) => {
    const body = DrainRequestSchema.parse(request.body);
    if (workspaceId) {
      if (!body.workspace_id) {
        return reply.code(400).send({ detail: "workspace_id required" });
      }
      if (body.workspace_id !== workspaceId) {
        return reply.code(403).send({ detail: "workspace mismatch" });
      }
    }
    if (body.deadline_at_ms > Date.now() + registry.maxDeadlineMs) {
      return reply.code(400).send({ detail: "deadline exceeds maximum" });
    }
    const verdict = registry.apply(body);
    return reply.code(verdict.status).send(verdict.body);
  });
}
