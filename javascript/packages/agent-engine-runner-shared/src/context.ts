/**
 * Per-execution context management using AsyncLocalStorage.
 *
 * Mirrors Python `agent_engine_runner_shared/context.py` which uses `contextvars.ContextVar`.
 * Each concurrent execution gets its own isolated context frame, preventing
 * cross-execution contamination of execution_id, wrapper, and OE URL.
 *
 * The single entry point is `runWithExecutionContext(store, fn)`, which runs
 * `fn` inside an isolated frame via `AsyncLocalStorage#run`. The frame is a
 * child that never mutates the parent and is torn down automatically when `fn`
 * settles — even if `fn` throws or spawns async work via setTimeout/Promise.
 *
 * Node's `AsyncLocalStorage#enterWith` (a set/clear-token style) is deliberately
 * avoided: it rebinds the current async frame in place, so the store leaks into
 * the surrounding context and cannot be reliably restored. The Python
 * `set_execution_context` / `clear_execution_context` token pair is collapsed
 * into this callback form, which is the equivalent of Python's
 * `set_execution_context(...)` + `try/finally clear_execution_context(...)`.
 */

import { AsyncLocalStorage } from "node:async_hooks";
import { randomUUID } from "node:crypto";
import { redactText } from "./error_reporting.js";
import type { ToolAuthorization } from "./models.js";

export interface ExecutionStore {
  executionId: string | null;
  wrapper: unknown | null;
  oeUrl: string | null;
  /**
   * Validated replica-specific OE owner callback URL, or null when
   * the request carried none / it failed validation. Owner-preferring
   * transports (e.g. {@link module:progress}) send here first and fall back to
   * `oeUrl` on any owner failure. Already validated against `oeUrl` by the
   * caller (see `server/owner_url.ts`), so consumers trust it as-is.
   */
  oeOwnerUrl: string | null;
  /**
   * One-way owner-failure latch: once an owner pre-attempt fails, later
   * owner-preferring posts in this execution skip the owner URL entirely
   * instead of re-paying the pre-attempt timeout on every emit. A nested
   * object for the same reason as `sessionFinish`: the reference survives
   * store spreads (e.g. {@link runWithSuspendRequestContext}), a reassigned
   * field would not. Mirrors the AER stream path's `onOwnerFailure` discard.
   */
  ownerUrlFailure: { failed: boolean };
  requestId: string | null;
  /** Platform trace ID for log correlation. */
  traceId: string | null;
  userId: string | null;
  sessionId: string | null;
  workspaceId: string | null;
  customHeaders: Record<string, string> | null;
  executionMetadata: Record<string, unknown> | null;
  /** Delegated credential injected by OE for tool execution. */
  authorization: ToolAuthorization | null;
  /** Opaque caller-provided invocation payload (the request body beyond `message`). */
  payload: Record<string, unknown> | null;
  /**
   * Execution-wide abort signal (fires on the AER execution timeout). Combined
   * into in-flight OE/LLM fetches via {@link withExecutionSignal} so a timeout
   * cancels active network I/O instead of leaving it to run. Null outside an
   * AER execution (e.g. a single Tool Pod call).
   */
  signal: AbortSignal | null;
  /**
   * Holder for a pending session-finish request. A nested object
   * rather than a bare boolean: the AER reads it from the frame that started
   * the run, while agent code deep in the call chain mutates it in place —
   * the object reference is shared across that chain, a reassigned boolean
   * field on the store would not be.
   */
  sessionFinish: { requested: boolean; closed: boolean };
  /**
   * Out-of-band suspend signal. Only the author-facing
   * `suspendPayloadToJson` writes it, so untrusted tool-result content — which
   * cannot reach this process-local frame — can never forge a HITL suspend. A
   * nested holder for the same reason as `sessionFinish`: a tool offloaded to a
   * worker thread mutates the shared object, not a reassigned store field.
   */
  suspendRequest: { payload: Record<string, unknown> | null };
}

export interface OwnerUrlFailureState {
  failed: boolean;
}

export type SessionFinishStatus =
  | "requested"
  | "already_requested"
  | "unavailable";

// Frozen so accidental mutations to the fallback don't bleed into real stores.
const EMPTY_STORE: ExecutionStore = Object.freeze({
  executionId: null,
  wrapper: null,
  oeUrl: null,
  oeOwnerUrl: null,
  ownerUrlFailure: Object.freeze({ failed: false }),
  requestId: null,
  traceId: null,
  userId: null,
  sessionId: null,
  workspaceId: null,
  customHeaders: null,
  executionMetadata: null,
  authorization: null,
  payload: null,
  signal: null,
  // closed: true — nothing outside a run can finish.
  sessionFinish: Object.freeze({ requested: false, closed: true }),
  suspendRequest: Object.freeze({ payload: null }),
});

const storage = new AsyncLocalStorage<ExecutionStore>();

function current(): ExecutionStore {
  return storage.getStore() ?? EMPTY_STORE;
}

export interface SetExecutionContextArgs {
  executionId: string;
  wrapper: unknown;
  oeUrl: string;
  /** Validated replica-specific OE owner callback URL. */
  oeOwnerUrl?: string | null;
  ownerUrlFailure?: OwnerUrlFailureState;
  requestId?: string | null;
  traceId?: string | null;
  userId?: string | null;
  sessionId?: string | null;
  workspaceId?: string | null;
  customHeaders?: Record<string, string> | null;
  authorization?: ToolAuthorization | null;
  payload?: Record<string, unknown> | null;
  signal?: AbortSignal | null;
}

function buildStore(args: SetExecutionContextArgs): ExecutionStore {
  const parent = storage.getStore();
  return {
    executionId: args.executionId,
    wrapper: args.wrapper,
    oeUrl: args.oeUrl,
    oeOwnerUrl: args.oeOwnerUrl ?? null,
    ownerUrlFailure:
      args.ownerUrlFailure ??
      (parent?.executionId === args.executionId
        ? parent.ownerUrlFailure
        : { failed: false }),
    traceId: args.traceId ?? null,
    requestId:
      args.requestId || `req-${randomUUID().replace(/-/g, "").slice(0, 12)}`,
    userId: args.userId ?? null,
    sessionId: args.sessionId ?? null,
    workspaceId: args.workspaceId ?? null,
    customHeaders: args.customHeaders ?? null,
    executionMetadata: {},
    authorization: args.authorization ?? null,
    payload: args.payload ?? null,
    signal: args.signal ?? null,
    sessionFinish: { requested: false, closed: false },
    suspendRequest: { payload: null },
  };
}

/**
 * Run `fn` with an isolated execution context frame.
 *
 * `storage.run` creates a child frame that never mutates the parent, so cleanup
 * is automatic even if `fn` throws or spawns work via setTimeout/Promise. This
 * is the equivalent of Python's `set_execution_context(...)` followed by a
 * try/finally `clear_execution_context(...)`.
 *
 * @example
 *   await runWithExecutionContext({ executionId, wrapper, oeUrl, userId }, async () => {
 *     await runAgentLogic()  // getCurrentUserId() works anywhere in this call chain
 *   })
 */
export function runWithExecutionContext<T>(
  args: SetExecutionContextArgs,
  fn: () => T,
): T {
  return storage.run(buildStore(args), fn);
}

// =========================================================================
// Accessors — read-only views of the current store
// =========================================================================

export function getCurrentExecutionId(): string | null {
  return current().executionId;
}

export function getCurrentWrapper(): unknown | null {
  return current().wrapper;
}

export function getCurrentOeUrl(): string | null {
  return current().oeUrl;
}

/**
 * Validated replica-specific OE owner callback URL for the current execution,
 * or null when none was forwarded or a previous owner pre-attempt already
 * failed (see {@link reportOeOwnerUrlFailure}). Owner-preferring transports
 * send here first and fall back to {@link getCurrentOeUrl} on any owner
 * failure.
 */
export function getCurrentOeOwnerUrl(): string | null {
  const store = current();
  return store.ownerUrlFailure.failed ? null : store.oeOwnerUrl;
}

/**
 * Mark the current execution's owner URL unusable. One-way: after this,
 * {@link getCurrentOeOwnerUrl} returns null for the rest of the execution so
 * repeated emits stop re-paying the pre-attempt timeout against a dead owner.
 * Reads `storage.getStore()` directly (like {@link recordSuspendRequest}) so a
 * call outside a run is a no-op instead of mutating the frozen fallback.
 */
export function reportOeOwnerUrlFailure(): void {
  const store = storage.getStore();
  if (store !== undefined) {
    store.ownerUrlFailure.failed = true;
  }
}

export function getCurrentRequestId(): string | null {
  return current().requestId;
}

export function getCurrentTraceId(): string | null {
  return current().traceId;
}

export function getCurrentUserId(): string | null {
  return current().userId;
}

export function getCurrentSessionId(): string | null {
  return current().sessionId;
}

export function getCurrentWorkspaceId(): string | null {
  return current().workspaceId;
}

/** Delegated authorization for the current execution, if OE injected one. */
export function getCurrentAuthorization(): ToolAuthorization | null {
  return current().authorization;
}

/**
 * The opaque caller-provided invocation payload for the current execution
 * (the request body beyond `message`), or null if none was forwarded.
 */
export function getCurrentPayload(): Record<string, unknown> | null {
  return current().payload;
}

/**
 * Caller-provided custom headers, with platform-internal `a2a-` entries
 * stripped — agent code should never see A2A tokens or routing metadata.
 * Internal platform code that needs the full set (e.g. the A2A client) should
 * call {@link getAllCustomHeaders} instead.
 */
export function getCurrentCustomHeaders(): Record<string, string> {
  const headers = current().customHeaders ?? {};
  return Object.fromEntries(
    Object.entries(headers).filter(([key]) => !key.startsWith("a2a-")),
  );
}

/** All custom headers including platform-internal `a2a-` entries. */
export function getAllCustomHeaders(): Record<string, string> {
  return current().customHeaders ?? {};
}

export function getCurrentExecutionMetadata(): Record<string, unknown> {
  return current().executionMetadata ?? {};
}

/**
 * Combine a per-call `AbortSignal` (e.g. a request/read timeout) with the
 * current execution's abort signal, if one is set. The returned signal aborts
 * when *either* fires, so an execution-wide timeout cancels the in-flight OE/LLM
 * fetch instead of leaving it to run until its own deadline. Returns
 * `callSignal` unchanged when there is no execution signal (e.g. a Tool Pod
 * call outside an AER execution). Mirrors the effect of Python's
 * `asyncio.wait_for` cancelling in-flight I/O on timeout.
 */
export function withExecutionSignal(callSignal: AbortSignal): AbortSignal {
  const execSignal = current().signal;
  return execSignal ? AbortSignal.any([callSignal, execSignal]) : callSignal;
}

/**
 * Whether the current execution's abort signal has fired (execution timeout,
 * drain, or teardown). Lets a catch block tell "the run is being torn down"
 * apart from a genuine failure of the call it was making — the abort reason
 * itself arrives as whichever value the controller was aborted with, which is
 * not reliably an Error.
 */
export function isExecutionAborted(): boolean {
  return current().signal?.aborted ?? false;
}

// =========================================================================
// Memory metadata recorder
// =========================================================================

export interface RecordMemoryMetadataArgs {
  action: string;
  memoryType: string;
  content?: string;
  relevanceScore?: number;
  query?: string;
}

/**
 * Attach explicit memory metadata to the current execution step, if one exists.
 * Mutates the metadata object in-place (matching the Python behaviour).
 */
export function recordCurrentMemoryMetadata(
  args: RecordMemoryMetadataArgs,
): void {
  const metadata = storage.getStore()?.executionMetadata;
  if (metadata === null || metadata === undefined) {
    return;
  }

  const memory: Record<string, unknown> = {
    action: args.action,
    type: args.memoryType,
  };
  if (args.content) {
    // Content and query are user/LLM-derived and this metadata is returned
    // to the UI — scrub credential-shaped fragments before persisting.
    memory["content"] = redactText(args.content);
  }
  if (args.relevanceScore !== undefined) {
    memory["relevance_score"] = args.relevanceScore;
  }
  if (args.query) {
    memory["query"] = redactText(args.query);
  }

  let memoryEvents = metadata["memory_events"];
  if (!Array.isArray(memoryEvents)) {
    memoryEvents = [];
    metadata["memory_events"] = memoryEvents;
  }
  (memoryEvents as unknown[]).push(memory);
  metadata["memory"] = memory;
}

// =========================================================================
// Session finish request
// =========================================================================

/**
 * Record that the agent considers this session finished. Reads directly off
 * `storage.getStore()` (not `current()`) so a call outside a run reports
 * "unavailable" without ever touching the frozen `EMPTY_STORE` fallback.
 * `wrapper` is null for Tool Pod / Function contexts (see server/tool.ts,
 * server/function.ts) — only the AER holds the finish latch, so those
 * contexts must also report "unavailable" rather than a misleading success.
 */
export function requestSessionFinish(): SessionFinishStatus {
  const store = storage.getStore();
  if (store === undefined || store.wrapper === null) {
    return "unavailable";
  }
  // No lock: this function is synchronous and non-async, so it cannot be
  // preempted mid-call by another callback on the same event loop turn.
  // Only cross-turn ordering (this call vs. closeSessionFinish() from a
  // *different* turn) matters, and that's a plain sequential read/write on
  // this store's own object — never shared with another turn's store.
  if (store.sessionFinish.closed) {
    return "unavailable";
  }
  if (store.sessionFinish.requested) {
    return "already_requested";
  }
  store.sessionFinish.requested = true;
  return "requested";
}

/**
 * Close the current execution's session-finish latch.
 *
 * Called from the AER's `finally` once the execute frame ends (covers
 * success, error, policy-denied, and suspend paths alike). After this,
 * requestSessionFinish() reports "unavailable" instead of promising a
 * release nothing will act on — e.g. a setTimeout or floating promise
 * scheduled during the turn but resolving after it.
 */
export function closeSessionFinish(): void {
  const store = storage.getStore();
  if (store !== undefined) {
    store.sessionFinish.closed = true;
  }
}

export function isSessionFinishRequested(): boolean {
  return current().sessionFinish.requested;
}

// =========================================================================
// Suspend request signal
// =========================================================================

/**
 * Record an author-intended HITL suspend for the current tool call. Called
 * only by `suspendPayloadToJson`, so the signal's provenance is the tool
 * author's code, not tool-result data. Reads `storage.getStore()` directly so
 * a call outside a run is a no-op instead of mutating the frozen fallback.
 */
export function recordSuspendRequest(payload: Record<string, unknown>): void {
  const store = storage.getStore();
  if (store !== undefined) {
    store.suspendRequest.payload = payload;
  }
}

/**
 * The suspend payload the current tool call requested via
 * `suspendPayloadToJson`, or null if it did not. The Tool Pod reads this after
 * the tool returns to decide whether to report `status: "suspend"`.
 */
export function getRequestedSuspend(): Record<string, unknown> | null {
  return current().suspendRequest.payload;
}

/** Run one in-process tool call with an isolated suspend marker. */
export function runWithSuspendRequestContext<T>(fn: () => T): T {
  const store = storage.getStore();
  if (store === undefined) {
    return fn();
  }
  return storage.run({ ...store, suspendRequest: { payload: null } }, fn);
}

const CUSTOMER_ORIGIN = "customer";
const customerOriginStorage = new AsyncLocalStorage<boolean>();

/** `"customer"` inside a customer-code boundary, else null. */
export function getCurrentLogOrigin(): string | null {
  return customerOriginStorage.getStore() === true ? CUSTOMER_ORIGIN : null;
}

/**
 * Mark the dynamic extent of customer agent/tool code for log attribution.
 * Nested scopes are a no-op. Missing origin is unclassified, not proven
 * platform-authored.
 */
export function runWithCustomerOrigin<T>(fn: () => T): T {
  if (customerOriginStorage.getStore() === true) {
    return fn();
  }
  return customerOriginStorage.run(true, fn);
}

const callAbortStorage = new AsyncLocalStorage<AbortSignal>();

/**
 * The per-call stop signal for the in-flight callback-routed tool call.
 * Defined only inside a tool body that declared call-interrupt support; a
 * cooperative body checks it (or forwards it to `fetch` etc.) to stop at its
 * next checkpoint. Undefined everywhere else.
 */
export function getCallAbortSignal(): AbortSignal | undefined {
  return callAbortStorage.getStore();
}

/** Run one callback-routed tool body with its per-call abort signal attached. */
export function runWithCallAbortSignal<T>(signal: AbortSignal, fn: () => T): T {
  return callAbortStorage.run(signal, fn);
}
