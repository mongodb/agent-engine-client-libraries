/**
 * Agent Execution Runtime (AER) server — framework-agnostic agent execution.
 *
 * Responsibilities:
 * - Execute agents via the BaseAgent protocol (framework-neutral)
 * - Use SecureToolWrapper for all tool/LLM calls through OE
 * - Handle HITL via StreamEvent suspend/resume pattern
 * - Report completion/suspension to OE via callback
 *
 * Mirrors Python's `AERServer` in `server/aer.py`.
 */

import type { FastifyInstance } from "fastify";
import { trace, type Span } from "@opentelemetry/api";
import { fromJson } from "@bufbuild/protobuf";
import {
  JsonValueSchema,
  type AgentInput,
  type BaseAgent,
  type JsonValue,
  type RequestContext,
  type StreamEvent,
} from "@mongodb-js/agent-engine-sdk";
import { explicitFeatures, RuntimeAgentConfig } from "../agent_config.js";
import { fatalWithTermination } from "../launcher.js";
import { getLogger } from "../logger.js";
import { TimeoutError, withTimeout } from "../timeout.js";
import { withMetrics } from "../metrics.js";
import {
  AERExecuteResponseSchema,
  ExecuteRequestSchema,
  ExecutorCallbackRequestSchema,
  SuspendPayloadSchema,
  type AERExecuteResponse,
  type ExecuteRequest,
  type ExecutorCallbackRequest,
  type InterruptResult,
  type StreamingResult,
  type ToolsListResponse,
} from "../models.js";
import { NodeExecutionLogger } from "../node_logger.js";
import { DRAIN_ABORT_REASON } from "./drain.js";
import {
  ExternalAPICallError,
  LLMInvocationError,
  PolicyDeniedException,
  SecureToolWrapper,
  ToolCallTimeoutError,
} from "../secure_wrapper.js";
import {
  closeSessionFinish,
  getCurrentExecutionId,
  getCurrentOeOwnerUrl,
  isSessionFinishRequested,
  runWithCustomerOrigin,
  reportOeOwnerUrlFailure,
  runWithExecutionContext,
  type OwnerUrlFailureState,
} from "../context.js";
import {
  AttemptHeartbeat,
  DurableMemoryState,
  WorkflowClient,
  cancelledHistoryReport,
  currentCancelledHistory,
  reportCancelledHistoryBeforeCleanup,
  type CancelledHistoryCapture,
  type CancelledHistoryProvider,
  WorkflowMessageSchema,
  attemptStartRequestFromExecute,
  newDurabilityOwnerId,
  runWithAttemptContext,
  validateDurableMemoryIdentity,
} from "../workflow/index.js";
import type {
  AttemptContext,
  AttemptStartRequest,
} from "../generated/workflow/v1/runtime_pb.js";
import {
  closeAllTLSAgents,
  fetchPlatform,
  getFetchOptionsWithTLS,
} from "../tls_client.js";
import {
  filterThinkingTokens,
  getEnvFloat,
  getRequestTimeout,
  logExecutionCallback,
  logSection,
  stripThinking,
} from "../utils.js";
import { getQueryPlugin } from "../hooks.js";
import {
  noteCheckpointWireWorkspaceId,
  resolveCheckpointWorkspaceId,
} from "../checkpoint_workspace.js";
import { BaseServer } from "./base.js";
import { CallbackDelivery } from "./callback_delivery.js";
import { DurableAckError, DurableChunkSender } from "./chunk_delivery.js";
import {
  DONE,
  DURABLE_ACK_INCOMPLETE_ERROR_CODE,
  ERROR,
  LLM_CREDENTIAL_REJECTED_ERROR_CODE,
  LLM_INVOCATION_ERROR_CODE,
  LLM_INVOCATION_ERROR_SOURCE,
  POLICY_DENIED_ERROR_CODE,
  RUN_TIMEOUT_ERROR_CODE,
  TEXT,
  TIMEOUT_ERROR_CODE,
  TOOL_CREDENTIAL_REJECTED_ERROR_CODE,
} from "./chunk_types.js";
import { resolveOeUrl } from "./oe_url.js";
import { resolveOwnerUrl } from "./owner_url.js";
import { discardResponseBody, postWithRetries } from "./http_retry.js";
import {
  StreamJournalManager,
  journalSseLines,
  type JournalReason,
} from "./stream_journal.js";
import { pipeSseLinesToResponse } from "./tool.js";
import { forceFlushTracing, getTracer } from "../tracing/index.js";
import { runWithInboundTraceContext } from "./inbound_trace_span.js";
import {
  AER_BUILD_AGENT,
  AER_EXECUTE,
  OPENINFERENCE_SPAN_KIND,
  OpenInferenceSpanKind,
} from "../span_names.js";

const logger = getLogger("agent_engine_runner_shared.server.aer");

// Whole-turn execution timeout. The platform stamps it in vm-mode; this default
// covers local development. It sits under the 70-minute stream
// deadline so the runner, not the stream, ends a long turn.
const EXECUTION_TIMEOUT_MS =
  getEnvFloat("RUNNER_EXECUTION_TIMEOUT", 3600.0) * 1000;

/** A per-execution build whose caller gave up while it was still running. */
interface AbandonedBuild {
  build: Promise<unknown>;
  graceTimer: NodeJS.Timeout;
}

/**
 * How long an abandoned per-execution build may keep running past its deadline
 * before the pod gives up on it and exits for replacement.
 *
 * Read per call so it stays a deployment knob. The build already ran for the
 * whole execution timeout before being abandoned, and the spec asks builders to
 * stay lightweight, so anything past this is not going to settle.
 */
function abandonedBuildGraceMs(): number {
  return getEnvFloat("RUNNER_ABANDONED_BUILD_GRACE", 60.0) * 1000;
}

// Ordered durable chunk acknowledgement (negotiated: the engine stamps
// durable_ack_enabled when the workspace advertised the durable_ack
// capability). The whole turn's budget bounds the sender's in-order retries:
// the platform's stream deadline is what the caller was promised, so retrying
// a batch past it would only produce output nobody is left to read. Mirrors
// Python's DURABLE_ACK_TURN_BUDGET_S.
const DURABLE_ACK_TURN_BUDGET_MS = EXECUTION_TIMEOUT_MS;
// A failing turn drains its outstanding chunks best-effort, bounded well below
// the turn budget: the ERROR terminal and its settlement must not be held on
// the network while the engine is unresponsive. Mirrors Python's
// DURABLE_ACK_ERROR_DRAIN_TIMEOUT_S.
const DURABLE_ACK_ERROR_DRAIN_TIMEOUT_MS = 5_000;
// Tolerance for deciding that a settlement that ran out of budget did so at
// the execution's deadline (and is therefore the timeout's to settle): the
// settle's own deadline is computed from the same instant and can raise a
// fraction of a millisecond early on float rounding.
const SETTLEMENT_DEADLINE_TOLERANCE_MS = 5;

// Per-request OE callback timeout.
const REQUEST_TIMEOUT_MS = getRequestTimeout() * 1000;

// Timeout for the capability advertise POST. Mirrors Python's
// `_advertise_capabilities` `timeout=10.0`.
const CAPABILITY_ADVERTISE_TIMEOUT_MS = 10_000;

// Bounded wait for queued Memory writes before asking OE to release a finished
// session. Mirrors the Python AER contract.
const SESSION_FINISH_MEMORY_DRAIN_TIMEOUT_MS = 10_000;

// Bounded wait for the tracer provider to flush before asking OE to release a
// finished session. The run root (`aer.execute`) ends with the turn, but a
// batch processor has not necessarily shipped it yet; the OE may kill this pod
// with zero grace once it honours /finish.
const SESSION_FINISH_TRACE_FLUSH_TIMEOUT_MS = 5_000;
const CANCELLED_HISTORY_TIMEOUT_MS = 3_000;

// Bound on the cancelled-turn checkpoint write during drain teardown. The OE's
// cancel drain deadline is 5s and the write is one Mongo upsert; 3s leaves the
// remaining settlement phases their share. Mirrors Python's
// CANCELLED_TURN_COMMIT_TIMEOUT_S.
const CANCELLED_TURN_COMMIT_TIMEOUT_MS = 3000;

/** Reject after CANCELLED_TURN_COMMIT_TIMEOUT_MS when the cancelled-turn
 * checkpoint write has not settled; a no-op for agents without the hook.
 * Resolves with the work's value — the commit's checkpoint coordinate. */
async function withCancelledTurnDeadline(
  work: Promise<unknown> | undefined,
): Promise<unknown> {
  if (work === undefined) return undefined;
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      work,
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(
          () => reject(new Error("cancelled-turn commit timed out")),
          CANCELLED_TURN_COMMIT_TIMEOUT_MS,
        );
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

// `language` fallback when agent.yaml omits it — how the TS SDK identifies
// itself to OE (mirrors agent.yaml's own `language: typescript` convention
// for scaffolded TS agents; Python's equivalent fallback is "python").
const DEFAULT_LANGUAGE = "typescript";

// Mirrors Python's Query(max_length=200); bounds the $in scan if the OE
// proxy is ever bypassed (trust model in server/query.ts).
const MAX_QUERY_SESSION_IDS = 200;

const QUERY_PLUGIN_UNSUPPORTED_DETAIL =
  "session queries are not supported by this framework";

// Plugin errors can echo raw persisted state (user content): bound what is
// logged and keep the 500 response body generic.
const QUERY_ERROR_LOG_MAX_CHARS = 200;

function describeQueryError(exc: unknown): string {
  const name = exc instanceof Error ? exc.name : typeof exc;
  const message = exc instanceof Error ? exc.message : String(exc);
  return `${name}: ${message.slice(0, QUERY_ERROR_LOG_MAX_CHARS)}`;
}

/**
 * Structured metadata for an LLM-provider failure after OE approval.
 * `error_code` is persisted on the execution so unary invokes can classify
 * without the live stream chunk; `code` is the same value so stream
 * consumers that read `metadata.code` (the gateway) match
 * `metadata.error_code` (AER). Mirrors Python's _llm_invocation_metadata.
 *
 * When the tool pod stamped a specific classification (e.g. a credential
 * rejection), it replaces the generic code and `source` is omitted: the
 * gateway maps that bare code to client-owned, while `llm` attribution
 * would read as provider flakiness for a failure the customer can fix.
 *
 * Only the pod's credential stamp is honored. The live relay forwards the
 * pod's frames verbatim and LLMInvocationError is agent-raisable, so the
 * value arriving here is workload-authored: an unrecognized code falls back
 * to the generic classification rather than letting the workload rename its
 * failure for the gateway's allowlist and owner attribution.
 */
function llmInvocationMetadata(errorCode?: string): Record<string, string> {
  if (errorCode === LLM_CREDENTIAL_REJECTED_ERROR_CODE) {
    return { error_code: errorCode, code: errorCode };
  }
  return {
    error_code: LLM_INVOCATION_ERROR_CODE,
    code: LLM_INVOCATION_ERROR_CODE,
    source: LLM_INVOCATION_ERROR_SOURCE,
  };
}

/**
 * Structured metadata when the run failed on a rejected credential.
 *
 * Two shapes qualify, and only these: a tool call whose classified
 * external-API failure was an auth rejection (the tool's configured
 * credential), and an LLMInvocationError the framework wrapped before it
 * reached the dedicated catch (the pod's stamped code rides along).
 * Frameworks may nest either under `cause`, so the walk covers the chain;
 * any other failure shape returns undefined and the error stays
 * unclassified. Mirrors Python's _credential_rejection_metadata.
 */
function credentialRejectionMetadata(
  err: unknown,
): Record<string, string> | undefined {
  const seen = new Set<unknown>();
  const stack: unknown[] = [err];
  while (stack.length > 0) {
    const current = stack.pop();
    if (current === null || current === undefined || seen.has(current)) {
      continue;
    }
    seen.add(current);
    // Literal matches the closed classification the pod stamps
    // (tool_api_error's status map) and the OE allowlists.
    if (
      current instanceof ExternalAPICallError &&
      current.tool_api_error.classification === "AUTH_FAILED"
    ) {
      return {
        error_code: TOOL_CREDENTIAL_REJECTED_ERROR_CODE,
        code: TOOL_CREDENTIAL_REJECTED_ERROR_CODE,
      };
    }
    // The exception type is agent-raisable, so only the exact code the pod
    // can have stamped is honored; anything else falls back to generic.
    if (
      current instanceof LLMInvocationError &&
      current.error_code === LLM_CREDENTIAL_REJECTED_ERROR_CODE
    ) {
      return { error_code: current.error_code, code: current.error_code };
    }
    if (current instanceof Error) {
      stack.push(current.cause);
    }
  }
  return undefined;
}

/**
 * Normalize Fastify's query-string parsing to a string list: a repeated
 * `session_ids` param arrives as an array, a single occurrence as a string,
 * and an absent param as undefined.
 */
function normalizeSessionIds(raw: unknown): string[] {
  if (raw === undefined || raw === null) return [];
  if (Array.isArray(raw)) return raw.map(String);
  return [String(raw)];
}

function getOwnerCallbackUrls(server: {
  ownerCallbackUrl?: Map<string, string>;
}): Map<string, string> {
  if (!(server.ownerCallbackUrl instanceof Map)) {
    server.ownerCallbackUrl = new Map();
  }
  return server.ownerCallbackUrl;
}

function getPendingSessionFinishes(server: {
  pendingSessionFinish?: Map<string, ExecutorCallbackRequest>;
}): Map<string, ExecutorCallbackRequest> {
  if (!(server.pendingSessionFinish instanceof Map)) {
    server.pendingSessionFinish = new Map();
  }
  return server.pendingSessionFinish;
}

/**
 * Build a Promise that rejects when `signal` fires (or immediately if it is
 * already aborted). Used to race against `iter.next()` so a timeout can unwind
 * a pending `await` instead of waiting for the next event.
 *
 * The listener is registered once per execution (the returned promise is
 * reused across every iter.next() race), so this does not accumulate listeners.
 */
function abortPromise(signal: AbortSignal): Promise<never> {
  return new Promise<never>((_, reject) => {
    const err = new Error("Agent execution aborted");
    err.name = "AbortError";
    if (signal.aborted) {
      reject(err);
      return;
    }
    signal.addEventListener("abort", () => reject(err), { once: true });
  });
}

/** Fields resolved from a request's payload with top-level fallback. */
interface ResolvedInvocationParams {
  sessionId: string;
  message: string;
}

interface ExecutorCallbackFields {
  attempt_id?: string;
  attempt_fence?: number;
  suspend_generation?: number;
  result?: unknown;
  error?: string;
  suspend_reason?: string;
  suspend_context?: Record<string, unknown>;
  interrupts?: InterruptResult["interrupts"];
  resume_schema?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

function buildExecutorCallback(
  executionId: string,
  status: string,
  fields: ExecutorCallbackFields = {},
) {
  validateCallbackInterrupts(fields.interrupts);
  const callback = ExecutorCallbackRequestSchema.parse({
    execution_id: executionId,
    status,
    attempt_id: fields.attempt_id,
    attempt_fence: fields.attempt_fence,
    suspend_generation: fields.suspend_generation,
    result: fields.result,
    error: fields.error,
    suspend_reason: fields.suspend_reason,
    suspend_context: fields.suspend_context,
    interrupts: fields.interrupts,
    resume_schema: fields.resume_schema,
    metadata: fields.metadata,
  });
  // Serialize before cleanup so validation and delivery use the same detached
  // payload even if agent-owned values are mutated while resources close.
  return JSON.parse(JSON.stringify(callback)) as ExecutorCallbackRequest;
}

/**
 * Resolve the continuation id and message for this execution. Mirrors Python's
 * `_resolve_invocation_params` in `server/aer.py`.
 *
 * `sessionId` falls back to `execution_id` when the caller did not supply
 * one. Framework resume state (e.g. a checkpoint id) is *not* resolved here —
 * it is opaque state the adapter reads from `ctx.metadata`.
 */
function resolveInvocationParams(
  request: ExecuteRequest,
): ResolvedInvocationParams {
  const reqPayload = request.payload ?? {};

  // Mirror Python's `session_id or execution_id`: an empty-string session_id
  // (e.g. a partially-migrated caller's zero value) must fall through rather
  // than collapse unrelated executions into one session.
  const sessionId = request.session_id || request.execution_id;

  const payloadMessage = reqPayload["message"];
  const message =
    (typeof payloadMessage === "string" ? payloadMessage : undefined) ??
    request.message;

  return { sessionId, message };
}

/**
 * Reject a request that carries `message` both top-level and inside `payload`.
 *
 * The AER is the only component that unpacks the opaque payload, so the
 * collision is caught here and raised as a clean 400. Doing it here (rather
 * than in the model schema) keeps the payload out of the error body and the
 * logs. Mirrors Python's `_reject_ambiguous_message` in `server/aer.py`.
 */
function rejectAmbiguousMessage(request: ExecuteRequest): void {
  const payloadMsg = request.payload ? request.payload["message"] : undefined;
  if (request.message && typeof payloadMsg === "string" && payloadMsg) {
    const err = new Error(
      "Ambiguous request: 'message' was provided both at the top level " +
        "and inside 'payload'. Send the message in exactly one place.",
    ) as Error & { statusCode: number };
    err.statusCode = 400;
    throw err;
  }
}

async function postJson(url: string, body: unknown): Promise<void> {
  const tlsOptions = getFetchOptionsWithTLS(url);
  const response = await fetchPlatform(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    ...tlsOptions,
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} from ${url}`);
  }
}

function validateCallbackInterrupts(
  interrupts: InterruptResult["interrupts"],
): void {
  for (const interrupt of interrupts ?? []) {
    try {
      if (JsonValueSchema.safeParse(interrupt.value).success) continue;
    } catch {
      // Recursive values can overflow the recursive schema before returning a result.
    }
    throw new TypeError(
      `Interrupt "${interrupt.id}" has a non-JSON-serializable value`,
    );
  }
}

/**
 * Parse a path segment that must be an integer.
 *
 * Returns null when it is not one, so the route can answer the same 422 the
 * Python twin's typed path parameter produces. A parsed but unknown generation
 * (including a negative one) is a 404, not a validation error.
 */
function parseIntegerSegment(raw: string): number | null {
  if (!/^-?\d+$/.test(raw)) return null;
  const value = Number.parseInt(raw, 10);
  return Number.isSafeInteger(value) ? value : null;
}

/**
 * Parse the `after_seq` query value.
 *
 * Mirrors the Python route's `Query(default=0, ge=0)`: absent means "from the
 * start", and a supplied value that is not a non-negative integer is a
 * validation error rather than a silent reset. A resume cursor must never be
 * reinterpreted as a different replay range.
 */
function parseAfterSeq(raw: string | undefined): number | "invalid" {
  if (raw === undefined) return 0;
  if (!/^\d+$/.test(raw)) return "invalid";
  const value = Number.parseInt(raw, 10);
  return Number.isSafeInteger(value) ? value : "invalid";
}

export class AERServer extends BaseServer {
  private chunkSeq: Map<string, number> = new Map();
  // Ordered durable chunk senders per in-flight execution, registered only
  // when the engine negotiated durable acknowledgement. Absent for every other
  // execution, which keeps the legacy per-chunk path byte-for-byte unchanged.
  private durableAckSenders: Map<string, DurableChunkSender> = new Map();
  // Runner-wide stream journal and the generation each journaled execution
  // opened. Registered only when the engine negotiated a journal dispatch, so
  // the legacy per-chunk path stays byte-for-byte unchanged. Lazily created
  // (like `drainRegistry`) so tests that skip the constructor still work.
  private _streamJournal?: StreamJournalManager;
  private _journalGeneration?: Map<string, number>;
  // Per-execution validated OE owner callback URL, keyed by
  // execution_id like `chunkSeq`. Set in `doHandleExecute` (only when the
  // request's owner URL validates against the resolved service URL) and read by
  // `sendStreamChunk` / `reportCallback` so those two endpoints prefer the
  // owning OE replica. Registered and popped inside the same try/finally as
  // `chunkSeq`, so an entry can never leak when execution setup fails early.
  ownerCallbackUrl: Map<string, string> = new Map();
  private callbackDelivery = new CallbackDelivery();
  // Executions whose agent called finishSession() and completed normally;
  // drained by the /execute route once the response has been flushed.
  pendingSessionFinish: Map<string, ExecutorCallbackRequest> = new Map();
  private readonly durabilityOwnerId = newDurabilityOwnerId();
  // Per-execution builds whose caller gave up while the build was still running,
  // keyed by execution_id. The execution stays registered as drain work until
  // the build settles, so a drain cannot report this pod idle while author code
  // is still running against shared state, and a build that never settles takes
  // the pod out of service rather than leaving it wedged and healthy-looking.
  private readonly abandonedBuilds = new Map<string, AbandonedBuild>();
  // One-shot capability advertise, mirrors Python's `_capabilities_advertised`.
  // Set only after OE accepts the declaration.
  private capabilitiesAdvertised = false;
  get modeName(): string {
    return "aer";
  }

  /**
   * Runner-wide stream journal, lazily created so a harness that skips the
   * constructor still gets a working store.
   */
  get streamJournal(): StreamJournalManager {
    return (this._streamJournal ??= new StreamJournalManager());
  }

  /** Per-execution journal generation, lazily created (see the field). */
  private get journalGeneration(): Map<string, number> {
    return (this._journalGeneration ??= new Map());
  }

  /**
   * Build the graph for this invocation, bounded by the execution timeout.
   *
   * The builder does its own I/O and may be async, so it is awaited here: this
   * is author I/O running before the whole-turn timeout starts, and a stalled
   * builder would otherwise hold /execute open indefinitely. A global agent
   * resolves to `undefined` without awaiting anything.
   */
  private async buildGraphForInvocation(
    executionId: string,
    agentInput: AgentInput,
    ctx: RequestContext,
    timeoutMs: number = EXECUTION_TIMEOUT_MS,
  ): Promise<unknown> {
    const pendingGraph =
      this.runtime.buildGraph?.(agentInput, ctx) ?? Promise.resolve(undefined);
    let buildSettled = false;
    const markSettled = (): void => {
      buildSettled = true;
    };
    void pendingGraph.then(markSettled, markSettled);

    try {
      return await withTimeout(
        pendingGraph,
        timeoutMs,
        () =>
          `the per-execution graph build did not complete within ${timeoutMs / 1000} seconds`,
      );
    } catch (error) {
      if (error instanceof TimeoutError && !buildSettled) {
        // The caller is gone but the build is still running. Keep its execution
        // registered as work until it settles, so a drain cannot report this
        // pod idle while author code is live. The build itself is not
        // cancelled: it holds the registry lock, and the next invocation
        // serializes behind it.
        this.trackAbandonedBuild(executionId, pendingGraph);
      }
      throw error;
    }
  }

  /**
   * Record a per-execution build whose caller gave up while it was still
   * running, so the /execute route keeps its execution registered as drain work
   * until the build settles.
   *
   * The entry clears on settlement, which is also what returns the pod to
   * normal service: the next invocation waits on the build lock and proceeds
   * once the abandoned build releases it.
   */
  private trackAbandonedBuild(
    executionId: string,
    build: Promise<unknown>,
  ): void {
    const entry: AbandonedBuild = {
      build,
      graceTimer: setTimeout(() => {
        this.exitForUnsettledBuild(executionId);
      }, abandonedBuildGraceMs()),
    };
    // Never hold the process open just to fire this.
    entry.graceTimer.unref();
    this.abandonedBuilds.set(executionId, entry);

    const settle = (): void => {
      if (this.abandonedBuilds.get(executionId) !== entry) return;
      clearTimeout(entry.graceTimer);
      this.abandonedBuilds.delete(executionId);
    };
    void build.then(settle, settle);
  }

  /**
   * Report a deadline breach the way the platform expects: a terminal ERROR
   * chunk and callback carrying the timeout error code, then a 504 that the
   * outer handler re-raises without reporting a second time.
   *
   * Shared by the whole-turn deadline and the graph-build deadline, which are
   * the same bound and reach the caller the same way.
   */
  private async reportDeadlineBreach(
    request: ExecuteRequest,
    stageCallback: (
      status: string,
      fields?: ExecutorCallbackFields,
    ) => ExecutorCallbackRequest,
  ): Promise<never> {
    const errorMsg = `Execution timed out after ${EXECUTION_TIMEOUT_MS / 1000} seconds`;
    try {
      await this.sendStreamChunk(
        request.platform_api_url,
        request.execution_id,
        ERROR,
        "",
        errorMsg,
        { error_code: RUN_TIMEOUT_ERROR_CODE },
      );
    } catch (chunkErr) {
      logger.error(
        { err: chunkErr },
        `Failed to deliver ERROR chunk for execution ${request.execution_id} after retries`,
      );
    }
    stageCallback("ERROR", {
      error: errorMsg,
      metadata: { error_code: RUN_TIMEOUT_ERROR_CODE },
    });
    const httpError = new Error(errorMsg) as Error & {
      statusCode: number;
      alreadyReported?: boolean;
    };
    httpError.statusCode = 504;
    // Terminal ERROR chunk + callback are already sent above. Mark the error so
    // the outer catch re-raises without reporting a second time — mirrors
    // Python's `except HTTPException: raise`.
    httpError.alreadyReported = true;
    throw httpError;
  }

  /**
   * Take the pod out of service after a build that never settled.
   *
   * The build still holds the registry lock, so nothing this process does can
   * make it serve the agent again. Exiting hands the replacement to the
   * orchestrator, and the termination message carries the cause to the deploy
   * timeline, which is what the runner's fatal path exists for.
   */
  private exitForUnsettledBuild(executionId: string): void {
    void fatalWithTermination(
      `A per-execution graph build for execution ${executionId} did not ` +
        `settle within ${abandonedBuildGraceMs() / 1000}s of its invocation ` +
        "deadline, so this pod can no longer serve the agent.",
    );
  }

  /**
   * End an execution's drain work, or defer it to an abandoned build's
   * settlement.
   *
   * Ending it now would let a drain report this pod idle while author code from
   * that build is still running against shared state, so the execution stays
   * registered as work until the build settles.
   */
  private endWorkForExecution(executionId: string): void {
    const abandoned = this.abandonedBuilds.get(executionId);
    if (abandoned === undefined) {
      this.drainRegistry.endWork(executionId);
      return;
    }
    const endWork = (): void => this.drainRegistry.endWork(executionId);
    void abandoned.build.then(endWork, endWork);
  }

  async onStartup(): Promise<void> {
    if (this.runtime.graphBuilder === null) {
      throw new Error(
        "Agent graph builder not initialized. " +
          "Did you pass a graph_builder to register_and_run()?",
      );
    }
    logger.info("AER graph builder ready");

    const oeUrl = (process.env["OE_URL"] ?? "").trim();
    const workspaceId = (process.env["APP_ID"] ?? "").trim();
    if (!oeUrl || !workspaceId) {
      throw new Error("Capability registration requires OE_URL and APP_ID");
    }
    await this.ensureOeRegistrations(oeUrl, workspaceId);
  }

  /**
   * Run pending one-shot OE registrations (capability advertise).
   *
   * Mirrors Python's `_ensure_oe_registrations`. Called from `onStartup` and
   * defensively from `/execute` so agent work cannot run without a persisted
   * declaration even if startup was bypassed.
   */
  private async ensureOeRegistrations(
    oeUrl: string,
    workspaceId: string,
  ): Promise<void> {
    if (!this.capabilitiesAdvertised) {
      await this.advertiseCapabilities(oeUrl, workspaceId);
    }
  }

  /** Push the current declaration to OE, failing startup if it is not accepted. */
  private async advertiseCapabilities(
    oeUrl: string,
    workspaceId: string,
  ): Promise<void> {
    const orgId = this.runtime.orgId;
    const projectId = this.runtime.projectId;
    if (!orgId || !projectId) {
      throw new Error(
        "Capability registration requires ORG_ID and PROJECT_ID " +
          `for workspace ${workspaceId}`,
      );
    }
    const agentCfg =
      typeof this.runtime.getAgentConfig === "function"
        ? this.runtime.getAgentConfig()
        : new RuntimeAgentConfig();

    const language = (agentCfg.language ?? "").trim() || DEFAULT_LANGUAGE;
    const framework = (agentCfg.framework ?? "").trim();
    const features: Record<string, boolean> = {
      owner_callback_fallback: true,
      ...explicitFeatures(agentCfg.features),
    };
    // SDK-injected too: this runner delivers its chunk stream in order and
    // retries until the engine confirms durable persistence, so the engine may
    // negotiate durable acknowledgement (durable_ack_enabled) on dispatch.
    // Inert until the engine stamps it. Mirrors Python's
    // `features.setdefault("durable_ack", True)`.
    if (!("durable_ack" in features)) features["durable_ack"] = true;
    // SDK-injected too: this runner journals a turn's chunk stream on its own
    // disk and serves it back over /stream/journal/* on resume, so the engine
    // may dispatch with stream_journal_enabled. Inert until the engine stamps
    // it. Mirrors Python's `features.setdefault("stream_journal", True)`.
    if (!("stream_journal" in features)) features["stream_journal"] = true;

    const payload = {
      workspace_id: workspaceId,
      org_id: orgId,
      project_id: projectId,
      language,
      framework,
      features,
    };

    const url = `${oeUrl}/agent/capabilities`;
    const tlsOptions = getFetchOptionsWithTLS(url);
    const response = await fetchPlatform(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(CAPABILITY_ADVERTISE_TIMEOUT_MS),
      redirect: "manual",
      ...tlsOptions,
    });

    // Only the status is inspected; cancel the body so undici can release the
    // connection instead of retaining an unread response.
    await discardResponseBody(response);
    if (response.status !== 200) {
      throw new Error(
        `Capability registration was rejected for ${workspaceId} ` +
          `with status ${response.status}`,
      );
    }

    logger.info(
      "Agent capabilities advertised for workspace %s " +
        "(language=%s framework=%s durable_workflow=%s)",
      workspaceId,
      language,
      framework || "-",
      features["durable_workflow"],
    );
    this.capabilitiesAdvertised = true;
  }

  async onShutdown(): Promise<void> {
    await this.callbackDelivery.shutdown();
    await this.runtime.memoryWriter?.shutdown();
    this.chunkSeq.clear();
    getOwnerCallbackUrls(this).clear();
    // Close all cached TLS agents for graceful shutdown (parity with Python AER)
    closeAllTLSAgents();
  }

  getHealthDetails(): Record<string, unknown> {
    return {
      graph_builder_ready: this.runtime.graphBuilder !== null,
      tools_registered: Object.keys(this.runtime.tools),
    };
  }

  registerRoutes(app: FastifyInstance): void {
    app.post("/warm-up", async (_request, reply) => {
      // TypeScript graph builders are synchronous and run on Node's main event
      // loop. Calling one here would make this best-effort optimization block
      // health and every other request, so TypeScript retains lazy first-use
      // graph construction while preserving OE's cross-runtime endpoint.
      return reply.code(204).send();
    });

    app.post("/execute", async (request, reply) => {
      const body = ExecuteRequestSchema.parse(request.body);
      // Throws 409 if the execution was drained after OE dispatched.
      this.drainRegistry.beginWork(body.execution_id);
      try {
        const result = await this.executeWithInboundTraceContext(
          body,
          request.headers,
        );
        const responseBody = AERExecuteResponseSchema.parse(result);
        const finishCallback = getPendingSessionFinishes(this).get(
          body.execution_id,
        );
        getPendingSessionFinishes(this).delete(body.execution_id);
        if (finishCallback !== undefined) {
          // After the response is flushed, never before: the OE blocks on this
          // call and turns a transport error into a 502 that overwrites the
          // finished run's result. Guard on status too - an error thrown or a
          // client disconnect after this point must not still release.
          reply.raw.once("finish", () => {
            if (reply.raw.statusCode < 400) {
              void this.releaseFinishedSession(body, finishCallback);
            }
          });
        }
        return reply.send(responseBody);
      } finally {
        this.endWorkForExecution(body.execution_id);
      }
    });

    app.get(
      "/tools",
      async (): Promise<ToolsListResponse> => ({
        tools: Object.entries(this.runtime.toolDefinitions).map(
          ([name, defn]) => ({
            name,
            ...defn,
          }),
        ),
      }),
    );

    // Session query routes — mirror Python's /query/sessions routes in
    // server/aer.py, including status-code behavior.
    app.get("/query/sessions", async (request, reply) => {
      const query = request.query as Record<string, unknown>;
      const sessionIds = normalizeSessionIds(query["session_ids"]);
      if (sessionIds.length > MAX_QUERY_SESSION_IDS) {
        return reply.code(422).send({
          detail: `session_ids accepts at most ${MAX_QUERY_SESSION_IDS} items`,
        });
      }
      if (sessionIds.length === 0) {
        return reply.send({ sessions: [] });
      }
      const plugin = getQueryPlugin();
      if (plugin === null) {
        return reply
          .code(501)
          .send({ detail: QUERY_PLUGIN_UNSUPPORTED_DETAIL });
      }
      try {
        return await reply.send(
          await plugin.getSummariesForSessions(sessionIds),
        );
      } catch (exc) {
        logger.error(
          `session summary query failed: ${describeQueryError(exc)}`,
        );
        return reply.code(500).send({ detail: "Internal Server Error" });
      }
    });

    app.get<{ Params: { session_id: string } }>(
      "/query/sessions/:session_id/messages",
      async (request, reply) => {
        const plugin = getQueryPlugin();
        if (plugin === null) {
          return reply
            .code(501)
            .send({ detail: QUERY_PLUGIN_UNSUPPORTED_DETAIL });
        }
        try {
          return await reply.send(
            await plugin.getMessagesForSession(request.params.session_id),
          );
        } catch (exc) {
          logger.error(
            `session messages query failed: ${describeQueryError(exc)}`,
          );
          return reply.code(500).send({ detail: "Internal Server Error" });
        }
      },
    );

    // Runner stream journal read routes. The runner owns the journal; these
    // serve it back to whichever engine replica resumes the turn. They sit
    // behind the same auth hook as every other route, which is inactive on a
    // runner with no RUNNER_AUTH_TOKEN; the executor network policy (only the
    // per-project OE may reach this port) is the control.
    app.get<{ Params: { execution_id: string } }>(
      "/stream/journal/:execution_id",
      async (request, reply) => {
        const index = this.streamJournal.getIndex(request.params.execution_id);
        if (index === null) {
          return reply
            .code(404)
            .send({ detail: "unknown stream journal execution" });
        }
        return reply.send(index);
      },
    );

    app.get<{
      Params: { execution_id: string; generation: string };
      Querystring: { after_seq?: string };
    }>("/stream/journal/:execution_id/:generation", async (request, reply) => {
      const executionId = request.params.execution_id;
      const generation = parseIntegerSegment(request.params.generation);
      if (generation === null) {
        return reply
          .code(422)
          .send({ detail: "invalid stream journal generation" });
      }
      const afterSeq = parseAfterSeq(request.query.after_seq);
      if (afterSeq === "invalid") {
        return reply
          .code(422)
          .send({ detail: "invalid stream journal after_seq" });
      }
      const journal = this.streamJournal.getJournal(executionId, generation);
      // An evicted body survives only in the index; a follow must fail closed
      // rather than serve an empty stream as if it were complete.
      if (journal === null || !journal.retained) {
        return reply
          .code(404)
          .send({ detail: "unknown stream journal generation" });
      }
      const release = this.streamJournal.acquireReader(executionId, generation);
      if (release === null) {
        return reply
          .code(429)
          .send({ detail: "stream journal reader limit reached" });
      }
      // SSE is a raw stream: hijack so Fastify does not try to serialize a
      // body, then let the shared pipe own the socket for the journal's
      // lifetime (disconnect cancel, backpressure, safe end).
      reply.hijack();
      const res = reply.raw;
      res.statusCode = 200;
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      res.flushHeaders();
      try {
        await pipeSseLinesToResponse(
          res,
          (signal) => journalSseLines(journal, afterSeq, signal),
          { onGeneratorSettled: release },
        );
      } catch (error) {
        // The pipe already settled the generator on most paths; release is
        // idempotent so this only covers a failure before it started.
        release();
        throw error;
      }
    });
  }

  private async reportCancelledHistory(
    request: ExecuteRequest,
    attempt: AttemptContext,
    agent: BaseAgent,
    signal: AbortSignal,
  ): Promise<void> {
    if (!signal.aborted || signal.reason !== DRAIN_ABORT_REASON) return;
    const deadline = this.drainRegistry.cancellationDeadline(
      request.execution_id,
    );
    if (deadline === null) return;
    const provider = agent as BaseAgent & Partial<CancelledHistoryProvider>;
    if (typeof provider.captureCancelledHistory !== "function") return;
    const reportDeadline = Math.min(
      deadline,
      Date.now() + CANCELLED_HISTORY_TIMEOUT_MS,
    );
    if (reportDeadline <= Date.now()) return;
    let timer: NodeJS.Timeout | undefined;
    try {
      const capture = provider.captureCancelledHistory();
      if (capture === null) return;
      const report = cancelledHistoryReport(attempt, capture);
      const timeoutMs = reportDeadline - Date.now();
      if (timeoutMs <= 0) return;
      await Promise.race([
        new WorkflowClient(request.platform_api_url, {
          timeoutMs,
        }).reportCancelledHistory(report),
        new Promise<never>((_, reject) => {
          timer = setTimeout(
            () => reject(new Error("cancelled history deadline exceeded")),
            timeoutMs,
          );
        }),
      ]);
    } catch (error) {
      logger.warn(
        `Failed to retain cancelled execution history (${error instanceof Error ? error.name : typeof error})`,
      );
    } finally {
      if (timer !== undefined) clearTimeout(timer);
    }
  }

  private async reportSuspendedHistory(
    request: ExecuteRequest,
    attempt: AttemptContext,
    capture: CancelledHistoryCapture,
  ): Promise<void> {
    // A suspended execution ends its attempt, so a later Stop finds no execute
    // handler: report the display transcript now so it survives cancellation.
    let timer: NodeJS.Timeout | undefined;
    try {
      const report = cancelledHistoryReport(attempt, capture, true);
      const timeoutMs = CANCELLED_HISTORY_TIMEOUT_MS;
      await Promise.race([
        new WorkflowClient(request.platform_api_url, {
          timeoutMs,
        }).reportCancelledHistory(report),
        new Promise<never>((_, reject) => {
          timer = setTimeout(
            () => reject(new Error("suspended history deadline exceeded")),
            timeoutMs,
          );
        }),
      ]);
    } catch (error) {
      logger.warn(
        `Failed to retain suspended execution history (${error instanceof Error ? error.name : typeof error})`,
      );
    } finally {
      if (timer !== undefined) clearTimeout(timer);
    }
  }

  /**
   * Open `aer.execute` as the run's root, joining an inbound W3C traceparent
   * when one arrived.
   *
   * Mirrors Python's `_execute_with_inbound_trace_context`. Without this the
   * AER's own spans — and the trace_id persisted onto tool-call records via
   * `getCurrentTraceContext()` — start a disconnected root trace even though
   * the caller's traceparent now arrives over the A2A hop. With no inbound
   * traceparent the span is a fresh local root, the same as in Python.
   */
  private executeWithInboundTraceContext(
    body: ExecuteRequest,
    headers: Record<string, string | string[] | undefined>,
  ): Promise<AERExecuteResponse> {
    // Outermost step of an invocation, so it declares AGENT rather than leaving
    // a backend to guess from the span name.
    return runWithInboundTraceContext(
      {
        name: AER_EXECUTE,
        kind: OpenInferenceSpanKind.AGENT,
        tracerName: "agent-engine-runner-shared.aer",
        attributes: { "execution.id": body.execution_id },
      },
      headers,
      (span) => this.handleExecute(body, span),
    );
  }

  private handleExecute = withMetrics(
    "aer_execute",
    (request: ExecuteRequest, rootSpan?: Span) =>
      this.doHandleExecute(request, rootSpan),
  );

  private async doHandleExecute(
    request: ExecuteRequest,
    // The run root, passed explicitly by `executeWithInboundTraceContext` so
    // the identity stamping below cannot land on an unrelated active span.
    // Omitted only by direct callers (tests); the active span is the same root
    // there, so the fallback is equivalent, never a silent wrong target.
    rootSpan?: Span,
  ): Promise<AERExecuteResponse> {
    logSection();
    logger.debug(
      "AER: Execute %s... (resume=%s)",
      request.execution_id.slice(0, 8),
      request.resume,
    );
    // Pin the callback base to the runner's own configured OE before any of
    // it is used. Every outbound call below derives from this field, and the
    // runner treats the responses as authoritative results and policy
    // decisions — so it must not be chosen by the request. Rebinding here
    // rather than at each use site means a call site added later inherits
    // the trusted value automatically.
    request.platform_api_url = resolveOeUrl(request.platform_api_url);
    logger.info("AER: OE callback URL: %s", request.platform_api_url);

    // Validate the request-supplied owner URL against the trusted
    // service URL just resolved. A forged value validates to null and is
    // ignored; a valid one lets stream chunks, the terminal callback, and
    // in-process tool results prefer the owning OE replica.
    const ownerUrl = resolveOwnerUrl(
      request.platform_api_owner_url,
      request.platform_api_url,
    );
    const ownerUrlFailure: OwnerUrlFailureState = { failed: false };

    // Fail fast on a message supplied both top-level and inside payload,
    // before any side effects (execution, callbacks). Mirrors Python.
    rejectAmbiguousMessage(request);

    // Defensively enforce readiness registration if the startup lifecycle was
    // bypassed, using the resolved OE URL. Capability identity comes from
    // trusted deployment configuration, not the caller-controlled payload.
    const registrationWorkspaceId = (process.env["APP_ID"] ?? "").trim();
    if (registrationWorkspaceId) {
      await this.ensureOeRegistrations(
        request.platform_api_url,
        registrationWorkspaceId,
      );
    }

    // Resolve message/session_id from payload with top-level fallback.
    // Mirrors Python's _resolve_invocation_params.
    const resolved = resolveInvocationParams(request);
    const message = resolved.message;
    const sessionId = resolved.sessionId;

    noteCheckpointWireWorkspaceId(request.workspace_id ?? null);
    const workspaceId = resolveCheckpointWorkspaceId(
      request.workspace_id ?? null,
    );

    // `aer.execute` opened before `runWithExecutionContext`, so the
    // ExecutionContextSpanProcessor could not stamp session/workspace on it.
    // Stamp them on the run root explicitly — `executeWithInboundTraceContext`
    // threads the root span in — so the identity cannot land on an unrelated
    // active span if a wrapper ever opens one in between, and so it stays
    // visible to the execution/session trace filters. The exporter reads
    // attributes at span end, so the top-level session_id/execution_id land on
    // the stored document.
    const span = rootSpan ?? trace.getActiveSpan();
    if (span !== undefined) {
      if (sessionId) {
        span.setAttribute("session.id", sessionId);
        // thread.id mirrors session.id: the runner uses session_id as the
        // LangGraph checkpoint thread_id, and some OTLP consumers key off
        // "thread.id" — the same encoding Python's tracing setup writes.
        span.setAttribute("thread.id", sessionId);
      }
      if (workspaceId) {
        span.setAttribute("workspace.id", workspaceId);
      }
    }

    logger.debug(
      "AER: Message: %s",
      message.length > 50 ? message.slice(0, 50) + "..." : message,
    );

    const wrapper = new SecureToolWrapper(
      request.platform_api_url,
      request.execution_id,
      request.custom_headers,
      ownerUrl,
      this.drainRegistry,
    );

    // Execution-wide cancellation. Python relies on asyncio.wait_for to cancel
    // the inner coroutine on timeout; Node has no implicit cancellation, so a
    // timeout (below) — or an execution drain, via the registry attach below —
    // aborts this controller, which is (1) raced against
    // iter.next() in executeViaAgentStream to stop consuming promptly and (2)
    // threaded through the execution context (`signal`) so requestOeApproval and
    // the SSE stream abort their in-flight OE/LLM fetches — together cancelling
    // active network I/O rather than leaving it to run.
    const abortController = new AbortController();
    // Give the drain receiver the same handle the execution timeout uses, so
    // a drain has exactly the timeout's reach. If the drain landed before this
    // point (between admission and here), the attach aborts immediately.
    this.drainRegistry.attachController(request.execution_id, abortController);

    let durableAttempt: AttemptContext | null = null;
    let pendingCallback: ExecutorCallbackRequest | undefined;
    // Lifecycle reason for the stream journal, set by the paths that settle
    // cleanly. Left null by an error exit (the finally maps that to "error")
    // and overridden by cancellation.
    let journalReason: JournalReason | null = null;
    const stageCallback = (
      status: string,
      fields: ExecutorCallbackFields = {},
    ): ExecutorCallbackRequest => {
      pendingCallback = buildExecutorCallback(request.execution_id, status, {
        ...fields,
        ...(durableAttempt !== null &&
        (status === "COMPLETED" || status === "ERROR")
          ? {
              attempt_id: durableAttempt.attemptId,
              attempt_fence: Number(durableAttempt.fencingToken),
            }
          : {}),
        suspend_generation: request.suspend_generation ?? undefined,
      });
      return pendingCallback;
    };

    let cleanupError: unknown;
    let cleanupPhase: string | undefined;
    let hasCleanupError = false;
    const rememberCleanupError = (phase: string, error: unknown): void => {
      if (!hasCleanupError) {
        cleanupError = error;
        cleanupPhase = phase;
        hasCleanupError = true;
        return;
      }
      logger.error(
        {
          err: error,
          execution_id: request.execution_id,
          cleanup_phase: phase,
        },
        "Additional AER cleanup failure",
      );
    };

    const execution = runWithExecutionContext(
      {
        executionId: request.execution_id,
        traceId: request.platform_trace_id,
        wrapper,
        oeUrl: request.platform_api_url,
        // Owner-preferring transports (e.g. emit() from an in-process tool)
        // read this from the execution context.
        oeOwnerUrl: ownerUrl,
        ownerUrlFailure,
        userId: request.user_id,
        sessionId: sessionId,
        workspaceId: workspaceId,
        customHeaders: request.custom_headers,
        // Forward the opaque caller payload so agent tools can read it via
        // getCurrentPayload(). Mirrors Python set_execution_context(payload=...).
        payload: request.payload,
        signal: abortController.signal,
        // A journaled turn routes in-process progress emits through the turn's
        // own chunk path, so a local tool's step is sequenced and recorded like
        // every other chunk. Other turns keep progress's own transport.
        emitChunk:
          request.stream_journal_enabled &&
          Number.isInteger(request.stream_generation) &&
          request.stream_generation > 0
            ? (event, data) =>
                this.sendStreamChunk(
                  request.platform_api_url,
                  request.execution_id,
                  event,
                  data,
                  "",
                  {},
                )
            : null,
      },
      async () => {
        // The execution's original absolute deadline: successful settlement
        // must not outlive it, so a turn whose budget expired cannot be
        // presented as completed.
        const executionDeadlineMs = performance.now() + EXECUTION_TIMEOUT_MS;
        const timeoutHandle = setTimeout(
          () => abortController.abort(),
          EXECUTION_TIMEOUT_MS,
        );

        const ctx: RequestContext = {
          executionId: request.execution_id,
          sessionId: sessionId,
          userId: request.user_id,
          workspaceId: workspaceId ?? undefined,
          requestHeaders: request.custom_headers,
          // Framework metadata travels on the context, not the caller payload.
          // AER passes it through without interpreting adapter-owned keys.
          resume: request.resume,
          resumeData: (request.resume_data ?? null) as JsonValue,
          metadata: (request.metadata ?? null) as Record<
            string,
            JsonValue
          > | null,
          previousExecutionCancelled: request.previous_execution_cancelled,
          signal: abortController.signal,
        };

        // AgentInput.payload is the opaque caller blob, with the resolved
        // message normalized in. Platform plumbing (identity, resume,
        // checkpoint) travels on ctx, not here.
        const agentInput: AgentInput = {
          payload: {
            ...(request.payload ?? {}),
            message,
          } as JsonValue,
        };

        // Native-vs-durable eligibility is decidable without I/O, but only
        // once the agent is materialized — building the graph is what
        // registers (or clears) the workflow adapter eligibility consults.
        // Until then the routing mode is unknown (undefined), and an unknown
        // mode never writes native history.
        let durableStartRequest: AttemptStartRequest | null | undefined;
        let agent: BaseAgent | undefined;
        let durableRoutingResolved = false;
        let heartbeat: AttemptHeartbeat | null = null;

        // A cancelled run still owes history its final turn — including a stop
        // landing during preparation, before the stream starts. Best-effort:
        // never changes the settlement path.
        const commitCancelledTurnBestEffort = async (): Promise<void> => {
          // An acquired attempt vetoes native history unconditionally.
          // Eligibility not yet computed (teardown before agent
          // materialization) or a lease call that never returned both leave
          // the mode unknown — never write native history for what may be a
          // durable run.
          const nativeConfirmed =
            durableAttempt === null &&
            durableStartRequest !== undefined &&
            (durableStartRequest === null || durableRoutingResolved);
          if (
            !nativeConfirmed ||
            this.drainRegistry.drainReason(request.execution_id) !==
              "execution_cancelled"
          ) {
            return;
          }
          try {
            const coordinate = await withCancelledTurnDeadline(
              agent?.commitCancelledTurn?.(ctx, agentInput),
            );
            // The coordinate rides the existing executor-callback metadata
            // channel: OE's terminal guard persists it for the branch-source
            // selector — the runner is the only party that knows it.
            if (
              coordinate !== null &&
              typeof coordinate === "object" &&
              typeof (coordinate as { thread_id?: unknown }).thread_id ===
                "string" &&
              typeof (coordinate as { checkpoint_id?: unknown })
                .checkpoint_id === "string"
            ) {
              stageCallback("CANCELLED", {
                metadata: {
                  langgraph_checkpoint: coordinate as {
                    thread_id: string;
                    checkpoint_id: string;
                  },
                },
              });
            }
          } catch (commitErr) {
            logger.warn(
              { err: commitErr, execution_id: request.execution_id },
              "cancelled-turn checkpoint commit failed",
            );
          }
        };
        try {
          // Register the owner callback URL inside the same try whose finally
          // pops it, so a failure between here and the first
          // sendStreamChunk / reportCallback can never strand an entry. Nothing
          // between context setup and this point delivers a chunk or callback.
          if (ownerUrl) {
            getOwnerCallbackUrls(this).set(request.execution_id, ownerUrl);
          }

          if (request.resume) {
            logger.info(
              "AER: Resume execution for %s",
              request.execution_id.slice(0, 8),
            );
          }

          logger.info(
            "AER /execute: execution_id=%s, session_id=%s",
            request.execution_id.slice(0, 8),
            sessionId.slice(0, 8),
          );

          const nodeLogger = new NodeExecutionLogger({
            oeUrl: request.platform_api_url,
            executionId: request.execution_id,
            sessionId,
            userId: request.user_id,
            orgId: request.org_id,
            projectId: request.project_id,
          });
          logger.debug(
            "Created NodeExecutionLogger for session %s",
            sessionId.slice(0, 8),
          );

          // Stream journal, only when the engine negotiated it: from here on
          // every chunk this turn emits is appended to a journal on the
          // runner's own disk, so a resume on any replica can read the turn's
          // stream back. Admission happens before preparation, and the single
          // finally below closes the journal after any outcome handler has
          // emitted its terminal frame — including for a preparation failure.
          await this.openStreamJournal(request, sessionId);

          let suspendedCapture: CancelledHistoryCapture | null = null;
          let executionOutcome: InterruptResult | StreamingResult;
          let graph: unknown;
          try {
            graph = await this.buildGraphForInvocation(
              request.execution_id,
              agentInput,
              ctx,
            );
            // A build that settles normally can still belong to an invocation
            // that was cancelled while it ran. Re-check before materializing,
            // or the author's input-preparation hook starts after the Stop:
            // the stream's abort race cannot prevent it, because the iterator's
            // first next() is evaluated before the race.
            if (abortController.signal.aborted) {
              throw new Error(
                "Execution stopped: cancelled while the graph was building",
              );
            }
          } catch (error) {
            // The build deadline is the execution deadline, so a breach is
            // reported as a timeout rather than falling through to the generic
            // error path as a 500. Classify against the controller as well as
            // the error: a builder that honors ctx.signal reports the deadline
            // as an ordinary abort, and the deadline is what aborted the
            // controller. A drain is a cancellation, not a breach, so it keeps
            // the existing cancelled handling in the outer catch.
            const drained =
              abortController.signal.aborted &&
              abortController.signal.reason === DRAIN_ABORT_REASON;
            if (
              !drained &&
              (error instanceof TimeoutError || abortController.signal.aborted)
            ) {
              await this.reportDeadlineBreach(request, stageCallback);
            }
            throw error;
          }

          agent = getTracer("runner-shared.aer").startActiveSpan(
            AER_BUILD_AGENT,
            {
              attributes: {
                [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN,
              },
            },
            (span) => {
              try {
                return this.runtime.getAgent({
                  callbacks: [nodeLogger],
                  graph,
                });
              } finally {
                span.end();
              }
            },
          );

          // Materializing the agent registered (or cleared) the workflow
          // adapter, so eligibility is decidable here — before the first
          // preparation await — even on this worker's first request.
          durableStartRequest = attemptStartRequestFromExecute(
            request,
            sessionId,
            this.durabilityOwnerId,
            this.runtime.appName,
            this.runtime.appVersion,
            // != null: memoryWriter is optional, and the declaration must agree
            // with the wrapper setup guard on absent vs disabled writers.
            this.runtime.memoryWriter != null,
          );

          // Lease the attempt before tenant work. An old OE (bare 404) stays
          // native; other start failures fail closed.
          durableAttempt = await this.startDurableAttempt(
            request,
            durableStartRequest,
          );
          durableRoutingResolved = true;
          if (durableAttempt !== null && this.runtime.memoryWriter != null) {
            // != null (not !==): memoryWriter is optional, and an absent
            // writer must behave like a disabled one.
            validateDurableMemoryIdentity(
              durableAttempt.workflowIdentity,
              request.user_id,
            );
            wrapper.durableMemory = new DurableMemoryState(message || null);
          }
          if (
            request.resume_from_step !== undefined &&
            request.resume_from_step !== null &&
            durableAttempt === null
          ) {
            // Raise the shared allocator watermark so the next mint continues
            // past the resumed step (not restarting at 1).
            wrapper.observeOperationalStep(request.resume_from_step);
          }
          heartbeat =
            durableAttempt === null
              ? null
              : new AttemptHeartbeat(
                  durableAttempt,
                  // Bind the execution signal so a Stop cancels the initial
                  // renewal in flight: the reporting finally cannot run until
                  // that await settles, and OE's drain window is shorter than
                  // the transport timeout.
                  new WorkflowClient(request.platform_api_url, {
                    signal: abortController.signal,
                  }),
                );
          // Ordered chunk delivery through one coalescing poster, started
          // only when the engine asked for it: a journal dispatch (the
          // journal is the durable copy, so live delivery is best effort) or
          // a negotiated durable acknowledgement (live delivery is the
          // durable copy, so a failed drain fails the turn). A dispatch that
          // carries both is a journal dispatch. The sender is torn down by the
          // finally below. Nothing is registered otherwise, so the legacy path
          // is untouched.
          if (request.stream_journal_enabled || request.durable_ack_enabled) {
            // The poster and the settlement waits observe a drain/Stop
            // promptly, but a whole-turn timeout is not a cancellation — the
            // sender's own budget bounds it — so accepted chunks can still use
            // the bounded error drain. Production (a producer blocked on the
            // queue cap) observes any abort so the timeout owner can run.
            const senderAbort = new AbortController();
            abortController.signal.addEventListener(
              "abort",
              () => {
                if (abortController.signal.reason === DRAIN_ABORT_REASON) {
                  senderAbort.abort(abortController.signal.reason);
                }
              },
              { once: true },
            );
            this.startDurableChunkSender(
              request,
              ownerUrlFailure,
              senderAbort.signal,
              abortController.signal,
            );
          }

          try {
            // agent is assigned by the build step above (getAgent throws
            // rather than returning undefined); the declared type stays wide
            // only because the teardown closure may run before the build.
            const builtAgent: BaseAgent = agent;
            // Bound once so the reporting closure below keeps the non-null
            // narrowing the ternary establishes.
            const activeAttempt = durableAttempt;
            const execute = () =>
              this.executeViaAgentStream(
                builtAgent,
                ctx,
                agentInput,
                request.platform_api_url,
                request.execution_id,
              );
            executionOutcome =
              activeAttempt === null
                ? await execute()
                : await runWithAttemptContext(activeAttempt, async () => {
                    const history = currentCancelledHistory();
                    if (history !== null) {
                      const opener = fromJson(WorkflowMessageSchema, {
                        role: "MESSAGE_ROLE_USER",
                        content: message,
                        id: `durable-input:${request.execution_id}`,
                      });
                      history.setInput(opener);
                      history.reporter = () =>
                        this.reportCancelledHistory(
                          request,
                          activeAttempt,
                          builtAgent,
                          abortController.signal,
                        );
                    }
                    try {
                      // Tenant code stays behind successful activation, but the
                      // capture above must already be in place: a Stop during
                      // this await makes OE refuse the renewal (a cancelled
                      // execution cannot renew), and the finally below is the
                      // only path that reports the turn's opener.
                      if (heartbeat !== null) await heartbeat.start();
                      return await execute();
                    } finally {
                      if (history !== null)
                        suspendedCapture = history.capture();
                      await reportCancelledHistoryBeforeCleanup();
                    }
                  });
          } catch (err) {
            // The controller fires for the execution-timeout setTimeout and
            // for a drain. A drain is a cancellation, not a deadline breach —
            // report it truthfully instead of claiming a phantom timeout.
            if (abortController.signal.aborted) {
              if (abortController.signal.reason === DRAIN_ABORT_REASON) {
                throw new Error(
                  "Execution stopped: drained after the execution was cancelled",
                  { cause: err },
                );
              }
              await this.reportDeadlineBreach(request, stageCallback);
            }
            throw err;
          }

          if (isInterruptResult(executionOutcome)) {
            const hasEnvelope =
              executionOutcome.interrupts !== undefined &&
              executionOutcome.resume_schema !== undefined;
            let suspendReason: string;
            let suspendContext: Record<string, unknown> | undefined;
            if (hasEnvelope) {
              if (executionOutcome.interrupts?.length === 0) {
                const err = new Error(
                  "Agent produced an empty interrupt snapshot; cannot suspend.",
                ) as Error & { statusCode: number };
                err.statusCode = 500;
                throw err;
              }
              const parsedPayload = SuspendPayloadSchema.safeParse(
                executionOutcome.interrupts?.[0]?.value,
              );
              if (parsedPayload.success) {
                suspendReason = parsedPayload.data.suspend_reason;
                suspendContext = parsedPayload.data.suspend_context;
              } else {
                suspendReason = "agent_interrupt";
              }
            } else {
              const payload = SuspendPayloadSchema.parse(
                executionOutcome.suspend_payload,
              );
              suspendReason = payload.suspend_reason;
              suspendContext = payload.suspend_context;
            }

            logExecutionCallback(
              request.execution_id,
              "SUSPENDED",
              null,
              null,
              suspendReason,
              "AER",
            );

            if (durableAttempt !== null && suspendedCapture !== null) {
              await this.reportSuspendedHistory(
                request,
                durableAttempt,
                suspendedCapture,
              );
            }

            if (this.runtime.memoryWriter && durableAttempt === null) {
              try {
                this.runtime.memoryWriter.writeTurnAsync({
                  message,
                  resultMessages: executionOutcome.messages,
                  userId: request.user_id,
                  sessionId,
                  includeUserTurn: !request.resume,
                });
              } catch (error) {
                const kind = error instanceof Error ? error.name : typeof error;
                logger.warn(
                  `Failed to queue pre-suspend memory write (${kind})`,
                );
              }
            }

            // Settlement ordering (negotiated durable ack): the suspension
            // must not be presented before every chunk produced up to it is
            // acknowledged — a resume replays the turn's history, and a gap
            // there would be permanent. A failed drain fails the turn instead.
            // A journal dispatch is exempt: the journal, not live delivery, is
            // the durable copy, so settlement does not wait for the poster to
            // drain.
            const suspendSender = this.durableAckSenders.get(
              request.execution_id,
            );
            if (suspendSender !== undefined && !suspendSender.journalMode) {
              try {
                // Same original-deadline bound as completion settlement.
                await suspendSender.settle(
                  Math.max(0, executionDeadlineMs - performance.now()),
                );
                // Same original-deadline bound as completion settlement.
                if (
                  performance.now() >=
                  executionDeadlineMs - SETTLEMENT_DEADLINE_TOLERANCE_MS
                ) {
                  await this.failIncompleteDurableTurn(
                    request,
                    stageCallback,
                    new DurableAckError(
                      `durable chunk acknowledgement incomplete for execution ` +
                        `${request.execution_id}: settlement exceeded the ` +
                        `execution deadline`,
                    ),
                  );
                }
              } catch (ackErr) {
                if (
                  abortController.signal.aborted &&
                  abortController.signal.reason === DRAIN_ABORT_REASON
                ) {
                  throw ackErr;
                }
                if (!(ackErr instanceof DurableAckError)) throw ackErr;
                await this.failIncompleteDurableTurn(
                  request,
                  stageCallback,
                  ackErr,
                );
              }
            }

            stageCallback("SUSPENDED", {
              suspend_reason: suspendReason,
              suspend_context: suspendContext,
              interrupts: executionOutcome.interrupts,
              resume_schema: executionOutcome.resume_schema,
              metadata: executionOutcome.metadata,
            });
            // A suspension ends this generation's journal; the chunks before
            // it are what a resume replays. The finally writes the end marker.
            journalReason = "suspended";
            return {
              status: "suspended",
              suspend_reason: suspendReason,
            };
          }

          if (
            this.runtime.memoryWriter &&
            executionOutcome.messages.length > 0 &&
            durableAttempt === null
          ) {
            try {
              this.runtime.memoryWriter.writeTurnAsync({
                message,
                resultMessages: executionOutcome.messages,
                userId: request.user_id,
                sessionId,
                includeUserTurn: !request.resume,
              });
            } catch (error) {
              const kind = error instanceof Error ? error.name : typeof error;
              logger.warn(`Failed to queue memory write (${kind})`);
            }
          }

          // Settlement ordering (negotiated durable ack): every chunk produced
          // by the turn must be acknowledged before the terminal frame and the
          // settlement. A failed drain means chunks are not durable, so the
          // turn is failed rather than presented as complete. A journal
          // dispatch is exempt: the journal, not live delivery, is the durable
          // copy, so settlement does not wait for the poster to drain.
          const doneSender = this.durableAckSenders.get(request.execution_id);
          if (doneSender !== undefined && !doneSender.journalMode) {
            try {
              // Successful settlement uses the execution's original absolute
              // deadline: an already timed-out turn must not be presented as
              // completed.
              await doneSender.settle(
                Math.max(0, executionDeadlineMs - performance.now()),
              );
              // A successful settle must still be inside the execution's
              // original deadline: the settle's own wait can be woken by a
              // late acknowledgement after the deadline passed, and an
              // already timed-out turn must never be presented as completed.
              if (
                performance.now() >=
                executionDeadlineMs - SETTLEMENT_DEADLINE_TOLERANCE_MS
              ) {
                await this.failIncompleteDurableTurn(
                  request,
                  stageCallback,
                  new DurableAckError(
                    `durable chunk acknowledgement incomplete for execution ` +
                      `${request.execution_id}: settlement exceeded the ` +
                      `execution deadline`,
                  ),
                );
              }
            } catch (ackErr) {
              // A drain/Stop during settlement keeps its own cancellation
              // owner; anything else — including the settlement running out
              // of the execution's original budget with chunks still
              // unacknowledged — is the incomplete-turn failure, which keeps
              // its durable_ack_incomplete discriminator.
              if (
                abortController.signal.aborted &&
                abortController.signal.reason === DRAIN_ABORT_REASON
              ) {
                throw ackErr;
              }
              if (!(ackErr instanceof DurableAckError)) throw ackErr;
              await this.failIncompleteDurableTurn(
                request,
                stageCallback,
                ackErr,
              );
            }
          }

          try {
            await this.sendStreamChunk(
              request.platform_api_url,
              request.execution_id,
              DONE,
              executionOutcome.content,
              "",
              { status: "completed" },
            );
          } catch (chunkErr) {
            logger.error(
              { err: chunkErr },
              `Failed to deliver DONE chunk for execution ${request.execution_id} after retries`,
            );
          }

          const completedCallback = stageCallback("COMPLETED", {
            result: executionOutcome.content,
            metadata: executionOutcome.metadata,
          });
          if (isSessionFinishRequested()) {
            // Latched here, acted on after the response: the OE blocks on
            // this /execute call and turns a transport error into a 502 that
            // overwrites the finished run's result, so the session's pods
            // must not be killed before the response is delivered.
            getPendingSessionFinishes(this).set(
              request.execution_id,
              completedCallback,
            );
          }
          journalReason = "completed";
          return { status: "completed", result: executionOutcome.content };
        } catch (err) {
          const httpError = err as Error & {
            statusCode?: number;
            alreadyReported?: boolean;
          };
          // A path that already emitted its terminal ERROR callback (e.g. the
          // execution-timeout branch) re-raises without reporting again, so OE
          // sees exactly one terminal callback. Mirrors Python's
          // `except HTTPException: raise`.
          if (httpError.alreadyReported) {
            throw httpError;
          }
          // A drain (or pod teardown) aborts the execution-wide controller
          // with DRAIN_ABORT_REASON: the OE's cancel path owns the terminal
          // Cancelled settlement, so the held /execute must answer cleanly
          // rather than 500 — a dispatch failure would race that settlement
          // and mark the run ERROR. No ERROR callback is staged for the same
          // reason. Key on the controller, not the error: the inner catch
          // wraps the drain AbortError in a plain Error, so name-matching
          // misses real drains — and an AbortError agent code throws with no
          // drain in flight is a failure, not a cancellation. Mirrors
          // runner-shared (py).
          if (
            abortController.signal.aborted &&
            abortController.signal.reason === DRAIN_ABORT_REASON
          ) {
            logger.debug(
              `AER: execution ${request.execution_id.slice(0, 8)} cancelled (drain or teardown)`,
            );
            // Single commit site for every cancelled path (stream-phase and
            // preparation): the staged CANCELLED callback carries the
            // rescued checkpoint's coordinate for OE's branch-source selector.
            await commitCancelledTurnBestEffort();
            journalReason = "aborted";
            return { status: "cancelled" };
          }
          // A policy denial is a deliberate, terminal outcome. Attach a
          // machine-readable discriminator (metadata.error_code="policy_denied"
          // plus the bare reason) so consumers can render a "blocked by policy"
          // result instead of string-matching the message.
          //
          // The human-readable `error` keeps the full exception text
          // ("Policy denied: <reason>") so consumers that don't branch on
          // metadata see no change vs. the generic catch-all below; the
          // discriminator and bare reason ride in metadata.
          //
          // The discriminator is guaranteed on the live stream ERROR chunk. It
          // is also attached to the terminal callback, but the OE currently
          // drops terminal error metadata (and the status / SSE-fallback paths
          // don't expose it), so non-streaming consumers can't rely on it until
          // the OE follow-up lands — sending it now is forward-compatible.
          // Mirrors runner-shared (py).
          if (err instanceof PolicyDeniedException) {
            // Guardrail identity is included when the denial came from a
            // guardrail policy (plain tool/model/budget denials carry only the
            // reason). Mirrors runner-shared `_policy_denied_metadata`.
            const policyMeta: Record<string, string> = {
              error_code: POLICY_DENIED_ERROR_CODE,
              reason: err.reason,
            };
            if (err.guardrailMeta != null) {
              policyMeta.guardrail_id = err.guardrailMeta.guardrail_id;
              policyMeta.guardrail_category =
                err.guardrailMeta.guardrail_category;
            }
            logExecutionCallback(
              request.execution_id,
              "POLICY_DENIED",
              null,
              err.reason,
              null,
              "AER",
            );
            try {
              await this.sendStreamChunk(
                request.platform_api_url,
                request.execution_id,
                ERROR,
                "",
                err.message,
                policyMeta,
              );
            } catch (chunkErr) {
              logger.error(
                { err: chunkErr },
                `Failed to deliver policy-denied ERROR chunk for execution ${request.execution_id} after retries`,
              );
            }
            stageCallback("ERROR", {
              error: err.message,
              metadata: policyMeta,
            });
            if (!httpError.statusCode) httpError.statusCode = 500;
            throw httpError;
          }
          if (err instanceof ToolCallTimeoutError) {
            // Per-call timeouts remain retryable independently of the run limit.
            const timeoutMeta: Record<string, string | number> = {
              error_code: TIMEOUT_ERROR_CODE,
              tool_name: err.toolName,
              elapsed_seconds: err.elapsedSeconds,
            };
            logExecutionCallback(
              request.execution_id,
              "ERROR",
              null,
              err.message,
              null,
              "AER",
            );
            try {
              await this.sendStreamChunk(
                request.platform_api_url,
                request.execution_id,
                ERROR,
                "",
                err.message,
                timeoutMeta,
              );
            } catch (chunkErr) {
              logger.error(
                { err: chunkErr },
                `Failed to deliver timeout ERROR chunk for execution ${request.execution_id} after retries`,
              );
            }
            stageCallback("ERROR", {
              error: err.message,
              metadata: timeoutMeta,
            });
            httpError.statusCode = 504;
            throw httpError;
          }
          if (err instanceof LLMInvocationError) {
            // Stamp source=llm only for provider-owned failures. Relay,
            // truncation, and guardrail plumbing share this exception type
            // without that source and must stay uncoded.
            if (err.source !== LLM_INVOCATION_ERROR_SOURCE) {
              const uncodedMsg = err.message;
              logExecutionCallback(
                request.execution_id,
                "ERROR",
                null,
                uncodedMsg,
                null,
                "AER",
              );
              // The turn is failing: drain accepted output best-effort
              // (bounded) before the settlement, preserving the original
              // uncoded error.
              await this.drainDurableChunksBestEffort(
                request.execution_id,
                "uncoded LLM error",
              );
              stageCallback("ERROR", {
                error: uncodedMsg,
              });
              if (!httpError.statusCode) httpError.statusCode = 500;
              throw httpError;
            }
            const llmMeta = llmInvocationMetadata(err.error_code);
            logExecutionCallback(
              request.execution_id,
              "ERROR",
              null,
              err.message,
              null,
              "AER",
            );
            try {
              await this.sendStreamChunk(
                request.platform_api_url,
                request.execution_id,
                ERROR,
                "",
                err.message,
                llmMeta,
              );
            } catch (chunkErr) {
              logger.error(
                { err: chunkErr },
                `Failed to deliver LLM-invocation ERROR chunk for execution ${request.execution_id} after retries`,
              );
            }
            stageCallback("ERROR", {
              error: err.message,
              metadata: llmMeta,
            });
            if (!httpError.statusCode) httpError.statusCode = 500;
            throw httpError;
          }
          if (err instanceof DurableAckError) {
            // A sender failure surfaced during production (budget exhausted,
            // fatal engine response, undeliverable chunk): the turn must not
            // be presented as complete, so it takes the same coded ERROR
            // terminal + callback path as a failed settlement drain.
            await this.failIncompleteDurableTurn(request, stageCallback, err);
          }
          // A turn that is failing still drains its outstanding chunks
          // best-effort (bounded) so accepted output is not lost with the
          // sender; the original error is preserved either way.
          await this.drainDurableChunksBestEffort(
            request.execution_id,
            "execution error",
          );
          const errorMsg = err instanceof Error ? err.message : String(err);
          logExecutionCallback(
            request.execution_id,
            "ERROR",
            null,
            errorMsg,
            null,
            "AER",
          );
          const credentialMeta = credentialRejectionMetadata(err);
          stageCallback("ERROR", {
            error: errorMsg,
            ...(credentialMeta ? { metadata: credentialMeta } : {}),
          });
          if (!httpError.statusCode) httpError.statusCode = 500;
          throw httpError;
        } finally {
          // Close the session-finish latch before the context that holds it
          // is torn down: covers the success, error,
          // policy-denied, and suspend paths alike, so post-turn work (e.g.
          // a setTimeout or floating promise scheduled during the turn)
          // that calls requestSessionFinish() after this point gets
          // "unavailable" instead of a release promise nothing will act on.
          closeSessionFinish();
          clearTimeout(timeoutHandle);
          this.chunkSeq.delete(request.execution_id);
          // The settlement paths have already drained or failed the turn, so
          // anything still queued here can no longer be presented; stop the
          // poster before the callback delivery below.
          try {
            const sender = this.durableAckSenders.get(request.execution_id);
            if (sender !== undefined) {
              this.durableAckSenders.delete(request.execution_id);
              await sender.close();
            }
          } catch (error) {
            rememberCleanupError("chunk sender shutdown", error);
          }

          // Close this turn's journal before the callback delivery: the writer
          // drains every chunk already accepted and writes the end marker
          // naming the outcome, so a follower sees the terminal chunk before
          // the end. Every exit reaches this, including suspension (which
          // sends no closing chunk of its own); an unset reason means an
          // unexpected exit, which closes as "error" so a follower is never
          // told an untrusted prefix is complete.
          try {
            await this.closeStreamJournal(request, journalReason ?? "error");
          } catch (error) {
            rememberCleanupError("stream journal close", error);
          }

          try {
            await wrapper.close();
          } catch (error) {
            rememberCleanupError("wrapper close", error);
          }
          try {
            if (heartbeat !== null) await heartbeat.stop();
          } catch (error) {
            rememberCleanupError("heartbeat shutdown", error);
          }
        }
      },
    );

    let response: AERExecuteResponse | undefined;
    let executionError: unknown;
    let hasExecutionError = false;
    try {
      response = await execution;
    } catch (error) {
      executionError = error;
      hasExecutionError = true;
    }
    try {
      if (pendingCallback !== undefined) {
        const { execution_id, status, ...fields } = pendingCallback;
        await this.reportCallback(
          request.platform_api_url,
          execution_id,
          status,
          fields,
          pendingCallback,
          ownerUrlFailure,
          ownerUrl,
        );
      }
    } finally {
      getOwnerCallbackUrls(this).delete(request.execution_id);
    }
    if (hasExecutionError) {
      if (hasCleanupError) {
        logger.error(
          {
            err: cleanupError,
            execution_id: request.execution_id,
            cleanup_phase: cleanupPhase,
          },
          "Additional AER cleanup failure",
        );
      }
      throw executionError;
    }
    if (hasCleanupError) {
      throw cleanupError;
    }
    if (response === undefined) {
      throw new Error("AER execution completed without a response");
    }
    return response;
  }

  /**
   * Ask OE to activate a fenced attempt, or return null for native routing.
   *
   * The start request arrives precomputed after agent materialization (the
   * step that registers the workflow adapter): null there is definitive —
   * incomplete tenant identity or a framework without an adapter — and is
   * never re-derived here. An old OE without workflow routes (bare 404) also
   * yields null. Any other failure propagates so the invocation fails closed
   * rather than running unfenced.
   */
  private async startDurableAttempt(
    request: ExecuteRequest,
    startRequest: AttemptStartRequest | null,
  ) {
    if (startRequest === null) return null;
    return new WorkflowClient(request.platform_api_url).startAttempt(
      startRequest,
    );
  }

  /**
   * Open this turn's stream journal and register its generation for chunk
   * recording.
   *
   * Built only when the engine set `stream_journal_enabled` with a non-zero
   * generation, and closed by the execution's finally. The registry keeps the
   * body (and, after eviction, its index) for the runner's lifetime.
   */
  private async openStreamJournal(
    request: ExecuteRequest,
    sessionId: string,
  ): Promise<void> {
    const generation = request.stream_generation;
    // The generation is the journal's key. A dispatch that omits it (or
    // carries 0) is not a journal dispatch, so the legacy path stays untouched
    // rather than producing an unaddressable body.
    if (
      !request.stream_journal_enabled ||
      !Number.isInteger(generation) ||
      generation <= 0
    ) {
      return;
    }
    this.journalGeneration.set(request.execution_id, generation);
    await this.streamJournal.openJournal(
      request.execution_id,
      sessionId,
      generation,
    );
  }

  /**
   * Close this execution's journal with a lifecycle reason.
   *
   * Idempotent, so both the preparation guard and the execution's finally can
   * call it: whichever runs first owns the end marker.
   */
  private async closeStreamJournal(
    request: ExecuteRequest,
    reason: JournalReason,
  ): Promise<void> {
    const generation = this.journalGeneration.get(request.execution_id);
    if (generation === undefined) return;
    this.journalGeneration.delete(request.execution_id);
    await this.streamJournal.closeJournal(
      request.execution_id,
      generation,
      reason,
    );
  }

  /**
   * Register this execution's ordered durable chunk sender.
   *
   * Built only when the engine asked for it (a journal dispatch or a
   * negotiated durable acknowledgement), and torn down by the execution's
   * finally. A journal dispatch builds the poster in journal mode, where live
   * delivery is best effort and settlement is not gated; a dispatch that
   * carries both fields is a journal dispatch. The owner pre-attempt follows
   * the shared owner-callback policy through the same helpers the legacy chunk
   * POST uses: one best-effort attempt, service fallback, owner never retried,
   * and the failure latched so later settlement posts skip the owner.
   */
  private startDurableChunkSender(
    request: ExecuteRequest,
    ownerUrlFailure: OwnerUrlFailureState,
    cancellationSignal: AbortSignal,
    productionSignal: AbortSignal,
  ): void {
    const executionId = request.execution_id;
    this.durableAckSenders.set(
      executionId,
      new DurableChunkSender({
        executionId,
        chunkUrl: `${request.platform_api_url}/stream/chunk`,
        deadlineMs: performance.now() + DURABLE_ACK_TURN_BUDGET_MS,
        cancellationSignal,
        productionSignal,
        journalMode: request.stream_journal_enabled,
        ownerUrlGetter: () => {
          const ownerCallbacks = getOwnerCallbackUrls(this);
          const owner =
            getCurrentExecutionId() === executionId
              ? getCurrentOeOwnerUrl()
              : (ownerCallbacks.get(executionId) ?? null);
          return owner ? `${owner}/stream/chunk` : null;
        },
        onOwnerFailure: () => {
          getOwnerCallbackUrls(this).delete(executionId);
          if (getCurrentExecutionId() === executionId) {
            reportOeOwnerUrlFailure();
          }
          // Explicit latch: the poster runs as its own async chain, and
          // setting it here keeps the failure visible to the settlement paths
          // regardless of async-context propagation.
          ownerUrlFailure.failed = true;
        },
      }),
    );
  }

  /**
   * Drain the turn's outstanding chunks within a bounded window.
   *
   * Best-effort: a turn that is already failing must not be held on the
   * network while the engine is unresponsive, so a drain that cannot finish
   * within the window is abandoned with a warning and the caller proceeds.
   */
  private async drainDurableChunksBestEffort(
    executionId: string,
    label: string,
  ): Promise<void> {
    const sender = this.durableAckSenders.get(executionId);
    if (sender === undefined) return;
    try {
      await sender.settle(DURABLE_ACK_ERROR_DRAIN_TIMEOUT_MS);
    } catch (ackErr) {
      logger.warn(
        `Draining ${label} execution ${executionId} chunks before its ERROR ` +
          `terminal failed within the bounded window; proceeding anyway: ${
            ackErr instanceof Error ? ackErr.message : String(ackErr)
          }`,
      );
    }
  }

  /**
   * Fail a turn whose chunks could not be acknowledged, never as complete.
   *
   * Reached when the settlement gate found unacknowledged chunks (budget
   * exhausted, or a fatal engine response): the turn is failed with an error
   * terminal and an ERROR callback so the engine settles it as errored — a
   * resume can retry — instead of completing a turn whose stream has a gap.
   * Throws an `alreadyReported` 500 so the outer catch re-raises without
   * reporting a second time and the /execute response reports the failure.
   */
  private async failIncompleteDurableTurn(
    request: ExecuteRequest,
    stageCallback: (
      status: string,
      fields?: ExecutorCallbackFields,
    ) => ExecutorCallbackRequest,
    error: unknown,
  ): Promise<never> {
    const detail = error instanceof Error ? error.message : String(error);
    const errorMsg = `Stream chunk delivery could not be durably acknowledged: ${detail}`;
    logger.warn(
      `Refusing to present execution ${request.execution_id} as complete: ${detail}`,
    );
    try {
      await this.sendStreamChunk(
        request.platform_api_url,
        request.execution_id,
        ERROR,
        "",
        errorMsg,
        { error_code: DURABLE_ACK_INCOMPLETE_ERROR_CODE },
      );
    } catch (chunkErr) {
      logger.error(
        { err: chunkErr },
        `Failed to deliver durable-ack ERROR chunk for execution ${request.execution_id} after retries`,
      );
    }
    stageCallback("ERROR", {
      error: errorMsg,
      metadata: { error_code: DURABLE_ACK_INCOMPLETE_ERROR_CODE },
    });
    const httpError = new Error(errorMsg) as Error & {
      statusCode: number;
      alreadyReported?: boolean;
    };
    httpError.statusCode = 500;
    // Terminal ERROR chunk + callback are already sent above. Mark the error
    // so the outer catch re-raises without reporting a second time.
    httpError.alreadyReported = true;
    throw httpError;
  }

  private async sendStreamChunk(
    oeUrl: string,
    executionId: string,
    chunkType: string,
    content = "",
    error = "",
    // Numbers allowed so numeric discriminator fields (elapsed_seconds) stay
    // numeric on the wire rather than being stringified for TS's benefit.
    metadata: Record<string, string | number> = {},
  ): Promise<void> {
    const seq = (this.chunkSeq.get(executionId) ?? 0) + 1;
    this.chunkSeq.set(executionId, seq);

    const chunk: Record<string, unknown> = {
      execution_id: executionId,
      chunk_type: chunkType,
      content,
      error,
      metadata,
      seq,
    };
    // Stream journal, only for a journaled turn: stamp the generation and
    // record the chunk (terminal included) before it is posted, so the journal
    // is a complete, ordered copy of what the engine was sent. The recording
    // never blocks; a bound drops the chunk and marks the journal incomplete.
    const generation = this.journalGeneration.get(executionId);
    if (generation !== undefined) {
      chunk["generation"] = generation;
      this.streamJournal.append(executionId, generation, {
        kind: "chunk",
        seq,
        generation,
        chunk,
      });
    }
    const terminal = chunkType === DONE || chunkType === ERROR;
    const durableSender = this.durableAckSenders.get(executionId);
    if (durableSender !== undefined && !terminal) {
      // Ordered poster path: queue for the coalescing poster, which retries in
      // order until the engine confirms durable persistence. On the
      // durable-ack path a sender failure propagates — dropping the chunk
      // would silently present an incomplete turn as complete. On the journal
      // path the poster is in journal mode: it never blocks or fails the turn,
      // and a chunk it cannot deliver live is dropped (the journal keeps it).
      await durableSender.send(chunk);
      return;
    }
    if (durableSender !== undefined && chunkType === ERROR) {
      // A failing turn drains its outstanding chunks best-effort first
      // (bounded, so the error terminal is not held), then delivers the error
      // regardless: the turn is already failing.
      await this.drainDurableChunksBestEffort(executionId, "error terminal");
    }
    const ownerCallbacks = getOwnerCallbackUrls(this);
    const owner =
      getCurrentExecutionId() === executionId
        ? getCurrentOeOwnerUrl()
        : (ownerCallbacks.get(executionId) ?? null);
    try {
      await postWithRetries(
        `${oeUrl}/stream/chunk`,
        chunk,
        owner ? `${owner}/stream/chunk` : null,
        owner
          ? () => {
              ownerCallbacks.delete(executionId);
              if (getCurrentExecutionId() === executionId) {
                reportOeOwnerUrlFailure();
              }
            }
          : null,
      );
    } catch (e) {
      if (terminal) {
        // Terminal chunks re-raise on exhaustion so the caller can fall
        // through to reportCallback — OE's exec.Status fallback then
        // synthesizes a terminal SSE chunk from exec.Result / exec.Error.
        // Mirrors Python's `_send_stream_chunk`.
        this.chunkSeq.delete(executionId);
        throw e;
      }
      // Non-terminal chunk (text / subagent boundary): the run continues;
      // the chunk is dropped.
      logger.warn(
        `Failed to deliver ${chunkType} chunk (seq=${seq}) to OE after retries: ${e}`,
      );
      return;
    }

    if (terminal) {
      this.chunkSeq.delete(executionId);
    }
  }

  private async executeViaAgentStream(
    agent: BaseAgent,
    ctx: RequestContext,
    agentInput: AgentInput,
    oeUrl: string,
    executionId: string,
  ): Promise<InterruptResult | StreamingResult> {
    // Per-stream <think> filter state keyed by (source, tool_call_id).
    // source alone is insufficient: two parallel dispatches of the same-named
    // subagent share a source string but have distinct tool_call_ids, so their
    // thinking buffers must be isolated to prevent token cross-contamination.
    const thinkingState = new Map<string, [buf: string, inside: boolean]>();

    // Manual iteration so we can race iter.next() against ctx.signal. for-await
    // would block inside iter.next() with no way to unwind on abort. The race
    // unblocks promptly when the signal fires; iter.return() in `finally` lets
    // the underlying generator run any cleanup it has.
    // Only stream creation and iterator steps run customer code; the
    // rest of the loop is platform-owned.
    const iter = runWithCustomerOrigin(() =>
      agent.execute(ctx, agentInput)[Symbol.asyncIterator](),
    );
    const signal = ctx.signal;
    const abortRacePromise = signal ? abortPromise(signal) : null;
    try {
      while (true) {
        const result: IteratorResult<StreamEvent> = abortRacePromise
          ? await runWithCustomerOrigin(() =>
              Promise.race([iter.next(), abortRacePromise]),
            )
          : await runWithCustomerOrigin(() => iter.next());
        if (result.done) break;
        const streamEvent = result.value;
        if (
          typeof streamEvent.data !== "object" ||
          streamEvent.data === null ||
          Array.isArray(streamEvent.data)
        ) {
          throw new TypeError(
            `Expected object event data, got ${typeof streamEvent.data}`,
          );
        }
        const eventData = streamEvent.data as Record<string, unknown>;

        if (streamEvent.event === "token") {
          const tokenContent =
            (eventData["content"] as string | undefined) ?? "";
          const source = String(eventData["source"] ?? "");
          const toolCallId = String(eventData["tool_call_id"] ?? "");
          if (tokenContent) {
            const stateKey = `${source}::${toolCallId}`;
            const [buf, inside] = thinkingState.get(stateKey) ?? ["", false];
            const [streamable, newBuf, newInside] = filterThinkingTokens(
              tokenContent,
              buf,
              inside,
            );
            thinkingState.set(stateKey, [newBuf, newInside]);
            if (streamable) {
              await this.sendStreamChunk(
                oeUrl,
                executionId,
                TEXT,
                streamable,
                "",
                {
                  source,
                  tool_call_id: toolCallId,
                },
              );
            }
          }
        } else if (streamEvent.event === "suspend") {
          const rawMetadata = eventData["metadata"];
          const metadata =
            typeof rawMetadata === "object" &&
            rawMetadata !== null &&
            !Array.isArray(rawMetadata)
              ? (rawMetadata as Record<string, unknown>)
              : {};
          return {
            suspend_payload: eventData["suspend_payload"] ?? {},
            interrupts: eventData["interrupts"] as
              | InterruptResult["interrupts"]
              | undefined,
            resume_schema: eventData["resume_schema"] as
              | Record<string, unknown>
              | undefined,
            metadata,
            messages: ((eventData["messages"] as unknown[] | null) ??
              []) as unknown as InterruptResult["messages"],
          };
        } else if (streamEvent.event === "result") {
          const content = stripThinking(
            (eventData["response"] as string | undefined) ?? "",
          );
          // Cast through unknown: the agent stream sends raw message objects that match
          // the Message wire shape but TypeScript cannot verify the deeply nested types.
          const messages = ((eventData["messages"] as unknown[]) ??
            []) as unknown as StreamingResult["messages"];
          return {
            content,
            messages,
            metadata: eventData["metadata"] as
              | StreamingResult["metadata"]
              | undefined,
          };
        }
      }
    } finally {
      // Hint to the underlying generator that we are done so it can run its
      // own cleanup (e.g. close streams). Fire-and-forget: awaiting
      // `iter.return()` would block on the generator's current pending
      // `await`, which defeats the abort race we just performed. On a timeout
      // the execution signal also aborts the in-flight OE/LLM fetch (see the
      // execution-context `signal`), so that pending `await` rejects promptly
      // and the generator unwinds rather than running on; the return is then
      // processed at its next checkpoint, after which it is eligible for GC.
      Promise.resolve(
        // Generator cleanup during return() runs customer code — keep it
        // inside the customer scope for attribution.
        runWithCustomerOrigin(() => iter.return?.()),
      ).catch(() => {
        /* ignore */
      });
    }

    throw new Error(
      `Agent stream ended without result or suspend (execution_id=${executionId})`,
    );
  }

  private async reportCallback(
    oeUrl: string,
    executionId: string,
    status: string,
    fields: ExecutorCallbackFields = {},
    stagedCallback?: ExecutorCallbackRequest,
    ownerUrlFailure?: OwnerUrlFailureState,
    ownerUrlOverride?: string | null,
  ): Promise<void> {
    const callback =
      stagedCallback ?? buildExecutorCallback(executionId, status, fields);
    const ownerCallbacks = getOwnerCallbackUrls(this);
    const owner = ownerUrlFailure
      ? ownerUrlFailure.failed
        ? null
        : (ownerUrlOverride ?? null)
      : getCurrentExecutionId() === executionId
        ? getCurrentOeOwnerUrl()
        : (ownerCallbacks.get(executionId) ?? null);
    await this.callbackDelivery.send(
      oeUrl,
      callback,
      owner,
      owner
        ? () => {
            ownerCallbacks.delete(executionId);
            if (ownerUrlFailure) ownerUrlFailure.failed = true;
            if (getCurrentExecutionId() === executionId) {
              reportOeOwnerUrlFailure();
            }
          }
        : null,
    );
  }

  /**
   * Ask the OE to free this session's compute now that the turn is over.
   *
   * Called from the /execute route only after its response has been flushed,
   * because the OE will kill this pod with zero grace. Wait for any retained
   * terminal callback and queued Memory writes first so the finish request
   * cannot destroy their only in-memory copies. Best-effort: a failure just
   * leaves the pods to the idle sweep.
   */
  private async releaseFinishedSession(
    request: ExecuteRequest,
    callback: ExecutorCallbackRequest = {
      execution_id: request.execution_id,
      status: "COMPLETED",
    },
  ): Promise<void> {
    if (
      !(await this.callbackDelivery.waitForTerminal(
        request.execution_id,
        callback,
      ))
    ) {
      return;
    }

    const writer = this.runtime.memoryWriter;
    if (
      writer &&
      !(await writer.drain(SESSION_FINISH_MEMORY_DRAIN_TIMEOUT_MS))
    ) {
      logger.warn(
        `Session finish: ${writer.pendingWrites} memory write(s) still pending after 10s; releasing anyway`,
      );
    }

    // The run root ended with the turn, but a batch processor ships spans
    // asynchronously. Flush (bounded) before the release request so the root
    // reaches the trace store, instead of dying with the pod under the OE's
    // zero-grace teardown. Best-effort: a slow store must not hold the release
    // open, and a failure just leaves the run to the idle sweep.
    try {
      if (!(await forceFlushTracing(SESSION_FINISH_TRACE_FLUSH_TIMEOUT_MS))) {
        logger.warn(
          "Session finish: tracing flush did not complete before releasing; the run root may be missing from traces",
        );
      }
    } catch (e) {
      logger.debug(
        { err: (e as Error).message },
        "Trace flush before session release failed",
      );
    }

    try {
      const baseUrl = request.platform_api_url.replace(/\/+$/, "");
      await postJson(
        `${baseUrl}/executions/${request.execution_id}/finish`,
        {},
      );
    } catch (e) {
      logger.error({ err: e }, "Session finish request failed");
    }
  }
}

function isInterruptResult(
  outcome: InterruptResult | StreamingResult,
): outcome is InterruptResult {
  return "suspend_payload" in outcome;
}
