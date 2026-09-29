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
import {
  JsonValueSchema,
  type AgentInput,
  type BaseAgent,
  type JsonValue,
  type RequestContext,
  type StreamEvent,
} from "@mongodb-js/agent-engine-sdk";
import { explicitFeatures, RuntimeAgentConfig } from "../agent_config.js";
import { getLogger } from "../logger.js";
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
  attemptStartRequestFromExecute,
  newDurabilityOwnerId,
  runWithAttemptContext,
  validateDurableMemoryIdentity,
} from "../workflow/index.js";
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
import {
  DONE,
  ERROR,
  LLM_CREDENTIAL_REJECTED_ERROR_CODE,
  LLM_INVOCATION_ERROR_CODE,
  LLM_INVOCATION_ERROR_SOURCE,
  POLICY_DENIED_ERROR_CODE,
  TEXT,
  TIMEOUT_ERROR_CODE,
  TOOL_CREDENTIAL_REJECTED_ERROR_CODE,
} from "./chunk_types.js";
import { resolveOeUrl } from "./oe_url.js";
import { resolveOwnerUrl } from "./owner_url.js";
import { discardResponseBody, postWithRetries } from "./http_retry.js";
import { getTracer } from "../tracing/index.js";
import {
  AER_BUILD_AGENT,
  OPENINFERENCE_SPAN_KIND,
  OpenInferenceSpanKind,
} from "../span_names.js";

const logger = getLogger("agent_engine_runner_shared.server.aer");

// Whole-turn execution timeout. Matches the platform's stream deadline so a turn
// is not cut off short of the limit the caller was promised.
const EXECUTION_TIMEOUT_MS =
  getEnvFloat("RUNNER_EXECUTION_TIMEOUT", 600.0) * 1000;

// Per-request OE callback timeout.
const REQUEST_TIMEOUT_MS = getRequestTimeout() * 1000;

// Timeout for the capability advertise POST. Mirrors Python's
// `_advertise_capabilities` `timeout=10.0`.
const CAPABILITY_ADVERTISE_TIMEOUT_MS = 10_000;

// Bounded wait for queued Memory writes before asking OE to release a finished
// session. Mirrors the Python AER contract.
const SESSION_FINISH_MEMORY_DRAIN_TIMEOUT_MS = 10_000;

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
  pendingSessionFinish?: Set<string>;
}): Set<string> {
  if (!(server.pendingSessionFinish instanceof Set)) {
    server.pendingSessionFinish = new Set();
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

export class AERServer extends BaseServer {
  private chunkSeq: Map<string, number> = new Map();
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
  pendingSessionFinish: Set<string> = new Set();
  private readonly durabilityOwnerId = newDurabilityOwnerId();
  // One-shot capability advertise, mirrors Python's `_capabilities_advertised`.
  // Set only after OE accepts the declaration.
  private capabilitiesAdvertised = false;
  get modeName(): string {
    return "aer";
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
        const result = await this.handleExecute(body);
        const responseBody = AERExecuteResponseSchema.parse(result);
        if (getPendingSessionFinishes(this).delete(body.execution_id)) {
          // After the response is flushed, never before: the OE blocks on this
          // call and turns a transport error into a 502 that overwrites the
          // finished run's result. Guard on status too - an error thrown or a
          // client disconnect after this point must not still release.
          reply.raw.once("finish", () => {
            if (reply.raw.statusCode < 400) {
              void this.releaseFinishedSession(body);
            }
          });
        }
        return reply.send(responseBody);
      } finally {
        this.drainRegistry.endWork(body.execution_id);
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
  }

  private handleExecute = withMetrics(
    "aer_execute",
    (request: ExecuteRequest) => this.doHandleExecute(request),
  );

  private async doHandleExecute(
    request: ExecuteRequest,
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

    let pendingCallback: ExecutorCallbackRequest | undefined;
    const stageCallback = (
      status: string,
      fields: ExecutorCallbackFields = {},
    ): void => {
      pendingCallback = buildExecutorCallback(request.execution_id, status, {
        ...fields,
        suspend_generation: request.suspend_generation ?? undefined,
      });
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
      },
      async () => {
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

        let heartbeat: AttemptHeartbeat | null = null;
        try {
          // Register the owner callback URL inside the same try whose finally
          // pops it, so a failure between here and the first
          // sendStreamChunk / reportCallback can never strand an entry. Nothing
          // between context setup and this point delivers a chunk or callback.
          if (ownerUrl) {
            getOwnerCallbackUrls(this).set(request.execution_id, ownerUrl);
          }

          // AgentInput.payload is the opaque caller blob, with the resolved
          // message normalized in. Platform plumbing (identity, resume,
          // checkpoint) travels on ctx, not here.
          const agentInput: AgentInput = {
            payload: {
              ...(request.payload ?? {}),
              message,
            } as JsonValue,
          };

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

          const agent = getTracer("runner-shared.aer").startActiveSpan(
            AER_BUILD_AGENT,
            {
              attributes: {
                [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN,
              },
            },
            (span) => {
              try {
                return this.runtime.getAgent({ callbacks: [nodeLogger] });
              } finally {
                span.end();
              }
            },
          );

          // Materialize the adapter before asking OE for an attempt. Missing
          // identity, an old OE (bare 404), or an ineligible framework stays
          // native; other start failures fail closed before tenant work.
          const durableAttempt = await this.startDurableAttempt(
            request,
            sessionId,
          );
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
                  new WorkflowClient(request.platform_api_url),
                );
          if (heartbeat !== null) await heartbeat.start();

          let executionOutcome: InterruptResult | StreamingResult;
          try {
            const execute = () =>
              this.executeViaAgentStream(
                agent,
                ctx,
                agentInput,
                request.platform_api_url,
                request.execution_id,
              );
            executionOutcome =
              durableAttempt === null
                ? await execute()
                : await runWithAttemptContext(durableAttempt, execute);
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
              const errorMsg = `Execution timed out after ${EXECUTION_TIMEOUT_MS / 1000} seconds`;
              try {
                await this.sendStreamChunk(
                  request.platform_api_url,
                  request.execution_id,
                  ERROR,
                  "",
                  errorMsg,
                  { error_code: TIMEOUT_ERROR_CODE },
                );
              } catch (chunkErr) {
                logger.error(
                  { err: chunkErr },
                  `Failed to deliver ERROR chunk for execution ${request.execution_id} after retries`,
                );
              }
              stageCallback("ERROR", {
                error: errorMsg,
                metadata: { error_code: TIMEOUT_ERROR_CODE },
              });
              const httpError = new Error(errorMsg) as Error & {
                statusCode: number;
                alreadyReported?: boolean;
              };
              httpError.statusCode = 504;
              // Terminal ERROR chunk + callback are already sent above. Mark the
              // error so the outer catch re-raises without reporting a second time
              // — mirrors Python's `except HTTPException: raise`.
              httpError.alreadyReported = true;
              throw httpError;
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

            stageCallback("SUSPENDED", {
              suspend_reason: suspendReason,
              suspend_context: suspendContext,
              interrupts: executionOutcome.interrupts,
              resume_schema: executionOutcome.resume_schema,
              metadata: executionOutcome.metadata,
            });
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

          stageCallback("COMPLETED", {
            result: executionOutcome.content,
            metadata: executionOutcome.metadata,
          });
          if (isSessionFinishRequested()) {
            // Latched here, acted on after the response: the OE blocks on
            // this /execute call and turns a transport error into a 502 that
            // overwrites the finished run's result, so the session's pods
            // must not be killed before the response is delivered.
            getPendingSessionFinishes(this).add(request.execution_id);
          }
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
            // A deadline breach, not a crash: carry the same discriminator as
            // the whole-turn timeout so a consumer handles both the same way,
            // and 504 rather than 500 for the same reason.
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
   * Null covers incomplete tenant identity, a framework that never registered
   * a workflow adapter, and an old OE without workflow routes (bare 404). Any
   * other failure propagates so the invocation fails closed rather than
   * running unfenced.
   */
  private async startDurableAttempt(
    request: ExecuteRequest,
    sessionId: string,
  ) {
    const startRequest = attemptStartRequestFromExecute(
      request,
      sessionId,
      this.durabilityOwnerId,
      this.runtime.appName,
      this.runtime.appVersion,
      // != null: memoryWriter is optional, and the declaration must agree
      // with the wrapper setup guard on absent vs disabled writers.
      this.runtime.memoryWriter != null,
    );
    if (startRequest === null) return null;
    return new WorkflowClient(request.platform_api_url).startAttempt(
      startRequest,
    );
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

    const chunk = {
      execution_id: executionId,
      chunk_type: chunkType,
      content,
      error,
      metadata,
      seq,
    };
    const terminal = chunkType === DONE || chunkType === ERROR;
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
  private async releaseFinishedSession(request: ExecuteRequest): Promise<void> {
    if (!(await this.callbackDelivery.waitForTerminal(request.execution_id))) {
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
